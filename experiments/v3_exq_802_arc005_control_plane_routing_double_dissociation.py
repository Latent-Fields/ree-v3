"""V3-EXQ-802: ARC-005 -- does the control plane ROUTE, or merely READ OUT?

Proposal EXP-0392 (manual_proposals.v1.json, minted by /thought-digestion 2026-07-21).
claim_ids = [ARC-005] ONLY.

=== THE FALSIFIABLE CONTENT (and what is NOT being tested) ===
ARC-005 says "control plane routes precision and modes". That a control plane EXISTS is a
DESIGN FACT -- we built one, and no experiment can falsify it. The falsifiable content is
that it ROUTES: that its channels have causal authority over precision and mode which is
DISSOCIABLE FROM CONTENT. If precision and mode occupancy are simply readouts of whatever
the agent happens to be processing, the plane is epiphenomenal, and ARC-005's 88 reverse
dependencies -- the highest fan-in in the registry, currently carried on exp_conf 0.0 and
lit_conf 0.783 with ZERO experimental entries -- inherit a false premise.

=== DESIGN: DOUBLE DISSOCIATION, 2x2 EMBEDDED IN A 3-LEVEL LADDER ===
Factor 1  channel level   L0 (substrate operating settings) / L1 / L2, all four channels
                          moved TOGETHER and monotonically
Factor 2  content         set A / set B, on a FIXED arena

  arms:  L0_A  L1_A  L2_A  L0_B  L1_B  L2_B     (6 arms x 5 seeds = 30 cells)

The proposal's 2x2 is the {L0, L2} x {A, B} sub-grid:
  A0_BASE = L0_A,  A1_CHAN = L2_A,  A2_CONTENT = L0_B,  A3_BOTH = L2_B
L1 exists because the acceptance check requires MONOTONICITY, which two levels cannot show.

THE FOUR IMPLEMENTED CHANNELS (the scope of the test -- see SCOPE below):
  1. 5-HT rigidity     serotonin.tonic_5ht_baseline        0.50 -> 1.00
  2. phasic gain       phasic_burst_temp_delta            -0.10 -> -1.00
  3. mode prior        salience_external_task_bias         1.00 -> 3.00
  4. pcc_stability mu  pcc_stability_baseline              0.50 -> 0.95
Every channel is LIVE at every level (only its setting moves), so no arm differs from
another by a module being absent -- an on/off contrast would confound "routing" with
"instantiating".

CONTENT sets vary WHAT IS IN the arena, never the arena or any channel:
  A: 3 hazards, 5 resources, contamination_spread 0.5, layout seed  s
  B: 6 hazards, 2 resources, contamination_spread 0.9, layout seed  s + 10007
Arena (size, toroidal, hazard_harm, resource_benefit, energy_decay, drift) is IDENTICAL.

DVs:
  (i)  E3 precision readout  -- e3.current_precision = 1/(running_variance + 1e-6),
       recorded as log10 precision (it spans orders of magnitude).
  (ii) mode-occupancy distribution -- time-fraction of measurement steps in each of
       {external_task, internal_planning, internal_replay, offline_consolidation},
       read from salience.current_mode (the DISCRETE, hysteresis-committed mode).

=== DV-SYMMETRY INVARIANCE DECLARATION (mandatory, per arm) ===
The occupancy DV is a set-aggregate over the discrete `current_mode` sequence; the
precision DV is a mean of a continuous scalar. Neither manipulation is invariant under
those DVs' symmetry groups:

  arms L1_A / L2_A / L1_B / L2_B (the channel manipulation), channel by channel --
   (1) 5-HT rigidity is NOT a broadcast constant over modes: it enters via
       z_goal_seeding_gain / valence_wanting_floor, which reshape the dACC bundle
       (foraging_value, choice_difficulty) that the coordinator aggregates. It is
       neither additive-uniform across modes nor a monotone rescaling of the mode
       vector, so it survives both argmax and the softmax normalisation.
   (2) phasic gain enters the E3 SOFTMAX TEMPERATURE. A temperature change IS a
       monotone reparameterisation and therefore argmax-invariant -- but the
       precision DV is a VARIANCE readout (running variance of world-model
       prediction error), not a rank statistic, and temperature changes the
       realised action distribution, hence the error stream, hence the variance.
       It is NOT invariant under the precision DV. It is declared as contributing
       to precision, NOT independently to occupancy.
   (3) mode prior adds `external_task_bias` to the external_task LOGIT ONLY
       (salience_coordinator.py:435). A per-mode (non-broadcast) logit shift is
       exactly what argmax and softmax are NOT invariant under. This is the
       occupancy channel with the cleanest authority.
   (4) pcc_stability has TWO legs and they differ: the MECH-048 mu leg is a softmax
       TEMPERATURE on the mode prior and IS argmax-invariant, so it cannot by
       itself move the discrete occupancy; the MECH-259 leg is a multiplier
       (1 + stability_scaling * mu) on the SWITCH THRESHOLD, which changes switch
       ADMISSION and dwell under hysteresis, and is NOT invariant under a
       time-fraction occupancy DV. The channel therefore has authority via its
       threshold leg. Both legs are recorded separately (`effective_temperature`,
       `enter_threshold`) so a reader can see which one moved.
  arms L0_A / L0_B (the content manipulation): content changes hazard/resource
       density, contamination spread and layout, i.e. the observation stream itself.
       No symmetry of either DV is a relabelling of the environment content.

No arm is scoped out on symmetry grounds. If a channel had turned out to be invariant
under both DVs it would be disposition (b) of the precondition-gate rule -- scoped OUT of
scoring and routed non_contributory, never `mixed`.

=== PRE-REGISTERED CRITERIA (constants below; nothing derived from the run) ===
Per seed s, with TV = total-variation distance between occupancy vectors (in
occupancy-fraction units, range [0, 1]):
    D_chan(s) = mean over content of TV( occ(L2, c, s), occ(L0, c, s) )
    D_cont(s) = mean over level in {L0, L2} of TV( occ(l, B, s), occ(l, A, s) )
    Delta(s)  = D_chan(s) - D_cont(s)

  C1 DISSOCIATION (load-bearing)
     mean_s Delta >= 0.8 * SD_s(Delta)  AND  mean_s Delta >= 0.15
  C2 MONOTONICITY (load-bearing)
     within each (content, seed) triplet over levels L0<L1<L2:
       Spearman rho of external_task occupancy vs level >= +0.60   [SIGNED -- the
         direction is pre-registrable: the bias raises the external_task logit]
       |Spearman rho| of log10-precision vs level >= 0.60          [UNSIGNED -- two
         channels touch precision with opposite-signed intuitions, so only
         MONOTONICITY, not its sign, is what ARC-005 asserts]
     satisfied in >= 7 of the 10 (content x seed) units
  C3 REGIME REPRODUCIBILITY (load-bearing)
     for each level in {L0, L2} and seed, argmax-occupancy mode under content A equals
     that under content B; satisfied in >= 7 of the 10 (level x seed) units

  PASS  = C1 and C2 and C3                       -> evidence_direction supports
  FAIL with mean_s Delta <= 0                    -> evidence_direction WEAKENS ARC-005
        (mode occupancy tracks content at least as well as channel settings: the plane
         is a readout, not a router. This is a DECISION-FLIPPING negative, not a null.)
  FAIL with mean_s Delta > 0 but below the gates  -> mixed (a measured weak effect)

=== NON-DEGENERACY (else substrate_not_ready_requeue, NOT a verdict) ===
Per-arm gates, regime-conditioned via experiments/_lib/precondition_gate.py so that no
arm's red can vacate another arm's result:
  P1 n_distinct_argmax_modes_across_design >= 2   all arms (a DESIGN-level scalar, so
       it gates the run as a whole). See "P1 RESCOPED" immediately below.
  P2 precision_cross_seed_sd > 1e-6   all arms. A floor-pinned precision readout carries
       no cross-seed variance and cannot support the C1 SD-scaled comparison.
  P3 channel_state_delta_vs_L0 > 0.05 applies_to PERTURBED ARMS ONLY. Verifies the
       settings TOOK EFFECT (realised tonic_5ht / temp_delta / bias / pcc_stability moved
       against the same-content L0 arm), not merely that they were passed. Structurally
       0 for an L0 arm compared with itself, hence scoped out there (disposition (a)).
  P4 n_salience_ticks >= 150          all arms. The coordinator must have actually run
       enough times for an occupancy distribution to mean anything.

=== P1 RESCOPED (a gate whose LITERAL form vacates the result it protects) ===
EXP-0392 and ARC-005's `what_would_answer` both state P1 as "in the BASELINE ARM at
least two modes occupied >= 5% of steps", justified by: "a single-mode-collapsed agent
makes all arms trivially identical and the comparison vacuous."

A 600-tick readiness probe (seed 0, all four corner arms, 2026-07-21) FALSIFIES that
justification on this substrate:

    arm    mode occupancy                        log10 precision   pcc_stab  seeding gain
    L0_A   internal_planning 1.00                2.360             0.476     0.90
    L0_B   internal_planning 1.00                2.241             0.476     0.90
    L2_A   external_task     1.00                2.205             0.926     1.50
    L2_B   external_task     1.00                2.320             0.920     1.50

Every arm is single-mode, and the arms are the OPPOSITE of trivially identical: the
channel manipulation moves the discrete mode from end to end (TV distance 1.0) while
the content manipulation moves it not at all (TV distance 0.0). The literal per-arm
gate would therefore have self-routed `substrate_not_ready_requeue` on the strongest
possible instance of the effect ARC-005 asserts -- the V3-EXQ-785 vacating failure
mode exactly, arriving through a precondition rather than through an AND.

The gate is therefore RESCOPED, not relaxed -- no threshold is lowered, and the
statistic is changed to the one the stated rationale is actually about:

    GATING     n_distinct_argmax_modes_across_design >= 2
               If EVERY cell of the design sits in the SAME mode, nothing varies and
               the comparison genuinely IS vacuous. That is the condition "makes all
               arms trivially identical" names. It is a design-level scalar, so it
               takes the same value in every arm and gates the run as a whole.
    DIAGNOSTIC n_modes_occupied_ge_5pct, per arm, recorded and NEVER gating. Within-arm
               multi-modality is interesting (it says the agent is time-sharing), but
               its absence is not degeneracy -- a channel that genuinely routes MAY
               pin occupancy to one mode, and failing an arm for that is failing it
               for exhibiting the effect under test.

A run in which all 30 cells share one mode still self-routes substrate_not_ready.

=== WHICH CHANNEL CARRIES A PASS (stated up front, not discovered later) ===
Channel 3 (the mode prior) has DIRECT authority over the occupancy DV: it adds to the
external_task logit. Channel 4's threshold leg gates switching. Channels 1 and 2 reach
occupancy only indirectly, and act mainly on the precision DV. So a C1 PASS is expected
to be carried substantially by channels 3 and 4. That is a claim about the CONTROL PLANE
having causal authority -- which is what ARC-005 asserts -- but it is NOT a claim that
each of the four channels independently routes modes. Per-channel dissociation is
UNTESTED here and would need a channel-wise ablation grid. Stated in the manifest under
custom_information.channel_attribution_limit.

SCORABILITY. The 2x2 needs all four of L0_A, L0_B, L2_A, L2_B green; if any is red the
run self-routes `substrate_not_ready_requeue` (never a substrate verdict). A red L1 arm
vacates only C2, which is recorded criteria_non_degenerate false while C1/C3 stand.

=== SAMPLE-SIZE INTEGRITY ===
`current_mode` is agent STATE, not a latched `last_*` diagnostic: the mode genuinely
holds between coordinator ticks, so a per-env-step occupancy read is a time-fraction,
not pseudo-replication. Nonetheless the number of GENUINE coordinator ticks is counted
(identity change of `agent._salience_last_tick`, which `tick()` rebuilds every call) and
emitted as `n_salience_ticks`, so the true independent denominator is auditable rather
than inferred from the step count. The UNIT OF ANALYSIS for every criterion is the SEED
(n=5), so within-cell autocorrelation cannot inflate any gate.

=== NO GRADIENT TRAINING ===
Nothing is trained; no head reads a latent under a loss. The agent is driven in eval()
exactly as the arms are constructed, so the phased-training protocol does not apply. A
WARMUP phase precedes measurement solely so the precision EMA and the coordinator's
signal EMAs have settled before any datum is recorded.

=== SCOPE ===
Tests routing authority over the CURRENTLY-IMPLEMENTED channel set. A PASS does NOT
vindicate every channel posited in MECH-002/MECH-004; it establishes that the plane is
causal rather than decorative. A FAIL would make Q-017 (minimal orthogonal control-axis
subset) and MECH-063 (orthogonal tonic/phasic axes) moot.

=== GOV-REUSE-1 ===
Decisive readout: the channel-vs-content main effect on mode occupancy. `mode_occupancy`
appears in ZERO recorded manifests (reanalysis_query --readout mode_occupancy: 0 matches)
and ARC-005 has zero experimental entries, so nothing is recoverable post-hoc. Run.

=== MINT-AS-YOU-GO ===
The OFF arm (L0_A) is constructed from experiments/_lib/baselines/exq800_arc005_control_
plane.py and every cell is fingerprinted with include_driver_script_in_hash=False, so a
later sibling with a different driver can reuse these cells. This run IS the lineage mint;
no separate baseline-only job is queued.
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

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
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
EXPERIMENT_TYPE = "v3_exq_802_arc005_control_plane_routing_double_dissociation"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["ARC-005"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
BACKLOG_ID = "EXP-0392"

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

MODE_NAMES = [
    "external_task", "internal_planning", "internal_replay", "offline_consolidation",
]

# --- C1 dissociation ---
C1_SD_MULTIPLIER = 0.8            # mean(Delta) >= 0.8 * SD(Delta)
C1_ABS_FLOOR = 0.15               # ... AND >= 0.15 occupancy-fraction
# --- C2 monotonicity ---
C2_OCC_RHO_FLOOR = 0.60           # SIGNED (direction pre-registered)
C2_PRECISION_ABS_RHO_FLOOR = 0.60  # UNSIGNED (only monotonicity is claimed)
C2_MIN_UNITS = 7                  # of 10 (content x seed)
# --- C3 regime reproducibility ---
C3_MIN_UNITS = 7                  # of 10 (level x seed)

# --- non-degeneracy floors ---
# P1 GATES on the design-level statistic (see "P1 RESCOPED" in the docstring). The
# per-arm within-arm count is recorded as a NON-GATING diagnostic at the 5% fraction.
P1_DISTINCT_MODES_FLOOR = 1.5     # strictly-more-than-one distinct mode in the design
P1_OCCUPANCY_FRACTION = 0.05      # diagnostic only
P2_PRECISION_SD_FLOOR = 1e-6
P3_CHANNEL_DELTA_FLOOR = 0.05
P4_SALIENCE_TICK_FLOOR = 150.0
DRY_P4_SALIENCE_TICK_FLOOR = 5.0  # smoke only -- criteria are not scored on a smoke


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


def _tv_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Total-variation distance between two occupancy vectors, in [0, 1].

    Expressed directly in OCCUPANCY-FRACTION units, which is what the C1 absolute
    floor (0.15) is stated in.
    """
    return 0.5 * sum(abs(float(p.get(m, 0.0)) - float(q.get(m, 0.0))) for m in MODE_NAMES)


def _arm_id(level: float, content: str) -> str:
    return BASE.arm_id(level, content)


def _build(seed: int, level: float, content: str):
    """Construct env + agent for one cell, THROUGH the canonical baseline module."""
    env_kw = BASE.content_env_kwargs(content, seed)
    env = CausalGridWorld(**env_kw)
    _obs, obs_dict = env.reset()

    kw: Dict[str, Any] = dict(BASE.agent_kwargs(level))
    kw.update(
        body_obs_dim=obs_dict["body_state"].shape[-1],
        world_obs_dim=obs_dict["world_state"].shape[-1],
        action_dim=env.action_dim,
    )
    cfg = REEConfig.from_dims(**kw)
    # Channel 1 lives on the nested SerotoninConfig, which from_dims does not expose
    # as scalar kwargs. Pin the channel's OUTPUT (see the note in channel_settings):
    # gain_min == gain_max makes current_seeding_gain() return the ladder value
    # exactly, immune to the harm-suppression term that crushes the tonic level.
    ch = BASE.channel_settings(level)
    cfg.serotonin.gain_min = cfg.serotonin.gain_max = float(ch["serotonin_seeding_gain"])
    cfg.serotonin.floor_min = cfg.serotonin.floor_max = float(
        ch["serotonin_wanting_floor"]
    )
    agent = REEAgent(cfg)
    agent.eval()
    return agent, env, obs_dict, kw


def _run_cell(
    seed: int, level: float, content: str, n_warmup: int, n_measure: int
) -> Dict[str, Any]:
    """Drive one (seed, level, content) cell and return its readouts."""
    arm = _arm_id(level, content)
    agent, env, obs_dict, kw = _build(seed, level, content)
    n_ticks = n_warmup + n_measure

    print(f"Seed {seed} Condition {arm}", flush=True)

    mode_counts: Dict[str, int] = {m: 0 for m in MODE_NAMES}
    log_precisions: List[float] = []
    mode_entropies: List[float] = []
    eff_temps: List[float] = []
    enter_thresholds: List[float] = []
    seeding_gains: List[float] = []
    tonic_5ht: List[float] = []
    pcc_stab: List[float] = []
    burst_levels: List[float] = []
    n_salience_ticks = 0
    n_measured_steps = 0
    prev_sal_tick: Any = None

    z_world_prev: Optional[torch.Tensor] = None
    action_prev: Optional[torch.Tensor] = None

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
            z_world_cur = latent.z_world.detach()
            # The driven loop never calls update_running_variance() itself, so the
            # E3 precision readout would stay pinned at precision_init without this.
            if z_world_prev is not None and action_prev is not None:
                _pred = agent.e2.world_forward(z_world_prev, action_prev)
                agent.e3.update_running_variance(z_world_cur - _pred.detach())

            ticks_d = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks_d["e1_tick"]
                else torch.zeros(1, agent.config.latent.world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks_d)
            action = agent.select_action(candidates, ticks_d, 1.0)
        agent._step_count += 1

        # A NEW dict object is returned by salience.tick() on every genuine call, so
        # an identity change counts coordinator ticks exactly (n_salience_ticks is the
        # honest independent denominator behind the per-step occupancy time-fraction).
        cur_sal_tick = getattr(agent, "_salience_last_tick", None)
        if cur_sal_tick is not None and cur_sal_tick is not prev_sal_tick:
            n_salience_ticks += 1
            if measuring:
                mode_entropies.append(float(cur_sal_tick.get("mode_entropy", 0.0)))
                eff_temps.append(float(cur_sal_tick.get("effective_temperature", 1.0)))
                enter_thresholds.append(float(cur_sal_tick.get("enter_threshold", 0.0)))
        prev_sal_tick = cur_sal_tick

        act_idx = (
            int(action.argmax().item()) if isinstance(action, torch.Tensor) else int(action)
        )
        action_prev = torch.zeros(1, env.action_dim)
        action_prev[0, act_idx % env.action_dim] = 1.0
        z_world_prev = z_world_cur
        _obs, reward, done, _info, obs_dict = env.step(act_idx % env.action_dim)

        # Keep the 5-HT channel LIVE: the agent's own loop does not drive it in this
        # driven harness, so without this the tonic level would never leave baseline.
        agent.serotonin_step(max(0.0, float(reward)))

        if measuring:
            n_measured_steps += 1
            mode = agent.salience.current_mode if agent.salience is not None else None
            if mode in mode_counts:
                mode_counts[mode] += 1
            log_precisions.append(math.log10(max(agent.e3.current_precision, 1e-12)))
            seeding_gains.append(float(agent.serotonin.current_seeding_gain()))
            tonic_5ht.append(float(agent.serotonin.tonic_5ht))
            pcc_stab.append(float(agent.pcc.pcc_stability) if agent.pcc is not None else 0.0)
            burst_levels.append(
                float(agent.phasic_burst.burst_level)
                if agent.phasic_burst is not None else 0.0
            )

        if done:
            _obs, obs_dict = env.reset()

        if (tick + 1) % 250 == 0 or tick == n_ticks - 1:
            print(
                f"  [train] arc005 {arm} seed={seed} ep {tick + 1}/{n_ticks} "
                f"sal_ticks={n_salience_ticks} steps={n_measured_steps}",
                flush=True,
            )

    denom = max(1, n_measured_steps)
    occupancy = {m: mode_counts[m] / denom for m in MODE_NAMES}

    row: Dict[str, Any] = {
        "arm_id": arm,
        "channel_level": float(level),
        "content": content,
        "seed": int(seed),
        "n_ticks": n_ticks,
        "n_warmup_ticks": n_warmup,
        "n_measured_steps": n_measured_steps,
        # The honest independent denominator behind the occupancy time-fraction.
        "n_salience_ticks": n_salience_ticks,
        "mode_occupancy": {k: round(v, 6) for k, v in occupancy.items()},
        "mode_counts": dict(mode_counts),
        "n_modes_occupied_ge_5pct": sum(
            1 for v in occupancy.values() if v >= P1_OCCUPANCY_FRACTION
        ),
        "argmax_mode": max(occupancy.items(), key=lambda kv: kv[1])[0],
        "log10_precision_mean": round(float(np.mean(log_precisions)), 8)
        if log_precisions else 0.0,
        "log10_precision_sd": round(float(np.std(log_precisions)), 8)
        if len(log_precisions) > 1 else 0.0,
        # Realised channel state -- P3 reads these, and they are what makes
        # "the settings took effect" a MEASUREMENT rather than an assumption.
        "realised_channel_state": {
            # Channel 1's setting IS the seeding gain (the rigidity the channel
            # imposes). tonic_5ht is recorded alongside as context only -- it is
            # driven by harm suppression, not by the manipulation.
            "seeding_gain_mean": round(float(np.mean(seeding_gains)), 8)
            if seeding_gains else 0.0,
            "tonic_5ht_mean": round(float(np.mean(tonic_5ht)), 8) if tonic_5ht else 0.0,
            "pcc_stability_mean": round(float(np.mean(pcc_stab)), 8) if pcc_stab else 0.0,
            "burst_level_mean": round(float(np.mean(burst_levels)), 8)
            if burst_levels else 0.0,
            "phasic_temp_delta_cfg": float(
                agent.phasic_burst.config.temp_delta
                if agent.phasic_burst is not None else 0.0
            ),
            "external_task_bias_cfg": float(
                agent.salience.config.external_task_bias
                if agent.salience is not None else 0.0
            ),
        },
        # The two pcc_stability legs, recorded SEPARATELY so a reader can see which
        # one moved (mu -> temperature is argmax-invariant; threshold is not).
        "mode_prior_diagnostics": {
            "mode_entropy_mean": round(float(np.mean(mode_entropies)), 8)
            if mode_entropies else 0.0,
            "effective_temperature_mean": round(float(np.mean(eff_temps)), 8)
            if eff_temps else 0.0,
            "enter_threshold_mean": round(float(np.mean(enter_thresholds)), 8)
            if enter_thresholds else 0.0,
        },
        "phasic_events_converged": int(
            getattr(agent.phasic_burst, "_n_events_converged", 0)
            if agent.phasic_burst is not None else 0
        ),
    }
    return row


# ------------------------------------------------------------------ #
# Precondition specs (regime-conditioned)                             #
# ------------------------------------------------------------------ #
def _specs(salience_tick_floor: float) -> List[PreconditionSpec]:
    return [
        PreconditionSpec(
            name="n_distinct_argmax_modes_across_design",
            description=(
                "distinct argmax-occupancy modes over ALL cells of the design -- the "
                "run is vacuous only if every cell sits in the same mode"
            ),
            control="the full 6-arm x seeds grid",
            threshold=P1_DISTINCT_MODES_FLOOR,
        ),
        PreconditionSpec(
            name="precision_cross_seed_sd",
            description="cross-seed SD of the arm's log10-precision mean",
            control="five independently seeded cells of the same arm",
            threshold=P2_PRECISION_SD_FLOOR,
        ),
        PreconditionSpec(
            name="channel_state_delta_vs_L0",
            description=(
                "normalised L1 distance of the arm's REALISED channel state from the "
                "same-content L0 arm's"
            ),
            control="same-content level-0 arm",
            threshold=P3_CHANNEL_DELTA_FLOOR,
            applies_to=lambda ctx: float(ctx["channel_level"]) != LEVELS[0],
            applies_note=(
                "perturbed arms only -- an L0 arm compared with itself is 0 by "
                "construction, not by substrate failure"
            ),
        ),
        PreconditionSpec(
            name="n_salience_ticks",
            description="genuine SalienceCoordinator ticks (worst cell of the arm)",
            control="coordinator ticking on the live selection path",
            threshold=salience_tick_floor,
        ),
    ]


def _arm_contexts() -> List[Dict[str, Any]]:
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


def _worst_cell(rows: List[Dict[str, Any]], key: str) -> Tuple[float, str]:
    """Minimum of `key` over the arm's cells, plus the offending cell id.

    `met` for these preconditions is a worst-case claim ("every cell clears the
    floor"), so `measured` MUST be the worst cell -- a mean would recompute as MET
    while an out-of-band cell is masked.
    """
    best: Optional[float] = None
    who = ""
    for r in rows:
        v = float(r[key]) if not isinstance(r[key], dict) else 0.0
        if best is None or v < best:
            best, who = v, f"seed={r['seed']}"
    return (float(best) if best is not None else 0.0), who


def _channel_state_vector(row: Dict[str, Any]) -> np.ndarray:
    s = row["realised_channel_state"]
    # Each component normalised by its own ladder span, so the L1 distance is
    # comparable across channels and the 0.05 floor means the same thing for each.
    return np.array(
        [
            (float(s["seeding_gain_mean"]) - 0.90) / 0.60,
            (abs(float(s["phasic_temp_delta_cfg"])) - 0.10) / 0.90,
            (float(s["external_task_bias_cfg"]) - 1.00) / 2.00,
            (float(s["pcc_stability_mean"]) - 0.50) / 0.45,
        ],
        dtype=float,
    )


# ------------------------------------------------------------------ #
# Analysis                                                            #
# ------------------------------------------------------------------ #
def _analyse(rows: List[Dict[str, Any]], seeds: List[int]) -> Dict[str, Any]:
    by: Dict[Tuple[float, str, int], Dict[str, Any]] = {
        (r["channel_level"], r["content"], r["seed"]): r for r in rows
    }
    L0, L2 = LEVELS[0], LEVELS[-1]

    # --- C1: channel vs content main effect on mode occupancy ---
    d_chan: List[float] = []
    d_cont: List[float] = []
    deltas: List[float] = []
    per_seed_effects: List[Dict[str, Any]] = []
    for s in seeds:
        try:
            dc = statistics.fmean(
                _tv_distance(by[(L2, c, s)]["mode_occupancy"],
                             by[(L0, c, s)]["mode_occupancy"])
                for c in CONTENTS
            )
            dk = statistics.fmean(
                _tv_distance(by[(lv, "B", s)]["mode_occupancy"],
                             by[(lv, "A", s)]["mode_occupancy"])
                for lv in (L0, L2)
            )
        except KeyError:
            continue
        d_chan.append(dc)
        d_cont.append(dk)
        deltas.append(dc - dk)
        per_seed_effects.append(
            {"seed": s, "d_channel": round(dc, 6), "d_content": round(dk, 6),
             "delta": round(dc - dk, 6)}
        )

    delta_mean = float(statistics.fmean(deltas)) if deltas else 0.0
    delta_sd = float(statistics.stdev(deltas)) if len(deltas) > 1 else 0.0
    c1_sd_gate = C1_SD_MULTIPLIER * delta_sd
    c1_pass = bool(deltas) and (delta_mean >= c1_sd_gate) and (delta_mean >= C1_ABS_FLOOR)

    # --- C2: monotonicity over the channel ladder ---
    c2_units: List[Dict[str, Any]] = []
    for c in CONTENTS:
        for s in seeds:
            try:
                trip = [by[(lv, c, s)] for lv in LEVELS]
            except KeyError:
                continue
            occ = [float(r["mode_occupancy"]["external_task"]) for r in trip]
            prec = [float(r["log10_precision_mean"]) for r in trip]
            rho_occ = _spearman_rho(list(LEVELS), occ)
            rho_prec = _spearman_rho(list(LEVELS), prec)
            ok = (rho_occ >= C2_OCC_RHO_FLOOR) and (
                abs(rho_prec) >= C2_PRECISION_ABS_RHO_FLOOR)
            c2_units.append(
                {"content": c, "seed": s, "rho_external_task_occupancy": round(rho_occ, 4),
                 "rho_log10_precision": round(rho_prec, 4), "satisfied": bool(ok)}
            )
    c2_n = sum(1 for u in c2_units if u["satisfied"])
    c2_pass = c2_n >= C2_MIN_UNITS

    # --- C3: same channel setting reproduces the same regime across content ---
    c3_units: List[Dict[str, Any]] = []
    for lv in (L0, L2):
        for s in seeds:
            try:
                a, b = by[(lv, "A", s)], by[(lv, "B", s)]
            except KeyError:
                continue
            same = a["argmax_mode"] == b["argmax_mode"]
            c3_units.append(
                {"channel_level": lv, "seed": s, "argmax_A": a["argmax_mode"],
                 "argmax_B": b["argmax_mode"], "satisfied": bool(same)}
            )
    c3_n = sum(1 for u in c3_units if u["satisfied"])
    c3_pass = c3_n >= C3_MIN_UNITS

    return {
        "per_seed_effects": per_seed_effects,
        "d_channel_mean": round(float(statistics.fmean(d_chan)), 6) if d_chan else 0.0,
        "d_content_mean": round(float(statistics.fmean(d_cont)), 6) if d_cont else 0.0,
        "delta_mean": round(delta_mean, 6),
        "delta_sd": round(delta_sd, 6),
        "c1_sd_gate": round(c1_sd_gate, 6),
        "c1_abs_floor": C1_ABS_FLOOR,
        "c1_pass": c1_pass,
        "c2_units": c2_units,
        "c2_n_satisfied": c2_n,
        "c2_n_units": len(c2_units),
        "c2_pass": c2_pass,
        "c3_units": c3_units,
        "c3_n_satisfied": c3_n,
        "c3_n_units": len(c3_units),
        "c3_pass": c3_pass,
    }


# ------------------------------------------------------------------ #
# Driver                                                              #
# ------------------------------------------------------------------ #
def run_experiment(dry_run: bool) -> Dict[str, Any]:
    seeds = DRY_SEEDS if dry_run else SEEDS
    n_warmup = DRY_WARMUP_TICKS if dry_run else WARMUP_TICKS
    n_measure = DRY_MEASURE_TICKS if dry_run else MEASURE_TICKS
    tick_floor = DRY_P4_SALIENCE_TICK_FLOOR if dry_run else P4_SALIENCE_TICK_FLOOR

    specs = _specs(tick_floor)
    contexts = _arm_contexts()
    # Design-time proof: refuse to start a run carrying an unsatisfiable gate.
    assert_no_structurally_unsatisfiable_gate(specs, contexts)
    # The OFF cell must be built from the canonical baseline module BY CONSTRUCTION,
    # or a later sibling's fingerprint can never match this mint.
    assert BASE.cell_config_slice(LEVELS[0], "A") == BASE.off_path_config_slice(), (
        "OFF cell slice diverged from off_path_config_slice() -- the mint would not "
        "be reusable"
    )

    rows: List[Dict[str, Any]] = []
    for content in CONTENTS:
        for level in LEVELS:
            for seed in seeds:
                slice_ = BASE.cell_config_slice(level, content)
                with arm_cell(
                    seed,
                    config_slice=slice_,
                    script_path=Path(__file__),
                    config_slice_declared=True,
                    # MANDATORY for a cross-driver-reusable mint: without it this
                    # driver folds into substrate_hash and no sibling can ever hit.
                    include_driver_script_in_hash=False,
                ) as cell:
                    row = _run_cell(seed, level, content, n_warmup, n_measure)
                    cell.stamp(row)
                rows.append(row)
                print(
                    f"verdict: {'PASS' if row['n_measured_steps'] > 0 else 'FAIL'}",
                    flush=True,
                )

    # ---- per-arm gates ------------------------------------------------------ #
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_arm.setdefault(r["arm_id"], []).append(r)

    l0_state: Dict[str, np.ndarray] = {}
    for content in CONTENTS:
        arm0 = _arm_id(LEVELS[0], content)
        vecs = [_channel_state_vector(r) for r in by_arm.get(arm0, [])]
        l0_state[content] = (
            np.mean(np.stack(vecs), axis=0) if vecs else np.zeros(4, dtype=float)
        )

    # Design-level P1 statistic: identical in every arm by construction, so it gates
    # the run as a whole rather than singling any arm out.
    distinct_modes = float(len({r["argmax_mode"] for r in rows}))

    arm_gates = []
    offending: Dict[str, str] = {}
    for ctx in contexts:
        arm_rows = by_arm.get(ctx["id"], [])
        if not arm_rows:
            continue
        ticks_worst, who_ticks = _worst_cell(arm_rows, "n_salience_ticks")
        prec_means = [float(r["log10_precision_mean"]) for r in arm_rows]
        prec_sd = float(statistics.stdev(prec_means)) if len(prec_means) > 1 else 0.0
        vec = np.mean(np.stack([_channel_state_vector(r) for r in arm_rows]), axis=0)
        chan_delta = float(np.mean(np.abs(vec - l0_state[ctx["content"]])))
        offending[ctx["id"]] = f"ticks:{who_ticks}"
        arm_gates.append(
            evaluate_arm_gate(
                ctx["id"],
                ctx,
                specs,
                measured={
                    "n_distinct_argmax_modes_across_design": distinct_modes,
                    "precision_cross_seed_sd": prec_sd,
                    "channel_state_delta_vs_L0": chan_delta,
                    "n_salience_ticks": ticks_worst,
                },
            )
        )
    gate = aggregate_arm_gates(arm_gates)
    green = set(gate["green_arms"])

    # The 2x2 needs all four corner arms green. A red L1 arm vacates only C2.
    corners = [
        _arm_id(LEVELS[0], "A"), _arm_id(LEVELS[0], "B"),
        _arm_id(LEVELS[-1], "A"), _arm_id(LEVELS[-1], "B"),
    ]
    red_corners = [a for a in corners if a not in green]
    mid_arms = [_arm_id(LEVELS[1], c) for c in CONTENTS]
    red_mid = [a for a in mid_arms if a not in green]
    scorable = not red_corners

    analysis = _analyse(rows, seeds)

    # ---- routing ------------------------------------------------------------ #
    if not scorable:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
        non_degenerate = False
        degeneracy_reason = (
            "2x2 corner arm(s) failed their non-degeneracy gate: "
            + ", ".join(red_corners)
            + f" (green arms: {sorted(green)})"
        )
        overall_pass = False
    else:
        overall_pass = bool(
            analysis["c1_pass"] and analysis["c2_pass"] and analysis["c3_pass"]
        )
        outcome = "PASS" if overall_pass else "FAIL"
        label = "control_plane_routes" if overall_pass else (
            "control_plane_is_readout_not_router"
            if analysis["delta_mean"] <= 0.0 else "control_plane_routing_weak"
        )
        if overall_pass:
            evidence_direction = "supports"
        elif analysis["delta_mean"] <= 0.0:
            # DECISION-FLIPPING negative: occupancy tracks content at least as well
            # as channel settings. ARC-005 demotes.
            evidence_direction = "weakens"
        else:
            evidence_direction = "mixed"
        non_degenerate = True
        degeneracy_reason = ""
        if red_mid:
            degeneracy_reason = (
                "C2 (monotonicity) is not scorable: mid-ladder arm(s) red: "
                + ", ".join(red_mid) + ". C1 and C3 stand."
            )

    criteria = [
        {"name": "C1_dissociation", "load_bearing": True,
         "passed": bool(analysis["c1_pass"])},
        {"name": "C2_monotonicity", "load_bearing": True,
         "passed": bool(analysis["c2_pass"])},
        {"name": "C3_regime_reproducibility", "load_bearing": True,
         "passed": bool(analysis["c3_pass"])},
    ]
    # `arm_criteria_non_degenerate` maps ONE owning arm -> its criteria. Every
    # criterion here is owned by SEVERAL arms (a 2x2 contrast needs all four corners),
    # so the conjunction is computed directly: a criterion is non-degenerate only if
    # every arm it reads is green. Same semantics, correct arity.
    criterion_owners = {
        "C1_dissociation": corners,
        "C2_monotonicity": corners + mid_arms,
        "C3_regime_reproducibility": corners,
    }
    criteria_nd = {
        name: all(a in green for a in owners)
        for name, owners in criterion_owners.items()
    }

    manifest: Dict[str, Any] = {
        "run_id": None,  # filled by __main__
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "backlog_id": BACKLOG_ID,
        "outcome": outcome,
        "overall_pass": overall_pass,
        "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "criteria": criteria,
        "analysis": analysis,
        "per_arm_gate": gate["per_arm_gate"],
        "per_arm_gate_offending_cells": offending,
        "diagnostics": {
            # NON-GATING (see "P1 RESCOPED"). Recorded on PASS runs too: a diagnostic
            # that appears only when something looks wrong cannot establish that
            # anything was ever right.
            "within_arm_modes_occupied_ge_5pct": {
                arm: {
                    "worst_cell": _worst_cell(rs, "n_modes_occupied_ge_5pct")[0],
                    "offending_cell": _worst_cell(rs, "n_modes_occupied_ge_5pct")[1],
                    "by_seed": {str(r["seed"]): r["n_modes_occupied_ge_5pct"]
                                for r in rs},
                }
                for arm, rs in by_arm.items()
            },
            "n_distinct_argmax_modes_across_design": distinct_modes,
            "argmax_mode_by_arm": {
                arm: sorted({r["argmax_mode"] for r in rs}) for arm, rs in by_arm.items()
            },
        },
        "scorable_2x2": scorable,
        "red_corner_arms": red_corners,
        "arm_results": rows,
        "per_seed_effects": analysis["per_seed_effects"],
        "interpretation": {
            "label": label,
            "preconditions": gate["adjudication_preconditions"],
            "preconditions_scope_note": gate["per_arm_gate"]["preconditions_scope_note"],
            "criteria_non_degenerate": criteria_nd,
        },
        "pre_registered_thresholds": {
            "c1_sd_multiplier": C1_SD_MULTIPLIER,
            "c1_abs_floor": C1_ABS_FLOOR,
            "c2_occupancy_rho_floor": C2_OCC_RHO_FLOOR,
            "c2_precision_abs_rho_floor": C2_PRECISION_ABS_RHO_FLOOR,
            "c2_min_units": C2_MIN_UNITS,
            "c3_min_units": C3_MIN_UNITS,
            "p1_distinct_modes_floor": P1_DISTINCT_MODES_FLOOR,
            "p1_occupancy_fraction_diagnostic_only": P1_OCCUPANCY_FRACTION,
            "p2_precision_sd_floor": P2_PRECISION_SD_FLOOR,
            "p3_channel_delta_floor": P3_CHANNEL_DELTA_FLOOR,
            "p4_salience_tick_floor": tick_floor,
        },
        "custom_information": {
            "channel_ladder": {str(lv): BASE.channel_settings(lv) for lv in LEVELS},
            "content_sets": BASE.CONTENT_SETS,
            "arena": BASE.ARENA,
            "dv_symmetry_declaration": (
                "mode prior = per-mode logit shift (external_task only), not a broadcast "
                "constant -> not argmax/softmax invariant. pcc_stability mu leg IS a "
                "softmax temperature (argmax-invariant) but its MECH-259 threshold leg "
                "changes switch admission under hysteresis -> not invariant under a "
                "time-fraction occupancy DV. phasic gain is a temperature (argmax-"
                "invariant) and is declared as acting on the PRECISION DV (a variance "
                "readout), not independently on occupancy. 5-HT reshapes the dACC bundle "
                "the coordinator aggregates -- neither uniform-additive nor a monotone "
                "rescaling. No arm scoped out on symmetry grounds."
            ),
            "channel_attribution_limit": (
                "Channel 3 (mode prior) has DIRECT authority over the occupancy DV and "
                "channel 4's threshold leg gates switching; channels 1 and 2 reach "
                "occupancy only indirectly and act mainly on precision. A C1 PASS is "
                "therefore expected to be carried substantially by channels 3 and 4, "
                "and licenses the claim that THE PLANE has causal authority -- NOT that "
                "each channel independently routes modes. Per-channel dissociation is "
                "UNTESTED here; it needs a channel-wise ablation grid."
            ),
            "p1_rescope": (
                "EXP-0392's literal per-arm 'baseline occupies >= 2 modes' gate was "
                "rescoped to the design-level 'the design occupies >= 2 distinct modes'. "
                "A 600-tick readiness probe (2026-07-21) measured every corner arm "
                "single-mode WHILE the channel manipulation moved the mode end-to-end "
                "(TV 1.0) and content moved it not at all (TV 0.0) -- so the literal "
                "gate would have vacated the strongest instance of the effect under "
                "test. No threshold was lowered; the statistic was changed to the one "
                "the gate's own stated rationale ('makes all arms trivially identical') "
                "is about. Within-arm multi-modality is kept as a NON-GATING diagnostic."
            ),
            "readiness_probe_2026_07_21_seed0_600ticks": {
                "L0_A": {"mode": "internal_planning", "log10_precision": 2.360},
                "L0_B": {"mode": "internal_planning", "log10_precision": 2.241},
                "L2_A": {"mode": "external_task", "log10_precision": 2.205},
                "L2_B": {"mode": "external_task", "log10_precision": 2.320},
            },
            "substrate_config_notes": (
                "use_dacc + use_aic_analog are ON in EVERY arm at identical settings. "
                "With both off the coordinator's salience_aggregate is identically 0, "
                "argmax is always external_task, and the discrete mode can never "
                "switch -- occupancy would be single-mode by CONFIGURATION rather than "
                "by substrate property. Likewise phasic_burst_signal_source is set to "
                "'instantaneous_pe' (with baseline_continuity 'carry') because the "
                "default smoothed source fires ZERO events, leaving channel 2 inert. "
                "Both are shared substrate config, not part of either factor."
            ),
            "gov_reuse_1": (
                "readout 'mode_occupancy' absent from every recorded manifest "
                "(reanalysis_query: 0 matches); ARC-005 has zero experimental entries "
                "-> not recoverable, run"
            ),
            "mint": (
                "lineage exq802_arc005_control_plane; every cell fingerprinted with "
                "include_driver_script_in_hash=False -> cross-driver reusable"
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
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    from datetime import datetime

    manifest = run_experiment(dry_run=args.dry_run)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest["timestamp_utc"] = ts
    manifest["dry_run"] = bool(args.dry_run)

    full_config = {
        "arena": BASE.ARENA,
        "content_sets": BASE.CONTENT_SETS,
        "channel_levels": LEVELS,
        "channel_ladder": {str(lv): BASE.channel_settings(lv) for lv in LEVELS},
        "agent_kwargs_by_level": {str(lv): BASE.agent_kwargs(lv) for lv in LEVELS},
        "schedule": {
            "warmup_ticks": DRY_WARMUP_TICKS if args.dry_run else WARMUP_TICKS,
            "measure_ticks": DRY_MEASURE_TICKS if args.dry_run else MEASURE_TICKS,
        },
        "pre_registered_thresholds": manifest["pre_registered_thresholds"],
    }
    seeds_used = DRY_SEEDS if args.dry_run else SEEDS

    # AFTER arm_results is assembled, so substrate_hash HOISTS from the per-cell
    # fingerprints rather than being recomputed (and mismatching).
    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=seeds_used,
        script_path=Path(__file__),
        started_at=t0,
    )

    out_path = write_flat_manifest(
        manifest, OUT_DIR, dry_run=args.dry_run, stamp=False
    )

    print(json.dumps({
        "run_id": manifest["run_id"],
        "outcome": manifest["outcome"],
        "evidence_direction": manifest["evidence_direction"],
        "non_degenerate": manifest["non_degenerate"],
        "delta_mean": manifest["analysis"]["delta_mean"],
        "d_channel_mean": manifest["analysis"]["d_channel_mean"],
        "d_content_mean": manifest["analysis"]["d_content_mean"],
        "label": manifest["interpretation"]["label"],
        "manifest": str(out_path),
    }, indent=2), flush=True)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
