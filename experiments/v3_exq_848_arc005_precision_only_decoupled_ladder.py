"""V3-EXQ-848: ARC-005 -- does the control plane route PRECISION, decoupled from occupancy?

Successor to V3-EXQ-802. claim_ids = [ARC-005] ONLY.

=== WHY THIS EXPERIMENT EXISTS (802's confound + the fix that unblocks a clean test) ===
802 tested a single 3-level ladder that moved ALL FOUR control-plane channels together
(5-HT seeding gain, phasic-burst temp delta, mode-prior bias, pcc_stability) and required
the SAME joint ladder to move BOTH mode-occupancy AND precision monotonically. It landed
FAIL/mixed: occupancy dissociation was clean and strong (C1 PASS d_channel=1.0, C3 PASS
10/10), but log10_precision_mean was BIT-IDENTICAL across all three levels for every one
of 10 (content,seed) units (rho=0.0, not weak -- exactly zero).

Root-cause diagnosis (chip-20260731-arc005-802-precision-anomaly, documented in
REE_assembly/evidence/planning/arc_005_control_plane_routing_plan.md,
GAP-A-precision-diagnostic node) found THREE separate, now-understood reasons -- and 802's
joint-ladder design meant only TWO of the four channels (1 and 2) were ever architecturally
capable of reaching precision, while the other two (3 and 4) were never expected to (see
this plan doc's own channel table: 1/2 -> precision, 3/4 -> occupancy). Yoking all four
into one ladder therefore diluted any real 1/2 signal with two channels that structurally
cannot move precision -- a design confound independent of any bug.

  (1) Channels 3/4 never feed E3 trajectory scoring at all -- only SalienceCoordinator's
      own mode classification. Scoped OUT of this experiment's precision criterion a
      priori (see SCOPE below), never left to "fail" it.
  (2) Channel 1 (5-HT): a real substrate bug (SD-057 L7 dACC-consume reading
      get_world_state_sequence()[0, 0, :] -- the collapsed proposer STARTING state,
      identical across every candidate) was found and FIXED this session
      (ree_core/agent.py, mirrors the existing ARC-065 GAP-A _candidate_world_summaries()
      pattern). This experiment is the first to exercise the fix
      (use_mech_consume=True, candidate_summary_source="e2_world_forward").
      HOWEVER: quantified verification (same diagnosis session) showed dacc_adapter's
      response to a genuine post-fix goal_proximity signal is still ~1e-8 in scale -- four
      to seven orders of magnitude below the ~0.87 range of the OTHER dACC score_bias
      components -- consistent with dacc_adapter never having been trained on non-degenerate
      values of this input (use_mech_consume has never been ON in any prior training run).
      So channel 1 is EXPECTED to still show no detectable precision effect here, for a
      deeper (untrained-consumer) reason out of scope for this experiment to fix.
  (3) Channel 2 (phasic-burst temperature) is architecturally argmax-invariant under REE's
      normal committed operation: action selection is a deterministic argmin over trajectory
      scores whenever the agent is "committed" (MECH-090 latch), measured at ~97% of ticks
      in a matched-RNG probe (34/35 genuine e3.select() calls committed=True over 310 ticks).
      Temperature only matters in the rare uncommitted stochastic-softmax branch. This
      experiment runs under NORMAL (committed) operation, so channel 2 is ALSO expected to
      show no detectable effect here -- a distinct, already-understood, architectural reason.

=== WHAT THIS EXPERIMENT IS, AND IS NOT ===
This is TRACK A of the three-track remedy the diagnosis settled on (user-confirmed):
  TRACK A (this experiment): decouple the ladder so ONLY channels 1 and 2 vary; channels 3
    and 4 are held at their L0 substrate-default values in EVERY cell. Apply the channel-1
    wiring fix so it gets its best honest post-fix shot. PRE-REGISTER the expectation (from
    (2) and (3) above) that this will likely STILL show a null -- the point is to convert
    802's CONFOUNDED FAIL/mixed into a correctly-scoped, mechanistically-interpretable
    result, not to guess at a different outcome. A null here is a CLEAN CONFIRMATION of an
    already-diagnosed mechanism, not a surprise.
  TRACK B (NOT built here -- a candidate future experiment, noted in the queue entry only):
    test channel 2 under an uncommitted-only regime (MECH-090 commitment disabled), the only
    way to let temperature actually influence a selection. A narrower, differently-framed
    claim ("does temperature matter when the agent is free to express it"), not normal-
    operation routing.
  TRACK C (reflected here, not built): channel 1's null is expected to persist until
    dacc_adapter is exposed to genuine goal_proximity signal during training -- a substrate/
    training project, out of scope for a DV redesign. This experiment's dv_symmetry
    declaration states this explicitly so a null here is not mistaken for "the wiring still
    doesn't work" (the wiring is now correct; the CONSUMER of the signal is untrained).

The mode-occupancy side of ARC-005 does NOT need re-testing: 802's C1/C3 results were clean
and strong and are not in question. GAP-B (V3-EXQ-846, already landed) covers per-channel
occupancy attribution. This experiment is PRECISION-ONLY.

=== DESIGN: SINGLE-FACTOR 3-LEVEL LADDER ON CHANNELS 1+2 ONLY ===
Factor   channel level (1+2 only)   L0 (substrate defaults) / L1 / L2, laddered together
Held fixed EVERY cell               channels 3+4 at their L0 substrate-default values
Content  set A / set B (independent units for the monotonicity criterion, not a
                          dissociation factor -- occupancy dissociation is 802/GAP-B's job)

  cells: L0_A L1_A L2_A L0_B L1_B L2_B  (6 cells x 5 seeds = 30 cells; same cell count as 802)

THE TWO CHANNELS UNDER TEST (channels 3+4 explicitly excluded, see SCOPE):
  1. 5-HT rigidity     serotonin.tonic_5ht_baseline        0.50 -> 1.00 (unchanged from 802)
  2. phasic gain       phasic_burst_temp_delta            -0.10 -> -1.00 (unchanged from 802)
NEW relative to 802 (the channel-1 fix's prerequisites, all default-off, all opt-in here):
  use_mech_consume=True, cfg.goal.z_goal_enabled=True,
  candidate_summary_source="e2_world_forward"

DV: E3 precision readout -- e3.current_precision = 1/(running_variance + 1e-6), recorded as
log10 precision. Precision is driven via the CANONICAL update_residue() path (drives
e3.post_action_update against the winning trajectory's own predicted next-state), NOT 802's
manual e2.world_forward recomputation -- diagnosed (this session) to make no difference to
the invariance finding, but canonical is the correct default going forward and this
experiment also needs the canonical update_z_goal() call (802 never called it; it is the
SOLE z_goal writer, and z_goal must be genuinely live for the channel-1 fix to have any
chance of mattering).

=== DV-SYMMETRY INVARIANCE DECLARATION (mandatory, per arm) ===
  channel 1 (5-HT): reaches dACC's score_bias via candidate_goal_proximity (post-fix, now a
    genuine per-candidate signal, NOT a broadcast constant -- confirmed non-zero
    cross-candidate range post-fix). NOT invariant under the precision DV in principle. This
    experiment's own diagnosis found the downstream adapter's response magnitude is
    currently negligible (~1e-8 vs ~0.87 scale) -- so a null here is an EXPECTED, EXPLAINED
    finding about an untrained consumer network, not evidence the wiring itself is inert.
  channel 2 (phasic gain): a genuine E3 SOFTMAX TEMPERATURE -- a monotone reparameterisation,
    and therefore provably argmax-invariant under DETERMINISTIC (committed) selection, which
    this experiment runs under (measured ~97% commitment rate). NOT invariant in principle
    under an uncommitted/stochastic selection regime (that is TRACK B, not built here). A
    null here is the confirmed, architecturally-expected reading for the committed regime.
  channels 3/4: excluded from this experiment entirely (SCOPE) -- never varied, so no
    DV-symmetry claim is made about them here; see 802/GAP-B for their (occupancy) authority.

=== PRE-REGISTERED CRITERION (constant below; nothing derived from the run) ===
Per (content, seed) unit, over the L0<L1<L2 ladder of channels 1+2:
  C_PRECISION_MONOTONICITY (load-bearing)
    |Spearman rho of log10-precision vs level| >= 0.60   [UNSIGNED -- channels 1 and 2 touch
      precision with opposite-signed intuitions (5-HT rigidity vs phasic sharpening), so only
      monotonicity, not its sign, is the claim -- same convention as 802's C2]
    satisfied in >= 7 of the 10 (content x seed) units

  PASS  = C_PRECISION_MONOTONICITY                -> evidence_direction supports
  FAIL, all 10 units rho in [-0.1, 0.1] (near-zero, not just sub-threshold)
                                                    -> evidence_direction non_contributory
    (the PRE-REGISTERED expected outcome per the DV-symmetry declaration above: a clean,
     mechanistically-explained null, not evidence against ARC-005 -- routed non_contributory,
     never "weakens", exactly the disposition (b) the mandatory_design_check below requires
     for a DV-symmetry-invariant manipulation)
  FAIL, otherwise (some real but sub-threshold trend) -> evidence_direction mixed

=== NON-DEGENERACY (else substrate_not_ready_requeue, NOT a verdict) ===
Per-cell gates via experiments/_lib/precondition_gate.py:
  P1 channel_state_delta_vs_L0 > 0.05  applies_to PERTURBED CELLS ONLY (channels 1+2 moved
       vs the same-content L0 cell). Verifies the settings TOOK EFFECT.
  P2 precision_cross_seed_sd > 1e-6    all cells. A precision readout permanently floor-
       pinned across DIFFERENT SEEDS (different arenas/layouts) would be a substrate-
       readiness failure distinct from -- and prior to -- the monotonicity question this
       experiment asks (that question is about the SAME seed across LEVELS, a different
       axis; see the criterion above).
  P3 n_salience_ticks >= 150           all cells. The coordinator must have actually ticked
       enough for a genuine precision trajectory to accumulate.

=== SCOPE ===
Channels 3 and 4 are DELIBERATELY held fixed at L0 in every cell and are not tested here --
per the plan doc's own channel table they were only ever expected to carry occupancy
authority (confirmed by 802's clean C1/C3), never precision. Including them in a precision
criterion would repeat 802's dilution confound. Their occupancy authority is 802/GAP-B's
result, not re-litigated here. TRACK B (channel 2 under uncommitted selection) and TRACK C
(retraining dacc_adapter on live goal_proximity) are NOT built in this experiment -- see the
module docstring "WHAT THIS EXPERIMENT IS, AND IS NOT" above.

=== NO GRADIENT TRAINING ===
Nothing is trained; no head reads a latent under a loss. The agent is driven in eval()
exactly as the cells are constructed. A WARMUP phase precedes measurement so the precision
EMA and the coordinator's signal EMAs have settled before any datum is recorded.

=== SAMPLE-SIZE INTEGRITY ===
`current_mode`/`current_precision` are agent STATE, held between coordinator ticks; a
per-env-step read is a time-fraction/EMA-continuation, not pseudo-replication. The number of
genuine coordinator ticks is counted (`n_salience_ticks`) and gated (P3) rather than inferred
from the step count. UNIT OF ANALYSIS for the criterion is the SEED (n=5) x CONTENT (n=2).
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
EXPERIMENT_TYPE = "v3_exq_848_arc005_precision_only_decoupled_ladder"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["ARC-005"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
BACKLOG_ID = None
SUPERSEDES_NOTE = (
    "Not a lettered supersession of V3-EXQ-802 (802 stays on record: its C1/C3 occupancy "
    "result is sound and unchallenged). This is a NEW, precision-ONLY, decoupled-ladder "
    "design correcting 802's joint-ladder confound on the precision DV specifically -- "
    "see the module docstring."
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

# --- precision monotonicity criterion ---
C_RHO_ABS_FLOOR = 0.60            # UNSIGNED
C_MIN_UNITS = 7                   # of 10 (content x seed)
C_NULL_BAND = 0.10                # |rho| <= this in ALL 10 units -> non_contributory, not mixed

# --- non-degeneracy floors ---
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

    This is the decoupling fix: 802 laddered all four channels together via
    BASE.channel_settings(level) alone, diluting any real 1/2 signal with two
    channels the plan doc's own channel table never expected to carry precision.
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

    Identical to BASE.agent_kwargs(level) EXCEPT: (a) channels 3+4 are fixed at L0
    (via _precision_channel_settings), and (b) three new flags -- use_mech_consume,
    z_goal_enabled, candidate_summary_source="e2_world_forward" -- give channel 1's
    now-fixed dACC-consume pathway its best honest shot. All three default off/absent
    in BASE.agent_kwargs, so this is a strict superset of new opt-in behaviour, not a
    change to any existing default.
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
        # --- NEW vs 802: the channel-1 fix's prerequisites (all opt-in) ---
        use_mech_consume=True,
        z_goal_enabled=True,
        candidate_summary_source="e2_world_forward",
    )


def _build(seed: int, level: float, content: str):
    """Construct env + agent for one cell, THROUGH the canonical baseline module
    for env/content (unchanged from 802) with the decoupled agent config above."""
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
    agent = REEAgent(cfg)
    agent.eval()
    return agent, env, obs_dict


def _run_cell(
    seed: int, level: float, content: str, n_warmup: int, n_measure: int,
    zg_acc: ZGoalStreamAccumulator,
) -> Dict[str, Any]:
    """Drive one (seed, level, content) cell and return its readouts.

    Uses the CANONICAL update_z_goal() / update_residue() calls (802 used neither --
    z_goal was never live there since use_mech_consume was off, and precision was
    tracked via a manual e2.world_forward recomputation). update_z_goal is the SOLE
    z_goal writer (agent.py); without it goal_state.is_active() never becomes True and
    the channel-1 fix has nothing to activate. update_residue drives the canonical
    e3.post_action_update precision path -- diagnosed (this session) to make no
    difference to the invariance finding vs the manual recomputation, but it is the
    correct default and needed alongside update_z_goal regardless.
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

        # Keep the 5-HT channel LIVE (as 802 did).
        agent.serotonin_step(max(0.0, float(reward)))

        # Canonical precision-update path: resense the post-action observation,
        # then update_residue() drives e3.post_action_update against the winning
        # trajectory's own predicted next-state.
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
                f"  [train] arc005precision {arm} seed={seed} ep {tick + 1}/{n_ticks} "
                f"sal_ticks={n_salience_ticks} steps={n_measured_steps}",
                flush=True,
            )

    zg_acc.observe(agent)

    row: Dict[str, Any] = {
        "arm_id": arm,
        "channel_level": float(level),
        "content": content,
        "seed": int(seed),
        "n_ticks": n_ticks,
        "n_warmup_ticks": n_warmup,
        "n_measured_steps": n_measured_steps,
        "n_salience_ticks": n_salience_ticks,
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
        },
        "goal_state_active_at_end": bool(
            agent.goal_state is not None and agent.goal_state.is_active()
        ),
    }
    return row


# ------------------------------------------------------------------ #
# Precondition specs (regime-conditioned)                             #
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
# Analysis                                                            #
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
                print(
                    f"verdict: {'PASS' if row['n_measured_steps'] > 0 else 'FAIL'}",
                    flush=True,
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
    scorable = len(green) >= len(contexts) - 0  # need ALL 6 cells green to score every unit
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
            label = "control_plane_routes_precision_decoupled"
            evidence_direction = "supports"
        elif analysis["all_near_zero_null"]:
            label = "precision_channel_authority_null_expected"
            evidence_direction = "non_contributory"
        else:
            label = "precision_channel_authority_weak"
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
        },
        "custom_information": {
            "channel_ladder": {str(lv): _precision_channel_settings(lv) for lv in LEVELS},
            "content_sets": BASE.CONTENT_SETS,
            "arena": BASE.ARENA,
            "dv_symmetry_declaration": (
                "channel 1 (5-HT) reaches dACC score_bias via candidate_goal_proximity "
                "(post-fix this session, a genuine non-uniform per-candidate signal) -- "
                "NOT invariant in principle, but this run's own diagnosis found the "
                "downstream dacc_adapter's response magnitude is currently negligible "
                "(untrained consumer), so a null is EXPECTED and EXPLAINED, not a wiring "
                "failure. channel 2 (phasic gain) is a genuine E3 softmax TEMPERATURE, a "
                "monotone reparameterisation and therefore provably argmax-invariant under "
                "the DETERMINISTIC (committed) selection this experiment runs under "
                "(~97% commitment rate measured) -- a null is the architecturally expected "
                "reading for this regime, not a measurement of absence. Channels 3/4 are "
                "held fixed at L0 throughout and make no DV-symmetry claim here -- see SCOPE."
            ),
            "scope_note": (
                "Channels 3 (mode prior) and 4 (pcc_stability) are DELIBERATELY held fixed "
                "at their L0 values in every cell and excluded from this experiment's "
                "criterion a priori -- they were only ever expected to carry OCCUPANCY "
                "authority (802's clean C1/C3), never precision. This decouples this "
                "experiment's precision test from the dilution confound 802's joint ladder "
                "carried. TRACK B (channel 2 under uncommitted/exploratory selection) and "
                "TRACK C (retraining dacc_adapter on live goal_proximity) are explicitly "
                "OUT OF SCOPE here -- see the module docstring."
            ),
            "supersedes_802": SUPERSEDES_NOTE,
            "expected_outcome": (
                "PRE-REGISTERED: given the DV-symmetry declaration above, this experiment "
                "is expected to land near-zero rho on ALL 10 units (all_near_zero_null=True, "
                "evidence_direction=non_contributory), a clean confirmation of an "
                "already-diagnosed mechanism rather than a surprising finding. A different "
                "outcome (PASS, or a real-but-sub-threshold 'mixed' trend) would itself be "
                "decision-relevant and should be flagged for follow-up."
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
        "manifest": str(out_path),
    }, indent=2), flush=True)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
