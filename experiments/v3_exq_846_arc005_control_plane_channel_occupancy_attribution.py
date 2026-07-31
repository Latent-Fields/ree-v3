"""V3-EXQ-846: ARC-005 GAP-B -- WHICH control-plane channel carries mode-occupancy authority?

Plan of record: REE_assembly/evidence/planning/arc_005_control_plane_routing_plan.md,
node arc_005_control_plane_routing:GAP-B (re-scoped 2026-07-31, reconcile chip
chip-20260731-arc005-802-reconcile). claim_ids = [ARC-005] ONLY.

=== WHY THIS RUN EXISTS, AND WHY IT IS SCOPED TO OCCUPANCY ONLY ===
V3-EXQ-802 (GAP-A) established that the control plane, as a WHOLE, has causal authority
over mode occupancy (C1 PASS, d_channel_mean=1.0 vs d_content_mean=0.0 -- channel settings
move the discrete mode end-to-end while content moves it not at all) but showed literally
ZERO measured response of the continuous precision readout (log10_precision_mean) to any
of the four channels -- bit-identical across L0/L1/L2 for every (content, seed) pair. 802's
own docstring and manifest custom_information.channel_attribution_limit both state that
PER-CHANNEL dissociation is UNTESTED: a C1 PASS licenses "the plane routes" but not "each
channel independently routes". This run answers that question for OCCUPANCY. It does NOT
attempt to answer it for precision: building a leave-one-out contrast on log10_precision_mean
would measure a difference of two constant (non-responsive) terms -- an ARITHMETIC IDENTITY,
not a measurement, for every one of the four channels in this harness (confirmed empirically
by 802, not merely anticipated). Precision is still RECORDED per the generous-recording
convention (per-arm log10_precision_mean/sd, exactly as 802 recorded it), but no criterion in
this script is scored on it; any precision-side reading is pre-declared non_contributory
under substrate_ceiling, per the plan doc's mandatory_design_check.

=== DESIGN: LEAVE-ONE-OUT AT L2, CONTENT SET A ONLY ===
802 already showed content has ~zero effect on occupancy (d_content_mean=0.0), so a single
content set suffices for channel attribution -- adding content B would not discriminate
channels, only re-confirm content-invariance GAP-A already established.

  ARM_ALL_ON    all four channels at L2 (fully perturbed)         = 802's L2_A arm exactly
  ARM_5HT_OFF   channel 1 (5-HT) at L0, channels 2/3/4 at L2
  ARM_PHASIC_OFF channel 2 (phasic burst) at L0, channels 1/3/4 at L2
  ARM_MODEPRIOR_OFF channel 3 (mode prior) at L0, channels 1/2/4 at L2
  ARM_PCC_OFF   channel 4 (pcc_stability) at L0, channels 1/2/3 at L2
  ARM_ALL_OFF   all four channels at L0 (substrate baseline)      = 802's L0_A arm exactly

  6 arms x 5 seeds = 30 cells. ARM_ALL_ON and ARM_ALL_OFF are BYTE-IDENTICAL in construction
  to 802's L2_A / L0_A cells (built via BASE.cell_config_slice(level, "A") unchanged), so
  BOTH are attempted for reuse via try_reuse_cell before falling back to a fresh run (see
  REUSE below) -- an extension beyond the plan doc's minimum ask of reusing only ARM_ALL_OFF,
  justified because the mechanism, justification and cost are identical for both cells and
  GOV-REUSE-1 favours reuse whenever it is available.

THE FOUR CHANNELS, per-channel config key(s) held at L0 for that arm's ablation (all other
keys taken from BASE.channel_settings(L2) unchanged) -- see
experiments/_lib/baselines/exq802_arc005_control_plane.py for the exact formulas:
  1. 5-HT rigidity     serotonin_seeding_gain, serotonin_wanting_floor
  2. phasic-burst gain  phasic_burst_temp_delta
  3. mode prior         salience_external_task_bias
  4. pcc_stability mu   pcc_stability_baseline

DV: mode-occupancy distribution only (time-fraction of measurement steps in each of
{external_task, internal_planning, internal_replay, offline_consolidation}, read from
salience.current_mode). log10 precision is recorded but not scored (see above).

=== SCORING: PER-CHANNEL OCCUPANCY-AUTHORITY DROP ===
Per seed s, with TV = total-variation distance between occupancy vectors (occupancy-fraction
units, range [0, 1]), and arms abbreviated ON=ARM_ALL_ON, OFF=ARM_ALL_OFF:

    full_effect(s)        = TV( occ(ON, s), occ(OFF, s) )
    retained_effect_X(s)  = TV( occ(X_OFF, s), occ(OFF, s) )      for X in {5HT, PHASIC,
                                                                    MODEPRIOR, PCC}
    authority_X(s)        = full_effect(s) - retained_effect_X(s)

authority_X(s) is the DROP in the channel-vs-baseline occupancy effect attributable to
channel X: if X_OFF's occupancy collapses back toward ARM_ALL_OFF (retained_effect_X -> 0),
authority_X -> full_effect (X was carrying essentially all of the routing); if X_OFF still
looks like ARM_ALL_ON (retained_effect_X -> full_effect), authority_X -> 0 (X was not
necessary for the routing observed). This is exactly the "drop in the channel-vs-L0
occupancy effect when that channel alone is returned to baseline vs ARM_ALL_ON" the plan
doc's design_sketch specifies.

Secondary (non-gating) diagnostic per channel: argmax_reverts_frac -- the fraction of seeds
in which X_OFF's argmax mode equals ARM_ALL_OFF's argmax mode (a discrete corroboration of
the continuous TV-based authority score, in the same spirit as 802's C3).

=== DV-SYMMETRY INVARIANCE DECLARATION (mandatory, per the 5 non-baseline arms) ===
The occupancy DV is a set-aggregate (time-fraction) over the discrete `current_mode`
sequence; its symmetry group is relabellings that preserve inter-mode TV distance. A
manipulation is safe to score only if it is NOT invariant under that group (else any
measured delta is an arithmetic identity, not a measurement -- the V3-EXQ-604c failure
class the plan doc's mandatory_design_check names explicitly for this lineage).

  ARM_ALL_ON (all four channels at L2): identical construction to 802's L2_A arm, whose
    occupancy (measured, not assumed) was pure external_task (TV=1.0 from L0_A's pure
    internal_planning). Channel 3 alone (a per-mode, non-broadcast logit shift on
    external_task ONLY -- salience_coordinator.py) is sufficient to break both argmax and
    softmax invariance; channel 1 reshapes the dACC bundle the coordinator aggregates
    (neither uniform-additive nor a monotone rescaling); channel 4's MECH-259 threshold leg
    changes switch admission under hysteresis (not invariant under a time-fraction DV). Not
    an identity -- already empirically confirmed non-invariant by 802.

  ARM_5HT_OFF (channels 2/3/4 ON at L2, channel 1 OFF at L0): channel 3's non-broadcast
    external_task logit shift remains fully active regardless of channel 1's state, so this
    arm cannot be occupancy-invariant. The measured authority_5HT is channel 1's OWN marginal
    contribution against that active channel-3 backdrop -- a genuine (possibly small)
    measurement, not a structural zero.

  ARM_PHASIC_OFF (channels 1/3/4 ON at L2, channel 2 OFF at L0): channel 3 remains active
    (see above) -> this arm is not occupancy-invariant as a WHOLE. Channel 2 in isolation is
    the specific case the plan doc's mandatory_design_check flags: it is a PURE E3
    softmax-temperature knob (phasic_surprise_burst.py) with NO direct entry point into
    SalienceCoordinator's logit/threshold computation (unlike channels 1/3/4, which all have
    one) -- so a naive argument would call it argmax-invariant. But E3's live selection path
    samples via torch.multinomial (CLAUDE.md "Running the test suite" / cross-machine-class
    divergence note; ree_core/predictors/e3_selector.py), NOT deterministic argmax over the
    temperature-scaled scores -- so a temperature change genuinely alters the REALISED
    (stochastically sampled) action sequence at a fixed seed, hence the environment
    trajectory, hence the dACC/AIC inputs SalienceCoordinator's aggregate reads on
    subsequent ticks. Channel 2's occupancy pathway is therefore real but INDIRECT and
    state-dependent, not a provable identity -- 802 itself declared exactly this ("declared
    as contributing to precision, NOT independently to occupancy", not "provably zero on
    occupancy"). If authority_PHASIC measures near-zero, that is a genuine substrate finding
    (channel 2 has no material occupancy pathway), not an arithmetic default -- scored
    normally, never pre-routed to substrate_ceiling.

  ARM_MODEPRIOR_OFF (channels 1/2/4 ON at L2, channel 3 OFF at L0): the critical leg -- does
    the channel 802 flagged as having DIRECT occupancy authority actually carry the effect
    alone? With channel 3 removed, channel 1 (dACC reshape, non-broadcast) and channel 4's
    MECH-259 threshold leg (switch-admission change, non-invariant under a time-fraction DV)
    remain active at L2 -- so this arm is NOT occupancy-invariant even with channel 3 gone;
    a null or reduced result here is a genuine measurement of how much of the routing
    survives on the OTHER three channels, not an identity.

  ARM_PCC_OFF (channels 1/2/3 ON at L2, channel 4 OFF at L0): channel 3 remains fully active
    -> not occupancy-invariant. Channel 4 as ablated here is its WHOLE setting
    (pcc_stability_baseline), which drives BOTH the MECH-048 mu leg (a softmax temperature on
    the mode prior -- THAT leg alone would be argmax-invariant, which is exactly why the plan
    doc's mandatory_design_check calls out "channel 4-mu" specifically as a risk) AND the
    MECH-259 threshold leg (switch-admission, non-invariant) SIMULTANEOUSLY -- both legs share
    the single `pcc_stability_baseline` scalar in this substrate, so this arm ablates the
    threshold leg too, not "only mu". It is therefore not at the arithmetic-identity risk the
    mandatory_design_check names; that risk would apply only to a design that isolated the mu
    leg alone, which this design does not attempt (no channel-4-leg-splitting arm is queued
    here -- out of scope, noted under custom_information.leg_split_untested).

No arm is scoped out on symmetry grounds. If a channel's authority estimate later reads a
provable identity from something not anticipated above (e.g. a design bug pins it), that
result is routed non_contributory/substrate_ceiling, never 'mixed' -- but nothing in the
substrate as verified above forces that outcome a priori.

=== PRE-REGISTERED CRITERIA (constants below; nothing derived from the run) ===
Per channel X in {5HT, PHASIC, MODEPRIOR, PCC}, using the SAME gate shape 802 used for its
own dissociation criterion (continuity of thresholds across the lineage):
    C1_X ATTRIBUTION: mean_s authority_X >= 0.8 * SD_s(authority_X)  AND  mean_s authority_X
                       >= 0.15
    (load-bearing, one per channel -- this is the whole point of a leave-one-out ablation:
    each channel's contribution is its own finding, not a joint conjunction)

  attribution_achieved = C1_5HT or C1_PHASIC or C1_MODEPRIOR or C1_PCC (>= 1 channel
    individually clears the gate)

  PASS  = attribution_achieved (>= 1 channel individually carries occupancy authority)
          -> evidence_direction supports (ARC-005's routing is attributable to specific
          channel(s); if the channels are 3 and/or 4 as the plan doc's readiness-probe
          EXPECTATION predicted, note that in interpretation; if channel 1 or 2 unexpectedly
          clears the gate, that STILL supports ARC-005 -- it is a stronger, more distributed
          routing than expected, not evidence against it. Expectation-mismatch is flagged
          under custom_information.expectation_match, not routed to a different direction.)
  FAIL  = not attribution_achieved (no channel individually clears the gate; the routing 802
          established is a JOINT/non-decomposable effect, not attributable to any single
          channel in isolation) -> evidence_direction mixed. This does NOT weaken ARC-005 --
          802's own joint C1 result stands unchanged -- it means GAP-B's decomposition
          question is answered in the negative (no single channel is individually
          sufficient), which is itself informative.

=== NON-DEGENERACY (else substrate_not_ready_requeue, NOT a verdict) ===
Regime-conditioned via experiments/_lib/precondition_gate.py, mirroring 802's structure:
  P1 n_distinct_argmax_modes_across_design >= 2   all arms (DESIGN-level scalar, carried
       over from 802's substrate_notes -- gates the run as a whole, never a single arm).
  P2 channel_ablation_component_delta > 0.05      applies_to the 4 leave-one-out arms ONLY.
       Verifies the ablated channel's OWN realised setting actually moved back toward L0
       relative to ARM_ALL_ON (the manipulation took effect), not merely that it was passed
       to the constructor. Scoped out for ARM_ALL_ON/ARM_ALL_OFF (structurally 0 there --
       disposition (a), not a substrate failure).
  P3 n_salience_ticks >= 150                      all arms (worst cell). Same floor as 802 --
       the coordinator must have actually ticked enough for an occupancy distribution to
       mean anything.
Precision is NOT gated (see "WHY THIS RUN EXISTS" above) -- 802's P2 (precision_cross_seed_sd)
has no analogue here because precision is out of scope for scoring in this design.

=== SAMPLE-SIZE INTEGRITY ===
Identical to 802: current_mode is agent STATE (not a latched last_* diagnostic), so a
per-env-step occupancy read is a time-fraction, not pseudo-replication. n_salience_ticks
(identity-change count of agent._salience_last_tick) is recorded as the honest independent
tick denominator. Unit of analysis for every criterion is the SEED (n=5).

=== NO GRADIENT TRAINING ===
Nothing is trained; the agent is driven in eval() exactly as 802's arms were. Phased-training
protocol does not apply. A WARMUP phase (identical length to 802: 200 ticks) precedes
measurement so the precision EMA and coordinator signal EMAs settle first.

=== REUSE (GOV-REUSE-1 / arm-reuse fingerprint) ===
ARM_ALL_OFF (L0, content A) and ARM_ALL_ON (L2, content A) are attempted for reuse via
try_reuse_cell, citing reuse_baseline_from=v3_exq_802_arc005_control_plane_routing_double_
dissociation_20260722T212125Z_v3, include_driver_script_in_hash=False, config_slice built
via BASE.cell_config_slice(level, "A") verbatim (byte-identical to 802's own L0_A/L2_A
construction). NOTE (recorded here for the record, not merely at run time): 19 commits have
touched ree_core/** since 802 landed (2026-07-22), none confirmed to touch the four channel
modules or agent.py's selection path in a way that changes THIS design's behaviour, but the
default whole-tree substrate_hash makes reuse HIT-sensitive to ALL of ree_core, not just the
files this design reads -- so a REFUSAL (fingerprint mismatch -> run fresh) is the likely,
safe outcome, not a bug. Refuse-by-default: either outcome is correct.

=== GOV-REUSE-1 (decisive-readout recorded-manifest check) ===
Decisive readout: per-channel occupancy-authority drop (authority_X per channel, defined
above). This exact per-channel decomposition appears in ZERO recorded manifests -- 802 is
the only 'mode_occupancy'-bearing manifest for ARC-005 and it explicitly declares per-channel
attribution UNTESTED (custom_information.channel_attribution_limit). Not recoverable from
existing evidence except via the two whole-arm reuses above. Run (the residual, non-reusable
part: 4 leave-one-out arms x 5 seeds = 20 fresh cells).

=== MINT-AS-YOU-GO ===
The 4 leave-one-out arms are fingerprinted with include_driver_script_in_hash=False (lineage
exq802_arc005_control_plane, same lineage as 802 -- these are new CELLS of that lineage, not
a new lineage), so a future sibling with a different driver can reuse them too.
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
from experiments._lib.arm_reuse import try_reuse_cell  # noqa: E402
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
EXPERIMENT_TYPE = "v3_exq_846_arc005_control_plane_channel_occupancy_attribution"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["ARC-005"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

OUT_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

REUSE_CITE_RUN_ID = (
    "v3_exq_802_arc005_control_plane_routing_double_dissociation_20260722T212125Z_v3"
)
LINEAGE = "exq802_arc005_control_plane"

# ------------------------------------------------------------------ #
# Pre-registered constants (NOT derived from the run's own statistics) #
# ------------------------------------------------------------------ #
SEEDS = [0, 1, 2, 3, 4]
CONTENT = "A"
L0, L2 = BASE.CHANNEL_LEVELS[0], BASE.CHANNEL_LEVELS[-1]
WARMUP_TICKS = BASE.WARMUP_TICKS
MEASURE_TICKS = BASE.MEASURE_TICKS
TOTAL_TICKS = BASE.TOTAL_TICKS

DRY_WARMUP_TICKS = 10
DRY_MEASURE_TICKS = 40
DRY_SEEDS = [0, 1]

MODE_NAMES = [
    "external_task", "internal_planning", "internal_replay", "offline_consolidation",
]

# Channel identity: name -> BASE.channel_settings() key(s) that channel controls, and
# the index into the 4-component normalised state vector (mirrors 802's
# _channel_state_vector ordering: 5ht, phasic, modeprior, pcc).
CHANNELS: Dict[str, Dict[str, Any]] = {
    "5HT": {"keys": ["serotonin_seeding_gain", "serotonin_wanting_floor"], "vec_idx": 0},
    "PHASIC": {"keys": ["phasic_burst_temp_delta"], "vec_idx": 1},
    "MODEPRIOR": {"keys": ["salience_external_task_bias"], "vec_idx": 2},
    "PCC": {"keys": ["pcc_stability_baseline"], "vec_idx": 3},
}
CHANNEL_ORDER = ["5HT", "PHASIC", "MODEPRIOR", "PCC"]

ARM_ALL_ON = "ARM_ALL_ON"
ARM_ALL_OFF = "ARM_ALL_OFF"
LEAVE_ONE_OUT_ARMS = {f"ARM_{c}_OFF": c for c in CHANNEL_ORDER}
ALL_ARM_IDS = [ARM_ALL_ON] + list(LEAVE_ONE_OUT_ARMS.keys()) + [ARM_ALL_OFF]

# --- C1 per-channel attribution (same gate shape as 802's C1) ---
C1_SD_MULTIPLIER = 0.8
C1_ABS_FLOOR = 0.15

# --- non-degeneracy floors ---
P1_DISTINCT_MODES_FLOOR = 1.5      # design-level: strictly more than one distinct mode
P2_CHANNEL_ABLATION_DELTA_FLOOR = 0.05
P3_SALIENCE_TICK_FLOOR = 150.0
DRY_P3_SALIENCE_TICK_FLOOR = 5.0   # smoke only -- criteria are not scored on a smoke

NEEDED_REUSE_KEYS = [
    "mode_occupancy", "argmax_mode", "n_salience_ticks", "realised_channel_state",
    "log10_precision_mean", "n_modes_occupied_ge_5pct",
]


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #
def _tv_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Total-variation distance between two occupancy vectors, in [0, 1]."""
    return 0.5 * sum(abs(float(p.get(m, 0.0)) - float(q.get(m, 0.0))) for m in MODE_NAMES)


def _channel_settings_for_arm(arm_id: str) -> Dict[str, Any]:
    """The four channel-key settings for one arm: all keys at L2 EXCEPT the ablated
    channel's key(s) (if any), which sit at L0. Values are taken VERBATIM from
    BASE.channel_settings(level) at the appropriate level for each key -- never
    re-derived -- so a change to the substrate's channel formulas in the shared _lib
    baseline module propagates here automatically rather than silently drifting.
    """
    on = BASE.channel_settings(L2)
    off = BASE.channel_settings(L0)
    if arm_id == ARM_ALL_ON:
        return dict(on)
    if arm_id == ARM_ALL_OFF:
        return dict(off)
    channel = LEAVE_ONE_OUT_ARMS[arm_id]
    result = dict(on)
    for key in CHANNELS[channel]["keys"]:
        result[key] = off[key]
    return result


def _agent_kwargs_for_arm(arm_id: str) -> Dict[str, Any]:
    """Mirrors BASE.agent_kwargs(level)'s structure exactly, but sourced from a
    per-key mixed channel-settings dict (_channel_settings_for_arm) instead of a
    single scalar level -- BASE.agent_kwargs only accepts one level for all four
    channels at once, which the leave-one-out design cannot use directly.
    """
    ch = _channel_settings_for_arm(arm_id)
    return dict(
        alpha_world=0.9,
        tonic_5ht_enabled=True,
        use_phasic_burst=True,
        phasic_burst_temp_delta=ch["phasic_burst_temp_delta"],
        phasic_burst_signal_source="instantaneous_pe",
        phasic_burst_baseline_continuity="carry",
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
        use_harm_stream=True,
        use_affective_harm_stream=True,
        use_support_preserving_cem=True,
        support_preserving_min_first_action_classes=2,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=4.0,
    )


def _cell_config_slice(arm_id: str, seed: int) -> Dict[str, Any]:
    """Declared config slice for one cell's fingerprint.

    ARM_ALL_ON / ARM_ALL_OFF return BASE.cell_config_slice(L2/L0, "A") VERBATIM,
    including its "L2_A"/"L0_A" internal arm_id and "channel_level" field -- this is
    load-bearing, not cosmetic: compute_arm_fingerprint hashes this dict, and it must
    be byte-identical to what 802 hashed for its own L2_A/L0_A cells or the reuse
    lookup can never hit even when the substrate hasn't drifted. The row's own
    displayed arm_id (ARM_ALL_ON/ARM_ALL_OFF) is set separately in _run_cell / after
    a reuse hit -- unrelated to this dict's internal "arm_id" fingerprint key.

    Leave-one-out arms have no single BASE.cell_config_slice analogue (two different
    levels are active at once), so they declare their own slice -- lineage, the
    ablated channel + both ladder levels, content, env/agent kwargs, schedule. These
    are never reuse targets against 802 (802 never ran a leave-one-out arm); they are
    fresh mint cells for this lineage, fingerprinted the same way as every other
    Phase-0 emission.
    """
    if arm_id == ARM_ALL_ON:
        return BASE.cell_config_slice(L2, CONTENT)
    if arm_id == ARM_ALL_OFF:
        return BASE.cell_config_slice(L0, CONTENT)
    channel = LEAVE_ONE_OUT_ARMS[arm_id]
    env_kw = dict(BASE.ARENA)
    spec = BASE.CONTENT_SETS[CONTENT]
    env_kw.update(
        num_hazards=spec["num_hazards"],
        num_resources=spec["num_resources"],
        contamination_spread=spec["contamination_spread"],
        content_seed_offset=spec["seed_offset"],
    )
    return {
        "lineage": LINEAGE,
        "arm_id": arm_id,
        "ablated_channel": channel,
        "on_level": L2,
        "off_level": L0,
        "content": CONTENT,
        "env_kwargs": env_kw,
        "agent_kwargs": _agent_kwargs_for_arm(arm_id),
        "schedule": {
            "warmup_ticks": WARMUP_TICKS,
            "measure_ticks": MEASURE_TICKS,
            "total_ticks": TOTAL_TICKS,
        },
    }


def _build(seed: int, arm_id: str):
    """Construct env + agent for one cell."""
    env_kw = BASE.content_env_kwargs(CONTENT, seed)
    env = CausalGridWorld(**env_kw)
    _obs, obs_dict = env.reset()

    kw: Dict[str, Any] = dict(_agent_kwargs_for_arm(arm_id))
    kw.update(
        body_obs_dim=obs_dict["body_state"].shape[-1],
        world_obs_dim=obs_dict["world_state"].shape[-1],
        action_dim=env.action_dim,
    )
    cfg = REEConfig.from_dims(**kw)
    # Channel 1 (5-HT) lives on the nested SerotoninConfig, which from_dims does not
    # expose as scalar kwargs -- same pin 802 used, at the appropriate per-arm level.
    ch = _channel_settings_for_arm(arm_id)
    cfg.serotonin.gain_min = cfg.serotonin.gain_max = float(ch["serotonin_seeding_gain"])
    cfg.serotonin.floor_min = cfg.serotonin.floor_max = float(ch["serotonin_wanting_floor"])
    agent = REEAgent(cfg)
    agent.eval()
    return agent, env, obs_dict


_ZG = ZGoalStreamAccumulator()


def _run_cell(seed: int, arm_id: str, n_warmup: int, n_measure: int) -> Dict[str, Any]:
    """Drive one (seed, arm) cell and return its readouts. Structurally identical to
    802's _run_cell -- same measured fields, same driven-loop mechanics -- just keyed
    by arm_id (which now encodes a per-channel ablation) instead of (level, content).
    """
    agent, env, obs_dict = _build(seed, arm_id)
    n_ticks = n_warmup + n_measure

    print(f"Seed {seed} Condition {arm_id}", flush=True)

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
                f"  [train] arc005gapb {arm_id} seed={seed} ep {tick + 1}/{n_ticks} "
                f"sal_ticks={n_salience_ticks} steps={n_measured_steps}",
                flush=True,
            )

    denom = max(1, n_measured_steps)
    occupancy = {m: mode_counts[m] / denom for m in MODE_NAMES}

    row: Dict[str, Any] = {
        "arm_id": arm_id,
        "content": CONTENT,
        "seed": int(seed),
        "n_ticks": n_ticks,
        "n_warmup_ticks": n_warmup,
        "n_measured_steps": n_measured_steps,
        "n_salience_ticks": n_salience_ticks,
        "mode_occupancy": {k: round(v, 6) for k, v in occupancy.items()},
        "mode_counts": dict(mode_counts),
        "n_modes_occupied_ge_5pct": sum(
            1 for v in occupancy.values() if v >= 0.05
        ),
        "argmax_mode": max(occupancy.items(), key=lambda kv: kv[1])[0],
        "log10_precision_mean": round(float(np.mean(log_precisions)), 8)
        if log_precisions else 0.0,
        "log10_precision_sd": round(float(np.std(log_precisions)), 8)
        if len(log_precisions) > 1 else 0.0,
        "realised_channel_state": {
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
    _ZG.observe(agent)  # AFTER stepping -- reads the z_goal counters at call time
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
            name="channel_ablation_component_delta",
            description=(
                "normalised delta of the ABLATED channel's own realised setting vs "
                "ARM_ALL_ON, isolated to that channel's component of the state vector"
            ),
            control="ARM_ALL_ON (same-content, all channels at L2)",
            threshold=P2_CHANNEL_ABLATION_DELTA_FLOOR,
            applies_to=lambda ctx: ctx["id"] in LEAVE_ONE_OUT_ARMS,
            applies_note=(
                "leave-one-out arms only -- ARM_ALL_ON/ARM_ALL_OFF have no ablated "
                "channel to measure a delta for, hence 0 by construction there, not "
                "by substrate failure"
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
    return [{"id": a} for a in ALL_ARM_IDS]


def _worst_cell(rows: List[Dict[str, Any]], key: str) -> Tuple[float, str]:
    best: Optional[float] = None
    who = ""
    for r in rows:
        v = float(r[key]) if not isinstance(r[key], dict) else 0.0
        if best is None or v < best:
            best, who = v, f"seed={r['seed']}"
    return (float(best) if best is not None else 0.0), who


def _channel_state_vector(row: Dict[str, Any]) -> np.ndarray:
    """Same normalisation 802 used, so the 0.05 floor means the same thing here."""
    s = row["realised_channel_state"]
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
    by: Dict[Tuple[str, int], Dict[str, Any]] = {(r["arm_id"], r["seed"]): r for r in rows}

    per_channel: Dict[str, Any] = {}
    per_seed_full_effect: List[Dict[str, Any]] = []
    full_effects: List[float] = []
    for s in seeds:
        try:
            fe = _tv_distance(
                by[(ARM_ALL_ON, s)]["mode_occupancy"], by[(ARM_ALL_OFF, s)]["mode_occupancy"]
            )
        except KeyError:
            continue
        full_effects.append(fe)
        per_seed_full_effect.append({"seed": s, "full_effect": round(fe, 6)})
    full_effect_mean = float(statistics.fmean(full_effects)) if full_effects else 0.0

    for channel in CHANNEL_ORDER:
        arm_id = f"ARM_{channel}_OFF"
        authorities: List[float] = []
        retained: List[float] = []
        argmax_reverts = 0
        n_units = 0
        per_seed: List[Dict[str, Any]] = []
        for s in seeds:
            try:
                off_row = by[(ARM_ALL_OFF, s)]
                on_row = by[(ARM_ALL_ON, s)]
                x_row = by[(arm_id, s)]
            except KeyError:
                continue
            fe = _tv_distance(on_row["mode_occupancy"], off_row["mode_occupancy"])
            re_ = _tv_distance(x_row["mode_occupancy"], off_row["mode_occupancy"])
            auth = fe - re_
            authorities.append(auth)
            retained.append(re_)
            n_units += 1
            reverts = bool(x_row["argmax_mode"] == off_row["argmax_mode"])
            if reverts:
                argmax_reverts += 1
            per_seed.append(
                {"seed": s, "full_effect": round(fe, 6), "retained_effect": round(re_, 6),
                 "authority": round(auth, 6), "argmax_reverts_to_baseline": reverts}
            )
        auth_mean = float(statistics.fmean(authorities)) if authorities else 0.0
        auth_sd = float(statistics.stdev(authorities)) if len(authorities) > 1 else 0.0
        gate = C1_SD_MULTIPLIER * auth_sd
        c1_pass = bool(authorities) and (auth_mean >= gate) and (auth_mean >= C1_ABS_FLOOR)
        per_channel[channel] = {
            "authority_mean": round(auth_mean, 6),
            "authority_sd": round(auth_sd, 6),
            "retained_effect_mean": round(
                float(statistics.fmean(retained)), 6
            ) if retained else 0.0,
            "c1_sd_gate": round(gate, 6),
            "c1_abs_floor": C1_ABS_FLOOR,
            "c1_pass": c1_pass,
            "argmax_reverts_frac": round(argmax_reverts / n_units, 4) if n_units else 0.0,
            "n_units": n_units,
            "per_seed": per_seed,
        }

    attribution_achieved = any(v["c1_pass"] for v in per_channel.values())

    return {
        "full_effect_mean": round(full_effect_mean, 6),
        "per_seed_full_effect": per_seed_full_effect,
        "per_channel": per_channel,
        "attribution_achieved": attribution_achieved,
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
    contexts = _arm_contexts()
    assert_no_structurally_unsatisfiable_gate(specs, contexts)

    # ARM_ALL_ON / ARM_ALL_OFF's fingerprint slice must be BYTE-IDENTICAL to 802's own
    # L2_A / L0_A slices (_cell_config_slice returns BASE.cell_config_slice(...)
    # verbatim for these two arm_ids -- see its docstring), or the reuse attempt below
    # can never hit even when the substrate hasn't drifted. Also confirm the
    # constructed AGENT config (what _build actually uses) agrees, independent of the
    # fingerprint-slice plumbing -- a divergence here would mean this script measures
    # a different cell than the one it fingerprints as, which the fingerprint alone
    # cannot catch.
    assert _cell_config_slice(ARM_ALL_ON, 0) == BASE.cell_config_slice(L2, CONTENT), (
        "ARM_ALL_ON fingerprint slice diverged from BASE.cell_config_slice(L2, 'A') -- "
        "reuse of 802's L2_A cells would never hit even on a stable substrate"
    )
    assert _cell_config_slice(ARM_ALL_OFF, 0) == BASE.cell_config_slice(L0, CONTENT), (
        "ARM_ALL_OFF fingerprint slice diverged from BASE.cell_config_slice(L0, 'A') -- "
        "reuse of 802's L0_A cells would never hit even on a stable substrate"
    )
    assert _agent_kwargs_for_arm(ARM_ALL_ON) == BASE.agent_kwargs(L2), (
        "ARM_ALL_ON's constructed agent_kwargs diverged from BASE.agent_kwargs(L2)"
    )
    assert _agent_kwargs_for_arm(ARM_ALL_OFF) == BASE.agent_kwargs(L0), (
        "ARM_ALL_OFF's constructed agent_kwargs diverged from BASE.agent_kwargs(L0)"
    )
    assert BASE.cell_config_slice(L0, CONTENT) == BASE.off_path_config_slice(), (
        "BASE's own L0_A slice diverged from off_path_config_slice() -- upstream "
        "invariant from 802 no longer holds"
    )

    rows: List[Dict[str, Any]] = []
    reuse_log: List[str] = []
    for arm_id in ALL_ARM_IDS:
        for seed in seeds:
            slice_ = _cell_config_slice(arm_id, seed)
            reused_row: Optional[Dict[str, Any]] = None
            if arm_id in (ARM_ALL_ON, ARM_ALL_OFF):
                # Attempted on --dry-run too (seeds 0/1 are a subset of 802's [0..4]) so
                # the smoke exercises this code path; a MISS (expected -- see the
                # substrate-drift note in custom_information.reuse) falls back to
                # _run_cell exactly like a real run would, so coverage is unaffected
                # either way.
                reused_row = try_reuse_cell(
                    config_slice=slice_,
                    seed=seed,
                    script_path=Path(__file__),
                    needed_keys=NEEDED_REUSE_KEYS,
                    cite_run_id=REUSE_CITE_RUN_ID,
                    include_driver_script_in_hash=False,
                    logger=lambda m: reuse_log.append(m),
                )
            if reused_row is not None:
                reused_row["arm_id"] = arm_id  # local arm-id spelling, not 802's L0_A/L2_A
                rows.append(reused_row)
                print(f"verdict: PASS (reused seed={seed} arm={arm_id})", flush=True)
                continue

            with arm_cell(
                seed,
                config_slice=slice_,
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=False,
            ) as cell:
                row = _run_cell(seed, arm_id, n_warmup, n_measure)
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

    vecs = [_channel_state_vector(r) for r in by_arm.get(ARM_ALL_ON, [])]
    all_on_vec_mean = np.mean(np.stack(vecs), axis=0) if vecs else np.zeros(4, dtype=float)

    distinct_modes = float(len({r["argmax_mode"] for r in rows}))

    arm_gates = []
    offending: Dict[str, str] = {}
    for ctx in contexts:
        arm_id = ctx["id"]
        arm_rows = by_arm.get(arm_id, [])
        if not arm_rows:
            continue
        ticks_worst, who_ticks = _worst_cell(arm_rows, "n_salience_ticks")
        measured: Dict[str, float] = {
            "n_distinct_argmax_modes_across_design": distinct_modes,
            "n_salience_ticks": ticks_worst,
        }
        offending[arm_id] = f"ticks:{who_ticks}"
        if arm_id in LEAVE_ONE_OUT_ARMS:
            channel = LEAVE_ONE_OUT_ARMS[arm_id]
            idx = CHANNELS[channel]["vec_idx"]
            vec = np.mean(np.stack([_channel_state_vector(r) for r in arm_rows]), axis=0)
            component_delta = float(abs(vec[idx] - all_on_vec_mean[idx]))
            measured["channel_ablation_component_delta"] = component_delta
        else:
            measured["channel_ablation_component_delta"] = 0.0  # scoped out for these arms
        arm_gates.append(evaluate_arm_gate(arm_id, ctx, specs, measured=measured))
    gate = aggregate_arm_gates(arm_gates)
    green = set(gate["green_arms"])

    # All 6 arms must be green for the leave-one-out contrasts to be meaningful --
    # every channel's authority score reads ARM_ALL_ON and ARM_ALL_OFF, so those two
    # are load-bearing for every channel; each leave-one-out arm is load-bearing only
    # for its own channel's C1.
    red_core = [a for a in (ARM_ALL_ON, ARM_ALL_OFF) if a not in green]
    red_leave_one_out = [a for a in LEAVE_ONE_OUT_ARMS if a not in green]
    scorable_channels = {
        c: (f"ARM_{c}_OFF" not in red_leave_one_out) and not red_core
        for c in CHANNEL_ORDER
    }
    scorable = not red_core and any(scorable_channels.values())

    analysis = _analyse(rows, seeds)

    if not scorable:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
        non_degenerate = False
        degeneracy_reason = (
            "core arm(s) failed their non-degeneracy gate: " + ", ".join(red_core)
            + f" (green arms: {sorted(green)})"
            if red_core else
            "every leave-one-out arm failed its non-degeneracy gate: "
            + ", ".join(red_leave_one_out) + f" (green arms: {sorted(green)})"
        )
        overall_pass = False
    else:
        attribution = analysis["attribution_achieved"]
        overall_pass = bool(attribution)
        outcome = "PASS" if overall_pass else "FAIL"
        contributing = [c for c in CHANNEL_ORDER if analysis["per_channel"][c]["c1_pass"]]
        if overall_pass:
            label = "channel_occupancy_authority_attributed"
            evidence_direction = "supports"
        else:
            label = "occupancy_authority_not_individually_attributable"
            evidence_direction = "mixed"
        non_degenerate = True
        degeneracy_reason = ""
        if red_leave_one_out:
            degeneracy_reason = (
                "channel(s) not scorable (leave-one-out arm red): "
                + ", ".join(sorted({LEAVE_ONE_OUT_ARMS.get(a, a) for a in red_leave_one_out}))
                + ". Remaining channels' attribution stands."
            )

    criteria = [
        {"name": f"C1_{c}_attribution", "load_bearing": True,
         "passed": bool(analysis["per_channel"][c]["c1_pass"])}
        for c in CHANNEL_ORDER
    ]
    criterion_owners = {
        f"C1_{c}_attribution": [ARM_ALL_ON, ARM_ALL_OFF, f"ARM_{c}_OFF"] for c in CHANNEL_ORDER
    }
    criteria_nd = {
        name: all(a in green for a in owners)
        for name, owners in criterion_owners.items()
    }

    expected_channels = {"MODEPRIOR", "PCC"}
    contributing_set = {c for c in CHANNEL_ORDER if analysis["per_channel"][c]["c1_pass"]}
    expectation_match = (
        "matches_readiness_probe_expectation"
        if contributing_set and contributing_set.issubset(expected_channels)
        else "no_channel_attributed" if not contributing_set
        else "diverges_from_readiness_probe_expectation"
    )

    manifest: Dict[str, Any] = {
        "run_id": None,  # filled by __main__
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
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
            "n_distinct_argmax_modes_across_design": distinct_modes,
            "argmax_mode_by_arm": {
                arm: sorted({r["argmax_mode"] for r in rs}) for arm, rs in by_arm.items()
            },
            "scorable_channels": scorable_channels,
        },
        "scorable": scorable,
        "red_core_arms": red_core,
        "red_leave_one_out_arms": red_leave_one_out,
        "arm_results": rows,
        "reuse_log": reuse_log,
        "interpretation": {
            "label": label,
            "preconditions": gate["adjudication_preconditions"],
            "preconditions_scope_note": gate["per_arm_gate"]["preconditions_scope_note"],
            "criteria_non_degenerate": criteria_nd,
        },
        "pre_registered_thresholds": {
            "c1_sd_multiplier": C1_SD_MULTIPLIER,
            "c1_abs_floor": C1_ABS_FLOOR,
            "p1_distinct_modes_floor": P1_DISTINCT_MODES_FLOOR,
            "p2_channel_ablation_delta_floor": P2_CHANNEL_ABLATION_DELTA_FLOOR,
            "p3_salience_tick_floor": tick_floor,
        },
        "custom_information": {
            "channel_ladder": {
                "ARM_ALL_ON": BASE.channel_settings(L2),
                "ARM_ALL_OFF": BASE.channel_settings(L0),
            },
            "content_set": BASE.CONTENT_SETS[CONTENT],
            "arena": BASE.ARENA,
            "dv_symmetry_declaration": (
                "occupancy DV is a time-fraction set-aggregate over discrete current_mode; "
                "its symmetry group is inter-mode relabellings preserving TV distance. "
                "ARM_ALL_ON is 802's own L2_A construction (already empirically confirmed "
                "non-invariant: TV=1.0 from ARM_ALL_OFF/L0_A). Every leave-one-out arm "
                "retains channel 3 (mode prior, a non-broadcast external_task-only logit "
                "shift) EXCEPT ARM_MODEPRIOR_OFF, which instead retains channel 1 (dACC "
                "reshape) and channel 4's MECH-259 threshold leg (switch-admission change) "
                "-- so no arm is provably invariant under the occupancy DV's symmetry group. "
                "ARM_PHASIC_OFF is the specific risk mandatory_design_check names: channel 2 "
                "is a pure E3 softmax temperature with no direct SalienceCoordinator entry "
                "point, but E3's live selection path samples via torch.multinomial (not "
                "deterministic argmax), so a temperature change alters the realised action "
                "sequence at a fixed seed and therefore the downstream dACC/AIC context -- a "
                "real but indirect pathway, not a provable identity. ARM_PCC_OFF ablates the "
                "WHOLE pcc_stability_baseline scalar (both the argmax-invariant MECH-048 mu "
                "leg AND the non-invariant MECH-259 threshold leg simultaneously), so it is "
                "NOT the 'channel-4-mu-only' identity risk mandatory_design_check flags -- "
                "that risk applies only to a leg-split design, which is out of scope here "
                "(see leg_split_untested below). No arm scoped out on symmetry grounds."
            ),
            "leg_split_untested": (
                "pcc_stability_baseline drives BOTH the MECH-048 mu leg (mode-prior softmax "
                "temperature, argmax-invariant in isolation) and the MECH-259 threshold leg "
                "(switch admission, non-invariant) via one shared scalar in this substrate. "
                "This design ablates the channel as a whole; it does NOT test whether the "
                "mu leg alone has any occupancy authority (it should not, by symmetry) or "
                "isolate the threshold leg's authority from the mu leg's. A leg-split "
                "design would need independent config knobs for the two legs, which the "
                "current substrate does not expose."
            ),
            "expectation_match": expectation_match,
            "readiness_probe_expectation": (
                "the plan doc's 'channels' field states a 600-tick readiness-probe "
                "EXPECTATION (not an established finding) that channels 3 (mode prior) "
                "and 4 (pcc_stability, via its threshold leg) carry most of the occupancy "
                "authority; channels 1 (5-HT) and 2 (phasic) are expected to act mainly on "
                "precision. expectation_match records whether the measured "
                "contributing-channel set is a subset of {MODEPRIOR, PCC} "
                "(matches_readiness_probe_expectation), empty (no_channel_attributed), or "
                "includes 5HT/PHASIC (diverges_from_readiness_probe_expectation -- still "
                "evidence_direction supports, since ARC-005's routing is even MORE "
                "distributed than predicted, not contradicted)."
            ),
            "precision_out_of_scope": (
                "log10_precision_mean/sd are RECORDED per arm (generous-recording "
                "convention) but are NOT scored: 802 measured literally zero response of "
                "this readout to any of the four channels (bit-identical across L0/L1/L2 "
                "for every content x seed pair), so a leave-one-out contrast on precision "
                "would measure a difference of two constant terms -- an arithmetic identity, "
                "not a measurement, for all four channels in this harness. Any reference to "
                "precision in this manifest is non_contributory/substrate_ceiling context, "
                "per the plan doc's re-scope note, never a scored finding."
            ),
            "reuse": {
                "lineage": LINEAGE,
                "cite_run_id": REUSE_CITE_RUN_ID,
                "attempted_arms": [ARM_ALL_ON, ARM_ALL_OFF],
                "note": (
                    "ARM_ALL_ON and ARM_ALL_OFF are byte-identical in construction to "
                    "802's L2_A / L0_A cells (asserted at startup); reuse is attempted for "
                    "both via try_reuse_cell citing 802's run_id. 19 commits touched "
                    "ree_core/** between 802 landing (2026-07-22) and this run's authoring "
                    "(2026-07-31), which is very likely to bust the default whole-tree "
                    "substrate_hash even though none confirmed to touch the four channel "
                    "modules or the selection path this design reads -- so a REFUSAL "
                    "(fingerprint mismatch -> run fresh) is the expected, safe outcome, not "
                    "a bug. See reuse_log on the manifest for the actual per-cell outcome."
                ),
            },
            "gov_reuse_1": (
                "decisive readout is the per-channel occupancy-authority DROP, which "
                "appears in ZERO recorded manifests -- 802 explicitly declares per-channel "
                "attribution UNTESTED (channel_attribution_limit). Not recoverable except "
                "via the two whole-arm reuses above; residual 4 arms x 5 seeds run fresh."
            ),
            "mint": (
                "lineage exq802_arc005_control_plane (same lineage as 802, new cells); "
                "every FRESH cell fingerprinted with include_driver_script_in_hash=False "
                "-> cross-driver reusable by a future sibling"
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
        "content": CONTENT,
        "channel_ladder": {
            "ARM_ALL_ON": BASE.channel_settings(L2),
            "ARM_ALL_OFF": BASE.channel_settings(L0),
        },
        "arms": ALL_ARM_IDS,
        "schedule": {
            "warmup_ticks": DRY_WARMUP_TICKS if args.dry_run else WARMUP_TICKS,
            "measure_ticks": DRY_MEASURE_TICKS if args.dry_run else MEASURE_TICKS,
        },
        "pre_registered_thresholds": manifest["pre_registered_thresholds"],
    }
    seeds_used = DRY_SEEDS if args.dry_run else SEEDS

    # AFTER arm_results is assembled, so substrate_hash HOISTS from the per-cell
    # fingerprints (fresh cells) / carries the reused cells' own hashes consistently.
    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=seeds_used,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )

    out_path = write_flat_manifest(
        manifest, OUT_DIR, dry_run=args.dry_run, stamp=False
    )

    print(json.dumps({
        "run_id": manifest["run_id"],
        "outcome": manifest["outcome"],
        "evidence_direction": manifest["evidence_direction"],
        "non_degenerate": manifest["non_degenerate"],
        "attribution_achieved": manifest["analysis"]["attribution_achieved"],
        "full_effect_mean": manifest["analysis"]["full_effect_mean"],
        "per_channel_authority_mean": {
            c: manifest["analysis"]["per_channel"][c]["authority_mean"]
            for c in CHANNEL_ORDER
        },
        "label": manifest["interpretation"]["label"],
        "manifest": str(out_path),
    }, indent=2), flush=True)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
