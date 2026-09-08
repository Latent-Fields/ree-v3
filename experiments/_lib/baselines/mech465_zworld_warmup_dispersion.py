"""Canonical baseline module for the MECH-465 z_world warmup-budget dispersion lineage.

Factored out of V3-EXQ-1015 (the first experiment of the lineage) so that its COLD
(no-warmup) arm -- the exact configuration that the 2026-08-27 boundary-regime probe and the
2026-09-04 warmup spike measured -- is reusable BY CONSTRUCTION by any later, different-driver
sibling (a WARM1600 extension, a residual-DV run on a headroom regime). Mint-as-you-go:
`arm_reuse_fingerprint_plan.md` sec 7b / 9.

The substrate config is a verbatim port of the 2026-08-27 harness
(`REE_assembly/evidence/planning/mech465_boundary_regime_probe_20260827.py`) as re-used
unchanged by the 2026-09-04 spike
(`REE_assembly/evidence/planning/mech465_zworld_warmup_dispersion_probe_20260904.py`): the
V3-EXQ-785a exogenous-urgency configuration (alpha_world 0.9, modulatory authority 0.5,
support-preserving CEM, structured curiosity). Keeping it identical is what makes the COLD arm
of any consumer reproduce the spike's COLD readings (the harness-fidelity check every consumer
records).

One thing is deliberately NOT ported: the spike passed `commitment_threshold` through
`REEConfig.from_dims`, which does NOT land it (confirmed 3/3 probes 2026-08-27; instance of
`reference-reeconfig-from-dims-silent-kwargs`). Consumers set
`agent.e3.config.commitment_threshold` POST-CONSTRUCTION; `make_agent_and_env` returns the
from_dims-landed default so the driver can record the provenance.

ASCII-only stdout throughout (Windows-runner safety).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


LINEAGE = "mech465_zworld_warmup_dispersion"
CANONICAL_BASELINE_ID = "mech465_zworld_warmup_dispersion_cold_v1"

# --- environment (verbatim: 2026-08-27 harness / 2026-09-04 spike) -----------------------
ENV_KWARGS: Dict[str, Any] = dict(use_proxy_fields=True, hazard_harm=0.5)

# --- exogenous urgency instrument (verbatim: V3-EXQ-785a grid) ----------------------------
URG = [0.04, 0.10, 0.16, 0.22, 0.28, 0.34]
BASE_URGENCY_WEIGHT = 0.12

# --- measurement schedule ----------------------------------------------------------------
# Calibration pass: threshold at the never-commit sentinel so every tick is a fresh select
# and the running variance is sampled free of the commitment latch. The median over its
# SECOND HALF becomes the cell's own threshold (per-cell recalibration, GFLAG-0136 decision).
#
# WHY 360 / 180 AND NOT THE SPIKE'S 90. rv is a symmetric EMA (alpha 0.05) initialised at
# precision_init 0.5, so the init transient decays as 0.5 * 0.95^t: 4.9e-3 at t=90 (the
# SAME ORDER as the converged rv, 3e-3..1e-2), 2.3e-4 at t=150, 5e-5 at t=180. A window
# opening at t=90 is still inside the transient for its first ~80 ticks, which inflates a
# dispersion statistic (and did so, identically across arms, in the 08-27 probe and the
# 09-04 spike -- recorded as a finding by V3-EXQ-1015, not a defect in either). The
# calibration median must be transient-free or the threshold lands an order of magnitude
# above the converged rv (measured in the 1015 smoke: 0.070 vs 0.007).
CAL_TICKS = 360
CAL_EXCLUDE_TICKS = 180
# WHAT THE SENTINEL DOES AND DOES NOT DO (fable red-team of V3-EXQ-1015, F2). rv is fed by the
# harness EVERY tick regardless of E3's cadence, so the trace is latch-free. The E3 SELECT is
# not every tick: the heartbeat clock fires E3 every e3_steps_per_tick = 10 env steps unless a
# phase reset forces one (ree_core/heartbeat/clock.py:146-157), and the resets come from
# commitment entry and the urgency interrupt (ree_core/agent.py), neither of which fires at the
# sentinel. So the calibration MEDIAN is taken over the every-tick rv TRACE (180 points), never
# over the ~18 fresh-select diagnostic rows.
CAL_TICKS = 360
CAL_EXCLUDE_TICKS = 180
# Threshold placement (red-team F3): thr := cal_median / (1 + U_MID), so that the urgency
# grid's effective thresholds thr * (1 + u), u in [0.04, 0.34], BRACKET the free-running
# median (0.874x .. 1.126x). A raw-median threshold puts every level ABOVE the median and the
# gate commits on ~every select (measured 0.93-1.00 at all six levels in the 1015 smoke).
U_MID = 0.19   # mean of URG
# Scored pass, at the recalibrated threshold. rv is converged by then (no re-init between
# passes); the 90-tick exclusion skips the latch-onset transient after the threshold switch.
# 1200 ticks (2x the spike's 600): rv is an alpha-0.05 EMA, so consecutive samples are
# autocorrelated (~20-tick memory) and the per-cell DV carries a 15-30% relative SE at 510
# scored ticks (block bootstrap on the 1015 smoke); doubling the window is the cheapest
# variance lever available to the trend test.
SCORED_TICKS = 1200
WARMUP_EXCLUDE_TICKS = 90
# Readout constants the cell records (declared in the slice: they change what a cell
# reports, so a consumer with different values must MISS the cache).
P1_BAND = (0.05, 0.95)          # MECH-465 conjunct 1: commit rate per urgency level
GATE_MARGIN_BAND = (0.5, 2.0)   # MECH-465 conjunct 1: median rv / effective_threshold
P0B_BUFFER_MAXLEN = 512         # P0b transition buffer (PHASED arm only)
BOOT_BLOCK_TICKS = 40           # moving-block bootstrap block (2x the EMA time constant)
BOOT_REPLICATES = 400
CI_LEVEL = 0.95                 # two-sided bootstrap CI recorded per cell
# SD-070 P0a rollout length per episode (verbatim spike: p0_steps=40).
P0_STEPS_PER_EPISODE = 40
# Never-commit sentinel for the calibration pass: committed = rv < thr, rv ~ 1e-3..1e-2.
CAL_SENTINEL_THRESHOLD = 1e-9

# --- substrate config (verbatim spike `build()` kwargs, minus commitment_threshold) -------
AGENT_KWARGS_STATIC: Dict[str, Any] = dict(
    alpha_world=0.9,
    use_harm_stream=True, use_affective_harm_stream=True, urgency_weight=BASE_URGENCY_WEIGHT,
    use_support_preserving_cem=True, support_preserving_min_first_action_classes=2,
    support_preserving_stratified_elites=True, support_preserving_ao_std_floor=0.2,
    use_per_stream_vs=True, use_per_region_vs=True, use_event_segmenter=True,
    use_invalidation_trigger=True, use_anchor_sets=True,
    e2_action_contrastive_enabled=True, e2_action_contrastive_weight=0.1,
    e2_rollout_output_norm_clamp_enabled=True, e2_rollout_output_norm_clamp_ratio=4.0,
    use_structured_curiosity=True, use_curiosity_novelty=True,
    curiosity_bias_scale=0.1, curiosity_novelty_weight=0.05,
    use_modulatory_selection_authority=True, modulatory_authority_gain=0.5,
    modulatory_authority_min_range_floor=1e-6,
    use_e3_score_diversity=False, use_e3_diversity_entropy_bonus=False,
)


def make_env(seed: int) -> CausalGridWorldV2:
    """A fresh environment instance. Every rollout stream gets its own (RNG isolation)."""
    return CausalGridWorldV2(seed=int(seed), **ENV_KWARGS)


def make_agent_and_env(seed: int) -> Tuple[REEAgent, CausalGridWorldV2, Dict[str, Any], float]:
    """Build the measurement env + agent exactly as the spike did.

    Returns (agent, env, obs_dict_after_reset, from_dims_default_commitment_threshold).
    The agent is in eval mode with E3 score decomposition enabled (the spike's setting); the
    commitment threshold is UNTOUCHED here -- the driver sets it post-construction.
    """
    env = make_env(seed)
    _o, od = env.reset()
    kw = dict(
        body_obs_dim=od["body_state"].shape[-1],
        world_obs_dim=od["world_state"].shape[-1],
        action_dim=env.action_dim,
        **AGENT_KWARGS_STATIC,
    )
    cfg = REEConfig.from_dims(**kw)
    cfg.e3.use_finer_channel_gating = True
    agent = REEAgent(cfg)
    agent.eval()
    agent.e3.e3_score_decomp_enabled = True
    landed = float(agent.e3.config.commitment_threshold)
    return agent, env, od, landed


def arm_config_slice(p0a_episodes: int, p0b_episodes: int, p0b_steps: int = 0) -> Dict[str, Any]:
    """Everything a cell's computation reads, and nothing it does not.

    Declares env kwargs + the substrate-operating config + the schedule + the arm's own
    warmup budgets. NOT declared: acceptance thresholds, labels, ON-arm-only analysis
    constants (none of those change what the cell computes).
    """
    return {
        "lineage": LINEAGE,
        "env_kwargs": dict(ENV_KWARGS),
        "agent_kwargs_static": dict(AGENT_KWARGS_STATIC),
        "e3_use_finer_channel_gating": True,
        "e3_score_decomp_enabled": True,
        "urgency_grid": list(URG),
        "schedule": {
            "cal_ticks": CAL_TICKS,
            "cal_exclude_ticks": CAL_EXCLUDE_TICKS,
            "scored_ticks": SCORED_TICKS,
            "warmup_exclude_ticks": WARMUP_EXCLUDE_TICKS,
            "cal_sentinel_threshold": CAL_SENTINEL_THRESHOLD,
            "calibration": "free_running_trace_second_half_median",
            "threshold_placement": "cal_median_over_1_plus_u_mid",
            "u_mid": U_MID,
            "p0_steps_per_episode": P0_STEPS_PER_EPISODE,
        },
        "readout_constants": {
            "P1_BAND": list(P1_BAND),
            "GATE_MARGIN_BAND": list(GATE_MARGIN_BAND),
            "P0B_BUFFER_MAXLEN": P0B_BUFFER_MAXLEN,
            "BOOT_BLOCK_TICKS": BOOT_BLOCK_TICKS,
            "BOOT_REPLICATES": BOOT_REPLICATES,
            "CI_LEVEL": CI_LEVEL,
            "dv_source": "every_tick_rv_trace",
        },
        "p0a_episodes": int(p0a_episodes),
        "p0b_episodes": int(p0b_episodes),
        "p0b_steps_per_episode": int(p0b_steps),
        "rv_update": "manual_e2_world_forward_residual_every_tick",
    }


def off_path_config_slice() -> Dict[str, Any]:
    """The COLD arm's slice: no P0a, no P0b. Consumers cite this via reuse_baseline_from."""
    return arm_config_slice(0, 0, 0)
