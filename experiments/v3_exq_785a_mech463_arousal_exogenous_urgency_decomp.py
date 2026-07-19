"""V3-EXQ-785a: MECH-463 -- arousal as a variance amplifier of the incumbent selection
channel, re-tested under EXOGENOUS urgency manipulation with hazard proximity measured.

Supersedes the SCORING of V3-EXQ-785 (same scientific question, corrected design).
Re-queued per failure_autopsy_V3-EXQ-785_2026-07-19 (REE_assembly master ccb4a35e67),
targets[0].requeue_spec. claim_ids = [MECH-463] ONLY; MECH-439 deliberately NOT tagged.

=== WHAT 785 ESTABLISHED, AND THE ONE THING IT COULD NOT DECIDE ===
785's harm_incumbent arm passed its gate 6/6 and produced a strong result: incumbent share
fell 0.970 -> 0.831 across urgency deciles (rho -0.8303 against a pre-registered +0.6).
Decomposing share x total separated MECH-463's two premises with OPPOSITE signs:

    decile 0 -> decile 9
      TOTAL cross-candidate variance    x14.1   <- AMPLIFICATION premise: CONFIRMED
        incumbent absolute variance     x12.1
        non-incumbent absolute variance x79.2   <- CONCENTRATION premise: CONTRADICTED

Arousal amplifies selection variance, but amplifies the SUBORDINATE channels ~6.5x faster
than the incumbent, so the dominant channel's SHARE falls.

THE OPEN QUESTION: urgency_applied was ENDOGENOUS in 785 (emergent from z_harm_a), so
high-urgency ticks were also NEAR-HAZARD ticks (env hazard_harm 0.5). The leading rival is
that the whole profile reflects near-hazard candidate GEOMETRY, not arousal. This run
exists solely to break that confound.

=== THE LOAD-BEARING CHANGE: EXOGENOUS URGENCY (and why it earns a new letter) ===
e3_selector.py:2704-2711 computes
    urgency_applied = min(||z_harm_a|| * config.urgency_weight, urgency_max)
Both operands are readable in the driver and `config.urgency_weight` is read LIVE at each
select() call, so setting
    e3.config.urgency_weight = target / ||z_harm_a||
lands urgency_applied on `target` EXACTLY, with no substrate change. Verified 2026-07-19,
600 ticks: max |realized - assigned| = 2.8e-17 (float roundoff), 6 distinct levels.

Assignment is drawn i.i.d. uniform from a pre-registered grid, INDEPENDENT of state, so
the confound is broken BY CONSTRUCTION rather than adjusted away. Measured on the probe:
    corr(assigned_urgency, hazard_prox_max) = -0.038   (endogenous 785: confounded by design)
The grid spans 0.04..0.34, BRACKETING and widening 785's endogenous span (0.070..0.125),
so the contrast is better powered as well as unconfounded.

`urgency_weight` is the INSTRUMENT here, not a fixed config. The 785 value 0.12 is retained
as the declared baseline and each tick also records `natural_urgency` = the value 0.12 WOULD
have produced, so the exogenous levels stay interpretable against 785's endogenous range.
(At urgency_weight 0.5 urgency saturates against urgency_max and collapses the top bins --
the reason 785 used 0.12; under exogenous control the cap is respected by the grid instead,
whose top level 0.34 sits below urgency_max 0.5.)

=== SECOND REGIME: DROPPED, WITH EVIDENCE. CHANNEL-AGNOSTICISM IS UNTESTED. ===
The autopsy required replacing OR dropping 785's entropy_bias_scale=1.0 regime, whose
pre-registered expected_incumbent_share was 1.043 -- above unity, so (shares summing to 1.0
by construction) every other component is forced below the |0.01| floor and the regime could
NEVER pass its own >=2-component P7 gate. Replacement was attempted: entropy_bias_scale was
scanned over {0.15, 0.30, 0.50, 1.00} (seed 0, 150 ticks each, 2026-07-19):

    ebs    incumbent      share    runner-up              n_nontrivial
    0.15   CH:mech341    1.1478    f            0.0165    3
    0.30   CH:mech341    0.9972    harm_weighted 0.0025   1
    0.50   CH:mech341    0.9927    harm_weighted 0.0078   1
    1.00   CH:mech341    0.9911    harm_weighted 0.0125   2

NO viable second regime exists on this axis: CH:mech341 absorbs ~99-115% of cross-candidate
variance at every setting, so the share statistic is arithmetically forced wherever the
entropy bonus is the incumbent. The regime is therefore DROPPED rather than replaced, and
this run states plainly: **MECH-463's CHANNEL-AGNOSTICISM half is UNTESTED here.** This run
tests the mechanism at ONE incumbent identity (harm_weighted). A PASS licenses no claim
about channel-agnosticism; that requires a substrate offering a genuine second
multi-component cross-candidate regime, which this one does not.

=== A MEASUREMENT DEFECT IN 785: ~9x PSEUDO-REPLICATION (fixed here) ===
785 read `agent.e3.last_score_diagnostics` WITHOUT clearing it. E3 select() only runs when
the commitment latch is open (agent.py:4968, beta_gate.is_elevated holds the committed
trajectory and skips fresh selection), and on latched ticks the PREVIOUS tick's diagnostics,
decomposition and channel terms are all still resident -- so 785 appended them again as new
rows. Measured 2026-07-19 over 600 ticks: 67 genuine select() calls, but the 785 read
pattern yields 600 rows -- a ~9.0x pseudo-replication factor.

Consequence for 785's reported precision (NOT for its direction): its "3959 committed ticks,
~396/decile, SE ~0.003, ~40 SE" reflects ~440 independent selections, so the true SE is ~3x
larger (SE ~ 1/sqrt(n)) and the span is ~15 SE, not ~40. The 785 finding SURVIVES this
correction comfortably -- it is a precision correction, not a reversal.

FIX: `e3.last_score_diagnostics` is set to None immediately before every select_action(),
and a row is recorded ONLY if it is repopulated. Every row in this run is one genuine,
independent E3 selection.

=== THE CANDIDATE MECHANISM, AND WHY ITS SIGN IS RESTATED HERE ===
The autopsy flagged (as candidate, not established) that
`effective_threshold * (1.0 - urgency_applied)` is an ADMISSION threshold whose lowering is
"LESS selective ... which predicts dilution". READING THE GATE, THAT SIGN IS BACKWARDS:

    e3_selector.py:2696-2711, 2718-2725
        effective_threshold = commit_threshold * (1.0 - urgency_applied)
        committed = <variance> < effective_threshold          # commit when variance is LOW

The gate is an UPPER BOUND on a variance. Raising urgency LOWERS the bound, making the gate
STRICTER -- it admits FEWER ticks, and only the LOWEST-variance ones. Confirmed empirically
(600 ticks, exogenous): commit rate falls 1.000 (u=0.04) -> 0.889 (u=0.34), monotone in the
tail. So arousal is a MORE-selective admission filter, not a diluting one.

Two further precisions that matter for interpreting any share change:
  (a) The gated quantity in this config is the E3 z_world RUNNING VARIANCE (world-model
      stability), NOT cross-candidate score separation -- `use_harm_variance_commit` is off,
      so the harm-score-variance reframe branch is not taken. Arousal therefore selects on
      WORLD-STABILITY, and any induced share change is a selection-on-stability effect, not
      a separation effect. This is a sharper mechanism than "admits lower-separation ticks".
  (b) Commit rate is near-saturated (~0.97 overall). A near-ceiling rate has little room to
      rise, which BOUNDS how much differential admission can explain. If the rate is flat
      across levels while the share moves, differential admission is RULED OUT as the
      mechanism -- an informative null either way. Hence C3 below is scored and declared,
      but is explicitly NOT load-bearing.

=== PRE-REGISTERED CRITERIA -- AMPLIFICATION AND CONCENTRATION SCORED SEPARATELY ===
785's compound criterion returned FAIL for a claim whose amplification half was confirmed at
14.1x. They are separated here and BOTH are load-bearing (MECH-463 asserts both):

  C1 AMPLIFICATION -- total cross-candidate score variance RISES with exogenous urgency
                      (Spearman rho over level means >= +0.60, and a positive log-ratio gap).
  C2 CONCENTRATION -- the incumbent's SHARE rises with exogenous urgency (rho >= +0.60).
  C3 ADMISSION     -- commit RATE per urgency level (mechanism probe; NOT load-bearing;
                      its null is declared below).

INTERPRETATION GRID (six cells; the monotone-DECREASE cell is the one 785 lacked, which is
why a strong contrary result had no valid self-route and was buried):

  C1 rises + C2 rises  -> SUPPORTS            arousal amplifies AND concentrates.
  C1 rises + C2 FALLS  -> AMPLIFIES_BUT_DILUTES  <- NEW CELL. The 785 signature, now under
                          exogenous control: amplification confirmed, concentration
                          contradicted. Weakens MECH-463's concentration premise ON THE
                          MERITS and opens arousal as a candidate DIVERSITY lever.
  C1 rises + C2 flat   -> AMPLIFIES_ONLY      amplification only; concentration untested-null.
  C1 flat  + C2 FALLS  -> REDISTRIBUTES_ONLY  share moves with no variance gain.
  C1 flat  + C2 flat   -> REFUTES             arousal causally inert w.r.t. the selection-
                          variance distribution (gate GREEN required).
  gate RED             -> substrate_not_ready_requeue. NEVER a substrate verdict.

"FALLS" is a SEPARATE, SIGNED cell: rho <= -0.60 with a negative gap clearing the same
effect-size floor. It is distinct from "flat" (|rho| < 0.60), which is the REFUTES cell.

=== THE RIVAL IS MEASURED, NOT ARGUED ===
`hazard_field_view` (25-dim 5x5 proximity gradient) is recorded per tick and the share
profile is reported CONDITIONED on its tertile. This is deliberately the LEARNER'S OWN
observable, not a privileged global oracle distance -- the V3-EXQ-732a confound (a global
oracle ceiling scored against a 5x5-local-view learner) is exactly the error to avoid here.
Randomisation already makes proximity orthogonal to assigned urgency IN EXPECTATION; the
conditional profile is the check that it held IN SAMPLE, and the realised correlation is
recorded as a precondition.

=== DURABLE RECORDING: THE SINK GOES INSIDE THE MANIFEST ===
785 declared `custom_information.per_tick_sink` = "<run_id>_per_tick.jsonl" and the file is
ABSENT from disk repo-wide -- which blocked the commit-rate-per-level reanalysis that would
have tested the mechanism above at zero extra compute. ROOT CAUSE (diagnosed 2026-07-19, not
a forgotten write): 785 ran on ree-cloud-2 (manifest `machine`: ree-cloud-2). Under Phase 3
the worker POSTs /result and the coordinator spools ONLY `manifest_bytes`; runner-side result
pushes are disabled (PHASE3_DISABLE_RUNNER_RESULT_PUSH=1). A sidecar written next to the
manifest on the worker's local disk is therefore NEVER transported to REE_assembly. The
sidecar pattern CANNOT work for any cloud-run experiment.

FIX: the per-tick record is embedded IN the manifest under
`custom_information.per_tick_rows`, which is inside `manifest_bytes` and so travels. Per the
Experimental Recording Standard's `custom_information` catch-all (section 3c / principle 3,
"over-recording is lossless"). Rows are compact (fixed scalar keys, rounded) and the
per-candidate component vectors -- the bulky part -- are retained only for a bounded
pre-registered SAMPLE of ticks, so the manifest stays a sane size while every scalar needed
for the decile/level, commit-rate and proximity-conditioned reanalyses is present for EVERY
row. A sidecar .jsonl is ALSO written when running locally, as a convenience only; nothing
in the analysis depends on it.

=== NON-VACUITY GATE (P1/P2 regime-conditioned; P7 NOW regime-conditioned too) ===
785's P7 (>=2 components with non-trivial share) was applied WHOLE-RUN, so the structurally
single-component entropy arm's RED vacated the harm arm's valid, well-powered result. The
autopsy's sharpest finding. Every precondition here is evaluated PER REGIME and the run-level
gate is the conjunction over regimes that are actually scored -- so no regime can vacate
another's result. (With the entropy regime dropped there is one regime, but the conditioning
is implemented structurally so it cannot regress if a second regime is ever added.)

  P1 modulatory_authority_active_frac   -- ONLY where the incumbent is a modulatory CHANNEL
  P2 incumbent cross-candidate RANGE    -- the SAME statistic the decomposition routes on
                                           (never a magnitude/mean-abs proxy: the 643 GAP)
  P3 assigned urgency non-constant      -- >= 5 of 6 levels populated
  P4 total cross-candidate variance     -- a share of ~0 variance is undefined
  P5 fresh committed-tick count         -- FRESH selections only (the 785 ~9x defect)
  P6 pre-registered incumbent IS the incumbent
  P7 >= 2 components with |share| > 0.01  -- REGIME-CONDITIONED (the 785 defect)
  P8 exogenous assignment fidelity      -- max |realized - assigned| below tolerance;
                                           if the injection did not take, the manipulation
                                           did not happen and nothing here is interpretable
  P9 randomisation held in sample       -- |corr(assigned_urgency, hazard_prox)| below a
                                           CEILING (direction "upper"), the one precondition
                                           here that is not a floor
Any RED -> outcome FAIL, label substrate_not_ready_requeue, non_degenerate=false.

=== TWO DRIVER CONSTRAINTS THAT SILENTLY VACATE THIS PROBE (preserved from 785) ===
(1) act_with_split_obs() calls sense(obs_body, obs_world) with NO obs_harm_a, so z_harm_a is
    None on that path and urgency_applied is pinned at 0 -- which ALSO makes the exogenous
    injection a no-op (the branch at :2705 requires z_harm_a is not None). The harm stream
    MUST be fed explicitly and the wrapper replicated:
    clock.advance() -> _e1_tick -> generate_trajectories -> select_action.
(2) V3-EXQ-396a: update_running_variance() has NO CALLER ANYWHERE IN ree_core, so
    _running_variance stays pinned at precision_init (0.5) while effective_threshold tops out
    near 0.39 -- the commit gate never fires and every tick is uncommitted. The driver must
    drive the EMA itself. Measured commit rate 0.000 -> 0.970.
Do not "simplify" either away.

SCOPE: not gated on MECH-457 or INV-088 (a decomposition probe on score geometry, not a
behavioural conversion test). Does NOT re-open the 689/485/445/625 selection-face lineages.
MECH-439 NOT tagged (11 substrate_ceiling autopsies); nothing here bears on it.
GOV-REUSE-1: the decisive readout is the incumbent share under EXOGENOUS urgency. No
recorded manifest carries it -- exogenous urgency assignment has never been run (785 is the
only run with urgency_applied at all, and it is endogenous by construction) -- so it is not
recoverable by reanalysis and must run.

Run:
  /opt/local/bin/python3 experiments/v3_exq_785a_mech463_arousal_exogenous_urgency_decomp.py --dry-run
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402

# ------------------------------------------------------------------ #
# Identity                                                           #
# ------------------------------------------------------------------ #
EXPERIMENT_TYPE = "v3_exq_785a_mech463_arousal_exogenous_urgency_decomp"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["MECH-463"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
SUPERSEDES = "v3_exq_785_mech463_arousal_variance_amplifier_decomp"

OUT_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# ------------------------------------------------------------------ #
# Pre-registered constants (NOT derived from the run's own statistics) #
# ------------------------------------------------------------------ #
SEEDS = [0, 1, 2, 3, 4]
TICKS_PER_SEED = 3000
DRY_RUN_TICKS = 120

# EXOGENOUS urgency levels, assigned i.i.d. uniform per tick, independent of state.
# Brackets and widens 785's endogenous span (0.070..0.125). Top level 0.34 < urgency_max
# 0.5, so no level saturates against the cap.
URGENCY_LEVELS = [0.04, 0.10, 0.16, 0.22, 0.28, 0.34]

PRIMARY_COMPONENTS = [
    "f", "harm_weighted", "residue_weighted",
    "benefit_weighted", "novelty_weighted", "goal_weighted",
]

# Single regime. The entropy regime is DROPPED with recorded evidence (see docstring):
# CH:mech341 absorbs ~99-115% of cross-candidate variance at every entropy_bias_scale
# scanned, so its share statistic is arithmetically forced. Channel-agnosticism is UNTESTED.
REGIMES = [
    {"id": "harm_incumbent", "entropy_bias_scale": 0.0,
     "expected_incumbent": "harm_weighted", "expected_incumbent_share": 0.9368},
]
DROPPED_REGIME_EVIDENCE = {
    "dropped": "entropy_incumbent (entropy_bias_scale=1.0)",
    "reason": (
        "pre-registered expected_incumbent_share 1.043 > 1.0; shares sum to 1.0 by "
        "construction, so every other component is forced below the |0.01| floor and the "
        "regime could never satisfy its own >=2-component P7 gate"
    ),
    "replacement_scan_2026_07_19_seed0_150ticks": {
        "0.15": {"incumbent": "CH:mech341", "share": 1.1478, "n_nontrivial": 3},
        "0.30": {"incumbent": "CH:mech341", "share": 0.9972, "n_nontrivial": 1},
        "0.50": {"incumbent": "CH:mech341", "share": 0.9927, "n_nontrivial": 1},
        "1.00": {"incumbent": "CH:mech341", "share": 0.9911, "n_nontrivial": 2},
    },
    "conclusion": (
        "no viable second regime on the entropy axis -- CH:mech341 absorbs ~99-115% of "
        "cross-candidate variance at every setting. Regime DROPPED, not replaced. "
        "MECH-463's CHANNEL-AGNOSTICISM half is UNTESTED by this run."
    ),
}

MIN_TICKS_PER_LEVEL = 30
MIN_LEVELS_POPULATED = 5

# Non-vacuity gate floors (FLOORS unless marked).
AUTHORITY_ACTIVE_FRAC_FLOOR = 0.05
CHANNEL_RANGE_FLOOR = 1e-4
SCORE_VARIANCE_FLOOR = 1e-12
COMMITTED_TICK_FLOOR = float(MIN_TICKS_PER_LEVEL * MIN_LEVELS_POPULATED)
INCUMBENT_MARGIN_FLOOR = 0.10
N_NONTRIVIAL_FLOOR = 1.5
NONTRIVIAL_SHARE_FLOOR = 0.01
# P8: exogenous fidelity. CEILING -- measured 2.8e-17 on the probe; 1e-6 is a generous bound.
FIDELITY_TOL = 1e-6
# P9: randomisation check. CEILING -- measured |corr| 0.038 on the probe.
RANDOMISATION_CORR_CEILING = 0.25

# PASS criteria.
MONOTONIC_RHO_FLOOR = 0.60
GAP_ABS_FLOOR = 0.02
GAP_SD_MULTIPLIER = 2.0
# C1 amplification is scored on a LOG ratio (variance spans ~14x), so it needs its own floor.
LOG_VAR_GAP_FLOOR = 0.10

URGENCY_WEIGHT_BASELINE = 0.12   # 785's value; the declared baseline for natural_urgency
MODULATORY_AUTHORITY_GAIN = 0.5
ALPHA_WORLD = 0.9

# Bound the bulky per-candidate vectors retained in the manifest (scalars kept for ALL rows).
# Sized against a measured projection: at 3000 ticks x 5 seeds the run yields ~1800 rows, and
# a per_candidate row costs ~2.4 KB against ~0.5 KB for a scalar row. 25/cell holds the whole
# manifest near ~1.2 MB -- within the existing corpus range (largest run manifest to date:
# 0.96 MB, v3_exq_779b) rather than an outlier, while still banking a per-candidate sample
# for every cell. The scalar fields every reanalysis needs are kept for EVERY row.
PER_CANDIDATE_SAMPLE_PER_CELL = 25
HAZARD_TERTILE_MIN_N = 20


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #
def _spearman_rho(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation. Returns 0.0 when undefined (n < 3 or a flat side)."""
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


def _pearson(x: List[float], y: List[float]) -> float:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return 0.0
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def _component_shares(components: Dict[str, np.ndarray]) -> Optional[Dict[str, float]]:
    """Covariance-correct cross-candidate variance share for EVERY component.

    score_k = sum_c C_ck. Attributing to component c its own variance plus its full
    covariance with the rest:
        share_c = (Var(C_c) + Cov(C_c, sum_{d != c} C_d)) / Var(score)
    These sum to EXACTLY 1, since
        sum_c [Var(C_c) + Cov(C_c, rest_c)] = Var(sum_c C_c).
    Retaining the per-candidate [K] tensors unreduced is what makes the covariance terms
    computable at all -- marginal variances alone cannot form them.
    """
    keys = [k for k, v in components.items() if v.size >= 2]
    if not keys:
        return None
    n = components[keys[0]].size
    total = np.zeros(n, dtype=float)
    for k in keys:
        if components[k].size != n:
            return None
        total = total + components[k]
    var_total = float(np.var(total))
    if not np.isfinite(var_total) or var_total <= SCORE_VARIANCE_FLOOR:
        return None
    out: Dict[str, float] = {}
    for k in keys:
        c = components[k]
        rest = total - c
        cov = float(np.mean((c - c.mean()) * (rest - rest.mean())))
        out[k] = (float(np.var(c)) + cov) / var_total
    out["__var_total__"] = var_total
    return out


def _build_agent_and_env(seed: int, regime: Dict[str, Any]):
    # CausalGridWorld carries its OWN np.random.default_rng(seed) -- omitting seed= here
    # makes the run non-reproducible (and any bit-identity check meaningless).
    env = CausalGridWorldV2(use_proxy_fields=True, seed=seed, hazard_harm=0.5)
    _obs, obs_dict = env.reset()

    kw: Dict[str, Any] = dict(
        body_obs_dim=obs_dict["body_state"].shape[-1],
        world_obs_dim=obs_dict["world_state"].shape[-1],
        action_dim=env.action_dim,
        alpha_world=ALPHA_WORLD,
        # SD-011 affective harm stream -> z_harm_a. Without BOTH, z_harm_a is None, the
        # urgency branch at e3_selector.py:2705 is skipped entirely, and the EXOGENOUS
        # injection is a silent no-op (not just a pinned-at-0 endogenous urgency).
        use_harm_stream=True,
        use_affective_harm_stream=True,
        urgency_weight=URGENCY_WEIGHT_BASELINE,
        use_support_preserving_cem=True,
        support_preserving_min_first_action_classes=2,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        # SD-056: action-divergent candidates + the clamp keeping E3 scores bounded
        # (~0.034 raw range) rather than the ~1e32 that killed 643 by float32 cancellation.
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=0.1,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=4.0,
        use_structured_curiosity=True,
        use_curiosity_novelty=True,
        curiosity_bias_scale=0.1,
        curiosity_novelty_weight=0.05,
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_min_range_floor=1e-6,
    )
    es = float(regime["entropy_bias_scale"])
    if es > 0.0:
        kw.update(
            use_e3_score_diversity=True,
            use_e3_diversity_entropy_bonus=True,
            use_e3_diversity_stratified_select=False,
            e3_diversity_entropy_bias_scale=es,
        )
    else:
        kw.update(use_e3_score_diversity=False, use_e3_diversity_entropy_bonus=False)

    cfg = REEConfig.from_dims(**kw)
    # Split last_channel_terms into NAMED channels rather than one compressed
    # "score_bias" slot -- the decomposition needs the parts separable.
    cfg.e3.use_finer_channel_gating = True

    agent = REEAgent(cfg)
    agent.eval()
    agent.e3.e3_score_decomp_enabled = True  # MECH-463 instrumentation gate (ree-v3 435322f)
    return agent, env, obs_dict, kw


def _urgency_signal(agent: REEAgent, latent) -> Optional[torch.Tensor]:
    """The tensor E3 will actually norm for urgency.

    agent.select_action (agent.py:4934-4952) passes z_harm_a to E3.select, EXCEPT that
    SD-019a redirects it to z_harm_un when use_harm_un is active. The injected weight must
    be computed against the SAME tensor E3 norms, or the realised urgency misses its target
    (P8 would catch it, but this reproduces the redirect so it does not arise).
    """
    sig = latent.z_harm_a
    if getattr(agent.config.latent, "use_harm_un", False) and latent.z_harm_un is not None:
        sig = latent.z_harm_un
    return sig


def _collect_cell(seed: int, regime: Dict[str, Any], n_ticks: int, rng: np.random.Generator):
    """One (regime, seed) cell under EXOGENOUS urgency. Returns rows + telemetry."""
    agent, env, obs_dict, _cfg_slice = _build_agent_and_env(seed, regime)

    rows: List[Dict[str, Any]] = []
    authority_hits = 0
    channel_ranges: List[float] = []
    incumbent_ranges: List[float] = []
    fidelity_errs: List[float] = []
    n_fresh_select = 0
    n_latched = 0
    # 396a: the agent loop never calls update_running_variance(), so the driver must.
    z_world_prev: Optional[torch.Tensor] = None
    action_prev: Optional[torch.Tensor] = None

    print(f"Seed {seed} Condition {regime['id']}", flush=True)

    for tick in range(n_ticks):
        # EXOGENOUS assignment: drawn independently of state, BEFORE anything is observed.
        assigned = float(rng.choice(URGENCY_LEVELS))
        with torch.no_grad():
            # Explicit harm feed -- act_with_split_obs() would drop obs_harm_a.
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

            # --- the manipulation ---------------------------------------------------- #
            sig = _urgency_signal(agent, latent)
            sig_norm = float(sig.norm(dim=-1).mean().item()) if sig is not None else 0.0
            natural_urgency = min(sig_norm * URGENCY_WEIGHT_BASELINE,
                                  agent.e3.config.urgency_max)
            # urgency_applied = min(sig_norm * urgency_weight, urgency_max); solving for the
            # weight lands it exactly on `assigned`. urgency_weight is read live at select().
            agent.e3.config.urgency_weight = (
                (assigned / sig_norm) if sig_norm > 1e-9 else 0.0)

            # Freshness marker: E3 select() only runs when the commitment latch is open.
            # Without this clear, a latched tick re-reads the PREVIOUS tick's diagnostics
            # and the row is a duplicate -- the ~9x pseudo-replication defect in 785.
            agent.e3.last_score_diagnostics = None

            ticks_d = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks_d["e1_tick"]
                else torch.zeros(1, agent.config.latent.world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks_d)
            action = agent.select_action(candidates, ticks_d, 1.0)
        agent._step_count += 1

        diag = agent.e3.last_score_diagnostics
        if diag is None or "urgency_applied" not in diag:
            n_latched += 1   # commitment latch held; no fresh selection this tick
        else:
            n_fresh_select += 1
            realized = float(diag["urgency_applied"])
            fidelity_errs.append(abs(realized - assigned))
            if float(diag.get("modulatory_authority_active", 0.0) or 0.0) > 0.0:
                authority_hits += 1

            decomp = agent.e3.last_score_decomp or {}
            chan = agent.e3.last_channel_terms or {}
            per_cand = decomp.get("per_candidate") or []
            comps: Dict[str, np.ndarray] = {}
            for name in PRIMARY_COMPONENTS:
                comps[name] = np.asarray(
                    [float(c.get(name, 0.0) or 0.0) for c in per_cand], dtype=float
                )
            for cname, cvec in chan.items():
                v = cvec.detach().cpu().numpy().astype(float)
                comps["CH:" + cname] = v
                if v.size >= 2:
                    channel_ranges.append(float(v.max() - v.min()))

            inc = regime["expected_incumbent"]
            if inc in comps and comps[inc].size >= 2:
                incumbent_ranges.append(float(comps[inc].max() - comps[inc].min()))

            shares = _component_shares(comps) if len(per_cand) >= 2 else None
            if shares is not None:
                var_total = shares.pop("__var_total__")
                # Hazard proximity from the LEARNER'S OWN observable (5x5 local view), not a
                # privileged global oracle distance -- the 732a confound.
                hz = obs_dict.get("hazard_field_view")
                if hz is not None:
                    hzv = hz.detach().cpu().numpy().astype(float).reshape(-1)
                    hz_max = float(hzv.max())
                    hz_center = float(hzv[hzv.size // 2])
                    hz_mean = float(hzv.mean())
                else:
                    hz_max = hz_center = hz_mean = 0.0
                row: Dict[str, Any] = {
                    "regime": regime["id"],
                    "seed": seed,
                    "tick": tick,
                    "assigned_urgency": assigned,
                    "realized_urgency": realized,
                    "natural_urgency": natural_urgency,
                    "effective_threshold": float(diag.get("effective_threshold", 0.0)),
                    "commit_variance": float(diag.get("commit_variance", 0.0)),
                    "commit_gate_mode": str(diag.get("commit_gate_mode", "")),
                    "committed": bool(diag.get("committed", False)),
                    "temperature": float(
                        diag.get("gap_scaled_commit_temperature_eff", 1.0) or 1.0),
                    "da": float(diag.get("loop_d1_d2_conflict_signal", 0.0) or 0.0),
                    "hazard_prox_max": round(hz_max, 6),
                    "hazard_prox_center": round(hz_center, 6),
                    "hazard_prox_mean": round(hz_mean, 6),
                    "var_total": round(var_total, 14),
                    # 5 dp on shares: the criteria compare level MEANS whose spread is
                    # O(0.01-0.1), so the 6th decimal is noise, and the shares dict is the
                    # single largest contributor to per-row size. Every component is
                    # retained (never pruned to the non-trivial ones) so the sum-to-1.0
                    # audit that validates the decomposition stays possible.
                    "shares": {k: round(v, 5) for k, v in shares.items()},
                }
                # Bulky per-candidate vectors: bounded sample only (scalars kept for ALL
                # rows), so the manifest carries the full analysis set at a sane size.
                if len(rows) < PER_CANDIDATE_SAMPLE_PER_CELL:
                    row["per_candidate"] = {
                        k: [round(x, 6) for x in v.tolist()] for k, v in comps.items()}
                rows.append(row)

        if (tick + 1) % 250 == 0 or tick == n_ticks - 1:
            print(f"  [train] mech463x {regime['id']} seed={seed} ep {tick + 1}/{n_ticks} "
                  f"fresh={n_fresh_select} rows={len(rows)}", flush=True)

        act_idx = int(action.argmax().item()) if isinstance(action, torch.Tensor) else int(action)
        action_prev = torch.zeros(1, env.action_dim)
        action_prev[0, act_idx % env.action_dim] = 1.0
        z_world_prev = z_world_cur
        _obs, _r, done, _info, obs_dict = env.step(act_idx % env.action_dim)
        if done:
            _obs, obs_dict = env.reset()

    telemetry = {
        "regime": regime["id"],
        "seed": seed,
        "n_ticks": n_ticks,
        "n_fresh_select": n_fresh_select,
        "n_latched_ticks": n_latched,
        "fresh_select_yield": (n_fresh_select / n_ticks) if n_ticks else 0.0,
        "n_rows": len(rows),
        "n_committed": sum(1 for r in rows if r["committed"]),
        "authority_active_frac": (authority_hits / n_fresh_select) if n_fresh_select else 0.0,
        "channel_range_mean": float(np.mean(channel_ranges)) if channel_ranges else 0.0,
        "incumbent_range_mean": (
            float(np.mean(incumbent_ranges)) if incumbent_ranges else 0.0),
        "fidelity_err_max": float(np.max(fidelity_errs)) if fidelity_errs else 0.0,
        "fidelity_err_mean": float(np.mean(fidelity_errs)) if fidelity_errs else 0.0,
    }
    return rows, telemetry


def _level_profile(rows: List[Dict[str, Any]], incumbent: str) -> Dict[str, Any]:
    """Group COMMITTED rows by the EXOGENOUSLY ASSIGNED urgency level."""
    committed = [r for r in rows if r["committed"] and incumbent in r["shares"]]
    levels = []
    for u in URGENCY_LEVELS:
        sel = [r for r in committed if r["assigned_urgency"] == u]
        if len(sel) < MIN_TICKS_PER_LEVEL:
            levels.append({"assigned_urgency": u, "n": len(sel), "scored": False})
            continue
        inc_vals = [r["shares"][incumbent] for r in sel]
        vt = [r["var_total"] for r in sel]
        # Absolute incumbent / non-incumbent variance = share x total. Recording BOTH is
        # what let the 785 autopsy separate amplification from concentration at all
        # (learning 3: "always record the DENOMINATOR of a share criterion").
        inc_abs = [s * v for s, v in zip(inc_vals, vt)]
        non_abs = [(1.0 - s) * v for s, v in zip(inc_vals, vt)]
        levels.append({
            "assigned_urgency": u, "n": len(sel), "scored": True,
            "incumbent_share_mean": float(np.mean(inc_vals)),
            "incumbent_share_sd": float(np.std(inc_vals)),
            "var_total_mean": float(np.mean(vt)),
            "var_total_median": float(np.median(vt)),
            "incumbent_abs_var_mean": float(np.mean(inc_abs)),
            "non_incumbent_abs_var_mean": float(np.mean(non_abs)),
            "commit_variance_mean": float(np.mean([r["commit_variance"] for r in sel])),
            "effective_threshold_mean": float(
                np.mean([r["effective_threshold"] for r in sel])),
            "hazard_prox_max_mean": float(np.mean([r["hazard_prox_max"] for r in sel])),
        })
    return {"levels": levels, "n_committed": len(committed)}


def _commit_rate_by_level(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """C3: commit RATE per assigned-urgency level, over ALL fresh selections."""
    out = []
    for u in URGENCY_LEVELS:
        sel = [r for r in rows if r["assigned_urgency"] == u]
        if not sel:
            out.append({"assigned_urgency": u, "n": 0, "commit_rate": None})
            continue
        out.append({
            "assigned_urgency": u,
            "n": len(sel),
            "commit_rate": float(sum(1 for r in sel if r["committed"]) / len(sel)),
            "commit_variance_mean": float(np.mean([r["commit_variance"] for r in sel])),
            "effective_threshold_mean": float(
                np.mean([r["effective_threshold"] for r in sel])),
        })
    return out


def _hazard_conditioned_profile(rows: List[Dict[str, Any]], incumbent: str):
    """The RIVAL, MEASURED: incumbent-share-vs-urgency within hazard-proximity tertiles.

    If the profile holds within every tertile, near-hazard candidate geometry cannot be
    what produces it -- which is precisely what the endogenous 785 design could not show.
    """
    committed = [r for r in rows if r["committed"] and incumbent in r["shares"]]
    if len(committed) < 3 * HAZARD_TERTILE_MIN_N:
        return {"tertiles": [], "note": "insufficient committed rows for tertile split"}
    hz = np.asarray([r["hazard_prox_max"] for r in committed])
    lo, hi = float(np.quantile(hz, 1 / 3)), float(np.quantile(hz, 2 / 3))
    bands = [("low", -np.inf, lo), ("mid", lo, hi), ("high", hi, np.inf)]
    tertiles = []
    for name, a, b in bands:
        sel = [r for r, h in zip(committed, hz) if (a <= h < b) or (name == "high" and h >= a)]
        if len(sel) < HAZARD_TERTILE_MIN_N:
            tertiles.append({"band": name, "n": len(sel), "scored": False})
            continue
        xs, ys, vs = [], [], []
        for u in URGENCY_LEVELS:
            s2 = [r for r in sel if r["assigned_urgency"] == u]
            if len(s2) >= max(5, MIN_TICKS_PER_LEVEL // 3):
                xs.append(u)
                ys.append(float(np.mean([r["shares"][incumbent] for r in s2])))
                vs.append(float(np.mean([r["var_total"] for r in s2])))
        tertiles.append({
            "band": name, "n": len(sel), "scored": len(xs) >= 3,
            "hazard_prox_mean": float(np.mean([r["hazard_prox_max"] for r in sel])),
            "levels_scored": len(xs),
            "share_rho": _spearman_rho(xs, ys) if len(xs) >= 3 else 0.0,
            "var_total_rho": _spearman_rho(xs, vs) if len(xs) >= 3 else 0.0,
            "share_by_level": [{"u": u, "share": y} for u, y in zip(xs, ys)],
        })
    return {"tertiles": tertiles, "tertile_edges": [lo, hi]}


def _mean_shares(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    committed = [r for r in rows if r["committed"]]
    acc: Dict[str, List[float]] = {}
    for r in committed:
        for k, v in r["shares"].items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _analyse_regime(regime, rows, telem_list) -> Dict[str, Any]:
    """PER-REGIME gate + criteria. Every precondition is regime-scoped, INCLUDING P7 --
    785 applied P7 whole-run, so one arm's structural RED vacated the other's valid result.
    """
    incumbent = regime["expected_incumbent"]
    mean_shares = _mean_shares(rows)

    ranked = sorted(mean_shares.items(), key=lambda kv: -kv[1])
    actual_incumbent = ranked[0][0] if ranked else ""
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    incumbent_margin = (mean_shares.get(incumbent, 0.0) - runner_up) if ranked else 0.0

    auth = float(np.mean([t["authority_active_frac"] for t in telem_list])) if telem_list else 0.0
    var_total_mean = float(np.mean([r["var_total"] for r in rows])) if rows else 0.0
    n_committed = sum(1 for r in rows if r["committed"])
    inc_range = (float(np.mean([t["incumbent_range_mean"] for t in telem_list]))
                 if telem_list else 0.0)
    n_nontrivial = sum(1 for v in mean_shares.values() if abs(v) > NONTRIVIAL_SHARE_FLOOR)
    fid_max = float(np.max([t["fidelity_err_max"] for t in telem_list])) if telem_list else 0.0

    prof = _level_profile(rows, incumbent)
    scored = [d for d in prof["levels"] if d.get("scored")]
    n_levels_scored = len(scored)

    # P9: did randomisation hold IN SAMPLE? Correlation between the assigned level and the
    # learner-observable hazard proximity, over fresh selections.
    rand_corr = _pearson([r["assigned_urgency"] for r in rows],
                         [r["hazard_prox_max"] for r in rows]) if len(rows) >= 3 else 0.0

    def _p(name, desc, control, measured, threshold, met=None, direction="lower"):
        d = {"name": f"{regime['id']}::{name}", "kind": "readiness",
             "description": desc, "control": control, "direction": direction,
             "measured": float(measured), "threshold": float(threshold)}
        if met is None:
            d["met"] = bool(measured < threshold) if direction == "upper" \
                else bool(measured > threshold)
        else:
            d["met"] = bool(met)
        return d

    preconditions = []
    # P1 applies ONLY where the incumbent is a MODULATORY CHANNEL. In a primary-component
    # regime the modulatory channels legitimately contribute ~0 and the authority gate never
    # fires -- asserting it there would make the regime structurally un-passable.
    if incumbent.startswith("CH:"):
        preconditions.append(_p(
            "modulatory_authority_active_frac",
            "fraction of fresh selections where the authority gate fired (channel-incumbent "
            "regimes only -- N/A when the incumbent is a primary score component)",
            "candidates that genuinely differ", auth, AUTHORITY_ACTIVE_FRAC_FLOOR))
    preconditions += [
        _p("incumbent_cross_candidate_range",
           f"mean cross-candidate RANGE of the incumbent component '{incumbent}' -- the "
           f"SAME statistic the decomposition routes on, never a mean-abs proxy (643 GAP)",
           "SD-056 action-contrastive candidates", inc_range, CHANNEL_RANGE_FLOOR),
        # P7, REGIME-CONDITIONED (the 785 defect this run exists partly to fix).
        _p("n_components_with_nontrivial_share",
           f"components holding |share| > {NONTRIVIAL_SHARE_FLOOR}; a decomposition where "
           f"only one component is non-trivial is arithmetically forced, not measured. "
           f"Evaluated PER REGIME so a structurally single-component regime cannot vacate "
           f"another regime's valid result (the V3-EXQ-785 whole-run application)",
           "multi-component primary score", float(n_nontrivial), N_NONTRIVIAL_FLOOR),
        _p("urgency_levels_populated",
           "assigned urgency levels with enough committed rows to score; the exogenous "
           "analogue of 785's 'urgency non-constant' -- no spread means no contrast",
           "i.i.d. uniform assignment over the pre-registered grid",
           float(n_levels_scored), float(MIN_LEVELS_POPULATED - 0.5)),
        _p("cross_candidate_score_variance",
           "mean total cross-candidate score variance (a share of ~0 is undefined)",
           "SD-056 action-contrastive candidates + rollout clamp",
           var_total_mean, SCORE_VARIANCE_FLOOR),
        _p("fresh_committed_tick_count",
           "committed rows from FRESH E3 selections only. 785 read last_score_diagnostics "
           "without clearing it, so commitment-latched ticks re-recorded the previous "
           "tick's diagnostics as new rows (~9.0x pseudo-replication, measured). Every row "
           "here is one genuine independent selection",
           "diagnostics cleared before every select_action",
           float(n_committed), COMMITTED_TICK_FLOOR),
        _p("incumbent_identity_as_preregistered",
           f"margin of pre-registered incumbent '{incumbent}' over the runner-up; guards "
           f"against scoring 'the incumbent's share moves' for the wrong incumbent "
           f"(observed top: '{actual_incumbent}')",
           "785 harm arm, 5 seeds: harm_weighted 0.9368 / f 0.0552 / residue 0.0080",
           incumbent_margin, INCUMBENT_MARGIN_FLOOR,
           met=(actual_incumbent == incumbent and incumbent_margin > INCUMBENT_MARGIN_FLOOR)),
        # P8 -- CEILING. If the injection did not take, the manipulation did not happen and
        # nothing in this run is interpretable.
        _p("exogenous_assignment_fidelity",
           "max |realized_urgency - assigned_urgency| over all fresh selections. The "
           "manipulation IS the experiment: a miss here means urgency was not actually set",
           "probe 2026-07-19, 600 ticks: max err 2.8e-17 (float roundoff)",
           fid_max, FIDELITY_TOL, direction="upper"),
        # P9 -- CEILING. The randomisation check; the rival is broken by construction only
        # if assignment really was orthogonal to proximity in this sample.
        _p("randomisation_orthogonal_to_hazard_proximity",
           "|corr(assigned_urgency, hazard_prox_max)| over fresh selections. Near-zero is "
           "what makes near-hazard geometry an implausible explanation BY CONSTRUCTION "
           "rather than by adjustment (785's endogenous urgency was confounded by design)",
           "probe 2026-07-19: corr -0.038", abs(rand_corr), RANDOMISATION_CORR_CEILING,
           direction="upper"),
    ]
    gate_green = all(p["met"] for p in preconditions)

    # ---- C1 AMPLIFICATION and C2 CONCENTRATION, scored SEPARATELY ---- #
    xs = [d["assigned_urgency"] for d in scored]
    share_ys = [d["incumbent_share_mean"] for d in scored]
    var_ys = [d["var_total_mean"] for d in scored]

    share_rho = _spearman_rho(xs, share_ys) if len(scored) >= 3 else 0.0
    share_gap = float(share_ys[-1] - share_ys[0]) if len(scored) >= 2 else 0.0
    var_rho = _spearman_rho(xs, var_ys) if len(scored) >= 3 else 0.0
    # Variance spans ~14x in 785, so amplification is scored on a LOG ratio, not a raw gap.
    if len(scored) >= 2 and var_ys[0] > 0.0 and var_ys[-1] > 0.0:
        log_var_gap = float(math.log10(var_ys[-1] / var_ys[0]))
        var_fold = float(var_ys[-1] / var_ys[0])
    else:
        log_var_gap, var_fold = 0.0, 0.0

    # Per-seed share gaps -> effect-size floor scaled on the SD of the DELTA, plus a floor.
    per_seed_share_gaps: List[float] = []
    per_seed_log_var_gaps: List[float] = []
    for t in telem_list:
        srows = [r for r in rows if r["seed"] == t["seed"]]
        sp = _level_profile(srows, incumbent)
        ss = [d for d in sp["levels"] if d.get("scored")]
        if len(ss) >= 2:
            per_seed_share_gaps.append(
                float(ss[-1]["incumbent_share_mean"] - ss[0]["incumbent_share_mean"]))
            if ss[0]["var_total_mean"] > 0 and ss[-1]["var_total_mean"] > 0:
                per_seed_log_var_gaps.append(
                    float(math.log10(ss[-1]["var_total_mean"] / ss[0]["var_total_mean"])))
    share_gap_sd = float(np.std(per_seed_share_gaps)) if len(per_seed_share_gaps) >= 2 else 0.0
    log_var_gap_sd = (float(np.std(per_seed_log_var_gaps))
                      if len(per_seed_log_var_gaps) >= 2 else 0.0)
    share_gap_floor = max(GAP_ABS_FLOOR, GAP_SD_MULTIPLIER * share_gap_sd)
    log_var_gap_floor = max(LOG_VAR_GAP_FLOOR, GAP_SD_MULTIPLIER * log_var_gap_sd)

    enough = len(scored) >= MIN_LEVELS_POPULATED
    # SIGNED three-way verdicts. "falls" is a first-class cell distinct from "flat" --
    # 785's strong monotone decrease had no valid self-route in a rises/flat-only grid.
    share_rises = bool(enough and share_rho >= MONOTONIC_RHO_FLOOR
                       and share_gap > share_gap_floor)
    share_falls = bool(enough and share_rho <= -MONOTONIC_RHO_FLOOR
                       and share_gap < -share_gap_floor)
    var_rises = bool(enough and var_rho >= MONOTONIC_RHO_FLOOR
                     and log_var_gap > log_var_gap_floor)
    var_falls = bool(enough and var_rho <= -MONOTONIC_RHO_FLOOR
                     and log_var_gap < -log_var_gap_floor)

    commit_rates = _commit_rate_by_level(rows)
    cr_x = [c["assigned_urgency"] for c in commit_rates if c["commit_rate"] is not None]
    cr_y = [c["commit_rate"] for c in commit_rates if c["commit_rate"] is not None]
    commit_rate_rho = _spearman_rho(cr_x, cr_y) if len(cr_x) >= 3 else 0.0
    commit_rate_span = (float(max(cr_y) - min(cr_y)) if cr_y else 0.0)

    return {
        "regime": regime["id"],
        "expected_incumbent": incumbent,
        "observed_incumbent": actual_incumbent,
        "incumbent_margin": incumbent_margin,
        "mean_shares": mean_shares,
        "gate_green": gate_green,
        "preconditions": preconditions,
        "level_profile": prof,
        "levels_scored": len(scored),
        "randomisation_corr_assigned_vs_hazard": rand_corr,
        "fidelity_err_max": fid_max,
        # C2 concentration
        "share_rho": share_rho,
        "share_gap": share_gap,
        "share_gap_floor_applied": share_gap_floor,
        "share_gap_sd_across_seeds": share_gap_sd,
        "incumbent_share_rises": share_rises,
        "incumbent_share_falls": share_falls,
        # C1 amplification
        "var_total_rho": var_rho,
        "log10_var_total_gap": log_var_gap,
        "var_total_fold": var_fold,
        "log_var_gap_floor_applied": log_var_gap_floor,
        "var_total_rises": var_rises,
        "var_total_falls": var_falls,
        # C3 admission mechanism
        "commit_rate_by_level": commit_rates,
        "commit_rate_rho": commit_rate_rho,
        "commit_rate_span": commit_rate_span,
        # rival
        "hazard_conditioned": _hazard_conditioned_profile(rows, incumbent),
        "n_committed": n_committed,
        "per_seed_share_gaps": per_seed_share_gaps,
        "per_seed_log_var_gaps": per_seed_log_var_gaps,
    }


def run_experiment(dry_run: bool):
    t0 = time.perf_counter()
    n_ticks = DRY_RUN_TICKS if dry_run else TICKS_PER_SEED
    seeds = SEEDS[:2] if dry_run else SEEDS

    all_rows: List[Dict[str, Any]] = []
    arm_results: List[Dict[str, Any]] = []
    regime_analyses: List[Dict[str, Any]] = []

    for regime in REGIMES:
        regime_rows: List[Dict[str, Any]] = []
        regime_telem: List[Dict[str, Any]] = []
        for seed in seeds:
            probe_slice = _build_agent_and_env(seed, regime)[3]
            # arm_cell discharges BOTH per-cell obligations: full RNG reset on enter and
            # the fingerprint stamp. include_driver_script_in_hash=False MINTS this cell
            # reuse-eligible for a future, different-driver consumer (mint-as-you-go).
            with arm_cell(
                seed,
                config_slice=probe_slice,
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=False,
            ) as cell:
                # Assignment RNG is seeded per cell and reset by arm_cell on entry, so the
                # exogenous schedule is reproducible.
                rng = np.random.default_rng(10_000 + seed)
                rows, telem = _collect_cell(seed, regime, n_ticks, rng)
                row: Dict[str, Any] = {"arm_id": regime["id"], "seed": seed, **telem}
                cell.stamp(row)
            regime_rows.extend(rows)
            regime_telem.append(telem)
            arm_results.append(row)
            print(f"verdict: {'PASS' if telem['n_committed'] > 0 else 'FAIL'}", flush=True)

        all_rows.extend(regime_rows)
        regime_analyses.append(_analyse_regime(regime, regime_rows, regime_telem))

    # ---------------- outcome routing (six-cell grid) ---------------- #
    gate_green = all(a["gate_green"] for a in regime_analyses)
    a0 = regime_analyses[0]
    preconditions = [p for a in regime_analyses for p in a["preconditions"]]

    if not gate_green:
        outcome, evidence_direction = "FAIL", "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        failed = [p["name"] for p in preconditions if not p["met"]]
        degeneracy_reason = (
            "MECH-463 non-vacuity gate RED: " + ", ".join(failed)
            + ". A numerical/config degeneracy (or a failed exogenous injection) yields a "
              "profile indistinguishable from a genuine result, so this run is NOT scored "
              "and is NOT a refutation of MECH-463."
        )
    else:
        non_degenerate, degeneracy_reason = True, ""
        amp, conc_up, conc_dn = a0["var_total_rises"], a0["incumbent_share_rises"], \
            a0["incumbent_share_falls"]
        if amp and conc_up:
            outcome, evidence_direction = "PASS", "supports"
            label = "arousal_amplifies_and_concentrates"
        elif amp and conc_dn:
            # THE NEW CELL. 785's signature, now under exogenous control.
            outcome, evidence_direction = "FAIL", "mixed"
            label = "arousal_amplifies_but_dilutes_incumbent"
        elif amp:
            outcome, evidence_direction = "FAIL", "mixed"
            label = "arousal_amplifies_only_concentration_null"
        elif conc_dn:
            outcome, evidence_direction = "FAIL", "weakens"
            label = "arousal_redistributes_without_amplifying"
        elif conc_up:
            outcome, evidence_direction = "FAIL", "mixed"
            label = "arousal_concentrates_without_amplifying"
        else:
            outcome, evidence_direction = "FAIL", "does_not_support"
            label = "arousal_causally_inert_on_selection_variance"

    criteria = [
        {"name": "C1_amplification_var_total_rises_with_exogenous_urgency",
         "load_bearing": True, "passed": a0["var_total_rises"],
         "measured_rho": a0["var_total_rho"], "threshold_rho": MONOTONIC_RHO_FLOOR,
         "measured_log10_gap": a0["log10_var_total_gap"],
         "threshold_log10_gap": a0["log_var_gap_floor_applied"],
         "var_total_fold": a0["var_total_fold"],
         "levels_scored": a0["levels_scored"],
         "null_note": (
             "a null here means exogenous arousal does not amplify cross-candidate "
             "selection variance -- which would REVERSE 785's endogenously-measured 14.1x "
             "and attribute it to near-hazard geometry")},
        {"name": "C2_concentration_incumbent_share_rises_with_exogenous_urgency",
         "load_bearing": True, "passed": a0["incumbent_share_rises"],
         "signed_falls": a0["incumbent_share_falls"],
         "measured_rho": a0["share_rho"], "threshold_rho": MONOTONIC_RHO_FLOOR,
         "measured_gap": a0["share_gap"],
         "threshold_gap": a0["share_gap_floor_applied"],
         "levels_scored": a0["levels_scored"],
         "null_note": (
             "FLAT means arousal does not redistribute share (concentration untested-null); "
             "FALLS is a distinct, signed verdict -- arousal DILUTES the incumbent, "
             "contradicting MECH-463's concentration premise on the merits")},
        {"name": "C3_admission_commit_rate_varies_with_exogenous_urgency",
         "load_bearing": False, "passed": abs(a0["commit_rate_rho"]) >= MONOTONIC_RHO_FLOOR,
         "measured_rho": a0["commit_rate_rho"], "commit_rate_span": a0["commit_rate_span"],
         "commit_rate_by_level": a0["commit_rate_by_level"],
         "null_note": (
             "MECHANISM PROBE, deliberately NOT load-bearing. The gate is "
             "committed = variance < commit_threshold*(1-urgency), an UPPER BOUND, so "
             "raising urgency makes admission STRICTER (fewer, lower-variance ticks) -- the "
             "OPPOSITE of the autopsy's 'less selective / admits lower-separation' framing. "
             "The gated quantity is the z_world running variance (world-model stability), "
             "NOT candidate separation, since use_harm_variance_commit is off. A FLAT "
             "commit rate while the share moves RULES OUT differential admission as the "
             "mechanism; that is an informative null, not a failure. Rate is near-saturated "
             "(~0.97 measured), which bounds how much this can explain either way")},
    ]
    criteria_non_degenerate = {
        "C1_amplification_var_total_rises_with_exogenous_urgency": bool(
            a0["levels_scored"] >= MIN_LEVELS_POPULATED
            and len(a0["per_seed_log_var_gaps"]) >= 2),
        "C2_concentration_incumbent_share_rises_with_exogenous_urgency": bool(
            a0["levels_scored"] >= MIN_LEVELS_POPULATED
            and len(a0["per_seed_share_gaps"]) >= 2),
        "C3_admission_commit_rate_varies_with_exogenous_urgency": bool(
            a0["commit_rate_span"] > 0.0
            and sum(1 for c in a0["commit_rate_by_level"]
                    if c["commit_rate"] is not None) >= 3),
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    full_config = {
        "seeds": seeds,
        "ticks_per_seed": n_ticks,
        "regimes": REGIMES,
        "urgency_manipulation": "exogenous_iid_uniform_over_pre_registered_grid",
        "urgency_levels": URGENCY_LEVELS,
        "urgency_weight_baseline": URGENCY_WEIGHT_BASELINE,
        "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
        "alpha_world": ALPHA_WORLD,
        "env": {"cls": "CausalGridWorldV2", "use_proxy_fields": True, "hazard_harm": 0.5},
        "thresholds": {
            "MONOTONIC_RHO_FLOOR": MONOTONIC_RHO_FLOOR,
            "GAP_ABS_FLOOR": GAP_ABS_FLOOR,
            "GAP_SD_MULTIPLIER": GAP_SD_MULTIPLIER,
            "LOG_VAR_GAP_FLOOR": LOG_VAR_GAP_FLOOR,
            "AUTHORITY_ACTIVE_FRAC_FLOOR": AUTHORITY_ACTIVE_FRAC_FLOOR,
            "CHANNEL_RANGE_FLOOR": CHANNEL_RANGE_FLOOR,
            "SCORE_VARIANCE_FLOOR": SCORE_VARIANCE_FLOOR,
            "COMMITTED_TICK_FLOOR": COMMITTED_TICK_FLOOR,
            "INCUMBENT_MARGIN_FLOOR": INCUMBENT_MARGIN_FLOOR,
            "N_NONTRIVIAL_FLOOR": N_NONTRIVIAL_FLOOR,
            "NONTRIVIAL_SHARE_FLOOR": NONTRIVIAL_SHARE_FLOOR,
            "FIDELITY_TOL": FIDELITY_TOL,
            "RANDOMISATION_CORR_CEILING": RANDOMISATION_CORR_CEILING,
            "MIN_TICKS_PER_LEVEL": MIN_TICKS_PER_LEVEL,
            "MIN_LEVELS_POPULATED": MIN_LEVELS_POPULATED,
        },
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "timestamp_utc": ts,
        "dry_run": dry_run,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "config": full_config,
        "seeds": seeds,
        "arm_results": arm_results,
        "regime_analyses": regime_analyses,
        "metrics": {
            "incumbent": a0["expected_incumbent"],
            "share_rho": a0["share_rho"],
            "share_gap": a0["share_gap"],
            "var_total_rho": a0["var_total_rho"],
            "var_total_fold": a0["var_total_fold"],
            "commit_rate_rho": a0["commit_rate_rho"],
            "randomisation_corr": a0["randomisation_corr_assigned_vs_hazard"],
            "n_committed": a0["n_committed"],
        },
        "criteria": criteria,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "design_note": (
                "EXOGENOUS urgency: urgency_applied is SET, not observed, by solving "
                "e3.config.urgency_weight = target / ||z_harm_a|| against "
                "e3_selector.py:2704-2711. Assignment is i.i.d. uniform over a "
                "pre-registered grid, independent of state, so the 785 endogeneity "
                "confound (high urgency == near hazard) is broken BY CONSTRUCTION rather "
                "than adjusted away. Fidelity and the realised randomisation correlation "
                "are both gate preconditions (P8/P9)."
            ),
            "grid_note": (
                "Six cells. AMPLIFICATION (C1) and CONCENTRATION (C2) are scored "
                "SEPARATELY -- 785's compound criterion returned FAIL for a claim whose "
                "amplification half was confirmed at 14.1x. 'falls' is a first-class "
                "SIGNED cell distinct from 'flat': 785's strong monotone decrease had no "
                "valid self-route in a rises/flat-only grid and was buried."
            ),
            "channel_agnosticism_untested_note": (
                "MECH-463's CHANNEL-AGNOSTICISM half is UNTESTED by this run. The second "
                "regime was dropped, not replaced: an entropy_bias_scale scan over "
                "{0.15,0.30,0.50,1.00} found CH:mech341 absorbing ~99-115% of "
                "cross-candidate variance at every setting, so its share statistic is "
                "arithmetically forced. This run tests the mechanism at ONE incumbent "
                "identity (harm_weighted). No result here licenses any claim about "
                "channel-agnosticism."
            ),
            "mechanism_sign_note": (
                "The autopsy flagged (as candidate) that the commit threshold is an "
                "ADMISSION threshold whose lowering is 'LESS selective ... predicts "
                "dilution'. Reading the gate, that sign is BACKWARDS: "
                "committed = variance < commit_threshold*(1-urgency) is an UPPER BOUND, so "
                "raising urgency LOWERS the bound and makes admission STRICTER. Measured "
                "(600 ticks, exogenous): commit rate 1.000 at u=0.04 -> 0.889 at u=0.34. "
                "Further, the gated quantity is the z_world RUNNING VARIANCE "
                "(world-model stability), not cross-candidate separation, because "
                "use_harm_variance_commit is off -- so any induced share change is a "
                "selection-on-stability effect. C3 measures this directly."
            ),
            "pseudo_replication_note": (
                "785 read e3.last_score_diagnostics without clearing it. E3 select() runs "
                "only when the commitment latch is open (agent.py:4968), so latched ticks "
                "re-recorded the previous tick's diagnostics as new rows: 600 rows from 67 "
                "genuine selections, ~9.0x. 785's '3959 committed ticks / ~40 SE' reflects "
                "~440 independent selections, so its true SE is ~3x larger and the span is "
                "~15 SE. Its DIRECTION survives; only its precision is overstated. Fixed "
                "here by clearing the diagnostics before every select_action."
            ),
            "rival_measured_note": (
                "hazard_field_view (the LEARNER'S OWN 5x5 local observable, not a "
                "privileged global oracle distance -- the 732a confound) is recorded per "
                "tick and the share profile is reported within hazard-proximity tertiles. "
                "Randomisation makes proximity orthogonal to assigned urgency in "
                "expectation; P9 checks it held in sample; the tertile profile shows "
                "whether the effect survives WITHIN proximity bands."
            ),
            "caveat_z_world_under_differentiation": (
                "Under an under-differentiated z_world (participation ratio ~1.06 at "
                "world_dim=128) the per-candidate channels have little to range over. That "
                "objection bites the FIX, not the measurement -- but a does_not_support "
                "verdict must be read against it, not as unconditional."
            ),
            "scope_note": (
                "NOT gated on MECH-457 or INV-088 (decomposition probe on score geometry, "
                "not a behavioural conversion test). Does NOT re-open the 689/485/445/625 "
                "selection-face lineages. MECH-439 deliberately NOT tagged (11 "
                "substrate_ceiling autopsies); nothing here bears on it. A confirmed "
                "dilution under exogenous urgency would weaken MECH-463's concentration "
                "premise on the merits and open arousal as a candidate DIVERSITY lever for "
                "the conversion-ceiling programme."
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
        "custom_information": {
            "gov_reuse_1_check": (
                "Decisive readout: incumbent cross-candidate variance share under EXOGENOUS "
                "urgency. Checked v3_exq_785_..._20260718T212815Z_v3 (the only manifest "
                "carrying urgency_applied at all): its urgency is ENDOGENOUS by "
                "construction, so it cannot answer the exogenous question, and its per-tick "
                "sink is absent from disk so no reanalysis path exists. Not recoverable -> "
                "must run."
            ),
            "instrumentation_commit": "ree-v3 435322f (e3_score_decomp_enabled)",
            "dropped_regime_evidence": DROPPED_REGIME_EVIDENCE,
            "per_tick_sink_root_cause_785": (
                "785 declared custom_information.per_tick_sink = '<run_id>_per_tick.jsonl'; "
                "the file is absent repo-wide. Root cause: 785 ran on ree-cloud-2, and "
                "under Phase 3 the worker POSTs /result with ONLY manifest_bytes while "
                "runner-side result pushes are disabled "
                "(PHASE3_DISABLE_RUNNER_RESULT_PUSH=1) -- so a sidecar written next to the "
                "manifest on the worker's local disk is never transported. The sidecar "
                "pattern cannot work for any cloud-run experiment. Fixed by embedding the "
                "per-tick record IN the manifest (custom_information.per_tick_rows), which "
                "travels inside manifest_bytes."
            ),
            "per_tick_rows_schema": {
                "note": (
                    "One entry per GENUINE fresh E3 selection. Scalars are present for "
                    "every row; the bulky per-candidate component vectors are retained for "
                    "the first "
                    f"{PER_CANDIDATE_SAMPLE_PER_CELL} rows of each cell only, to bound "
                    "manifest size while keeping every scalar the level / commit-rate / "
                    "proximity-conditioned reanalyses need."
                ),
                "fields": [
                    "regime", "seed", "tick", "assigned_urgency", "realized_urgency",
                    "natural_urgency", "effective_threshold", "commit_variance",
                    "commit_gate_mode", "committed", "temperature", "da",
                    "hazard_prox_max", "hazard_prox_center", "hazard_prox_mean",
                    "var_total", "shares", "per_candidate (sampled)",
                ],
            },
            "per_tick_rows": all_rows,
            "driver_constraints": (
                "(1) act_with_split_obs() drops obs_harm_a -> z_harm_a None -> the urgency "
                "branch at e3_selector.py:2705 is skipped and the EXOGENOUS injection is a "
                "silent no-op; (2) update_running_variance() has no caller in ree_core "
                "(396a) -> commit gate never fires (rate 0.000 -> 0.970 once the driver "
                "drives the EMA)."
            ),
        },
    }

    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
    )
    return manifest, all_rows


# ------------------------------------------------------------------ #
# Entry point                                                        #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("V3-EXQ-785a: MECH-463 incumbent-share decomposition under EXOGENOUS urgency",
          flush=True)
    print(f"  dry_run={args.dry_run}", flush=True)

    t_start = time.perf_counter()
    manifest, rows = run_experiment(args.dry_run)

    out_path = write_flat_manifest(
        manifest,
        OUT_DIR,
        dry_run=args.dry_run,
        config=manifest.get("config"),
        seeds=manifest.get("seeds"),
        script_path=Path(__file__),
        started_at=t_start,
    )

    # Convenience sidecar ONLY. The durable record is inside the manifest
    # (custom_information.per_tick_rows) -- a sidecar cannot survive a cloud run, which is
    # exactly why 785's declared sink does not exist. Nothing in the analysis reads this.
    if not args.dry_run:
        sink = Path(out_path).parent / f"{manifest['run_id']}_per_tick.jsonl"
        try:
            with open(sink, "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            print(f"  per-tick sidecar (convenience): {sink} ({len(rows)} rows)", flush=True)
        except OSError as exc:
            print(f"  per-tick sidecar skipped ({exc}); manifest copy is authoritative",
                  flush=True)

    print(f"  outcome={manifest['outcome']} direction={manifest['evidence_direction']} "
          f"label={manifest['interpretation']['label']}", flush=True)
    for a in manifest["regime_analyses"]:
        print(f"  [{a['regime']}] incumbent={a['observed_incumbent']} "
              f"(expected {a['expected_incumbent']}) "
              f"C1 var_rho={a['var_total_rho']:.4f} fold={a['var_total_fold']:.2f}x "
              f"C2 share_rho={a['share_rho']:.4f} gap={a['share_gap']:.4f} "
              f"C3 commit_rho={a['commit_rate_rho']:.4f} "
              f"levels={a['levels_scored']} committed={a['n_committed']} "
              f"gate={'GREEN' if a['gate_green'] else 'RED'}", flush=True)
        print(f"      fidelity_err_max={a['fidelity_err_max']:.3e} "
              f"randomisation_corr={a['randomisation_corr_assigned_vs_hazard']:.4f}",
              flush=True)
        for p in a["preconditions"]:
            if not p["met"]:
                print(f"      RED {p['name']}: measured={p['measured']:.6g} "
                      f"threshold={p['threshold']:.6g} ({p['direction']})", flush=True)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
