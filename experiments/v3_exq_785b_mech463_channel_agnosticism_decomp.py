"""V3-EXQ-785b: MECH-463 -- is the arousal variance-amplification CHANNEL-AGNOSTIC? The
same exogenous-urgency probe as 785a, run at a SECOND incumbent identity.

Same scientific question as 785a, second incumbent identity. Does NOT supersede 785a --
785a's harm_weighted result stands and is the comparison arm for this run.
claim_ids = [MECH-463] ONLY; MECH-439 deliberately NOT tagged.

=== WHY THIS RUN EXISTS ===
V3-EXQ-785a (adjudicated 2026-07-19, REE_assembly e8d4325cbd) showed global scalar arousal
is causally INERT on committed-selection variance geometry: under exogenous urgency across
an 8.5x range, var_total fold 0.970 and incumbent share gap +0.0036 (under 1 SE). That
refuted MECH-463's registered concentration prediction. But MECH-463 was deliberately left
at `candidate` rather than demoted, and channel-agnosticism is the first of three reasons:

    785a tested ONE incumbent identity (harm_weighted). MECH-463 asserts the amplification
    is channel-agnostic ACROSS incumbent identities, so a single-identity refutation
    cannot settle it.

This run supplies the second identity. It is the direct consumer of the scoping spike
`REE_assembly/evidence/planning/mech463_channel_agnosticism_scoping_spike_2026-07-19.md`.

=== WHAT THE SPIKE ESTABLISHED (and why the obvious second regime is NOT used) ===
785a's authors already tried and REJECTED the entropy_incumbent regime with recorded
evidence: an entropy_bias_scale scan over {0.15,0.30,0.50,1.00} found CH:mech341 absorbing
~99-115% of cross-candidate variance at EVERY setting. Shares sum to 1.0 by construction,
so that forces every other component below the |0.01| floor and the regime can never satisfy
its own >=2-component P7 gate. Its share statistic is arithmetically FORCED, not measured.
That axis is EXCLUDED here, not re-tested. In particular: do NOT cite the entropy regime's
rho +0.5879 from V3-EXQ-785 as support for MECH-463 -- it is that same forced
single-component artifact and is that run's only claim-favourable number.

The spike then searched the remaining axes and found the baseline decomposition contains
only THREE live components (aggregated over 785a's 1765 embedded per-tick rows):

    harm_weighted   +0.9390 (sd 0.0778)   <- the 785a incumbent
    f               +0.0533 (sd 0.0684)
    residue_weighted +0.0077 (sd 0.0251)
    benefit_weighted / novelty_weighted / goal_weighted / CH:residual   all EXACTLY 0.0000

Three of those four zeros are STRUCTURAL, and that bounds what this run can license:
  - novelty_weighted is DEAD BY CONSTRUCTION. e3_selector.py:963-971 hardcodes
    _dc_novelty_w = 0.0; the MECH-111 broadcast branch that populated it was deleted
    2026-05-25 as dead-by-construction (a uniform scalar shift is argmin-invariant). A
    novelty-weighted incumbent is NOT reachable by configuration.
  - benefit_weighted is triple-gated (benefit_eval_enabled AND benefit_weight>0 AND
    _benefit_samples_seen >= 50) and the warmup did not clear inside the spike horizon.
  - goal_weighted needs an active goal_state, which this env family does not supply.

=== THE TWO GLOBAL-SCALAR ROUTES ARE STRUCTURALLY INELIGIBLE (checked in code, not scanned)
The scoping brief named D1/D2 dopamine gain and softmax temperature as candidate axes.
NEITHER can change the incumbent identity, so neither is used:
  - D1/D2 gain (e3_selector.py:1553-1570, applied 1728-1729) operates on a loop ACCUMULATOR
    (assoc_accum / limbic_accum) AFTER channel composition. It re-gains the aggregate; it is
    not an additive component in the decomposition.
  - Softmax temperature (e3_selector.py:2682, 3102) divides the composed score inside the
    softmax, strictly downstream of the per-candidate component vectors the shares are
    computed from. It can move which candidate commits; it cannot move a component share.
The viable axis is therefore the COMPONENT-WEIGHT axis (lambda_ethical, rho_residue).

=== THE TWO REGIMES, AND WHY THE PRIMARY KEEPS HARM ON ===
Scanned seed 0 / 150 ticks, then replicated across seeds 0/1/2 with torch seeded:

  PRIMARY  residue_incumbent  rho_residue = 20.0, harm channel ON (lambda_ethical default)
      seed 0: residue 0.585 / harm 0.358 / f 0.057
      seed 1: residue 0.887 / harm 0.097 / f 0.016
      seed 2: residue 0.868 / harm 0.123
    Incumbency is stable (residue_weighted at all three seeds, never harm_weighted), every
    share is BELOW 1.0 with no negative runner-up, and harm_weighted survives as a genuine
    second component. Critically it is a REWEIGHTING, not an ablation: the harm channel
    stays ON, so a result here speaks to channel-agnosticism rather than to harm-channel
    necessity. Mean incumbent share 0.780 -- comfortably under 1.0, so P7 is SATISFIABLE in
    principle, which is exactly what the entropy regime could never claim.

  SECONDARY  f_incumbent  lambda_ethical = 0.05
      seed 0: f 0.825 / harm 0.090 / residue 0.084
      seed 1: f 0.984 / residue 0.047 / harm -0.031
      seed 2: f 0.683 / residue 0.185 / harm 0.132
    Replicates, but seed 1 lands at 0.984 with a NEGATIVE harm share -- close to the forced
    boundary -- so it is the weaker of the two and is scored as a replication check.

REJECTED as forced (incumbent ~1.0 with a NEGATIVE remainder -- the same cancellation
pathology as the entropy regime in a different costume, even though they technically clear
n_nontrivial>=2): lambda_ethical=0.0 (f, +1.049), lambda=0.0+rho_residue=5.0
(residue, +1.014), goal_weight=1.0+lambda=0.0 (f, +1.015). The DISCRIMINATING criterion is
incumbent share comfortably below 1.0 WITH same-signed runners-up.

=== THE 785a DEFECT THIS RUN FIXES: WHOLE-RUN GATE AGGREGATION ===
785a computed `gate_green = all(a["gate_green"] for a in regime_analyses)` and routed all
criteria off `regime_analyses[0]`. With ONE regime that never bit. With TWO regimes it is
EXACTLY the whole-run AND-aggregation that the V3-EXQ-785 autopsy identified as its sharpest
finding -- one arm's structurally impossible precondition silently vacating another arm's
valid, well-powered result. Fixed here with experiments/_lib/precondition_gate.py:
`aggregate_arm_gates` makes non_degenerate = ANY arm green, criteria are scored on the
highest-priority GREEN regime, and `arm_criteria_non_degenerate` keys each criterion to its
owning arm's gate. A red secondary can no longer vacate a green primary.

=== A NON-DEFECT, RECORDED SO IT IS NOT RE-RAISED: torch SEEDING IN 785a ===
The scoping spike initially reported 785a as failing to seed torch, on the grounds that the
file never calls torch.manual_seed directly and an identical baseline config produced
incumbent share 0.9313 and 0.9649 on two runs of the same seed. THAT FINDING WAS WRONG and
is retracted here. 785a's operative agent is built inside _collect_cell, which runs INSIDE
the `with arm_cell(seed, ...)` context, and arm_cell's entry calls reset_all_rng(seed) ->
torch.manual_seed(seed) (+cuda, numpy, random, and the _harness fallback RNG). So 785a's
cells ARE pure functions of (substrate, config, seed) and its arm_fingerprint
reuse-eligibility is sound. The variation the spike measured came from the spike's OWN
scratch harness, which built agents directly with no arm_cell wrapper and therefore no RNG
reset -- a defect in the throwaway scan script, not in 785a.

This run keeps an explicit torch.manual_seed(seed) in _build_agent_and_env anyway. It is
REDUNDANT with arm_cell's reset on the production path and is retained only so the builder
is self-contained for a direct caller (a future scratch probe or contract test that
constructs an agent outside a cell) -- i.e. so the exact confusion above cannot recur. It
is not a fix for anything in 785a.

=== EVERYTHING ELSE IS 785a's DESIGN, DELIBERATELY UNCHANGED ===
Preserved verbatim because they are what make the readout interpretable at all:
  - EXOGENOUS urgency. e3.config.urgency_weight = target / ||z_harm_a|| lands
    urgency_applied on target exactly (e3_selector.py:2704-2711, weight read live at
    select()). Assignment i.i.d. uniform over a pre-registered grid, INDEPENDENT of state,
    so the near-hazard confound is broken BY CONSTRUCTION. Grid 0.04..0.34 brackets 785's
    endogenous span 0.070..0.125; top level sits below urgency_max 0.5.
  - e3.last_score_diagnostics cleared before EVERY select_action. E3 select() runs only when
    the commitment latch is open (agent.py:4968), so without the clear a latched tick
    re-records the previous tick's diagnostics as a new row -- the ~9.0x pseudo-replication
    defect measured in 785. Every row here is one genuine independent selection.
  - Per-tick rows embedded IN the manifest (custom_information.per_tick_rows), never a
    sidecar. Under Phase 3 the worker POSTs /result and the coordinator spools ONLY
    manifest_bytes, so a sidecar written next to the manifest on a cloud worker is NEVER
    transported -- the root cause of 785's absent per_tick sink.
  - hazard_field_view (the LEARNER'S OWN 5x5 local observable, not a privileged global
    oracle distance -- the 732a confound) recorded per tick; share profile reported within
    hazard-proximity tertiles.
  - The two driver constraints that silently vacate the probe: (1) act_with_split_obs()
    passes no obs_harm_a so z_harm_a is None and the exogenous injection becomes a silent
    no-op -- the harm stream MUST be fed explicitly and the wrapper replicated; (2)
    V3-EXQ-396a: update_running_variance() has no caller in ree_core, so the driver must
    drive the EMA itself or the commit gate never fires. Do not "simplify" either away.

=== PRE-REGISTERED CRITERIA (unchanged from 785a; scored on the primary GREEN regime) ===
  C1 AMPLIFICATION -- total cross-candidate score variance RISES with exogenous urgency
                      (Spearman rho over level means >= +0.60, positive log-ratio gap).
  C2 CONCENTRATION -- the incumbent's SHARE rises with exogenous urgency (rho >= +0.60).
  C3 ADMISSION     -- commit RATE per urgency level (mechanism probe; NOT load-bearing).

INTERPRETATION GRID (six cells; "falls" is a first-class SIGNED cell, rho <= -0.60 with a
negative gap clearing the same effect-size floor -- distinct from "flat", |rho| < 0.60):

  C1 rises + C2 rises  -> SUPPORTS at this second identity -> channel-agnosticism SUPPORTED
  C1 rises + C2 FALLS  -> AMPLIFIES_BUT_DILUTES
  C1 rises + C2 flat   -> AMPLIFIES_ONLY
  C1 flat  + C2 FALLS  -> REDISTRIBUTES_ONLY
  C1 flat  + C2 flat   -> REFUTES: arousal causally inert here TOO. Combined with 785a's
                          harm_weighted null this is the decisive reading -- inert at BOTH
                          tested identities, which closes the channel-agnosticism escape
                          route and clears MECH-463 to be demoted on the merits.
  gate RED             -> substrate_not_ready_requeue. NEVER a substrate verdict.

=== WHAT A PASS AND A NULL EACH LICENSE (breadth is BOUNDED -- read this) ===
Because novelty_weighted is dead by construction and benefit/goal are unreachable in this
env family, a result across {harm_weighted, residue_weighted} licenses agnosticism across
TWO COST-SIDE PRIMARY COMPONENTS -- NOT across the full registered channel set, and NOT
across the modulatory CH:* channels (whose only reachable instance is the arithmetically
forced entropy regime). State that bound in any governance action citing this run.

=== CAVEAT: z_world UNDER-DIFFERENTIATION (weakened, not resolved) ===
The standing concern (participation ratio ~1.06 at world_dim=128, absolute variances
~1.2e-05) predicts that forced-single-component outcomes are a symptom of degenerate world
representation rather than of decomposition design. The primary regime is mild evidence
AGAINST that reading: at UNCHANGED world_dim a three-way split of 0.585/0.358/0.057 is
attainable, so the baseline's single-component profile is at least partly a
component-weight fact rather than purely a z_world fact. This weakens the caveat for the
SHARE readout only; it does NOT clear it for the absolute-variance readout.

=== SCOPE ===
rho_residue=20.0 against a default of 0.5 is a 40x reweighting -- a legitimate probe of
channel-agnosticism (the claim is about incumbent identity, not about the default operating
point) but NOT a claim about how the agent normally behaves. Not gated on MECH-457 or
INV-088 (a decomposition probe on score geometry, not a behavioural conversion test). Does
NOT re-open the 689/485/445/625 selection-face lineages. MECH-439 NOT tagged.
GOV-REUSE-1: the decisive readout is the incumbent share under exogenous urgency AT A
NON-harm INCUMBENT. Checked 785 and 785a (the only runs carrying urgency_applied at all):
both run rho_residue at its 0.5 default, so no recorded manifest carries this readout and it
is not recoverable by reanalysis. Must run.

Run:
  /opt/local/bin/python3 experiments/v3_exq_785b_mech463_channel_agnosticism_decomp.py --dry-run
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
from experiments._lib.precondition_gate import (  # noqa: E402
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
)

# ------------------------------------------------------------------ #
# Identity                                                           #
# ------------------------------------------------------------------ #
EXPERIMENT_TYPE = "v3_exq_785b_mech463_channel_agnosticism_decomp"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["MECH-463"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
# Deliberately None: this run does NOT supersede 785a. Same scientific question at a
# SECOND incumbent identity -- 785a's harm_weighted result stands and is this run's
# comparison arm. Stamping `supersedes` here would mark 785a inactive in the indexer and
# destroy exactly the contrast the channel-agnosticism question is built on.
SUPERSEDES = None

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

# TWO regimes, both at entropy_bias_scale 0.0 (the entropy axis is EXCLUDED -- see the
# docstring; CH:mech341 absorbs ~99-115% of variance at every setting, arithmetically
# forced). `config_overrides` are applied to cfg.e3 AFTER REEConfig.from_dims, so they
# reweight the primary score components without touching substrate code.
#
# Order is PRIORITY order: criteria are scored on the highest-priority GREEN regime.
# expected_incumbent_share values are the seed-mean from the scoping spike replication
# (seeds 0/1/2, 150 ticks) -- all comfortably BELOW 1.0, so P7 is satisfiable in
# principle, which is what the dropped entropy regime could never claim.
REGIMES = [
    {"id": "residue_incumbent", "entropy_bias_scale": 0.0,
     "config_overrides": {"rho_residue": 20.0},
     "expected_incumbent": "residue_weighted", "expected_incumbent_share": 0.780,
     "role": "primary",
     "scan_note": "seeds 0/1/2: residue .585/.887/.868, harm survives at .358/.097/.123; "
                  "harm channel stays ON -- a reweighting, not an ablation"},
    {"id": "f_incumbent", "entropy_bias_scale": 0.0,
     "config_overrides": {"lambda_ethical": 0.05},
     "expected_incumbent": "f", "expected_incumbent_share": 0.831,
     "role": "secondary_replication",
     "scan_note": "seeds 0/1/2: f .825/.984/.683; seed 1 carries a NEGATIVE harm share and "
                  "sits close to the forced boundary, so this is the weaker regime"},
]
# Provenance of the two regimes above. The entropy axis stays EXCLUDED (785a's recorded
# evidence, reproduced here so this manifest is self-contained); the component-weight axis
# is where the viable second identity was found.
REGIME_SELECTION_EVIDENCE = {
    "scoping_spike": (
        "REE_assembly/evidence/planning/"
        "mech463_channel_agnosticism_scoping_spike_2026-07-19.md"
    ),
    "excluded_axis_entropy": {
        "reason": (
            "CH:mech341 absorbs ~99-115% of cross-candidate variance at every "
            "entropy_bias_scale scanned, so its share statistic is arithmetically FORCED. "
            "Recorded by 785a; NOT re-tested here."
        ),
        "scan_seed0_150ticks": {
            "0.15": {"incumbent": "CH:mech341", "share": 1.1478, "n_nontrivial": 3},
            "0.30": {"incumbent": "CH:mech341", "share": 0.9972, "n_nontrivial": 1},
            "0.50": {"incumbent": "CH:mech341", "share": 0.9927, "n_nontrivial": 1},
            "1.00": {"incumbent": "CH:mech341", "share": 0.9911, "n_nontrivial": 2},
        },
        "do_not_cite": (
            "V3-EXQ-785's entropy-regime rho +0.5879 must NOT be cited as support for "
            "MECH-463 -- it is this same forced single-component artifact."
        ),
    },
    "excluded_axis_global_scalars": {
        "d1_d2_dopamine_gain": (
            "e3_selector.py:1553-1570, applied 1728-1729 -- operates on a loop ACCUMULATOR "
            "after channel composition; re-gains the aggregate, not an additive component."
        ),
        "softmax_temperature": (
            "e3_selector.py:2682, 3102 -- divides the composed score inside the softmax, "
            "downstream of the per-candidate component vectors shares are computed from."
        ),
        "conclusion": "neither can change incumbent IDENTITY; structurally ineligible.",
    },
    "structurally_unreachable_incumbents": {
        "novelty_weighted": (
            "DEAD BY CONSTRUCTION -- e3_selector.py:963-971 hardcodes _dc_novelty_w = 0.0; "
            "the MECH-111 broadcast branch was deleted 2026-05-25 as argmin-invariant. "
            "Not reachable by configuration; needs substrate work."
        ),
        "benefit_weighted": (
            "triple-gated (benefit_eval_enabled AND benefit_weight>0 AND "
            "_benefit_samples_seen >= 50); warmup did not clear inside the spike horizon."
        ),
        "goal_weighted": "needs an active goal_state, absent in this env family.",
    },
    "rejected_as_forced": {
        "note": (
            "incumbent ~1.0 with a NEGATIVE remainder -- the same cancellation pathology "
            "as the entropy regime, even though these technically clear n_nontrivial>=2. "
            "The discriminating criterion is incumbent share comfortably below 1.0 WITH "
            "same-signed runners-up."
        ),
        "lambda_ethical=0.0": {"incumbent": "f", "share": 1.0491},
        "lambda=0.0+rho_residue=5.0": {"incumbent": "residue_weighted", "share": 1.0140},
        "goal_weight=1.0+lambda=0.0": {"incumbent": "f", "share": 1.0151},
    },
    "licensed_breadth_bound": (
        "A result across {harm_weighted (785a), residue_weighted (here)} licenses "
        "agnosticism across TWO COST-SIDE PRIMARY COMPONENTS -- NOT the full registered "
        "channel set, and NOT the modulatory CH:* channels, whose only reachable instance "
        "is the arithmetically forced entropy regime. State this bound in any governance "
        "action citing this run."
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
    # REDUNDANT on the production path, deliberately kept. arm_cell's entry already calls
    # reset_all_rng(seed) -> torch.manual_seed(seed), and _collect_cell builds the agent
    # inside that context -- so 785a was NOT unseeded (see the docstring retraction; the
    # spike's contrary finding came from its own arm_cell-less scratch harness). This line
    # exists so the builder is self-contained for a DIRECT caller outside a cell -- a
    # scratch probe or contract test -- where the RNG would otherwise be whatever the
    # previous work left it.
    torch.manual_seed(seed)
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

    # THE MANIPULATION THAT DEFINES THE REGIME: reweight the primary score components so a
    # component OTHER than harm_weighted becomes the incumbent. Applied AFTER from_dims
    # because these are e3-config scalars read live in score_trajectory
    # (e3_selector.py:987 `score = f + lambda_eff * m + rho_residue * phi`), not
    # constructor dims. No substrate change; the harm channel stays ON in the primary
    # regime, which is what makes this a reweighting rather than an ablation.
    for _k, _v in (regime.get("config_overrides") or {}).items():
        if not hasattr(cfg.e3, _k):
            raise AttributeError(
                f"regime '{regime['id']}' config_override '{_k}' is not an e3 config "
                f"field -- a silently-ignored override would run the BASELINE regime "
                f"under this regime's label and self-route a false channel-agnosticism "
                f"result"
            )
        setattr(cfg.e3, _k, float(_v))

    agent = REEAgent(cfg)
    agent.eval()
    agent.e3.e3_score_decomp_enabled = True  # MECH-463 instrumentation gate (ree-v3 435322f)

    # The returned slice is what arm_cell hashes into the per-cell fingerprint, so it MUST
    # carry the regime overrides. `kw` alone does not: the overrides are applied to cfg.e3
    # AFTER from_dims, so both regimes would hash IDENTICALLY at the same seed and a
    # reuse consumer could serve a residue_incumbent cell for an f_incumbent request --
    # a false cache HIT, which corrupts a conclusion rather than merely wasting compute
    # (arm_reuse_fingerprint_plan.md sec 2, the governing asymmetry).
    config_slice = dict(kw)
    config_slice["regime_id"] = regime["id"]
    config_slice["e3_config_overrides"] = dict(regime.get("config_overrides") or {})
    return agent, env, obs_dict, config_slice


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
           "scoping spike 2026-07-19, seeds 0/1/2 x 150 ticks: residue_incumbent -> "
           "residue .585/.887/.868 (harm .358/.097/.123); f_incumbent -> f "
           ".825/.984/.683. Incumbency was stable across all three seeds in both regimes",
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
    # 785a computed `all(a["gate_green"] for a in regime_analyses)` and routed every
    # criterion off regime_analyses[0]. With ONE regime that never bit; with TWO it is
    # EXACTLY the whole-run AND-aggregation the V3-EXQ-785 autopsy identified as its
    # sharpest finding -- one arm's failed precondition silently vacating another arm's
    # valid, well-powered result. aggregate_arm_gates makes non_degenerate = ANY arm
    # green, and criteria are scored on the highest-PRIORITY GREEN regime (REGIMES is in
    # priority order), so a red secondary cannot vacate a green primary.
    arm_gates = [
        {"arm": a["regime"],
         "gate_green": a["gate_green"],
         "preconditions": a["preconditions"],
         "failed_preconditions": [p["name"] for p in a["preconditions"] if not p["met"]],
         "scoped_out": a.get("scoped_out", [])}
        for a in regime_analyses
    ]
    aggregate = aggregate_arm_gates(arm_gates)
    gate_green = aggregate["any_green"]
    # The SCORED regime: highest-priority green one. None when every regime is red.
    a0 = next((a for a in regime_analyses if a["gate_green"]), regime_analyses[0])
    scored_regime_id = a0["regime"] if gate_green else None
    # Green-arms-only on a partial run: build_experiment_indexes._compute_adjudication
    # reads interpretation.preconditions FLAT and ARM-BLIND and returns precondition_unmet
    # for the WHOLE RUN on the first unmet entry -- so flattening every arm's
    # preconditions here would reproduce the 785 vacating at adjudication time even with
    # the routing above fixed. Red arms are carried in full under per_arm_gate.
    preconditions = aggregate["adjudication_preconditions"]

    if not gate_green:
        outcome, evidence_direction = "FAIL", "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = (
            aggregate["degeneracy_reason"]
            + " EVERY regime is red, so no incumbent identity was measured cleanly and "
              "this run is NOT a refutation of MECH-463 -- neither of its concentration "
              "premise nor of its channel-agnosticism premise."
        )
    else:
        non_degenerate = True
        # Non-empty only on a PARTIAL run (some regime red): names the red arms AND the
        # green ones that ARE scored, so a reader is never left inferring which is which.
        degeneracy_reason = aggregate["degeneracy_reason"]
        amp, conc_up, conc_dn = a0["var_total_rises"], a0["incumbent_share_rises"], \
            a0["incumbent_share_falls"]
        # Labels are suffixed with the SCORED incumbent identity: the whole point of this
        # run is that a verdict is only ever ABOUT the identity it was measured at, and an
        # unsuffixed label is exactly how 785a's single-identity result came to be read as
        # though it settled the channel-agnostic claim.
        _inc = a0["expected_incumbent"]
        if amp and conc_up:
            outcome, evidence_direction = "PASS", "supports"
            label = f"arousal_amplifies_and_concentrates_at_{_inc}"
        elif amp and conc_dn:
            outcome, evidence_direction = "FAIL", "mixed"
            label = f"arousal_amplifies_but_dilutes_incumbent_at_{_inc}"
        elif amp:
            outcome, evidence_direction = "FAIL", "mixed"
            label = f"arousal_amplifies_only_concentration_null_at_{_inc}"
        elif conc_dn:
            outcome, evidence_direction = "FAIL", "weakens"
            label = f"arousal_redistributes_without_amplifying_at_{_inc}"
        elif conc_up:
            outcome, evidence_direction = "FAIL", "mixed"
            label = f"arousal_concentrates_without_amplifying_at_{_inc}"
        else:
            # The decisive cell for the channel-agnosticism question: combined with 785a's
            # harm_weighted null this is inert at BOTH tested identities, which closes the
            # escape route that kept MECH-463 at `candidate`. Bounded by
            # REGIME_SELECTION_EVIDENCE["licensed_breadth_bound"] -- two cost-side primary
            # components, NOT the full channel set.
            outcome, evidence_direction = "FAIL", "does_not_support"
            label = f"arousal_causally_inert_on_selection_variance_at_{_inc}"

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
    # Each criterion is keyed to the gate of the arm that OWNS it (all three are scored on
    # the same green regime a0), with the independent power checks passed as `extra`. This
    # is the per-criterion channel build_experiment_indexes.py honours, and it is what
    # makes a green arm's result visibly separable at adjudication time instead of needing
    # an autopsy to recover.
    _c1 = "C1_amplification_var_total_rises_with_exogenous_urgency"
    _c2 = "C2_concentration_incumbent_share_rises_with_exogenous_urgency"
    _c3 = "C3_admission_commit_rate_varies_with_exogenous_urgency"
    criteria_non_degenerate = arm_criteria_non_degenerate(
        {a0["regime"]: [_c1, _c2, _c3]},
        aggregate,
        extra={
            _c1: bool(a0["levels_scored"] >= MIN_LEVELS_POPULATED
                      and len(a0["per_seed_log_var_gaps"]) >= 2),
            _c2: bool(a0["levels_scored"] >= MIN_LEVELS_POPULATED
                      and len(a0["per_seed_share_gaps"]) >= 2),
            _c3: bool(a0["commit_rate_span"] > 0.0
                      and sum(1 for c in a0["commit_rate_by_level"]
                              if c["commit_rate"] is not None) >= 3),
        },
    )

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
        # TOP LEVEL, deliberately: 785's manifest DID carry per-regime gate_green inside
        # regime_analyses, but nothing the indexer or pending_review reads ever saw it,
        # which is why its clean arm's result was recoverable only by autopsy.
        "per_arm_gate": aggregate["per_arm_gate"],
        "scored_regime": scored_regime_id,
        "metrics": {
            "incumbent": a0["expected_incumbent"],
            "scored_regime": scored_regime_id,
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
            "channel_agnosticism_scope_note": (
                "THIS is the run that tests MECH-463's channel-agnosticism half, which "
                "785a explicitly left untested. It measures a SECOND incumbent identity "
                "(residue_weighted primary, f secondary) against 785a's harm_weighted. "
                "BREADTH IS BOUNDED: novelty_weighted is dead by construction "
                "(e3_selector.py:963-971), benefit/goal are unreachable in this env "
                "family, and the only reachable CH:* incumbent is the arithmetically "
                "forced entropy regime -- so a result here licenses agnosticism across "
                "TWO COST-SIDE PRIMARY COMPONENTS, NOT the full registered channel set. "
                "See custom_information.regime_selection_evidence.licensed_breadth_bound."
            ),
            "combined_reading_with_785a_note": (
                "The decisive combination: 785a measured var_total fold 0.970 and share "
                "gap +0.0036 (under 1 SE) at harm_weighted. If this run is ALSO flat at "
                "residue_weighted, arousal is causally inert at BOTH tested identities, "
                "which closes the channel-agnosticism escape route that kept MECH-463 at "
                "`candidate` rather than demoted. If instead it CONCENTRATES here, "
                "MECH-463 survives in an identity-CONDITIONAL form and 785a's null is "
                "revealed as specific to the harm channel -- a materially different claim "
                "from the registered channel-agnostic one, and it should be re-registered "
                "as such rather than counted as a plain confirmation."
            ),
            "gate_aggregation_note": (
                "non_degenerate is ANY-arm-green, not all-arms-green, via "
                "_lib/precondition_gate.aggregate_arm_gates. 785a used "
                "all(a['gate_green'] ...) and routed off regime_analyses[0]; harmless at "
                "one regime, but at two it is exactly the whole-run AND-aggregation the "
                "V3-EXQ-785 autopsy identified (sections 2a/8) as one arm's structurally "
                "impossible precondition vacating another arm's valid result. Criteria "
                "are scored on the highest-priority GREEN regime; red arms are carried "
                "in full at top-level per_arm_gate, and interpretation.preconditions "
                "carries GREEN arms only because _compute_adjudication reads that list "
                "flat and arm-blind."
            ),
            "torch_seeding_retraction_note": (
                "RETRACTION. The scoping spike initially reported 785a as failing to seed "
                "torch (it never calls torch.manual_seed directly, and an identical config "
                "gave incumbent share 0.9313 and 0.9649 on two runs of one seed). That "
                "finding was WRONG. 785a builds its operative agent inside _collect_cell, "
                "which runs INSIDE the arm_cell context, and arm_cell entry calls "
                "reset_all_rng(seed) -> torch.manual_seed(seed) (+cuda/numpy/random/harness). "
                "785a's cells ARE pure functions of (substrate, config, seed) and its "
                "reuse-eligibility is sound. The variation came from the spike's OWN scratch "
                "harness, which built agents with no arm_cell wrapper. The explicit "
                "torch.manual_seed here is REDUNDANT on the production path and retained "
                "only to make the builder self-contained for direct callers."
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
            "regime_selection_evidence": REGIME_SELECTION_EVIDENCE,
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

    print("V3-EXQ-785b: MECH-463 channel-agnosticism -- incumbent-share decomposition "
          "under EXOGENOUS urgency at a SECOND incumbent identity",
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
