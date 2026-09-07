"""NOT QUEUED -- DESIGN REFUSED AT /queue-experiment STEP 4.5 (red-team BLOCKING, fable,
2026-09-07, confirmed by two independent probes in this directory). Lives under
experiments/_scratch/ deliberately: design-complete and smoke-green, but at production
CEM settings the elite-selection channel cannot move the proposal centroid above the
pre-registered 0.02 floor on this bench (oracle-elite ceiling +0.0001..+0.001), while the
positive control moves the DV by ~0.27 through ao_std rescaling on a non-linear decoder --
a channel the scored arms shut by construction. A null here would be unattributable.
Refusal record + successor design + spike:
  REE_assembly/evidence/planning/exq1005_mech267_location_dv_redteam_blocking_20260907.md
Do not move this file back into experiments/ or queue it as-is.

V3-EXQ-1005: MECH-267 mode-conditioned proposal CONTENT (location) separation at
production num_cem_iterations=3, under a properly-powered paired-contrast gate (EVIDENCE).

WHY THIS RUN EXISTS. MECH-267 asserts mode-conditioned hippocampal proposal
CONTENT: "external_task mode proposes task-relevant trajectories; internal_replay
mode replays past-successful trajectories" -- i.e. different operating_modes should
propose trajectories drawn from DIFFERENT REGIONS of action-object space. Every
experiment in the lineage to date (V3-EXQ-869 / 869a / 923 / 927 / 928) has instead
measured proposal BREADTH -- `mean_raw_std_by_dim`, the std across candidates of the
decoder raw output. Breadth is a DISPERSION statistic; it is silent about WHERE in
action-object space the proposals sit. The claim's own registered falsifiers are
content claims, and the claim's `what_would_answer` block explicitly asks for a
"properly-powered paired contrast" to replace V3-EXQ-928's acknowledged
under-powered gate (across-seed mean vs an absolute FLOOR_PRODUCTION=0.01, where
per-seed SD 0.011-0.018 exceeds the floor itself).

THIS RUN measures the never-measured half: the LOCATION of the proposal
distribution, per operating_mode, at the production num_cem_iterations=3, with a
paired per-seed contrast against a mode-blind control.

PRIMARY DV -- standardized cross-mode centroid separation. For a mode pair (a, b),
from ONE propose_trajectories call per mode, read `mean_by_action_dim` and
`std_by_action_dim` off the SAME diagnostic dict the lineage already uses
(`action_object_decoder_raw_output_stats`; the mean vector has simply never been
read before). Then

    d(a, b) = sqrt( mean_over_dims[ (mu_a[i] - mu_b[i])^2 / (0.5 * (sd_a[i]^2 + sd_b[i]^2)) ] )

a per-dimension-standardized RMS separation: dimensionless, and by construction a
LOCATION statistic -- two modes with the same centroid and different spread give
d ~ the sampling floor, not a large value. The run-level DV is the mean of d over
all SIX unordered mode pairs (V3-EXQ-928 scored ONE hand-picked extreme pair,
internal_planning vs offline_consolidation, which -- see the horizon note below --
happens to be degenerate for one of the two facets under test).

THE CONFOUND THIS DESIGN EXISTS TO CONTROL. d is not automatically clean: a
mode-dependent noise scale changes the CEM sampling breadth, and a wider sample
feeding the top-k elite refit pulls the refit ao_mean further from the terrain
prior -- so a pure BREADTH manipulation moves LOCATION downstream, through
elite selection, without any content selection whatsoever. Authoring-time probe
(8 seeds, this exact instrument) measured that path directly: the noise-scale +
mode_partitioned_cem arm moved d by +0.024 to +0.171 while DOUBLING the cross-mode
breadth spread (0.0124 -> 0.0267). So an arm that changes breadth cannot be read as
content evidence. The design closes this by scoring ONLY breadth-matched arms
(per-mode noise scale pinned UNIFORM at 1.0, so the breadth channel is shut by
construction) and by carrying an explicit per-arm breadth-invariance guard (C4).

ARMS (5) -- all at num_cem_iterations=3, all sharing seed, terrain weights and
(z_world, z_self) within a seed, so arms differ ONLY by the named config facets:

  1. CTRL_OFF      mode_conditioning_enabled=False. operating_mode is ignored
                   entirely by the substrate. NULL CALIBRATION: whatever d it
                   reports is pure per-cell sampling noise, and it is the paired
                   baseline every other arm is differenced against.
  2. CTRL_UNIFORM  conditioning ON, mode_noise_scale and mode_horizon_scale both
                   pinned UNIFORM (1.0), no H2, no H3. INSTRUMENT LEAK CHECK:
                   mode conditioning is switched on but carries no per-mode
                   information, so this arm MUST reproduce CTRL_OFF. (Authoring
                   probe: bit-identical, delta exactly 0.0 on 8/8 seeds.) A
                   non-zero delta here means the instrument manufactures
                   separation from the switch alone -- an instrument defect, and
                   the run self-routes substrate_not_ready_requeue.
  3. H2_ONLY       conditioning ON, noise + horizon UNIFORM, mode_value_weight SET
                   (distinct per-mode world_dim vectors), mode_partitioned_cem
                   False. SCORED CONTENT ARM. H2 is the one facet whose design
                   intent is content selection: it subtracts a mode-dependent
                   w . mean_z_world term from the CEM elite score, so the elite
                   SET -- which trajectories get selected -- stays mode-dependent
                   on every refit. V3-EXQ-928 declared H2 "a clean null"
                   (t=+0.48), but measured it on the BREADTH DV, which a pure
                   ranking term is not expected to move. This arm re-asks that
                   question on the DV H2 was built for.
  4. HORIZON_ONLY  conditioning ON, noise UNIFORM, mode_horizon_scale at its live
                   DEFAULTS, no H2/H3. SCORED CONTENT ARM. Mode-conditioned
                   scoring-window depth changes WHICH candidates are elite, so it
                   is a location route, not a breadth route. At horizon=4 the
                   default map {external_task 0.5, internal_planning 1.0,
                   internal_replay 0.7, offline_consolidation 1.0} yields
                   effective horizons 2 / 4 / 3 / 4 -- three distinct windows, and
                   note that V3-EXQ-928's scored extreme pair (internal_planning
                   vs offline_consolidation) maps to 4 vs 4, i.e. this facet was
                   INERT on the only pair 928 scored. Averaging over all six pairs
                   removes that accident.
  5. NOISE_H3_REF  conditioning ON, mode_noise_scale at DEFAULTS, horizon UNIFORM,
                   mode_partitioned_cem=True. REFERENCE / POSITIVE CONTROL ONLY --
                   deliberately SCOPED OUT of the content criteria. Its
                   manipulation acts on breadth and reaches d through the
                   elite-refit confound above, so a large d here is NOT content
                   evidence. It is in the design because it is the DV-CAN-MOVE
                   positive control (P4): without it, a null on arms 3 and 4 is
                   unattributable between "no content effect" and "this DV cannot
                   move on this instrument".

PRE-REGISTERED ACCEPTANCE (paired, per seed; scored arms = H2_ONLY, HORIZON_ONLY).
Per seed s and arm A, delta(A, s) = dbar(A, s) - dbar(CTRL_OFF, s), where dbar is
the six-pair mean. An arm CLEARS iff all four hold:

  C1 floor      mean_s delta(A) >= max(CONTENT_FLOOR_ABS, CONTENT_FLOOR_REL * mean_s dbar(CTRL_OFF))
                = max(0.02, 0.15 * the measured sampling-noise scale). Both an
                absolute floor and a floor referenced to the run's own null, so a
                degenerate near-zero null cannot make the relative floor trivial.
  C2 power      paired t = mean/(sd/sqrt(n_seeds)) >= 3.0. At n=30 this is
                dz >= 0.55, i.e. the gate scales on the SD OF THE PAIRED DELTA --
                the specific repair V3-EXQ-928's under-powered absolute-floor gate
                was asked for.
  C3 consistency  >= 21/30 seeds with delta(A, s) > 0.
  C4 breadth-invariance (CONFOUND GUARD). mean_s breadth_spread(A) must not exceed
                mean_s breadth_spread(CTRL_OFF) by more than BREADTH_TOL (20%),
                where breadth_spread is max-minus-min over modes of the per-mode
                mean std. This is what LICENSES reading the arm's location delta as
                content rather than as the elite-refit consequence of a breadth
                change. An arm failing C4 is reported but does not count as
                content evidence.

  OUTCOME: PASS iff >= 1 scored arm clears C1 AND C2 AND C3 AND C4, with all
  preconditions met. FAIL otherwise.

PRECONDITIONS (readiness; measured, not asserted).
  P1 instrument_null_calibrated  mean |delta(CTRL_UNIFORM)| <= 1e-9 (upper bound).
     Violated -> substrate_not_ready_requeue: the instrument reports separation
     from switching mode conditioning on with no per-mode information.
  P2 manipulation_engaged        every cell's engagement diagnostics match its arm
     spec (_last_mode_value_weight_active, _last_mode_partitioned_cem,
     _last_mode_noise_scale, and for HORIZON_ONLY that _last_effective_horizon
     actually takes >= 2 distinct values across the four modes).
  P3 readout_populated           mean_by_action_dim and std_by_action_dim are
     non-empty finite vectors on every cell, and pooled variance is > 0.
  P4 location_dv_can_move        NOISE_H3_REF's mean delta clears the C1 floor.
     THE positive control: it proves d is capable of moving on this instrument, so
     that a null on the scored arms is a finding about content and not about the
     readout. Below floor -> substrate_not_ready_requeue, never a claim verdict.

DV_HEADROOM DECLARED (kind 'dv_headroom', experiments/_metrics.dv_headroom_check):
  dv = paired location delta vs CTRL_OFF; criterion_threshold = the APPLIED content
  floor (the same `content_floor` C1 reads); achievable = max |delta| over seeds on
  NOISE_H3_REF (statistic 'max_abs': a signed deviation against an absolute floor);
  margin 1.0. Emitted in interpretation.recorded_preconditions -- RECORDED, NOT
  ADJUDICATING, deliberately: P4 already adjudicates the same quantity with the
  STRICTER statistic (the across-seed MEAN of the reference delta must clear the
  floor), so a second gating entry would double-gate one measurement; this entry
  exists so the indexer's headroom channel sees the standard-shape declaration.

EVIDENCE DIRECTION, pre-registered both ways.
  supports    a scored arm clears -- mode conditioning differentiates proposal
              CONTENT at production settings, with breadth held matched. This is
              the CONFIRMING branch of the claim's what_would_answer.
  weakens     no scored arm clears, P1-P4 all met. A well-powered null on content
              with a demonstrably movable DV NARROWS MECH-267 from a
              content-selection mechanism to a DISPERSION mechanism: mode
              conditioning would then move only how BROADLY the proposer samples,
              never WHERE -- which is decision-relevant to the pending
              mode_partitioned_cem production default flip, since flipping it
              would buy a breadth knob rather than the mode-conditioned proposal
              content MECH-267 registers.
  non_contributory  any precondition unmet (instrument defect / DV cannot move).

DV-SYMMETRY INVARIANCE (mandatory per-arm declaration; DV = per-dim-standardized
cross-mode centroid separation, a LOCATION statistic).
  CTRL_OFF: no manipulation reaches the substrate at all (conditioning disabled) --
      the control, and the empirical null, not an invariance artefact.
  CTRL_UNIFORM: the manipulation is the identity by construction (uniform maps), so
      it IS invariant under every DV. That is precisely its job as a leak check; it
      is scoped out of the content criteria and its predicted value is exact zero.
  H2_ONLY: subtracts a PER-TRAJECTORY term (w . mean_z_world -- each candidate's own
      mean world state), NOT a broadcast constant across candidates, so it can move
      the argsort elite selection, hence the elite set, hence the refit ao_mean,
      hence the centroid. Not invariant under a location DV, and not invariant under
      candidate permutation.
  HORIZON_ONLY: truncates the scoring window per mode, changing which candidates
      rank as elite. It is a re-ranking, not a monotone rescaling of one common
      score, so it is not invariant under the rank-preserving symmetries; and it
      shifts the refit MEAN, so not invariant under the location DV.
  NOISE_H3_REF: multiplicatively rescales ao_std. A pure scale change is invariant
      under the per-dim STANDARDIZED statistic in expectation -- but NOT through the
      elite-refit path, which is exactly the confound documented above and the
      reason this arm is a reference control rather than a scored content arm.

BLOCKER CHECK (Step 2.5 / 2.5a): none. Every facet is IMPLEMENTED
(mode_noise_scale + mode_horizon_scale 2026-04-20 / 2026-08-02;
SD-MECH267-CEM-SELECTION-FIX H2 + H3 2026-08-14) and all five arms were confirmed
reachable at runtime by an authoring-time probe before this script was written:
mean_by_action_dim populated (len 4) on every cell; HORIZON_ONLY effective horizons
[4, 2, 3, 4] across the four modes; CTRL_UNIFORM bit-identical to CTRL_OFF;
_last_mode_value_weight_active True on H2_ONLY; _last_mode_partitioned_cem True on
NOISE_H3_REF.

RE-DERIVE BRAKE (Step 2.5b): MECH-267's counted-autopsy total is 4 (869, 869a, 923,
928), above the threshold of 2. The brake is RELEASED on two independent grounds.
(a) This is not a lettered iteration of a braked design: it is a new EXQ NUMBER
measuring a DIFFERENT DV (proposal LOCATION) on a different design axis
(measurement / readout), where every braked run measured proposal BREADTH -- the
skill's stated non-braked case of "a commitment-free read of the same claim" on a
redesign. (b) The upstream substrate the braked lineage named,
SD-MECH267-CEM-SELECTION-FIX, is IMPLEMENTED (2026-08-14), and the most recent
adjudication (failure_autopsy_927-928-mech267-cluster_2026-08-16) states the owed
retest is gated on a default flip rather than on a further build -- a decision this
run's result directly informs.

GOV-REUSE-1 (Step 2.4): the decisive readout is the per-mode CENTROID
(`mean_by_action_dim`) at iters=3. Checked every recorded manifest in the lineage --
869, 869a, 923, 927, 928 -- and NONE records it: each banks only
`mean_raw_std_by_dim` (a scalar mean over dims of the STD), plus entropy and
unique-class counts. The mean vector, and the per-dim std needed to standardize it,
are absent, so the statistic is neither recorded nor derivable post-hoc. Not
recoverable -> run.

claim_ids: ['MECH-267'] -- experiment_purpose=evidence: this tests the claim's
registered CONTENT assertion directly under a pre-registered discriminative gate,
rather than discriminating among loci for an established failure.

RED-TEAM (Step 4.5): BLOCKING (fable, 2026-09-07) -- finding 1 CONFIRMED by two probes;
NOT QUEUED. Verdict + dispositions in the refusal record named at the top of this file.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_1005_mech267_mode_content_location_separation.py [--dry-run]

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""
from __future__ import annotations

import argparse
import itertools
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.hippocampal.module import HippocampalModule  # noqa: E402
from ree_core.predictors.e2_fast import E2FastPredictor  # noqa: E402
from ree_core.residue.field import ResidueField  # noqa: E402
from ree_core.utils.config import (  # noqa: E402
    E2Config,
    HippocampalConfig,
    ResidueConfig,
)
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._metrics import dv_headroom_check  # noqa: E402

EVIDENCE_ROOT = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

EXPERIMENT_PURPOSE = "evidence"

RELATED_EXQ = ["V3-EXQ-869", "V3-EXQ-869a", "V3-EXQ-923", "V3-EXQ-928"]

SEEDS: List[int] = list(range(30))  # identical seed set to 869/869a/923/928

# Substrate dims -- identical to V3-EXQ-928 so the two runs are directly
# comparable (that run's breadth DV is reported here as a secondary readout).
WORLD_DIM = 32
SELF_DIM = 16
ACTION_DIM = 4
ACTION_OBJECT_DIM = 16
NUM_CANDIDATES = 16
HORIZON = 4
NUM_CEM_ITERATIONS = 3  # the PRODUCTION setting -- the regime that washes out

MODES: List[str] = [
    "internal_planning",
    "external_task",
    "internal_replay",
    "offline_consolidation",
]
MODE_PAIRS: List[Tuple[str, str]] = list(itertools.combinations(MODES, 2))  # 6 pairs

# Uniform maps: pinning these is what shuts the breadth channel by construction.
UNIFORM_NOISE_SCALE: Dict[str, float] = {m: 1.0 for m in MODES}
UNIFORM_HORIZON_SCALE: Dict[str, float] = {m: 1.0 for m in MODES}

# --- Pre-registered thresholds (constants; never derived from this run) -------
CONTENT_FLOOR_ABS = 0.02   # absolute floor on the paired location delta
CONTENT_FLOOR_REL = 0.15   # ...or 15% of the measured sampling-noise null, whichever is larger
POWER_T_MIN = 3.0          # paired t (n=30 -> dz >= 0.55): the SD-scaled half of the gate
CONSISTENCY_MIN_FRAC = 0.70  # >= 21/30 seeds positive
BREADTH_TOL = 0.20         # C4: scored arm's cross-mode breadth spread within 20% of control
NULL_LEAK_TOL = 1e-9       # P1: CTRL_UNIFORM must reproduce CTRL_OFF
DV_HEADROOM_MARGIN = 1.0   # recorded dv_headroom: bare feasibility (achievable >= floor x 1.0)

# Distinct per-mode value-weight vectors for the H2 arm (world_dim length), built
# with a LOCAL generator so the global RNG stream is untouched. Same construction
# and same seed (2672) as V3-EXQ-928, so H2 is the identical manipulation that run
# measured on the breadth DV -- only the readout differs.
_MODE_VALUE_WEIGHT_SCALE = 1.0
_mvw_gen = torch.Generator().manual_seed(267_2)
_MODE_VALUE_WEIGHT_TENSOR = (
    torch.randn(len(MODES), WORLD_DIM, generator=_mvw_gen) * _MODE_VALUE_WEIGHT_SCALE
)
MODE_VALUE_WEIGHT: Dict[str, List[float]] = {
    mode: _MODE_VALUE_WEIGHT_TENSOR[i].tolist() for i, mode in enumerate(MODES)
}

# Arm definitions. `noise`/`horizon` None means "leave at the HippocampalConfig
# live default"; a dict pins the map explicitly.
ARMS: Dict[str, Dict[str, Any]] = {
    "CTRL_OFF": {
        "mode_conditioning_enabled": False,
        "noise": None, "horizon": None,
        "mode_value_weight": {}, "mode_partitioned_cem": False,
    },
    "CTRL_UNIFORM": {
        "mode_conditioning_enabled": True,
        "noise": UNIFORM_NOISE_SCALE, "horizon": UNIFORM_HORIZON_SCALE,
        "mode_value_weight": {}, "mode_partitioned_cem": False,
    },
    "H2_ONLY": {
        "mode_conditioning_enabled": True,
        "noise": UNIFORM_NOISE_SCALE, "horizon": UNIFORM_HORIZON_SCALE,
        "mode_value_weight": MODE_VALUE_WEIGHT, "mode_partitioned_cem": False,
    },
    "HORIZON_ONLY": {
        "mode_conditioning_enabled": True,
        "noise": UNIFORM_NOISE_SCALE, "horizon": None,
        "mode_value_weight": {}, "mode_partitioned_cem": False,
    },
    "NOISE_H3_REF": {
        "mode_conditioning_enabled": True,
        "noise": None, "horizon": UNIFORM_HORIZON_SCALE,
        "mode_value_weight": {}, "mode_partitioned_cem": True,
    },
}
ARM_ORDER: List[str] = [
    "CTRL_OFF", "CTRL_UNIFORM", "H2_ONLY", "HORIZON_ONLY", "NOISE_H3_REF",
]
BASELINE_ARM = "CTRL_OFF"
LEAK_ARM = "CTRL_UNIFORM"
SCORED_ARMS: List[str] = ["H2_ONLY", "HORIZON_ONLY"]   # the content arms
REFERENCE_ARM = "NOISE_H3_REF"                          # DV-can-move positive control

# Expected per-arm engagement (the P2 manipulation check).
_EXPECTED: Dict[str, Dict[str, Any]] = {
    "CTRL_OFF":     {"mvw": False, "part": False, "noise_scale": None, "multi_horizon": False},
    "CTRL_UNIFORM": {"mvw": False, "part": False, "noise_scale": 1.0,  "multi_horizon": False},
    "H2_ONLY":      {"mvw": True,  "part": False, "noise_scale": 1.0,  "multi_horizon": False},
    "HORIZON_ONLY": {"mvw": False, "part": False, "noise_scale": 1.0,  "multi_horizon": True},
    "NOISE_H3_REF": {"mvw": False, "part": True,  "noise_scale": None, "multi_horizon": False},
}


def _make_hippocampal(arm: str) -> HippocampalModule:
    e2 = E2FastPredictor(
        E2Config(
            self_dim=SELF_DIM, world_dim=WORLD_DIM, action_dim=ACTION_DIM,
            action_object_dim=ACTION_OBJECT_DIM, hidden_dim=64,
        )
    )
    res = ResidueField(
        ResidueConfig(world_dim=WORLD_DIM, hidden_dim=32, num_basis_functions=8)
    )
    spec = ARMS[arm]
    kwargs: Dict[str, Any] = dict(
        world_dim=WORLD_DIM, action_dim=ACTION_DIM,
        action_object_dim=ACTION_OBJECT_DIM, hidden_dim=32,
        horizon=HORIZON, num_candidates=NUM_CANDIDATES,
        num_cem_iterations=NUM_CEM_ITERATIONS,
        mode_conditioning_enabled=spec["mode_conditioning_enabled"],
        mode_value_weight=spec["mode_value_weight"],
        mode_partitioned_cem=spec["mode_partitioned_cem"],
    )
    if spec["noise"] is not None:
        kwargs["mode_noise_scale"] = spec["noise"]
    if spec["horizon"] is not None:
        kwargs["mode_horizon_scale"] = spec["horizon"]
    return HippocampalModule(HippocampalConfig(**kwargs), e2, res)


_MODE_OFFSET = {m: i * 7_919 for i, m in enumerate(MODES)}


def _cell_sampling_seed(seed: int, arm: str, mode: str) -> int:
    """Reproducible per-(seed, mode) CEM sampling seed -- SHARED ACROSS ARMS.

    COMMON RANDOM NUMBERS, and this is a deliberate departure from
    V3-EXQ-869/923/928, which added an arm offset to keep the arms' CEM draws
    disjoint. Two reasons the matched-draw design is the right one here:

    (1) It is what makes the paired contrast tight. With independent draws per
        arm, every arm-minus-control delta carries two independent samples'
        worth of noise, which is a pure power cost -- and a properly-powered
        paired contrast is precisely the repair MECH-267's what_would_answer
        asks for after V3-EXQ-928's gate was found under-powered.
    (2) It is what makes the P1 leak check SATISFIABLE. CTRL_UNIFORM's predicted
        value is EXACT zero (an inert manipulation), which is only a meaningful
        prediction when it faces the identical draw sequence as CTRL_OFF. Under
        decorrelated draws the two arms differ by sampling noise no matter what
        the substrate does, so a zero-tolerance leak check could never pass --
        an unsatisfiable pre-registered precondition. The fix is to scope the
        comparison correctly (identical draws), never to loosen the tolerance.

    The mode offset is retained at the 869/923/928 value (7919) so per-mode
    sampling stays comparable with the lineage. Every arm re-seeds immediately
    before its propose_trajectories call, so all arms consume the SAME
    underlying standard-normal sequence and differ only by how the config
    transforms it.
    """
    return seed * 104_729 + _MODE_OFFSET[mode]


def _finite_vec(v: Any) -> bool:
    return (
        isinstance(v, (list, tuple))
        and len(v) > 0
        and all(x is not None and math.isfinite(float(x)) for x in v)
    )


def _location_separation(
    mu_a: List[float], sd_a: List[float], mu_b: List[float], sd_b: List[float]
) -> Optional[float]:
    """Per-dimension-standardized RMS centroid separation between two modes.

    Dimensionless and LOCATION-only: identical centroids give ~the sampling
    floor regardless of how the two spreads differ.
    """
    if not (len(mu_a) == len(sd_a) == len(mu_b) == len(sd_b)) or not mu_a:
        return None
    acc: List[float] = []
    for i in range(len(mu_a)):
        pooled_var = 0.5 * (float(sd_a[i]) ** 2 + float(sd_b[i]) ** 2)
        if not math.isfinite(pooled_var) or pooled_var <= 0.0:
            return None
        acc.append((float(mu_a[i]) - float(mu_b[i])) ** 2 / pooled_var)
    return math.sqrt(sum(acc) / len(acc))


def _ranking_authority(
    hip_with: HippocampalModule,
    hip_without: HippocampalModule,
    trajectories: List[Any],
    mode: str,
) -> Dict[str, Any]:
    """How much authority does the arm's scoring manipulation have over elite selection?

    RECORDED, NON-GATING. This exists to make a NULL attributable. P2 only
    establishes that a facet engaged (a boolean). If the scored arm returns no
    content shift, the obvious rival explanation is "the knob was set too weak
    to matter" -- and nothing in a boolean engagement check rules that out.
    This measures it directly: the across-candidate spread of the manipulation's
    score contribution against the across-candidate spread of the underlying
    terrain score, whether the argsort ranking changed at all, and how much of
    the top-k elite set was swapped. An authority ratio >> 1 with a large elite
    swap and no centroid movement is a finding about the MECHANISM (re-ranking a
    mode-blind candidate pool cannot relocate the pool's centroid), not about an
    under-powered knob.

    Authoring-time probe (6 seeds x 4 modes, H2): ratio mean 215.6 (min 18.1),
    ranking changed on 24/24 cells, elite overlap 0.29.
    """
    if not trajectories:
        return {"measurable": False}
    try:
        with torch.no_grad():
            s_with = [float(hip_with._score_trajectory(t, operating_mode={mode: 1.0}))
                      for t in trajectories]
            s_without = [float(hip_without._score_trajectory(t)) for t in trajectories]
    except Exception:
        return {"measurable": False}
    if not s_with or len(s_with) != len(s_without):
        return {"measurable": False}
    contrib = [s_with[i] - s_without[i] for i in range(len(s_with))]
    terrain_spread = max(s_without) - min(s_without)
    mode_spread = max(contrib) - min(contrib)
    rank_with = sorted(range(len(s_with)), key=lambda i: s_with[i])
    rank_without = sorted(range(len(s_without)), key=lambda i: s_without[i])
    n_elite = max(1, int(len(trajectories) * 0.2))
    overlap = len(set(rank_with[:n_elite]) & set(rank_without[:n_elite])) / n_elite
    return {
        "measurable": True,
        "terrain_score_spread_across_candidates": terrain_spread,
        "mode_term_spread_across_candidates": mode_spread,
        "authority_ratio": (
            (mode_spread / terrain_spread) if terrain_spread > 0 else None
        ),
        "ranking_changed": bool(rank_with != rank_without),
        "elite_set_overlap_frac": overlap,
        "n_elite": n_elite,
        "n_candidates": len(trajectories),
    }


def _run_seed(seed: int, arms: List[str]) -> Dict[str, Any]:
    """One seed: shared terrain weights and (z_world, z_self) across all arms."""
    cells: Dict[str, Dict[str, Any]] = {}
    # Companion module with the H2 term OFF but identical weights (same seed),
    # used only to score the SAME trajectory objects both ways for the recorded
    # ranking-authority diagnostic below. Never proposes anything itself.
    torch.manual_seed(seed)
    hip_ref = _make_hippocampal(BASELINE_ARM)
    for arm in arms:
        # Terrain-prior weights are a pure function of seed -- the facets under
        # test add no network parameters, so every arm sees the identical network.
        torch.manual_seed(seed)
        hip = _make_hippocampal(arm)

        # Decorrelated but reproducible latent draw, identical across arms.
        torch.manual_seed(seed + 900_000)
        z_world = torch.randn(1, WORLD_DIM)
        z_self = torch.randn(1, SELF_DIM)

        for mode in MODES:
            torch.manual_seed(_cell_sampling_seed(seed, arm, mode))
            trajectories = hip.propose_trajectories(
                z_world, z_self,
                num_candidates=NUM_CANDIDATES,
                operating_mode={mode: 1.0},
            )
            diag = hip.get_last_propose_diagnostics()
            authority = (
                _ranking_authority(hip, hip_ref, trajectories, mode)
                if arm == "H2_ONLY" else {"measurable": False, "not_applicable": True}
            )
            stats = diag.get("action_object_decoder_raw_output_stats", {}) or {}
            raw_mean = stats.get("mean_by_action_dim", [])
            raw_std = stats.get("std_by_action_dim", [])
            populated = _finite_vec(raw_mean) and _finite_vec(raw_std)
            cells[f"{arm}::{mode}"] = {
                "arm": arm,
                "mode": mode,
                "num_cem_iterations": NUM_CEM_ITERATIONS,
                "mean_by_action_dim": [float(v) for v in raw_mean] if populated else [],
                "std_by_action_dim": [float(v) for v in raw_std] if populated else [],
                # Secondary readout: the V3-EXQ-869/923/928 breadth DV, kept so
                # this run is directly comparable to the lineage it extends.
                "mean_raw_std_by_dim": (
                    float(statistics.fmean(raw_std)) if populated else 0.0
                ),
                "readout_populated": bool(populated),
                "mode_value_weight_active": bool(hip._last_mode_value_weight_active),
                "mode_partitioned_cem": bool(hip._last_mode_partitioned_cem),
                "mode_noise_scale_used": (
                    None if hip._last_mode_noise_scale is None
                    else float(hip._last_mode_noise_scale)
                ),
                "effective_horizon": (
                    None if hip._last_effective_horizon is None
                    else int(hip._last_effective_horizon)
                ),
                "ranking_authority": authority,
            }
    return {"seed": seed, "cells": cells}


def _arm_dbar(cells: Dict[str, Dict[str, Any]], arm: str) -> Optional[float]:
    """Six-pair mean standardized centroid separation for one arm, one seed."""
    vals: List[float] = []
    for a, b in MODE_PAIRS:
        ca, cb = cells.get(f"{arm}::{a}"), cells.get(f"{arm}::{b}")
        if not ca or not cb or not ca["readout_populated"] or not cb["readout_populated"]:
            return None
        d = _location_separation(
            ca["mean_by_action_dim"], ca["std_by_action_dim"],
            cb["mean_by_action_dim"], cb["std_by_action_dim"],
        )
        if d is None:
            return None
        vals.append(d)
    return statistics.fmean(vals) if vals else None


def _arm_pairwise(cells: Dict[str, Dict[str, Any]], arm: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for a, b in MODE_PAIRS:
        ca, cb = cells.get(f"{arm}::{a}"), cells.get(f"{arm}::{b}")
        if not ca or not cb or not ca["readout_populated"] or not cb["readout_populated"]:
            out[f"{a}|{b}"] = None
        else:
            out[f"{a}|{b}"] = _location_separation(
                ca["mean_by_action_dim"], ca["std_by_action_dim"],
                cb["mean_by_action_dim"], cb["std_by_action_dim"],
            )
    return out


def _arm_breadth_spread(cells: Dict[str, Dict[str, Any]], arm: str) -> Optional[float]:
    """max-minus-min over modes of the per-mode mean std -- the C4 confound guard."""
    vals: List[float] = []
    for mode in MODES:
        c = cells.get(f"{arm}::{mode}")
        if not c or not c["readout_populated"]:
            return None
        vals.append(float(c["mean_raw_std_by_dim"]))
    return max(vals) - min(vals)


def _paired_t(deltas: List[float]) -> Optional[float]:
    if len(deltas) < 2:
        return None
    sd = statistics.stdev(deltas)
    if sd <= 0.0:
        return None
    return statistics.fmean(deltas) / (sd / math.sqrt(len(deltas)))


def _cell_engaged(cell: Dict[str, Any], seed_cells: Dict[str, Dict[str, Any]]) -> bool:
    exp = _EXPECTED[cell["arm"]]
    if bool(cell["mode_value_weight_active"]) != exp["mvw"]:
        return False
    if bool(cell["mode_partitioned_cem"]) != exp["part"]:
        return False
    got_scale = cell["mode_noise_scale_used"]
    if exp["noise_scale"] is None:
        # CTRL_OFF: conditioning disabled, so no scale at all. NOISE_H3_REF: the
        # live default map, which must be a real per-mode value (not 1.0-uniform).
        if cell["arm"] == "CTRL_OFF" and got_scale is not None:
            return False
        if cell["arm"] != "CTRL_OFF" and got_scale is None:
            return False
    else:
        if got_scale is None or abs(float(got_scale) - float(exp["noise_scale"])) > 1e-9:
            return False
    if exp["multi_horizon"]:
        horizons = {
            seed_cells[f"{cell['arm']}::{m}"]["effective_horizon"] for m in MODES
        }
        if len(horizons) < 2:
            return False
    return True


def main(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = SEEDS[:3] if dry_run else SEEDS
    arms = ARM_ORDER  # every arm runs even in the smoke: the controls are the point
    print(
        f"[v3_exq_1005] MECH-267 mode-content LOCATION separation "
        f"({len(arms)} arms x {len(MODES)} modes x {len(seeds)} seed(s), "
        f"num_cem_iterations={NUM_CEM_ITERATIONS}) "
        f"({'dry-run' if dry_run else 'full'})...",
        flush=True,
    )

    per_seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        r = _run_seed(seed, arms)
        cells = r["cells"]
        r["arm_dbar"] = {a: _arm_dbar(cells, a) for a in arms}
        r["arm_pairwise"] = {a: _arm_pairwise(cells, a) for a in arms}
        r["arm_breadth_spread"] = {a: _arm_breadth_spread(cells, a) for a in arms}
        base = r["arm_dbar"].get(BASELINE_ARM)
        r["arm_delta_vs_control"] = {
            a: (None if (base is None or r["arm_dbar"].get(a) is None)
                else r["arm_dbar"][a] - base)
            for a in arms
        }
        per_seed_results.append(r)

        # Progress instrumentation: one boundary + verdict per seed x condition
        # unit, so the runner's total_runs (seeds x conditions) matches exactly.
        for arm in arms:
            print(f"Seed {seed} Condition {arm}", flush=True)
            d = r["arm_dbar"].get(arm)
            delta = r["arm_delta_vs_control"].get(arm)
            print(
                f"  [probe] seed={seed} arm={arm} ep 1/1 dbar="
                f"{'na' if d is None else round(d, 6)} delta_vs_control="
                f"{'na' if delta is None else round(delta, 6)}",
                flush=True,
            )
            seed_ok = (
                arm == BASELINE_ARM
                or (delta is not None and delta > 0.0)
            )
            print(f"verdict: {'PASS' if seed_ok else 'FAIL'}", flush=True)

    # ---- P2 / P3: manipulation engagement + readout integrity -----------------
    all_cells = [(r, c) for r in per_seed_results for c in r["cells"].values()]
    n_cells = len(all_cells)
    n_engaged = sum(1 for r, c in all_cells if _cell_engaged(c, r["cells"]))
    n_populated = sum(1 for _, c in all_cells if c["readout_populated"])
    manipulation_ok = n_cells > 0 and n_engaged == n_cells
    readout_ok = n_cells > 0 and n_populated == n_cells

    # ---- Aggregate per arm ----------------------------------------------------
    def _deltas(arm: str) -> List[float]:
        return [
            r["arm_delta_vs_control"][arm]
            for r in per_seed_results
            if r["arm_delta_vs_control"].get(arm) is not None
        ]

    control_dbars = [
        r["arm_dbar"][BASELINE_ARM] for r in per_seed_results
        if r["arm_dbar"].get(BASELINE_ARM) is not None
    ]
    control_dbar_mean = statistics.fmean(control_dbars) if control_dbars else 0.0
    control_breadths = [
        r["arm_breadth_spread"][BASELINE_ARM] for r in per_seed_results
        if r["arm_breadth_spread"].get(BASELINE_ARM) is not None
    ]
    control_breadth_mean = statistics.fmean(control_breadths) if control_breadths else 0.0

    # The pre-registered floor: absolute OR referenced to the measured null.
    content_floor = max(CONTENT_FLOOR_ABS, CONTENT_FLOOR_REL * control_dbar_mean)
    consistency_min = math.ceil(CONSISTENCY_MIN_FRAC * len(seeds))
    breadth_ceiling = control_breadth_mean * (1.0 + BREADTH_TOL)

    per_arm: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        deltas = _deltas(arm)
        breadths = [
            r["arm_breadth_spread"][arm] for r in per_seed_results
            if r["arm_breadth_spread"].get(arm) is not None
        ]
        mean_delta = statistics.fmean(deltas) if deltas else None
        t_stat = _paired_t(deltas)
        n_pos = sum(1 for d in deltas if d > 0.0)
        mean_breadth = statistics.fmean(breadths) if breadths else None
        c1 = mean_delta is not None and mean_delta >= content_floor
        c2 = t_stat is not None and t_stat >= POWER_T_MIN
        c3 = n_pos >= consistency_min
        c4 = mean_breadth is not None and mean_breadth <= breadth_ceiling
        per_arm[arm] = {
            "mean_delta_vs_control": mean_delta,
            "sd_delta": statistics.stdev(deltas) if len(deltas) > 1 else None,
            "paired_t": t_stat,
            "n_seeds_positive": n_pos,
            "n_seeds": len(deltas),
            "mean_dbar": (
                statistics.fmean([
                    r["arm_dbar"][arm] for r in per_seed_results
                    if r["arm_dbar"].get(arm) is not None
                ]) if deltas else None
            ),
            "mean_breadth_spread": mean_breadth,
            "C1_floor_met": bool(c1),
            "C2_power_met": bool(c2),
            "C3_consistency_met": bool(c3),
            "C4_breadth_invariance_met": bool(c4),
            "clears_content_gate": bool(c1 and c2 and c3 and c4),
            "role": (
                "baseline" if arm == BASELINE_ARM
                else "leak_check" if arm == LEAK_ARM
                else "reference_dv_can_move" if arm == REFERENCE_ARM
                else "scored_content_arm"
            ),
        }

    # ---- P1 / P4 --------------------------------------------------------------
    leak_deltas = _deltas(LEAK_ARM)
    leak_abs_mean = statistics.fmean([abs(d) for d in leak_deltas]) if leak_deltas else None
    p1_met = leak_abs_mean is not None and leak_abs_mean <= NULL_LEAK_TOL

    ref_mean_delta = per_arm[REFERENCE_ARM]["mean_delta_vs_control"]
    p4_met = ref_mean_delta is not None and ref_mean_delta >= content_floor

    preconditions = [
        {
            "name": "instrument_null_calibrated",
            "description": (
                "CTRL_UNIFORM (mode conditioning ON, all per-mode maps uniform) must "
                "reproduce CTRL_OFF -- switching conditioning on with no per-mode "
                "information must not manufacture centroid separation"
            ),
            "control": "uniform-map arm; predicted exactly zero by construction",
            "measured": leak_abs_mean,
            "threshold": NULL_LEAK_TOL,
            "direction": "upper",
            "met": bool(p1_met),
        },
        {
            "name": "manipulation_engaged_all_cells",
            "description": (
                "every cell's engagement diagnostics match its arm spec "
                "(mode_value_weight_active / mode_partitioned_cem / noise scale, and "
                "HORIZON_ONLY takes >=2 distinct effective horizons across modes)"
            ),
            "control": "per-cell substrate diagnostics vs the declared arm table",
            "measured": float(n_engaged),
            "threshold": float(n_cells),
            "direction": "lower",
            "met": bool(manipulation_ok),
        },
        {
            "name": "location_readout_populated",
            "description": (
                "mean_by_action_dim and std_by_action_dim are non-empty finite "
                "vectors with positive pooled variance on every cell"
            ),
            "control": "per-cell decoder raw-output stats",
            "measured": float(n_populated),
            "threshold": float(n_cells),
            "direction": "lower",
            "met": bool(readout_ok),
        },
        {
            "name": "location_dv_can_move",
            "description": (
                "POSITIVE CONTROL: the NOISE_H3_REF reference arm's paired location "
                "delta clears the same pre-registered floor the scored arms are held "
                "to, proving this DV is capable of moving on this instrument -- "
                "without it a null on the scored arms is unattributable"
            ),
            "control": "mode_noise_scale defaults + mode_partitioned_cem (known to move breadth)",
            "measured": ref_mean_delta,
            "threshold": content_floor,
            "direction": "lower",
            "met": bool(p4_met),
        },
    ]
    all_preconditions_met = bool(p1_met and manipulation_ok and readout_ok and p4_met)

    # ---- dv_headroom (RECORDED, not adjudicating) -----------------------------
    # Standard-shape headroom declaration on the reference arm's per-seed paired
    # deltas. P4 above is the adjudicating twin (mean >= floor, stricter); this
    # entry records max |delta| against the SAME applied floor so the indexer's
    # headroom channel sees the declaration without double-gating.
    ref_headroom_values = _deltas(REFERENCE_ARM) or [float("nan")]
    dv_headroom_entry = dv_headroom_check(
        "dv_headroom_paired_location_delta",
        dv_name="paired_location_delta_vs_control",
        criterion_threshold=content_floor,
        control_values=ref_headroom_values,
        statistic="max_abs",
        margin=DV_HEADROOM_MARGIN,
        control=(
            f"{REFERENCE_ARM} (mode_noise_scale defaults + mode_partitioned_cem): the "
            "DV-can-move positive control; achievable = max |paired delta vs "
            f"{BASELINE_ARM}| over seeds, against the applied content floor "
            "max(CONTENT_FLOOR_ABS, CONTENT_FLOOR_REL x control dbar) that C1 reads"
        ),
        gating=False,
        recorded_only=True,
        adjudicating_twin="location_dv_can_move",
        why_not_adjudicating=(
            "P4 location_dv_can_move adjudicates the same quantity with the "
            "stricter across-seed MEAN and routes substrate_not_ready_requeue on "
            "failure; this entry is the standard-shape declaration for the "
            "indexer's headroom channel and must not double-gate one measurement."
        ),
        per_seed={
            f"seed{r['seed']}": r["arm_delta_vs_control"].get(REFERENCE_ARM)
            for r in per_seed_results
        },
    )
    dv_headroom_entry["met"] = bool(
        math.isfinite(dv_headroom_entry["measured"])
        and dv_headroom_entry["measured"] >= dv_headroom_entry["threshold"]
    )
    dv_headroom_entry["comparator"] = ">="
    recorded_preconditions: List[Dict[str, Any]] = [dv_headroom_entry]

    # ---- Outcome routing ------------------------------------------------------
    clearing = [a for a in SCORED_ARMS if per_arm[a]["clears_content_gate"]]
    if not (manipulation_ok and readout_ok):
        outcome = "FAIL"
        label = "measurement_degenerate"
        direction = "non_contributory"
        note = (
            "Manipulation did not engage as configured, or the location readout was "
            "unpopulated, on at least one cell. Nothing about MECH-267 follows."
        )
    elif not p1_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
        note = (
            "INSTRUMENT DEFECT: the uniform-map leak arm did not reproduce the "
            "mode-blind control, so the instrument manufactures separation from the "
            "conditioning switch alone. Not a claim verdict."
        )
    elif not p4_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
        note = (
            "POSITIVE CONTROL FAILED: the reference arm's location delta did not "
            "clear the floor, so this DV is not demonstrably movable on this "
            "instrument and a null on the scored arms is unattributable. Not a "
            "claim verdict."
        )
    elif clearing:
        outcome = "PASS"
        label = "mode_content_location_separation_confirmed::" + "+".join(clearing)
        direction = "supports"
        note = (
            "At production num_cem_iterations=3, mode conditioning differentiates "
            "proposal CONTENT (centroid location) with cross-mode breadth held "
            "matched to the control, on a properly-powered paired contrast. This is "
            "the CONFIRMING branch of MECH-267's what_would_answer."
        )
    else:
        outcome = "FAIL"
        label = "mode_content_location_null_dispersion_only"
        direction = "weakens"
        note = (
            "Well-powered null on proposal CONTENT with every precondition met and a "
            "demonstrably movable DV: at production num_cem_iterations=3 no "
            "breadth-matched mode-conditioning facet shifts WHERE the proposer "
            "samples, only HOW BROADLY. NARROWS MECH-267 from a content-selection "
            "mechanism to a dispersion mechanism. Decision-relevant to the pending "
            "mode_partitioned_cem production default flip, which on this reading "
            "would buy a breadth knob rather than mode-conditioned proposal content."
        )

    criteria = [
        {
            "name": "C_manipulation_and_readout_integrity",
            "load_bearing": True,
            "passed": bool(manipulation_ok and readout_ok),
            "detail": {
                "n_cells": n_cells, "n_engaged": n_engaged, "n_populated": n_populated,
            },
        },
        {
            "name": "C_instrument_null_calibrated",
            "load_bearing": True,
            "passed": bool(p1_met),
            "detail": {"leak_abs_mean": leak_abs_mean, "tol": NULL_LEAK_TOL},
        },
        {
            "name": "C_location_dv_can_move",
            "load_bearing": True,
            "passed": bool(p4_met),
            "detail": {"reference_mean_delta": ref_mean_delta, "floor": content_floor},
        },
        {
            "name": "C_at_least_one_scored_arm_clears_content_gate",
            "load_bearing": True,
            "passed": bool(any(per_arm[a]["clears_content_gate"] for a in SCORED_ARMS)),
            "detail": {
                "scored_arms": SCORED_ARMS,
                "clearing": [a for a in SCORED_ARMS if per_arm[a]["clears_content_gate"]],
                "note": (
                    "AGGREGATE gate for the disjunctive combination rule. The "
                    "per-arm members below are load_bearing:False on purpose -- the "
                    "rule accepts on >=1 arm, so an arm that does NOT clear is the "
                    "informative half of the localisation, not a gate cleared on "
                    "nothing."
                ),
            },
        },
    ] + [
        {
            "name": f"C_{arm}_clears_content_gate",
            "load_bearing": False,
            "passed": bool(per_arm[arm]["clears_content_gate"]),
            "detail": {
                "mean_delta": per_arm[arm]["mean_delta_vs_control"],
                "floor": content_floor,
                "paired_t": per_arm[arm]["paired_t"],
                "t_min": POWER_T_MIN,
                "n_seeds_positive": per_arm[arm]["n_seeds_positive"],
                "consistency_min": consistency_min,
                "mean_breadth_spread": per_arm[arm]["mean_breadth_spread"],
                "breadth_ceiling": breadth_ceiling,
                "C1": per_arm[arm]["C1_floor_met"],
                "C2": per_arm[arm]["C2_power_met"],
                "C3": per_arm[arm]["C3_consistency_met"],
                "C4": per_arm[arm]["C4_breadth_invariance_met"],
            },
        }
        for arm in SCORED_ARMS
    ]

    # Non-degeneracy: the gate is degenerate only if the scored arms' paired
    # deltas carry no variance at all (a structurally pinned readout).
    ref_deltas = _deltas(REFERENCE_ARM)
    ref_spread = (max(ref_deltas) - min(ref_deltas)) if ref_deltas else 0.0
    scored_delta_pool = [d for a in SCORED_ARMS for d in _deltas(a)]
    # Non-degeneracy is a property of the INSTRUMENT, not of the result: the
    # question is whether this measurement COULD have discriminated. It could,
    # iff the DV-can-move positive control both cleared the floor and varied
    # across seeds. A scored arm sitting at exactly zero beside a control that
    # moves is the most informative null available, and must NOT self-report as
    # degenerate (which would exclude it from scoring).
    non_degenerate = bool(ref_deltas) and ref_spread > 0.0 and bool(p4_met)
    criteria_non_degenerate = {
        c["name"]: bool(
            non_degenerate if c["name"].endswith("clears_content_gate") else True
        )
        for c in criteria
    }

    elapsed = time.perf_counter() - t0

    full_config = {
        "arms": arms,
        "arm_definitions": {
            a: {
                "mode_conditioning_enabled": ARMS[a]["mode_conditioning_enabled"],
                "mode_noise_scale": (
                    "live_default" if ARMS[a]["noise"] is None else "uniform_1.0"
                ),
                "mode_horizon_scale": (
                    "live_default" if ARMS[a]["horizon"] is None else "uniform_1.0"
                ),
                "mode_value_weight_set": bool(ARMS[a]["mode_value_weight"]),
                "mode_partitioned_cem": ARMS[a]["mode_partitioned_cem"],
                "role": per_arm[a]["role"],
            }
            for a in arms
        },
        "modes": MODES,
        "mode_pairs": [f"{a}|{b}" for a, b in MODE_PAIRS],
        "scored_arms": SCORED_ARMS,
        "baseline_arm": BASELINE_ARM,
        "leak_arm": LEAK_ARM,
        "reference_arm": REFERENCE_ARM,
        "num_cem_iterations": NUM_CEM_ITERATIONS,
        "num_candidates": NUM_CANDIDATES,
        "horizon": HORIZON,
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "action_dim": ACTION_DIM,
        "action_object_dim": ACTION_OBJECT_DIM,
        "mode_value_weight_scale": _MODE_VALUE_WEIGHT_SCALE,
        "mode_value_weight_seed": 2672,
        "primary_dv": "six_pair_mean_standardized_centroid_separation",
        "secondary_dv": "mean_raw_std_by_dim",
        "content_floor_abs": CONTENT_FLOOR_ABS,
        "content_floor_rel": CONTENT_FLOOR_REL,
        "content_floor_applied": content_floor,
        "power_t_min": POWER_T_MIN,
        "consistency_min_frac": CONSISTENCY_MIN_FRAC,
        "consistency_min_seeds": consistency_min,
        "breadth_tol": BREADTH_TOL,
        "breadth_ceiling_applied": breadth_ceiling,
        "null_leak_tol": NULL_LEAK_TOL,
        "seeds": seeds,
        "related_exq": RELATED_EXQ,
    }

    run_id = (
        "v3_exq_1005_mech267_mode_content_location_separation_"
        + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "_v3"
    )

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "queue_id": "V3-EXQ-1005",
        "experiment_type": "v3_exq_1005_mech267_mode_content_location_separation",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": ["MECH-267"],
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_note": note,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": (
            None if non_degenerate
            else (
                "the DV-can-move positive control did not both clear the floor "
                "and vary across seeds, so this measurement could not have "
                "discriminated content separation either way"
            )
        ),
        "interpretation": {
            "label": label,
            "combination_rule": (
                "PASS iff >=1 SCORED arm (H2_ONLY, HORIZON_ONLY) clears ALL of C1 "
                "(mean paired delta >= max(0.02, 0.15 x control dbar)), C2 (paired "
                "t >= 3.0), C3 (>=70% of seeds positive) and C4 (cross-mode breadth "
                "spread within 20% of control -- the confound guard that licenses "
                "reading the delta as CONTENT rather than as the elite-refit "
                "consequence of a breadth change), AND all four preconditions hold. "
                "The reference arm NOISE_H3_REF is deliberately EXCLUDED from the "
                "content criteria (its manipulation reaches the DV through the "
                "breadth confound) and serves only as the DV-can-move positive "
                "control. Precondition failure routes non_contributory / "
                "substrate_not_ready_requeue, never a claim verdict."
            ),
            "criteria": criteria,
            "criteria_non_degenerate": criteria_non_degenerate,
            "preconditions": preconditions,
            "all_preconditions_met": all_preconditions_met,
            "recorded_preconditions": recorded_preconditions,
            "preconditions_scope_note": (
                "interpretation.preconditions carries the four ADJUDICATING readiness "
                "gates (P1 instrument_null_calibrated, P2 manipulation_engaged_all_cells, "
                "P3 location_readout_populated, P4 location_dv_can_move). "
                "interpretation.recorded_preconditions carries the standard-shape "
                "dv_headroom declaration on the reference arm, RECORDED ONLY: P4 is "
                "its adjudicating twin (stricter mean statistic, same applied floor), "
                "so a gating copy here would double-gate one measurement."
            ),
            "scored_arms_clearing": clearing,
        },
        "manipulation_check": {
            "all_cells_engaged_as_configured": bool(manipulation_ok),
            "n_engaged": n_engaged,
            "n_cells": n_cells,
            "n_readout_populated": n_populated,
            "expected_engagement_by_arm": _EXPECTED,
        },
        "per_arm_results": per_arm,
        "control_dbar_mean": control_dbar_mean,
        "control_breadth_spread_mean": control_breadth_mean,
        "content_floor_applied": content_floor,
        "per_seed_results": per_seed_results,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "elapsed_sec": elapsed,
        "dry_run": bool(dry_run),
        "custom_information": {
            "why_location_not_breadth": (
                "V3-EXQ-869/869a/923/927/928 all scored mean_raw_std_by_dim, a "
                "DISPERSION statistic. MECH-267's registered assertion is about "
                "proposal CONTENT. mean_by_action_dim was present in the same "
                "diagnostic dict throughout and never read."
            ),
            "breadth_to_location_confound": (
                "A wider CEM sample feeding top-k elite refit pulls the refit "
                "ao_mean away from the terrain prior, so a pure breadth "
                "manipulation moves the centroid with no content selection. "
                "Authoring probe (8 seeds): the noise+H3 arm moved the location DV "
                "by +0.024 to +0.171 while doubling cross-mode breadth spread "
                "(0.0124 -> 0.0267). Hence C4 and the uniform-noise pinning on "
                "every scored arm."
            ),
            "exq_928_pair_degeneracy": (
                "At horizon=4 the default mode_horizon_scale map gives effective "
                "horizons internal_planning=4, external_task=2, internal_replay=3, "
                "offline_consolidation=4. V3-EXQ-928 scored ONLY the "
                "internal_planning vs offline_consolidation pair, i.e. 4 vs 4 -- "
                "the horizon facet was inert on the one pair it measured. This run "
                "averages over all six pairs."
            ),
        },
    }

    out_dir = EVIDENCE_ROOT / "v3_exq_1005_mech267_mode_content_location_separation"
    out_file = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=dry_run,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
    )

    print(
        f"[v3_exq_1005] outcome={outcome} label={label} direction={direction} "
        f"content_floor={content_floor:.5f} control_dbar={control_dbar_mean:.5f}",
        flush=True,
    )
    for arm in arms:
        pa = per_arm[arm]
        print(
            f"  arm={arm:13s} role={pa['role']:22s} dbar="
            f"{'na' if pa['mean_dbar'] is None else round(pa['mean_dbar'], 5)} "
            f"delta={'na' if pa['mean_delta_vs_control'] is None else round(pa['mean_delta_vs_control'], 5)} "
            f"t={'na' if pa['paired_t'] is None else round(pa['paired_t'], 2)} "
            f"pos={pa['n_seeds_positive']}/{pa['n_seeds']} "
            f"breadth={'na' if pa['mean_breadth_spread'] is None else round(pa['mean_breadth_spread'], 5)} "
            f"clears={pa['clears_content_gate']}",
            flush=True,
        )
    print(
        f"  dv_headroom(recorded): measured={dv_headroom_entry['measured']:.5f} "
        f"threshold={dv_headroom_entry['threshold']:.5f} met={dv_headroom_entry['met']} "
        f"n_adjudicating={len(preconditions)} n_recorded={len(recorded_preconditions)}",
        flush=True,
    )
    if not dry_run:
        print(f"Result written to: {out_file}", flush=True)

    return {
        "outcome": outcome,
        "manifest_path": out_file,
        "run_id": run_id,
        "dry_run": dry_run,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="3 seeds, all 5 arms; relocates the smoke manifest, no evidence/ write.",
    )
    args = parser.parse_args()
    _result = main(dry_run=args.dry_run)
    emit_outcome(
        outcome=_result["outcome"],
        manifest_path=_result["manifest_path"],
        run_id=_result["run_id"],
        dry_run=_result["dry_run"],
    )
    sys.exit(0 if _result["outcome"] == "PASS" else 1)
