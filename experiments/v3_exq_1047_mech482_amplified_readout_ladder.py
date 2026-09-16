"""V3-EXQ-1047 -- MECH-482 / SD-102: is the epistemic-deficit readout MAGNITUDE-limited or
PATTERN-limited?

V3-EXQ-964b settled reachability and instrument validity and left exactly one question open:
the real readout never moves the committed action AT ITS OWN MAGNITUDE, while a synthetic
deficit at the SAME shipped selection authority moves it on 3/3 seeds. Two live hypotheses
remain on the registered question `mech482_deficit_selection_authority`:

  H-mag  the readout's CONTENT is fine and only its MAGNITUDE is too small. Amplify it until
         its realised post-clamp range reaches the same rail the synthetic reaches, and it
         fires like the synthetic does.
  H-pat  the readout's PATTERN is selection-irrelevant. At the rail it still does not fire,
         while a range-matched synthetic and a permutation of the readout's OWN values do.

  MANIPULATION: a GAIN k in {1, 10, 40, 100} on the SUBJECT'S OWN readout, at the SHIPPED
                `curiosity_bias_scale = 0.1`. The authority knob is NOT lifted on the subject
                -- that was V3-EXQ-964b's control and is retained here only as such.
  HELD FIXED:   the readiness knobs, the env seed, the yoking discipline, the corrected
                reachability instrument, and every threshold V3-EXQ-964b pre-registered. This
                file patches exactly two names on that module and reuses its yoked group.

EXPERIMENT_PURPOSE = "diagnostic" -- it discriminates WHY the mechanism is inert and routes a
build-vs-config decision. Excluded from governance confidence/conflict scoring. `claim_ids`
MECH-482; bears_on `orienting_epistemic_deficit_v3_plan:ORNT-2`.

SLEEP: none. No sleep flag is set anywhere in this driver. sleep_driver_pattern="none".

red-team: see the RED-TEAM RECORD at the end of this docstring and the queue entry note.

Authority: CONFIRMED autopsy `failure_autopsy_V3-EXQ-964b_2026-09-16` (+ .json), user gate
2026-09-16T11:10:54Z (all three Step 8 decisions approved, including the explicit re-derive
brake PRODUCER RELEASE), applied by governance-20260916 (REE_assembly db6d20ebee:
substrate_queue `sd_epistemic_deficit_multitarget_readiness` amended; the 964b record's
`target` is this run's spec). A NEW EXQ NUMBER, not a lettered 964 re-test: the autopsy is
explicit that this is a new question. H-instr and H-auth were resolved by V3-EXQ-964b; H-mag
and H-pat are the alive legs this run adjudicates, and GOV-FANOUT-1 requires both nulls
declared, which they are below.

=== WHY THIS DESIGN IS REACHABLE -- READ OFF THE LANDED PREDECESSOR, NOT ASSUMED ===

From `v3_exq_964b_mech482_reachability_verify_lift_20260915T215220Z_v3.json`:

  ladder_max_selection_divergence_frac__s0.1   0.2105    ladder_seeds_firing__s0.1   3 of 3
  ladder_max_action_divergence_frac__s0.1      0.0944
  readiness_max_selection_divergence_frac      0.0       (the SUBJECT, at its own magnitude)
  readiness_max_action_divergence_frac         0.0
  subject_clamp_saturated_frac_worst           0.0       (the real readout never rails)
  max_lp_dev_range_worst                       0.0021222 (the real PRE-clamp range)

The s=0.1 rung IS the shipped authority, and its realised post-clamp range is exactly
2 * 0.1 = 0.2. So a post-clamp range of 0.2 demonstrably moves BOTH readouts on every seed,
while the subject's own realised range is ~0.0021 -- about two orders of magnitude short. The
gain needed to reach the rail is therefore ~37-94x depending on seed (36.6 / 94.2 / 47.5 on
seeds 71 / 101 / 202, per the amended substrate_queue entry), which is what sizes the ladder
k in {1, 10, 40, 100}: k=1 is the subject unchanged, k=100 clears the rail on the WORST seed.

This is the check V3-EXQ-1012b's refusal (GFLAG-0297) shows must be done BEFORE queuing: a
control that cannot bite makes the whole criterion degenerate. Here the control is already
MEASURED biting, at the exact realised range the top rungs will reach.

=== THE ARMS -- three families at four matched rungs, plus two retained controls ===

All followers are yoked on ONE reference-driven rollout per seed, so every comparison is a
PAIRED flip against the same observation sequence. Each family wraps
`agent._curiosity_per_candidate_learning_progress` (agent.py:6332) -- the exact seam the real
deficit travels -- and RETURNS None WHENEVER THE REAL METHOD DOES, so every arm fires on
exactly the ticks the subject's own mechanism fires on. Authority and cadence are untouched;
only the vector's magnitude and its pattern differ.

  ARM_READINESS          the SUBJECT at k=1, imported unchanged. (V3-EXQ-964b's ARM_RDY.)
  ARM_REAL_K{10,40,100}  the subject's OWN readout multiplied by k. ** THE MANIPULATION. **
  ARM_SYN_K{1,10,40,100}   RANGE-MATCHED SYNTHETIC. A linear ramp rescaled WITHIN THE TICK so
                         its pre-clamp cross-candidate range EQUALS that rung's real range.
                         Same magnitude, different pattern.
  ARM_PERM_K{1,10,40,100}  PERMUTED-REAL. That rung's real vector with its values PERMUTED
                         ACROSS CANDIDATES within the tick. The value MULTISET is identical,
                         so the range matches by construction and not merely by rescaling --
                         a strictly tighter control than the synthetic.
  ARM_VERIFY_LIFT_S10    V3-EXQ-964b's top positive control, retained verbatim. Fires 3/3.

The synthetic and permuted arms exist because they answer different objections. The synthetic
shows a DIFFERENT pattern at the same range fires; the permuted shows THIS pattern's own
values, re-assigned, fire. Only the permuted arm holds the value distribution EXACTLY fixed,
so only it isolates candidate-value CORRESPONDENCE as the manipulated thing.

=== DV-SYMMETRY INVARIANCE, DECLARED PER ARM (mandatory) ===

Both DVs are argmax-derived: the E3 SELECTION index on E3-fresh ticks, and the committed
action. The symmetry group is (i) addition of a candidate-UNIFORM constant and (ii) monotone
rescaling of the whole score vector; neither moves an argmax.

  ARM_PREREADINESS   IS invariant under (i) -- a single enclosing target makes the deficit
      candidate-uniform. DISPOSITION (b): scoped OUT of divergence scoring, retained as the
      yoked reference and the collapse control. Imported verbatim from V3-EXQ-964a/b.
  ARM_READINESS and ARM_REAL_Kk   NOT invariant. `rbf_weighted` varies continuously with
      candidate position, and multiplying a non-uniform vector by k > 0 keeps it non-uniform.
      Note k is NOT a monotone rescaling of the SCORE: the bias is one additive term inside
      the score, and it is CLAMPED, so k changes the argmin-relevant deviation's shape as well
      as its size. That is exactly why the clamp rail is the interesting point.
  ARM_SYN_Kk         NOT invariant. A linear ramp is uniform for no positive scale.
  ARM_PERM_Kk        NOT invariant, and this is the one that needs stating carefully. A
      permutation of interchangeable units IS a symmetry of any SET-AGGREGATE DV (mean, sum,
      variance). Our DVs are NOT set-aggregates: an argmax is a statement about WHICH index
      carries the extreme value, so relocating values across candidates moves it generically.
      The permuted arm is therefore a live control for these DVs and would NOT be one for a
      pooled-statistics DV.
  ARM_VERIFY_LIFT_S10  NOT invariant (a ramp, railed). Retained control.

=== WHAT FALSIFIES WHAT -- PRE-REGISTERED, BOTH LEGS DECLARED (GOV-FANOUT-1) ===

`RAIL_RUNG` is the smallest k whose REAL arm realises a post-clamp range at the rail
(>= `RAIL_FRACTION` x 2 * curiosity_bias_scale) on a seed majority. Firing means selection
divergence above `DIVERGENCE_FLOOR` on a seed majority.

  F_MAG   the REAL arm FIRES at `RAIL_RUNG`.
          -> `deficit_readout_magnitude_limited`. H-mag supported, H-pat weakened. The
             readout's content is selection-relevant and only its size was short, so a ~50-100x
             gain on `curiosity_learning_progress_weight` is a CONFIG fix and MECH-482's first
             genuine behavioural test follows. ROUTING: /governance to ratify the gain, then
             /queue-experiment for that behavioural test. NOT a substrate build.
  F_PAT   the REAL arm does NOT fire at `RAIL_RUNG` while BOTH the range-matched synthetic AND
          the permuted-real arm DO, at the same rung and on a seed majority.
          -> `deficit_readout_pattern_limited`. H-pat supported, H-mag eliminated. The readout
             carries no selection-relevant candidate ordering: its structure, not its size, is
             the problem, so the repair is on the READOUT or the TARGET FRAME. This is a
             GENUINE `substrate_ceiling` reading and the re-derive brake FIRES. ROUTING:
             /implement-substrate on the readout, NOT another lettered magnitude iteration.

F_MAG and F_PAT are MUTUALLY EXCLUSIVE BY CONSTRUCTION (one requires the real arm to fire at
the rail rung, the other requires it not to), so no combination of them can produce an
aggregation-vacuity PASS. Any other pattern -- notably the real arm silent AND the controls
also silent at the rail -- is `inconclusive_controls_did_not_separate` and routes to
/failure-autopsy: it means this harness cannot separate the two legs and neither is licensed.

=== AVOIDING THE ORDERED-GATE VACUOUS-PASS FALSE FLAG (carried caveat) ===

V3-EXQ-964b was logged `outcome PASS` with `C3 FAIL`, which the indexer's ordered-gate handling
reads as aggregation vacuity -- an open defect
(`chip-20260916-ordered-gate-vacuous-pass-false-flag`). This run is structured so the shape
cannot recur: **the only `load_bearing` criterion is the INSTRUMENT gate** (the positive
control fires AND the top rung reaches the rail AND the range matching holds), which is what
`outcome` tracks. F_MAG and F_PAT are `load_bearing: false` FINDING criteria that route the
LABEL, with an explicit `combination_rule`. A PASS therefore always means "a valid,
instrument-verified measurement was obtained", and the scientific content is read off the
label -- never off a failed load-bearing criterion sitting under a PASS.

=== PRECONDITIONS (any red -> `substrate_not_ready_requeue`, never a substrate verdict) ===

  G0  the retained verify-lift positive control fires on a seed majority -- the detector works
      in THIS run, not just in V3-EXQ-964b.
  G1  the REAL arm at the TOP rung (k=100) reaches the clamp rail on a seed majority. If the
      amplification never reaches the range at which the synthetic is known to fire, the H-mag
      question was never posed and a silent real arm means nothing. THIS IS THE GATE THAT
      MAKES A NULL INTERPRETABLE.
  G2  per rung, the synthetic's realised post-clamp range matches the real arm's to within
      `RANGE_MATCH_TOL` (relative) on every cell -- otherwise "range-matched" is false and the
      pattern contrast is confounded with a magnitude difference.
  G3  the permuted arm's realised post-clamp range matches the real arm's to the same
      tolerance (it should match essentially exactly -- same multiset).
  G4  the self-yoked instrument control diverges exactly 0 on both DVs (imported discipline).
  G5  no permutation was the identity (a resampled identity would make ARM_PERM == ARM_REAL).

=== THE ROUTED DV IS COMMITTED-TICK DIVERGENCE (red-team pass 1, finding 1) ===

E3 selects by deterministic `argmin` only while COMMITTED; otherwise it draws from
`softmax(-scores / temperature)`, and `last_selected_idx` is written either way. Every runner
restores its own torch RNG state around each tick and all runners start from one
`reset_all_rng(seed)`, so on a SAMPLED tick the follower and the reference consume the SAME
uniform draw and a follower "diverges" iff its bias nudged the CDF across that draw -- a
MAGNITUDE-scaled lottery that is essentially PATTERN-BLIND. Counting those ticks would let one
lottery produce EITHER verdict under EITHER hypothesis: F_MAG from the real arm winning it,
F_PAT from both controls winning it while the real lost. V3-EXQ-964b's own corrected instrument
already refuses to treat a sampled flip as reachability; this run's routed DV now agrees with
it.

So `_TracedRunner` records `(e3_fresh, selected_idx, committed_now)` per tick for the reference
AND every follower, F_MAG / F_PAT route on COMMITTED-tick divergence only, and sampled-tick
divergence is recorded per rung as a clearly-named DIAGNOSTIC that nothing routes on.
`G7_committed_partition_present` gates the fallback so the pooled fraction can never silently
reach a verdict.

`G6_committed_channel_reachable_at_rail` is the matching headroom certificate: the clamp bounds
the argmin-relevant deviation at `2 * curiosity_bias_scale = 0.2`, so if every committed margin
exceeds that, NO gain on the readout can move a committed selection at any k and a silent real
arm would mean nothing. SCOPE NOTE, because it decides whether this design is posable at all:
the 2-episode authoring smoke recorded a smallest positive margin of ~0.39 and
`corrected_n_flip_reachable_total = 0` on every rung, which read as "unreachable by
construction". That is a SMALL-SAMPLE artifact of the toy schedule. At the FULL 3-episode
schedule V3-EXQ-964b's landed manifest records smallest positive margins of 0.171 / 0.0021 /
0.0002 and `corrected_n_flip_reachable_total` of 2 / 42 / 25 on seeds 71 / 101 / 202 for its
shipped-authority arm at this same realised range -- so the committed channel IS live on 3 of 3
seeds where this run actually operates. G6 measures it in-run rather than inheriting it.

=== RED-TEAM RECORD (Step 4.5) ===

See the queue entry note for the verdict, the reviewing model, and the disposition of every
finding.
"""

from __future__ import annotations

import argparse
import datetime
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

import experiments.v3_exq_964a_mech482_epistemic_deficit_multitarget_readiness as x964a  # noqa: E402
import experiments.v3_exq_964b_mech482_reachability_verify_lift as x964b  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1047_mech482_amplified_readout_ladder"
QUEUE_ID = "V3-EXQ-1047"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = ["MECH-482"]
BEARS_ON = ["orienting_epistemic_deficit_v3_plan:ORNT-2"]

# Cells are opened and stamped by x964b's own yoked-group machinery, which this driver reuses
# unchanged; it never opens a cell itself.
ARM_FINGERPRINT_EXEMPT = (
    "this driver reuses x964b.run_yoked_group, which owns the per-cell RNG reset discipline "
    "(reset_all_rng(seed) before every runner construction); the manifest's arm_results rows "
    "are per-seed yoked groups, not independently re-runnable single-arm cells.")

# G0/G1/G2/G3/G5 are definitional or are anchored on V3-EXQ-964b's OWN measured values, which
# are carried as literals below and re-scored in-run by the shipped predicate.
ANCHOR_REACHABILITY_EXEMPT = (
    "G0's control is V3-EXQ-964b's retained verify-lift arm, MEASURED firing 3/3 seeds at "
    "every rung in the landed 964b manifest; G1's rail is the arithmetic 2 * "
    "curiosity_bias_scale that the same manifest shows the synthetic reaching at s=0.1; "
    "G2/G3/G5 are construction identities. None is a hand-tuned approximation of a signature.")

# ---- THE LADDER, sized off V3-EXQ-964b's own measured numbers ------------------------------
GAIN_LADDER: List[int] = [1, 10, 40, 100]
TOP_GAIN = GAIN_LADDER[-1]
# The gain that takes the real readout to the clamp rail, per seed, from the amended
# substrate_queue entry. Recorded so the ladder's sizing is auditable, not re-derived here.
REF_GAIN_TO_RAIL_964B = {71: 36.6, 101: 94.2, 202: 47.5}
# V3-EXQ-964b's landed readouts, carried as CONTEXT and never as a comparator.
REF_964B = {
    "ladder_max_selection_divergence_frac__s0.1": 0.21052631578947367,
    "ladder_max_action_divergence_frac__s0.1": 0.09444444444444444,
    "ladder_seeds_firing__s0.1": 3,
    "readiness_max_selection_divergence_frac": 0.0,
    "readiness_max_action_divergence_frac": 0.0,
    "subject_clamp_saturated_frac_worst": 0.0,
    "max_lp_dev_range_worst": 0.0021221935749053955,
}

# ---- PRE-REGISTERED, ABSOLUTE ---------------------------------------------------------------
RAIL_FRACTION = 0.95        # "at the rail" = >= 0.95 * (2 * curiosity_bias_scale)
RANGE_MATCH_TOL = 0.05      # relative tolerance for "range-matched"
DIVERGENCE_FLOOR = x964b.DIVERGENCE_FLOOR       # 1e-9, imported
SEED_MAJORITY = x964b.SEED_MAJORITY             # 2 of 3, imported
SEEDS = list(x964b.SEEDS)                       # [71, 101, 202]
DRY_RUN_SEEDS = list(x964b.DRY_RUN_SEEDS)
EPISODES = x964b.EPISODES
STEPS_PER_EPISODE = x964b.STEPS_PER_EPISODE
DRY_RUN_EPISODES = x964b.DRY_RUN_EPISODES
DRY_RUN_STEPS = x964b.DRY_RUN_STEPS
SHIPPED_BIAS_SCALE = 0.1                        # config.py:4646, NOT lifted on any real arm
RAIL_RANGE = 2.0 * SHIPPED_BIAS_SCALE           # = 0.2, the realised post-clamp range at rail

ARM_PRE = x964b.ARM_PRE
ARM_RDY = x964b.ARM_RDY
RETAINED_LIFT_ARM = "ARM_VERIFY_LIFT_S10"


def _real_arm(k: int) -> str:
    return ARM_RDY if int(k) == 1 else "ARM_REAL_K%d" % int(k)


def _syn_arm(k: int) -> str:
    return "ARM_SYN_K%d" % int(k)


def _perm_arm(k: int) -> str:
    return "ARM_PERM_K%d" % int(k)


FOLLOWER_ARM_IDS: List[str] = (
    [_real_arm(k) for k in GAIN_LADDER]
    + [_syn_arm(k) for k in GAIN_LADDER]
    + [_perm_arm(k) for k in GAIN_LADDER]
    + [RETAINED_LIFT_ARM]
)

_ZG = ZGoalStreamAccumulator()

# Per-seed registry of the traced runners, so the COMMITTED-channel partition can be computed
# after x964b.run_yoked_group returns (it does not expose its runner objects).
_TRACE_REGISTRY: Dict[str, Any] = {}


class _TracedRunner(x964b._InstrumentedRunner):
    """x964b's runner, recording per tick whether E3 was FRESH, which candidate it selected,
    and whether the selection was COMMITTED (deterministic argmin) or SAMPLED.

    WHY THE PARTITION IS LOAD-BEARING (red-team pass 1, finding 1). E3 selects by deterministic
    `argmin` only while committed; otherwise it DRAWS from `softmax(-scores / temperature)`
    (`e3_selector.py`), and `last_selected_idx` is set either way. Every runner restores its own
    torch RNG state around each tick and all runners start from the same `reset_all_rng(seed)`,
    so on a SAMPLED tick the follower and the reference consume the SAME uniform draw and a
    follower "diverges" iff its bias nudged the CDF across that draw -- a MAGNITUDE-scaled
    lottery that is essentially PATTERN-BLIND. Counting those ticks would let the same lottery
    produce either verdict under either hypothesis: `F_MAG` from the real arm winning it, and
    `F_PAT` from the two controls winning it while the real lost.

    x964b's own corrected instrument already refuses to treat a sampled flip as reachability
    ("a stochastic flip is not a deterministic one"), and this run's routed DV must agree with
    it. So F_MAG / F_PAT route on COMMITTED-tick divergence only, and sampled-tick divergence
    is recorded separately as a diagnostic.
    """

    def __init__(self, cfg: Any, trace_tag: str = ARM_PRE) -> None:
        super().__init__(cfg)
        self.trace: List[Any] = []
        self.trace_tag = str(trace_tag)
        _TRACE_REGISTRY[self.trace_tag] = self

    def choose(self, obs: Any) -> Any:
        a = super().choose(obs)
        try:
            state = self.agent.e3.get_commitment_state() or {}
            committed = bool(state.get("committed_now"))
        except Exception:
            committed = False
        self.trace.append((bool(self.last_tick_e3_fresh),
                           self.last_tick_selected_idx, committed))
        return a


def _committed_partition(seed_groups_tag: str = ARM_PRE) -> Dict[str, Dict[str, Any]]:
    """Divergence split into COMMITTED and SAMPLED ticks, from this seed's traces."""
    ref = _TRACE_REGISTRY.get(seed_groups_tag)
    out: Dict[str, Dict[str, Any]] = {}
    if ref is None:
        return out
    for tag, f in _TRACE_REGISTRY.items():
        if tag == seed_groups_tag:
            continue
        n_c = n_c_div = n_s = n_s_div = 0
        for (rf, ri, rc), (ff, fi, fc) in zip(ref.trace, f.trace):
            if not (rf and ff and ri is not None and fi is not None):
                continue
            if rc and fc:
                n_c += 1
                if fi != ri:
                    n_c_div += 1
            else:
                n_s += 1
                if fi != ri:
                    n_s_div += 1
        out[tag] = {
            "n_committed_both_fresh": n_c,
            "n_committed_diverged": n_c_div,
            "committed_selection_divergence_frac": (n_c_div / n_c) if n_c else 0.0,
            "n_sampled_both_fresh": n_s,
            "n_sampled_diverged": n_s_div,
            "sampled_selection_divergence_frac": (n_s_div / n_s) if n_s else 0.0,
        }
    return out


# --------------------------------------------------------------------------------------
# THE THREE ARM FAMILIES
# --------------------------------------------------------------------------------------
class _GainRunner(_TracedRunner):
    """The SUBJECT's own readout, multiplied by k. Magnitude only: the vector's SHAPE, its
    cadence and the selection authority are all untouched."""

    FAMILY = "real"

    def __init__(self, cfg: Any, gain: int, seed: int, arm_id: str) -> None:
        super().__init__(cfg, trace_tag=str(arm_id))
        self.gain = int(gain)
        self.family = self.FAMILY
        self.arm_id = str(arm_id)
        self._perm_rng = random.Random("%s|%d|%d" % (arm_id, int(seed), int(gain)))
        self.n_identity_permutations_resampled = 0
        self.n_synthetic_endpoint_match_failures = 0
        self.max_real_pre_clamp_range = 0.0
        agent = self.agent
        real = agent._curiosity_per_candidate_learning_progress

        def _wrapped(candidates: Any) -> Optional[torch.Tensor]:
            v = real(candidates)
            if v is None:
                return None          # cadence-matched to the subject, always
            k = int(v.numel())
            if k < 2:
                return v             # a single candidate carries no cross-candidate range
            rng = float(v.max().item() - v.min().item())
            if rng > self.max_real_pre_clamp_range:
                self.max_real_pre_clamp_range = rng
            out = self.transform(v, rng)
            self.n_injected_ticks += 1
            orng = float(out.max().item() - out.min().item())
            if orng > self.max_injected_range:
                self.max_injected_range = orng
            return out

        agent._curiosity_per_candidate_learning_progress = _wrapped

    def transform(self, v: torch.Tensor, rng: float) -> torch.Tensor:
        return v * float(self.gain)

    def instrument_summary(self) -> Dict[str, Any]:
        """x964b's summary plus this run's own per-arm provenance.

        `n_identity_permutations_resampled` and `max_real_pre_clamp_range` are read back by
        preconditions G5 and G1, so they have to reach the manifest rather than staying on the
        runner object.
        """
        out = super().instrument_summary()
        out.update({
            "arm_family": self.family,
            "arm_gain": int(self.gain),
            "arm_id_1047": self.arm_id,
            "n_identity_permutations_resampled": int(self.n_identity_permutations_resampled),
            "n_synthetic_endpoint_match_failures": int(self.n_synthetic_endpoint_match_failures),
            "max_real_pre_clamp_range": float(self.max_real_pre_clamp_range),
            "curiosity_bias_scale": float(getattr(self.agent.config, "curiosity_bias_scale",
                                                  SHIPPED_BIAS_SCALE)),
        })
        return out


class _RangeMatchedSyntheticRunner(_GainRunner):
    """A LINEAR RAMP rescaled WITHIN THE TICK to the real arm's range at this rung.

    Same realised magnitude as `_GainRunner` at the same k (asserted in-run by G2), different
    pattern. It answers "does SOME pattern of this size move the selection?".
    """

    FAMILY = "synthetic"

    def transform(self, v: torch.Tensor, rng: float) -> torch.Tensor:
        """A MONOTONE interior between the REAL vector's own deviation endpoints at this rung.

        WHAT HAS TO MATCH, read off the source rather than assumed:
        `StructuredCuriosity.compute_score_bias` (structured_curiosity.py:607-641) computes
        `raw_deviation = total - total.mean()`, clamps THAT to +/- curiosity_bias_scale, and
        reports `_last_bias_range = deviation.max() - deviation.min()`. So the quantity this
        control must match is the MEAN-CENTRED DEVIATION'S ENDPOINTS -- not the raw vector's
        endpoints and not its range.

        TWO EARLIER DRAFTS GOT THIS WRONG AND THEIR OWN SMOKES CAUGHT IT, both at k=40 with a
        27.7% mismatch (real 0.1566 vs control 0.2000 -- the control railed on BOTH sides while
        the real railed on one):
          * scaling a ramp to `rng * gain` matched the PRE-clamp range, which the clamp then
            maps differently for differently-shaped vectors;
          * interpolating between the raw vector's min and max matched the raw endpoints, but
            the deviation subtracts the MEAN, and a linspace's mean is not the real vector's.
        A control advantaged in realised magnitude confounds the pattern contrast with a
        magnitude difference, in the direction that falsely supports H-pat.

        THE CONSTRUCTION. Let `d = scaled - scaled.mean()`, with `lo = d.min() < 0 < hi =
        d.max()` (true whenever the readout is not constant, since d is zero-mean). Build
        `e(alpha) = lo + (hi - lo) * r**alpha` on `r = linspace(0, 1, K)`: monotone, with
        endpoints exactly `lo` and `hi` for every alpha > 0, and with `mean(e)` decreasing
        continuously from ~hi (alpha -> 0) to ~lo (alpha -> inf). Bisect alpha until
        `mean(e) == 0`; then `e - mean(e) == e` has EXACTLY the real deviation's endpoints, so
        it clamps to exactly the same post-clamp range -- by construction, not by rescaling.
        What differs is the INTERIOR: a smooth monotone curve instead of the readout's own
        per-candidate shape. That is the manipulated thing.
        """
        scaled = v * float(self.gain)
        k = int(scaled.numel())
        d = scaled - scaled.mean()
        lo = float(d.min().item())
        hi = float(d.max().item())
        if not (lo < 0.0 < hi):
            # Degenerate (constant readout): no zero-mean vector has these endpoints. Fall
            # back to the real deviation itself and COUNT it, so G2 cannot be satisfied by a
            # control that silently stopped being one.
            self.n_synthetic_endpoint_match_failures += 1
            return scaled
        r = torch.linspace(0.0, 1.0, k, dtype=scaled.dtype, device=scaled.device)
        a_lo, a_hi = 1e-3, 1e3
        e = None
        for _ in range(60):
            a = 0.5 * (a_lo + a_hi)
            e = lo + (hi - lo) * torch.pow(r, a)
            m = float(e.mean().item())
            if abs(m) <= 1e-12 * max(1.0, abs(hi - lo)):
                break
            # mean(e) DECREASES in alpha (more mass pushed toward lo).
            if m > 0.0:
                a_lo = a
            else:
                a_hi = a
        if e is None or abs(float(e.mean().item())) > 1e-6 * max(1.0, abs(hi - lo)):
            self.n_synthetic_endpoint_match_failures += 1
            return scaled
        return e + float(scaled.mean().item())


class _PermutedRealRunner(_GainRunner):
    """This rung's REAL vector with its values PERMUTED ACROSS CANDIDATES, within the tick.

    The value MULTISET is identical, so the range matches by construction rather than by
    rescaling -- the tightest available control. It answers "does THIS readout's own
    candidate-value CORRESPONDENCE carry the selection-relevant content?".

    The permutation is seeded from a STRING built from (arm_id, seed, gain) plus a per-tick
    counter -- never from `hash()`, which is PYTHONHASHSEED-salted and would make the null
    differ between runs of the same seed.
    """

    FAMILY = "permuted"

    def transform(self, v: torch.Tensor, rng: float) -> torch.Tensor:
        scaled = v * float(self.gain)
        k = int(scaled.numel())
        idx = list(range(k))
        for _attempt in range(16):
            self._perm_rng.shuffle(idx)
            if any(i != j for i, j in enumerate(idx)):
                break
            self.n_identity_permutations_resampled += 1
        return scaled[torch.as_tensor(idx, device=scaled.device)]


class _TracedVerifyLiftRunner(_GainRunner):
    """V3-EXQ-964b's verify-lift positive control, re-expressed through this run's traced
    plumbing so it is partitioned into committed and sampled ticks like every other arm.

    Behaviourally identical to `x964b._VerifyLiftRunner`: a linear ramp at `INJECTED_MAGNITUDE`,
    ignoring the real vector's VALUES (magnitude only) while still returning None on exactly
    the ticks the real method does (cadence matched). The lifted `curiosity_bias_scale` is set
    on its config by `_build_follower`. Re-expressed rather than subclassed because that class
    binds the UNTRACED base at x964b import time, so patching the base cannot reach it -- and
    an untraced positive control would leave G0 reading the POOLED fraction, which is the
    permissive direction for the one gate that certifies the detector.
    """

    FAMILY = "verify_lift"

    def transform(self, v: torch.Tensor, rng: float) -> torch.Tensor:
        k = int(v.numel())
        ramp = torch.arange(k, dtype=v.dtype, device=v.device) / float(k - 1)
        return ramp * float(x964b.INJECTED_MAGNITUDE)


def _build_follower(arm_id: str) -> Any:
    """Every follower is ARM_READINESS's config, differing in ONE thing: what the wrapper on
    `_curiosity_per_candidate_learning_progress` returns. The SHIPPED `curiosity_bias_scale`
    is left alone on every real / synthetic / permuted arm -- only the retained control lifts
    it, and it is labelled as a control."""
    seed = int(_CURRENT_SEED["seed"])
    if arm_id == RETAINED_LIFT_ARM:
        cfg = x964a.build_config(readiness=True)
        cfg.curiosity_bias_scale = 10.0
        return _TracedVerifyLiftRunner(cfg, 1, seed, arm_id)
    if arm_id == ARM_RDY:
        cfg = x964a.build_config(readiness=True)
        return _GainRunner(cfg, 1, seed, arm_id)
    family, gain = arm_id.rsplit("_K", 1)
    gain = int(gain)
    cfg = x964a.build_config(readiness=True)
    if family == "ARM_REAL":
        return _GainRunner(cfg, gain, seed, arm_id)
    if family == "ARM_SYN":
        return _RangeMatchedSyntheticRunner(cfg, gain, seed, arm_id)
    if family == "ARM_PERM":
        return _PermutedRealRunner(cfg, gain, seed, arm_id)
    raise ValueError("unknown arm id %r" % (arm_id,))


_CURRENT_SEED: Dict[str, int] = {"seed": SEEDS[0]}


def _install() -> None:
    """Idempotent. x964b.run_yoked_group looks both names up as module globals."""
    x964b.FOLLOWER_ARM_IDS = list(FOLLOWER_ARM_IDS)
    x964b._build_follower = _build_follower
    x964b.VERIFY_LIFT_ARM_IDS = [RETAINED_LIFT_ARM]
    # run_yoked_group builds the yoked REFERENCE as `_InstrumentedRunner(...)` by module-global
    # lookup, so patching the class traces the reference too -- without which there is nothing
    # to partition the followers against.
    x964b._InstrumentedRunner = _TracedRunner


# --------------------------------------------------------------------------------------
def _sel_frac(group: Dict[str, Any], arm_id: str) -> Optional[float]:
    """THE ROUTED DV: selection divergence on COMMITTED, both-fresh ticks only.

    Falls back to the unpartitioned fraction ONLY if the trace is absent, which
    `G7_committed_partition_present` gates on -- so a silent fallback cannot reach a verdict.
    """
    a = group.get(arm_id)
    if not isinstance(a, dict):
        return None
    if "committed_selection_divergence_frac" in a:
        return a["committed_selection_divergence_frac"]
    return a.get("yoked_selection_divergence_frac")


def _sampled_frac(group: Dict[str, Any], arm_id: str) -> Optional[float]:
    a = group.get(arm_id)
    return None if not isinstance(a, dict) else a.get("sampled_selection_divergence_frac")


def _n_committed(group: Dict[str, Any], arm_id: str) -> Optional[int]:
    a = group.get(arm_id)
    return None if not isinstance(a, dict) else a.get("n_committed_both_fresh")


def _act_frac(group: Dict[str, Any], arm_id: str) -> Optional[float]:
    a = group.get(arm_id)
    return None if not isinstance(a, dict) else a.get("yoked_action_divergence_frac_fresh")


def _inst(group: Dict[str, Any], arm_id: str, key: str) -> Optional[float]:
    a = group.get(arm_id)
    if not isinstance(a, dict):
        return None
    return (a.get("instrument") or {}).get(key)


def _seeds_firing(groups: List[Dict[str, Any]], arm_id: str) -> int:
    n = 0
    for g in groups:
        v = _sel_frac(g, arm_id)
        if v is not None and float(v) > DIVERGENCE_FLOOR:
            n += 1
    return n


def _seeds_at_rail(groups: List[Dict[str, Any]], arm_id: str) -> int:
    n = 0
    for g in groups:
        r = _inst(g, arm_id, "max_post_clamp_bias_range")
        if r is not None and float(r) >= RAIL_FRACTION * RAIL_RANGE:
            n += 1
    return n


def _range_match_worst(groups: List[Dict[str, Any]], ctrl_arm: str,
                       real_arm: str) -> Dict[str, Any]:
    worst, worst_cell = 0.0, None
    for g in groups:
        a = _inst(g, real_arm, "max_post_clamp_bias_range")
        b = _inst(g, ctrl_arm, "max_post_clamp_bias_range")
        if a is None or b is None or float(a) <= 0.0:
            continue
        rel = abs(float(b) - float(a)) / float(a)
        if rel > worst:
            worst, worst_cell = rel, "%s@seed%s" % (ctrl_arm, g.get("seed"))
    return {"worst_relative_mismatch": worst, "offending_cell": worst_cell}


def run_experiment(episodes: int, steps: int, seeds: List[int],
                   dry_run: bool = False) -> Dict[str, Any]:
    _install()
    # A MAJORITY OF THE SEEDS ACTUALLY RUN. At the full 3 seeds this is 2, i.e. exactly
    # V3-EXQ-964b's inherited SEED_MAJORITY; it only differs on a reduced-seed smoke, where a
    # fixed 2-of-3 bar would be unmeetable and would leave every gate and both finding legs
    # unexercised by the very run that is supposed to test them.
    majority = max(1, (len(seeds) + 1) // 2)
    groups: List[Dict[str, Any]] = []
    for seed in seeds:
        _CURRENT_SEED["seed"] = int(seed)
        _TRACE_REGISTRY.clear()
        print("Seed %d Condition amplified_readout_ladder" % seed, flush=True)
        g = x964b.run_yoked_group(seed, episodes, steps, _ZG)
        part = _committed_partition()
        for tag, pv in part.items():
            if isinstance(g.get(tag), dict):
                g[tag].update(pv)
        g["committed_partition_present"] = bool(part)
        groups.append(g)
        print("verdict: PASS", flush=True)

    control = x964b.paired_control_divergence(
        seeds[0], readiness=True, episodes=1,
        steps=(x964b.DRY_RUN_CONTROL_STEPS if dry_run else steps))
    control_worst = max(float(control.get("action_divergence_frac", 0.0)),
                        float(control.get("selection_divergence_frac", 0.0)))

    # ---- per-rung summary --------------------------------------------------------------
    rungs: Dict[str, Any] = {}
    for k in GAIN_LADDER:
        real, syn, perm = _real_arm(k), _syn_arm(k), _perm_arm(k)
        rungs[str(k)] = {
            "gain": int(k),
            "real_arm": real, "synthetic_arm": syn, "permuted_arm": perm,
            "real_seeds_firing": _seeds_firing(groups, real),
            "synthetic_seeds_firing": _seeds_firing(groups, syn),
            "permuted_seeds_firing": _seeds_firing(groups, perm),
            "real_seeds_at_rail": _seeds_at_rail(groups, real),
            "real_max_post_clamp_range": [_inst(g, real, "max_post_clamp_bias_range")
                                          for g in groups],
            "real_clamp_saturated_frac": [_inst(g, real, "max_clamp_saturated_frac")
                                          for g in groups],
            "synthetic_max_post_clamp_range": [_inst(g, syn, "max_post_clamp_bias_range")
                                               for g in groups],
            "permuted_max_post_clamp_range": [_inst(g, perm, "max_post_clamp_bias_range")
                                              for g in groups],
            "real_committed_selection_divergence_frac": [_sel_frac(g, real) for g in groups],
            "synthetic_committed_selection_divergence_frac": [_sel_frac(g, syn) for g in groups],
            "permuted_committed_selection_divergence_frac": [_sel_frac(g, perm) for g in groups],
            "real_SAMPLED_selection_divergence_frac_DIAGNOSTIC": [_sampled_frac(g, real)
                                                                 for g in groups],
            "synthetic_SAMPLED_selection_divergence_frac_DIAGNOSTIC": [_sampled_frac(g, syn)
                                                                      for g in groups],
            "permuted_SAMPLED_selection_divergence_frac_DIAGNOSTIC": [_sampled_frac(g, perm)
                                                                     for g in groups],
            "real_n_committed_both_fresh": [_n_committed(g, real) for g in groups],
            "real_corrected_n_flip_reachable_total": [
                _inst(g, real, "corrected_n_flip_reachable_total") for g in groups],
            "real_selection_divergence_frac": [_sel_frac(g, real) for g in groups],
            "synthetic_selection_divergence_frac": [_sel_frac(g, syn) for g in groups],
            "permuted_selection_divergence_frac": [_sel_frac(g, perm) for g in groups],
            "real_action_divergence_frac_fresh": [_act_frac(g, real) for g in groups],
            "synthetic_range_match": _range_match_worst(groups, syn, real),
            "permuted_range_match": _range_match_worst(groups, perm, real),
        }

    rail_rung = None
    for k in GAIN_LADDER:
        if rungs[str(k)]["real_seeds_at_rail"] >= majority:
            rail_rung = k
            break

    # ---- PRECONDITIONS ------------------------------------------------------------------
    g0_firing = _seeds_firing(groups, RETAINED_LIFT_ARM)
    g1_top_at_rail = rungs[str(TOP_GAIN)]["real_seeds_at_rail"]
    g2_worst = max([rungs[str(k)]["synthetic_range_match"]["worst_relative_mismatch"]
                    for k in GAIN_LADDER] or [0.0])
    g3_worst = max([rungs[str(k)]["permuted_range_match"]["worst_relative_mismatch"]
                    for k in GAIN_LADDER] or [0.0])
    g6_reachable_seeds = 0
    _rail_probe = None
    for k in GAIN_LADDER:
        if rungs[str(k)]["real_seeds_at_rail"] >= majority:
            _rail_probe = k
            break
    if _rail_probe is not None:
        _ra = _real_arm(_rail_probe)
        for g in groups:
            v = _inst(g, _ra, "corrected_n_flip_reachable_total")
            if v is not None and float(v) >= 1.0:
                g6_reachable_seeds += 1

    n_identity = 0
    for g in groups:
        for k in GAIN_LADDER:
            a = g.get(_perm_arm(k)) or {}
            n_identity += int((a.get("instrument") or {}).get(
                "n_identity_permutations_resampled", 0) or 0)

    preconditions = [
        {"name": "G0_positive_control_fires",
         "description": ("the retained V3-EXQ-964b verify-lift arm fires in THIS run -- the "
                         "detector works here, not only in the predecessor"),
         "measured": float(g0_firing), "threshold": float(majority), "direction": "lower",
         "control": "V3-EXQ-964b measured this arm firing on 3 of 3 seeds at every rung",
         "met": bool(g0_firing >= majority)},
        {"name": "G1_top_rung_reaches_clamp_rail",
         "description": ("the REAL arm at k=%d reaches the clamp rail (>= %.2f x %.2f). Without "
                         "this the H-mag question was never posed and a silent real arm means "
                         "nothing" % (TOP_GAIN, RAIL_FRACTION, RAIL_RANGE)),
         "measured": float(g1_top_at_rail), "threshold": float(majority),
         "direction": "lower",
         "control": ("V3-EXQ-964b measured the real readout's pre-clamp range at 0.0021222 and "
                     "the gain to rail at 36.6 / 94.2 / 47.5 per seed, so k=%d clears it on the "
                     "worst seed" % TOP_GAIN),
         "met": bool(g1_top_at_rail >= majority)},
        {"name": "G2_synthetic_range_matched",
         "description": "the synthetic control's realised post-clamp range matches the real arm's",
         "measured": float(g2_worst), "threshold": float(RANGE_MATCH_TOL), "direction": "upper",
         "control": "worst relative mismatch over every rung and seed",
         "met": bool(g2_worst <= RANGE_MATCH_TOL)},
        {"name": "G3_permuted_range_matched",
         "description": ("the permuted control's realised post-clamp range matches the real "
                         "arm's -- it should match essentially exactly, same value multiset"),
         "measured": float(g3_worst), "threshold": float(RANGE_MATCH_TOL), "direction": "upper",
         "control": "worst relative mismatch over every rung and seed",
         "met": bool(g3_worst <= RANGE_MATCH_TOL)},
        {"name": "G6_committed_channel_reachable_at_rail",
         "description": ("at the rail rung the REAL arm records at least one ARITHMETICALLY "
                         "REACHABLE committed flip (x964b's corrected "
                         "`corrected_n_flip_reachable_total`, i.e. a committed tick whose "
                         "positive margin the post-clamp perturbation could cross) on a seed "
                         "majority. WITHOUT THIS the H-mag leg is unposeable on the "
                         "deterministic channel: the clamp bounds the argmin-relevant "
                         "deviation at 2 * curiosity_bias_scale = %.2f, so if every committed "
                         "margin exceeds that, no gain on the readout can move a committed "
                         "selection at ANY k and a silent real arm means nothing."
                         % RAIL_RANGE),
         "measured": float(g6_reachable_seeds), "threshold": float(majority),
         "direction": "lower",
         "control": ("V3-EXQ-964b measured corrected_n_flip_reachable_total = 2 / 42 / 25 on "
                     "seeds 71 / 101 / 202 for its shipped-authority arm at this same realised "
                     "range of %.2f, so the committed channel is live at full scale on 3 of 3 "
                     "seeds" % RAIL_RANGE),
         "met": bool(g6_reachable_seeds >= majority)},
        {"name": "G7_committed_partition_present",
         "description": ("every seed group carries the committed/sampled trace partition. The "
                         "routed DV is COMMITTED-tick divergence; without the partition it "
                         "would silently fall back to the pooled fraction, which mixes in "
                         "softmax-SAMPLED ticks where follower and reference share one uniform "
                         "draw and divergence is a magnitude-scaled, PATTERN-BLIND lottery"),
         "measured": float(sum(
             1 for g in groups
             if g.get("committed_partition_present")
             and all("committed_selection_divergence_frac" in (g.get(a) or {})
                     for a in FOLLOWER_ARM_IDS))),
         "threshold": float(len(groups)), "direction": "lower",
         "control": "one per seed group, EVERY follower arm partitioned, by construction",
         "met": bool(all(
             g.get("committed_partition_present")
             and all("committed_selection_divergence_frac" in (g.get(a) or {})
                     for a in FOLLOWER_ARM_IDS) for g in groups))},
        {"name": "G4_self_yoked_control_exactly_zero",
         "description": "an arm yoked against itself must diverge exactly 0 on both DVs",
         "measured": float(control_worst), "threshold": float(DIVERGENCE_FLOOR),
         "direction": "upper",
         "control": "V3-EXQ-964b's own seeded self-yoked control, imported",
         "met": bool(control_worst <= DIVERGENCE_FLOOR)},
        {"name": "G5_no_identity_permutations",
         "description": ("no permuted-real tick fell back to the identity permutation, which "
                         "would make that arm a copy of the real one"),
         "measured": float(n_identity), "threshold": 0.0, "direction": "upper",
         "control": "count over every permuted cell; the transform resamples up to 16 times",
         "met": bool(n_identity == 0)},
    ]
    gate_green = all(bool(p["met"]) for p in preconditions)
    failed = [p["name"] for p in preconditions if not p["met"]]

    # ---- THE FINDING --------------------------------------------------------------------
    f_mag = f_pat = False
    rr = rungs[str(rail_rung)] if rail_rung is not None else None
    if gate_green and rr is not None:
        f_mag = bool(rr["real_seeds_firing"] >= majority)
        f_pat = bool(rr["real_seeds_firing"] < majority
                     and rr["synthetic_seeds_firing"] >= majority
                     and rr["permuted_seeds_firing"] >= majority)

    if not gate_green:
        label, outcome = "substrate_not_ready_requeue", "FAIL"
        routing = ("instrument not ready -- %s. NOT a substrate verdict and NOT evidence about "
                   "MECH-482." % ", ".join(failed))
    elif rail_rung is None:
        label, outcome = "substrate_not_ready_requeue", "FAIL"
        routing = ("no ladder rung reached the clamp rail on a seed majority, so the H-mag leg "
                   "was never posed. Instrument/sizing result, not a substrate verdict.")
    elif f_mag:
        label, outcome = "deficit_readout_magnitude_limited", "PASS"
        routing = ("H-MAG SUPPORTED, H-pat weakened: the real readout's own ORDERING moves the "
                   "COMMITTED E3 selection once amplified to the rail (k=%d), so its content "
                   "is selection-relevant and the shortfall is SIZE. WHAT THIS DOES AND DOES "
                   "NOT LICENSE (red-team pass 1, finding 2): it licenses 'the ordering is "
                   "selection-relevant'. It does NOT by itself license 'a gain on "
                   "curiosity_learning_progress_weight makes the mechanism behaviourally "
                   "live' -- at the shipped curiosity_bias_scale the clamp pins every railed "
                   "candidate, the real arm is already partially saturated at the rail, and "
                   "further weight is ABSORBED rather than transmitted. The next lever is "
                   "therefore AUTHORITY (curiosity_bias_scale) or the weight AND the clamp "
                   "together, not the weight alone. Read "
                   "`real_clamp_saturated_frac__k*` and `real_action_divergence_frac_fresh` at "
                   "the rail rung before sizing anything. Route to /governance to decide the "
                   "lever; the re-derive brake does NOT fire." % rail_rung)
    elif f_pat:
        label, outcome = "deficit_readout_pattern_limited", "PASS"
        routing = ("H-PAT SUPPORTED, H-mag ELIMINATED: at the rail (k=%d) the real readout is "
                   "silent while BOTH a range-matched synthetic AND a permutation of the "
                   "readout's OWN values move the selection. Its candidate ordering carries no "
                   "selection-relevant content, so amplification cannot help. This is a GENUINE "
                   "substrate_ceiling reading and the re-derive brake FIRES. Route to "
                   "/implement-substrate on the READOUT or the TARGET FRAME -- NOT another "
                   "lettered magnitude iteration." % rail_rung)
    else:
        label, outcome = "inconclusive_controls_did_not_separate", "PASS"
        routing = ("at the rail rung (k=%s) the real arm fired on %d seeds, the synthetic on "
                   "%d and the permuted on %d -- neither leg's pre-registered pattern held. "
                   "This harness did not separate H-mag from H-pat and NEITHER is licensed. "
                   "Route to /failure-autopsy. FIRST THING TO CHECK THERE: clamp SATURATION. "
                   "The rail rung is by definition the most saturated one, and "
                   "compute_score_bias pins every railed candidate at +/- "
                   "curiosity_bias_scale (structured_curiosity.py:607-641), so at high "
                   "saturation the real, synthetic and permuted vectors all collapse toward "
                   "the SAME railed pattern and the three arms stop being distinguishable BY "
                   "CONSTRUCTION rather than by finding. `real_clamp_saturated_frac__k*` and "
                   "the full per-rung firing table in `rungs` are recorded for exactly this: "
                   "the LEAST-saturated rung whose range is still large (k=40 realised ~0.157 "
                   "against the 0.20 rail in the authoring smoke) is where the three patterns "
                   "differ most and is the right place to re-pose the contrast."
                   % (rail_rung, rr["real_seeds_firing"], rr["synthetic_seeds_firing"],
                      rr["permuted_seeds_firing"]))

    print("[measurement] %s | rail_rung=%s | real/syn/perm seeds firing at rail = %s"
          % (label, rail_rung,
             (rr["real_seeds_firing"], rr["synthetic_seeds_firing"], rr["permuted_seeds_firing"])
             if rr else None), flush=True)

    flat: Dict[str, Any] = {
        "n_seeds": len(seeds),
        "seed_majority": int(majority),
        "seed_majority_full_run": int(SEED_MAJORITY),
        "rail_range": float(RAIL_RANGE),
        "rail_fraction": float(RAIL_FRACTION),
        "rail_rung_gain": (float(rail_rung) if rail_rung is not None else None),
        "gate_green": 1 if gate_green else 0,
        "f_mag_magnitude_limited": 1 if f_mag else 0,
        "f_pat_pattern_limited": 1 if f_pat else 0,
        "positive_control_seeds_firing": float(g0_firing),
        "top_rung_seeds_at_rail": float(g1_top_at_rail),
        "synthetic_range_match_worst": float(g2_worst),
        "permuted_range_match_worst": float(g3_worst),
        "self_yoked_control_worst": float(control_worst),
        "n_identity_permutations": float(n_identity),
    }
    for k in GAIN_LADDER:
        r = rungs[str(k)]
        flat["real_seeds_firing__k%d" % k] = float(r["real_seeds_firing"])
        flat["synthetic_seeds_firing__k%d" % k] = float(r["synthetic_seeds_firing"])
        flat["permuted_seeds_firing__k%d" % k] = float(r["permuted_seeds_firing"])
        flat["real_seeds_at_rail__k%d" % k] = float(r["real_seeds_at_rail"])
        rmax = [v for v in r["real_max_post_clamp_range"] if v is not None]
        if rmax:
            flat["real_max_post_clamp_range__k%d" % k] = float(max(rmax))
        sat = [v for v in r["real_clamp_saturated_frac"] if v is not None]
        if sat:
            flat["real_clamp_saturated_frac__k%d" % k] = float(max(sat))
    flat = {k: v for k, v in flat.items()
            if v is not None and (not isinstance(v, float) or v == v)}

    return {
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": list(CLAIM_IDS),
        "bears_on": list(BEARS_ON),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "sleep_driver_pattern": "none",
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "readout": flat,
        "arm_results": groups,
        "rungs": rungs,
        "self_yoked_control": control,
        "diagnostics": {"ref_964b": REF_964B, "ref_gain_to_rail_964b": REF_GAIN_TO_RAIL_964B},
        "interpretation": {
            "label": label,
            "routing": routing,
            "rail_rung_gain": rail_rung,
            "preconditions": preconditions,
            "criteria": [
                {"name": "C0_instrument_gate",
                 "load_bearing": True,
                 "passed": bool(gate_green),
                 "measured": float(sum(1 for p in preconditions if p["met"])),
                 "threshold": float(len(preconditions)),
                 "threshold_note": ("THE ONLY load-bearing criterion, and what `outcome` "
                                    "tracks: a PASS means a valid instrument-verified "
                                    "measurement was obtained. The scientific content is read "
                                    "off the LABEL, never off a failed load-bearing criterion "
                                    "under a PASS -- the ordered-gate vacuous-pass shape "
                                    "V3-EXQ-964b was false-flagged for.")},
                {"name": "F_MAG_real_fires_at_rail",
                 "load_bearing": False,
                 "passed": bool(f_mag),
                 "measured": (float(rr["real_seeds_firing"]) if rr else None),
                 "threshold": float(majority),
                 "threshold_note": "routes the LABEL, not the outcome. H-mag leg."},
                {"name": "F_PAT_controls_fire_where_real_does_not",
                 "load_bearing": False,
                 "passed": bool(f_pat),
                 "measured": (float(min(rr["synthetic_seeds_firing"],
                                        rr["permuted_seeds_firing"])) if rr else None),
                 "threshold": float(majority),
                 "measured_real_seeds_firing": (float(rr["real_seeds_firing"]) if rr else None),
                 "threshold_note": ("routes the LABEL, not the outcome. H-pat leg: requires the "
                                    "real arm BELOW the majority and BOTH controls at or above "
                                    "it, at the same rung.")},
            ],
            "combination_rule": ("C0 alone decides the OUTCOME. F_MAG and F_PAT decide the "
                                 "LABEL and are MUTUALLY EXCLUSIVE BY CONSTRUCTION -- one "
                                 "requires the real arm to fire at the rail rung, the other "
                                 "requires it not to -- so no combination of them can produce "
                                 "an aggregation-vacuity PASS. Neither firing is "
                                 "`inconclusive_controls_did_not_separate`."),
            "criteria_non_degenerate": {
                "C0_instrument_gate": True,
                "F_MAG_real_fires_at_rail": bool(gate_green and rail_rung is not None),
                "F_PAT_controls_fire_where_real_does_not": bool(
                    gate_green and rail_rung is not None),
            },
            "gate_reason": (None if gate_green
                            else "preconditions unmet: %s" % ", ".join(failed)),
            "null_reading": ("`inconclusive_controls_did_not_separate` means the harness could "
                             "not separate H-mag from H-pat -- NOT that the mechanism is fine "
                             "and NOT that it is broken. Neither leg is licensed and the "
                             "re-derive brake does NOT fire on it."),
        },
        "non_degenerate": bool(gate_green),
        "degeneracy_reason": (None if gate_green
                              else "preconditions unmet: %s" % ", ".join(failed)),
    }


def _run_self_test() -> int:
    fails = 0

    def _chk(ok: bool, msg: str) -> None:
        nonlocal fails
        if not ok:
            fails += 1
            print("[self-test] FAIL: %s" % msg, flush=True)

    _install()
    _chk(x964b._build_follower is _build_follower, "the follower patch did not install")
    _chk(list(x964b.FOLLOWER_ARM_IDS) == list(FOLLOWER_ARM_IDS), "the arm list did not install")
    _chk(x964b.VERIFY_LIFT_ARM_IDS and x964b.VERIFY_LIFT_ARM_IDS[-1] in FOLLOWER_ARM_IDS,
         "run_yoked_group's progress print indexes VERIFY_LIFT_ARM_IDS[-1]; it must be an arm "
         "this run actually builds")
    _chk(ARM_RDY in FOLLOWER_ARM_IDS, "run_yoked_group's print also indexes ARM_RDY")
    _chk(len(set(FOLLOWER_ARM_IDS)) == len(FOLLOWER_ARM_IDS), "duplicate arm id")
    _chk(_real_arm(1) == ARM_RDY, "k=1 must BE the subject arm, not a copy of it")

    # THE SIZING: the ladder must actually reach the rail on the worst seed, from the
    # predecessor's own measured numbers. This is the check GFLAG-0297 (V3-EXQ-1012b) shows
    # must happen before queuing.
    worst_gain = max(REF_GAIN_TO_RAIL_964B.values())
    _chk(TOP_GAIN >= worst_gain,
         "TOP_GAIN %d is below the worst seed's measured gain-to-rail %.1f -- G1 would be "
         "unmeetable and the H-mag leg unposeable" % (TOP_GAIN, worst_gain))
    _chk(GAIN_LADDER[0] == 1, "the bottom rung must be the subject unchanged")
    _chk(any(g < worst_gain for g in GAIN_LADDER),
         "the ladder needs at least one rung BELOW the rail, or there is no dose-response")
    # NEGATIVE CONTROL: the subject's own measured range must NOT already reach the rail, or
    # the whole amplification question is moot.
    _chk(REF_964B["max_lp_dev_range_worst"] < RAIL_FRACTION * RAIL_RANGE,
         "the subject already reaches the rail at k=1; re-read why this run exists")
    _chk(REF_964B["ladder_seeds_firing__s0.1"] >= SEED_MAJORITY,
         "the predecessor's control did NOT fire at the shipped authority, so a 0.2 realised "
         "range is not known to move the DV and this design is not reachable")

    # The transforms.
    v = torch.tensor([0.0, 1.0, 2.0, 3.0])
    g = _GainRunner.transform.__get__(type("S", (), {"gain": 10})(), None)
    _chk(bool(torch.allclose(g(v, 3.0), v * 10.0)), "the gain transform is wrong")

    class _P:
        gain = 10
        _perm_rng = random.Random("t")
        n_identity_permutations_resampled = 0
    p = _PermutedRealRunner.transform.__get__(_P(), None)
    out = p(v, 3.0)
    _chk(sorted(out.tolist()) == sorted((v * 10.0).tolist()),
         "the permuted transform must preserve the value MULTISET exactly")
    _chk(float(out.max() - out.min()) == float((v * 10).max() - (v * 10).min()),
         "the permuted transform must preserve the RANGE exactly")

    class _S:
        gain = 10
        n_synthetic_endpoint_match_failures = 0
    _sobj = _S()
    s = _RangeMatchedSyntheticRunner.transform.__get__(_sobj, None)
    outs = s(v, 3.0)
    scaled = v * 10.0
    # THE PROPERTY THAT MATTERS: the MEAN-CENTRED DEVIATION endpoints must match exactly, since
    # that is what compute_score_bias clamps (structured_curiosity.py:607-641).
    for _tv in (torch.tensor([0.0, 0.1, 0.2, 3.0]),
                torch.tensor([1.0, 1.05, 1.2, 1.25, 4.0]),
                torch.tensor([-2.0, 0.3, 0.31, 0.32])):
        _o = s(_tv, 0.0)
        _d_real = (_tv * 10.0) - (_tv * 10.0).mean()
        _d_syn = _o - _o.mean()
        _chk(abs(float(_d_syn.min()) - float(_d_real.min())) < 1e-4
             and abs(float(_d_syn.max()) - float(_d_real.max())) < 1e-4,
             "synthetic DEVIATION endpoints must match the real's: got [%r, %r] vs [%r, %r] "
             "on %r" % (float(_d_syn.min()), float(_d_syn.max()),
                        float(_d_real.min()), float(_d_real.max()), _tv.tolist()))
        _chk(not bool(torch.allclose(_d_syn, _d_real, atol=1e-6)),
             "on a NON-linear readout the synthetic's interior must differ from the real's")
    _chk(_sobj.n_synthetic_endpoint_match_failures == 0,
         "the endpoint-matching bisection failed on a well-formed vector")



    print("[self-test] %d failure(s)" % fails, flush=True)
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="V3-EXQ-1047: MECH-482 amplified-readout ladder (magnitude vs pattern)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_run_self_test())

    t0 = time.perf_counter()
    seeds = list(DRY_RUN_SEEDS if args.dry_run else SEEDS)
    episodes = DRY_RUN_EPISODES if args.dry_run else EPISODES
    steps = DRY_RUN_STEPS if args.dry_run else STEPS_PER_EPISODE
    print("=== %s ===" % EXPERIMENT_TYPE, flush=True)
    print("Queue ID: %s  Claims: %s" % (QUEUE_ID, CLAIM_IDS), flush=True)
    print("Ladder k=%s at SHIPPED curiosity_bias_scale=%.2f (rail range %.2f)"
          % (GAIN_LADDER, SHIPPED_BIAS_SCALE, RAIL_RANGE), flush=True)
    print("Arms: %d followers" % len(FOLLOWER_ARM_IDS), flush=True)

    result = run_experiment(episodes, steps, seeds, dry_run=bool(args.dry_run))

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["timestamp_utc"] = ts
    result["run_timestamp"] = ts
    result["run_id"] = "%s_%s_v3" % (EXPERIMENT_TYPE, ts)

    full_config = {
        "seeds": seeds,
        "episodes": episodes, "steps_per_episode": steps,
        "gain_ladder": list(GAIN_LADDER),
        "shipped_curiosity_bias_scale": float(SHIPPED_BIAS_SCALE),
        "rail_range": float(RAIL_RANGE),
        "arms": list(FOLLOWER_ARM_IDS),
        "reference_arm": ARM_PRE,
        "pre_registered_thresholds": {
            "RAIL_FRACTION": float(RAIL_FRACTION),
            "RANGE_MATCH_TOL": float(RANGE_MATCH_TOL),
            "DIVERGENCE_FLOOR": float(DIVERGENCE_FLOOR),
            "SEED_MAJORITY": int(SEED_MAJORITY),
        },
        "inherited_from": "V3-EXQ-964b",
        "dry_run": bool(args.dry_run),
    }
    out_path = write_flat_manifest(
        result, dry_run=bool(args.dry_run), config=full_config, seeds=seeds,
        script_path=Path(__file__), started_at=t0, z_goal_stream_stats=_ZG.stats())

    print("\nWrote manifest to: %s" % out_path, flush=True)
    print("outcome: %s (%s)" % (result["outcome"], result["interpretation"]["label"]), flush=True)
    _raw = str(result["outcome"]).upper()
    emit_outcome(outcome=_raw if _raw in ("PASS", "FAIL") else "FAIL",
                 manifest_path=str(out_path), dry_run=bool(args.dry_run))
