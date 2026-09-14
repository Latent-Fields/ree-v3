"""NOT A QUEUEABLE EXPERIMENT -- REFERENCE HARNESS ONLY (moved out of experiments/ 2026-09-11).

This file was V3-EXQ-1024. It was NEVER queued: four successive designs built from it were each
killed by adversarial design review, and its criteria as they stand are known-broken (the
clause-(ii) criterion is algebraically dominated by its own guarding precondition, and the
design-time geometry guard certifies idealised values the run does not realise on 4/5 seeds).
DO NOT queue it, and do not copy its criteria.

It is kept only for the parts that ARE sound and were expensive to find: the direct-drive
CandidateRuleField harness, the settled-baseline warmup via observe(), the straddling-recurrence
ecology, the overlapping-cluster geometry, measured recurrence-at-mint, and tag-space duplicate
detection. Renamed off the v3_exq_*.py pattern so it cannot trip another session's queue
atomicity audit.

Full verdict and the four-pass design history: chip chip-20260911-mech349-validation-redesign,
and GFLAG-0265 / GFLAG-0266 / GFLAG-0268 on MECH-349.
"""

"""V3-EXQ-1024: MECH-349 -- CandidateRuleField CREATE-face mint-trigger diagnostics.

red-team: PASS 4 PENDING (filled in before queueing). Passes 1-3 all returned BLOCKING and all
were acted on; their confirmed findings are recorded under DESIGN HISTORY below.

=== THE QUESTION THIS RUN SETTLES ===
MECH-349 (ARC-063 CREATE face) asserts that a NON-GRADIENT structural event mints a distinct
CandidateRule slot when a (context-bucket -> action-object) regularity
    (i) RECURS at least `crf_mint_recurrence_threshold` times, AND
    (ii) is NOT already covered by an existing rule's context_tag.
The claim has ZERO entries in claim_evidence.v1.json; its CREATE-face diagnostics exist on disk
only under neighbouring SD-078/ARC-063 attribution (V3-EXQ-806), recorded as prose in its
evidence_quality_note by GFLAG-0198 option B. This run is the owed option-C residual. C1 tests
clause (i); C2 tests clause (ii). Each is TWO-SIDED: the ecology can present both the state the
clause says should mint and the state it says should be refused, and the criteria require the
substrate to separate them in BOTH directions. Nothing else votes.

=== THE ECOLOGY: OVERLAPPING CLUSTERS ===
N_CLUSTERS clusters, each presenting TWO context regimes -- a base direction and a NEAR-TWIN at
centered cosine ~0.94-0.96 -- against a cross-cluster maximum of ~0.52. Cluster c recurs
CLUSTER_COUNTS[c] times per member, in shuffled order, with all regimes live from the first
counted tick and the SD-078 common-mode baseline settled beforehand by observe()-only ticks.

Why the twins are the whole point: clause (ii) says a regularity already covered by an existing
rule must be REFUSED. In an ecology of well-separated regimes that state is UNPRESENTABLE, so
the clause is untestable -- measured at the substrate's own default block threshold of 0.8, the
gate fired 190 times in a 254-tick run and every single fire was a regime blocking ITSELF on
re-presentation, with ZERO cross-regime blocks, because the maximum attainable cross-regime
cosine (0.529) sits structurally below the threshold. The twins make "a DISTINCT regularity that
an existing rule already covers" an actual state of the ecology. Recorded as GFLAG-0266.

Why the counts differ per cluster: if every regularity recurred far above every swept threshold
-- as it does in an equal-frequency ecology -- the threshold is inert and C1 measures nothing.
Straddling counts make clause (i) discriminating and yield a point prediction.

SCOPE LIMIT: this run tests the CREATE-face TRIGGER LOGIC under constructed input. It does NOT
test whether a real encoder supplies separable regimes (SD-078's question, evidenced by
V3-EXQ-806), and per MECH-349's own `what_would_answer` a null on any downstream
committed-action or behavioural DV is NOT falsifying for this claim and is not measured here.

=== PRE-REGISTERED CRITERIA (constants below; never derived from the run) ===
C1 (LOAD-BEARING) -- clause (i), RECURRENCE-gating. Both conjuncts required:
    (a) COUNTER IS READ AT ALL. Mint count at the counter-blind reference threshold (1, where
        the guard at candidate_rule_field.py's `_maybe_mint` can never block) strictly EXCEEDS
        mint count at the reference threshold, on >= C1_SEEDS_REQUIRED seeds. A substrate that
        ignores its recurrence counter produces the same count at both and fails this.
    (b) DOSE RESPONSE. Mint count is non-increasing across the 4x sweep (3 -> 6 -> 12) and the
        drop from 3 to 12 is at least C1_MIN_DROP clusters, on >= C1_SEEDS_REQUIRED seeds.
    NOTE what is deliberately NOT a criterion: "100% of mint events occur at recurrence >=
    threshold". That is an ARITHMETIC TAUTOLOGY of any substrate shaped like `_maybe_mint`
    (it increments the counter, then early-returns below the bar), so it cannot fail and carries
    no information. It is recorded as a structural fact instead. See GFLAG-0266.
C2 (LOAD-BEARING) -- clause (ii), NOVELTY-gating. Both conjuncts required, and together they are
    a two-sided test of coverage detection on ONE ecology:
    (a) COVERED IS REFUSED. At the operating block threshold (0.8, below the twin cosine), mint
        count equals the number of DISTINCT CLUSTERS minted -- i.e. every near-twin was refused
        as already covered, and no cluster minted twice -- on >= C2_SEEDS_REQUIRED seeds.
    (b) NOVEL IS ADMITTED. At the high block threshold (0.97, ABOVE the twin cosine, so a twin
        no longer counts as covered), mint count exceeds the operating-threshold count by at
        least C2_MIN_ADMIT, on >= C2_SEEDS_REQUIRED seeds.
    An inert gate produces the same count at both and fails (b); a gate that blocks everything
    fails (b); a gate that blocks nothing fails (a).

PASS iff every precondition is met AND C1 AND C2.

=== DESIGN-TIME GUARDS (refuse before compute; never lower a pre-registered threshold) ===
G1 every swept threshold has >= MIN_CLUSTERS_ABOVE_THRESHOLD clusters able to reach it;
G2 the threshold sweep's point predictions strictly DECREASE by >= C1_MIN_DROP across it --
   without this a narrow-band ecology silently makes C1(b) unmeetable and routes `weakens`;
G3 the measured geometry actually separates: the twin cosines are SANDWICHED in
   (BLOCK_OPERATING, BLOCK_HIGH) -- refused at the operating threshold, admitted at the high
   one -- and the max cross-cluster cosine stays below BLOCK_OPERATING. This is what makes CONTEXT_DIM / TWIN_EPS safe -- if either drifts
   so the block can no longer tell twins from strangers, the run REFUSES instead of reporting a
   flat sweep as a substrate finding.

=== DESIGN HISTORY -- three BLOCKING adversarial reviews ===
PASS 1. The distinctness criterion `crf_max_pairwise_rule_dist >= 1.5` was an arithmetic
  identity (nothing retires -> slots fill contiguously -> the statistic is a lookup on live-rule
  count off the fixed pinned_seed=6063 matrix; the floor is exactly `live_rules >= 3`), and its
  non-churn conjunct was pinned at 0 (retirement unreachable under availability_maintenance at
  ANY outcome sign). => both REMOVED from scoring; MECH-350 UNTAGGED.
PASS 2. The entire sweep signal came from a single-regime opening transient: regimes 1-7 minted
  exactly once each while all variance sat in regime 0, whose rules were mutual DUPLICATES (253
  of 435 tag pairs at cosine >= 0.95, = C(23,2)). Making all regimes simultaneously available --
  a free parameter of the driver, not of the claim -- drove the criterion from 5/5 to 0/5 with
  preconditions still met. => ecology rebuilt with straddling counts + a settled baseline;
  duplicate detection added as a gating precondition.
PASS 3. Clause (ii) was still untestable: at the operating block threshold every gate fire was a
  self-block (0 cross-regime blocks, max attainable cross-regime cosine 0.529 < 0.8), so the
  novelty criterion drew its whole signal from low-threshold arms where the gate FALSE-blocks
  distinct regimes. The recurrence conjunct was a tautology. Two driver constants (the count band
  and CONTEXT_DIM) could route `weakens`, and the not-ready branch emitted an `unknown`-direction
  row against a claim whose own governance ruling forbids exactly that. => overlapping-cluster
  ecology (this version), the tautology demoted to a structural fact, guards G2/G3 added, and the
  not-ready branch now emits NO claim tag.

=== RECORDED BUT NOT SCORED ===
* crf_max_pairwise_rule_dist -- structural identity (pass 1); MECH-350 NOT tagged.
* crf_n_retired_total -- structurally 0 here, so MECH-349 FALSIFYING(3)'s churn signature is
  UNREACHABLE and this run cannot bear on it. A rule mints at maintenance_floor 0.45 and, once
  it stops activating, FREEZES rather than decaying (credit() only updates a rule while its
  eligibility trace is alive, and eligibility is set only when the rule is ACTIVE), so it never
  reaches the 0.05 retire floor. Only the SIGN of the outcome signal is read. Owed separately:
  chip chip-20260911-mech349-churn-leg-maintenance-floor.
* recurrence-at-mint -- measured and recorded, but a tautology of the implementation (above).
* point-prediction agreement -- recorded; the sign-bucket can fragment a cluster's occurrences
  across keys, so the prediction is an upper bound rather than an identity.

=== NON-DEGENERACY PRECONDITIONS (MECH-349 what_would_answer (a)-(d)) ===
(a) mint fires; (b) mints are DISTINCT -- no cluster minted twice at the operating threshold;
(c) pool does not saturate; (d) the ARC-062 top-down seed leg is NOT under test (no arc062_seed
    is supplied at any call site) and is reported as untested, never as a null.

SLEEP DRIVER: not applicable (no SleepLoopManager; this driver steps the CandidateRuleField
directly and never constructs an agent).
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.run_id import make_run_id
from ree_core.policy.candidate_rule_field import (
    CandidateRuleField,
    CandidateRuleFieldConfig,
)

EXPERIMENT_TYPE = "v3_exq_1024_mech349_crf_mint_trigger_diagnostics"
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-349"]
RELATED_EXQ = ["V3-EXQ-806", "V3-EXQ-666c", "V3-EXQ-822"]

SEEDS = [0, 1, 2, 3, 4]

# --- ecology -----------------------------------------------------------------
CONTEXT_DIM = 16
N_CLUSTERS = 8
TWIN_EPS = 0.3                                  # near-twin offset -> centered cos ~0.94-0.96
CLUSTER_COUNTS = [1, 2, 4, 6, 9, 13, 19, 26]    # per member; straddles every swept threshold
REGIME_MAGNITUDE = 2.0
COMMON_MODE_SCALE = 3.0                         # SD-008 geometry
OBS_NOISE = 0.005
BASELINE_WARMUP_TICKS = 300                     # observe()-only; no mint, no recurrence count
OUTCOME_SIGNAL = 0.1

# --- substrate stack under test ---------------------------------------------
N_SLOTS = 64
RULE_DIM = 16
THRESH_COUNTER_BLIND = 1        # C1(a) reference: the guard can never block here
THRESH_DEFAULT = 3              # reference cell, shared by C1 and C2
THRESH_SWEEP = [3, 6, 12]       # C1(b), 4x
BLOCK_OPERATING = 0.8           # substrate default; BELOW the twin cosine -> twins refused
BLOCK_HIGH = 0.97               # ABOVE the twin cosine -> twins admitted

# --- pre-registered thresholds ----------------------------------------------
C1_SEEDS_REQUIRED = 4
C1_MIN_DROP = 2                 # clusters, across threshold 3 -> 12
C2_SEEDS_REQUIRED = 4
C2_MIN_ADMIT = 3                # extra mints when the block rises above the twin cosine
MIN_CLUSTERS_ABOVE_THRESHOLD = 2
PRECOND_A_MIN_MINTED = 2
PRECOND_A_SEED_FRACTION = 2.0 / 3.0


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _base_direction(c: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(1000 + int(c))
    v = torch.randn(CONTEXT_DIM, generator=g)
    return v / v.norm().clamp_min(1e-8)


def _twin_direction(c: int) -> torch.Tensor:
    """A near-twin of cluster c: same direction nudged along a fixed orthogonal."""
    g = torch.Generator()
    g.manual_seed(5000 + int(c))
    d = _base_direction(c)
    o = torch.randn(CONTEXT_DIM, generator=g)
    o = o - (o @ d) * d
    o = o / o.norm().clamp_min(1e-8)
    v = d + TWIN_EPS * o
    return v / v.norm().clamp_min(1e-8)


def build_ecology(n_clusters: int) -> Tuple[List[torch.Tensor], List[int]]:
    """Returns (directions, cluster_of_regime) with two regimes per cluster."""
    dirs: List[torch.Tensor] = []
    cluster_of: List[int] = []
    for c in range(n_clusters):
        dirs.extend([_base_direction(c), _twin_direction(c)])
        cluster_of.extend([c, c])
    return dirs, cluster_of


def measure_geometry(n_clusters: int) -> Dict[str, float]:
    """Centered pairwise cosines. Seed-independent: the directions are fixed constants."""
    dirs, cluster_of = build_ecology(n_clusters)
    mean = sum(dirs) / len(dirs)
    cen = [d - mean for d in dirs]
    twin_cos: List[float] = []
    cross_cos: List[float] = []
    for a, b in itertools.combinations(range(len(dirs)), 2):
        x, y = cen[a], cen[b]
        c = float((x @ y) / (x.norm() * y.norm()).clamp_min(1e-8))
        (twin_cos if cluster_of[a] == cluster_of[b] else cross_cos).append(c)
    return {
        "min_twin_cosine": min(twin_cos),
        "max_twin_cosine": max(twin_cos),
        "max_cross_cluster_cosine": max(cross_cos),
    }


def predicted_mints(threshold: int, counts: List[int]) -> int:
    return sum(1 for c in counts if c >= threshold)


def _build_config(*, mint_recurrence_threshold: int, mature_mint_block_threshold: float,
                  n_slots: int) -> CandidateRuleFieldConfig:
    return CandidateRuleFieldConfig(
        n_slots=n_slots,
        rule_dim=RULE_DIM,
        mint_recurrence_threshold=int(mint_recurrence_threshold),
        mature_mint_block_threshold=float(mature_mint_block_threshold),
        cue_centering=True,
        mature_pool_dynamics=True,
        availability_maintenance=True,
    )


def config_slice(*, mint_recurrence_threshold: int, mature_mint_block_threshold: float,
                 n_slots: int, counts: List[int], warmup: int) -> Dict[str, Any]:
    return {
        "context_dim": CONTEXT_DIM,
        "rule_dim": RULE_DIM,
        "n_slots": int(n_slots),
        "mint_recurrence_threshold": int(mint_recurrence_threshold),
        "mature_mint_block_threshold": float(mature_mint_block_threshold),
        "cue_centering": True,
        "mature_pool_dynamics": True,
        "availability_maintenance": True,
        "cluster_counts": list(counts),
        "twin_eps": TWIN_EPS,
        "regime_magnitude": REGIME_MAGNITUDE,
        "common_mode_scale": COMMON_MODE_SCALE,
        "obs_noise": OBS_NOISE,
        "baseline_warmup_ticks": int(warmup),
        "outcome_signal": OUTCOME_SIGNAL,
    }


def _run_field(*, seed: int, mint_recurrence_threshold: int, mature_mint_block_threshold: float,
               n_slots: int, counts: List[int], warmup: int) -> Dict[str, Any]:
    counts = list(counts)
    n_clusters = len(counts)
    dirs, cluster_of = build_ecology(n_clusters)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    common = torch.randn(CONTEXT_DIM, generator=gen) * COMMON_MODE_SCALE

    def context_for(i: int) -> torch.Tensor:
        return (common + dirs[i] * REGIME_MAGNITUDE
                + torch.randn(CONTEXT_DIM, generator=gen) * OBS_NOISE)

    cfg = _build_config(mint_recurrence_threshold=mint_recurrence_threshold,
                        mature_mint_block_threshold=mature_mint_block_threshold,
                        n_slots=n_slots)
    crf = CandidateRuleField(context_dim=CONTEXT_DIM, config=cfg)

    for i in range(warmup):
        crf.observe(context_for(i % len(dirs)))

    sequence: List[int] = []
    for i in range(len(dirs)):
        sequence.extend([i] * int(counts[cluster_of[i]]))
    sequence = [sequence[j] for j in torch.randperm(len(sequence), generator=gen).tolist()]

    mint_regimes: List[int] = []
    mint_ticks: List[int] = []
    recurrence_at_mint: List[int] = []
    matched_trace: List[int] = []

    for t, i in enumerate(sequence):
        if (t + 1) % 50 == 0 or (t + 1) == len(sequence):
            print("  [train] crf seed=%d ep %d/%d" % (seed, t + 1, len(sequence)), flush=True)
        ctx = context_for(i)
        crf.step(ctx, action_object_idx=int(i), outcome_signal=OUTCOME_SIGNAL)
        if crf._last_minted_this_step:
            mint_ticks.append(t)
            mint_regimes.append(int(i))
            key = (crf._context_bucket(crf._centered(ctx)), int(i))
            recurrence_at_mint.append(int(crf._recurrence.get(key, -1)))
        matched_trace.append(int(crf._last_n_matched))

    st = crf.get_state()
    minted = int(st["crf_n_minted_total"])
    live = int(st["crf_n_slots_minted"])
    clusters_minted = sorted({cluster_of[i] for i in mint_regimes})
    per_cluster: Dict[int, int] = {}
    for i in mint_regimes:
        c = cluster_of[i]
        per_cluster[c] = per_cluster.get(c, 0) + 1

    base = crf._baseline
    tags = [(r.context_tag.reshape(-1) - base) for r in crf._rules.values()]
    max_tag_cos = 0.0
    for a, b in itertools.combinations(range(len(tags)), 2):
        x, y = tags[a], tags[b]
        max_tag_cos = max(max_tag_cos,
                          float((x @ y) / (x.norm() * y.norm()).clamp_min(1e-8)))

    return {
        "seed": int(seed),
        "n_clusters": n_clusters,
        "n_regimes": len(dirs),
        "counted_ticks": len(sequence),
        "mint_recurrence_threshold": int(mint_recurrence_threshold),
        "mature_mint_block_threshold": float(mature_mint_block_threshold),
        "n_slots": int(n_slots),
        "crf_n_minted_total": minted,
        "crf_n_retired_total": int(st["crf_n_retired_total"]),
        "crf_live_rules_final": live,
        "crf_max_pairwise_rule_dist": float(st["crf_max_pairwise_rule_dist"]),
        "crf_live_over_slots": live / max(1, n_slots),
        # clause (ii) instruments
        "distinct_clusters_minted": len(clusters_minted),
        "mints_per_cluster": per_cluster,
        "max_cluster_multiplicity": max(per_cluster.values()) if per_cluster else 0,
        "mints_equal_distinct_clusters": bool(minted == len(clusters_minted)),
        "max_tag_cosine": max_tag_cos,
        # clause (i) instruments (recorded; the 100%-above-threshold fact is a tautology)
        "recurrence_at_mint": recurrence_at_mint,
        "n_mints_below_threshold": len([v for v in recurrence_at_mint
                                        if v < int(mint_recurrence_threshold)]),
        "predicted_mints": predicted_mints(int(mint_recurrence_threshold), counts),
        # banked internals
        "crf_frac_active": float(st["crf_frac_active"]),
        "crf_n_matched_mean": float(statistics.fmean(matched_trace)) if matched_trace else 0.0,
        "crf_n_maintained_reactivatable": int(st["crf_n_maintained_reactivatable"]),
        "n_recurrence_buckets": int(len(crf._recurrence)),
        "mint_ticks": mint_ticks,
    }


def _arm_thresh(th: int) -> str:
    return "ARM_THRESH_%d" % th


def _arm_block_high() -> str:
    return "ARM_BLOCK_HIGH"


def _by_seed(rows, arm, key) -> Dict[int, Any]:
    return {r["seed"]: r[key] for r in rows if r["arm"] == arm}


def _worst_cell(rows, key, mode="min") -> Tuple[float, str]:
    vals = [(float(r[key]), "%s:s%d" % (r["arm"], r["seed"]))
            for r in rows if _finite(r.get(key))]
    if not vals:
        return float("nan"), "none"
    return min(vals) if mode == "min" else max(vals)


def main(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = SEEDS[:1] if dry_run else SEEDS
    # The dry run uses the FULL ecology: it is only ~160 counted ticks, and every subset
    # tried during design either starved a swept threshold (G1) or flattened the
    # prediction drop (G2), making the smoke vacuous. Only seeds and warmup are reduced,
    # so the smoke exercises the same design the real run does.
    counts = CLUSTER_COUNTS
    warm = 80 if dry_run else BASELINE_WARMUP_TICKS
    n_seeds = len(seeds)
    n_clusters = len(counts)

    # ---------- DESIGN-TIME GUARDS: refuse before compute --------------------
    reach = {th: predicted_mints(th, counts) for th in [THRESH_COUNTER_BLIND] + THRESH_SWEEP}
    bad = [th for th in THRESH_SWEEP if reach[th] < MIN_CLUSTERS_ABOVE_THRESHOLD]
    if bad:
        raise SystemExit("G1 FAILED: threshold(s) %s reachable by fewer than %d clusters "
                         "(counts=%s, reachable=%s). Widen CLUSTER_COUNTS; never lower a "
                         "pre-registered threshold." % (bad, MIN_CLUSTERS_ABOVE_THRESHOLD,
                                                        counts, reach))
    drop = reach[THRESH_SWEEP[0]] - reach[THRESH_SWEEP[-1]]
    if drop < C1_MIN_DROP:
        raise SystemExit("G2 FAILED: the threshold sweep's point predictions fall by only %d "
                         "across %s (reachable=%s), below the pre-registered C1_MIN_DROP of %d. "
                         "C1(b) would be unmeetable BY CONSTRUCTION and the run would route "
                         "'weakens' on an ecology choice. Widen CLUSTER_COUNTS."
                         % (drop, THRESH_SWEEP, reach, C1_MIN_DROP))
    geom = measure_geometry(n_clusters)
    # The block fires when cosine >= threshold, so the twins must be SANDWICHED:
    #   BLOCK_OPERATING <= twin cosines < BLOCK_HIGH
    # -> at the operating threshold every twin is judged "already covered" and REFUSED;
    # -> at the high threshold none of them is, so the same twins are ADMITTED.
    # That sandwich is what makes C2 two-sided on one ecology.
    if not (geom["min_twin_cosine"] > BLOCK_OPERATING):
        raise SystemExit("G3 FAILED: min twin cosine %.4f is not above BLOCK_OPERATING %.2f, so "
                         "some twin would NOT be refused at the operating threshold and C2(a) "
                         "cannot distinguish coverage detection from a gate that simply admits "
                         "everything. Lower TWIN_EPS." % (geom["min_twin_cosine"],
                                                          BLOCK_OPERATING))
    if not (geom["max_twin_cosine"] < BLOCK_HIGH):
        raise SystemExit("G3 FAILED: max twin cosine %.4f is not below BLOCK_HIGH %.2f, so the "
                         "high-block arm would still refuse some twin and C2(b) is unmeetable. "
                         "Raise BLOCK_HIGH or raise TWIN_EPS." % (geom["max_twin_cosine"],
                                                                  BLOCK_HIGH))
    if not (geom["max_cross_cluster_cosine"] < BLOCK_OPERATING):
        raise SystemExit("G3 FAILED: max cross-cluster cosine %.4f reaches BLOCK_OPERATING "
                         "%.2f, so the operating gate would also block distinct clusters and "
                         "C2(a) could not distinguish coverage from false-blocking."
                         % (geom["max_cross_cluster_cosine"], BLOCK_OPERATING))
    print("[gate] clusters reaching each threshold: %s" % reach, flush=True)
    print("[gate] geometry: twin cos %.4f-%.4f sandwiched in (BLOCK_OPERATING %.2f, "
          "BLOCK_HIGH %.2f); cross-cluster max %.4f < %.2f"
          % (geom["min_twin_cosine"], geom["max_twin_cosine"], BLOCK_OPERATING, BLOCK_HIGH,
             geom["max_cross_cluster_cosine"], BLOCK_OPERATING), flush=True)

    arms: List[Tuple[str, int, float]] = [
        (_arm_thresh(THRESH_COUNTER_BLIND), THRESH_COUNTER_BLIND, BLOCK_OPERATING),
    ]
    arms += [(_arm_thresh(th), th, BLOCK_OPERATING) for th in THRESH_SWEEP]
    arms += [(_arm_block_high(), THRESH_DEFAULT, BLOCK_HIGH)]
    total = len(arms) * n_seeds

    rows: List[Dict[str, Any]] = []
    for arm, th, bl in arms:
        for sd in seeds:
            print("Seed %d Condition %s" % (sd, arm))
            slice_ = config_slice(mint_recurrence_threshold=th, mature_mint_block_threshold=bl,
                                  n_slots=N_SLOTS, counts=counts, warmup=warm)
            with arm_cell(sd, config_slice=slice_, script_path=Path(__file__),
                          config_slice_declared=True,
                          # the ecology is driver-resident -> a cross-driver reuse would be a
                          # false-HIT, which corrupts a conclusion
                          include_driver_script_in_hash=True) as cell:
                row = _run_field(seed=sd, mint_recurrence_threshold=th,
                                 mature_mint_block_threshold=bl, n_slots=N_SLOTS,
                                 counts=counts, warmup=warm)
                row["arm"] = arm
                cell.stamp(row)
            print("verdict: %s" % ("PASS" if row["crf_n_minted_total"] >= 2 else "FAIL"),
                  flush=True)
            rows.append(row)
            print("  [cell] %s seed=%d done (%d/%d)" % (arm, sd, len(rows), total), flush=True)

    ref_arm = _arm_thresh(THRESH_DEFAULT)
    blind_arm = _arm_thresh(THRESH_COUNTER_BLIND)
    high_arm = _arm_block_high()
    base_rows = [r for r in rows if r["arm"] == ref_arm]
    operating_rows = [r for r in rows if r["mature_mint_block_threshold"] == BLOCK_OPERATING]

    # ---------------- preconditions ----------------------------------------
    a_ok = sum(1 for r in base_rows if r["crf_n_minted_total"] >= PRECOND_A_MIN_MINTED)
    a_req = math.ceil(PRECOND_A_SEED_FRACTION * n_seeds)
    worst_mult, worst_mult_cell = _worst_cell(operating_rows, "max_cluster_multiplicity", "max")
    worst_sat, worst_sat_cell = _worst_cell(rows, "crf_live_over_slots", "max")

    preconditions = [
        {"name": "mint_fires", "kind": "readiness",
         "description": "MECH-349 (a): seeds on the reference arm with >= 2 mints",
         "measured": float(a_ok), "threshold": float(a_req), "direction": "lower",
         "control": "ecology affords %d clusters, %d above the reference threshold"
                    % (n_clusters, reach[THRESH_DEFAULT]),
         "met": bool(a_ok >= a_req)},
        {"name": "no_cluster_minted_twice", "kind": "readiness",
         "description": ("MECH-349 (b): at the OPERATING block threshold no cluster may mint "
                         "more than once -- the ecology-independent duplicate test (a cluster "
                         "minting twice IS the vacuous-positive signature). Worst cell reported."),
         "measured": float(worst_mult), "threshold": 2.0, "direction": "upper",
         "comparator": "<", "offending_cell": worst_mult_cell,
         "control": "each cluster presents a base and a near-twin; only one may survive the gate",
         "met": bool(_finite(worst_mult) and worst_mult < 2.0)},
        {"name": "pool_does_not_saturate", "kind": "readiness",
         "description": "MECH-349 (c): crf_live_rules_final / crf_n_slots stays below 1.0",
         "measured": float(worst_sat), "threshold": 1.0, "direction": "upper",
         "comparator": "<", "offending_cell": worst_sat_cell,
         "control": "n_slots=%d against %d regimes" % (N_SLOTS, 2 * n_clusters),
         "met": bool(_finite(worst_sat) and worst_sat < 1.0)},
        {"name": "arc062_seed_leg_not_under_test", "kind": "scope",
         "description": ("MECH-349 (d): no arc062_seed is supplied at any call site, so the "
                         "top-down leg is UNTESTED and must not be read as a null"),
         "measured": 0.0, "threshold": 0.0, "direction": "upper",
         "control": "arc062_seed argument omitted at every step() call site", "met": True},
    ]

    # ---------------- C1 ----------------------------------------------------
    blind_by_seed = _by_seed(rows, blind_arm, "crf_n_minted_total")
    thr_by_seed = {th: _by_seed(rows, _arm_thresh(th), "crf_n_minted_total")
                   for th in THRESH_SWEEP}
    c1a_seed = {sd: bool(blind_by_seed.get(sd, 0) > thr_by_seed[THRESH_DEFAULT].get(sd, 0))
                for sd in seeds}
    c1a_n = sum(1 for v in c1a_seed.values() if v)
    c1a_pass = c1a_n >= min(C1_SEEDS_REQUIRED, n_seeds)
    c1b_seed = {}
    for sd in seeds:
        seq = [thr_by_seed[th].get(sd, 0) for th in THRESH_SWEEP]
        c1b_seed[sd] = bool(all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
                            and (seq[0] - seq[-1]) >= C1_MIN_DROP)
    c1b_n = sum(1 for v in c1b_seed.values() if v)
    c1b_pass = c1b_n >= min(C1_SEEDS_REQUIRED, n_seeds)
    c1_pass = bool(c1a_pass and c1b_pass)

    # ---------------- C2 ----------------------------------------------------
    op_by_seed = _by_seed(rows, ref_arm, "crf_n_minted_total")
    op_clusters = _by_seed(rows, ref_arm, "distinct_clusters_minted")
    high_by_seed = _by_seed(rows, high_arm, "crf_n_minted_total")
    c2a_seed = {sd: bool(op_by_seed.get(sd, -1) == op_clusters.get(sd, -2)) for sd in seeds}
    c2a_n = sum(1 for v in c2a_seed.values() if v)
    c2a_pass = c2a_n >= min(C2_SEEDS_REQUIRED, n_seeds)
    c2b_seed = {sd: bool(high_by_seed.get(sd, 0) - op_by_seed.get(sd, 0) >= C2_MIN_ADMIT)
                for sd in seeds}
    c2b_n = sum(1 for v in c2b_seed.values() if v)
    c2b_pass = c2b_n >= min(C2_SEEDS_REQUIRED, n_seeds)
    c2_pass = bool(c2a_pass and c2b_pass)

    load_bearing_pass = bool(c1_pass and c2_pass)
    preconditions_met = all(p["met"] for p in preconditions)
    outcome = "PASS" if (load_bearing_pass and preconditions_met) else "FAIL"

    criteria = [
        {"name": "C1_recurrence_gating", "load_bearing": True, "passed": c1_pass,
         "measured": float(min(c1a_n, c1b_n)),
         "threshold": float(min(C1_SEEDS_REQUIRED, n_seeds)),
         "units": "seeds satisfying BOTH conjuncts (worst of the two reported)",
         "conjunct_a_counter_is_read": {
             "passed": c1a_pass, "measured": float(c1a_n),
             "threshold": float(min(C1_SEEDS_REQUIRED, n_seeds)), "per_seed": c1a_seed,
             "units": "seeds where the counter-blind threshold mints strictly more than the "
                      "reference threshold (a counter-ignoring substrate ties and fails)",
             "minted_counter_blind": blind_by_seed, "minted_reference": op_by_seed},
         "conjunct_b_dose_response": {
             "passed": c1b_pass, "measured": float(c1b_n),
             "threshold": float(min(C1_SEEDS_REQUIRED, n_seeds)), "per_seed": c1b_seed,
             "min_drop_required": C1_MIN_DROP,
             "mint_by_threshold": {str(th): thr_by_seed[th] for th in THRESH_SWEEP}},
         "tests_claim_clause": "(i) recurs at least `threshold` times"},
        {"name": "C2_novelty_gating", "load_bearing": True, "passed": c2_pass,
         "measured": float(min(c2a_n, c2b_n)),
         "threshold": float(min(C2_SEEDS_REQUIRED, n_seeds)),
         "units": "seeds satisfying BOTH conjuncts (worst of the two reported)",
         "conjunct_a_covered_is_refused": {
             "passed": c2a_pass, "measured": float(c2a_n),
             "threshold": float(min(C2_SEEDS_REQUIRED, n_seeds)), "per_seed": c2a_seed,
             "minted_at_operating_block": op_by_seed,
             "distinct_clusters_at_operating_block": op_clusters,
             "units": "seeds where every near-twin was refused (mints == distinct clusters)"},
         "conjunct_b_novel_is_admitted": {
             "passed": c2b_pass, "measured": float(c2b_n),
             "threshold": float(min(C2_SEEDS_REQUIRED, n_seeds)), "per_seed": c2b_seed,
             "min_extra_admitted_required": C2_MIN_ADMIT,
             "minted_at_high_block": high_by_seed,
             "units": "seeds where raising the block above the twin cosine admitted at least "
                      "C2_MIN_ADMIT additional rules"},
         "tests_claim_clause": "(ii) not already covered by an existing rule's context_tag"},
    ]

    combination_rule = ("PASS iff ALL preconditions met AND C1 AND C2. Each criterion is itself "
                        "a two-sided conjunction: C1 requires the counter to be read at all (a) "
                        "AND a dose response to it (b); C2 requires a covered regularity to be "
                        "REFUSED at the operating threshold (a) AND the same regularity to be "
                        "ADMITTED once the threshold rises above its cosine (b). Every arm is "
                        "scored; nothing is reported-only.")

    criteria_non_degenerate = {
        # keyed ACROSS ARMS WITHIN A SEED -- a set pooled over arms and seeds is satisfied by
        # seed spread alone and cannot detect a flat sweep
        "C1": bool(any(len({thr_by_seed[th].get(sd) for th in THRESH_SWEEP}
                           | {blind_by_seed.get(sd)}) > 1 for sd in seeds)),
        "C2": bool(any(op_by_seed.get(sd) != high_by_seed.get(sd) for sd in seeds)),
    }

    if not preconditions_met:
        label = "substrate_not_ready_requeue"
        summary = "A MECH-349 non-degeneracy precondition was not met; no verdict is admissible."
        direction = "non_contributory"
        # MECH-349's own governance ruling forbids attaching an uninterpretable row to this
        # claim, so a not-ready run carries NO claim tag at all.
        claim_ids_out: List[str] = []
    elif load_bearing_pass:
        label = "crf_create_face_trigger_confirmed"
        summary = ("Both asserted clauses hold two-sidedly: the recurrence counter is read and "
                   "mint count falls across a 4x threshold sweep (C1), and a near-twin "
                   "regularity is REFUSED at the operating block threshold yet ADMITTED once "
                   "that threshold rises above its cosine (C2). Distinctness and non-churn were "
                   "NOT measured -- both are structural in this configuration.")
        direction = "supports"
        claim_ids_out = list(CLAIM_IDS)
    else:
        failed = [c["name"] for c in criteria if not c["passed"]]
        label = "crf_create_face_trigger_not_confirmed"
        summary = ("Load-bearing criteria failed under met preconditions and satisfied design "
                   "guards: %s. C1 failing is MECH-349 FALSIFYING(1); C2 failing is its "
                   "inert-novelty-clause pole." % ", ".join(failed))
        direction = "weakens"
        claim_ids_out = list(CLAIM_IDS)

    def _flat(v: Any) -> Optional[float]:
        if isinstance(v, bool):
            return float(int(v))
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
        return None

    _readout_raw: Dict[str, Any] = {
        "c1_passed": c1_pass, "c1a_seeds": c1a_n, "c1b_seeds": c1b_n,
        "c1_seeds_required": min(C1_SEEDS_REQUIRED, n_seeds), "c1_min_drop": C1_MIN_DROP,
        "c2_passed": c2_pass, "c2a_seeds": c2a_n, "c2b_seeds": c2b_n,
        "c2_seeds_required": min(C2_SEEDS_REQUIRED, n_seeds), "c2_min_admit": C2_MIN_ADMIT,
        "mean_minted_counter_blind": statistics.fmean(blind_by_seed.values()) if blind_by_seed else 0.0,
        "mean_minted_operating": statistics.fmean(op_by_seed.values()) if op_by_seed else 0.0,
        "mean_minted_high_block": statistics.fmean(high_by_seed.values()) if high_by_seed else 0.0,
        "mean_minted_max_threshold": (statistics.fmean(thr_by_seed[THRESH_SWEEP[-1]].values())
                                      if thr_by_seed[THRESH_SWEEP[-1]] else 0.0),
        "worst_cluster_multiplicity": worst_mult,
        "worst_live_over_slots": worst_sat,
        "min_twin_cosine": geom["min_twin_cosine"],
        "max_cross_cluster_cosine": geom["max_cross_cluster_cosine"],
        "block_operating": BLOCK_OPERATING, "block_high": BLOCK_HIGH,
        "n_clusters": n_clusters,
        "total_retired_all_cells": sum(r["crf_n_retired_total"] for r in rows),
        "preconditions_met": preconditions_met, "load_bearing_pass": load_bearing_pass,
        "n_seeds": n_seeds,
    }
    readout = {k: v for k, v in ((k, _flat(v)) for k, v in _readout_raw.items()) if v is not None}

    full_config = {
        "context_dim": CONTEXT_DIM, "n_clusters": n_clusters, "cluster_counts": list(counts),
        "twin_eps": TWIN_EPS, "regime_magnitude": REGIME_MAGNITUDE,
        "common_mode_scale": COMMON_MODE_SCALE, "obs_noise": OBS_NOISE,
        "baseline_warmup_ticks": warm, "n_slots": N_SLOTS, "rule_dim": RULE_DIM,
        "threshold_counter_blind": THRESH_COUNTER_BLIND, "threshold_sweep": THRESH_SWEEP,
        "block_operating": BLOCK_OPERATING, "block_high": BLOCK_HIGH,
        "outcome_signal": OUTCOME_SIGNAL, "cue_centering": True,
        "mature_pool_dynamics": True, "availability_maintenance": True,
        "seed_from_arc062": False, "seeds": seeds,
        "design_guards": {"clusters_reaching_threshold": reach, "geometry": geom},
    }

    manifest: Dict[str, Any] = {
        "run_id": make_run_id(EXPERIMENT_TYPE),
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": claim_ids_out,
        "claim_ids_intended": list(CLAIM_IDS),
        "related_exq": RELATED_EXQ,
        "outcome": outcome,
        "evidence_direction": direction,
        "sleep_driver_pattern": "not_applicable_no_sleep_loop",
        "readout": readout,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "interpretation": {
            "label": label, "summary": summary, "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "design_guards_passed": {"G1_thresholds_reachable": reach,
                                     "G2_prediction_drop": drop,
                                     "G3_geometry": geom},
            "point_prediction": {
                "rule": "#{c : CLUSTER_COUNTS[c] >= threshold}",
                "predicted": {str(th): reach[th] for th in THRESH_SWEEP},
                "observed": {str(th): thr_by_seed[th] for th in THRESH_SWEEP},
                "note": ("RECORDED, NOT SCORED: the sign-bucket can fragment a cluster's "
                         "occurrences across recurrence keys, so the prediction is an upper "
                         "bound rather than an identity."),
            },
            "structural_facts_not_measurements": {
                "recurrence_at_mint_is_a_tautology": (
                    "n_mints_below_threshold is 0 everywhere and CANNOT be otherwise: "
                    "_maybe_mint increments the recurrence counter and then early-returns when "
                    "it is below the bar, so a sub-threshold mint is unreachable for any "
                    "substrate of that shape. Recorded, never scored. See GFLAG-0266."),
                "distinctness_is_an_identity": (
                    "crf_max_pairwise_rule_dist is a deterministic lookup on live-rule count off "
                    "the pinned_seed=6063 matrix. MECH-350 is NOT tagged."),
                "retirement_unreachable": (
                    "crf_n_retired_total is 0 throughout, so MECH-349 FALSIFYING(3)'s churn "
                    "signature is UNREACHABLE here. A rule that stops activating freezes rather "
                    "than decaying, because credit() only updates it while its eligibility trace "
                    "is alive and eligibility is set only when the rule is ACTIVE. Owed "
                    "separately: chip-20260911-mech349-churn-leg-maintenance-floor."),
            },
        },
        "claim_scope_note": (
            "MECH-350/351/352 are NOT tagged: distinctness is a structural identity here, and "
            "neither conflict nor credit is manipulated. Their diagnostics ARE recorded per cell "
            "so a future targeted run need not re-derive them."
        ),
        "arm_results": rows,
        "cell_summary": {
            arm: {"minted_by_seed": _by_seed(rows, arm, "crf_n_minted_total"),
                  "distinct_clusters_by_seed": _by_seed(rows, arm, "distinct_clusters_minted"),
                  "max_cluster_multiplicity_by_seed": _by_seed(rows, arm,
                                                               "max_cluster_multiplicity")}
            for arm in sorted({r["arm"] for r in rows})
        },
    }

    out_path = write_flat_manifest(manifest, dry_run=dry_run, config=full_config, seeds=seeds,
                                   script_path=Path(__file__), started_at=t0)
    manifest["_out_path"] = out_path
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-1024 MECH-349 CandidateRuleField mint-trigger diagnostics")
    parser.add_argument("--dry-run", action="store_true",
                        help="1 seed, full ecology, short warmup; manifest out of evidence/")
    args = parser.parse_args()

    result = main(dry_run=args.dry_run)
    out_path = result["_out_path"]
    print()
    print("=== V3-EXQ-1024 MECH-349 CRF mint-trigger diagnostics ===")
    print("label:   %s" % result["interpretation"]["label"])
    print("outcome: %s" % result["outcome"])
    print("summary: %s" % result["interpretation"]["summary"])
    print("--- preconditions ---")
    for p in result["interpretation"]["preconditions"]:
        print("  %-30s measured=%-9.3f thr=%-7.3f met=%s"
              % (p["name"], p["measured"], p["threshold"], p["met"]))
    print("--- criteria (each a two-sided conjunction) ---")
    for c in result["criteria"]:
        print("  %-22s [LOAD-BEARING] passed=%s" % (c["name"], c["passed"]))
        for key in ("conjunct_a_counter_is_read", "conjunct_a_covered_is_refused",
                    "conjunct_b_dose_response", "conjunct_b_novel_is_admitted"):
            if key in c:
                print("      %-34s measured=%-6.1f thr=%-5.1f passed=%s"
                      % (key, c[key]["measured"], c[key]["threshold"], c[key]["passed"]))
    print("--- mint count by arm x seed ---")
    for arm, c in result["cell_summary"].items():
        vals = " ".join("s%d=%s" % (s, v) for s, v in sorted(c["minted_by_seed"].items()))
        print("  %-16s %s" % (arm, vals))
    print("manifest: %s" % out_path)
    print("overall_outcome: %s" % result["outcome"])

    _o = str(result["outcome"]).upper()
    emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                 manifest_path=out_path, dry_run=args.dry_run)
