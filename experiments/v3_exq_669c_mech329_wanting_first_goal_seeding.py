"""V3-EXQ-669c -- MECH-329 wanting-before-liking goal-seeding ordering test,
re-issued on the SD-077 centered super-ordinal cue key.

SUPERSEDES V3-EXQ-669b (v3_exq_669b_mech329_wanting_first_goal_seeding, run
2026-06-13, outcome FAIL / non_contributory), which self-routed on its own
pre-registered readiness gate R3 (max anchor_count across all arms/seeds >= 2):
observed anchor_count = 1 EVERYWHERE, so the load-bearing C1 (anchor_count delta)
and C3 (p01 complexity delta) had zero cross-arm range and could not
differentiate. That was correctly adjudicated a substrate-readiness limit, NOT a
falsification of MECH-329 / MECH-189.

THE SCIENCE IS UNCHANGED FROM 669b. Arms, schedule, forced-feed nursery,
pre-registered thresholds, readiness gates and routing are all identical. The ONLY
substantive change is two config values in _build_agent, enabled by the SD-077
substrate landing:
    super_ordinal_cue_centering = True    (SD-077 -- the actual fix)
    super_ordinal_n_slots        = 64     (was 16 -- see WHY 64 below)

WHY 669b SATURATED (measured 2026-07-21 on this exact nursery, 155 contexts,
seed 101 -- session scientific-dashboard-status-7d2d08):
  raw world_obs pairwise cosine  min 0.2164 / mean 0.6084, 90.7% of pairs < 0.8
  z_world      pairwise cosine   min 0.9641 / mean 0.9898, 0.0% of pairs < 0.8
  ||mean(z_world)|| / mean||z_world|| = 0.9949
The nursery is NOT context-poor -- it is richly diverse in the observation, and the
untrained z_world encoder buries that diversity under a dominant common-mode
offset (SD-008 under-differentiation). So the store's raw-z_world cue cosine
measured the shared offset, not the context: 1 allocation + 159 reinforcements into
slot 0.

WHY A THRESHOLD CHANGE WOULD NOT HAVE WORKED (and why 669b's own docstring advice
must NOT be followed): contextual_complexity = 1 - best_cosine, and best_cosine
>= 0.9641 everywhere, so complexity <= 0.036 under ANY (merge_similarity,
complexity_threshold) pair -- strictly below this experiment's pre-registered
COMPLEXITY_MARGIN of 0.05 (669b's measured mean over 160 fired writes: 0.0077).
C3 was UNSATISFIABLE BY CONSTRUCTION on a raw-z_world key. 669b's docstring
proposed a LOWER merge_similarity; that is the WRONG SIGN -- it moves more contexts
into the REINFORCE branch and makes saturation strictly worse. Only changing the
KEY SPACE helps, which is what SD-077 does (slow EMA common-mode baseline, cue
cosine taken on the centered residual z_world - baseline; the SD-066 pattern).

WHY n_slots = 64: with centering ON, this nursery allocates ~26 distinct anchors
over the run. 669b's n_slots=16 would CAP the bank and re-flatten C1 by saturating
every arm at the same ceiling -- a different degeneracy with the same symptom. 64
leaves headroom so anchor_count is free to vary between arms.

Measured with centering ON, same nursery / same 160 writes, n_slots=64 at 669b's
own thresholds (merge 0.8, cthr 0.2): 26 anchors, mean complexity 0.076. R3 clears
and C3's margin becomes reachable. Substrate record: SD-077
(REE_assembly/docs/architecture/sd_077_centered_super_ordinal_cue_key.md).

MECH-329 (the ordering claim): the wanting system (z_goal seeding / approach drive)
seeds super-ordinal goal anchors via accidental benefit contacts BEFORE the liking
system (benefit_eval / hedonic calibration) is calibrated. This is an ORDERING
claim -- it predicts a DIFFERENCE in the timing/quantity/complexity of super-ordinal
writes between a wanting-first and a liking-first developmental schedule.

Operationalisation on the MECH-189 write substrate: the super-ordinal WRITE is
driven by the wanting system (update_z_goal forms z_goal and writes the anchor when
salience clears threshold). The liking head (benefit_eval) exists in every arm; the
ordering lever is WHEN the wanting drive is active across the nursery curriculum.
Writes are enabled across the whole nursery (child + wean windows) and frozen only
at the end -- so the both_delayed control HAS a writable window once its wanting
drive turns on (the satisfiable positive control 669a lacked).

Three arms (matched seeds; agent built ONCE per arm, NO mid-run rebuild):
  A wanting_first -- wanting drive ON in BOTH the child and wean windows
                     (wanting active from the start of development).
  B liking_first  -- wanting drive DELAYED to the wean window only
                     (liking calibrates first; wanting seeds late).
  C both_delayed  -- wanting drive DELAYED to the wean window only; the
                     positive/sanity control: when wanting finally turns on, do
                     writes fire at all?

  "wanting ON in a window"  = update_z_goal(FORCED_BENEFIT, FORCED_DRIVE) per step
                              -> z_goal forms -> super-ordinal write fires.
  "wanting OFF in a window" = update_z_goal(0.0, 0.0) per step
                              -> no seeding, no super-ordinal write.

DV-SYMMETRY DECLARATION (one line per arm, per the Step 3.5 design audit). All
three arms share one DV family: anchor_count (a cardinality over allocated slots)
and p01_mean_complexity (a mean of 1 - best_cosine over fired writes). The
manipulation is the TEMPORAL SCHEDULE of the wanting drive, which changes WHICH
contexts reach the write path and HOW MANY writes fire.
  - wanting_first: not invariant. Its child window fires writes that liking_first's
    does not, so the set of contexts entering the anchor bank differs -- neither a
    broadcast constant, a monotone rescaling, nor a permutation of interchangeable
    units. anchor_count is a cardinality over a genuinely different input set.
  - liking_first: not invariant, same argument with the child window empty.
  - both_delayed: not invariant; it is the positive control and its whole function
    is that turning the drive on changes the write count from 0.
None of the three manipulations is a uniform additive constant over candidates, a
monotone transform of a rank-based DV, or a permutation of a set-aggregate's
inputs, so no arm's delta is an arithmetic identity fixed before the run.

PRE-REGISTERED NON-VACUITY READINESS GATE (the exact discipline V3-EXQ-588c used,
and the exact defect class that made 669a vacuous). ALL THREE preconditions must
hold before the load-bearing ordering criteria are scored; any failure self-routes
substrate_not_ready_requeue (outcome FAIL, evidence_direction unknown,
non_degenerate=False) -- NEVER a weakens FAIL:
  R1 (the user-mandated gate): the both_delayed positive control MUST produce
     anchor_count >= 1 AND total_writes >= 1 (writes fire at all once the substrate
     is fed -- the exact 669a failure).
  R2 (manipulation check): the wanting-schedule manipulation took --
     wanting_first p01_writes >= liking_first p01_writes + WRITE_MARGIN(3) on
     >= 2/3 seeds. (This is near-tautological by the schedule -- wanting_first
     writes in the child window by construction, liking_first does not -- so it is
     a PRECONDITION that the manipulation engaged, NOT a scored scientific result.)
  R3 (anchor-hierarchy non-degeneracy -- the SAME-STATISTIC guard the load-bearing
     criteria route on): max anchor_count across all arms/seeds >= 2. This is the
     gate 669b failed. It remains armed: if it fails AGAIN with SD-077 centering
     enabled, that is a NEW and informative fact (the centered key did not separate
     this nursery either), not a MECH-329 weakens.

LOAD-BEARING ACCEPTANCE (the genuine MECH-329 consequence; scored ONLY when R1-R3
hold). The ordering manipulation is "when is the wanting drive active"; the
non-trivial prediction is that EARLY wanting builds a RICHER super-ordinal goal
hierarchy than late wanting:
  C1: wanting_first anchor_count    >= liking_first anchor_count    + ANCHOR_MARGIN(2)
      (wanting-early forms more distinct super-ordinal anchors).
  C3: wanting_first p01_complexity  >= both_delayed p01_complexity  + COMPLEXITY_MARGIN(0.05)
      (wanting-driven child contacts span more novel contexts).
  PASS (supports MECH-329 ordering) = C1 AND C3 on >= 2/3 seeds. R1-R3 met but
  C1/C3 not -> MECH-329 weakens (a genuine null: the substrate forms a
  differentiable hierarchy and the manipulation took, yet early-wanting did NOT
  enrich it); MECH-189 still supported (the write+read substrate fired).

A trained z_world encoder is NOT required: z_world is a deterministic context cue
(same obs -> same z_world), so the contextual-complexity novelty proxy and the
anchor keys are well-defined; the substrate-readiness question is anchor formation
+ write timing, not encoder fidelity (mirrors V3-EXQ-588c). SD-077 is precisely
what makes that hold WITHOUT a trained encoder -- it removes the common-mode
offset that an untrained encoder imposes, rather than trying to train it away
(which SD-070 measured to collapse z_world outright).
"""
from __future__ import annotations

import sys
import json
import math
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_669c_mech329_wanting_first_goal_seeding"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-329", "MECH-189"]

# Pre-registered thresholds (defined here, NOT inferred post-hoc).
ANCHOR_MARGIN = 2          # C1: wanting_first anchors >= liking_first + this
WRITE_MARGIN = 3           # C2: wanting_first p01 writes >= liking_first + this
COMPLEXITY_MARGIN = 0.05   # C3: wanting_first p01 complexity >= both_delayed + this
SEED_PASS_FRACTION = 2.0 / 3.0  # >= 2/3 seeds must pass each criterion
FORCED_BENEFIT = 0.5       # nursery forced-feed benefit (salience 0.5*(1+2*0.9)=1.4 >> 0.5)
FORCED_DRIVE = 0.9

# Per-arm wanting-drive schedule across the two writable nursery windows.
# (child window, wean window) -> wanting drive active?
ARM_SCHEDULE = {
    "wanting_first": {"child": True, "wean": True},
    "liking_first": {"child": False, "wean": True},
    "both_delayed": {"child": False, "wean": True},
}
ARMS = ["wanting_first", "liking_first", "both_delayed"]


def _build_nursery_env(seed: int) -> CausalGridWorldV2:
    """Stage-0 forced-feed nursery: dense resources, hazard-free safe context
    (mirrors the scaffolded_sd054_onboarding run_stage0_nursery memo:
    num_resources=6, num_hazards=0). Decouples the test from goal_pipeline:GAP-2
    (the ecological foraging-contact ceiling that starved 669a)."""
    return CausalGridWorldV2(
        size=8,
        num_hazards=0,
        num_resources=6,
        use_proxy_fields=True,
        seed=seed,
    )


def _build_agent(env: CausalGridWorldV2) -> REEAgent:
    """One agent per arm. z_goal_enabled + drive_weight=2.0 are the two-part-fix
    precondition (without them update_z_goal early-returns / the SD-012 multiplier
    is absent); benefit_eval head exists in every arm (the liking system); the
    arm ordering lever is the forced wanting-drive schedule, NOT a rebuild."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        z_goal_enabled=True,
        drive_weight=2.0,
        alpha_world=0.9,  # SD-008
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        use_super_ordinal_goal_anchors=True,
        # SD-077: the fix. Cue cosine is taken on the centered residual
        # (z_world - slow-EMA common-mode baseline) instead of raw z_world, which
        # under SD-008 under-differentiation measured the shared offset rather than
        # the context and saturated 669b's anchor bank at 1.
        super_ordinal_cue_centering=True,
        # Raised from 669b's 16: with centering ON this nursery allocates ~26
        # anchors, so 16 would CAP the bank and re-flatten C1 by saturating every
        # arm at the same ceiling.
        super_ordinal_n_slots=64,
        super_ordinal_salience_threshold=0.5,
        super_ordinal_complexity_mode="novelty",
        super_ordinal_complexity_threshold=0.2,
        super_ordinal_merge_similarity=0.8,
        super_ordinal_write_alpha=0.3,
    )
    return REEAgent(cfg)


def _world_dim(agent: REEAgent) -> int:
    return agent.config.latent.world_dim


def _nursery_episode(agent: REEAgent, env: CausalGridWorldV2, steps: int,
                     wanting_active: bool, complexity_sink: List[float]) -> int:
    """One forced-feed nursery episode. When wanting_active, drives forced
    supra-threshold benefit each step (super-ordinal writes fire); otherwise drives
    a null update (liking-only window, no write). Returns the number of
    super-ordinal writes that fired this episode; appends the contextual-complexity
    of each fired write to complexity_sink."""
    _, obs_dict = env.reset()
    agent.reset()  # per-episode reset -- does NOT clear the super-ordinal store
    wd = _world_dim(agent)
    som = agent.super_ordinal_goal_memory
    writes_this_ep = 0
    for _ in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        latent = agent.sense(obs_body, obs_world)
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent) if ticks.get("e1_tick")
            else torch.zeros(1, wd, device=agent.device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        action = agent.select_action(candidates, ticks)
        action_idx = int(action.argmax(dim=-1).item())

        w0 = som._n_writes
        if wanting_active:
            # FORCED high-salience benefit -> z_goal seeds -> super-ordinal write.
            agent.update_z_goal(benefit_exposure=FORCED_BENEFIT, drive_level=FORCED_DRIVE)
        else:
            # Liking-only / pre-wanting window: no wanting drive, no write.
            agent.update_z_goal(benefit_exposure=0.0, drive_level=0.0)
        if som._n_writes > w0:
            writes_this_ep += 1
            complexity_sink.append(float(som._last_complexity))

        _, _harm, done, _, obs_dict = env.step(action_idx)
        if done:
            break
    return writes_this_ep


def _run_seed_arm(arm: str, seed: int, n_child: int, n_wean: int,
                  steps: int) -> Dict[str, Any]:
    sched = ARM_SCHEDULE[arm]
    print(f"Seed {seed} Condition {arm}", flush=True)
    full_config = {
        "arm": arm, "seed": seed, "n_child": n_child, "n_wean": n_wean,
        "steps": steps, "forced_benefit": FORCED_BENEFIT,
        "forced_drive": FORCED_DRIVE, "schedule": sched,
    }
    total_eps = n_child + n_wean
    with arm_cell(seed, config_slice=full_config, script_path=Path(__file__)) as cell:
        torch.manual_seed(seed)
        env = _build_nursery_env(seed)
        agent = _build_agent(env)
        som = agent.super_ordinal_goal_memory

        # CHILD window (writes enabled).
        p01_complexities: List[float] = []
        for ep in range(n_child):
            _nursery_episode(agent, env, steps, sched["child"], p01_complexities)
            if (ep + 1) % 2 == 0 or ep == n_child - 1:
                print(f"  [train] child seed={seed} arm={arm} "
                      f"ep {ep + 1}/{total_eps} writes={som._n_writes}", flush=True)
        # p01 = the child-window write counts (the store starts empty, so
        # _n_writes at child-end == writes that fired during the child window).
        p01_writes = int(som._n_writes)
        p01_mean_complexity = (
            sum(p01_complexities) / len(p01_complexities) if p01_complexities else 0.0
        )

        # WEAN window (still writable).
        wean_sink: List[float] = []  # not scored; isolates p01 to the child window
        for ep in range(n_wean):
            _nursery_episode(agent, env, steps, sched["wean"], wean_sink)
            print(f"  [train] wean seed={seed} arm={arm} "
                  f"ep {n_child + ep + 1}/{total_eps} writes={som._n_writes}", flush=True)

        # FREEZE at weaning (child->adult transition; defensive -- no further phase).
        agent.set_super_ordinal_write_enabled(False)

        anchor_count = int(som.n_occupied())
        total_writes = int(som._n_writes)

        # Per-cell progress verdict (rough; the load-bearing ordering criteria are
        # computed by pairing arms in run_experiment).
        cell_ok = total_writes > 0
        print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)

        row = {
            "arm": arm,
            "seed": seed,
            "anchor_count": anchor_count,
            "total_writes": total_writes,
            "p01_writes": p01_writes,
            "p01_mean_complexity": round(float(p01_mean_complexity), 6),
        }
        cell.stamp(row)
    return row


def run_experiment(n_child: int, n_wean: int, steps: int, seeds: List[int],
                   dry_run: bool) -> Dict[str, Any]:
    arm_results: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            arm_results.append(_run_seed_arm(arm, seed, n_child, n_wean, steps))

    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r in arm_results:
        by_seed.setdefault(r["seed"], {})[r["arm"]] = r

    n_seeds = len(seeds)
    # >= 2/3 threshold count: need at least ceil(SEED_PASS_FRACTION * n_seeds).
    seed_pass_n = math.ceil(SEED_PASS_FRACTION * n_seeds - 1e-9)

    # ---- READINESS GATE (pre-registered, scored FIRST; any failure -> requeue) ----
    # R1 (user-mandated): both_delayed positive control fires writes + forms an anchor.
    bd_rows = [by_seed[s]["both_delayed"] for s in seeds]
    bd_min_anchor = min(r["anchor_count"] for r in bd_rows)
    bd_min_writes = min(r["total_writes"] for r in bd_rows)
    r1_writes_fire = (bd_min_anchor >= 1) and (bd_min_writes >= 1)

    # R2 (manipulation check): the wanting schedule took (wanting_first wrote in the
    # child window, liking_first did not). Near-tautological -> a PRECONDITION, not a
    # scored result.
    def _c2(s) -> bool:
        return (by_seed[s]["wanting_first"]["p01_writes"]
                >= by_seed[s]["liking_first"]["p01_writes"] + WRITE_MARGIN)
    n_c2 = sum(1 for s in seeds if _c2(s))
    frac_c2 = n_c2 / float(n_seeds)
    r2_manip_took = n_c2 >= seed_pass_n

    # R3 (anchor-hierarchy non-degeneracy): the store forms >1 distinct anchor
    # somewhere, so anchor_count / complexity have cross-arm range for C1/C3 to route
    # on. max==1 everywhere == saturation == substrate not ready (NOT a weakens).
    max_anchor = max(r["anchor_count"] for r in arm_results)
    r3_hierarchy_non_degenerate = max_anchor >= 2

    readiness_met = r1_writes_fire and r2_manip_took and r3_hierarchy_non_degenerate

    # ---- LOAD-BEARING ordering criteria (the genuine MECH-329 consequence) ----
    def _c1(s) -> bool:
        return (by_seed[s]["wanting_first"]["anchor_count"]
                >= by_seed[s]["liking_first"]["anchor_count"] + ANCHOR_MARGIN)

    def _c3(s) -> bool:
        return (by_seed[s]["wanting_first"]["p01_mean_complexity"]
                >= by_seed[s]["both_delayed"]["p01_mean_complexity"] + COMPLEXITY_MARGIN)

    n_c1 = sum(1 for s in seeds if _c1(s))
    n_c3 = sum(1 for s in seeds if _c3(s))
    frac_c1 = n_c1 / float(n_seeds)
    frac_c3 = n_c3 / float(n_seeds)
    c1_pass = n_c1 >= seed_pass_n
    c2_pass = r2_manip_took  # reported; not load-bearing
    c3_pass = n_c3 >= seed_pass_n
    ordering_pass = c1_pass and c3_pass

    # Outcome / routing.
    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        overall_direction = "unknown"
        per_claim = {"MECH-329": "unknown", "MECH-189": "unknown"}
        non_degenerate = False
        if not r1_writes_fire:
            degeneracy_reason = (
                "R1 unmet: both_delayed positive control formed no anchors / fired no "
                f"writes (min anchor_count={bd_min_anchor}, min total_writes={bd_min_writes}) "
                "-- ordering would be measured over an empty set (the V3-EXQ-669a defect). "
                "Re-queue at adequate forced-feed budget."
            )
        elif not r2_manip_took:
            degeneracy_reason = (
                "R2 unmet: the wanting-schedule manipulation did not take "
                f"(C2 frac={round(frac_c2, 3)} < {round(SEED_PASS_FRACTION, 3)}); "
                "wanting_first did not out-write liking_first in the child window. "
                "Re-queue with a longer child window."
            )
        else:
            degeneracy_reason = (
                "R3 unmet: anchor store saturated to <=1 anchor "
                f"(max anchor_count={max_anchor}); merge_similarity collapses the nursery "
                "z_world contexts so anchor_count / complexity have zero cross-arm range "
                "and C1/C3 cannot differentiate. Substrate not ready. NOTE this run "
                "already has SD-077 centering ON, so the common-mode explanation is "
                "EXCLUDED -- do NOT re-queue with a lower super_ordinal_merge_similarity "
                "(that is the wrong sign: it moves more contexts into the REINFORCE "
                "branch and worsens saturation). Investigate the centered-residual "
                "geometry of this nursery directly before queueing another iteration."
            )
    elif ordering_pass:
        outcome = "PASS"
        label = "wanting_before_liking_ordering_supported"
        overall_direction = "supports"
        per_claim = {"MECH-329": "supports", "MECH-189": "supports"}
        non_degenerate = True
        degeneracy_reason = ""
    else:
        # R1-R3 met (substrate forms a differentiable hierarchy AND the manipulation
        # took) but early-wanting did not enrich it: a GENUINE null for the MECH-329
        # ordering prediction (weakens); the MECH-189 write+read substrate still fired
        # (supports).
        outcome = "FAIL"
        label = "wanting_before_liking_ordering_not_demonstrated"
        overall_direction = "mixed"
        per_claim = {"MECH-329": "weakens", "MECH-189": "supports"}
        non_degenerate = True
        degeneracy_reason = ""

    bd_positive_control = min(bd_min_anchor, bd_min_writes)

    result: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": "v3_exq_669b_mech329_wanting_first_goal_seeding",
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "outcome": outcome,
        "dry_run": dry_run,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "evidence_direction": overall_direction,
        "evidence_direction_per_claim": per_claim,
        "config": {
            "n_child": n_child, "n_wean": n_wean, "steps": steps, "seeds": seeds,
            "forced_benefit": FORCED_BENEFIT, "forced_drive": FORCED_DRIVE,
            "anchor_margin": ANCHOR_MARGIN, "write_margin": WRITE_MARGIN,
            "complexity_margin": COMPLEXITY_MARGIN,
            "seed_pass_fraction": SEED_PASS_FRACTION,
            "arm_schedule": ARM_SCHEDULE,
        },
        "metrics": {
            "readiness_met": readiness_met,
            "r1_writes_fire": r1_writes_fire,
            "r2_manip_took": r2_manip_took,
            "r3_hierarchy_non_degenerate": r3_hierarchy_non_degenerate,
            "both_delayed_min_anchor_count": bd_min_anchor,
            "both_delayed_min_total_writes": bd_min_writes,
            "max_anchor_count_across_arms": max_anchor,
            "frac_c1_anchor_count": round(frac_c1, 4),
            "frac_c2_p01_writes_manip_check": round(frac_c2, 4),
            "frac_c3_p01_complexity": round(frac_c3, 4),
            "c1_pass": c1_pass, "c2_manip_check_pass": c2_pass, "c3_pass": c3_pass,
            "seed_pass_n_required": seed_pass_n,
        },
        "arm_results": arm_results,
        "interpretation": {
            "label": label,
            "preconditions": [
                {
                    "name": "R1_both_delayed_positive_control_forms_anchors",
                    "description": "MANDATORY non-vacuity gate: the both_delayed positive "
                                   "control must produce anchor_count>=1 AND total_writes>=1 "
                                   "on every seed -- writes fire at all once fed (the exact "
                                   "V3-EXQ-669a defect). Below-floor -> substrate_not_ready_requeue.",
                    "measured": int(bd_positive_control),
                    "threshold": 1,
                    "direction": "lower",  # floor: met when measured >= threshold
                    "control": "both_delayed drives forced wanting in the wean window; "
                               "writes must fire when the substrate is fed",
                    "met": bool(r1_writes_fire),
                },
                {
                    "name": "R2_wanting_schedule_manipulation_took",
                    "description": "Manipulation check (near-tautological by the schedule, so a "
                                   "PRECONDITION not a scored result): wanting_first p01_writes "
                                   ">= liking_first + WRITE_MARGIN on >= 2/3 seeds, confirming the "
                                   "wanting-drive schedule engaged the child-window write path.",
                    "measured": int(n_c2),
                    "threshold": int(seed_pass_n),
                    "direction": "lower",  # floor: met when measured >= threshold
                    "control": "wanting_first child=ON vs liking_first child=OFF",
                    "met": bool(r2_manip_took),
                },
                {
                    "name": "R3_anchor_hierarchy_non_degenerate",
                    "description": "SAME-STATISTIC non-degeneracy guard the load-bearing C1/C3 "
                                   "route on: max anchor_count across all arms/seeds >= 2. If "
                                   "every arm saturates to a single anchor (merge_similarity "
                                   "collapses the nursery z_world contexts) anchor_count / "
                                   "complexity have ZERO cross-arm range and C1/C3 cannot "
                                   "differentiate -> substrate_not_ready_requeue, NOT a MECH-329 "
                                   "weakens.",
                    "measured": int(max_anchor),
                    "threshold": 2,
                    "direction": "lower",  # floor: met when measured >= threshold
                    "control": "max distinct anchors any arm forms under forced feed",
                    "met": bool(r3_hierarchy_non_degenerate),
                },
            ],
            "criteria_non_degenerate": {
                "C1": bool(r3_hierarchy_non_degenerate),
                "C3": bool(r3_hierarchy_non_degenerate),
            },
            "criteria": [
                {"name": "C1_wanting_more_anchors", "load_bearing": True, "passed": bool(c1_pass)},
                {"name": "C3_wanting_higher_p01_complexity", "load_bearing": True, "passed": bool(c3_pass)},
                {"name": "C2_wanting_more_p01_writes_MANIP_CHECK", "load_bearing": False, "passed": bool(c2_pass)},
            ],
            "evidence_direction": overall_direction,
        },
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        n_child, n_wean, steps, seeds = 2, 2, 20, [42]
    else:
        n_child, n_wean, steps, seeds = 8, 6, 100, [42, 43, 44]

    result = run_experiment(n_child, n_wean, steps, seeds, args.dry_run)

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        result,
        out_dir,
        dry_run=args.dry_run,
        config=result.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )

    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    m = result["metrics"]
    print(f"readiness_met: {m['readiness_met']} "
          f"(R1_writes_fire={m['r1_writes_fire']} R2_manip_took={m['r2_manip_took']} "
          f"R3_hierarchy_nondegen={m['r3_hierarchy_non_degenerate']}; "
          f"both_delayed min_anchor={m['both_delayed_min_anchor_count']} "
          f"min_writes={m['both_delayed_min_total_writes']} "
          f"max_anchor={m['max_anchor_count_across_arms']})", flush=True)
    print(f"C1(anchors)={m['c1_pass']} C3(complexity)={m['c3_pass']} "
          f"[C2 manip-check={m['c2_manip_check_pass']}] "
          f"(frac C1={m['frac_c1_anchor_count']} C3={m['frac_c3_p01_complexity']} "
          f"C2={m['frac_c2_p01_writes_manip_check']})", flush=True)
    print(f"wrote: {out_path}", flush=True)
    return result, out_path, args.dry_run


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
