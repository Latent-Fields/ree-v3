"""Append the V3-EXQ-861f/861g/861h GOV-FANOUT-1 portfolio to the ree-v3 queue.

Narrow structural append (load, append top-level items, write) -- CLAUDE.md
"Read-modify-write contamination" / Prevention. Re-reads the live file
immediately before writing and refuses on any id collision.
"""
import json
from pathlib import Path

Q = Path("/Users/dgolden/REE_Working/ree-v3/experiment_queue.json")

COMMON = (
 "GOV-FANOUT-1 PORTFOLIO LEG. Routed by confirmed "
 "failure_autopsy_V3-EXQ-861e_2026-08-21 (status confirmed, Step 7c red-team "
 "CONTESTED -> Step 8 interactive gate verdict 'adopt_redteam_portfolio'; "
 "routing 'queue-experiment'; re_derive_brake.fired false; refused_requeue "
 "false; recommended_substrate_queue_entry.action 'none'). Governance cycle "
 "gov-20260821-0203, applied REE_assembly 26891ec7fa. Hypothesis-space qid "
 "inv050_mech180_861e_producer_vs_intervention_isolation (Mode A, frozen set "
 "H1+H3 count 2, H2 a labelled follow-on). Node class: complex (probe-gated) / "
 "puzzle (known rules). "
 "| WHAT 861e ESTABLISHED: the CALIB_DRAWS 5->10 instrument repair WORKED (R3 "
 "3/3, SEM adequate) and the answer moved the WRONG way -- seed 271 "
 "ARM_3_HIGH_ON mean MEL fell from 2.8999705e-05 (861c, ABOVE its own "
 "reference, factor 1.2146) to 2.2980323e-05 (861e, BELOW it, factor 0.8845). "
 "The 861c n=10 projection is REFUTED at its premise: the numerator moved. "
 "Seed 271 is NOT 'like seed 7' (861c already HIGH-graded 271 at n=5; seed 7 "
 "was already below-reference then). "
 "| EXPERIMENT_PURPOSE = diagnostic. This leg is instrument isolation, NOT "
 "evidence: evidence_direction is pinned non_contributory and "
 "evidence_direction_per_claim to {INV-050: unknown, MECH-180: unknown} "
 "REGARDLESS of the replicated grid, which is still computed in full and "
 "recorded under replication_readout for comparability with 861c/861e. "
 "Neither claim's status, confidence or v3_pending may move on it. Do NOT bump "
 "CALIB_DRAWS again; do NOT stamp substrate_ceiling; do NOT amend or fork "
 "MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION (861d's, a different mechanism). "
 "| SUBSTRATE PIN (new, experiments/_lib/substrate_pin.py): ree_core is "
 "executed from a pinned historical commit via read-only `git archive` into a "
 "scratch dir first on sys.path; experiments/_lib, experiment_protocol and "
 "pack_writer still come from the live checkout. Proven by TWO fatal checks "
 "before any compute -- structural (ree_core.__file__ under the pin dir) and "
 "behavioural (ree_core.predictors.e3_selector.authority_spread_ratio, present "
 "at 17befb8c and absent at f810969). An unprovable pin raises and the runner "
 "classifies ERROR; there is no degraded mode. Per-cell arm_fingerprint uses "
 "repo_root=<pin dir> + substrate_scope=('ree_core/**/*.py',) so substrate_hash "
 "describes the code that ACTUALLY RAN; that scope is narrower than the default "
 "globs so every pinned cell is stamped reuse-INELIGIBLE "
 "(substrate_pinned_to_historical_commit) and must never be minted as a "
 "baseline. The pin also makes the leg immune to trunk drift between queue-time "
 "and run-time. "
 "| GOV-REUSE-1 (Step 2.4): decisive readout is seed 271 ARM_3_HIGH_ON mean_mel "
 "/ mean_duration_factor against that cell's own mel_reference under this leg's "
 "single varied condition. Checked 861c (f810969/n=5/no-reseed/legacy-write) and "
 "861e (17befb8c/n=10/no-reseed/legacy-write): every cell in this portfolio is a "
 "condition neither ran. NOT recoverable -> run. "
 "| RE-DERIVE BRAKE (Step 2.5b): examined and RELEASED. The literal run-keyed "
 "counter reads INV-050=7, MECH-180=5 (over threshold 2). Released on three "
 "independent grounds: (a) the skill's own 'Not braked' clause -- this is a "
 "`diagnostic` whose purpose is to discriminate WHY, not a lettered re-derive at "
 "the same granularity; (b) GOV-FANOUT-1 makes a diverse discrimination "
 "portfolio the PRESCRIBED route for a braked lineage carrying a "
 "fanout_recommendation, which is exactly what this autopsy emitted; (c) the "
 "autopsy's producer half stamped fired=false / route_to=queue-experiment and "
 "its category note says stamping ceiling 'would fire the re-derive brake and "
 "forbid the H1/H3 isolation the data now need'. Nearly all counted hits are "
 "standard/non_contributory targets counting only through the "
 "non_contributory-direction proxy the skill itself warns inverts the brake for "
 "instrument defects. "
 "| SUBSTRATE OVERLAP (Step 2.5c) -- a CORRUPTING overlap fired and is disclosed "
 "rather than waived: `contextmemory-write-path-addressing-degeneracy` "
 "(severity corrupting, status implemented_pending_validation, substrate_paths "
 "[ree_core/predictors/e1_deep.py]). MEASURED from both predecessors' recorded "
 "manifests, not assumed: seed 271 -- the seed the whole discrimination rests on "
 "-- is write-address LOCKED in BOTH runs on EVERY arm (861c per_cycle_n_touched "
 "~[1,1,1,1,2,1], 4-5/6 cycles insufficient; 861e ARM_3_HIGH_ON exactly "
 "[1,1,1,1,1,1], 6/6 insufficient, mean_sws_new_slot_diversity 0.0000), while "
 "seeds 7 and 883 rotate normally (3-14 slots) in both. Consequences, both "
 "pre-registered: (i) the lock is a CONSTANT across 861c and 861e so it cannot "
 "cause the DIFFERENCE -- H1 vs H3 stays well-posed; (ii) it is still a live "
 "corrupting condition on the decisive seed, which is why V3-EXQ-861h exists "
 "(the Step 2.5b(iv) adversarial-audit leg) -- it repeats the protocol with the "
 "already-built non-degenerate write selection ON, so the portfolio MEASURES the "
 "defect instead of inheriting it. 861f/861g deliberately keep the LEGACY argmin "
 "write path because changing it would destroy their single-variable isolation. "
 "Every cell records per_cycle_n_touched_slots + "
 "n_cycles_insufficient_touched_slots and the manifest carries a "
 "contextmemory_write_lock_audit block. Other overlaps, unchanged from 861e's own "
 "adjudication: MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION (corrupting) inert -- "
 "run_sws_schema_pass gates the whole relative_novelty consumption behind "
 "use_mech122_spindle_content_selection, False here; mode-governance-engagement "
 "(corrupting) unrelated subsystem, this driver never imports "
 "salience_coordinator.py or regime_occupancy_gate; SD-MECH267-CEM-SELECTION-FIX "
 "/ SD-MECH303-THRESHOLD-SOURCING / SD-SLEEP-ENTRY-PRESSURE / "
 "SD-E3-SCORER-COMPLETION / SD-ORIENTING-DECISION-SCALE (degrading) noted, "
 "arm-symmetric, not blocking. "
 "| ETHICS PREFLIGHT (2.6): all-false / decision allow (V3, no self-model, no "
 "negative-valence manipulation, no human data). "
 "| Step 2.5a probe: on the PINNED tree, all 43 REEConfig.from_dims kwargs this "
 "driver passes exist (so none is silently swallowed -- "
 "reference-reeconfig-from-dims-silent-kwargs), agent constructs, mel_consumer "
 "present, sleep force_cycle returns the MEL keys."
)

SMOKE = (" | Smoke PASS (--dry-run, rc=0): pin verified True, in-run control set "
         "and _discrimination() exercised on the DECISIVE seed (the smoke seed is "
         "DECISIVE_SEED, not SEEDS[0], so neither path's first populated "
         "execution is the multi-hour run), manifest relocated out of evidence/. "
         "validate_experiments.py --strict --paths OK (3 advisory "
         "anchor-reachability warnings, of which the leg's OWN new anchor is "
         "discharged by a setup-time assert_anchor_reachable against the frozen "
         "recorded predecessor cell using THE SHIPPED PREDICATE; the two "
         "remaining names are inherited verbatim from 861e). "
         "validate_recording.py --strict OK. "
         "PROMOTES/DEMOTES NOTHING: /governance applies the verdict.")

ITEMS = [
 {
  "queue_id": "V3-EXQ-861f",
  "title": ("INV-050/MECH-180 H1 leg (measurement axis): is 861e's seed-271 "
            "HIGH-arm MEL collapse an intervention-isolation defect? Reseed the "
            "measurement phase, substrate pinned to 861e's own 17befb8c"),
  "script": "experiments/v3_exq_861f_inv050_mech180_h1_measurement_rng_isolation.py",
  "priority": 58,
  "machine_affinity": "any",
  "status": "pending",
  "estimated_minutes": 420,
  "supersedes": None,
  "claim_id": "INV-050",
  "claim_ids": ["INV-050", "MECH-180"],
  "conditions": 6,
  "episodes_per_run": 132,
  "experiment_type": "v3_exq_861f_inv050_mech180_h1_measurement_rng_isolation",
  "seeds": 3,
  "note": COMMON + (
    " | THIS LEG = H1, measurement axis, REQUIRED. Varies EXACTLY ONE thing vs "
    "861e: torch and numpy are reseeded with a fixed derived offset "
    "(MEAS_RESEED_OFFSET, not 0, so the measurement stream is fresh rather than "
    "a replay of the convergence stream) immediately before the measurement "
    "phase -- before the measurement env is constructed, so env construction is "
    "inside the isolated stream. The measurement-phase RNG then depends only on "
    "the seed and is INDEPENDENT of how many calibration draws preceded it. The "
    "autopsy withdrew 'same seeds isolate the calibration change' precisely "
    "because extra calib draws are extra frozen-wake steps consuming torch "
    "policy RNG via select_action before an unreseeded measurement loop (calib "
    "is train=False, so they are not extra world-model training -- but they do "
    "displace the stream). Declared side effect, recorded not hidden: the reseed "
    "also makes the measurement stream identical across arms within a seed, "
    "making C1's dose-response a strictly PAIRED design -- an improvement, but a "
    "change from 861c/861e. "
    "| IN-RUN CONTROL: after the 5x3 reseeded grid, ARM_3_HIGH_ON is re-run for "
    "all three seeds with the reseed DISABLED, same CALIB_DRAWS=10, same pinned "
    "substrate, same process, same machine (cells reset all RNG at entry, so "
    "order is irrelevant). H1 is therefore decided WITHIN the run, so a machine "
    "difference vs ree-worker-1 cannot masquerade as the effect -- and comparing "
    "that control against 861e's RECORDED value MEASURES the machine delta "
    "directly, which is the part of H3 the autopsy could not separate (both runs "
    "report machine_class linux-x86_64-py3.10-torch2.12.0+cpu, so any difference "
    "is sub-machine-class). "
    "| DECLARED NULL (pre-registered): decisive cell is seed 271 ARM_3_HIGH_ON, "
    "reseeded vs unreseeded, each scored against its OWN mel_reference "
    "(mean_duration_factor IS that ratio, so >1.0 is the autopsy's own criterion, "
    "not a threshold invented here). H1 SUPPORTED if reseeded rises above 1.0 "
    "while the unreseeded control stays below (reproducing 861e) -> the collapse "
    "was an intervention-isolation defect and the whole 861c-vs-861e comparison "
    "must be re-read. H1 NOT SUPPORTED if both sit below 1.0 -> an INFORMATIVE "
    "null that removes measurement-phase RNG displacement from the live set and "
    "hands the question to H3, then H2; it does NOT make seed 271 'like seed 7'. "
    "UNINFORMATIVE (non_contributory) if the unreseeded control does not "
    "reproduce 861e's collapse -- this box then does not reproduce 861e at all "
    "and there is no baseline to isolate against. That is a pre-registered "
    "readiness precondition "
    "(in_run_unreseeded_control_reproduces_861e_collapse), guarded at setup by "
    "assert_anchor_reachable against 861e's frozen recorded cell (factor 0.8845) "
    "using the shipped predicate -- not a post-hoc excuse.") + SMOKE,
 },
 {
  "queue_id": "V3-EXQ-861g",
  "title": ("INV-050/MECH-180 H3 leg (algorithm axis): does 861e's seed-271 "
            "HIGH-arm MEL collapse survive on 861c's substrate? 861e protocol "
            "(CALIB_DRAWS=10 + R3) pinned to f810969"),
  "script": "experiments/v3_exq_861g_inv050_mech180_h3_substrate_pin_f810969.py",
  "priority": 58,
  "machine_affinity": "any",
  "status": "pending",
  "estimated_minutes": 380,
  "supersedes": None,
  "claim_id": "INV-050",
  "claim_ids": ["INV-050", "MECH-180"],
  "conditions": 6,
  "episodes_per_run": 132,
  "experiment_type": "v3_exq_861g_inv050_mech180_h3_substrate_pin_f810969",
  "seeds": 3,
  "note": COMMON + (
    " | THIS LEG = H3, algorithm axis, REQUIRED. Varies EXACTLY ONE thing vs "
    "861e: ree_core is pinned to f810969, 861c's own substrate. CALIB_DRAWS "
    "stays 10, R3 stays on, no reseed, legacy write path -- the 861e protocol run "
    "on the old code. That fills the missing cell of the 2x2 the lineage has been "
    "reasoning across: (f810969, n=5) = 861c, 271 HIGH-graded; (17befb8c, n=10) = "
    "861e, 271 collapsed; (f810969, n=10) = THIS LEG; (17befb8c, n=5) = H2, the "
    "labelled follow-on. The substrate delta is real and NOT merely default-off "
    "knobs: 1296 inserted lines across 8 ree_core files between f810969 and "
    "17befb8c, including UNCONDITIONAL agent.py changes (extra clock.phase_reset() "
    "sites, the cem_elite modulatory route backstop, orienting-decision tick "
    "decrements) and the whole ContextMemory write-selection machinery in "
    "e1_deep.py. Note 861c could not have had the write repair -- those knobs do "
    "not exist at f810969 -- so both predecessors necessarily ran the legacy "
    "write path, which is what makes the 2.5c lock a constant. "
    "| IN-RUN CONTROL doubling as the pin's POSITIVE control: after the n=10 grid, "
    "ARM_3_HIGH_ON is re-run for all three seeds at CALIB_DRAWS=5 -- 861c's exact "
    "decisive condition -- on the same pinned substrate, same process, same "
    "machine. It (i) verifies the pin behaviourally in a way no static check can "
    "(861c recorded factor 1.2146 for this cell; landing near 861e's 0.8845 "
    "instead would mean the pin did not take or the machine dominates), and (ii) "
    "isolates the CALIB_DRAWS change ON THE OLD SUBSTRATE on ONE box, which is "
    "the contrast 861c-vs-861e was supposed to be but was not. "
    "| DECLARED NULL (pre-registered): H3 SUPPORTED if seed 271 stays HIGH-graded "
    "(factor > 1.0) at n=10 on f810969 while 861e collapsed at n=10 on 17befb8c "
    "-> the collapse is a substrate/machine delta and the 861c/861e comparison is "
    "confounded by code drift, not calibration power. H3 NOT SUPPORTED if 271 "
    "collapses here too -> the old substrate does not rescue it, and the in-run "
    "n=5 control discriminates further: n=5 reproducing 1.215 while n=10 collapses "
    "on the SAME substrate and box is a clean, machine-free demonstration that "
    "CALIB_DRAWS alone moves the readout, i.e. H1's mechanism confirmed from the "
    "other side. UNINFORMATIVE (non_contributory) if the n=5 positive control does "
    "not reproduce 861c (pre-registered readiness precondition "
    "pin_positive_control_reproduces_861c_decisive_cell, threshold "
    "PIN_CONTROL_MIN_FACTOR=1.10 set ~10 percent below 861c's recorded 1.2146 and "
    "guarded at setup by assert_anchor_reachable against that frozen cell using "
    "the shipped predicate).") + SMOKE,
 },
 {
  "queue_id": "V3-EXQ-861h",
  "title": ("INV-050/MECH-180 substrate-defect CONTROL leg (representation axis): "
            "is the decisive seed's MEL readout trustworthy at all? 861e protocol "
            "on 17befb8c with non-degenerate ContextMemory write selection ON"),
  "script": "experiments/v3_exq_861h_inv050_mech180_contextmemory_write_lock_control.py",
  "priority": 56,
  "machine_affinity": "any",
  "status": "pending",
  "estimated_minutes": 420,
  "supersedes": None,
  "claim_id": "INV-050",
  "claim_ids": ["INV-050", "MECH-180"],
  "conditions": 6,
  "episodes_per_run": 132,
  "experiment_type": "v3_exq_861h_inv050_mech180_contextmemory_write_lock_control",
  "seeds": 3,
  "note": COMMON + (
    " | THIS LEG = CONTROL, representation axis. IT IS NOT A FOURTH FROZEN "
    "HYPOTHESIS and does not grow the qid's denominator (frozen set stays H1+H3, "
    "count 2, adopted at an interactive gate). It exists because two MANDATORY "
    "queue-experiment checks fired at queue time and neither could be discharged "
    "any other way: Step 2.5b(iv) (adversarially audit the portfolio for coverage "
    "and verdict-aliasing gaps BEFORE queuing, and add the leg that closes the "
    "worst one) and Step 2.5c (the corrupting overlap above, which BLOCKS by "
    "default). It converts 'running against a known corrupting defect' into "
    "'measuring under the repair'. "
    "| Varies EXACTLY ONE thing vs 861e: same pinned substrate 17befb8c, same "
    "CALIB_DRAWS=10, no reseed -- with contextmemory_write_selection switched "
    "from the substrate default 'argmin' (legacy, degenerate) to 'refractory' "
    "(contextmemory_write_refractory_k=2). 'refractory' was chosen over "
    "'usage_balancing' deliberately: V3-EXQ-943 validated BOTH (BIAS 16/16 "
    "occupied 5/5 seeds; REFRACTORY 5/5 seeds >=2, locking seeds settling at "
    "exactly k+1=3), but refractory's non-degeneracy is STRUCTURAL rather than "
    "counter-driven, so it is the smaller and more predictable perturbation to a "
    "run whose readout is a prediction error. Both knobs exist at 17befb8c and "
    "neither exists at f810969, which is why this leg pins 17befb8c and why the "
    "equivalent control cannot be run on 861c's substrate. "
    "| IN-RUN CONTROL: ARM_3_HIGH_ON re-run for all three seeds with the LEGACY "
    "argmin path, everything else held, same process and machine -- so the "
    "write-path contrast is decided WITHIN the run and the legacy cells double as "
    "a same-box replica of 861e's decisive cell. "
    "| DECLARED NULL (pre-registered): CONTROL FAILS if the refractory arm "
    "restores seed 271's HIGH grading (factor > 1.0) while the in-run legacy "
    "control reproduces the collapse -> the decisive seed's readout DEPENDS on a "
    "known CORRUPTING substrate defect, 861f/861g's H1/H3 readings are conditional "
    "on it, and a fourth hypothesis would owe registration against the qid -- "
    "which is a GOVERNANCE call this leg reports and does not make. CONTROL PASSES "
    "if seed 271's factor is materially unchanged under the repair -> the lock is "
    "not load-bearing for measured MEL, 861f/861g stand on their own, and the "
    "corrupting overlap is discharged empirically rather than by argument. "
    "UNINFORMATIVE (non_contributory) if the refractory arm does not actually "
    "reduce n_cycles_insufficient_touched_slots on seed 271 -- the repair did not "
    "engage, so nothing was controlled; that is the pre-registered readiness "
    "precondition write_repair_engaged_on_decisive_seed, whose LEGACY half is "
    "guarded at setup by assert_anchor_reachable against 861e's frozen recorded "
    "cell (6/6 insufficient cycles) using the shipped predicate. The refractory "
    "half is deliberately NOT anchored: no recorded run of that condition exists "
    "on this lineage, and a guard scored against a reference that does not exist "
    "would be theatre. "
    "| SCOPE: this leg does NOT validate or close "
    "`contextmemory-write-path-addressing-degeneracy` -- that entry's own "
    "validation is V3-EXQ-943's occupancy criterion, adjudicated separately. "
    "ORDERING: 861h is only interpretable alongside 861f/861g, so it is queued at "
    "a lower priority (56 vs 58); it does not gate them.") + SMOKE,
 },
]


def main():
    d = json.loads(Q.read_text())          # re-read the LIVE file, last
    existing = {i["queue_id"] for i in d["items"]}
    for it in ITEMS:
        if it["queue_id"] in existing:
            raise SystemExit(f"REFUSING: {it['queue_id']} already in the queue")
    for it in ITEMS:
        d["items"].append(it)
    Q.write_text(json.dumps(d, indent=2) + "\n")
    print("appended:", [i["queue_id"] for i in ITEMS],
          "-> items now", len(d["items"]))


if __name__ == "__main__":
    main()
