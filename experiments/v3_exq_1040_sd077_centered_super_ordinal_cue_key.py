"""V3-EXQ-1040 -- SD-077 centered super-ordinal goal-anchor cue key: direct
raw-key vs centered-key test on a live run.

RED-TEAM (Step 4.5, model fable, one pass): CONTESTED -> all findings dispositioned.
(1) HEADLINE, fixed: the original load-bearing C3 (mean contextual_complexity over
    ALL fired writes) DILUTES with run length. With merge_similarity 0.8 and
    complexity_threshold 0.2 the post-salience no-write branch is unreachable
    (best_sim < 0.8 implies complexity > 0.2, goal.py:531-555), so every
    salience-passing tick writes; allocations become rare as the bank fills, so the
    pooled mean decays and the cross-arm delta crossed the 0.05 margin in BOTH
    directions between tick 40 and tick 140 while C1's delta held at 5-14. A
    C1-pass/C3-fail run would then have recorded "weakens SD-077" for a denominator
    artifact. FIX: C3 re-denominated to the LENGTH-INVARIANT geometry statistic
    SD-077 actually asserts (P0 pairwise-cosine spread on a fixed-size probe); the
    old form is retained as reported-only C3R, and each cell now records allocation-
    only and reinforce-only complexity so the dilution is visible in the manifest.
(2) Fixed: observe() has two call sites (goal.py:514 write, :578 retrieve) and one
    tick can reach both, so the live baseline advances at up to 1-(1-a)^2 and the
    rate is state-dependent. The P0 replica now measures BOTH bounds and C3 routes
    on the WORST.
(3) Fixed: a raw arm that differentiated live (C2 false) with C1 false now routes
    to substrate_not_ready_requeue / unknown, not weakens -- the premise failed to
    reproduce, so the claim is untested rather than falsified.
(4) Fixed: the centered arm's SECOND anchor is an arithmetic identity (after lazy
    seeding, query and stored residuals are exactly anti-parallel, cosine -1,
    complexity clamps to 1.0, allocation guaranteed); the DV-symmetry declaration
    below now says so, and C1's margin of 4 exceeds that one free unit.
(5) Fixed: stale "256" in R3's recorded description (N_SLOTS is 1024).
(6) Dismissed as overclaim, text corrected: the "closed loop" was measured INERT at
    smoke length -- both arms produced bit-identical z_world streams over 40 ticks.
    Each cell now records its own arm_common_mode_ratio so this stays auditable.
No finding survived in families 1 (path open and sole), 3 (branch balance) or 4
(R1/R2/R3 self-certification).

THE CLAIM UNDER TEST (SD-077, claim_type design_decision, status candidate).
MECH-189's SuperOrdinalGoalMemory keys anchors on RAW z_world cosine. Under SD-008
z_world under-differentiation that cosine measures the shared common-mode offset
rather than the context, so every nursery contact reinforces one slot and
anchor_count saturates at 1. SD-077 asserts the repair: subtract a slow EMA
common-mode baseline from BOTH query and stored keys before every cue cosine
(ree_core/goal.py SuperOrdinalGoalMemory.observe/_centered/_best_match, config
GoalConfig.super_ordinal_cue_centering, default False = bit-identical OFF).

WHY THIS RUN EXISTS -- the claim has never been tested live, only recomputed
offline. SD-077's entire quantitative support is an OFFLINE recompute over a FIXED
historical dataset (the 155 contexts of V3-EXQ-669b seed 101, measured 2026-07-21):
raw z_world pairwise cosine min 0.9641 / mean 0.9898 with ZERO pairs below 0.8 and
||mean(z_world)|| / mean||z_world|| = 0.9949; centered residual pairwise cosine
spanning -0.760 to 1.000 with 97.7 percent of pairs below 0.8, giving 26 anchors
and mean complexity 0.076 at n_slots=64. No experiment has ever tagged SD-077
(claim_ids search over the whole evidence corpus: zero hits), and no run has ever
varied super_ordinal_cue_centering as the manipulation.

WHY V3-EXQ-669c IS NOT THAT TEST. 669c (run 2026-07-22, FAIL / mixed) enabled
super_ordinal_cue_centering=True in ALL THREE of its arms as a substrate ENABLER
for a MECH-329 ordering question; centering is a held-constant of that design, not
its manipulated variable, so 669c carries no OFF cell and cannot speak to SD-077.
It is however the reason this run raises the slot bank: 669c measured
anchor_count = 64 at n_slots = 64, i.e. SATURATED AT THE CAP, so a repeat at 64
would censor the ON arm's DV. n_slots = 1024 here, with a pre-registered cap-headroom
precondition (R3) that makes censoring visible instead of silent.

WHICH HALF OF C1 IS ACTUALLY MEASURED -- stated so no later reader mistakes the
control for the result. Given the measured geometry (R1: common_mode_ratio ~0.997,
so raw pairwise cosine is >= ~0.96 everywhere), the RAW arm's anchor_count = 1 is
close to ANALYTIC rather than empirical: contextual_complexity = 1 - best_cosine
<= ~0.04, strictly below the 0.2 allocation threshold, so after the first slot no
second allocation can fire. That arm is a within-run, within-substrate CONTROL
confirming the premise reproduces here, not a measurement. The OPEN quantity -- the
one this run exists to measure -- is the CENTERED arm's anchor_count, which is not
determined a priori: the centered residuals could still have been collinear (giving
1-2 anchors and a C1 failure) or could differentiate (the SD-077 prediction). C1's
margin of 4 is therefore a bar on the centered arm, and both sides of it are
reachable. What this adds over SD-077's offline recompute is the CLOSED LOOP: the
recompute replayed stored z_world through the key function in isolation, whereas
here the anchor bank feeds retrieve() -> z_goal -> action selection -> which
contexts are visited next, so the centered key has to differentiate a context
stream it is itself steering. NOTE the red-team measured that feedback INERT at
smoke length (both arms produced bit-identical z_world streams over 40 ticks), so
do not read a result as evidence about the closed loop unless the recorded per-cell
arm_common_mode_ratio and n_ticks actually differ between arms -- at 12x100 they may
well not, in which case this is simply a clean open-loop contrast.

GOV-REUSE-1 (Step 2.4). Decisive readout: anchor_count under raw vs centered cue
key, same substrate, same seeds. NOT recoverable from the record. 669b carries the
OFF reading (anchor_count = 1) but ran on a pre-SD-077 substrate at n_slots = 16;
669c carries an ON reading but on substrate_hash
0e3072c5c7f2f5e0beefb76e0478a5c0c49bd2fce366db778413273d6e4498a0 at n_slots = 64.
The two differ in BOTH substrate_hash and n_slots, so a cross-run delta confounds
the manipulation with a substrate change and a bank-size change. A within-run,
within-substrate contrast is the only uncontaminated form.

DESIGN. Two arms, matched seeds, one agent per cell, identical in every respect
except one config boolean:
  A raw_key       super_ordinal_cue_centering = False   (the SD-077 negation; the
                                                         geometry 669b ran on)
  B centered_key  super_ordinal_cue_centering = True    (the SD-077 repair)
The forced-feed wanting drive is ON in every episode of both arms -- 669c's
schedule manipulation is deliberately held constant here so the ONLY live
difference is the cue key.

DV-SYMMETRY DECLARATION (Step 3.5 design audit; one line per arm). Both arms share
one DV family: anchor_count (a cardinality over allocated slots) and n_allocate
(an uncensored allocation count), both produced by the branch
`best_sim >= super_ordinal_merge_similarity` inside SuperOrdinalGoalMemory.write.
The symmetry group of that DV is the group of transforms that leave the cue cosine
ranking unchanged: positive rescaling of the key vectors, and permutation of the
slots. The manipulation is an ADDITIVE SHIFT of both query and stored keys taken
BEFORE L2 normalisation (goal.py _centered then F.normalize in _best_match).
  - raw_key: not invariant. It is the identity transform, so it defines the
    reference geometry rather than being annihilated by it.
  - centered_key: not invariant, with ONE declared exception. An additive shift
    applied before normalisation is
    neither a positive rescaling nor a slot permutation -- it moves directions, so
    it moves cosines, so it moves the merge/allocate branch. Concretely: for keys
    sharing a large common component b, cos(x, y) is driven to 1 while
    cos(x - b, y - b) is not, which is exactly the 0.9898 -> spread-to--0.760
    change SD-077 measured. The delta is therefore a measurement, not an arithmetic
    identity fixed before the run -- EXCEPT for its first unit: after the lazy seed
    (goal.py:445-446) the second write's baseline is 0.98*k0 + 0.02*k1, so the query
    residual is 0.98*(k1-k0) and the stored residual -0.02*(k1-k0), exactly
    anti-parallel; cosine -1, complexity clamps to 1.0, and that second allocation
    fires for ANY k1 != k0. The centered arm's anchor floor is therefore 2, not 1,
    by arithmetic. C1's margin of 4 is set above that free unit deliberately, so no
    part of a C1 pass rests on it.
Neither arm's manipulation is a uniform additive constant over CANDIDATES (the
shift is common to all keys but is applied before a normalisation that is not
shift-equivariant), a monotone rescaling of a rank DV, or a permutation of a
set-aggregate's inputs.

PRE-REGISTERED READINESS GATE (scored FIRST; R1/R2 failure self-routes
substrate_not_ready_requeue with evidence_direction unknown and
non_degenerate=False -- NEVER a weakens).
  R1 PREMISE -- raw z_world carries a dominant common-mode offset on THIS nursery:
     common_mode_ratio = ||mean(z_world)|| / mean(||z_world||) >= 0.80, measured in
     P0 on a write-free probe pass. SD-077's motivating condition. If the encoder
     geometry has changed and the offset is absent, centering is a no-op BY
     CONSTRUCTION and the claim is UNTESTED, not falsified -- hence requeue, not
     weakens. Arm-invariant by construction: centering never touches the encoder,
     only how cue keys are COMPARED, so this single P0 measurement is valid for
     both arms and certifies both.
  R2 WRITES FIRE -- min total_writes across all cells >= 1. Without writes there is
     no anchor bank and anchor_count is measured over an empty set (the V3-EXQ-669a
     defect).
  R3 CAP HEADROOM (conditioned; see ROUTING) -- slot_headroom = n_slots -
     max(anchor_count over all cells) >= 1, i.e. no arm saturated the bank. This is
     the same-statistic guard on the DV the load-bearing C1 routes on: a censored
     anchor_count cannot support a NULL reading.

LOAD-BEARING ACCEPTANCE (scored only when R1 and R2 hold).
  C1 (load-bearing): anchor_count(centered_key) >= anchor_count(raw_key) +
     ANCHOR_DELTA_MARGIN on >= 2/3 seeds. The direct SD-077 consequence.
  C3 (load-bearing): the CUE-KEY GEOMETRY spreads. On the fixed-size P0 probe,
     frac(pairwise cosine < 0.8) for the centered residual, minus the same fraction
     for raw z_world, >= FRAC_BELOW_MARGIN on >= 2/3 seeds. SD-077 measured 0.977 vs
     0.000 offline. Deliberately LENGTH-INVARIANT: it is computed on a fixed probe,
     so unlike a mean over fired writes it cannot drift with the episode budget, the
     write count, or how full the bank is (see the red-team note above). It is also
     the asserted CAUSE, where C1 is the closed-loop CONSEQUENCE, so the two are
     independent and a C1-pass/C3-fail combination is informative rather than
     self-contradictory.
  C3R (reported, NOT load-bearing): the original mean-over-fired-writes complexity
     test, kept so the dilution is visible and comparable to 669b's published 0.0077.
  C2 (reported, NOT load-bearing): raw_key anchor_count <= RAW_SATURATION_CEILING on
     >= 2/3 seeds. Records whether the saturation SD-077 diagnoses reproduces live.
     Deliberately not load-bearing: a raw arm that differentiates somewhat while the
     centered arm still beats it by the margin is a supports reading for SD-077, and
     gating on C2 would convert that into a spurious requeue.

ROUTING (the 785 net: a precondition must not vacate a verdict it cannot have
caused). Cap censoring can only SHRINK the centered arm's anchor_count, so it can
never manufacture a C1 pass -- it can only manufacture a C1 FAILURE. R3 is
therefore scored against the NULL branch only:
  R1 or R2 unmet                  -> FAIL substrate_not_ready_requeue (unknown)
  C1 false AND C2 false            -> FAIL substrate_not_ready_requeue (unknown): the
                                     raw control differentiated live, so the premise
                                     did not reproduce and nothing was repaired
  C1 and C3 pass                  -> PASS supports SD-077 (if R3 also unmet, the
                                     MAGNITUDE is censored at the cap but the
                                     DIRECTION is unaffected; recorded, not vacated)
  C1/C3 not met AND R3 unmet       -> FAIL substrate_not_ready_requeue (unknown):
                                     a null under a censored DV is uninterpretable
  C1/C3 not met AND R3 met         -> FAIL weakens SD-077: a genuine null. The
                                     offset was present, writes fired, the bank had
                                     headroom, and centering still did not restore
                                     anchor differentiation live.

KNOWN OPEN SUBSTRATE DEFECT ON AN EXERCISED PATH (Step 2.5c disposition). The
driver steps E1DeepPredictor, which holds a ContextMemory, and substrate_queue
entry contextmemory-write-path-addressing-degeneracy (severity corrupting, status
implemented_pending_validation) lists ree_core/predictors/e1_deep.py::ContextMemory.write.
Disposition: the ADDRESSING half named by that substrate_paths entry is CONFIRMED
FIXED by V3-EXQ-436g (16/16 occupied slots on 5/5 seeds, transfer gate confirmed);
the half still open is the CONTENT half, whose own declared lever is SD-070 z_world
entropy -- i.e. the very z_world under-differentiation this experiment MEASURES as
its R1 premise rather than suffers as a confound. It is also identical in both
arms and cannot produce a cross-arm delta. Recorded, not blocking.

A trained z_world encoder is NOT required and is not wanted: z_world is a
deterministic context cue (same obs -> same z_world), and the claim is explicitly
about the geometry an UNTRAINED encoder imposes. Training it away is SD-070's
problem and was measured to collapse z_world outright.
"""
from __future__ import annotations

import sys
import math
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1040_sd077_centered_super_ordinal_cue_key"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["SD-077"]

# ---- Pre-registered thresholds (defined HERE, never inferred post-hoc) ----
COMMON_MODE_FLOOR = 0.80      # R1: ||mean(z)|| / mean||z|| on raw z_world
WRITES_FLOOR = 1              # R2: min total_writes across cells
ANCHOR_DELTA_MARGIN = 4       # C1: centered anchors >= raw anchors + this
FRAC_BELOW_MARGIN = 0.50      # C3: centered frac(pairwise cos < 0.8) >= raw frac + this
COMPLEXITY_MARGIN = 0.05      # reported only (see C3 note): mean-over-writes complexity
RAW_SATURATION_CEILING = 2    # C2 (reported): raw_key anchor_count <= this
SEED_PASS_FRACTION = 2.0 / 3.0

# ---- Substrate operating point (identical in both arms except the one flag) ----
N_SLOTS = 1024                # 669c saturated at 64. At 12x100 steps the centered arm
                              # allocates fast (smoke: 8 anchors in 40 writes), so the cap
                              # is set well clear of any plausible reach; R3 still checks it.
SALIENCE_THRESHOLD = 0.5
COMPLEXITY_THRESHOLD = 0.2
MERGE_SIMILARITY = 0.8
WRITE_ALPHA = 0.3
BASELINE_ALPHA = 0.02         # SD-077 default, matching SD-066
FORCED_BENEFIT = 0.5          # salience 0.5*(1+2*0.9) = 1.4 >> 0.5 threshold
FORCED_DRIVE = 0.9
PROBE_STEPS = 160             # P0 write-free geometry probe (669b measured 155)

ARM_CENTERING = {"raw_key": False, "centered_key": True}
ARMS = ["raw_key", "centered_key"]

_ZG = ZGoalStreamAccumulator()


def _build_nursery_env(seed: int) -> CausalGridWorldV2:
    """Stage-0 forced-feed nursery: dense resources, hazard-free safe context.
    Same construction as V3-EXQ-669b/669c so the geometry this run measures is the
    geometry SD-077's offline recompute was taken on."""
    return CausalGridWorldV2(
        size=8,
        num_hazards=0,
        num_resources=6,
        use_proxy_fields=True,
        seed=seed,
    )


def _build_agent(env: CausalGridWorldV2, centering: bool) -> REEAgent:
    """One agent per cell. The ONLY value that differs between arms is
    super_ordinal_cue_centering."""
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
        super_ordinal_cue_centering=centering,   # <-- THE MANIPULATION
        super_ordinal_cue_baseline_alpha=BASELINE_ALPHA,
        super_ordinal_n_slots=N_SLOTS,
        super_ordinal_salience_threshold=SALIENCE_THRESHOLD,
        super_ordinal_complexity_mode="novelty",
        super_ordinal_complexity_threshold=COMPLEXITY_THRESHOLD,
        super_ordinal_merge_similarity=MERGE_SIMILARITY,
        super_ordinal_write_alpha=WRITE_ALPHA,
    )
    return REEAgent(cfg)


def _world_dim(agent: REEAgent) -> int:
    return agent.config.latent.world_dim


def _step_once(agent: REEAgent, env: CausalGridWorldV2, obs_dict) -> Tuple[int, torch.Tensor]:
    """One sense-plan-act tick. Returns (action_idx, z_world row [d])."""
    latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
    ticks = agent.clock.advance()
    wd = _world_dim(agent)
    e1_prior = (
        agent._e1_tick(latent) if ticks.get("e1_tick")
        else torch.zeros(1, wd, device=agent.device)
    )
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    action = agent.select_action(candidates, ticks)
    zw = latent.z_world.detach()
    if zw.dim() == 2:
        zw = zw.mean(dim=0)
    return int(action.argmax(dim=-1).item()), zw.clone()


def _pairwise_stats(rows: torch.Tensor) -> Dict[str, float]:
    """min / mean pairwise cosine and the fraction of pairs below 0.8, over the
    upper triangle. Mirrors SD-077's own offline measurement."""
    n = int(rows.shape[0])
    if n < 2:
        return {"min": float("nan"), "mean": float("nan"), "frac_below_0p8": float("nan")}
    unit = F.normalize(rows, dim=-1)
    sims = unit @ unit.t()
    iu = torch.triu_indices(n, n, offset=1)
    vals = sims[iu[0], iu[1]]
    return {
        "min": round(float(vals.min().item()), 6),
        "mean": round(float(vals.mean().item()), 6),
        "frac_below_0p8": round(float((vals < 0.8).float().mean().item()), 6),
    }


def _geometry_probe(seed: int, steps: int) -> Dict[str, Any]:
    """P0, WRITE-FREE. Collects z_world over the nursery and measures the raw
    common-mode ratio (R1's statistic) plus the raw-vs-centered pairwise cosine
    spread. Uses a centering=False agent and never calls update_z_goal, so no
    anchor is written and no baseline is advanced inside the store -- the EMA
    baseline is replicated here offline at the SAME alpha the substrate uses.

    Arm-invariant by construction: super_ordinal_cue_centering changes only how cue
    keys are COMPARED inside SuperOrdinalGoalMemory, never the encoder that produces
    z_world, so this single measurement certifies the premise for BOTH arms."""
    torch.manual_seed(seed)
    env = _build_nursery_env(seed)
    agent = _build_agent(env, centering=False)
    agent.set_super_ordinal_write_enabled(False)  # belt and braces: no writes in P0
    _, obs_dict = env.reset()
    agent.reset()
    rows: List[torch.Tensor] = []
    for i in range(steps):
        action_idx, zw = _step_once(agent, env, obs_dict)
        rows.append(zw)
        _, _harm, done, _, obs_dict = env.step(action_idx)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
    raw = torch.stack(rows, dim=0)
    mean_vec = raw.mean(dim=0)
    common_mode_ratio = float(
        mean_vec.norm().item() / max(float(raw.norm(dim=-1).mean().item()), 1e-12)
    )
    # Replicate the substrate's lazy-seeded EMA baseline offline.
    #
    # ADVANCE-RATE CAVEAT (red-team finding 1a, verified at goal.py:514 and :578).
    # SuperOrdinalGoalMemory.observe() has TWO call sites -- write() and retrieve() --
    # and one agent tick can reach BOTH (retrieve fires whenever goal_norm falls below
    # super_ordinal_seed_below_norm), so the live baseline advances at an EFFECTIVE
    # alpha of up to 1 - (1 - a)^2, and the exact rate is state-dependent. A single
    # nominal-alpha replica would therefore not be the substrate's baseline
    # trajectory. Rather than guess the rate, the probe measures the geometry at BOTH
    # bounds (one advance per context, and two) and the criterion routes on the WORST
    # of the two -- conservative, and immune to how often retrieve() happens to fire.
    def _centered_at(n_advances: int) -> torch.Tensor:
        baseline = raw[0].clone()
        rows_out = []
        for i in range(raw.shape[0]):
            rows_out.append(raw[i] - baseline)
            for _ in range(n_advances):
                baseline = (1.0 - BASELINE_ALPHA) * baseline + BASELINE_ALPHA * raw[i]
        return torch.stack(rows_out, dim=0)

    centered_1x = _pairwise_stats(_centered_at(1))
    centered_2x = _pairwise_stats(_centered_at(2))
    raw_stats = _pairwise_stats(raw)
    worst_frac_below = min(centered_1x["frac_below_0p8"], centered_2x["frac_below_0p8"])
    return {
        "seed": seed,
        "n_contexts": int(raw.shape[0]),
        "common_mode_ratio": round(common_mode_ratio, 6),
        "raw_pairwise": raw_stats,
        "centered_pairwise_1x_advance": centered_1x,
        "centered_pairwise_2x_advance": centered_2x,
        # C3's measured statistic: worst-case centered spread minus raw spread.
        "frac_below_delta_worst": round(
            float(worst_frac_below - raw_stats["frac_below_0p8"]), 6),
        "centered_frac_below_worst": round(float(worst_frac_below), 6),
        "raw_frac_below": raw_stats["frac_below_0p8"],
    }


def _run_cell(arm: str, seed: int, n_child: int, steps: int) -> Dict[str, Any]:
    centering = ARM_CENTERING[arm]
    print(f"Seed {seed} Condition {arm}", flush=True)
    config_slice = {
        "arm": arm, "seed": seed, "n_child": n_child, "steps": steps,
        "centering": centering, "n_slots": N_SLOTS,
        "salience_threshold": SALIENCE_THRESHOLD,
        "complexity_threshold": COMPLEXITY_THRESHOLD,
        "merge_similarity": MERGE_SIMILARITY,
        "write_alpha": WRITE_ALPHA, "baseline_alpha": BASELINE_ALPHA,
        "forced_benefit": FORCED_BENEFIT, "forced_drive": FORCED_DRIVE,
    }
    with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
        torch.manual_seed(seed)
        env = _build_nursery_env(seed)
        agent = _build_agent(env, centering=centering)
        som = agent.super_ordinal_goal_memory
        complexities: List[float] = []
        alloc_complexities: List[float] = []     # complexity on ALLOCATE writes
        reinf_complexities: List[float] = []     # complexity on REINFORCE writes
        zw_rows: List[torch.Tensor] = []         # this ARM's own z_world stream

        for ep in range(n_child):
            _, obs_dict = env.reset()
            agent.reset()  # per-episode; does NOT clear the super-ordinal store
            for _ in range(steps):
                action_idx, _zw = _step_once(agent, env, obs_dict)
                zw_rows.append(_zw)
                w0, a0 = som._n_writes, som._n_allocate
                # Forced supra-threshold benefit -> z_goal seeds -> super-ordinal write.
                agent.update_z_goal(benefit_exposure=FORCED_BENEFIT,
                                    drive_level=FORCED_DRIVE)
                if som._n_writes > w0:
                    cx = float(som._last_complexity)
                    complexities.append(cx)
                    # Separating the two write kinds is what makes the mean-over-writes
                    # statistic readable: with merge_similarity 0.8 and
                    # complexity_threshold 0.2 the post-salience no-write branch is
                    # UNREACHABLE (best_sim < 0.8 implies complexity > 0.2), so every
                    # salience-passing tick writes and the denominator grows with run
                    # length while allocations become rarer as the bank fills. The
                    # pooled mean therefore DECAYS with run length -- which is exactly
                    # why it is reported here and is NOT the load-bearing criterion.
                    (alloc_complexities if som._n_allocate > a0
                     else reinf_complexities).append(cx)
                _, _harm, done, _, obs_dict = env.step(action_idx)
                if done:
                    break
            print(f"  [train] seed={seed} arm={arm} ep {ep + 1}/{n_child} "
                  f"anchors={som.n_occupied()} writes={som._n_writes}", flush=True)

        agent.set_super_ordinal_write_enabled(False)  # freeze at weaning

        anchor_count = int(som.n_occupied())
        total_writes = int(som._n_writes)
        mean_complexity = (
            sum(complexities) / len(complexities) if complexities else 0.0
        )
        cell_ok = total_writes > 0
        print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)

        # This ARM's own z_world common-mode ratio, so the P0 probe's premise
        # measurement can be checked against the stream the arm actually traversed
        # (the arms can in principle diverge once the anchor bank feeds z_goal).
        arm_zw = torch.stack(zw_rows, dim=0) if zw_rows else None
        arm_cmr = (
            round(float(arm_zw.mean(dim=0).norm().item()
                        / max(float(arm_zw.norm(dim=-1).mean().item()), 1e-12)), 6)
            if arm_zw is not None else float("nan")
        )
        _mean = lambda xs: round(float(sum(xs) / len(xs)), 6) if xs else 0.0

        row = {
            "arm": arm,
            "seed": seed,
            "centering": bool(centering),
            "anchor_count": anchor_count,
            "n_allocate": int(som._n_allocate),
            "n_reinforce": int(som._n_reinforce),
            "n_baseline_seeds": int(som._n_seeds),
            "total_writes": total_writes,
            "mean_complexity": round(float(mean_complexity), 6),
            "mean_complexity_allocate_only": _mean(alloc_complexities),
            "mean_complexity_reinforce_only": _mean(reinf_complexities),
            "n_complexity_samples": len(complexities),
            "n_allocate_samples": len(alloc_complexities),
            "n_reinforce_samples": len(reinf_complexities),
            "arm_common_mode_ratio": arm_cmr,
            "n_ticks": len(zw_rows),
            "slot_headroom": int(N_SLOTS - anchor_count),
        }
        cell.stamp(row)
    _ZG.observe(agent)
    return row


def run_experiment(n_child: int, steps: int, seeds: List[int],
                   probe_steps: int, dry_run: bool) -> Dict[str, Any]:
    # ---- P0: write-free geometry probe (R1's statistic), once per seed ----
    geometry = [_geometry_probe(s, probe_steps) for s in seeds]
    min_common_mode = min(g["common_mode_ratio"] for g in geometry)
    r1_common_mode_offset = min_common_mode >= COMMON_MODE_FLOOR
    print(f"[P0] min common_mode_ratio over seeds = {round(min_common_mode, 4)} "
          f"(floor {COMMON_MODE_FLOOR}) -> R1={'MET' if r1_common_mode_offset else 'UNMET'}",
          flush=True)

    arm_results: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            arm_results.append(_run_cell(arm, seed, n_child, steps))

    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r in arm_results:
        by_seed.setdefault(r["seed"], {})[r["arm"]] = r

    n_seeds = len(seeds)
    seed_pass_n = math.ceil(SEED_PASS_FRACTION * n_seeds - 1e-9)

    # ---- READINESS ----
    min_total_writes = min(r["total_writes"] for r in arm_results)
    r2_writes_fire = min_total_writes >= WRITES_FLOOR

    max_anchor = max(r["anchor_count"] for r in arm_results)
    slot_headroom = int(N_SLOTS - max_anchor)
    r3_cap_headroom = slot_headroom >= 1

    readiness_core = r1_common_mode_offset and r2_writes_fire

    # ---- CRITERIA ----
    def _c1(s) -> bool:
        return (by_seed[s]["centered_key"]["anchor_count"]
                >= by_seed[s]["raw_key"]["anchor_count"] + ANCHOR_DELTA_MARGIN)

    def _c2(s) -> bool:
        return by_seed[s]["raw_key"]["anchor_count"] <= RAW_SATURATION_CEILING

    geo_by_seed = {g["seed"]: g for g in geometry}

    def _c3(s) -> bool:
        # LENGTH-INVARIANT geometry criterion. The mean-over-fired-writes complexity
        # this criterion originally used is diluted by run length (see the write-kind
        # split in _run_cell), so it is reported, not scored. What SD-077 actually
        # asserts is a property of the CUE-KEY GEOMETRY: raw z_world pairwise cosine
        # had 0.0 percent of pairs below 0.8, the centered residual 97.7 percent. That
        # statistic is computed on a FIXED-SIZE P0 probe, so it cannot drift with the
        # number of writes, the bank state, or the episode budget.
        return (geo_by_seed[s]["frac_below_delta_worst"] >= FRAC_BELOW_MARGIN)

    def _c3_reported(s) -> bool:
        # Retained as REPORTED context only -- the original mean-over-writes form.
        return (by_seed[s]["centered_key"]["mean_complexity"]
                >= by_seed[s]["raw_key"]["mean_complexity"] + COMPLEXITY_MARGIN)

    n_c1 = sum(1 for s in seeds if _c1(s))
    n_c2 = sum(1 for s in seeds if _c2(s))
    n_c3 = sum(1 for s in seeds if _c3(s))
    n_c3r = sum(1 for s in seeds if _c3_reported(s))
    frac_c1, frac_c2, frac_c3 = (n_c1 / float(n_seeds), n_c2 / float(n_seeds),
                                 n_c3 / float(n_seeds))
    frac_c3r = n_c3r / float(n_seeds)
    c3_reported_pass = n_c3r >= seed_pass_n
    c1_pass = n_c1 >= seed_pass_n
    c2_raw_saturates = n_c2 >= seed_pass_n   # reported, NOT load-bearing
    c3_pass = n_c3 >= seed_pass_n
    criteria_pass = c1_pass and c3_pass

    # Worst-cell effect sizes, recorded so each criterion's verdict is re-derivable
    # from the manifest alone (validate_experiments criteria-threshold rule).
    min_anchor_delta = min(
        by_seed[s]["centered_key"]["anchor_count"] - by_seed[s]["raw_key"]["anchor_count"]
        for s in seeds
    )
    min_frac_below_delta = min(geo_by_seed[s]["frac_below_delta_worst"] for s in seeds)
    min_complexity_delta = min(
        by_seed[s]["centered_key"]["mean_complexity"]
        - by_seed[s]["raw_key"]["mean_complexity"]
        for s in seeds
    )
    max_raw_anchor = max(by_seed[s]["raw_key"]["anchor_count"] for s in seeds)

    # ---- ROUTING (R3 scored against the NULL branch only; see module docstring) ----
    magnitude_censored = False
    if not readiness_core:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
        overall_direction = "unknown"
        non_degenerate = False
        if not r1_common_mode_offset:
            degeneracy_reason = (
                f"R1 unmet: raw z_world common_mode_ratio min={round(min_common_mode, 4)} "
                f"< floor {COMMON_MODE_FLOOR}. SD-077's motivating condition (a dominant "
                "common-mode offset in z_world) is ABSENT on this nursery, so centering is "
                "a no-op by construction and the claim is UNTESTED, not falsified. Re-queue "
                "on a substrate/nursery that reproduces the offset, or route to SD-008 / "
                "SD-070 -- the encoder geometry has changed since the 2026-07-21 recompute."
            )
        else:
            degeneracy_reason = (
                f"R2 unmet: min total_writes across cells = {min_total_writes} < "
                f"{WRITES_FLOOR}. No anchor was written, so anchor_count is measured over an "
                "empty set (the V3-EXQ-669a defect). Re-queue at an adequate forced-feed "
                "budget."
            )
    elif criteria_pass:
        outcome, label = "PASS", "centered_cue_key_restores_anchor_differentiation"
        overall_direction = "supports"
        non_degenerate = True
        degeneracy_reason = ""
        if not r3_cap_headroom:
            magnitude_censored = True
            print("[note] anchor bank saturated at n_slots; the DIRECTION is unaffected "
                  "(censoring only shrinks the centered arm) but the MAGNITUDE is a lower "
                  "bound.", flush=True)
    elif not r3_cap_headroom:
        # 785 net: a censored DV can only manufacture a NULL, never a pass. Only here
        # does R3 vacate.
        outcome, label = "FAIL", "substrate_not_ready_requeue"
        overall_direction = "unknown"
        non_degenerate = False
        degeneracy_reason = (
            f"R3 unmet on the NULL branch: max anchor_count={max_anchor} reached "
            f"n_slots={N_SLOTS} (slot_headroom={slot_headroom}). The centered arm's "
            "anchor_count is censored at the cap, so C1's failure cannot be distinguished "
            "from a truncated delta. Re-queue with a larger super_ordinal_n_slots. (Had C1 "
            "PASSED this would not have fired: censoring can only shrink the centered arm.)"
        )
    elif not c1_pass and not c2_raw_saturates:
        # The RAW control differentiated live (C2 false) AND the centered arm did not
        # beat it (C1 false): SD-077's premise -- a raw key that saturates -- did not
        # reproduce, so there was nothing for centering to repair. That is an UNTESTED
        # reading, not a weakens. (R1 can pass on the common-mode ratio while the raw
        # arm still allocates, so this branch is not dead.)
        outcome, label = "FAIL", "substrate_not_ready_requeue"
        overall_direction = "unknown"
        non_degenerate = False
        degeneracy_reason = (
            f"Premise did not reproduce: the raw_key control allocated more than "
            f"{RAW_SATURATION_CEILING} anchors (max raw anchor_count={max_raw_anchor}) on "
            f"{n_seeds - n_c2}/{n_seeds} seeds, so the saturation SD-077 repairs was not "
            "present and the centered arm had nothing to improve on. Not a weakens: the "
            "claim was untested, not falsified."
        )
    else:
        outcome, label = "FAIL", "centered_cue_key_did_not_restore_differentiation"
        overall_direction = "weakens"
        non_degenerate = True
        degeneracy_reason = ""

    per_claim = {"SD-077": overall_direction}

    result: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "outcome": outcome,
        "dry_run": dry_run,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "evidence_direction": overall_direction,
        "evidence_direction_per_claim": per_claim,
        "config": {
            "n_child": n_child, "steps": steps, "seeds": seeds,
            "probe_steps": probe_steps,
            "n_slots": N_SLOTS,
            "salience_threshold": SALIENCE_THRESHOLD,
            "complexity_threshold": COMPLEXITY_THRESHOLD,
            "merge_similarity": MERGE_SIMILARITY,
            "write_alpha": WRITE_ALPHA,
            "baseline_alpha": BASELINE_ALPHA,
            "forced_benefit": FORCED_BENEFIT, "forced_drive": FORCED_DRIVE,
            "arm_centering": ARM_CENTERING,
            "common_mode_floor": COMMON_MODE_FLOOR,
            "writes_floor": WRITES_FLOOR,
            "anchor_delta_margin": ANCHOR_DELTA_MARGIN,
            "frac_below_margin": FRAC_BELOW_MARGIN,
            "complexity_margin_reported_only": COMPLEXITY_MARGIN,
            "raw_saturation_ceiling": RAW_SATURATION_CEILING,
            "seed_pass_fraction": SEED_PASS_FRACTION,
        },
        "metrics": {
            "readiness_met": readiness_core,
            "r1_common_mode_offset": r1_common_mode_offset,
            "r2_writes_fire": r2_writes_fire,
            "r3_cap_headroom": r3_cap_headroom,
            "min_common_mode_ratio": round(min_common_mode, 6),
            "min_total_writes": int(min_total_writes),
            "max_anchor_count_across_arms": int(max_anchor),
            "slot_headroom": slot_headroom,
            "magnitude_censored_at_cap": magnitude_censored,
            "frac_c1_anchor_delta": round(frac_c1, 4),
            "frac_c2_raw_saturates": round(frac_c2, 4),
            "frac_c3_geometry_spread_delta": round(frac_c3, 4),
            "frac_c3reported_mean_complexity_delta": round(frac_c3r, 4),
            "c3_reported_mean_complexity_pass": c3_reported_pass,
            "min_frac_below_delta_worst": round(float(min_frac_below_delta), 6),
            "min_anchor_delta_centered_minus_raw": int(min_anchor_delta),
            "min_complexity_delta_centered_minus_raw": round(float(min_complexity_delta), 6),
            "max_raw_key_anchor_count": int(max_raw_anchor),
            "c1_pass": c1_pass,
            "c2_raw_saturates_reported": c2_raw_saturates,
            "c3_pass": c3_pass,
            "seed_pass_n_required": seed_pass_n,
        },
        "diagnostics": {"p0_geometry_per_seed": geometry},
        "arm_results": arm_results,
        "interpretation": {
            "label": label,
            "preconditions": [
                {
                    "name": "R1_raw_zworld_common_mode_offset_present",
                    "description": "PREMISE gate. SD-077 is conditioned on a dominant "
                                   "common-mode offset in raw z_world; without it centering "
                                   "is a no-op by construction and the claim is untested "
                                   "rather than falsified. Statistic is the WORST (minimum) "
                                   "seed, matching the all-seeds quantifier in met.",
                    "measured": round(min_common_mode, 6),
                    "threshold": COMMON_MODE_FLOOR,
                    "direction": "lower",
                    "control": "P0 write-free probe pass over the nursery with a "
                               "centering=False agent; arm-invariant because centering "
                               "never touches the encoder, only cue COMPARISON",
                    "offending_cell": min(geometry, key=lambda g: g["common_mode_ratio"])["seed"],
                    "met": bool(r1_common_mode_offset),
                },
                {
                    "name": "R2_super_ordinal_writes_fire",
                    "description": "Non-vacuity gate: every cell must fire at least one "
                                   "super-ordinal write, else anchor_count is measured over "
                                   "an empty set (the V3-EXQ-669a defect). Worst cell "
                                   "reported, matching the min quantifier in met.",
                    "measured": int(min_total_writes),
                    "threshold": WRITES_FLOOR,
                    "direction": "lower",
                    "control": "forced supra-threshold benefit every step in both arms",
                    "met": bool(r2_writes_fire),
                },
                {
                    "name": "R3_anchor_bank_cap_headroom",
                    "description": "SAME-STATISTIC censoring guard on the DV C1 routes on: "
                                   "no arm may saturate super_ordinal_n_slots, else "
                                   "anchor_count is censored and a NULL is uninterpretable. "
                                   "V3-EXQ-669c saturated at n_slots=64, which is why this "
                                   "run uses 1024. Scored against the NULL branch only -- "
                                   "censoring can only shrink the centered arm, so it can "
                                   "never manufacture a C1 pass.",
                    "measured": slot_headroom,
                    "threshold": 1,
                    "direction": "lower",
                    "control": "n_slots minus the maximum anchor_count over all cells",
                    "met": bool(r3_cap_headroom),
                },
            ],
            "criteria_non_degenerate": {
                # C1/C3 discriminate iff writes fired AND the DV was not censored
                # (or C1 passed, in which case censoring is irrelevant).
                "C1": bool(r2_writes_fire and (r3_cap_headroom or c1_pass)),
                # C3 is now a P0 geometry statistic over a fixed-size probe, so it is
                # non-degenerate whenever the probe collected >= 2 contexts -- it does
                # not depend on any write firing.
                "C3": bool(min(g["n_contexts"] for g in geometry) >= 2),
                "C2": bool(r2_writes_fire),
            },
            "combination_rule": "PASS requires C1 (behavioural: anchor cardinality) AND C3 "
                                "(geometric: cue-key pairwise spread), each on "
                                ">= ceil(2/3 * n_seeds) seeds, and only after R1 and R2 hold. "
                                "The two are deliberately independent -- C3 is the asserted "
                                "CAUSE measured on a fixed P0 probe, C1 the closed-loop "
                                "CONSEQUENCE -- so C1-pass/C3-fail is informative rather than "
                                "contradictory. C2 and C3R are reported context and never "
                                "enter the verdict.",
            "criteria": [
                {
                    "name": "C1_centered_more_anchors",
                    "load_bearing": True,
                    "passed": bool(c1_pass),
                    # `measured`/`threshold` are the SAME statistic `passed` tests: the
                    # seed-count against the >=2/3 bar. The per-seed delta bar it is
                    # built from is recorded alongside so the verdict is re-derivable
                    # from the manifest without opening this driver.
                    "measured": int(n_c1),
                    "threshold": int(seed_pass_n),
                    "measured_min_anchor_delta": int(min_anchor_delta),
                    "threshold_anchor_delta_margin": int(ANCHOR_DELTA_MARGIN),
                    "unit": "seeds passing (anchor_count centered - raw >= margin)",
                },
                {
                    "name": "C3_centered_key_spreads_cue_geometry",
                    "load_bearing": True,
                    "passed": bool(c3_pass),
                    "measured": int(n_c3),
                    "threshold": int(seed_pass_n),
                    "measured_min_frac_below_delta": round(float(min_frac_below_delta), 6),
                    "threshold_frac_below_margin": FRAC_BELOW_MARGIN,
                    "unit": "seeds passing (P0 frac of pairwise cosines < 0.8: "
                            "centered worst-case minus raw >= margin)",
                },
                {
                    "name": "C3R_centered_higher_mean_complexity_REPORTED",
                    "load_bearing": False,
                    "passed": bool(c3_reported_pass),
                    "measured": int(n_c3r),
                    "threshold": int(seed_pass_n),
                    "measured_min_complexity_delta": round(float(min_complexity_delta), 6),
                    "threshold_complexity_margin": COMPLEXITY_MARGIN,
                    "unit": "seeds passing (mean_complexity over fired writes, "
                            "centered - raw >= margin) -- DILUTES with run length, "
                            "reported only",
                },
                {
                    "name": "C2_raw_key_saturates_REPORTED",
                    "load_bearing": False,
                    "passed": bool(c2_raw_saturates),
                    "measured": int(n_c2),
                    "threshold": int(seed_pass_n),
                    "measured_max_raw_anchor_count": int(max_raw_anchor),
                    "threshold_raw_saturation_ceiling": int(RAW_SATURATION_CEILING),
                    "unit": "seeds passing (raw_key anchor_count <= ceiling)",
                },
            ],
            "evidence_direction": overall_direction,
        },
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()

    if args.dry_run:
        n_child, steps, seeds, probe_steps = 2, 20, [42], 24
    else:
        n_child, steps, seeds, probe_steps = 12, 100, [42, 43, 44], PROBE_STEPS

    result = run_experiment(n_child, steps, seeds, probe_steps, args.dry_run)

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        result,
        out_dir,
        dry_run=args.dry_run,
        config=result.get("config"),
        # Explicit list + started_at: the Experimental Recording Standard's always-core
        # requires `seeds` and `elapsed_seconds`, and passing seeds=None leaves BOTH absent
        # (caught by validate_recording.py --strict; the 669c lineage this driver follows
        # has the same gap).
        seeds=seeds,
        started_at=t0,
        script_path=Path(__file__),
        z_goal_stream_stats=_ZG.stats(),
    )

    m = result["metrics"]
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"readiness: R1_common_mode={m['r1_common_mode_offset']} "
          f"(min ratio {m['min_common_mode_ratio']}) "
          f"R2_writes_fire={m['r2_writes_fire']} (min writes {m['min_total_writes']}) "
          f"R3_cap_headroom={m['r3_cap_headroom']} (headroom {m['slot_headroom']})",
          flush=True)
    print(f"C1(anchor delta)={m['c1_pass']} C3(geometry spread)={m['c3_pass']} "
          f"[C2 raw saturates={m['c2_raw_saturates_reported']}; "
          f"C3R mean-complexity (reported)={m['c3_reported_mean_complexity_pass']}] "
          f"(frac C1={m['frac_c1_anchor_delta']} C3={m['frac_c3_geometry_spread_delta']} "
          f"C2={m['frac_c2_raw_saturates']} C3R={m['frac_c3reported_mean_complexity_delta']}) "
          f"min_frac_below_delta={m['min_frac_below_delta_worst']}", flush=True)
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
