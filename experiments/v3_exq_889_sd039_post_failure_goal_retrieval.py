"""V3-EXQ-889 -- SD-039 confirming DV: post-failure goal-identity retrieval.

PURPOSE (evidence). SD-039 claims: "Dual-trace anchors preserve a goal-state
snapshot plus wanting/arousal payload at write or invalidation time, so
unresolved goals remain QUERYABLE after the active plan fails." SD-039 is
status=candidate / v3_pending=true with ZERO scoring experimental evidence:
its only V3 run, V3-EXQ-494, is an implementation-acceptance probe
(experiment_purpose=diagnostic, adjudication=unverified) and is correctly
excluded from confidence by scoring_excluded="diagnostic_probe".

WHAT 494 ESTABLISHED, AND WHAT IT DID NOT. 494's UC3/UC4/UC5 showed the
payload is POPULATED, SURVIVES mark_inactive, and that a hand-constructed
goal-relevant phase scores higher than a hand-constructed irrelevant phase.
Every one of those is a PRESENCE property, measured on anchors the driver
wrote itself with hand-built payloads. None of them shows the preserved
payload is RECOVERABLE -- that a later query with a FRESH cue returns the
right preserved trace out of a pool of competing ones. "Queryable after the
plan fails" is a retrieval property, and retrieval is what this run measures.

WALL-INDEPENDENT. The DV is a property of the representation and of the
retrieval function, not of task performance: no reward, no success rate, no
episode return enters any criterion. The agent's rollout exists only to
generate a genuine z_goal stream and a genuine dual-trace anchor pool.

DESIGN. Per (arm, seed):
  Phase A (write + invalidate). N_GOAL_EPOCHS goal epochs, each in its own
    env instance so z_world -- and therefore the z_goal attractor -- differs
    by epoch. The agent runs EPISODES_PER_EPOCH episodes per epoch and every
    ANCHOR_EVERY steps writes an anchor through the REAL substrate path:
    the payload comes from HippocampalModule.build_goal_payload (the SD-039
    population layer) and the write goes through AnchorSet.write_anchor on
    ONE family, so each new segment_id remaps the prior anchor to inactive --
    the substrate's own plan-failure/invalidation write site. Segment ids
    carry their epoch, which is the retrieval ground truth.
  Phase B (probe). AFTER all epochs, each epoch's env is re-entered for
    REPROBE_STEPS to drive z_goal back toward that context, and the resulting
    FRESH z_goal is the cue. The cue is therefore NOT the stored vector --
    querying with the stored snapshot itself would score cosine 1.0 and
    measure nothing.
  Scoring. One query_by_goal_match(cue, threshold=-1.0) pass per cue in a
    fixed order, restricted to INACTIVE anchors. Top-1 identity accuracy =
    fraction of cues whose top-ranked inactive anchor comes from that cue's
    own epoch.

THE NULL IS MEASURED, NOT ASSUMED. Chance is not 1/N: the pool is unbalanced
and the top-1 anchor per cue is already fixed once the query pass is done.
So the null is the WRONG-CUE accuracy -- mean over ordered pairs (i, j), i
!= j, of 1[epoch(top1 for cue j) == epoch i] -- computed from the SAME score
matrix. A match function that merely amplifies noise has high score spread
and wrong-cue accuracy equal to true-cue accuracy; only a goal-INFORMATIVE
match separates them. This distinction is the whole point: V3-EXQ-807
established that the SD-079-centered cue restores goal_match SPREAD
(on_spread_mean 0.9985 vs off 0.0087), but spread is not selectivity.

WHY SD-079 CENTERING IS ON IN BOTH ARMS (instrumentation, not a treatment).
V3-EXQ-807 (diagnostic, PASS) measured the raw SD-039 cosine collapsing to a
spread of 0.0087 across an anchor pool, because z_goal is an EMA attractor
that concentrates the SD-008 common mode (measured z_goal cone min cosine
0.986). Reading SD-039's retrieval through the uncentered cosine would
measure that known instrument defect, not the claim. goal_cue_centering is
therefore held ON in BOTH arms so it cannot carry the contrast; the ONLY
thing that differs between arms is use_sd039_anchor_payload. SD-079 is not
tagged: it is not under test here.

GOV-REUSE-1 (Step 2.4). Decisive readout: post-failure top-1 goal-identity
retrieval accuracy of an inactive anchor under a fresh cue, with its
wrong-cue null. Checked V3-EXQ-494 (the only other SD-039-tagged run; no
substrate_hash, pre-standard, and it records only presence counts --
n_with_payload, n_query_results, mean_goal_match per hand-built phase -- with
no cue-identity structure, so accuracy is not derivable from it) and
V3-EXQ-807 (substrate_hash 959758ce3e..., SD-079-tagged; records
goal_match_spread / n_below_floor / gate counts under a SINGLE terminal cue,
so it has no cue x anchor matrix and no notion of a correct target). Spread
is recorded; IDENTITY is not, and cannot be derived from either. Partially
recoverable -> downgraded to exactly the missing piece: this run adds the
cue-identity matrix and its null, and banks the full per-cue top-1 provenance
so a successor need not re-run it.

DV-SYMMETRY (Step 3 mandatory declaration, per arm):
  ARM_OFF. DV = n_retrievable_inactive and top-1 identity accuracy. Symmetry
    group: permutation of the anchor pool (both are computed over an unordered
    pool) and any monotone rescaling of goal_match (top-1 is rank-based). The
    manipulation is NOT invariant under either: with use_sd039_anchor_payload
    False, build_goal_payload returns None, anchors carry goal_payload=None,
    and query_by_goal_match(threshold=-1.0) filters them out by presence
    BEFORE any score is taken (anchor_set.py, the `if anchor.goal_payload is
    not None` branch) -- so the retrievable set is empty rather than
    reordered. This is a presence contrast and is deliberately NOT the
    load-bearing criterion.
  ARM_ON true-cue vs wrong-cue. DV = top-1 identity accuracy. Symmetry group:
    permutation of the cue-to-target assignment. Here invariance IS the null
    hypothesis rather than a defect: if the centered match carries no
    goal-identity information, accuracy is exactly invariant under that
    permutation and C2's margin is 0. C2 asserts the manipulation is NOT
    invariant. Stated explicitly because this is the one place a permutation
    symmetry is intended rather than a hazard.
  Note on the statistic: a broadcast constant added to every anchor's score
    would cancel in top-1 (rank-based) but not in the raw spread. This run
    routes on top-1, so it is immune to that class; spread is recorded as a
    diagnostic only and is not load-bearing.

REFRESH-ON-INVALIDATE, and why it is recorded rather than designed around.
AnchorSet.write_anchor refreshes the OUTGOING anchor's payload with the
INCOMING one before marking it inactive (the documented SD-039
"cause-of-blockage payload"). So the LAST anchor of each epoch is preserved
carrying the NEXT epoch's snapshot, while every earlier anchor of that epoch
carries its own. With ANCHOR_EVERY over EPISODES_PER_EPOCH episodes that is
one boundary anchor out of many per epoch. It is not corrected for -- it is a
real property of the substrate under test -- but n_boundary_refreshed is
recorded per cell so a reader can see the size of the effect.

VERDICT ALIASING, and the precondition that breaks it. A chance-level
identity accuracy has TWO possible causes that produce an identical reading:
(i) the preserved payload carries no goal-identity information -- the real
falsification of SD-039's queryable-after-failure content; or (ii) the fresh
per-epoch cues did not actually differ, so every epoch presented the same
query and no retrieval function of any quality could have separated them --
a substrate-readiness gap, not a claim falsification. This is not
hypothetical: V3-EXQ-807 measured the z_goal cone at min pairwise cosine
0.986, i.e. z_goal barely varies at all, and the first smoke run of THIS
script returned true_cue_acc == wrong_cue_acc == 0.250 exactly, which is the
arithmetic signature of all four cues returning the SAME top-1 anchor.

So cue_goal_distinctness is a readiness precondition, measured on the
SD-079-CENTERED residual -- the same statistic goal_match routes on, not the
raw z_goal cosine and not a magnitude proxy for it -- against an upper
ceiling. Above the ceiling the run self-routes substrate_not_ready_requeue
and NEVER a verdict label. n_distinct_top1_anchors is recorded per cell so a
reader can see the collapse directly rather than inferring it from the two
accuracies being equal.

EPISODE-BOUNDARY POOL RESET -- why the rollout is continuous, and a recorded
scope limit on SD-039 itself. REEAgent.reset() calls
HippocampalModule.reset_anchor_set(), which calls AnchorSet.reset() and
CLEARS THE ENTIRE DUAL-TRACE POOL -- active and inactive alike (agent.py, the
reset_anchor_set() call in REEAgent.reset; anchor_set.py AnchorSet.reset
docstring: "Per-episode reset. Clears active + inactive anchors + tick").
Measured while building this run: with agent.reset() per episode the pool
returned to a single anchor at every episode boundary, so nothing survived to
be queried and the retrieval DV was structurally unmeasurable (the first
smoke run scored 0 inactive payload anchors for exactly this reason).

So this driver calls env.reset() per episode but NOT agent.reset(): the whole
cell is one continuous agent stream, and the "plan failure" events under test
are the substrate's OWN invalidation/remap write sites (write_anchor's
prior-active remap path), which is precisely what SD-039 names -- "at write or
invalidation time". Episode boundaries are a harness artefact and are not the
failure event the claim is about.

This is worth stating plainly because it BOUNDS the claim rather than
supporting it: whatever this run finds, SD-039's dual-trace preservation
currently holds only WITHIN a continuous agent stream. Across an
agent.reset() the preserved ghost-goal traces are destroyed, so any consumer
that expects unresolved goals to outlive an episode (MECH-292's bank ranks
over exactly this pool) gets an empty pool. That is a substrate observation,
not a result of this experiment, and it is recorded here so a governance
reader does not have to rediscover it.

READINESS PRECONDITIONS are scoped to ARM_ON only. ARM_OFF has zero anchors
with a payload BY CONSTRUCTION (that is what the control asserts), so
asserting "the inactive payload pool is non-empty" against it would be
structurally unsatisfiable and, under a whole-run AND, would vacate the
scorable arm -- the V3-EXQ-785 defect. ARM_OFF is disposition (a): the
preconditions are NOT MEANINGFUL for it, not failed by it. Only ARM_ON's
preconditions enter interpretation.preconditions (which the indexer reads
flat and arm-blind); the scope is stated in preconditions_scope_note.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_889_sd039_post_failure_goal_retrieval"
EXPERIMENT_PURPOSE = "evidence"

CLAIM_IDS = ["SD-039"]

SEEDS = [101, 202, 303]
ARMS = ["ARM_OFF", "ARM_ON"]

N_GOAL_EPOCHS = 4
EPISODES_PER_EPOCH = 3
STEPS_PER_EPISODE = 32
ANCHOR_EVERY = 8
REPROBE_STEPS = 16

# Total training episodes in ONE seed x arm cell -- must equal the denominator
# printed in the "[train] ... ep N/M" lines and the queue entry episodes_per_run.
EPISODES_PER_RUN = N_GOAL_EPOCHS * EPISODES_PER_EPOCH

FORCED_BENEFIT = 0.5
FORCED_DRIVE = 0.9

# ---- Pre-registered thresholds (constants; never derived from this run) ----
ACC_FLOOR = 0.60           # C2: ARM_ON true-cue top-1 identity accuracy
MARGIN_FLOOR = 0.25        # C2: true-cue accuracy minus wrong-cue accuracy
MIN_INACTIVE_PAYLOAD = 3   # C3 + readiness: inactive anchors carrying a payload
MIN_EPOCHS_REPRESENTED = 3 # readiness: distinct epochs present in the pool
ZGOAL_ACTIVE_FRAC_FLOOR = 0.50
# Readiness: the fresh per-epoch cues must actually DIFFER on the statistic
# retrieval consumes (the SD-079-centered residual). At or above this ceiling
# every epoch presents the same cue and identity retrieval is untestable --
# route substrate_not_ready_requeue, NEVER a falsification. See the
# VERDICT-ALIASING section of the module docstring.
CUE_DISTINCT_MAX_COSINE = 0.99
SEED_PASS_FRACTION = 2.0 / 3.0

# z_goal-stream liveness, pooled across arm x seed cells for the manifest block.
_ZG = ZGoalStreamAccumulator()


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(size=8, num_hazards=0, num_resources=6,
                             use_proxy_fields=True, seed=seed)


def _config_slice(payload_on: bool) -> Dict[str, Any]:
    """Declared config slice. Covers exactly what a cell's computation reads."""
    return {
        "env": {"size": 8, "num_hazards": 0, "num_resources": 6,
                "use_proxy_fields": True},
        "schedule": {"n_goal_epochs": N_GOAL_EPOCHS,
                     "episodes_per_epoch": EPISODES_PER_EPOCH,
                     "steps_per_episode": STEPS_PER_EPISODE,
                     "anchor_every": ANCHOR_EVERY,
                     "reprobe_steps": REPROBE_STEPS},
        "alpha_world": 0.9,
        "z_goal_enabled": True,
        "drive_weight": 2.0,
        "benefit_eval_enabled": True,
        "benefit_weight": 1.0,
        "forced_benefit": FORCED_BENEFIT,
        "forced_drive": FORCED_DRIVE,
        "use_anchor_sets": True,
        "use_sd039_anchor_payload": payload_on,
        # Instrumentation, identical in both arms -- see module docstring.
        "goal_cue_centering": True,
        "goal_cue_baseline_alpha": 0.05,
    }


def _max_pairwise_cosine(vecs: List[torch.Tensor]) -> float:
    """Max pairwise cosine over a list of (1, D) vectors. 1.0 when < 2 given."""
    if len(vecs) < 2:
        return 1.0
    m = torch.nn.functional.normalize(
        torch.cat([v.reshape(1, -1) for v in vecs], dim=0).float(), dim=-1)
    c = m @ m.t()
    n = c.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return float(c[iu[0], iu[1]].max())


def _epoch_of(segment_id: str) -> Optional[int]:
    """Ground-truth epoch from the segment id written in Phase A ('g<N>_...')."""
    if not segment_id.startswith("g"):
        return None
    head = segment_id[1:].split("_", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None


def _step_agent(agent: REEAgent, env: CausalGridWorldV2, obs: Dict[str, Any],
                wd: int) -> Tuple[Any, Dict[str, Any], bool]:
    """One env step through the full agent path. Returns (latent, obs, done)."""
    latent = agent.sense(obs["body_state"], obs["world_state"])
    ticks = agent.clock.advance()
    e1 = (agent._e1_tick(latent) if ticks.get("e1_tick")
          else torch.zeros(1, wd, device=agent.device))
    cands = agent.generate_trajectories(latent, e1, ticks)
    action = agent.select_action(cands, ticks)
    agent.update_z_goal(benefit_exposure=FORCED_BENEFIT, drive_level=FORCED_DRIVE)
    _, _h, done, _, obs = env.step(int(action.argmax(dim=-1).item()))
    return latent, obs, bool(done)


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    payload_on = (arm == "ARM_ON")
    print(f"Seed {seed} Condition {arm}", flush=True)

    with arm_cell(seed, config_slice=_config_slice(payload_on),
                  script_path=Path(__file__),
                  include_driver_script_in_hash=False) as cell:
        envs = [_build_env(seed * 100 + g) for g in range(N_GOAL_EPOCHS)]
        env0 = envs[0]
        cfg = REEConfig.from_dims(
            body_obs_dim=env0.body_obs_dim, world_obs_dim=env0.world_obs_dim,
            action_dim=4, alpha_world=0.9, z_goal_enabled=True, drive_weight=2.0,
            benefit_eval_enabled=True, benefit_weight=1.0,
            use_anchor_sets=True,
            use_sd039_anchor_payload=payload_on,
            goal_cue_centering=True, goal_cue_baseline_alpha=0.05,
        )
        agent = REEAgent(cfg)
        aset = agent.hippocampal.anchor_set
        wd = agent.config.latent.world_dim

        # ---------------- Phase A: write + invalidate across goal epochs -----
        episodes_done = 0
        n_payload_writes = 0
        for g in range(N_GOAL_EPOCHS):
            env = envs[g]
            for ep in range(EPISODES_PER_EPOCH):
                _, obs = env.reset()
                # NOTE: agent.reset() is deliberately NOT called -- see the
                # EPISODE-BOUNDARY POOL RESET section of the module docstring.
                latent = None
                for t in range(STEPS_PER_EPISODE):
                    latent, obs, done = _step_agent(agent, env, obs, wd)
                    if t % ANCHOR_EVERY == 0:
                        payload = agent.hippocampal.build_goal_payload(
                            latent_state=latent,
                            goal_state=agent.goal_state,
                            residue_field=agent.residue_field,
                            bla_output=agent._bla_last_output,
                            current_step=int(agent._step_count),
                            simulation_mode=False,
                        )
                        if payload is not None:
                            n_payload_writes += 1
                        # ONE family: each new segment_id remaps the prior
                        # anchor to inactive -- the substrate's own
                        # invalidation write site.
                        aset.write_anchor(
                            "fast", f"g{g}_e{ep}s{t}", ("world",),
                            latent.z_world.detach().reshape(-1),
                            goal_payload=payload,
                        )
                    if done:
                        break
                episodes_done += 1
                if episodes_done % 4 == 0:
                    print(f"  [train] goal seed={seed} arm={arm} "
                          f"ep {episodes_done}/{EPISODES_PER_RUN} "
                          f"anchors={len(aset.all_anchors())}", flush=True)

        # ---------------- Phase B: fresh cues by re-entering each epoch ------
        # The cue is a FRESH z_goal driven by re-exposure to the epoch's own
        # context, never the stored snapshot (which would score cosine 1.0).
        cues: List[Optional[torch.Tensor]] = []
        for g in range(N_GOAL_EPOCHS):
            env = envs[g]
            _, obs = env.reset()
            for _t in range(REPROBE_STEPS):
                _latent, obs, done = _step_agent(agent, env, obs, wd)
                if done:
                    _, obs = env.reset()
            zg = agent.goal_state.z_goal if agent.goal_state is not None else None
            cues.append(None if zg is None else zg.detach().reshape(1, -1).clone())

        # ---------------- Scoring --------------------------------------------
        inactive_keys = {a.key for a in aset.all_anchors() if not a.active}
        n_inactive = len(inactive_keys)
        n_inactive_payload = sum(
            1 for a in aset.all_anchors()
            if (not a.active) and a.goal_payload is not None
        )
        # Boundary anchors: last-written anchor of each epoch is refreshed with
        # the NEXT epoch's payload by write_anchor's remap path (docstring).
        n_boundary_refreshed = max(0, min(N_GOAL_EPOCHS - 1, n_inactive))

        # Cue distinctness on the SAME statistic retrieval consumes: the
        # SD-079-centered residual, not the raw z_goal. Measured BEFORE the
        # query pass so the query's own baseline advance cannot affect it.
        baseline = aset.goal_cue_baseline
        centered_cues = [
            (c.reshape(1, -1) if baseline is None
             else c.reshape(1, -1) - baseline.reshape(1, -1))
            for c in cues if c is not None
        ]
        cue_max_cos = _max_pairwise_cosine(centered_cues)

        top1_epoch: List[Optional[int]] = []
        spreads: List[float] = []
        n_scored: List[int] = []
        for g in range(N_GOAL_EPOCHS):
            cue = cues[g]
            pairs = aset.query_by_goal_match(cue, threshold=-1.0,
                                             simulation_mode=False)
            pairs = [(a, s) for (a, s) in pairs if a.key in inactive_keys]
            n_scored.append(len(pairs))
            if not pairs:
                top1_epoch.append(None)
                spreads.append(0.0)
                continue
            scores = torch.tensor([s for _, s in pairs])
            spreads.append(float(scores.max() - scores.min()))
            top1_epoch.append(_epoch_of(pairs[0][0].key[1]))

        epochs_represented = len({
            _epoch_of(a.key[1]) for a in aset.all_anchors()
            if (not a.active) and a.goal_payload is not None
            and _epoch_of(a.key[1]) is not None
        })

        n_cues_scored = sum(1 for e in top1_epoch if e is not None)
        if n_cues_scored > 0:
            true_acc = sum(
                1 for g in range(N_GOAL_EPOCHS)
                if top1_epoch[g] is not None and top1_epoch[g] == g
            ) / float(n_cues_scored)
            # Wrong-cue null over ordered pairs (i, j), i != j, from the SAME
            # score matrix: what accuracy would look like if the match carried
            # no goal-identity information.
            wrong = [
                1.0 if top1_epoch[j] == i else 0.0
                for i in range(N_GOAL_EPOCHS)
                for j in range(N_GOAL_EPOCHS)
                if i != j and top1_epoch[j] is not None
            ]
            wrong_acc = statistics.fmean(wrong) if wrong else 0.0
        else:
            true_acc = 0.0
            wrong_acc = 0.0

        row = {
            "arm": arm,
            "seed": seed,
            "n_anchors_total": len(aset.all_anchors()),
            "n_inactive": n_inactive,
            "n_inactive_with_payload": n_inactive_payload,
            "n_payload_writes": n_payload_writes,
            "n_boundary_refreshed": n_boundary_refreshed,
            "n_epochs_represented": epochs_represented,
            "n_cues_scored": n_cues_scored,
            "n_scored_per_cue": n_scored,
            "top1_epoch_per_cue": top1_epoch,
            "true_cue_top1_accuracy": true_acc,
            "wrong_cue_top1_accuracy": wrong_acc,
            "identity_margin": true_acc - wrong_acc,
            "goal_match_spread_mean": statistics.fmean(spreads) if spreads else 0.0,
            "n_cues_none": sum(1 for c in cues if c is None),
            "cue_centered_max_pairwise_cosine": cue_max_cos,
            "n_distinct_top1_anchors": len({
                e for e in top1_epoch if e is not None
            }),
        }
        cell.stamp(row)

    # z_goal liveness -- read AFTER this cell stepped; the agent is not retained.
    _ZG.observe(agent)
    passed = (row["identity_margin"] >= MARGIN_FLOOR and
              row["true_cue_top1_accuracy"] >= ACC_FLOOR) if payload_on else \
             (row["n_inactive_with_payload"] == 0)
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
    return row


def _worst_cell(rows: List[Dict[str, Any]], key: str, mode: str = "min"):
    fn = min if mode == "min" else max
    r = fn(rows, key=lambda x: x[key])
    return float(r[key]), f"{r['arm']}/seed{r['seed']}"


def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            rows.append(_run_cell(arm, seed))

    off = [r for r in rows if r["arm"] == "ARM_OFF"]
    on = [r for r in rows if r["arm"] == "ARM_ON"]

    # ---- Readiness preconditions: ARM_ON only (see module docstring) --------
    zg = _ZG.stats() or {}
    active_frac = zg.get("active_frac")
    active_frac_val = float(active_frac) if isinstance(active_frac, (int, float)) else 0.0
    zgoal_live = (
        isinstance(active_frac, (int, float))
        and active_frac_val >= ZGOAL_ACTIVE_FRAC_FLOOR
    )

    pool_worst, pool_cell = _worst_cell(on, "n_inactive_with_payload", mode="min")
    pool_ok = pool_worst >= MIN_INACTIVE_PAYLOAD

    epochs_worst, epochs_cell = _worst_cell(on, "n_epochs_represented", mode="min")
    epochs_ok = epochs_worst >= MIN_EPOCHS_REPRESENTED

    cue_worst, cue_cell = _worst_cell(
        on, "cue_centered_max_pairwise_cosine", mode="max")
    cue_distinct_ok = cue_worst < CUE_DISTINCT_MAX_COSINE

    readiness_ok = bool(zgoal_live and pool_ok and epochs_ok and cue_distinct_ok)

    # ---- Criteria -----------------------------------------------------------
    frac_c1 = sum(1 for r in off if r["n_inactive_with_payload"] == 0) / len(off)
    c1 = frac_c1 >= SEED_PASS_FRACTION

    frac_c2 = sum(
        1 for r in on
        if r["true_cue_top1_accuracy"] >= ACC_FLOOR
        and r["identity_margin"] >= MARGIN_FLOOR
    ) / len(on)
    c2 = (frac_c2 >= SEED_PASS_FRACTION) and readiness_ok

    frac_c3 = sum(
        1 for r in on if r["n_inactive_with_payload"] >= MIN_INACTIVE_PAYLOAD
    ) / len(on)
    c3 = (frac_c3 >= SEED_PASS_FRACTION) and readiness_ok

    # ---- Non-degeneracy -----------------------------------------------------
    # C2 discriminates only if the wrong-cue null is not itself saturated (a
    # null at 1.0 makes the margin unattainable and the criterion vacuous) and
    # the pool spans more than one epoch (single-epoch pool -> identity is
    # trivially satisfied or trivially impossible).
    null_not_saturated = all(r["wrong_cue_top1_accuracy"] < 1.0 for r in on)
    multi_epoch_pool = all(r["n_epochs_represented"] >= 2 for r in on)
    scores_move = all(r["goal_match_spread_mean"] > 0.0 for r in on)
    c2_nondeg = bool(null_not_saturated and multi_epoch_pool and scores_move
                     and readiness_ok)
    # C1 discriminates only if the ON arm really did populate (otherwise "OFF
    # has no payloads" is not news).
    on_populated = all(r["n_inactive_with_payload"] > 0 for r in on)
    c1_nondeg = bool(on_populated)
    c3_nondeg = bool(readiness_ok and multi_epoch_pool)

    overall = bool(c1 and c2 and c3)

    if not readiness_ok:
        label = "substrate_not_ready_requeue"
    elif overall:
        label = "sd039_payload_supports_post_failure_goal_identity_retrieval"
    else:
        label = "sd039_payload_does_not_support_post_failure_goal_identity_retrieval"

    non_degenerate = bool(readiness_ok and c2_nondeg)
    degeneracy_reason = ""
    if not readiness_ok:
        degeneracy_reason = (
            "ARM_ON readiness unmet (zgoal_live={} active_frac={:.3f}; "
            "inactive_payload_pool_worst={} at {}; epochs_represented_worst={} "
            "at {}; cue_centered_max_pairwise_cosine_worst={:.4f} at {} "
            "(ceiling {})) -- the retrieval DV was not exercised".format(
                zgoal_live, active_frac_val, int(pool_worst), pool_cell,
                int(epochs_worst), epochs_cell, cue_worst, cue_cell,
                CUE_DISTINCT_MAX_COSINE)
        )
    elif not c2_nondeg:
        degeneracy_reason = (
            "C2 vacuous: null_not_saturated={} multi_epoch_pool={} "
            "scores_move={}".format(null_not_saturated, multi_epoch_pool,
                                    scores_move)
        )

    metrics = {
        "on_true_cue_top1_accuracy_mean":
            statistics.fmean(r["true_cue_top1_accuracy"] for r in on),
        "on_wrong_cue_top1_accuracy_mean":
            statistics.fmean(r["wrong_cue_top1_accuracy"] for r in on),
        "on_identity_margin_mean":
            statistics.fmean(r["identity_margin"] for r in on),
        "on_identity_margin_worst": _worst_cell(on, "identity_margin", "min")[0],
        "on_n_inactive_with_payload_mean":
            statistics.fmean(r["n_inactive_with_payload"] for r in on),
        "off_n_inactive_with_payload_mean":
            statistics.fmean(r["n_inactive_with_payload"] for r in off),
        "on_goal_match_spread_mean":
            statistics.fmean(r["goal_match_spread_mean"] for r in on),
        "on_n_boundary_refreshed_mean":
            statistics.fmean(r["n_boundary_refreshed"] for r in on),
        "on_n_epochs_represented_worst": epochs_worst,
        "on_cue_centered_max_pairwise_cosine_worst": cue_worst,
        "on_cue_distinct_ok": cue_distinct_ok,
        "on_n_distinct_top1_anchors_mean":
            statistics.fmean(r["n_distinct_top1_anchors"] for r in on),
        "frac_c1_off_no_retrieval": frac_c1,
        "frac_c2_identity_above_null": frac_c2,
        "frac_c3_inactive_pool_sufficient": frac_c3,
        "c1_pass": c1, "c2_pass": c2, "c3_pass": c3,
        "zgoal_active_frac": active_frac_val,
        "readiness_ok": readiness_ok,
    }

    interpretation = {
        "label": label,
        "combination_rule": (
            "overall PASS = C1 AND C2 AND C3 (plain AND over all three). C2 and "
            "C3 additionally require the ARM_ON readiness gate. C1 is the "
            "bit-identical-OFF control and is NOT load-bearing."
        ),
        "preconditions": [
            {"name": "zgoal_writer_live",
             "description": ("z_goal stream is live across the run (writer "
                             "called and the benefit gate opened). A dead "
                             "z_goal makes every snapshot zero-init and the "
                             "retrieval DV unmeasurable."),
             "measured": active_frac_val,
             "threshold": ZGOAL_ACTIVE_FRAC_FLOOR,
             "direction": "lower",
             "control": ("pooled over every arm x seed cell via "
                         "ZGoalStreamAccumulator; the run drives update_z_goal "
                         "with a forced benefit above the gate threshold"),
             "met": bool(zgoal_live)},
            {"name": "inactive_payload_pool_size",
             "description": ("ARM_ON worst cell: inactive anchors carrying a "
                             "payload. Below the floor, top-1 over the pool is "
                             "structurally degenerate."),
             "measured": pool_worst,
             "threshold": float(MIN_INACTIVE_PAYLOAD),
             "direction": "lower",
             "control": ("worst ARM_ON cell ({}); counted on the agent's own "
                         "anchor set after Phase A".format(pool_cell)),
             "offending_cell": pool_cell,
             "met": bool(pool_ok)},
            {"name": "goal_epochs_represented",
             "description": ("ARM_ON worst cell: distinct goal epochs present "
                             "in the inactive payload pool. Identity retrieval "
                             "is undefined below two and underpowered below "
                             "the floor."),
             "measured": epochs_worst,
             "threshold": float(MIN_EPOCHS_REPRESENTED),
             "direction": "lower",
             "control": ("worst ARM_ON cell ({}); ground truth read from the "
                         "segment ids written in Phase A".format(epochs_cell)),
             "offending_cell": epochs_cell,
             "met": bool(epochs_ok)},
            {"name": "cue_goal_distinctness",
             "description": ("ARM_ON worst cell: max pairwise cosine between "
                             "the fresh per-epoch cues, measured on the "
                             "SD-079-centered residual -- the SAME statistic "
                             "goal_match routes on. At or above the ceiling "
                             "every epoch presents effectively one cue, so a "
                             "chance-level identity accuracy would mean 'the "
                             "cues did not differ', NOT 'retrieval is not "
                             "goal-informative'. Below-ceiling is what makes "
                             "C2 a test of the claim rather than of the cue."),
             "measured": cue_worst,
             "threshold": CUE_DISTINCT_MAX_COSINE,
             "direction": "upper",
             "control": ("worst ARM_ON cell ({}); cues are driven by "
                         "re-exposure to each epoch's own env, which is the "
                         "positive control for goal-context separation"
                         .format(cue_cell)),
             "offending_cell": cue_cell,
             "met": bool(cue_distinct_ok)},
        ],
        "preconditions_scope_note": (
            "All three preconditions are scoped to ARM_ON. ARM_OFF has zero "
            "anchors carrying a payload BY CONSTRUCTION -- that is exactly what "
            "the C1 control asserts -- so applying the pool/epoch gates to it "
            "would be structurally unsatisfiable and, under a whole-run AND, "
            "would vacate the scorable arm (the V3-EXQ-785 defect). ARM_OFF is "
            "disposition (a): NOT MEANINGFUL for these preconditions, not "
            "failed by them. ARM_OFF remains fully scored on C1."
        ),
        "criteria": [
            {"name": "C1_off_arm_no_retrieval", "load_bearing": False,
             "passed": c1},
            {"name": "C2_identity_accuracy_above_wrong_cue_null",
             "load_bearing": True, "passed": c2},
            {"name": "C3_retrieval_from_inactive_pool", "load_bearing": True,
             "passed": c3},
        ],
        "criteria_non_degenerate": {
            "C1_off_arm_no_retrieval": c1_nondeg,
            "C2_identity_accuracy_above_wrong_cue_null": c2_nondeg,
            "C3_retrieval_from_inactive_pool": c3_nondeg,
        },
    }

    return {
        "outcome": "PASS" if overall else "FAIL",
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": interpretation,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "last_agent_arm": "ARM_ON",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    global SEEDS, N_GOAL_EPOCHS, EPISODES_PER_EPOCH, EPISODES_PER_RUN
    if args.dry_run:
        SEEDS = [101]
        EPISODES_PER_EPOCH = 1
        EPISODES_PER_RUN = N_GOAL_EPOCHS * EPISODES_PER_EPOCH

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS, "arms": ARMS,
        "n_goal_epochs": N_GOAL_EPOCHS,
        "episodes_per_epoch": EPISODES_PER_EPOCH,
        "episodes_per_run": EPISODES_PER_RUN,
        "steps_per_episode": STEPS_PER_EPISODE,
        "anchor_every": ANCHOR_EVERY,
        "reprobe_steps": REPROBE_STEPS,
        "acc_floor": ACC_FLOOR,
        "margin_floor": MARGIN_FLOOR,
        "min_inactive_payload": MIN_INACTIVE_PAYLOAD,
        "min_epochs_represented": MIN_EPOCHS_REPRESENTED,
        "zgoal_active_frac_floor": ZGOAL_ACTIVE_FRAC_FLOOR,
        "seed_pass_fraction": SEED_PASS_FRACTION,
        "forced_benefit": FORCED_BENEFIT,
        "forced_drive": FORCED_DRIVE,
        "arm_config_slice_off": _config_slice(False),
        "arm_config_slice_on": _config_slice(True),
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if result["outcome"] == "PASS" else "weakens",
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
        "non_degenerate": result["non_degenerate"],
    }
    if result["degeneracy_reason"]:
        manifest["degeneracy_reason"] = result["degeneracy_reason"]
    if not result["non_degenerate"]:
        manifest["evidence_direction"] = "unknown"

    out_path = write_flat_manifest(
        manifest,
        Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments",
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    m = result["metrics"]
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"ON true_cue_acc={m['on_true_cue_top1_accuracy_mean']:.3f} "
          f"wrong_cue_acc={m['on_wrong_cue_top1_accuracy_mean']:.3f} "
          f"margin={m['on_identity_margin_mean']:.3f}", flush=True)
    print(f"ON inactive_with_payload={m['on_n_inactive_with_payload_mean']:.2f} | "
          f"OFF inactive_with_payload={m['off_n_inactive_with_payload_mean']:.2f}",
          flush=True)
    print(f"C1={m['c1_pass']} C2={m['c2_pass']} C3={m['c3_pass']} "
          f"readiness={m['readiness_ok']} "
          f"non_degenerate={result['non_degenerate']}", flush=True)
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
