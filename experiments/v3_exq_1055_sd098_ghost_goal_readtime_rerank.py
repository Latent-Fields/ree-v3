"""V3-EXQ-1055: SD-098 read-time goal-ness -- ghost-goal-bank re-ranking under a
mid-episode destination shift WITH REVERSAL (EVIDENCE).

experiment_purpose: evidence
RED-TEAM (opus, 2026-09-17): the forward-only version of this design was returned
BLOCKING and is NOT what this script implements. See "WHAT THE RED-TEAM KILLED".

WHAT THIS TESTS, AND THE SCOPE LIMIT (read before reading any result)
---------------------------------------------------------------------
SD-098 asserts goal-ness / subgoal-ness is a RELATIONAL property computed at READ
TIME from the conjunction of current state, active commitments and the currently
privileged destination -- not a node-type field stored on a topology node.

SD-098 is registered substrate_conditional on SD-097 (a typed possibility
topology), which does NOT exist. USER DECISION 2026-09-17 (Orchestrator decision
lane, session orchestrate-20260917-1532): do NOT build SD-097's typed topology;
falsify SD-098 against the EXISTING ghost-goal bank instead.

The substitution is principled. GhostGoalBank.rank()
(ree_core/hippocampal/ghost_goal_bank.py) already IS SD-098's mechanism at ONE
relation: an explicitly read-only per-call consumer computing each anchor's
goal-relevance from current_z_goal x stored payload, with NO stored node type
written anywhere. The SD-039 anchor pool stands in for SD-097's node set.

SECOND SCOPE LIMIT -- DESTINATION PRIVILEGE IS NOT SEPARATED FROM SENSORY CONTEXT
(red-team pass 2, finding F5; accepted, not fixed). Because the policy SEEKS the
active destination, returning the destination to D1 also returns the agent to D1's
sensory neighbourhood. So a positive result shows that goal-relevance is RECOMPUTED
AT READ TIME from the current state -- it does NOT isolate the "privileged
destination" term in SD-098's conjunction from plain context reinstatement.
The one condition that WOULD separate them (relocate the cluster without moving the
agent) is exactly the condition under which the manipulation is too weak to measure
at all: measured, it gives a paired cue_return of +0.003, positive 4/7. That is a
real limit of this substrate, not an oversight, and it is why the PASS label says
"read-time recomputation" rather than "destination privilege".
The confirmer is recorded rather than run here: per_seed carries both
contrast_return_minus_stay and the per-arm phase_c_occupancy, so an autopsy can
rank-correlate them directly.

SCOPE LIMIT -- HALF (i) ONLY. claims.yaml SD-098 what_would_answer has two halves:
  (i)  the relational implementation re-classifies a state S under a destination
       shift WITHOUT any write to S's stored attributes;   <- TESTED HERE
  (ii) a matched STORED-NODE-TYPE implementation needs an explicit re-label write
       to achieve the same.                                <- NOT TESTED HERE
Half (ii) requires the stored-node-type comparison arm, i.e. exactly the SD-097
schema commitment the user declined. A PASS here is PARTIAL support and does NOT
close SD-098. A clean FAIL is the more decisive outcome: it says the substrate's
one candidate read-time mechanism does not re-classify under a destination shift,
which is what would justify revisiting SD-097's node schema.

WHAT THE RED-TEAM KILLED, AND WHY THIS DESIGN IS DIFFERENT
-----------------------------------------------------------
The first draft was a FORWARD-ONLY shift (phase A at D1, phase B at D2), scoring
AUC over anchors grouped by which phase wrote them, against a same-region "sham"
relocation. It was returned BLOCKING for a reason that survives verification:

  z_goal is a fixed-alpha EMA pulled toward z_world every tick (goal.py:912-918),
  and each anchor's z_goal_snapshot is that same EMA sampled at its write step. So
  an anchor's goal_match to a later cue is a near-monotone function of WHEN it was
  written -- and payload_written_step, the provenance label, is a perfect proxy for
  that. The AUC was therefore a CUE-DISPLACEMENT METER, not a content meter, and in
  a forward-only design "the destination changed" and "the cue moved further" are
  collinear BY CONSTRUCTION. The sham arm could not absorb this (and, separately,
  that sham shared 4 of its 6 resource cells with the baseline layout, so it was
  not even a matched disruption).

THE FIX IS THE REVERSAL, and it is the whole point of this design. Three phases:

    phase A at D1  ->  phase B at D2  ->  phase C at D1 (RETURN) or D2 (STAY)

The DV compares only the two OLD groups, A and B, under the phase-C cue:

  (Measured on tuning: the STAY control lands at AUC 0.34-0.64, i.e. ~0.5, exactly
  the recency null; RETURN lands at 0.993-1.000. The dissociation is the result.)

  RECENCY / displacement account: group A is strictly OLDER than group B, and the
    cue only ever moves forward along its trajectory, so A can only LOSE rank to B.
    This account predicts AUC(A beats B) <= 0.5 in BOTH arms, and predicts no
    difference between them beyond displacement magnitude.
  CONTENT account (SD-098): when the destination RETURNS to D1, the D1-flavoured
    group A regains rank over group B DESPITE BEING OLDER, because role is computed
    at read time from the currently privileged destination.

These predictions are OPPOSITE, which is exactly what the forward-only design could
not arrange. A rank gain that is non-monotone in write time cannot be produced by
trajectory position.

STAY is a genuinely matched control, unlike the killed sham: both arms share
IDENTICAL phases A and B, and phase C has identical length, identical boundary
schedule and an identical env.reset_to relocation event -- the arms differ ONLY in
which region phase C's resources occupy. The same action sequence is replayed in
both arms of a seed.

OTHER RED-TEAM FINDINGS, ALL FIXED HERE
----------------------------------------
F2 top_k: GhostGoalBankConfig.top_k defaults to 32 and caps the RETURNED list, so
   at ~45 anchors the phase-A/B groups were truncated away entirely (measured: the
   groups came back EMPTY). top_k is set to None here and ASSERTED after
   construction -- REEConfig.from_dims silently swallows unknown kwargs
   (reference-reeconfig-from-dims-silent-kwargs), so a set-and-hope would be a
   silent no-op. The cap is a consumer convenience, not part of the ranking
   computation, so uncapping changes no scoring semantics.
   The red-team also noted that intersecting the two ranked lists DISCARDS the
   anchors that enter or leave the bank between cues -- which is literally SD-098's
   predicted re-classification. Those are no longer silently dropped: they are
   counted and recorded (bank_entries / bank_exits / churn) as first-class
   diagnostics, and max_abs_gain is recorded so the zero-sum assumption is auditable.
F3 self-certifying gate: the old goal_channel_authority gate was built from the
   same per-anchor quantity that drives rank and was read off the manipulated arm
   only -- the unmanipulated arm scored 7x its floor, so it could not be a
   manipulation check. It is replaced by cue_return_achieved, which asks whether
   the cue actually moved BACK toward the phase-A reference: a check the STAY arm
   fails by construction and the RETURN arm can genuinely fail.
F4 no instrument-not-ready branch: a run whose manipulation never reached the DV
   used to route `weakens`. It now routes manipulation_did_not_reach_dv ->
   non_contributory.
F5 reps[0]: the anchor-supply gate read the first replicate while all replicates
   were scored. It now reports the WORST cell across replicates.
F6 verdict lines: `verdict:` was printed once per cell chosen by a NaN test, into
   a stream the runner scrapes. It now reflects the cell's actual readiness.

WHY simulation_mode=True ON THE PROBE. GhostGoalBank.rank() calls
anchor_set.observe_goal_cue(), which ADVANCES the SD-079 slow-EMA common-mode
baseline unless simulation_mode=True (MECH-094: simulation cues must not shape
waking cue geometry). Probing via HippocampalModule.rank_ghost_goals(), which does
not expose the flag, would centre cue_mid against baseline_t and cue_end against
baseline_t+1 and confound the paired comparison. simulation_mode=True holds the
baseline fixed so BOTH cues are centred identically, and makes the probe a
genuinely side-effect-free read -- which is itself SD-098 half (i)'s assertion.

WHY goal_cue_centering=True. z_goal inherits and amplifies the SD-008 common-mode
offset (measured here: raw cos(cue_mid, cue_end) ~0.99 against a centred ~0.95).
Without SD-079 centring the goal channel is effectively dead and every arm reads
invariant for a representational reason. Same deliberate choice as V3-EXQ-868.

WHY NO E3, AND WHY THE POLICY NONETHELESS SEEKS. Step 2.5c substrate-path gate:
substrate_queue SD-082 is OPEN (implemented_pending_validation) and CORRUPTING on
ree_core/predictors/e3_selector.py. This driver never calls agent.select_action(),
so that path is never entered -- asserted per cell (e3_never_entered), not assumed.

But a uniform-random policy turned out to be too weak an instrument: the agent
never occupies the destination region, so the destination is barely encoded in
z_world/z_goal at all, and the paired manipulation check came back +0.003
(positive 4/7) -- i.e. the manipulation was NOT certifiable and any result would
have been unattributable. The policy therefore SEEKS the active destination's
centroid with EPSILON_EXPLORE random steps. It is still fully scripted -- a
hand-written greedy step, no selector, no learning -- so SD-082 stays untouched,
while phase-C occupancy rises to ~0.9 and the paired cue_return difference to
+0.55 (positive 7/7). Occupancy is recorded per cell as the env-level,
DV-independent record that the manipulation was applied in the world.

resource_respawn_on_consume=False is REQUIRED, not cosmetic: _respawn_resource()
spawns at a RANDOM interior cell, dissolving the relocated cluster within a few
consumptions. Measured: with respawn ON, relocated and unrelocated arms were
indistinguishable.

NOTE ON THE [smoke] z_goal_stream LINE. This driver reports
`active_frac=unmeasured ticks=0/0` with writer_calls>0 and goal_state_present=True.
That is EXPECTED, not a WRITER DEFECT: the tick counter is incremented inside
REEAgent.select_action(), which this driver deliberately never calls. update_z_goal
IS called every tick -- writer_calls is the field that shows it.

PRE-REGISTRATION AND THE TUNING/EVALUATION SPLIT
-------------------------------------------------
All tuning (phase lengths, boundary interval, replicate count, layouts, thresholds)
used TUNING seeds 0-6. This runs on EVALUATION seeds 100-114, DISJOINT, so every
threshold is out-of-sample. Tuning values are recorded under tuning_reference for
audit and are NOT evidence. (Disclosure: seeds 100 and 101 were executed end-to-end during
readiness/timing checks, so their contrasts were OBSERVED; both have therefore
been REMOVED from the evaluation set, which runs 102-116. No threshold was
derived from either.)

Pre-registered criteria:
  C1 structural   re-ranking occurs at all; recorded, NOT gating (the red-team
                  showed an inversion count is pinned true by membership churn).
  C2 LOAD-BEARING mean(AUC_return - AUC_stay) over GREEN seeds >= 0.10
                  AND positive in >= ceil(2/3 * n_green) seeds.
                  0.03 is ~2.3 standard errors above zero at n=15 given the tuning
                  spread (sd 0.050), and ~1/3 of the tuning effect (0.090) -- set
                  from the null, not fitted to the effect.
  C3 LOAD-BEARING ZERO writes to the anchor pool across both probe ranks (payload
                  fingerprint byte-identical) -- SD-098 half (i)'s "without any
                  explicit write". Recorded as structural: it is an API property
                  identical in both arms, so it cannot carry the verdict alone.
  combination_rule: PASS iff (C2 AND C3).

DV-SYMMETRY INVARIANCE (per arm, as required at design audit):
  Shared DV: AUC over normalized rank gains of group A vs group B -- a RANK-based
  statistic. Symmetry group: monotone transforms of ghost_priority applied uniformly
  within a cue (rank-preserving), and permutation within a provenance group.
  RETURN arm: the manipulation returns the privileged destination to D1, changing
    each anchor's goal_match by a SNAPSHOT-DEPENDENT, NON-UNIFORM amount (tuning:
    delta_goal_match range 0.46 vs mean_abs 0.05 -- emphatically not a broadcast
    constant, not a monotone rescaling). The DV is NOT invariant under it.
    Critically, the predicted direction is NON-MONOTONE IN WRITE TIME, which no
    uniform displacement of the cue can produce.
  STAY arm: the destination is deliberately held invariant while phase-C length,
    boundary schedule and relocation event are not. It is the subtrahend in C2 and
    is SCORED, never scoped out.

Readiness preconditions are evaluated PER SEED and are NOT ANDed whole-run
(V3-EXQ-785): a failing seed is scoped out of scoring and never counted as evidence
against SD-098. Fewer than MIN_GREEN_SEEDS green -> substrate_not_ready_requeue,
which is NOT a claim verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "sd098_ghost_goal_readtime_rerank"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["SD-098"]

# --- pre-registered constants (fixed on TUNING seeds 0-6; never re-derived) ---
EVAL_SEEDS: Tuple[int, ...] = tuple(range(102, 117))   # 15, disjoint from tuning
TUNING_SEEDS: Tuple[int, ...] = tuple(range(0, 7))
REPLICATES = 8
PHASE_TICKS = 90            # each of phases A, B, C
BOUNDARY_EVERY = 6
FORCE_BENEFIT = 0.5
FORCE_DRIVE = 0.9
GRID_SIZE = 8
# Scripted REGION-SEEKING policy. With a uniform-random policy the agent never
# occupies the destination region, so "the destination" is barely encoded in
# z_world/z_goal at all and the manipulation is not certifiable (measured: paired
# cue_return difference +0.003, positive 4/7). Seeking the active destination
# centroid raises phase-C occupancy to ~0.9 and the paired cue_return difference
# to +0.55, positive 7/7. This is still a SCRIPTED policy -- agent.select_action()
# is never called, so e3_selector (SD-082, open+corrupting) stays untouched.
EPSILON_EXPLORE = 0.3
DEST_REGION_RADIUS = 2      # Manhattan radius counted as "in the destination region"

D1: List[Tuple[int, int]] = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]
D2: List[Tuple[int, int]] = [(6, 6), (6, 5), (5, 6), (5, 5), (6, 4), (4, 6)]
HAZARDS: List[Tuple[int, int]] = [(3, 3)]
AGENT_START: Tuple[int, int] = (4, 2)

# Arms differ ONLY in phase C's destination.
ARMS: Tuple[Tuple[str, List[Tuple[int, int]]], ...] = (
    ("STAY", D2),     # control: destination held at D2
    ("RETURN", D1),   # manipulation: destination returns to D1
)

# Readiness floors (per seed)
MIN_GROUP_A_ANCHORS = 5
MIN_GROUP_B_ANCHORS = 5
MIN_GREEN_SEEDS = 8
# dv_headroom: contrast <= 1 - AUC_stay, so a STAY arm at the ceiling makes the C2
# bar unreachable BY CONSTRUCTION. Guards the arm that is actually the subtrahend.
# F1 (red-team pass 2): contrast <= 1 - AUC_stay = dv_headroom BY CONSTRUCTION, so a
# headroom floor BELOW the C2 bar admits seeds whose MAXIMUM attainable contrast is
# under the bar -- they can only drag the mean down and manufacture a `weakens`.
# Pinned equal to C2_MEAN_CONTRAST_FLOOR (0.10) so a green seed can always, in
# principle, reach the bar. Tuning headroom was 0.36-0.66, so this costs nothing there.
DV_HEADROOM_FLOOR = 0.10
# cue_return_achieved, PAIRED. The unpaired form does NOT discriminate the arms --
# the cue drifts back toward the phase-A reference in BOTH (measured +0.095 RETURN
# vs +0.093 STAY). The PAIRED difference does: the control arm sits at ~0 by
# construction. Floor 0.10 is ~2x the control arm's own worst drift (|0.05|);
# tuning min was +0.341.
CUE_RETURN_DIFF_FLOOR = 0.10
# F3 (red-team pass 2): cue_return is NOT independent of the DV -- it is a similarity
# to cue_ref_a, the phase-A z_goal, which is a proxy for the very group-A snapshot
# cloud that drives group A's rank gain. Gating on it makes green selection
# hypothesis-aligned. The manipulation gate is therefore phase-C OCCUPANCY, measured
# in the ENVIRONMENT and independent of anything the bank computes; cue_return is
# demoted to a recorded diagnostic. Tuning occupancy was 0.86-0.99.
PHASE_C_OCCUPANCY_FLOOR = 0.60

# A 0.10 shift in P(group A beats group B) is the smallest reinstatement worth
# calling content-specific. Deliberately conservative: tuning measured 0.469.
C2_MEAN_CONTRAST_FLOOR = 0.10
C2_CONSISTENCY_FRACTION = 2.0 / 3.0

TUNING_REFERENCE = {
    "note": "measured on TUNING seeds 0-6 during design; NOT evidence",
    "auc_stay_mean": 0.518,
    "auc_stay_range": "0.482-0.554",
    "auc_return_mean": 0.988,
    "contrast_mean": 0.470,
    "contrast_sd": 0.019,
    "dv_headroom_range": "0.446-0.518",
    "note_control_sits_on_the_exact_null": (
        "after the F6b fix (ranking WITHIN A-union-B, so group-C anchors cannot crowd "
        "either group), the STAY control lands at 0.482-0.554 -- the theoretical 0.500 "
        "null of the recency account, which is the strongest available evidence that "
        "the DV behaves as designed"),
    "contrast_positive_seeds": "7/7",
    "cue_return_diff_mean": 0.548,
    "cue_return_diff_min": 0.341,
    "cue_return_diff_positive_seeds": "7/7",
    "phase_c_occupancy": "0.86-1.00",
    "group_sizes": "n_A 5-7, n_B 14",
    "note_stay_arm_sits_at_null": (
        "STAY AUC ~0.5 -- exactly what the recency/displacement account predicts when "
        "the destination does not return"),
    "note_return_arm_saturates": (
        "RETURN AUC 0.993-1.000 is at ceiling, so the effect size is ceiling-limited; "
        "the contrast is protected because the STAY arm has ample headroom"),
    "superseded_random_policy": (
        "with a uniform-random policy the agent never occupied the destination, "
        "phase-C occupancy was incidental, and the paired cue_return difference was "
        "+0.003 (4/7) -- the manipulation was not certifiable"),
    "superseded_forward_only_design": (
        "returned BLOCKING by red-team; its null floor (mean |delta| 0.298, sd 0.325) "
        "exceeded its own effect (0.144), i.e. it could have returned a false supports"
    ),
}


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=len(D1),
        num_hazards=len(HAZARDS),
        hazard_harm=0.1,
        resource_benefit=0.5,
        # MUST stay False -- _respawn_resource() spawns at a RANDOM interior cell
        # and would dissolve the relocated cluster, erasing the manipulation.
        resource_respawn_on_consume=False,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.05,
        proximity_approach_threshold=0.15,
        use_proxy_fields=True,
    )


def _build_config(env: CausalGridWorldV2) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        use_per_stream_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        use_sd039_anchor_payload=True,
        use_mech292_ghost_bank=True,
        goal_cue_centering=True,   # SD-079: REQUIRED (see module docstring)
        drive_weight=2.0,
        use_resource_proximity_head=True,
    )
    # Must precede construction: REEAgent.__init__ builds goal_state from this flag.
    cfg.goal.z_goal_enabled = True
    # F2 fix. top_k caps the RETURNED list only; the full pool is still scored. The
    # default 32 truncates the phase-A/B groups away entirely at this pool size.
    # ASSERTED because from_dims silently swallows unknown kwargs.
    cfg.hippocampal.ghost_goal_bank_config.top_k = None
    assert cfg.hippocampal.ghost_goal_bank_config.top_k is None, (
        "ghost_goal_bank_config.top_k did not take -- the phase-A/B groups would be "
        "truncated away and every AUC would be NaN"
    )
    return cfg


def _config_slice(arm_name: str, phase_c: Sequence[Tuple[int, int]],
                  *, phase_ticks: int, boundary_every: int,
                  replicates: int) -> Dict[str, Any]:
    """Declared config slice for the arm fingerprint. The phase-C layout MUST be in
    it -- it is the only thing that differs between arms."""
    return {
        "grid_size": GRID_SIZE,
        "phase_ticks": phase_ticks,
        "boundary_every": boundary_every,
        "replicates": replicates,
        "force_benefit": FORCE_BENEFIT,
        "force_drive": FORCE_DRIVE,
        "phase_a_resources": [list(c) for c in D1],
        "phase_b_resources": [list(c) for c in D2],
        "phase_c_resources": [list(c) for c in phase_c],
        "arm": arm_name,
        "hazards": [list(c) for c in HAZARDS],
        "agent_start": list(AGENT_START),
        "goal_cue_centering": True,
        "ghost_bank_top_k": None,
        "resource_respawn_on_consume": False,
        "policy": "scripted_region_seeking_no_e3",
        "epsilon_explore": EPSILON_EXPLORE,
        "dest_region_radius": DEST_REGION_RADIUS,
    }


def _run_phase(agent: REEAgent, env: CausalGridWorldV2, obs_dict: Dict[str, Any],
               rng: np.random.Generator, tick: int, n_ticks: int,
               boundary_every: int, resources: Sequence[Tuple[int, int]]
               ) -> Tuple[Dict[str, Any], int, float]:
    """One phase of the continuous stream. NEVER calls agent.reset(), and NEVER
    calls agent.select_action() (so e3_selector is never entered).

    The policy seeks the active destination's centroid with EPSILON_EXPLORE random
    steps. Returns (obs, tick, occupancy) where occupancy is the fraction of ticks
    spent within DEST_REGION_RADIUS of that centroid -- the env-level, DV-independent
    record that the manipulation was actually applied in the world.
    """
    tx = sum(c[0] for c in resources) / len(resources)
    ty = sum(c[1] for c in resources) / len(resources)
    in_region = 0
    for i in range(n_ticks):
        latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
        agent.clock.advance()
        agent.update_z_goal(benefit_exposure=FORCE_BENEFIT, drive_level=FORCE_DRIVE)
        if i > 0 and (i % boundary_every) == 0:
            ev = agent.hippocampal.event_segmenter.force_boundary(
                "fast", reason=f"v3_exq_1055_t{tick}")
            payload = agent.hippocampal.build_goal_payload(
                latent_state=latent,
                goal_state=agent.goal_state,
                residue_field=agent.residue_field,
                bla_output=agent._bla_last_output,
                current_step=tick,      # our own monotone tick == provenance label
                simulation_mode=False,
            )
            agent.hippocampal.tick_anchor_set(latent, [ev], goal_payload=payload)
        if rng.random() < EPSILON_EXPLORE:
            act_idx = int(rng.integers(env.action_dim))
        else:
            dx, dy = tx - env.agent_x, ty - env.agent_y
            if abs(dx) >= abs(dy):
                act_idx = 0 if dx < 0 else (1 if dx > 0 else 4)
            else:
                act_idx = 2 if dy < 0 else (3 if dy > 0 else 4)
        action = torch.zeros(1, env.action_dim)
        action[0, act_idx] = 1.0
        agent._last_action = action
        _, _, done, _, obs_dict = env.step(action)
        if abs(env.agent_x - tx) + abs(env.agent_y - ty) <= DEST_REGION_RADIUS:
            in_region += 1
        if done:
            _, obs_dict = env.reset_to(
                agent_pos=(env.agent_x, env.agent_y),
                hazard_positions=list(HAZARDS),
                resource_positions=[tuple(c) for c in resources],
            )
        tick += 1
    return obs_dict, tick, (in_region / max(1, n_ticks))


def _pool_fingerprint(agent: REEAgent) -> str:
    """Content hash of every stored goal payload -- the C3 no-write check."""
    def _num(v: Any) -> str:
        return "none" if v is None else f"{float(v):.9g}"

    parts: List[str] = []
    for a in agent.hippocampal.anchor_set.all_anchors():
        p = a.goal_payload
        if p is None:
            parts.append("none")
            continue
        snap = p.z_goal_snapshot
        snap_h = (
            hashlib.sha256(snap.detach().reshape(-1).float().numpy().tobytes()).hexdigest()
            if snap is not None else "nosnap"
        )
        parts.append("|".join([
            snap_h, _num(p.wanting_strength), _num(p.arousal_tag), _num(p.last_vs),
            _num(p.staleness_at_write), str(int(p.payload_written_step or 0)),
            str(bool(a.active)), str(int(a.last_accessed)),
        ]))
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def _auc(xs: Sequence[float], ys: Sequence[float]) -> float:
    """P(a draw from xs exceeds a draw from ys); 0.5 = null."""
    if not xs or not ys:
        return float("nan")
    wins = 0.0
    for a in xs:
        for b in ys:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(xs) * len(ys))


def _cos(a: torch.Tensor, b: torch.Tensor,
         baseline: Optional[torch.Tensor] = None) -> float:
    x = a.detach().reshape(-1).float()
    y = b.detach().reshape(-1).float()
    if baseline is not None:
        z = baseline.detach().reshape(-1).float()
        if z.numel() == x.numel():
            x = x - z
            y = y - z
    if x.norm().item() < 1e-9 or y.norm().item() < 1e-9:
        return float("nan")
    return float(torch.nn.functional.cosine_similarity(
        x.unsqueeze(0), y.unsqueeze(0)).item())


def _mean(xs: Sequence[float]) -> float:
    vals = [x for x in xs if x == x]
    return (sum(vals) / len(vals)) if vals else float("nan")


def _pstdev(xs: Sequence[float]) -> float:
    vals = [x for x in xs if x == x]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def _run_replicate(seed: int, phase_c: Sequence[Tuple[int, int]], action_stream: int,
                   *, phase_ticks: int, boundary_every: int) -> Dict[str, Any]:
    rng = np.random.default_rng(action_stream)
    env = _make_env(seed)
    cfg = _build_config(env)
    torch.manual_seed(seed)
    agent = REEAgent(cfg)

    _, obs = env.reset_to(agent_pos=AGENT_START, hazard_positions=list(HAZARDS),
                          resource_positions=[tuple(c) for c in D1])
    agent.reset()

    obs, tick, _occ_a = _run_phase(agent, env, obs, rng, 0, phase_ticks,
                                   boundary_every, D1)
    end_a = tick
    cue_ref_a = agent.goal_state.z_goal.detach().clone()   # phase-A reference

    _, obs = env.reset_to(agent_pos=(env.agent_x, env.agent_y),
                          hazard_positions=list(HAZARDS),
                          resource_positions=[tuple(c) for c in D2])
    obs, tick, _occ_b = _run_phase(agent, env, obs, rng, tick, phase_ticks,
                                   boundary_every, D2)
    end_b = tick
    cue_mid = agent.goal_state.z_goal.detach().clone()

    # ---- phase C: the arms diverge here and ONLY here ---------------------- #
    _, obs = env.reset_to(agent_pos=(env.agent_x, env.agent_y),
                          hazard_positions=list(HAZARDS),
                          resource_positions=[tuple(c) for c in phase_c])
    obs, tick, occ_c = _run_phase(agent, env, obs, rng, tick, phase_ticks,
                                  boundary_every, phase_c)
    cue_end = agent.goal_state.z_goal.detach().clone()

    bank = agent.hippocampal.ghost_goal_bank
    aset = agent.hippocampal.anchor_set
    baseline = aset.goal_cue_baseline

    # ---- C3: the probe must not write anything ----------------------------- #
    fp_before = _pool_fingerprint(agent)
    # simulation_mode=True: does NOT advance the SD-079 baseline, so BOTH cues are
    # centred against the SAME baseline (see module docstring).
    bank_mid = bank.rank(cue_mid, simulation_mode=True)
    bank_end = bank.rank(cue_end, simulation_mode=True)
    no_write = (fp_before == _pool_fingerprint(agent))

    rank_mid = {id(e.anchor): i for i, e in enumerate(bank_mid)}
    rank_end = {id(e.anchor): i for i, e in enumerate(bank_end)}
    anchors = {id(e.anchor): e.anchor for e in bank_mid}
    anchors.update({id(e.anchor): e.anchor for e in bank_end})
    common = sorted(set(rank_mid) & set(rank_end))

    # F6 (red-team pass 2): group-C anchors are excluded from the DV but were NOT
    # excluded from the RANKING, so they crowded group B in STAY and group A in
    # RETURN -- oppositely signed and unmeasured. Rank WITHIN the A-union-B subset
    # so group C cannot displace either group.
    def _grp(key: int) -> str:
        st = anchors[key].goal_payload.payload_written_step or 0
        return "A" if st < end_a else ("B" if st < end_b else "C")

    ab_mid = [k for k, _ in sorted(
        ((k, rank_mid[k]) for k in rank_mid if _grp(k) in ("A", "B")),
        key=lambda kv: kv[1])]
    ab_end = [k for k, _ in sorted(
        ((k, rank_end[k]) for k in rank_end if _grp(k) in ("A", "B")),
        key=lambda kv: kv[1])]
    pos_mid = {k: i for i, k in enumerate(ab_mid)}
    pos_end = {k: i for i, k in enumerate(ab_end)}
    ab_common = sorted(set(pos_mid) & set(pos_end))
    denom = max(1, len(ab_common) - 1)

    # F2: entries/exits are the re-classification events SD-098 predicts. They are
    # excluded from the paired AUC (undefined rank on one side) but RECORDED, not
    # silently discarded.
    entries = sorted(set(rank_end) - set(rank_mid))
    exits = sorted(set(rank_mid) - set(rank_end))

    gain_a: List[float] = []
    gain_b: List[float] = []
    inversions = 0
    max_abs_gain = 0.0
    for k in ab_common:
        gain = (pos_mid[k] - pos_end[k]) / denom
        max_abs_gain = max(max_abs_gain, abs(gain))
        if pos_mid[k] != pos_end[k]:
            inversions += 1
        (gain_a if _grp(k) == "A" else gain_b).append(gain)
    n_group_c = sum(1 for k in common if _grp(k) == "C")

    def _entry_group(keys: Sequence[int]) -> Dict[str, int]:
        out = {"A": 0, "B": 0, "C": 0}
        for k in keys:
            st = anchors[k].goal_payload.payload_written_step or 0
            out["A" if st < end_a else ("B" if st < end_b else "C")] += 1
        return out

    e3_never_entered = (
        agent._last_e3_selection_result is None
        and not getattr(agent.e3, "last_score_diagnostics", {})
    )

    return {
        "auc_a_over_b": _auc(gain_a, gain_b),
        "n_group_a": len(gain_a),
        "n_group_b": len(gain_b),
        "n_group_c": int(n_group_c),
        "n_common": len(common),
        "n_bank_mid": len(bank_mid),
        "n_bank_end": len(bank_end),
        "n_bank_entries": len(entries),
        "n_bank_exits": len(exits),
        "bank_entries_by_group": _entry_group(entries),
        "bank_exits_by_group": _entry_group(exits),
        "max_abs_gain": float(max_abs_gain),
        "n_inversions": int(inversions),
        # cue_return_achieved: did the cue move BACK toward the phase-A reference?
        "cos_mid_to_ref_a": _cos(cue_mid, cue_ref_a, baseline),
        "cos_end_to_ref_a": _cos(cue_end, cue_ref_a, baseline),
        "cue_return": _cos(cue_end, cue_ref_a, baseline) - _cos(cue_mid, cue_ref_a, baseline),
        "cue_cos_mid_end_raw": _cos(cue_mid, cue_end),
        "cue_cos_mid_end_centered": _cos(cue_mid, cue_end, baseline),
        "phase_c_occupancy": float(occ_c),
        "no_write_verified": bool(no_write),
        "e3_never_entered": bool(e3_never_entered),
        "end_a": int(end_a), "end_b": int(end_b),
        "agent": agent,
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = list(EVAL_SEEDS[:1]) if dry_run else list(EVAL_SEEDS)
    replicates = 2 if dry_run else REPLICATES
    phase_ticks = 18 if dry_run else PHASE_TICKS
    boundary_every = 3 if dry_run else BOUNDARY_EVERY

    zg = ZGoalStreamAccumulator()
    arm_results: List[Dict[str, Any]] = []
    per_seed: List[Dict[str, Any]] = []
    ticks_per_cell = replicates * 3 * phase_ticks

    for seed in seeds:
        arm_auc: Dict[str, float] = {}
        arm_rows: Dict[str, Dict[str, Any]] = {}
        for arm_name, phase_c in ARMS:
            print(f"Seed {seed} Condition {arm_name}", flush=True)
            with arm_cell(
                seed,
                config_slice=_config_slice(
                    arm_name, phase_c, phase_ticks=phase_ticks,
                    boundary_every=boundary_every, replicates=replicates),
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=False,
            ) as cell:
                reps: List[Dict[str, Any]] = []
                for r in range(replicates):
                    out = _run_replicate(
                        seed, phase_c, 1000 + 100 * seed + r,
                        phase_ticks=phase_ticks, boundary_every=boundary_every)
                    zg.observe(out.pop("agent"))
                    reps.append(out)
                    print(f"  [train] seed={seed} arm={arm_name} "
                          f"ep {(r + 1) * 3 * phase_ticks}/{ticks_per_cell}", flush=True)
                aucs = [x["auc_a_over_b"] for x in reps]
                # F5: WORST cell across replicates, not reps[0].
                row: Dict[str, Any] = {
                    "arm": arm_name,
                    "seed": int(seed),
                    "auc_mean": _mean(aucs),
                    "auc_per_replicate": [float(a) for a in aucs],
                    "n_group_a_worst": int(min(x["n_group_a"] for x in reps)),
                    "n_group_b_worst": int(min(x["n_group_b"] for x in reps)),
                    "n_group_a_per_replicate": [int(x["n_group_a"]) for x in reps],
                    "n_group_b_per_replicate": [int(x["n_group_b"]) for x in reps],
                    "n_group_c_mean": _mean([float(x["n_group_c"]) for x in reps]),
                    "n_finite_auc_replicates": int(sum(1 for a in aucs if a == a)),
                    "n_common_mean": _mean([float(x["n_common"]) for x in reps]),
                    "n_bank_mid_mean": _mean([float(x["n_bank_mid"]) for x in reps]),
                    "n_bank_end_mean": _mean([float(x["n_bank_end"]) for x in reps]),
                    "n_bank_entries_mean": _mean([float(x["n_bank_entries"]) for x in reps]),
                    "n_bank_exits_mean": _mean([float(x["n_bank_exits"]) for x in reps]),
                    "bank_entries_by_group_first": reps[0]["bank_entries_by_group"],
                    "max_abs_gain_worst": max(x["max_abs_gain"] for x in reps),
                    "n_inversions_mean": _mean([float(x["n_inversions"]) for x in reps]),
                    "phase_c_occupancy_mean": _mean([x["phase_c_occupancy"] for x in reps]),
                    "cue_return_mean": _mean([x["cue_return"] for x in reps]),
                    # F4: the paired subtraction does NOT fully cancel -- baseline is
                    # read per-arm, so the centring vector differs. Surfaced so the
                    # asymmetry is auditable rather than assumed away.
                    "cos_mid_to_ref_a_mean": _mean([x["cos_mid_to_ref_a"] for x in reps]),
                    "cos_end_to_ref_a_mean": _mean([x["cos_end_to_ref_a"] for x in reps]),
                    "cue_return_worst": min(
                        (x["cue_return"] for x in reps if x["cue_return"] == x["cue_return"]),
                        default=float("nan")),
                    "cue_cos_mid_end_raw_mean": _mean([x["cue_cos_mid_end_raw"] for x in reps]),
                    "cue_cos_mid_end_centered_mean": _mean(
                        [x["cue_cos_mid_end_centered"] for x in reps]),
                    "no_write_verified_all": all(x["no_write_verified"] for x in reps),
                    "e3_never_entered_all": all(x["e3_never_entered"] for x in reps),
                }
                cell.stamp(row)
            arm_results.append(row)
            arm_rows[arm_name] = row
            arm_auc[arm_name] = row["auc_mean"]
            # F6: a real per-cell verdict -- did this cell produce a usable measurement?
            cell_ok = (
                row["auc_mean"] == row["auc_mean"]
                and row["n_group_a_worst"] >= (1 if dry_run else MIN_GROUP_A_ANCHORS)
                and row["n_group_b_worst"] >= (1 if dry_run else MIN_GROUP_B_ANCHORS)
            )
            print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)

        stay, ret = arm_rows["STAY"], arm_rows["RETURN"]
        min_a = 1 if dry_run else MIN_GROUP_A_ANCHORS
        min_b = 1 if dry_run else MIN_GROUP_B_ANCHORS
        groups_ok = (min(stay["n_group_a_worst"], ret["n_group_a_worst"]) >= min_a
                     and min(stay["n_group_b_worst"], ret["n_group_b_worst"]) >= min_b)
        auc_stay = arm_auc["STAY"]
        headroom = (1.0 - auc_stay) if auc_stay == auc_stay else float("nan")
        headroom_ok = headroom == headroom and headroom >= DV_HEADROOM_FLOOR
        # F3: the manipulation check is the ENV-level occupancy record -- the agent
        # must actually have occupied the region its arm's phase C names. This is
        # independent of everything the bank computes. cue_return is RECORDED but no
        # longer gates: it is a similarity to the phase-A z_goal and so is
        # hypothesis-aligned with the DV.
        cr_ret, cr_stay = ret["cue_return_mean"], stay["cue_return_mean"]
        cue_return_diff = (cr_ret - cr_stay) if (cr_ret == cr_ret and cr_stay == cr_stay) else float("nan")
        occ_ret = ret["phase_c_occupancy_mean"]
        occ_stay = stay["phase_c_occupancy_mean"]
        occupancy_ok = (occ_ret == occ_ret and occ_stay == occ_stay
                        and min(occ_ret, occ_stay) >= PHASE_C_OCCUPANCY_FLOOR)
        green = bool(groups_ok and headroom_ok and occupancy_ok)
        contrast = arm_auc["RETURN"] - arm_auc["STAY"]
        per_seed.append({
            "seed": int(seed),
            "auc_stay": auc_stay,
            "auc_return": arm_auc["RETURN"],
            "contrast_return_minus_stay": float(contrast),
            "dv_headroom": float(headroom),
            "cue_return_return_arm": float(cr_ret),
            "cue_return_stay_arm": float(cr_stay),
            "cue_return_diff": float(cue_return_diff),
            "phase_c_occupancy_return": float(ret["phase_c_occupancy_mean"]),
            "phase_c_occupancy_stay": float(stay["phase_c_occupancy_mean"]),
            "n_group_a_worst": int(min(stay["n_group_a_worst"], ret["n_group_a_worst"])),
            "n_group_b_worst": int(min(stay["n_group_b_worst"], ret["n_group_b_worst"])),
            # F6a: phases A and B are bit-identical across arms, so the group sizes
            # should match; a mismatch means eviction/floor censoring differed and is
            # recorded rather than assumed absent.
            "group_sizes_match_across_arms": bool(
                stay["n_group_a_worst"] == ret["n_group_a_worst"]
                and stay["n_group_b_worst"] == ret["n_group_b_worst"]),
            "n_group_c_stay": float(stay["n_group_c_mean"]),
            "n_group_c_return": float(ret["n_group_c_mean"]),
            "cos_mid_to_ref_a_stay": float(stay["cos_mid_to_ref_a_mean"]),
            "cos_mid_to_ref_a_return": float(ret["cos_mid_to_ref_a_mean"]),
            "green": green,
            "gate_failures": ([] if groups_ok else ["anchor_group_supply"])
            + ([] if headroom_ok else ["dv_headroom"])
            + ([] if occupancy_ok else ["phase_c_occupancy_achieved"]),
        })

    green_rows = [s for s in per_seed if s["green"]]
    n_green = len(green_rows)
    contrasts = [s["contrast_return_minus_stay"] for s in green_rows
                 if s["contrast_return_minus_stay"] == s["contrast_return_minus_stay"]]
    mean_contrast = _mean(contrasts)
    n_positive = sum(1 for c in contrasts if c > 0)
    # F5: denominate on the number of FINITE contrasts actually counted.
    n_required = math.ceil(C2_CONSISTENCY_FRACTION * len(contrasts)) if contrasts else 0

    # F2: with a saturating RETURN arm, green (AUC_stay <= 1 - 0.10) forces every
    # contrast positive, so the consistency clause CANNOT fail. Record that, so it is
    # never read as independent evidence.
    _min_ret = min((s["auc_return"] for s in green_rows if s["auc_return"] == s["auc_return"]),
                   default=float("nan"))
    consistency_clause_pinned = bool(
        _min_ret == _min_ret and _min_ret >= (1.0 - DV_HEADROOM_FLOOR))
    min_green = 1 if dry_run else MIN_GREEN_SEEDS
    gate_ok = n_green >= min_green
    manipulation_reached = any(
        min(s["phase_c_occupancy_return"], s["phase_c_occupancy_stay"]) >= PHASE_C_OCCUPANCY_FLOOR
        for s in per_seed)

    c1_inversions = _mean([r["n_inversions_mean"] for r in arm_results])
    c2_pass = bool(gate_ok and mean_contrast == mean_contrast
                   and mean_contrast >= C2_MEAN_CONTRAST_FLOOR
                   and n_positive >= n_required)
    c3_pass = all(r["no_write_verified_all"] for r in arm_results)
    overall_pass = bool(gate_ok and c2_pass and c3_pass)

    # F4: an explicit instrument-not-ready branch, so a run whose manipulation never
    # reached the DV routes non_contributory rather than `weakens`.
    if not manipulation_reached:
        label, direction = "manipulation_did_not_reach_dv", "non_contributory"
    elif not gate_ok:
        label, direction = "substrate_not_ready_requeue", "non_contributory"
    elif not c3_pass:
        label, direction = "probe_wrote_to_anchor_pool_instrument_defect", "non_contributory"
    elif c2_pass:
        label, direction = (
            "readtime_recomputation_reinstates_old_goal_anchors_supports_sd098_half_i",
            "supports")
    else:
        label, direction = (
            "destination_reversal_does_not_reinstate_weakens_sd098_on_this_substrate",
            "weakens")

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "evidence_direction": direction, "label": label,
        "gate_ok": gate_ok, "manipulation_reached": manipulation_reached,
        "n_green": n_green, "min_green_required": min_green,
        "mean_contrast": mean_contrast, "contrast_sd": _pstdev(contrasts),
        "n_positive": n_positive, "n_required": n_required,
        "n_finite_contrasts": len(contrasts),
        "c1_inversions": c1_inversions, "c1_pass": c1_inversions > 0,
        "c2_pass": c2_pass, "c3_pass": c3_pass,
        "consistency_clause_pinned": consistency_clause_pinned,
        "all_group_sizes_match": all(s["group_sizes_match_across_arms"] for s in per_seed),
        "auc_stay_mean": _mean([s["auc_stay"] for s in green_rows]),
        "auc_return_mean": _mean([s["auc_return"] for s in green_rows]),
        "worst_dv_headroom": min((s["dv_headroom"] for s in per_seed
                                  if s["dv_headroom"] == s["dv_headroom"]),
                                 default=float("nan")),
        "worst_cue_return_diff": min((s["cue_return_diff"] for s in per_seed
                                      if s["cue_return_diff"] == s["cue_return_diff"]),
                                     default=float("nan")),
        "mean_phase_c_occupancy": _mean(
            [s["phase_c_occupancy_return"] for s in per_seed]
            + [s["phase_c_occupancy_stay"] for s in per_seed]),
        "per_seed": per_seed, "arm_results": arm_results,
        "z_goal_stream_stats": zg.stats(), "seeds": seeds,
        "replicates": replicates, "phase_ticks": phase_ticks,
        "boundary_every": boundary_every,
    }


def _flat_scalar(res: Dict[str, Any]) -> Dict[str, Any]:
    """Flat SCALAR readout. Booleans as 0/1 ints; non-finite values DROPPED."""
    raw = {
        "mean_contrast_return_minus_stay": res["mean_contrast"],
        "contrast_sd": res["contrast_sd"],
        "auc_stay_mean": res["auc_stay_mean"],
        "auc_return_mean": res["auc_return_mean"],
        "n_green_seeds": float(res["n_green"]),
        "n_positive_contrast_seeds": float(res["n_positive"]),
        "n_required_positive_seeds": float(res["n_required"]),
        "n_finite_contrasts": float(res["n_finite_contrasts"]),
        "mean_rank_inversions": res["c1_inversions"],
        "worst_dv_headroom": res["worst_dv_headroom"],
        "worst_cue_return_diff": res["worst_cue_return_diff"],
        "mean_phase_c_occupancy": res["mean_phase_c_occupancy"],
        "c1_reranking_occurs": 1 if res["c1_pass"] else 0,
        "c2_content_specific": 1 if res["c2_pass"] else 0,
        "c3_no_write_verified": 1 if res["c3_pass"] else 0,
        "gate_ok": 1 if res["gate_ok"] else 0,
        "manipulation_reached_dv": 1 if res["manipulation_reached"] else 0,
        "consistency_clause_pinned": 1 if res["consistency_clause_pinned"] else 0,
        "all_group_sizes_match_across_arms": 1 if res["all_group_sizes_match"] else 0,
        "threshold_mean_contrast": C2_MEAN_CONTRAST_FLOOR,
        "threshold_dv_headroom": DV_HEADROOM_FLOOR,
        "threshold_cue_return_diff": CUE_RETURN_DIFF_FLOOR,
    }
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        fv = float(v)
        if math.isfinite(fv):
            out[k] = fv
    return out


def main(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    res = run_experiment(dry_run=dry_run)
    ps = res["per_seed"]

    def _worst(key: str) -> Tuple[float, Optional[Dict[str, Any]]]:
        rows = [s for s in ps if s[key] == s[key]]
        if not rows:
            return float("nan"), None
        w = min(rows, key=lambda s: s[key])
        return float(w[key]), w

    worst_hr, hr_row = _worst("dv_headroom")
    worst_cr, cr_row = _worst("cue_return_diff")
    for _r in ps:
        _r["_occ_min"] = min(_r["phase_c_occupancy_return"], _r["phase_c_occupancy_stay"])
    worst_occ, occ_row = _worst("_occ_min")
    worst_a = min((s["n_group_a_worst"] for s in ps), default=0)
    worst_b = min((s["n_group_b_worst"] for s in ps), default=0)
    min_a = 1 if dry_run else MIN_GROUP_A_ANCHORS
    min_b = 1 if dry_run else MIN_GROUP_B_ANCHORS

    preconditions = [
        {"name": "phase_c_occupancy_achieved",
         "description": ("ENV-LEVEL manipulation check: fraction of phase-C ticks the "
                         "agent spent inside the region its arm's phase C names. "
                         "Independent of everything the bank computes, which the "
                         "cue_return check was NOT (it is a similarity to the phase-A "
                         "z_goal, i.e. hypothesis-aligned with the DV). Worst cell."),
         "measured": worst_occ, "threshold": PHASE_C_OCCUPANCY_FLOOR, "direction": "lower",
         "control": "measured in the environment, not in latent space",
         "offending_cell": f"seed={occ_row['seed']}" if occ_row else "n/a",
         "met": bool(worst_occ == worst_occ and worst_occ >= PHASE_C_OCCUPANCY_FLOOR)},
        {"name": "cue_return_diff_DIAGNOSTIC_NOT_GATING",
         "description": ("PAIRED: how much further the RETURN arm's cue moved back "
                         "toward the phase-A reference than the STAY arm's. The "
                         "UNPAIRED form does not discriminate (measured +0.095 vs "
                         "+0.093) -- the cue drifts back in both arms. The paired form "
                         "is a manipulation check the control fails by construction, so "
                         "it cannot certify its own subject. Worst cell reported."),
         "measured": worst_cr, "threshold": CUE_RETURN_DIFF_FLOOR, "direction": "lower",
         "control": "STAY arm sits at ~0. RECORDED ONLY -- does not gate (see F3).",
         "offending_cell": f"seed={cr_row['seed']}" if cr_row else "n/a",
         "threshold_not_applicable": "diagnostic only; the gate is phase_c_occupancy_achieved",
         "met": True},
        {"name": "dv_headroom",
         "description": ("room above the STAY arm's AUC for the contrast to reach its "
                         "bar; AUC is bounded by 1.0 so a saturated STAY arm makes C2 "
                         "unreachable by construction. Worst cell reported."),
         "measured": worst_hr, "threshold": DV_HEADROOM_FLOOR, "direction": "lower",
         "control": "STAY arm -- the subtrahend in the C2 contrast",
         "offending_cell": f"seed={hr_row['seed']}" if hr_row else "n/a",
         "met": bool(worst_hr == worst_hr and worst_hr >= DV_HEADROOM_FLOOR)},
        {"name": "group_a_supply_worst_cell",
         "description": "phase-A anchors in the common ranked set, worst replicate",
         "measured": float(worst_a), "threshold": float(min_a), "direction": "lower",
         "control": "group A must be populated for the A-vs-B AUC to exist",
         "met": bool(worst_a >= min_a)},
        {"name": "group_b_supply_worst_cell",
         "description": "phase-B anchors in the common ranked set, worst replicate",
         "measured": float(worst_b), "threshold": float(min_b), "direction": "lower",
         "control": "group B must be populated for the A-vs-B AUC to exist",
         "met": bool(worst_b >= min_b)},
        {"name": "green_seed_supply",
         "description": "seeds passing the per-seed readiness gate",
         "measured": float(res["n_green"]), "threshold": float(res["min_green_required"]),
         "direction": "lower", "control": "per-seed gate, evaluated independently per seed",
         "met": bool(res["gate_ok"])},
    ]

    criteria = [
        {"name": "C1_reranking_occurs", "load_bearing": False,
         "passed": bool(res["c1_pass"]), "measured": float(res["c1_inversions"]),
         "threshold": 0.0,
         "description": ("mean rank inversions between cues. RECORDED, NOT GATING: the "
                         "red-team showed bank membership churn pins this true.")},
        {"name": "C2_reversal_reinstates_group_a", "load_bearing": True,
         "passed": bool(res["c2_pass"]),
         "measured": (float(res["mean_contrast"])
                      if res["mean_contrast"] == res["mean_contrast"] else None),
         "threshold": C2_MEAN_CONTRAST_FLOOR,
         "measured_positive_seeds": float(res["n_positive"]),
         "threshold_positive_seeds": float(res["n_required"]),
         "description": ("mean(AUC_return - AUC_stay) over green seeds >= 0.10 AND "
                         "positive in >= ceil(2/3) of the finite contrasts. The "
                         "consistency clause is PINNED TRUE whenever the RETURN arm "
                         "saturates (green implies AUC_stay <= 0.90, so a saturating "
                         "RETURN forces every contrast positive) -- see "
                         "consistency_clause_pinned in the readout; the MEAN clause is "
                         "the one that can fail.")},
        {"name": "C3_probe_performs_no_write", "load_bearing": True,
         "passed": bool(res["c3_pass"]), "measured": 1.0 if res["c3_pass"] else 0.0,
         "threshold": 1.0,
         "description": ("anchor-pool payload fingerprint byte-identical across both "
                         "probe ranks in every cell -- SD-098 half (i)'s 'without any "
                         "explicit write'. Structural: identical in both arms.")},
    ]

    non_degenerate = bool(res["gate_ok"] and res["manipulation_reached"])
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "timestamp_utc": ts,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": res["outcome"],
        "evidence_direction": res["evidence_direction"],
        "non_degenerate": non_degenerate,
        "degeneracy_reason": (None if non_degenerate else
                              "readiness gate failed, or the manipulation never reached the DV"),
        "readout": _flat_scalar(res),
        "criteria": criteria,
        "combination_rule": ("PASS iff (C2_reversal_reinstates_group_a AND "
                             "C3_probe_performs_no_write); C1 recorded, not gating"),
        "interpretation": {
            "label": res["label"],
            "preconditions": preconditions,
            "criteria_non_degenerate": {
                "C1_reranking_occurs": False,   # churn pins it true; never treated as discriminating
                "C2_reversal_reinstates_group_a": bool(
                    res["gate_ok"] and res["manipulation_reached"] and res["n_finite_contrasts"] > 0),
                "C3_probe_performs_no_write": True,
            },
            "scope_limit": (
                "TWO limits. (1) Tests SD-098 half (i) only (re-classification with no "
                "write). Half (ii), the matched stored-node-type comparison arm, is NOT "
                "built (user decision 2026-09-17: do not build SD-097's typed topology). "
                "(2) Because the policy seeks the active destination, destination "
                "privilege is NOT separated from sensory-context reinstatement: a PASS "
                "shows goal-relevance is RECOMPUTED AT READ TIME from current state, but "
                "does not isolate the destination term. The separating condition "
                "(relocate without moving the agent) is measurably too weak to run "
                "(paired cue_return +0.003, 4/7). A PASS is PARTIAL support and does not "
                "close SD-098."),
            "design_note": (
                "A-B-A reversal. Group A is OLDER than group B, so a recency / "
                "cue-displacement account predicts A can only LOSE rank in both arms; "
                "SD-098 predicts A is reinstated when the destination returns to D1. "
                "The predictions are opposite, which the superseded forward-only design "
                "could not arrange."),
        },
        "arm_results": res["arm_results"],
        "per_seed_results": res["per_seed"],
        "tuning_reference": TUNING_REFERENCE,
        "custom_information": {
            "tuning_seeds": list(TUNING_SEEDS),
            "evaluation_seeds": list(res["seeds"]),
            "seed_sets_disjoint": True,
            "arms": [a for a, _ in ARMS],
            "probe_simulation_mode": True,
            "ghost_bank_top_k": None,
            "e3_selector_entered": not all(r["e3_never_entered_all"] for r in res["arm_results"]),
            "substrate_gate_note": (
                "SD-082 (corrupting, e3_selector.py) is OPEN; this driver uses a scripted "
                "policy and never calls select_action, so that path is never entered -- "
                "asserted per cell."),
            "red_team": ("opus, 2026-09-17: BLOCKING on the superseded forward-only design "
                         "(cue-displacement collinear with content-change; sham shared 4/6 "
                         "cells with baseline). This A-B-A design is the fix."),
        },
    }

    full_config = {
        "phase_ticks": res["phase_ticks"], "boundary_every": res["boundary_every"],
        "replicates": res["replicates"], "force_benefit": FORCE_BENEFIT,
        "force_drive": FORCE_DRIVE, "grid_size": GRID_SIZE,
        "d1_resources": [list(c) for c in D1], "d2_resources": [list(c) for c in D2],
        "hazards": [list(c) for c in HAZARDS], "agent_start": list(AGENT_START),
        "goal_cue_centering": True, "ghost_bank_top_k": None,
        "resource_respawn_on_consume": False,
        "policy": "scripted_region_seeking_no_e3",
        "epsilon_explore": EPSILON_EXPLORE,
        "thresholds": {
            "c2_mean_contrast_floor": C2_MEAN_CONTRAST_FLOOR,
            "c2_consistency_fraction": C2_CONSISTENCY_FRACTION,
            "min_green_seeds": MIN_GREEN_SEEDS,
            "dv_headroom_floor": DV_HEADROOM_FLOOR,
            "phase_c_occupancy_floor": PHASE_C_OCCUPANCY_FLOOR,
            "cue_return_diff_floor_DIAGNOSTIC_ONLY": CUE_RETURN_DIFF_FLOOR,
            "min_group_a_anchors": MIN_GROUP_A_ANCHORS,
            "min_group_b_anchors": MIN_GROUP_B_ANCHORS,
        },
    }

    out_path = write_flat_manifest(
        manifest, dry_run=dry_run, config=full_config, seeds=res["seeds"],
        script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=res["z_goal_stream_stats"])

    print("")
    print(f"label:             {res['label']}")
    print(f"manipulation:      reached_dv={res['manipulation_reached']} "
          f"(worst paired cue_return {res['worst_cue_return_diff']:.4f}, "
          f"floor {CUE_RETURN_DIFF_FLOOR}; phase-C occupancy "
          f"{res['mean_phase_c_occupancy']:.2f})")
    print(f"gate_ok:           {res['gate_ok']} (green {res['n_green']}/{len(ps)}, "
          f"need {res['min_green_required']})")
    print(f"AUC stay/return:   {res['auc_stay_mean']:.4f} / {res['auc_return_mean']:.4f}")
    print(f"mean contrast:     {res['mean_contrast']:.4f} (threshold {C2_MEAN_CONTRAST_FLOOR})")
    print(f"positive seeds:    {res['n_positive']}/{res['n_finite_contrasts']} "
          f"(need {res['n_required']})")
    print(f"C1={res['c1_pass']} (not gating) C2={res['c2_pass']} C3={res['c3_pass']}")
    print(f"outcome:           {res['outcome']}  direction: {res['evidence_direction']}")
    print(f"manifest:          {out_path}")

    outcome_raw = str(res["outcome"]).upper()
    return {"outcome": outcome_raw if outcome_raw in ("PASS", "FAIL") else "FAIL",
            "manifest_path": out_path}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _result = main(dry_run=args.dry_run)
    emit_outcome(outcome=_result["outcome"],
                 manifest_path=_result["manifest_path"],
                 dry_run=args.dry_run)
    raise SystemExit(0)
