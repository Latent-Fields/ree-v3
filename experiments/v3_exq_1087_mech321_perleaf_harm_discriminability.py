#!/opt/local/bin/python3
"""V3-EXQ-1087 -- MECH-321 PER-LEAF HARM-INPUT DISCRIMINABILITY DIAGNOSTIC in
V3-EXQ-919's regime. Was 919's harm-aware selection actually harm-INFORMED?

red-team (fable): BLOCKING, 6 findings, all verified against source/data ->
F1 fixed (constant-index C2 null + chosen-index variety), F2 resolved by
orchestrator decision Q-1087 = Option A (post-death-share gate on the "919
weakens unconditional" route), F3 fixed (kept-leaf-executed rate reported),
F4 dismissed (tolerance targets collapse; C2 decides when no collapse), F5
relabelled descriptive-only, F6 named in the not-tracking label.

SLEEP DRIVER: not applicable -- no sleep phase entered in this run.

EXPERIMENT_PURPOSE = diagnostic. Excluded from governance confidence scoring.
Its job is to decide how V3-EXQ-919's `weakens` on MECH-321 should be READ
(ratified routing: failure_autopsy_MECH-320-defect-cluster_2026-09-23, user
decision rec-20260924-b52804ea; MECH-321 evidence_quality_note, 2026-09-24).

THE EXPOSURE UNDER TEST
    Both stages of SD-hazard-aware-policy-decomposition consume ONE per-leaf
    number, `harm_penalty`, read by HippocampalModule._decomposition_harm_
    penalty (ree_core/hippocampal/module.py:1128-1153) as the mean residue-
    field VALENCE_HARM_DISCRIMINATIVE over each leaf's OWN E2-predicted
    world_states (e2.rollout_with_world, module.py:1113). Stage 1
    (PolicyDecomposition.harm_bias, policy_decomposition.py:968-984) turns
    it into a clamped additive score bias; Stage 2
    (select_harm_aware_leaves, policy_decomposition.py:986-1019) keeps only
    the stable-argmin leaf of each withheld chunk. 919's Stage 2 fired on
    6831 / 6977 decompositions (97.9%). If the per-leaf penalties are
    indistinguishable, Stage 2 is a harm-blind pool cut and 919's behaviour
    change cannot be read as harm-informed selection.

WHAT THIS MEASURES (ON arm only -- the OFF arm never calls Stage 1/2)
    An instance-level wrapper on `agent.policy_decomposition.
    select_harm_aware_leaves` (the ONE call site, module.py:1271) captures,
    for every Stage-2 call: the leaves' penalties, their Stage-1 harm_bias
    (already written to leaf metadata before the call, module.py:1266-1270),
    z_harm_a_norm and w(h), the chosen index, the leaves' action sequences,
    their pairwise predicted-state distance, and a GROUND-TRUTH outcome per
    leaf from executing its own action sequence (seq[:horizon], the steps of
    the E2 rollout the leaf actually controls; the rollout pads the rest of
    the horizon with all-zero action vectors, which have no env
    counterpart) in a deep copy of the live env (common random numbers: every leaf starts from an
    identical copy, so the per-leaf contrast is paired). Stage 2 discards
    the non-kept leaves, so they cannot be recovered afterwards from
    candidate metadata -- hence the wrapper. PRE-FLIGHT P1: the "867a
    monkeypatch" the autopsy named was never committed (prose only in the
    867a docstring); this wrapper is written fresh.

    The wrapper returns the original method's result unchanged. All global
    RNG state (python / numpy / torch) is saved and restored around the
    ground-truth probe, and the env copy carries its own `_rng`
    (causal_grid_world.py:1604), so the probe cannot perturb the run. That is
    VERIFIED, not assumed: a probe-OFF control replicate of the first seed
    must reproduce the probe-ON action sequence exactly
    (precondition `probe_non_perturbing`).

PRE-FLIGHT PREMISE CORRECTIONS APPLIED (orchestrate-20260924 pre-flight, AMBER)
    P2. 919's agent is UNTRAINED (no optimizer / backward / prediction-loss
        anywhere in its driver; update_residue takes no training step). So
        "E2 world-forward collapse" here means RANDOM-INIT E2. The relevant
        prior bound is the May untrained-agent measurement (0.0000
        per-candidate spread), NOT the August trained-agent spreads
        (0.02-0.16) cited in the MECH-321 evidence_quality_note. This run
        faithfully replays 919's untrained regime and does not train either.

    CHANGE 1 (tolerance tie, not exact equality). Penalties are continuous
        RBF reads, so random-init E2 yields tiny non-zero differences and an
        EXACT-equality tie rate would read ~0 under near-collapse. A Stage-2
        call over >= 2 leaves is a TIE when
            spread < EPS_ABS  OR  spread / (|mean| + 1e-8) < EPS_REL
        with spread = max - min of the leaves' penalties. EPS_REL = 0.01:
        Stage 1's gain is 0.1 * w * p clamped at 0.1, so a 1% penalty
        difference moves the score by at most 1e-3, one percent of the
        clamp scale -- below any decision-relevant hazard contrast. EPS_ABS
        = 1e-6 catches the all-near-zero case. Both fixed here, before any
        data. The FULL spread distribution, the exact-tie rate, P(chosen
        index == 0) against the 1/n chance level, and pairwise predicted-
        state distance are reported alongside.

    CHANGE 2 (Stage-1 clamp saturation). valence_bounding_enabled=False by
        default (config.py:3506) and 919 never sets it, so the harm valence
        is an unbounded `+=` accumulator; harm_bias = clamp(0.1*w*p, 0, 0.1)
        saturates once w*p >= 1. Recorded: fraction of leaves with harm_bias
        == harm_bias_scale (and == 0), and within-call Stage-1 bias spread.
        A high saturated fraction is a second, independent harm-blind route
        for Stage 1. REPORTED as a label qualifier; it does not by itself
        change the routing (Stage 2, firing on ~98% of decompositions,
        carries the manipulation).

    CHANGE 3 (harm vs occupancy). The "harm" valence channel is written with
        z_harm.norm() on EVERY sense() tick (agent.py:5634-5650) at the
        nearest active residue centre, and HarmEncoder is an untrained
        Linear-ReLU-Linear with biases (latent/stack.py:156-172), so the
        channel may be largely a visit-count map. A LOW tie rate therefore
        does not by itself show harm-informed selection. Added: a
        ground-truth harm-tracking test (C2) as a precondition of the
        "unconditional weakens" branch, and a third pre-registered outcome
        "discriminative but not harm-tracking -> non_contributory". Also
        recorded per tick: z_harm.norm() split by hazard / no-hazard and
        harm / no-harm ticks, and its correlation with the hazard field at
        the agent's cell -- DESCRIPTIVE ONLY (red-team F5: harm_obs contains
        the dense hazard field, so that correlation partly reads the
        encoder's own input). C2 null #2 (red-team F1): the best CONSTANT
        chosen index -- leaves are always single primitives [0],[1],[2] and
        an untrained E2 can fix the argmin to one action, which a
        wall-blocked zero-harm move then makes "GT-best" trivially; tracking
        needs excess >= HT_EXCESS_MIN over BOTH nulls and >= 2 distinct
        chosen indices.

    CHANGE 4 (env done). 919's loop discards `done` (919 :415) and keeps
        stepping. This replay keeps 919's exact loop (so the regime is
        faithful and the action sequence is comparable to 919's), but READS
        done, records per-episode done tick + cause, and EXCLUDES Stage-2
        calls captured after done from every primary statistic (reported
        separately under `post_done`). No post-death battery is collected.
        Whether the replay still matches 919's schedule is recorded per
        seed as the elementwise action-sequence match against 919's own
        manifest (non-gating: ree_core has moved since 2026-08-11 and
        torch.multinomial differs across machine classes).

    CHANGE 5 (scope). ON arm only; 919 MEASUREMENT_SEEDS[:5] = 11, 23, 47,
        71, 3; 12 ep x 60 steps; identical HAZARD_TUNED overlay and flags;
        agent config built exactly as 919's `_build(seed, ARM_SELECTION_ON)`
        (asserted equal to 919's own `_arm_flags` at runtime when the 919
        module imports).

PRE-REGISTERED CRITERIA AND ROUTING (all constants below, fixed pre-data)
    Unit: a PRE-DONE Stage-2-ACTIVE call (w >= harm_override_w_threshold)
    over >= 2 leaves, pooled over the 5 measurement seeds.
    C1 (load-bearing) Stage-2 tolerance-tie rate T over those units.
    C2 (load-bearing only when T <= TIE_RATE_LOW) harm tracking, over the
        NON-tie units whose ground truth separates the leaves: excess =
        mean(1[chosen leaf is a ground-truth-best leaf]) - mean(chance),
        chance = (#GT-best leaves) / n_leaves; z = excess / (sqrt(sum
        chance*(1-chance)) / n). Ground truth is lexicographic: realised
        env harm (sum of negative harm over the executed rolled-horizon
        steps) first, then mean hazard-field value at the agent's cell
        (lower better) to break ties. Tracking iff excess >= HT_EXCESS_MIN
        and z >= HT_Z_MIN; anti-tracking iff excess <= -HT_EXCESS_MIN and
        z <= -HT_Z_MIN. CAVEAT (author): the chance null assumes the
        chosen index is uniform among leaves under the null. It is NOT --
        Stage 2 picks the first leaf on an exact tie, and leaf ORDER
        (_recursive_leaf_tiles) may correlate with ground truth. The
        exact-tie index-0 rate and P(first leaf is GT-best) are reported so
        an adjudicator can check that confound before reading C2.
    combination_rule:
        probe perturbed OR T-units < MIN_STAGE2_EVENTS
            -> substrate_not_ready_requeue (no 919 re-adjudication)
        T >= TIE_RATE_HIGH -> stage2_tie_rate_high_harm_blind
            -> 919 re-adjudicates to non_contributory
        TIE_RATE_LOW < T < TIE_RATE_HIGH
            -> stage2_tie_rate_intermediate_conditional_stands
               (the ratified routing names only HIGH and LOW; in between
               neither re-adjudication fires and 919 stays CONDITIONAL)
        T <= TIE_RATE_LOW and GT-units < MIN_GT_EVENTS
            -> stage2_discriminative_harm_tracking_undetermined (conditional stands)
        T <= TIE_RATE_LOW and tracking and 919 post-death harm share < 0.5
            -> stage2_discriminative_and_harm_tracking
            -> 919's conditional weakens becomes UNCONDITIONAL
        T <= TIE_RATE_LOW and tracking and share >= 0.5 (or unreadable)
            -> ..._harm_tracking__919_post_death_dominated_conditional_stands
            (OPTION A, orchestrator decision Q-1087: 919's own per_tick_harm
            shows 767/960 episodes die by median tick 7 of 60 and ~87% of its
            harm accrues at health 0, so a live Stage-2 input cannot make a
            post-death-dominated DV's weakens unconditional)
        T <= TIE_RATE_LOW and anti-tracking -> stage2_discriminative_anti_harm_tracking
            -> 919 non_contributory
        T <= TIE_RATE_LOW and neither       -> stage2_discriminative_not_harm_tracking
            -> 919 non_contributory (pre-flight B3 third outcome)
    Both directions are declared: a HIGH tie rate or non-tracking weakens the
    READING of 919 (not MECH-321 itself); a LOW tie rate with tracking
    strengthens it. This run emits no MECH-321 evidence of its own.
    SCOPE (author): the verdict covers the Stage-2-ACTIVE population only
    (the ~98% of 919's decompositions where the categorical override fired);
    Stage-1-only calls are reported under `stage2_inactive_calls`.

DV-SYMMETRY DECLARATION. DV C1 = max-min range of per-leaf penalties within a
call (symmetry: a broadcast additive constant across the call's leaves
cancels in the range). That invariance is the POINT, not a defect: a
per-leaf-identical penalty is exactly the harm-blind case the diagnostic is
built to detect, and the manipulation-free question here is whether the
penalties differ at all. C2 = chosen-vs-GT agreement (symmetry: any
monotone rescaling of the penalties); Stage 2's argmin is itself
order-based, so C2 reads exactly what Stage 2 consumes. Single arm, no
manipulation contrast.

GOV-REUSE-1. Decisive readout = per-leaf harm_penalty spread / Stage-2
tolerance-tie rate / chosen-vs-ground-truth agreement. Never recorded by any
867/867a/867b/919 run (autopsy MECH-320-defect-cluster line 104; 919's
manifest rows carry only per-cell counters). Not derivable post hoc: Stage 2
discards the non-kept leaves. Not recoverable -> run.

RE-DERIVE BRAKE (Step 2.5b): exempt -- a diagnostic discriminating WHY/
whether the harm input was live, not a re-test of the same claim.

SUBSTRATE-PATH GATE (Step 2.5c): open corrupting entries touching imported
modules -- MECH-320 (tonic_vigor.py; use_tonic_vigor not enabled, so its
execution condition is absent), SD-PP-B5 (trained world-forward
compression bounding CONSOLIDATION experiments; no training or consolidation
here, and a compressed per-leaf range is precisely what this run measures),
sd_zself_training_path (self_recurrence.py; DR-13 not enabled). Degrading
overlaps named in the queue note.

Z_GOAL: deliberately inert, as in 919 (update_z_goal called with
benefit_exposure=0.0, z_goal_enabled default False).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.episode_termination import EpisodeTerminationAccumulator  # noqa: E402
from experiments._lib.stats import spearman  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
import experiments._lib.baselines.sd084_midexec_reachability as baselines  # noqa: E402

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.policy import ChunkedPrimitive, ChunkState  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1087_mech321_perleaf_harm_discriminability"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-321"]
QUEUE_ID = "V3-EXQ-1087"
TARGET_RUN_ID = (
    "v3_exq_919_mech321_harm_aware_selection_unconditional_wholeepisode_"
    "20260811T225107Z_v3")
TARGET_MANIFEST = (
    REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    / (TARGET_RUN_ID + ".json"))

ARM_ON = "ARM_SELECTION_ON"  # 919's arm label, verbatim

# --- 919's harm-aware-selection parameters, VERBATIM (919 :204-208). ---
HARM_BIAS_GAIN = 0.1
HARM_BIAS_SCALE = 0.1
HARM_THREAT_FLOOR = 0.1
HARM_THREAT_REF = 0.5
HARM_OVERRIDE_W_THRESHOLD = 0.9

# --- Pre-registered thresholds. Constants, fixed before any data. ---
EPS_ABS = 1e-6            # absolute tie tolerance on per-call penalty spread
EPS_REL = 1e-2            # relative tie tolerance: spread / (|mean| + 1e-8)
TIE_RATE_HIGH = 0.5       # T >= this -> Stage 2 harm-blind (919 -> non_contributory)
TIE_RATE_LOW = 0.2        # T <= this -> discriminative (C2 then decides)
MIN_STAGE2_EVENTS = 30    # floor on pre-done Stage-2-active multi-leaf calls
MIN_GT_EVENTS = 30        # floor on GT-discriminable non-tie calls for C2
HT_EXCESS_MIN = 0.05      # C2 excess-over-chance bar
HT_Z_MIN = 2.0            # C2 z bar
SAT_TOL = 1e-9            # harm_bias == scale tolerance (float compare)
POST_DEATH_SHARE_MAX = 0.5  # Option A gate: 919 "weakens unconditional" only if
                            # 919's post-death harm share is BELOW this

# Pre-registered seed list: 919's MEASUREMENT_SEEDS[:5], verbatim order.
MEASUREMENT_SEEDS: Tuple[int, ...] = (11, 23, 47, 71, 3)
CONTROL_SEED = MEASUREMENT_SEEDS[0]   # probe-OFF non-perturbation replicate

ANCHOR_REACHABILITY_EXEMPT = (
    "the two count-floor preconditions ARE the degeneracy definition (minimum "
    "number of units a rate is read over), not a signature a positive control "
    "must reproduce; reachable by construction in 919's regime, which recorded "
    "6977 pre-commit decompositions over 40 seeds (~175/seed vs a floor of 30 "
    "pooled over 5 seeds).")

_ZGOAL = ZGoalStreamAccumulator()


# ---------------------------------------------------------------------------
# Config -- 919's ARM_SELECTION_ON construction, copied verbatim in shape
# (919 :289-347). Copied rather than imported so this driver's substrate_hash
# covers it; equality with 919's own _arm_flags is asserted at runtime below.
# ---------------------------------------------------------------------------
def _arm_flags_on() -> Dict[str, Any]:
    flags = dict(baselines.on_arm_flags())          # abort mechanism ON
    flags.update(baselines.HAZARD_TUNED_STREAM_FLAGS)
    flags.update({
        "decomposition_use_harm_aware_selection": True,
        "decomposition_harm_bias_gain": HARM_BIAS_GAIN,
        "decomposition_harm_bias_scale": HARM_BIAS_SCALE,
        "decomposition_harm_threat_floor": HARM_THREAT_FLOOR,
        "decomposition_harm_threat_ref": HARM_THREAT_REF,
        "decomposition_harm_override_w_threshold": HARM_OVERRIDE_W_THRESHOLD,
    })
    return flags


def _config_slice(episodes: int, probe_enabled: bool) -> Dict[str, Any]:
    slice_: Dict[str, Any] = {
        "env": dict(baselines.HAZARD_TUNED_ENV_OVERLAY),
        "env_seeded_per_cell": baselines.ENV_SEEDED_PER_CELL,
        "schedule": {
            "episodes": int(episodes),
            "steps_per_episode": baselines.STEPS_PER_EPISODE,
        },
        "self_dim": baselines.SELF_DIM,
        "world_dim": baselines.WORLD_DIM,
        "seeded_chunk_sequence": list(baselines.SEEDED_CHUNK_SEQUENCE),
        "seeded_chunk_depth": baselines.SEEDED_CHUNK_DEPTH,
        "seeded_chunk_selection_weight": baselines.SEEDED_CHUNK_SELECTION_WEIGHT,
        "stage2_probe_enabled": bool(probe_enabled),
    }
    slice_.update(_arm_flags_on())
    return slice_


def _config_matches_919() -> Optional[bool]:
    """True/False if 919's module imports; None (unverifiable) otherwise."""
    try:
        import importlib
        m919 = importlib.import_module(
            "experiments.v3_exq_919_mech321_harm_aware_selection_"
            "unconditional_wholeepisode")
        return bool(m919._arm_flags(m919.ARM_ON) == _arm_flags_on())
    except Exception:
        return None


def _build(seed: int) -> Tuple[CausalGridWorldV2, REEAgent]:
    env = CausalGridWorldV2(**baselines.env_kwargs_hazard_tuned(seed))
    env.reset()
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=baselines.SELF_DIM,
        world_dim=baselines.WORLD_DIM,
        reafference_action_dim=env.action_dim,
        **_arm_flags_on(),
    )
    return env, REEAgent(cfg)


def _register_chunk(agent: REEAgent) -> None:
    agent.policy_chunking.library.register(
        ChunkedPrimitive(
            sequence=baselines.SEEDED_CHUNK_SEQUENCE,
            depth=baselines.SEEDED_CHUNK_DEPTH,
            state=ChunkState.CRYSTALLISED,
            selection_weight=baselines.SEEDED_CHUNK_SELECTION_WEIGHT,
        )
    )


def _hazard_at_agent(env: CausalGridWorldV2) -> float:
    try:
        return float(np.clip(env.hazard_field[int(env.agent_x), int(env.agent_y)], 0.0, 1.0))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# The Stage-2 capture probe (pre-flight P1: written fresh).
# ---------------------------------------------------------------------------
class Stage2Probe:
    """Instance-level wrapper on agent.policy_decomposition.select_harm_aware_leaves.

    Records every Stage-2 call and returns the ORIGINAL method's result
    unchanged. The ground-truth execution runs on a deep copy of the env
    with all global RNG state saved/restored, so the live run is untouched
    (verified by the probe-OFF control replicate).
    """

    def __init__(self, agent: REEAgent, env: CausalGridWorldV2, horizon: int) -> None:
        self.agent = agent
        self.env = env
        self.horizon = int(horizon)
        self.source = agent.policy_decomposition
        self._orig = self.source.select_harm_aware_leaves   # bound method
        self.source.select_harm_aware_leaves = self._wrapped  # instance attr
        self.events: List[Dict[str, Any]] = []
        self.episode = 0
        self.tick = 0
        self.post_done = False
        self.n_gt_errors = 0

    # -- helpers -----------------------------------------------------------
    def _field_center_harm_spread(self) -> Optional[float]:
        """Positive control for the penalty instrument: max-min harm valence
        across the residue field's ACTIVE centres (distinct points). ~0
        means the field itself is flat, so any leaf tie is field-driven."""
        try:
            rf = self.agent.residue_field
            mask = rf.rbf_field.active_mask
            if not bool(mask.any()):
                return 0.0
            centres = rf.rbf_field.centers[mask]
            with torch.no_grad():
                v = rf.evaluate_valence(centres)[..., 2]  # VALENCE_HARM_DISCRIMINATIVE
            return float((v.max() - v.min()).item())
        except Exception:
            return None

    @staticmethod
    def _pairwise_state_dist(trajs: Sequence[Any]) -> Tuple[Optional[float], Optional[float]]:
        """Mean (over leaf pairs) of mean-over-horizon L2 between predicted
        world_states, and the same for the final predicted state."""
        stacks = []
        for t in trajs:
            ws = getattr(t, "world_states", None)
            if not ws:
                return None, None
            stacks.append(torch.stack([w.detach().float() for w in ws], dim=0))  # [H,B,D]
        if len(stacks) < 2:
            return None, None
        d_all: List[float] = []
        d_end: List[float] = []
        for i in range(len(stacks)):
            for j in range(i + 1, len(stacks)):
                a, b = stacks[i], stacks[j]
                h = min(a.shape[0], b.shape[0])
                diff = (a[:h] - b[:h]).norm(dim=-1)  # [H,B]
                d_all.append(float(diff.mean().item()))
                d_end.append(float(diff[h - 1].mean().item()))
        return statistics.fmean(d_all), statistics.fmean(d_end)

    def _ground_truth(self, seq: Sequence[int]) -> Dict[str, Any]:
        """Execute the leaf's own action sequence (seq[:horizon]) on a deep
        copy of the live env. Global RNG saved/restored."""
        py_state = random.getstate()
        np_state = np.random.get_state()
        th_state = torch.get_rng_state()
        try:
            env_c = copy.deepcopy(self.env)
            harm_sum = 0.0
            haz: List[float] = []
            died = False
            n_exec = 0
            for a in list(seq)[: self.horizon]:
                _f, h, d, _i, _o = env_c.step(int(a))
                harm_sum += min(0.0, float(h))
                haz.append(_hazard_at_agent(env_c))
                n_exec += 1
                if d:
                    died = True
                    break
            return {
                "realised_harm": harm_sum,
                "hazard_mean": statistics.fmean(haz) if haz else 0.0,
                "n_exec": n_exec,
                "done_in_copy": died,
                "ok": True,
            }
        except Exception as exc:  # recorded, never raised
            self.n_gt_errors += 1
            return {"ok": False, "error": repr(exc)[:200]}
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(th_state)

    # -- the wrapper --------------------------------------------------------
    def _wrapped(self, leaves_with_penalty, z_harm_a_norm):
        kept = self._orig(leaves_with_penalty, z_harm_a_norm)
        try:
            self._record(leaves_with_penalty, z_harm_a_norm, kept)
        except Exception as exc:  # never break the run
            self.events.append({"record_error": repr(exc)[:200],
                                "episode": self.episode, "tick": self.tick})
        return kept

    def _record(self, leaves_with_penalty, z_harm_a_norm, kept) -> None:
        items = [it for it, _ in leaves_with_penalty]
        pens = [float(p) for _, p in leaves_with_penalty]
        n = len(items)
        w = float(self.source.harm_threat_scale(z_harm_a_norm))
        thr = float(self.source.config.harm_override_w_threshold)
        scale = float(self.source.config.harm_bias_scale)
        stage2_active = bool(w >= thr)
        chosen_idx: Optional[int] = None
        if stage2_active and n >= 1 and len(kept) == 1:
            for i, it in enumerate(items):
                if it is kept[0]:
                    chosen_idx = i
                    break
        # the argmin Stage 2 would take (stable, first wins) -- always recorded
        argmin_idx = min(range(n), key=lambda i: pens[i]) if n else None
        biases = []
        seqs = []
        for it in items:
            meta = getattr(it, "metadata", None) or {}
            b = meta.get("decomposition_harm_bias")
            biases.append(float(b) if b is not None else None)
            seqs.append([int(a) for a in meta.get("chunk_sequence", ())])
        spread = (max(pens) - min(pens)) if n else 0.0
        mean_p = statistics.fmean(pens) if n else 0.0
        rel_spread = spread / (abs(mean_p) + 1e-8)
        tol_tie = bool(n >= 2 and (spread < EPS_ABS or rel_spread < EPS_REL))
        exact_tie = bool(n >= 2 and spread == 0.0)
        d_all, d_end = self._pairwise_state_dist(items) if n >= 2 else (None, None)
        gts = [self._ground_truth(s) for s in seqs] if n >= 2 else []
        ev: Dict[str, Any] = {
            "episode": self.episode,
            "tick": self.tick,
            "post_done": bool(self.post_done),
            "n_leaves": n,
            "penalties": pens,
            "penalty_spread": spread,
            "penalty_mean": mean_p,
            "penalty_rel_spread": rel_spread,
            "tolerance_tie": tol_tie,
            "exact_tie": exact_tie,
            "harm_biases": biases,
            "harm_bias_scale": scale,
            "z_harm_a_norm": float(z_harm_a_norm),
            "w": w,
            "stage2_active": stage2_active,
            "chosen_idx": chosen_idx,
            "argmin_idx": argmin_idx,
            "leaf_sequences": seqs,
            "n_distinct_sequences": len({tuple(s) for s in seqs}),
            "pairwise_pred_state_dist_mean": d_all,
            "pairwise_pred_state_dist_end": d_end,
            "field_center_harm_spread": self._field_center_harm_spread(),
            "agent_cell": [int(getattr(self.env, "agent_x", -1)), int(getattr(self.env, "agent_y", -1))],
            "ground_truth": gts,
        }
        self.events.append(ev)

    def detach(self) -> None:
        # remove the instance attribute so the class method is visible again
        try:
            del self.source.select_harm_aware_leaves
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# One cell -- 919's _run_cell loop (919 :358-482), same call order, plus done
# reading, per-tick harm-vs-occupancy recording and the optional probe.
# ---------------------------------------------------------------------------
def _run_cell(seed: int, episodes: int, steps: int, probe_enabled: bool,
              role: str) -> Dict[str, Any]:
    env, agent = _build(seed)
    _register_chunk(agent)
    world_dim = agent.config.latent.world_dim
    horizon = int(agent.hippocampal.config.horizon)
    probe = Stage2Probe(agent, env, horizon) if probe_enabled else None

    print(f"Seed {seed} Condition {role}", flush=True)
    term = EpisodeTerminationAccumulator(steps_configured=steps)
    actions: List[int] = []
    harm_ticks: List[float] = []
    per_tick: List[Dict[str, Any]] = []
    episode_done: List[Dict[str, Any]] = []
    n_ticks = 0
    max_z_harm_a_norm = 0.0

    for ep in range(episodes):
        _, obs = env.reset()
        agent.reset()
        if not agent.policy_chunking.library.all_chunks():
            _register_chunk(agent)
        done_tick: Optional[int] = None
        done_cause = ""
        post_done = False
        for t in range(steps):
            if probe is not None:
                probe.episode, probe.tick, probe.post_done = ep, t, post_done
            haz_now = _hazard_at_agent(env)
            latent = agent.sense(
                obs["body_state"], obs["world_state"],
                obs_harm=obs.get("harm_obs"),
                obs_harm_a=obs.get("harm_obs_a"),
            )
            zh = getattr(latent, "z_harm", None)
            zh_norm = float(zh.detach().norm().item()) if zh is not None else None
            if getattr(latent, "z_harm_a", None) is not None:
                na = float(latent.z_harm_a.detach().norm(dim=-1).mean().item())
                max_z_harm_a_norm = max(max_z_harm_a_norm, na)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick")
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            agent.update_z_goal(
                benefit_exposure=0.0,
                drive_level=REEAgent.compute_drive_level(obs["body_state"]),
            )
            action = agent.select_action(candidates, ticks)
            a_int = int(action.argmax(dim=-1).item())
            actions.append(a_int)
            _flat, harm, done, info, obs = env.step(a_int)
            harm_ticks.append(float(harm))
            agent.update_residue(harm)
            per_tick.append({
                "ep": ep, "t": t, "post_done": post_done,
                # fresh-selection flag: the action_sequence is accumulated per
                # env step only for the 919 replay-match / probe-identity
                # checks, never as a DV; Stage-2 DVs are captured only on ticks
                # where candidates were actually regenerated.
                "e3_tick": bool(ticks.get("e3_tick")),
                "z_harm_norm": zh_norm, "hazard_at_agent": haz_now,
                "harm": float(harm),
            })
            n_ticks += 1
            if done and not post_done:
                post_done = True
                done_tick = t + 1
                done_cause = str((info or {}).get("done_cause", "")) if isinstance(info, dict) else ""
        term.record(steps=(done_tick if done_tick is not None else steps),
                    cause=(done_cause if done_tick is not None else ""))
        episode_done.append({"ep": ep, "done_tick": done_tick, "done_cause": done_cause})
        print(f"  [train] replay seed={seed} role={role} ep {ep + 1}/{episodes} "
              f"ticks={n_ticks} stage2_calls="
              f"{len(probe.events) if probe is not None else 0}", flush=True)

    _ZGOAL.observe(agent)
    state = agent.get_policy_decomposition_state()
    row: Dict[str, Any] = {
        "arm_id": ARM_ON,
        "role": role,
        "seed": int(seed),
        "probe_enabled": bool(probe_enabled),
        "episodes": episodes,
        "steps_per_episode": steps,
        "n_ticks": n_ticks,
        "rollout_horizon": horizon,
        "decomp_n_decomposed_precommit": int(state.get("decomp_n_decomposed_precommit", 0)),
        "decomp_n_harm_bias_nonzero": int(state.get("decomp_n_harm_bias_nonzero", 0)),
        "decomp_n_harm_override_fires": int(state.get("decomp_n_harm_override_fires", 0)),
        "max_z_harm_a_norm": max_z_harm_a_norm,
        "mean_harm_signal": statistics.fmean(harm_ticks) if harm_ticks else 0.0,
        "action_sequence": actions,
        "n_fresh_select": sum(1 for x in per_tick if x["e3_tick"]),
        "n_latched": sum(1 for x in per_tick if not x["e3_tick"]),
        "fresh_select_yield": round(
            sum(1 for x in per_tick if x["e3_tick"]) / max(1, len(per_tick)), 6),
        "episode_done": episode_done,
        "per_tick": per_tick,
        "stage2_events": probe.events if probe is not None else [],
        "n_gt_errors": probe.n_gt_errors if probe is not None else 0,
        "_term": term,
    }
    if probe is not None:
        probe.detach()
    cell_pass = bool(n_ticks == episodes * steps)
    row["cell_pass"] = cell_pass
    print(f"verdict: {'PASS' if cell_pass else 'FAIL'}", flush=True)
    return row


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _gt_key(g: Dict[str, Any]) -> Tuple[float, float]:
    # lower is better: more-negative realised harm is worse -> use -harm
    return (-float(g["realised_harm"]), float(g["hazard_mean"]))


def _summarise_events(evs: List[Dict[str, Any]]) -> Dict[str, Any]:
    units = [e for e in evs if "record_error" not in e and e.get("stage2_active")
             and e.get("n_leaves", 0) >= 2]
    n_units = len(units)
    n_tie = sum(1 for e in units if e["tolerance_tie"])
    n_exact = sum(1 for e in units if e["exact_tie"])
    T = (n_tie / n_units) if n_units else None
    spreads = [e["penalty_spread"] for e in units]
    rels = [e["penalty_rel_spread"] for e in units]
    dists = [e["pairwise_pred_state_dist_mean"] for e in units
             if e.get("pairwise_pred_state_dist_mean") is not None]
    field = [e["field_center_harm_spread"] for e in units
             if e.get("field_center_harm_spread") is not None]
    idx0 = [1.0 if e.get("chosen_idx") == 0 else 0.0 for e in units if e.get("chosen_idx") is not None]
    chance0 = [1.0 / e["n_leaves"] for e in units if e.get("chosen_idx") is not None]
    idx0_exact = [1.0 if e.get("chosen_idx") == 0 else 0.0 for e in units
                  if e["exact_tie"] and e.get("chosen_idx") is not None]

    # tie-cause attribution (descriptive)
    tie_causes = {"identical_leaf_sequences": 0, "field_flat": 0, "other": 0}
    for e in units:
        if not e["tolerance_tie"]:
            continue
        if e["n_distinct_sequences"] <= 1:
            tie_causes["identical_leaf_sequences"] += 1
        elif (e.get("field_center_harm_spread") is not None
              and e["field_center_harm_spread"] < EPS_ABS):
            tie_causes["field_flat"] += 1
        else:
            tie_causes["other"] += 1

    # Stage-1 saturation over every leaf of every unit
    all_b = [b for e in units for b in e["harm_biases"] if b is not None]
    sat = [1.0 if b >= e["harm_bias_scale"] - SAT_TOL else 0.0
           for e in units for b in e["harm_biases"] if b is not None]
    zero = [1.0 if b <= 0.0 else 0.0 for b in all_b]
    b_spread = [max(bs) - min(bs) for bs in
                ([b for b in e["harm_biases"] if b is not None] for e in units) if len(bs) >= 2]

    # C2 harm tracking over non-tie, GT-discriminable units
    S: List[float] = []
    C: List[float] = []
    best_sets: List[List[int]] = []
    chosen_list: List[int] = []
    first_is_best: List[float] = []
    rho_list: List[float] = []
    n_gt_ok = 0
    for e in units:
        gts = e.get("ground_truth") or []
        if len(gts) != e["n_leaves"] or not all(g.get("ok") for g in gts):
            continue
        n_gt_ok += 1
        keys = [_gt_key(g) for g in gts]
        best = min(keys)
        best_set = [i for i, k in enumerate(keys) if k == best]
        if len(best_set) == len(keys):
            continue  # GT does not separate the leaves
        first_is_best.append(1.0 if 0 in best_set else 0.0)
        r = spearman(e["penalties"], [k[0] * 1e6 + k[1] for k in keys])
        if r is not None:
            rho_list.append(r)
        if e["tolerance_tie"] or e.get("chosen_idx") is None:
            continue
        S.append(1.0 if e["chosen_idx"] in best_set else 0.0)
        C.append(len(best_set) / len(keys))
        best_sets.append(best_set)
        chosen_list.append(int(e["chosen_idx"]))
    n_gt = len(S)
    excess = (statistics.fmean(S) - statistics.fmean(C)) if n_gt else None
    # red-team F1: the chance null assumes the chosen index varies; a FIXED
    # argmin (per-action embedding offset of an untrained E2) against a
    # ground truth that favours one action (e.g. a wall-blocked move) passes
    # the chance null trivially. Null #2 = the best CONSTANT index policy.
    const_hits = {}
    for bs in best_sets:
        for k in range(max([max(b) for b in best_sets] + [0]) + 1):
            const_hits.setdefault(k, 0.0)
            const_hits[k] += 1.0 if k in bs else 0.0
    const_null = (max(const_hits.values()) / n_gt) if (n_gt and const_hits) else None
    excess_vs_const = ((statistics.fmean(S) - const_null)
                       if (n_gt and const_null is not None) else None)
    n_distinct_chosen = len(set(chosen_list))
    var_sum = sum(c * (1.0 - c) for c in C)
    z = (excess / (math.sqrt(var_sum) / n_gt)) if (n_gt and var_sum > 0) else None

    def _q(xs: List[float], q: float) -> Optional[float]:
        if not xs:
            return None
        s = sorted(xs)
        return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]

    return {
        "n_calls_total": len(evs),
        "n_units": n_units,
        "n_tolerance_tie": n_tie,
        "n_exact_tie": n_exact,
        "tie_rate_tolerance": T,
        "tie_rate_exact": (n_exact / n_units) if n_units else None,
        "tie_causes": tie_causes,
        "penalty_spread_quantiles": {q: _q(spreads, q) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)},
        "penalty_rel_spread_quantiles": {q: _q(rels, q) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)},
        "pairwise_pred_state_dist_quantiles": {q: _q(dists, q) for q in (0.0, 0.25, 0.5, 0.75, 1.0)},
        "field_center_harm_spread_median": _q(field, 0.5),
        "p_chosen_idx0": statistics.fmean(idx0) if idx0 else None,
        "p_chosen_idx0_chance": statistics.fmean(chance0) if chance0 else None,
        "p_chosen_idx0_on_exact_ties": statistics.fmean(idx0_exact) if idx0_exact else None,
        "n_leaves_hist": {str(k): sum(1 for e in units if e["n_leaves"] == k)
                          for k in sorted({e["n_leaves"] for e in units})},
        "stage1_frac_saturated": statistics.fmean(sat) if sat else None,
        "stage1_frac_zero": statistics.fmean(zero) if zero else None,
        "stage1_within_call_bias_spread_median": _q(b_spread, 0.5),
        "n_units_gt_ok": n_gt_ok,
        "n_gt_discriminable_nontie": n_gt,
        "ht_hit_rate": statistics.fmean(S) if S else None,
        "ht_chance_rate": statistics.fmean(C) if C else None,
        "ht_excess": excess,
        "ht_z": z,
        "ht_constant_index_null": const_null,
        "ht_excess_vs_constant_index": excess_vs_const,
        "ht_n_distinct_chosen_idx": n_distinct_chosen,
        "p_first_leaf_is_gt_best": statistics.fmean(first_is_best) if first_is_best else None,
        "penalty_vs_gt_spearman_median": _q(rho_list, 0.5),
        "n_penalty_vs_gt_rho": len(rho_list),
    }


def _occupancy_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ticks = [t for r in rows for t in r["per_tick"]
             if not t["post_done"] and t["z_harm_norm"] is not None
             and not math.isnan(t["hazard_at_agent"])]
    zn = [t["z_harm_norm"] for t in ticks]
    hz = [t["hazard_at_agent"] for t in ticks]
    haz_t = [t["z_harm_norm"] for t in ticks if t["hazard_at_agent"] > 0.0]
    nohaz_t = [t["z_harm_norm"] for t in ticks if t["hazard_at_agent"] <= 0.0]
    harm_t = [t["z_harm_norm"] for t in ticks if t["harm"] < 0.0]
    noharm_t = [t["z_harm_norm"] for t in ticks if t["harm"] >= 0.0]
    return {
        "n_ticks_pre_done": len(ticks),
        "z_harm_norm_mean_hazard_ticks": statistics.fmean(haz_t) if haz_t else None,
        "z_harm_norm_mean_nohazard_ticks": statistics.fmean(nohaz_t) if nohaz_t else None,
        "z_harm_norm_mean_harm_ticks": statistics.fmean(harm_t) if harm_t else None,
        "z_harm_norm_mean_noharm_ticks": statistics.fmean(noharm_t) if noharm_t else None,
        "n_hazard_ticks": len(haz_t),
        "n_harm_ticks": len(harm_t),
        "z_harm_norm_vs_hazard_spearman": spearman(zn, hz) if len(zn) > 2 else None,
    }


def _match_919(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"target_manifest": str(TARGET_MANIFEST), "per_seed": {}}
    try:
        d = json.loads(TARGET_MANIFEST.read_text())
        ref = {int(r["seed"]): r["action_sequence"] for r in d.get("arm_results", [])
               if r.get("arm_id") == ARM_ON and r.get("role") == "measurement"}
    except Exception as exc:
        out["available"] = False
        out["error"] = repr(exc)[:200]
        return out
    out["available"] = True
    for r in rows:
        a = r["action_sequence"]
        b = ref.get(int(r["seed"]))
        if b is None:
            out["per_seed"][str(r["seed"])] = None
            continue
        n = min(len(a), len(b))
        eq = sum(1 for i in range(n) if a[i] == b[i])
        prefix = next((i for i in range(n) if a[i] != b[i]), n)
        out["per_seed"][str(r["seed"])] = {
            "elementwise_match_frac": eq / n if n else None,
            "identical_prefix_len": prefix, "n": n}
    return out


def _target_post_death_share() -> Optional[Dict[str, Any]]:
    """Red-team F2 / orchestrator decision Q-1087 = A. Reconstruct agent
    health per episode from 919's own per_tick_harm (health starts 1.0; harm
    subtracts |h|, clamp 0; benefit adds 0.5*h, cap 1.0 -- causal_grid_world.py
    :2408/:2642/:2742) and return the share of 919's harm (and ticks) that
    accrued AFTER the first tick with health <= 0. None when the manifest is
    unreadable -- the gate then counts as NOT cleared (conservative)."""
    try:
        d = json.loads(TARGET_MANIFEST.read_text())
        rows = [r for r in d.get("arm_results", []) if r.get("role") == "measurement"]
        spe = int(rows[0].get("steps_per_episode", 60))
        n_ep = n_died = pre_t = post_t = 0
        harm_pre = harm_post = 0.0
        death_ticks: List[int] = []
        for r in rows:
            h = r["per_tick_harm"]
            for e0 in range(0, len(h), spe):
                seg = h[e0:e0 + spe]
                n_ep += 1
                health, dead = 1.0, None
                for t, x in enumerate(seg):
                    x = float(x)
                    if x > 0:
                        health = min(1.0, health + 0.5 * x)
                    elif x < 0:
                        health = max(0.0, health - abs(x))
                    if dead is None and health <= 0.0:
                        dead = t
                    if dead is not None and t > dead:
                        post_t += 1
                        harm_post += min(0.0, x)
                    else:
                        pre_t += 1
                        harm_pre += min(0.0, x)
                if dead is not None:
                    n_died += 1
                    death_ticks.append(dead + 1)
        tot = harm_pre + harm_post
        return {
            "n_episodes": n_ep, "n_died": n_died,
            "median_death_tick": statistics.median(death_ticks) if death_ticks else None,
            "post_death_tick_share": post_t / max(1, pre_t + post_t),
            "post_death_harm_share": (harm_post / tot) if tot != 0 else 0.0,
        }
    except Exception:
        return None


def _analyse(meas: List[Dict[str, Any]], ctrl: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    pre = [e for r in meas for e in r["stage2_events"] if not e.get("post_done")]
    post = [e for r in meas for e in r["stage2_events"] if e.get("post_done")]
    inactive = [e for e in pre if not e.get("stage2_active")]
    s = _summarise_events(pre)
    s_post = _summarise_events(post)
    n_record_errors = sum(1 for r in meas for e in r["stage2_events"] if "record_error" in e)

    # red-team F3: did the Stage-2 kept leaf reach behaviour? (reported only)
    kept_exec: List[float] = []
    for r in meas:
        spe = int(r["steps_per_episode"])
        acts = r["action_sequence"]
        for e in r["stage2_events"]:
            if (e.get("post_done") or "record_error" in e or not e.get("stage2_active")
                    or e.get("n_leaves", 0) < 2 or e.get("chosen_idx") is None):
                continue
            gi = int(e["episode"]) * spe + int(e["tick"])
            seq = e["leaf_sequences"][e["chosen_idx"]]
            if seq and 0 <= gi < len(acts):
                kept_exec.append(1.0 if acts[gi] == seq[0] else 0.0)
    kept_leaf_executed_rate = statistics.fmean(kept_exec) if kept_exec else None

    target = _target_post_death_share()
    pd_share = target["post_death_harm_share"] if target else None
    pd_gate_cleared = bool(pd_share is not None and pd_share < POST_DEATH_SHARE_MAX)

    # probe non-perturbation: control (probe OFF) vs the same seed probe ON
    mismatch: Optional[int] = None
    if ctrl is not None:
        on = next((r for r in meas if r["seed"] == ctrl["seed"]), None)
        if on is not None:
            a, b = on["action_sequence"], ctrl["action_sequence"]
            mismatch = sum(1 for i in range(min(len(a), len(b))) if a[i] != b[i]) + abs(len(a) - len(b))
    probe_ok = mismatch == 0

    T = s["tie_rate_tolerance"]
    n_units = s["n_units"]
    enough = n_units >= MIN_STAGE2_EVENTS
    n_gt = s["n_gt_discriminable_nontie"]
    excess, z = s["ht_excess"], s["ht_z"]
    exc_c = s["ht_excess_vs_constant_index"]
    tracking = bool(n_gt >= MIN_GT_EVENTS and excess is not None and z is not None
                    and excess >= HT_EXCESS_MIN and z >= HT_Z_MIN
                    and exc_c is not None and exc_c >= HT_EXCESS_MIN
                    and s["ht_n_distinct_chosen_idx"] >= 2)
    anti = bool(n_gt >= MIN_GT_EVENTS and excess is not None and z is not None
                and excess <= -HT_EXCESS_MIN and z <= -HT_Z_MIN)

    preconditions: List[Dict[str, Any]] = [
        {"name": "probe_non_perturbing", "kind": "readiness",
         "description": ("probe-OFF control replicate of the first seed must reproduce the "
                         "probe-ON action sequence exactly (count of differing actions)."),
         "control": f"seed {CONTROL_SEED}, probe OFF vs probe ON",
         "measured": float(mismatch) if mismatch is not None else float("nan"),
         "threshold_low": 0.0, "threshold_high": 0.0,
         "comparator_low": ">=", "comparator_high": "<=", "direction": "interval",
         "met": bool(probe_ok)},
        {"name": "stage2_active_multileaf_units_min", "kind": "readiness",
         "description": ("pre-done Stage-2-active calls over >= 2 leaves (the C1 unit) must "
                         "reach the pre-registered floor for the tie rate to be read."),
         "control": "pooled over the 5 measurement seeds",
         "measured": float(n_units), "threshold": float(MIN_STAGE2_EVENTS),
         "direction": "lower", "met": bool(enough)},
    ]
    in_low = bool(enough and probe_ok and T is not None and T <= TIE_RATE_LOW)
    if in_low:
        preconditions.append(
            {"name": "gt_discriminable_nontie_units_min", "kind": "readiness",
             "description": ("non-tie units whose ground truth separates the leaves (the C2 "
                             "unit) must reach the floor before harm tracking is read."),
             "control": "ground truth from deep-copied env execution per leaf",
             "measured": float(n_gt), "threshold": float(MIN_GT_EVENTS),
             "direction": "lower", "met": bool(n_gt >= MIN_GT_EVENTS)})

    sat = s["stage1_frac_saturated"]
    sat_q = "" if sat is None or sat < 0.5 else "__stage1_clamp_saturated"
    recommended = None
    if not probe_ok or not enough:
        label = "substrate_not_ready_requeue"
        degeneracy_reason = (
            "probe perturbed the run (control action mismatch=%s)" % mismatch if not probe_ok
            else "only %d Stage-2-active multi-leaf units, below floor %d" % (n_units, MIN_STAGE2_EVENTS))
        outcome = "FAIL"
    elif T >= TIE_RATE_HIGH:
        label = "stage2_tie_rate_high_harm_blind" + sat_q
        recommended = "non_contributory"
        degeneracy_reason = None
        outcome = "PASS"
    elif T > TIE_RATE_LOW:
        label = "stage2_tie_rate_intermediate_conditional_stands" + sat_q
        degeneracy_reason = None
        outcome = "FAIL"
    elif n_gt < MIN_GT_EVENTS:
        label = "stage2_discriminative_harm_tracking_undetermined" + sat_q
        degeneracy_reason = ("only %d GT-discriminable non-tie units, below floor %d"
                             % (n_gt, MIN_GT_EVENTS))
        outcome = "FAIL"
    elif tracking and pd_gate_cleared:
        label = "stage2_discriminative_and_harm_tracking" + sat_q
        recommended = "weakens_unconditional"
        degeneracy_reason = None
        outcome = "PASS"
    elif tracking:
        # Option A (orchestrator, Q-1087): 919's DV is post-death dominated, so
        # a live, harm-tracking Stage-2 input cannot make its weakens
        # unconditional -- 919 stays CONDITIONAL.
        label = ("stage2_discriminative_and_harm_tracking__919_post_death_dominated"
                 "_conditional_stands" + sat_q)
        recommended = "conditional_stands"
        degeneracy_reason = None
        outcome = "PASS"
    elif anti:
        label = "stage2_discriminative_anti_harm_tracking" + sat_q
        recommended = "non_contributory"
        degeneracy_reason = None
        outcome = "PASS"
    else:
        # red-team F6: the penalty averages a zero-action-padded rollout of
        # length rollout_horizon while GT executes only the leaf's own steps.
        label = ("stage2_discriminative_not_harm_tracking__rollout_vs_gt_horizon_mismatch"
                 "_possible" + sat_q)
        recommended = "non_contributory"
        degeneracy_reason = None
        outcome = "PASS"

    c1_decisive = bool(enough and probe_ok and T is not None
                       and (T >= TIE_RATE_HIGH or T <= TIE_RATE_LOW))
    criteria = [
        {"name": "C1_STAGE2_TOLERANCE_TIE_RATE", "load_bearing": True,
         "passed": c1_decisive,
         "measured": T, "threshold_high_tie": TIE_RATE_HIGH, "threshold_low_tie": TIE_RATE_LOW,
         "n": n_units,
         "statement": ("Stage-2 tolerance-tie rate over pre-done Stage-2-active multi-leaf "
                       "calls is decisive: >= %.2f (harm-blind) or <= %.2f (discriminative)."
                       % (TIE_RATE_HIGH, TIE_RATE_LOW))},
        {"name": "C2_HARM_TRACKING", "load_bearing": in_low,
         "passed": tracking,
         "measured": excess, "threshold": HT_EXCESS_MIN,
         "measured_z": z, "threshold_z": HT_Z_MIN, "n": n_gt,
         "measured_vs_constant_index": exc_c, "threshold_vs_constant_index": HT_EXCESS_MIN,
         "statement": ("Among non-tie GT-discriminable calls, the chosen leaf is a "
                       "ground-truth-best leaf more often than chance by >= %.2f with z >= %.1f."
                       % (HT_EXCESS_MIN, HT_Z_MIN))},
        {"name": "R4_919_DV_PRE_DEATH_DOMINANT", "load_bearing": bool(in_low and tracking),
         "passed": pd_gate_cleared,
         "measured": pd_share, "threshold": POST_DEATH_SHARE_MAX,
         "statement": ("Option A gate (orchestrator decision Q-1087): the '919 weakens "
                       "unconditional' route requires 919's post-death harm share (health "
                       "reconstructed from its own per_tick_harm) < %.2f." % POST_DEATH_SHARE_MAX)},
        {"name": "R5_KEPT_LEAF_EXECUTED_RATE", "load_bearing": False,
         "passed": bool(kept_leaf_executed_rate is not None and kept_leaf_executed_rate > 0.0),
         "measured": kept_leaf_executed_rate, "threshold": 0.0,
         "statement": ("Reported (red-team F3): fraction of Stage-2-active units whose kept "
                       "leaf's first action was the action executed that tick.")},
        {"name": "R3_STAGE1_CLAMP_SATURATION", "load_bearing": False,
         "passed": bool(sat is not None and sat < 0.5),
         "measured": sat, "threshold": 0.5,
         "statement": "Reported: fraction of leaves whose Stage-1 harm_bias sits at the clamp."},
    ]
    criteria_non_degenerate = {
        "C1_STAGE2_TOLERANCE_TIE_RATE": bool(enough and probe_ok),
        "C2_HARM_TRACKING": bool(n_gt >= MIN_GT_EVENTS and s["ht_chance_rate"] is not None
                                 and s["ht_chance_rate"] < 1.0
                                 and s["ht_n_distinct_chosen_idx"] >= 2),
    }
    combination_rule = (
        "probe_non_perturbing AND units>=MIN_STAGE2_EVENTS else substrate_not_ready_requeue; "
        "T>=TIE_RATE_HIGH -> harm_blind (919 non_contributory); "
        "TIE_RATE_LOW<T<TIE_RATE_HIGH -> intermediate (919 stays conditional); "
        "T<=TIE_RATE_LOW -> C2 decides: tracking AND 919 post-death harm share < "
        "POST_DEATH_SHARE_MAX -> 919 weakens unconditional; tracking otherwise -> 919 stays "
        "conditional (Option A, Q-1087); "
        "anti or neither -> 919 non_contributory; GT units<MIN_GT_EVENTS -> undetermined. "
        "Stage-1 saturation (>=0.5) is a label qualifier only.")
    return {
        "outcome": outcome, "label": label, "degeneracy_reason": degeneracy_reason,
        "non_degenerate": bool(enough and probe_ok),
        "recommended_readjudication_of_919": recommended,
        "preconditions": preconditions, "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
        "combination_rule": combination_rule,
        "stage2_pre_done": s, "stage2_post_done": s_post,
        "stage2_inactive_calls": {"n": len(inactive),
                                  "summary": _summarise_events(
                                      [dict(e, stage2_active=True) for e in inactive])},
        "occupancy": _occupancy_summary(meas),
        "probe_control_action_mismatch": mismatch,
        "kept_leaf_executed_rate": kept_leaf_executed_rate,
        "n_kept_leaf_exec_units": len(kept_exec),
        "target_919_post_death": target,
        "n_record_errors": n_record_errors,
        "n_gt_errors": sum(r["n_gt_errors"] for r in meas),
    }


def _flat(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> Tuple[str, str, bool]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        seeds: Tuple[int, ...] = MEASUREMENT_SEEDS[:1]
        episodes, steps = 2, 20
    else:
        seeds = MEASUREMENT_SEEDS
        episodes, steps = baselines.EPISODES, baselines.STEPS_PER_EPISODE

    t0 = time.perf_counter()
    started = datetime.now(timezone.utc)
    config_matches_919 = _config_matches_919()

    meas: List[Dict[str, Any]] = []
    for seed in seeds:
        with arm_cell(seed, config_slice=_config_slice(episodes, True),
                      script_path=Path(__file__), config_slice_declared=True) as cell:
            row = _run_cell(seed, episodes, steps, probe_enabled=True, role="measurement")
            cell.stamp(row)
        meas.append(row)
    with arm_cell(CONTROL_SEED, config_slice=_config_slice(episodes, False),
                  script_path=Path(__file__), config_slice_declared=True) as cell:
        ctrl = _run_cell(CONTROL_SEED, episodes, steps, probe_enabled=False,
                         role="probe_off_control")
        cell.stamp(ctrl)

    res = _analyse(meas, ctrl)
    match = _match_919(meas)

    term = EpisodeTerminationAccumulator(steps_configured=steps)
    for r in meas + [ctrl]:
        for e in r["episode_done"]:
            term.record(steps=(e["done_tick"] if e["done_tick"] is not None else steps),
                        cause=e["done_cause"])
    for r in meas + [ctrl]:
        r.pop("_term", None)

    s = res["stage2_pre_done"]
    readout = {k: v for k, v in {
        "tie_rate_tolerance": _flat(s["tie_rate_tolerance"]),
        "tie_rate_exact": _flat(s["tie_rate_exact"]),
        "n_stage2_units": _flat(s["n_units"]),
        "penalty_spread_median": _flat(s["penalty_spread_quantiles"][0.5]),
        "penalty_rel_spread_median": _flat(s["penalty_rel_spread_quantiles"][0.5]),
        "pairwise_pred_state_dist_median": _flat(s["pairwise_pred_state_dist_quantiles"][0.5]),
        "field_center_harm_spread_median": _flat(s["field_center_harm_spread_median"]),
        "p_chosen_idx0": _flat(s["p_chosen_idx0"]),
        "p_chosen_idx0_chance": _flat(s["p_chosen_idx0_chance"]),
        "stage1_frac_saturated": _flat(s["stage1_frac_saturated"]),
        "stage1_frac_zero": _flat(s["stage1_frac_zero"]),
        "n_gt_discriminable_nontie": _flat(s["n_gt_discriminable_nontie"]),
        "ht_excess": _flat(s["ht_excess"]),
        "ht_z": _flat(s["ht_z"]),
        "ht_excess_vs_constant_index": _flat(s["ht_excess_vs_constant_index"]),
        "ht_n_distinct_chosen_idx": _flat(s["ht_n_distinct_chosen_idx"]),
        "ht_hit_rate": _flat(s["ht_hit_rate"]),
        "ht_chance_rate": _flat(s["ht_chance_rate"]),
        "p_first_leaf_is_gt_best": _flat(s["p_first_leaf_is_gt_best"]),
        "n_stage2_post_done": _flat(res["stage2_post_done"]["n_calls_total"]),
        "probe_control_action_mismatch": _flat(res["probe_control_action_mismatch"]),
        "z_harm_norm_vs_hazard_spearman": _flat(res["occupancy"]["z_harm_norm_vs_hazard_spearman"]),
        "kept_leaf_executed_rate": _flat(res["kept_leaf_executed_rate"]),
        "target_919_post_death_harm_share": _flat(
            (res["target_919_post_death"] or {}).get("post_death_harm_share")),
        "non_degenerate": _flat(res["non_degenerate"]),
    }.items() if v is not None}

    run_id = f"{EXPERIMENT_TYPE}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3"
    cfg_record = {
        "seeds": list(seeds), "control_seed": CONTROL_SEED,
        "episodes": episodes, "steps_per_episode": steps,
        "arm": ARM_ON, "arm_flags": _arm_flags_on(),
        "env_overlay": dict(baselines.HAZARD_TUNED_ENV_OVERLAY),
        "config_matches_919_arm_flags": config_matches_919,
        "thresholds": {
            "EPS_ABS": EPS_ABS, "EPS_REL": EPS_REL, "TIE_RATE_HIGH": TIE_RATE_HIGH,
            "TIE_RATE_LOW": TIE_RATE_LOW, "MIN_STAGE2_EVENTS": MIN_STAGE2_EVENTS,
            "MIN_GT_EVENTS": MIN_GT_EVENTS, "HT_EXCESS_MIN": HT_EXCESS_MIN,
            "HT_Z_MIN": HT_Z_MIN, "SAT_TOL": SAT_TOL,
            "POST_DEATH_SHARE_MAX": POST_DEATH_SHARE_MAX,
        },
    }
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "queue_id": QUEUE_ID,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "outcome": res["outcome"],
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "unknown",
        "bears_on": ["MECH-321", "SD-hazard-aware-policy-decomposition", "V3-EXQ-919"],
        "target_run_id": TARGET_RUN_ID,
        "recommended_readjudication_of_919": res["recommended_readjudication_of_919"],
        "non_degenerate": res["non_degenerate"],
        "degeneracy_reason": res["degeneracy_reason"],
        "readout": readout,
        "interpretation": {
            "label": res["label"],
            "preconditions": res["preconditions"],
            "criteria": res["criteria"],
            "criteria_non_degenerate": res["criteria_non_degenerate"],
            "combination_rule": res["combination_rule"],
        },
        "stage2_pre_done": res["stage2_pre_done"],
        "stage2_post_done": res["stage2_post_done"],
        "stage2_inactive_calls": res["stage2_inactive_calls"],
        "occupancy_vs_harm": res["occupancy"],
        "occupancy_tautology_note": (
            "red-team F5: harm_obs includes the hazard field view and the hazard field is "
            "dense (hazard>0 on essentially every tick), so the hazard/no-hazard split is "
            "empty and z_harm_norm_vs_hazard_spearman partly reads the encoder's own input. "
            "Do NOT cite it as harm-vs-occupancy evidence; C2 carries that question."),
        "instrument_horizon_note": (
            "red-team F6: each leaf penalty averages rollout_horizon predicted states, all "
            "but len(leaf) driven by zero action vectors; ground truth executes only the "
            "leaf's own steps. A not-tracking result may be this mismatch."),
        "target_919_post_death": res["target_919_post_death"],
        "kept_leaf_executed_rate": res["kept_leaf_executed_rate"],
        "replay_match_to_919": match,
        "n_record_errors": res["n_record_errors"],
        "n_gt_errors": res["n_gt_errors"],
        "arm_results": meas + [ctrl],
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False, "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False, "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False, "decision": "allow"},
        "sleep_driver_pattern": "not_applicable",
        "custom_information": {
            "source_autopsy": "REE_assembly/evidence/planning/failure_autopsy_MECH-320-defect-cluster_2026-09-23.md",
            "preflight": "orchestrate-20260924 pre-flight AMBER, 5 named changes applied",
            "comparator_note": ("919's agent is untrained: compare to the May untrained 0.0000 "
                                "per-candidate spread, not the August trained 0.02-0.16."),
        },
    }
    out_path = write_flat_manifest(
        manifest, None, dry_run=args.dry_run, config=cfg_record,
        seeds=list(seeds) + [CONTROL_SEED], script_path=Path(__file__),
        elapsed_seconds=round(time.perf_counter() - t0, 3),
        z_goal_stream_stats=_ZGOAL.stats(), episode_termination=term,
    )
    print(f"manifest: {out_path}", flush=True)
    print(f"outcome: {res['outcome']} label={res['label']} "
          f"recommend_919={res['recommended_readjudication_of_919']}", flush=True)
    print(f"  units={s['n_units']} tie_rate_tol={s['tie_rate_tolerance']} "
          f"tie_rate_exact={s['tie_rate_exact']} ht_excess={s['ht_excess']} ht_z={s['ht_z']} "
          f"n_gt={s['n_gt_discriminable_nontie']} stage1_sat={s['stage1_frac_saturated']}",
          flush=True)
    print(f"  probe_mismatch={res['probe_control_action_mismatch']} "
          f"record_errors={res['n_record_errors']} gt_errors={res['n_gt_errors']} "
          f"config_matches_919={config_matches_919} started={started.strftime('%Y%m%dT%H%M%SZ')}",
          flush=True)

    if args.dry_run:
        n_calls = sum(len(r["stage2_events"]) for r in meas)
        n_gt_exec = sum(1 for r in meas for e in r["stage2_events"]
                        for g in (e.get("ground_truth") or []) if g.get("ok"))
        print(f"[smoke] stage2_calls={n_calls} gt_exec_ok={n_gt_exec} "
              f"record_errors={res['n_record_errors']} gt_errors={res['n_gt_errors']} "
              f"probe_mismatch={res['probe_control_action_mismatch']}", flush=True)
        assert n_calls > 0, "SMOKE FAIL: the Stage-2 wrapper never fired -- do not queue."
        assert res["n_record_errors"] == 0, "SMOKE FAIL: probe record errors."
        assert res["n_gt_errors"] == 0, "SMOKE FAIL: ground-truth execution errors."
        assert n_gt_exec > 0, "SMOKE FAIL: no ground-truth execution happened."
        assert res["probe_control_action_mismatch"] == 0, (
            "SMOKE FAIL: the probe perturbed the run (control mismatch).")

    oc = str(res["outcome"]).upper()
    return (oc if oc in ("PASS", "FAIL") else "FAIL"), str(out_path), bool(args.dry_run)


if __name__ == "__main__":
    _o, _p, _d = main()
    emit_outcome(outcome=_o, manifest_path=_p, dry_run=_d)
