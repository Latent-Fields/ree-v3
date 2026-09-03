"""V3-EXQ-995 -- EXT-005 claim probe: does REE compute a genuine CAUSAL SIGNATURE?

EXT-005 ("Causal attribution gap: language describes causation without a causal
signature mechanism") asserts that LLMs produce fluent causal language without any
internal mechanism that distinguishes agent-caused from world-caused state change.
The claim's own notes name REE's remedy: a comparator over the world-state stream.
SD-003 stated it as a two-pass counterfactual and is SUPERSEDED; SD-031 supersedes
it with the single-pass form actually built in this substrate:

    residual_world = z_world_observed - E2WorldForward(z_world_{t-1}, a_actual)

Large residual -> the agent's own action did not predict the change -> externally
caused. Small residual -> self-caused. This probe asks whether that residual ACTUALLY
discriminates, i.e. whether REE has the mechanism EXT-005 says LLMs lack -- or whether
REE's "causal signature" is merely a change-magnitude detector wearing the name, which
would be EXT-005's failure mode reproduced one level down.

WHY THIS RUN IS THE FIRST ACTUAL MEASUREMENT OF IT (GOV-REUSE-1, Step 2.4)
    Decisive readout: `auroc_move_ok` -- AUROC of the comparator-residual norm
    discriminating exogenously-caused from purely-self-caused world change, measured
    in the one stratum where raw change magnitude is uninformative (below).
    Searched: every manifest in REE_assembly/evidence/experiments/. Exactly two carry
    an attribution readout at all -- v3_exq_177 (2026-03-29) and v3_exq_783
    (substrate_hash 7f856703..., 2026-07-18). 783 is the only run that reached
    attribution_ready=True (its D128_* arms), and ALL 16 of those arms recorded
      attribution_gap: null, "insufficient events of one or both causal classes".
    So the readout has never been obtained: the attribution arm has been starved of
    balanced causal-class events every time it has been attempted, exactly as
    failure_autopsy_zworld-integration-cluster_2026-06-06 predicted ("problem is
    exposure balance, not env"). NOT RECOVERABLE -> run. This driver's contribution is
    to MANUFACTURE that balance by construction rather than hope a learned policy
    supplies it (see CAUSAL-CLASS BALANCE below), which is also why it does not
    inherit 783's dependency on ARC-065 behavioural diversity.

WHAT MAKES THIS RUNNABLE ON THE CURRENT SUBSTRATE (Step 2.5 / 2.5a)
    Measurement-only. Nothing here needs E3 committed selection, the basal-ganglia
    commitment layer, or the conversion pathway, so it is untouched by the
    conversion-ceiling / F-dominance wall. The agent object is constructed but never
    driven: only `latent_stack.split_encoder` (to encode z_world) and a standalone
    `E2WorldForward` head are exercised. A random behaviour policy generates the
    stream. Empirically confirmed at authoring time (2026-09-03, seed 42):
      - world_dim=128 reachable via REEConfig.from_dims -> attribution_ready True
        (the dim=32 hard-assert is enforced; verified it refuses at 32)
      - E2WorldForward action pathway is WIRED: ||f(z,a0)-f(z,a2)|| = 0.263 at init
      - run_zworld_p0 (SD-070 P0a) moves 4/4 world_encoder tensors,
        p0a_holdout_mean_lift 0.583 at 250 episodes
    The 2026-06-06 dim-32 ceiling that blocked this question is REFUTED, by its own
    prescribed retest: failure_autopsy_V3-EXQ-783_2026-07-18 established that z_world
    under-differentiation is a TRAINING fault (SD-070 fixes it), not a dimensionality
    fault. Re-derive brake (Step 2.5b): 0 counting autopsies on EXT-005, 0 on SD-031.

CAUSAL-CLASS BALANCE, AND THE DEGENERACY THIS DESIGN HAD TO ROUTE AROUND
    The env supplies exogenous world change through hazard drift. With
    env_drift_interval=2 and env_drift_prob=1.0 a drift is attempted every other step.
    GROUND TRUTH IS TAKEN FROM HAZARD COORDINATES DIRECTLY (positions before vs after
    each step), NOT from info["env_drift_occurred"] -- that flag is set on every drift
    TICK regardless of whether any hazard actually moved (causal_grid_world.py:3063-65
    sets it unconditionally; _drift_hazards' own `drifted` result is discarded), so it
    over-reports. The flag is recorded alongside for audit. Measured at authoring:
    0/2000 disagreements at env_drift_prob=1.0, but the coordinate label is correct by
    construction and the flag is not.

    Each step is classed by ACTION OUTCOME -- STAY (action 4, a deliberate no-move),
    MOVE_OK (agent displaced), MOVE_BLOCKED (move attempted, agent did not displace;
    scheduled_action_block_enabled is False so this is always a wall/entity collision).
    Crossed with drift/no-drift this gives six well-populated cells (measured over 6000
    random steps: 627/537/2172/1848/458/358).

    THE VERDICT IS READ FROM MOVE_OK ONLY, AND THAT IS THE LOAD-BEARING DESIGN CHOICE.
    In STAY and MOVE_BLOCKED the agent does not displace, so the ONLY thing that can
    change the world observation is a hazard -- the label is trivially decodable from
    raw change magnitude with no forward model at all. Measured at authoring: bare
    ||z_obs - z_prev|| gives AUROC 0.996 (STAY) and 1.000 (MOVE_BLOCKED). A criterion
    read there would be an arithmetic identity, not a measurement, and no comparator
    could beat the ceiling. Those two strata are therefore retained as RECORDED
    POSITIVE CONTROLS (they prove z_world does represent hazard position at all) and
    are explicitly NOT verdict-bearing.
    In MOVE_OK the agent's own displacement dominates the change magnitude and the raw
    signal collapses to chance (measured 0.453). Any discrimination there MUST come
    from action-conditioned prediction. That is precisely EXT-005's question: can the
    system tell "the world changed because of me" from "the world changed on its own"
    while its own action is producing a large change at the same time?

PRE-REGISTERED THRESHOLDS (constants below; fixed before the run)
    Calibrated on a single-seed AUTHORING PILOT (seed 42, 30k transitions, held-out
    30%, 40 head epochs) whose values are recorded here for audit:
        bare MOVE_OK 0.4532 | BASE C1 0.5942  C2 +0.1409  C3 +0.0798
                            | INTERV C1 0.5891 C2 +0.1359 C3 +0.0689
    THE VERDICT SEEDS EXCLUDE 42 ENTIRELY (SEEDS = 43,45,46,47,48), so every scored
    seed is out-of-sample with respect to threshold calibration. Seed 44 is skipped as
    a standing precaution (per-seed early-death instability, EXQ-539/540/538a) even
    though this env is not a reef config.
    C1/C2 are NOISE-AWARE rather than bright lines: each must clear an absolute floor
    AND one cross-seed SD of its own paired distribution. That form is deliberate --
    failure_autopsy_V3-EXQ-783_2026-07-18 section 3 records a label decided by a
    0.0028 margin against a statistic whose own SEM was five times larger.

    C1 (LOAD-BEARING) mean(auroc_move_ok) - 0.5 >= max(0.05, 1.0 * sd)
        The comparator residual discriminates exogenous from self-caused change.
    C2 (LOAD-BEARING) mean(auroc_move_ok - auroc_bare_move_ok) >= max(0.05, 1.0 * sd)
        The forward-model conditioning adds discrimination BEYOND raw change
        magnitude. This is the EXT-005-critical criterion: fail it and REE's causal
        signature is a change detector, which is the charge EXT-005 levels at LLMs.
    C3 (supporting, not load-bearing) mean(auroc_move_ok - auroc_shuffled) >= 0.02
        The added discrimination is specifically ACTION-conditioned: recomputing the
        residual against a permuted action (marginal action distribution preserved,
        state-action pairing destroyed) should degrade it.
    COMBINATION RULE: overall PASS = C1 AND C2, both on the BASE arm. Recorded
    explicitly in the manifest as `combination_rule` so a reader need not open this
    file to know that C3 and the INTERV arm are not verdict-bearing.

ARM (single), AND DV-SYMMETRY INVARIANCE (declared per arm, as required)
    BASE -- E2WorldForward trained with plain MSE: the SD-031 single-pass form, which
    is the mechanism actually built in this substrate. VERDICT-BEARING.
    Its DV is a within-stratum AUROC of the per-step comparator-residual norm against
    the exogenous-change label. The symmetry group of that statistic is monotone
    rescalings of the residual norm and permutations WITHIN a label class. The
    manipulation is NOT invariant under it: training E2WorldForward changes the
    residual as a learned function of (z_world_prev, a) and therefore RE-RANKS steps
    ACROSS the label boundary -- it is not a broadcast additive constant (which would
    cancel in a rank statistic), not a monotone rescaling (which would leave AUROC
    exactly fixed), and not a within-class permutation. Confirmed empirically at
    authoring: BASE moved MOVE_OK AUROC from the bare 0.4532 to 0.5942, i.e. across the
    label boundary. The C2 reference term `auroc_bare_move_ok` involves no head at all,
    so it is a fixed per-seed reference, not a second manipulation.
    Cells are stamped reuse-INELIGIBLE (they share one P0a-trained encoder and one
    transition buffer per seed), so they are not independent functions of
    (substrate, config, seed).

THE SD-013 ARM THAT WAS REMOVED -- a design-time proof, recorded because it is a
FINDING about SD-013, not merely a dropped arm.
    An INTERV arm was designed and built: identical to BASE plus SD-013's contrastive
    interventional loss (`compute_interventional_loss`), the closest live analogue of
    SD-003's superseded two-pass counterfactual, which the SD-031 design doc names as
    higher priority for z_world precisely because ambient world correlations compress
    the action contribution. It was REMOVED before queueing, on measurement, because it
    is STRUCTURALLY INERT at this substrate's own default margin:
      - the loss is a hinge, max(0, interventional_margin - ||f(z,a) - f(z,a_cf)||),
        with E2WorldConfig.interventional_margin defaulting to 0.1;
      - the model's action-divergence ||f(z,a) - f(z,a_cf)|| is ALREADY ~0.234 at
        initialisation (measured, 2 seeds) and RISES with training (~2.2 by epoch 40 in
        the authoring pilot), so the hinge is slack from the first step onward and
        contributes exactly zero gradient. Measured engagement: 0 of 60 batches on
        every seed, where ENGAGED is defined as some row having 0 < l2_dist < margin,
        i.e. the hinge actually supplying gradient.
      - the arms came out BIT-IDENTICAL on every metric (wiring, AUROC, held-out MSE).
    Two traps this surfaced, both worth carrying forward: (1) a nonzero interventional
    LOSS is not evidence of engagement -- drawing a_cf uniformly lets it COINCIDE with
    a_actual, and a coincident pair has l2_dist == 0, so the hinge returns loss ==
    margin while `norm` at zero contributes zero gradient; that violates
    compute_interventional_loss's own documented contract (e2_world.py:341, "must
    differ") and makes an inert arm look active. (2) The authoring pilot APPEARED to
    show BASE/INTERV divergence (0.5942 vs 0.5891); that was RNG-consumption drift from
    drawing a_cf out of the shared generator, not the manipulation -- an artifact.
    The margin was deliberately NOT raised to force engagement: a pre-registered value
    that provably fails its gate is a design-time proof, not something to tune away.
    Testing SD-013 properly needs a margin calibrated to z_world's operating scale at
    world_dim=128, which is a separate question from EXT-005 and is left as follow-on.
    The verdict-bearing science is untouched -- C1/C2/C3 were never a function of this
    arm. The ARMS tuple retains its shape so a successor can restore it.

SUBSTRATE-PATH OVERLAP (Step 2.5c) -- checked, no exercised corrupting defect.
    Driver footprint: causal_grid_world.py, utils/config.py, agent.py (construction
    only), latent/stack.py, latent/zworld_p0.py, predictors/e2_world.py,
    _lib/zworld_p0_warmup.py, _lib/capability_eval.py. Open CORRUPTING entries that
    name any of these are all gated behind flags this driver never sets, VERIFIED live
    on the exact config below rather than assumed: mode-governance-engagement
    (salience_affinity_input_cap=None), SD-082 (use_lateral_pfc_analog=False);
    SD-e1-rollout-consistency-training and contextmemory-write-path-addressing-
    degeneracy sit in e1_deep.py, which is never reached because the agent is never
    driven (no sense(), no select_action()). Open DEGRADING entries that do overlap,
    recorded per the gate: SD-018 (latent/stack.py, latent/zworld_p0.py -- the
    resource-proximity/directional-field legs; both heads are absent here,
    use_resource_proximity_head=False and use_resource_field_head=False, so
    p0a_used_proximity_head / p0a_used_resource_field_head are recorded and expected
    False), mech357-freeze-incompatible-pressure-mechanism and
    SD-MECH303-THRESHOLD-SOURCING (causal_grid_world.py; reef_enabled False,
    hazard_agent_pursuit 0.0, hazard_food_attraction 0.0).

ETHICS PREFLIGHT (Step 2.6): all involvement flags false, decision allow. V3, not
    claimed sentient; no negative-valence self-model; the agent is not even driven.

PHASED TRAINING: P0a SD-070 z_world encoder warmup (run_zworld_p0) -> encoder FROZEN
    (requires_grad_(False) on the whole latent_stack) -> P1 E2WorldForward trained on
    .detach()ed z_world targets -> P2 held-out evaluation. No head ever back-propagates
    into the encoder.

RUNNER PROGRESS: the per-seed unit is emitted as TWO runner-visible conditions,
    P0A_ENCODER and P1_COLLECT, each with the same episode denominator
    (EPISODES_PER_PHASE), so the queue entry's episodes_per_run matches both loops and
    the boundary line resets the counter between them. seeds x conditions = 5 x 2 = 10
    `verdict:` lines.

red-team (fable), Step 4.5, ONE pass: CONTESTED -> all three findings fixed before
queueing, none dismissed. (1) family 4, the wiring precondition was measured AFTER
training, so a BASE head that LEARNS action-irrelevance -- which IS the
causal_signature_absent_change_detector_only finding -- would have driven the probe to
the floor and misrouted the run to substrate_not_ready_requeue, byte-identical to dead
wiring; fixed by gating on the AT-INIT probe and recording the trained value as a
per-arm diagnostic. (2) family 3, `overall = C1 and C2` never consulted C3, so C1+C2
passing with the action-permutation control FAILING would have recorded the unqualified
label causal_signature_present / supports; fixed with the qualified label
causal_signature_present_action_conditioning_unconfirmed. (3) family 3, EXT-005's
registered subject is `llm.causal_attribution`, which no V3 run can observe, so a bare
supports/weakens on it is unattributable between the LLM assertion and the REE remedy
annotation; fixed by recording a `claim_scope` block with an explicit attribution_caveat
scoping the direction to the remedy (the V3-EXQ-991 attribution_caveat precedent).
A fourth, MINOR finding -- that `substrate_not_ready_requeue` names no automated
requeue -- is DISMISSED: that label is the skill-mandated self-route for a below-floor
readiness precondition and the requeue is a governance action (pending_review ->
/failure-autopsy -> new letter), not automation. Recorded as `requeue_semantics` in the
manifest so it does not read as a broken promise. The reviewer separately verified
clean: the manipulation reaches the DV, the comparator's zero-sentinel is unreachable at
dim 128, the periodic drift label does not leak into z_world via body_state[9] (z_world
is encoded from world_obs only, stack.py:965), and C1/C2 are jointly satisfiable.
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell, seeded_construct  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.zworld_p0_warmup import run_zworld_p0  # noqa: E402
from experiments._lib.capability_eval import RandomPolicy  # noqa: E402
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot,
    assert_world_encoder_trained,
    zworld_precondition,
)
from experiments._metrics import check_degeneracy  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.predictors.e2_world import E2WorldForward, E2WorldConfig  # noqa: E402

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_995_claim_probe_ext_005_causal_signature"
QUEUE_ID = "V3-EXQ-995"
CLAIM_IDS = ["EXT-005"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# ---------------------------------------------------------------- configuration
SEEDS: List[int] = [43, 45, 46, 47, 48]   # 42 = authoring pilot, deliberately excluded
WORLD_DIM = 128                            # E2WorldForward hard-asserts >= 128
ALPHA_WORLD = 0.9                          # SD-008: z_world fidelity

ENV_KWARGS: Dict[str, Any] = dict(
    size=8,
    num_hazards=3,
    num_resources=3,
    contamination_spread=0.0,   # V3-EXQ-513 precedent: never leave this at its 0.5 default
    env_drift_interval=2,       # drift attempted every other step -> balanced classes
    env_drift_prob=1.0,         # every hazard attempts to move on a drift tick
    use_proxy_fields=True,
)

EPISODES_PER_PHASE = 250        # denominator for BOTH phases; == queue episodes_per_run
P0A_STEPS_PER_EPISODE = 100
TRANSITIONS_PER_BLOCK = 160     # 250 blocks x 160 = 40000 transitions per seed
N_TRANSITIONS = EPISODES_PER_PHASE * TRANSITIONS_PER_BLOCK
HELDOUT_FRACTION = 0.30
HEAD_EPOCHS = 60
HEAD_BATCH = 256
HEAD_LR = 3e-4
AUROC_CURVE_EVERY = 10          # record held-out C1 every N epochs (undertrained vs ceiling)

INTERVENTIONAL_FRACTION = 0.3
INTERVENTIONAL_MARGIN = 0.1

STAY_ACTION = 4                 # CausalGridWorldV2.ACTIONS[4] == (0, 0)

# SINGLE ARM. An SD-013 interventional arm was designed, built and then REMOVED before
# queueing on design-time proof that it is structurally inert at this substrate's own
# default margin -- see "THE SD-013 ARM THAT WAS REMOVED" in the module docstring. The
# tuple shape is retained so a successor with a calibrated margin can restore the arm
# without restructuring the driver.
ARMS: List[Tuple[str, bool]] = [("BASE", False)]

# ------------------------------------------------------- pre-registered thresholds
THRESH_C1_ABS_FLOOR = 0.05      # mean AUROC must clear 0.5 by this much
THRESH_C1_SD_MULT = 1.0         # ... and by one cross-seed SD of its own distribution
THRESH_C2_ABS_FLOOR = 0.05      # comparator must beat bare-delta by this much
THRESH_C2_SD_MULT = 1.0
THRESH_C3_MARGIN = 0.02         # supporting: action-conditioned component

# -------------------------------------------------------- readiness preconditions
FLOOR_STAY_BARE_AUROC = 0.80    # positive control: z_world DOES encode hazard change
BAND_MOVE_OK_BARE_LOW = 0.40    # negative control: no raw-magnitude confound in the
BAND_MOVE_OK_BARE_HIGH = 0.60   # ... verdict stratum (interval precondition)
FLOOR_MOVE_OK_CLASS_N = 500     # min(n_drift, n_nodrift) in held-out MOVE_OK, per seed
FLOOR_ACTION_PATHWAY = 1e-4     # WIRING check, not a learning check (see below)

PILOT_RECORD = {
    "seed": 42, "n_transitions": 30000, "head_epochs": 40, "heldout_fraction": 0.30,
    "bare_auroc_move_ok": 0.4532, "bare_auroc_stay": 0.9956, "bare_auroc_move_blocked": 1.0,
    "base_c1": 0.5942, "base_c2": 0.1409, "base_c3": 0.0798,
    "interv_c1": 0.5891, "interv_c2": 0.1359, "interv_c3": 0.0689,
    "note": "single-seed authoring pilot used ONLY to confirm the pre-registered "
            "thresholds are satisfiable and the criteria non-degenerate; seed 42 is "
            "excluded from SEEDS so no scored seed is in-sample.",
}


# --------------------------------------------------------------------- utilities
def _auroc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Rank-based AUROC. None when either class is empty."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n1 = int(y.sum())
    n0 = int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # average ranks over ties so a tied score cannot inflate the statistic
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts), dtype=float)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _mean_sd(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), sd


def _worst_low(values: List[Tuple[float, Any]]) -> Tuple[Optional[float], Any]:
    """Worst (minimum) cell of a floor precondition, with its offending cell id."""
    vals = [(v, k) for v, k in values if v is not None]
    if not vals:
        return None, None
    v, k = min(vals, key=lambda t: t[0])
    return float(v), k


def _worst_interval(values: List[Tuple[float, Any]], centre: float) -> Tuple[Optional[float], Any]:
    """Worst cell of a two-sided precondition: the one furthest from `centre`."""
    vals = [(v, k) for v, k in values if v is not None]
    if not vals:
        return None, None
    v, k = max(vals, key=lambda t: abs(t[0] - centre))
    return float(v), k


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _build_agent(env: CausalGridWorldV2) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
    )
    return REEAgent(cfg)


def _encode_world(split_encoder, obs_dict) -> torch.Tensor:
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    with torch.no_grad():
        return split_encoder(body, world)[1]


def _config_slice(seed: int, arm_id: str, interventional: bool) -> Dict[str, Any]:
    """The declared slice a cell's computation actually reads.

    `arm_id` / `interventional` MUST be in here: they are the only thing that differs
    between the two cells of a seed, so omitting them makes BASE and INTERV hash to the
    SAME arm_fingerprint (confirmed at authoring, before this fix) -- two distinct cells
    sharing one cell identity.
    """
    return {
        "arm_id": arm_id,
        "interventional": bool(interventional),
        "interventional_fraction": INTERVENTIONAL_FRACTION if interventional else None,
        "interventional_margin": INTERVENTIONAL_MARGIN if interventional else None,
        "env_kwargs": dict(ENV_KWARGS),
        "world_dim": WORLD_DIM,
        "alpha_world": ALPHA_WORLD,
        "episodes_per_phase": EPISODES_PER_PHASE,
        "p0a_steps_per_episode": P0A_STEPS_PER_EPISODE,
        "n_transitions": N_TRANSITIONS,
        "heldout_fraction": HELDOUT_FRACTION,
        "head_epochs": HEAD_EPOCHS,
        "head_batch": HEAD_BATCH,
        "head_lr": HEAD_LR,
        "seed": int(seed),
    }


# ------------------------------------------------------------------ P0a + P1 data
def _run_p0a(agent: REEAgent, seed: int, episodes: int, steps: int,
             dry_run: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """SD-070 z_world encoder warmup on a DEDICATED env (its rollout consumes env RNG)."""
    print("Seed %d Condition P0A_ENCODER" % seed, flush=True)
    before = latent_stack_snapshot(agent)
    warm_env = _build_env(seed)
    stats = run_zworld_p0(
        agent, warm_env, seed, episodes, steps,
        policy=RandomPolicy(seed), label="ext005", dry_run=dry_run,
    )
    guard = assert_world_encoder_trained(
        agent, before, p0=episodes, strict=not dry_run,
        context="V3-EXQ-995 EXT-005 causal-signature probe",
        escape_hint="P0a must train split_encoder.world_encoder; a frozen random "
                    "projection makes the comparator vacuous (MECH-353 / V3-EXQ-642).",
    )
    print("verdict: PASS", flush=True)
    return stats, guard


def _collect_transitions(agent: REEAgent, seed: int, blocks: int,
                         per_block: int) -> Dict[str, Any]:
    """P1 data collection under a random behaviour policy on a FROZEN encoder.

    Ground-truth labels come from agent + hazard COORDINATES, never from
    info["env_drift_occurred"] (which over-reports -- see module docstring).
    """
    print("Seed %d Condition P1_COLLECT" % seed, flush=True)
    split_encoder = agent.latent_stack.split_encoder
    env = _build_env(seed + 500)
    rng = random.Random(seed + 900)
    _flat, obs = env.reset()

    z_prev_l: List[torch.Tensor] = []
    act_l: List[torch.Tensor] = []
    z_obs_l: List[torch.Tensor] = []
    cls_l: List[str] = []
    drift_l: List[bool] = []
    flag_l: List[bool] = []
    n_flag_disagree = 0
    n_boundary_dropped = 0
    action_counts = [0] * env.action_dim

    for block in range(blocks):
        got = 0
        while got < per_block:
            z_prev = _encode_world(split_encoder, obs)
            ax, ay = env.agent_x, env.agent_y
            hz_before = [tuple(h[:2]) for h in env.hazards]
            action = rng.randrange(env.action_dim)
            _flat, _harm, done, info, obs = env.step(action)
            z_obs = _encode_world(split_encoder, obs)

            agent_moved = (env.agent_x, env.agent_y) != (ax, ay)
            hazards_moved = [tuple(h[:2]) for h in env.hazards] != hz_before
            drift_flag = bool(info.get("env_drift_occurred", False))
            if drift_flag != hazards_moved:
                n_flag_disagree += 1

            if action == STAY_ACTION:
                cls = "STAY"
            elif agent_moved:
                cls = "MOVE_OK"
            else:
                cls = "MOVE_BLOCKED"

            one_hot = torch.zeros(1, env.action_dim)
            one_hot[0, action] = 1.0
            z_prev_l.append(z_prev)
            act_l.append(one_hot)
            z_obs_l.append(z_obs)
            cls_l.append(cls)
            drift_l.append(bool(hazards_moved))
            flag_l.append(drift_flag)
            action_counts[action] += 1
            got += 1

            if done:
                # The reset-boundary transition is NOT a real (z_prev -> z_obs) pair.
                _flat, obs = env.reset()
                n_boundary_dropped += 1

        cur = block + 1
        if cur == 1 or cur % 25 == 0 or cur == blocks:
            print("  [train] ext005 seed=%d phase=P1 ep %d/%d (transition collection)"
                  % (seed, cur, blocks), flush=True)

    return {
        "z_prev": torch.cat(z_prev_l),
        "action": torch.cat(act_l),
        "z_obs": torch.cat(z_obs_l),
        "cls": np.array(cls_l),
        "drift": np.array(drift_l, dtype=bool),
        "flag": np.array(flag_l, dtype=bool),
        "n_flag_disagree": int(n_flag_disagree),
        "n_boundary_resets": int(n_boundary_dropped),
        "action_counts": action_counts,
        "n_transitions": len(cls_l),
    }


# ------------------------------------------------------------------- P1/P2 per arm
def _action_pathway_probe(model: E2WorldForward, z_prev: torch.Tensor,
                          action_dim: int) -> float:
    """mean ||f(z, a_i) - f(z, a_j)|| over a fixed z_world batch and two distinct actions.

    Measured BOTH at init (gating) and after training (diagnostic). Read the pair, never
    the post-training value alone: a head that LEARNS the action carries no information
    drives this toward zero, and that is the `causal_signature_absent_change_detector_only`
    FINDING (reported by C2/C3), not an instrument failure. Only a value that is already
    at the floor AT INIT means the action input never reaches the output.
    """
    n = min(256, z_prev.shape[0])
    a_i = torch.zeros(n, action_dim)
    a_i[:, 0] = 1.0
    a_j = torch.zeros(n, action_dim)
    a_j[:, min(2, action_dim - 1)] = 1.0
    with torch.no_grad():
        return float((model(z_prev[:n], a_i) - model(z_prev[:n], a_j)).norm(dim=-1).mean())


def _train_head(buf: Dict[str, Any], n_train: int, action_dim: int,
                interventional: bool, seed: int) -> Tuple[E2WorldForward, List[Dict[str, Any]], float]:
    model = E2WorldForward(E2WorldConfig(
        use_e2_world_forward=True,
        z_world_dim=WORLD_DIM,
        action_dim=action_dim,
        use_interventional=interventional,
        interventional_fraction=INTERVENTIONAL_FRACTION,
        interventional_margin=INTERVENTIONAL_MARGIN,
    ))
    opt = torch.optim.Adam(model.parameters(), lr=HEAD_LR)
    gen = torch.Generator().manual_seed(int(seed))
    z_prev, action, z_obs = buf["z_prev"], buf["action"], buf["z_obs"]
    curve: List[Dict[str, Any]] = []
    last_loss = float("nan")
    n_interv_batches = 0
    n_interv_engaged = 0
    interv_l2_sum = 0.0
    # BEFORE epoch 0: the wiring measurement, taken while it is still a pure property
    # of the module rather than of what training taught it.
    wiring_at_init = _action_pathway_probe(model, z_prev, action_dim)

    for epoch in range(HEAD_EPOCHS):
        perm = torch.randperm(n_train, generator=gen)
        for i in range(0, n_train, HEAD_BATCH):
            idx = perm[i:i + HEAD_BATCH]
            # P1 stop-gradient discipline: the encoder is already frozen, and the
            # target is detached as well so nothing can leak back into it.
            loss = model.compute_loss(model(z_prev[idx], action[idx]), z_obs[idx].detach())
            if interventional:
                k = max(1, int(INTERVENTIONAL_FRACTION * len(idx)))
                sub = idx[:k]
                # a_cf MUST DIFFER from a_actual -- compute_interventional_loss's own
                # contract (e2_world.py:341 "counterfactual action (must differ)").
                # Drawing it uniformly lets it COINCIDE with a_actual, and a coincident
                # pair has l2_dist == 0, so the hinge reports loss == margin (looks
                # engaged!) while norm-at-zero contributes ZERO gradient. Offsetting by
                # 1..action_dim-1 from the actual index guarantees a genuine alternative.
                a_idx = action[sub].argmax(dim=-1)
                offset = torch.randint(1, action_dim, (k,), generator=gen)
                a_cf = torch.zeros(k, action_dim)
                a_cf[torch.arange(k), (a_idx + offset) % action_dim] = 1.0
                # ENGAGEMENT ACCOUNTING (red-team family 1, caught at authoring): the
                # SD-013 term is a hinge, max(0, margin - ||pred_a - pred_cf||), with
                # margin 0.1 against an ||diff|| that starts near 0.26. If it never goes
                # slack the term is identically zero and INTERV is a SILENT NO-OP -- and
                # "INTERV == BASE" would then read as "the counterfactual constraint does
                # not help" when in truth it never fired. Measure it; do not infer it.
                interv_loss = model.compute_interventional_loss(
                    z_prev[sub].detach(), action[sub], a_cf)
                # ENGAGEMENT means the hinge contributes GRADIENT, i.e. some row has
                # 0 < l2_dist < margin. A nonzero LOSS is NOT sufficient evidence of
                # that (see the a_cf note above), which is exactly how an inert arm
                # can look active. Measure the distance distribution directly.
                with torch.no_grad():
                    d = (model(z_prev[sub].detach(), action[sub])
                         - model(z_prev[sub].detach(), a_cf)).norm(dim=-1)
                    binding = int(((d > 0.0) & (d < INTERVENTIONAL_MARGIN)).sum())
                    interv_l2_sum += float(d.mean())
                n_interv_batches += 1
                if binding > 0:
                    n_interv_engaged += 1
                loss = loss + interv_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        last_loss = float(loss.detach())
        if (epoch + 1) % AUROC_CURVE_EVERY == 0 or epoch == HEAD_EPOCHS - 1:
            curve.append({"epoch": epoch + 1, "train_loss": last_loss,
                          "heldout_auroc_move_ok": _eval_move_ok_auroc(model, buf, n_train)})
    engagement = {
        "n_interventional_batches": n_interv_batches,
        "n_interventional_engaged": n_interv_engaged,
        "interventional_engagement_fraction": (
            (n_interv_engaged / n_interv_batches) if n_interv_batches else None),
        "interventional_margin": INTERVENTIONAL_MARGIN,
        "mean_l2_dist_actual_vs_cf": (
            (interv_l2_sum / n_interv_batches) if n_interv_batches else None),
        "engagement_definition": (
            "a batch counts as ENGAGED only when some row has 0 < l2_dist < margin, "
            "i.e. the hinge actually contributes gradient. A nonzero loss alone does "
            "not qualify: a coincident a_cf gives l2_dist 0 -> loss == margin with "
            "zero gradient."),
    }
    return model, curve, wiring_at_init, engagement


def _eval_move_ok_auroc(model: E2WorldForward, buf: Dict[str, Any],
                        n_train: int) -> Optional[float]:
    n = buf["n_transitions"]
    held = np.zeros(n, dtype=bool)
    held[n_train:] = True
    with torch.no_grad():
        res = model.comparator_residual(
            buf["z_obs"], buf["z_prev"], buf["action"]).norm(dim=-1).numpy()
    sel = held & (buf["cls"] == "MOVE_OK")
    return _auroc(res[sel], buf["drift"][sel])


def _evaluate_arm(model: E2WorldForward, buf: Dict[str, Any], n_train: int,
                  action_dim: int, seed: int) -> Dict[str, Any]:
    n = buf["n_transitions"]
    held = np.zeros(n, dtype=bool)
    held[n_train:] = True
    gen = torch.Generator().manual_seed(int(seed) + 77)

    with torch.no_grad():
        res = model.comparator_residual(
            buf["z_obs"], buf["z_prev"], buf["action"]).norm(dim=-1).numpy()
        # ACTION-PERMUTATION control: marginal action distribution preserved, the
        # (state, action) pairing destroyed.
        perm = torch.randperm(n, generator=gen)
        res_shuf = model.comparator_residual(
            buf["z_obs"], buf["z_prev"], buf["action"][perm]).norm(dim=-1).numpy()
        fwd_mse = float(torch.nn.functional.mse_loss(
            model(buf["z_prev"][held], buf["action"][held]), buf["z_obs"][held]))

    bare = (buf["z_obs"] - buf["z_prev"]).norm(dim=-1).numpy()
    per_stratum: Dict[str, Any] = {}
    for cls in ("STAY", "MOVE_OK", "MOVE_BLOCKED"):
        sel = held & (buf["cls"] == cls)
        n1 = int(buf["drift"][sel].sum())
        n0 = int((~buf["drift"][sel]).sum())
        per_stratum[cls] = {
            "n_drift": n1, "n_no_drift": n0,
            "auroc_comparator": _auroc(res[sel], buf["drift"][sel]),
            "auroc_bare_delta": _auroc(bare[sel], buf["drift"][sel]),
            "auroc_shuffled_action": _auroc(res_shuf[sel], buf["drift"][sel]),
            "mean_residual_drift": float(res[sel][buf["drift"][sel]].mean()) if n1 else None,
            "mean_residual_no_drift": float(res[sel][~buf["drift"][sel]].mean()) if n0 else None,
            "verdict_bearing": cls == "MOVE_OK",
        }
    mo = per_stratum["MOVE_OK"]
    c = mo["auroc_comparator"]
    b = mo["auroc_bare_delta"]
    s = mo["auroc_shuffled_action"]
    return {
        "auroc_move_ok": c,
        "auroc_bare_move_ok": b,
        "auroc_shuffled_move_ok": s,
        "c1_lift_over_chance": (c - 0.5) if c is not None else None,
        "c2_gap_vs_bare": (c - b) if (c is not None and b is not None) else None,
        "c3_gap_vs_shuffled": (c - s) if (c is not None and s is not None) else None,
        "action_pathway_wiring_trained": _action_pathway_probe(
            model, buf["z_prev"], action_dim),
        "heldout_forward_mse": fwd_mse,
        "per_stratum": per_stratum,
    }


def _run_seed(seed: int, dry_run: bool, zg: ZGoalStreamAccumulator) -> Dict[str, Any]:
    env_probe = _build_env(seed)
    action_dim = env_probe.action_dim
    # seeded_construct resets ALL RNG to `seed` BEFORE the agent is built. Without it
    # the encoder's initial weights depend on the process's global torch RNG state at
    # construction time rather than on `seed`, so the per-seed results would not be
    # reproducible (validate_experiments agent_construction_before_seed_lint).
    agent = seeded_construct(seed, lambda: _build_agent(env_probe))

    episodes = 4 if dry_run else EPISODES_PER_PHASE
    p0a_steps = 30 if dry_run else P0A_STEPS_PER_EPISODE
    blocks = 4 if dry_run else EPISODES_PER_PHASE
    per_block = 90 if dry_run else TRANSITIONS_PER_BLOCK

    p0a_stats, guard = _run_p0a(agent, seed, episodes, p0a_steps, dry_run)

    # Encoder frozen for everything downstream (P1 stop-gradient discipline).
    for p in agent.latent_stack.parameters():
        p.requires_grad_(False)

    buf = _collect_transitions(agent, seed, blocks, per_block)
    n = buf["n_transitions"]
    n_train = int((1.0 - HELDOUT_FRACTION) * n)

    arm_rows: List[Dict[str, Any]] = []
    for arm_id, interventional in ARMS:
        with arm_cell(seed, config_slice=_config_slice(seed, arm_id, interventional),
                      script_path=Path(__file__), config_slice_declared=True,
                      extra_ineligible_reasons=[
                          "shared_p0a_encoder_and_transition_buffer_across_arms"]) as cell:
            model, curve, wiring_at_init, engagement = _train_head(
                buf, n_train, action_dim, interventional, seed)
            row: Dict[str, Any] = {
                "arm_id": arm_id, "seed": int(seed), "interventional": interventional,
                "verdict_bearing": arm_id == "BASE",
                "heldout_auroc_curve": curve,
                "action_pathway_wiring_at_init": wiring_at_init,
                "interventional_engagement": engagement,
                # A non-verdict-bearing arm whose manipulation never fired is INERT, not
                # a null result -- recorded so the two are never conflated.
                "manipulation_inert": bool(
                    interventional and engagement["n_interventional_batches"] > 0
                    and engagement["n_interventional_engaged"] == 0),
            }
            row.update(_evaluate_arm(model, buf, n_train, action_dim, seed))
            cell.stamp(row)
        arm_rows.append(row)

    # Records enabled_default_off_flags for drift detection as well as the z_goal
    # counters. This driver never DRIVES the agent (no sense/select_action), so the
    # stream is correctly reported unmeasured rather than measured-zero.
    zg.observe(agent)

    base = next(r for r in arm_rows if r["arm_id"] == "BASE")
    print("verdict: %s" % ("PASS" if (base["auroc_move_ok"] or 0.0) > 0.5 else "FAIL"),
          flush=True)

    return {
        "seed": int(seed),
        "p0a": p0a_stats,
        "zworld_encoder_guard": guard,
        "n_transitions": n,
        "n_train": n_train,
        "n_heldout": n - n_train,
        "n_env_drift_flag_disagreements": buf["n_flag_disagree"],
        "n_episode_boundary_resets": buf["n_boundary_resets"],
        "action_counts": buf["action_counts"],
        "class_counts": {c: int((buf["cls"] == c).sum())
                         for c in ("STAY", "MOVE_OK", "MOVE_BLOCKED")},
        "drift_fraction": float(buf["drift"].mean()),
        "arms": arm_rows,
    }


# ------------------------------------------------------------------- adjudication
def _preconditions(seed_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Readiness-kind preconditions. All GATING: below-floor self-routes to
    substrate_not_ready_requeue, never to a claim verdict."""
    out: List[Dict[str, Any]] = []

    # (1) SD-070 P0a actually moved the world encoder. Worst cell across seeds.
    worst = min(seed_rows,
                key=lambda r: r["zworld_encoder_guard"].get("world_encoder_max_abs_delta", 0.0))
    entry = zworld_precondition(worst["zworld_encoder_guard"],
                                context="V3-EXQ-995 worst seed %d" % worst["seed"])
    entry["offending_cell"] = "seed=%d" % worst["seed"]
    out.append(entry)

    # (2) POSITIVE CONTROL: z_world represents exogenous hazard change at all. Read in
    #     the STAY stratum, where the agent does not displace so the hazard is the only
    #     possible source of change. Same statistic as the verdict criterion (an AUROC
    #     over the same labels), measured where the answer is known.
    v, k = _worst_low([(r["arms"][0]["per_stratum"]["STAY"]["auroc_bare_delta"],
                        "seed=%d" % r["seed"]) for r in seed_rows])
    out.append({
        "name": "zworld_encodes_exogenous_change",
        "kind": "readiness",
        "description": "bare ||z_obs - z_prev|| AUROC for hazard-moved vs not, in the "
                       "STAY stratum (agent does not displace)",
        "control": "positive control -- the agent cannot have caused the change, so a "
                   "sub-floor reading means z_world does not represent hazard position "
                   "and NO comparator could be tested on it",
        "measured": v, "threshold": FLOOR_STAY_BARE_AUROC, "direction": "lower",
        "offending_cell": k, "met": (v is not None and v >= FLOOR_STAY_BARE_AUROC),
    })

    # (3) NEGATIVE CONTROL: no raw-magnitude confound in the verdict stratum. Two-sided,
    #     so it is declared as an interval -- a single bound would leave the other leg
    #     absent from the manifest and silently recompute as met (V3-EXQ-779b).
    v, k = _worst_interval([(r["arms"][0]["per_stratum"]["MOVE_OK"]["auroc_bare_delta"],
                             "seed=%d" % r["seed"]) for r in seed_rows], centre=0.5)
    out.append({
        "name": "move_ok_confound_absent",
        "kind": "readiness",
        "description": "bare ||z_obs - z_prev|| AUROC in MOVE_OK must sit near chance; "
                       "outside the band the verdict criteria are confounded by raw "
                       "change magnitude and the design is invalid, not falsified",
        "control": "negative control -- agent displacement dominates the change "
                   "magnitude here, so the raw signal should carry no drift information",
        "measured": v, "threshold_low": BAND_MOVE_OK_BARE_LOW,
        "threshold_high": BAND_MOVE_OK_BARE_HIGH, "direction": "interval",
        "offending_cell": k,
        "met": (v is not None and BAND_MOVE_OK_BARE_LOW <= v <= BAND_MOVE_OK_BARE_HIGH),
    })

    # (4) The failure 783 hit: enough events of BOTH causal classes. Count-gated
    #     criterion -> count readiness (same statistic).
    v, k = _worst_low([(float(min(r["arms"][0]["per_stratum"]["MOVE_OK"]["n_drift"],
                                  r["arms"][0]["per_stratum"]["MOVE_OK"]["n_no_drift"])),
                        "seed=%d" % r["seed"]) for r in seed_rows])
    out.append({
        "name": "causal_class_events_sufficient",
        "kind": "readiness",
        "description": "min(n_drift, n_no_drift) in the held-out MOVE_OK stratum",
        "control": "the exact precondition every prior attempt failed -- V3-EXQ-783 "
                   "recorded attribution_gap null on all 16 attribution_ready arms with "
                   "'insufficient events of one or both causal classes'",
        "measured": v, "threshold": float(FLOOR_MOVE_OK_CLASS_N), "direction": "lower",
        "offending_cell": k, "met": (v is not None and v >= FLOOR_MOVE_OK_CLASS_N),
    })

    # (5) The action pathway is WIRED (not: has learned the action matters). Worst cell
    #     across every seed x arm. Strict floor: a bit-zero reading is the failure.
    v, k = _worst_low([(a["action_pathway_wiring_at_init"],
                        "seed=%d arm=%s" % (r["seed"], a["arm_id"]))
                       for r in seed_rows for a in r["arms"]])
    out.append({
        "name": "e2world_action_pathway_live",
        "kind": "readiness",
        "description": "mean ||f(z, a_i) - f(z, a_j)|| AT INITIALISATION, over a fixed "
                       "z_world batch and two distinct actions (the post-training value "
                       "is recorded per arm as action_pathway_wiring_trained but does "
                       "NOT gate)",
        "control": "wiring check, measured before any training so it is a property of "
                   "the module and not of what training taught it. This is what "
                   "separates 'the action input never reaches the output' (instrument "
                   "not ready) from 'the model LEARNED the action carries little "
                   "information' -- the latter is the causal_signature_absent finding "
                   "C2/C3 report, and gating on a post-training probe would have "
                   "misrouted exactly that finding to substrate_not_ready_requeue",
        "measured": v, "threshold": FLOOR_ACTION_PATHWAY, "comparator": ">",
        "direction": "lower", "offending_cell": k,
        "met": (v is not None and v > FLOOR_ACTION_PATHWAY),
    })
    return out


def _adjudicate(seed_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    preconditions = _preconditions(seed_rows)
    gate_green = all(bool(p.get("met")) for p in preconditions)

    base_rows = [next(a for a in r["arms"] if a["arm_id"] == "BASE") for r in seed_rows]

    c1_vals = [a["auroc_move_ok"] for a in base_rows]
    c2_vals = [a["c2_gap_vs_bare"] for a in base_rows]
    c3_vals = [a["c3_gap_vs_shuffled"] for a in base_rows]
    c1_mean, c1_sd = _mean_sd(c1_vals)
    c2_mean, c2_sd = _mean_sd(c2_vals)
    c3_mean, c3_sd = _mean_sd(c3_vals)

    c1_req = max(THRESH_C1_ABS_FLOOR, THRESH_C1_SD_MULT * c1_sd)
    c2_req = max(THRESH_C2_ABS_FLOOR, THRESH_C2_SD_MULT * c2_sd)
    c1_pass = bool(gate_green and (c1_mean - 0.5) >= c1_req)
    c2_pass = bool(gate_green and c2_mean >= c2_req)
    c3_pass = bool(gate_green and c3_mean >= THRESH_C3_MARGIN)

    n_ok = sum(1 for a in base_rows
               if a["auroc_move_ok"] is not None and a["c2_gap_vs_bare"] is not None)
    criteria = [
        {"name": "C1_comparator_discriminates_move_ok", "load_bearing": True,
         "passed": c1_pass, "mean": c1_mean, "sd": c1_sd, "requirement": c1_req,
         "statistic": "mean(auroc_move_ok) - 0.5 >= max(%.3f, %.1f*sd)"
                      % (THRESH_C1_ABS_FLOOR, THRESH_C1_SD_MULT)},
        {"name": "C2_beats_bare_change_magnitude", "load_bearing": True,
         "passed": c2_pass, "mean": c2_mean, "sd": c2_sd, "requirement": c2_req,
         "statistic": "mean(auroc_move_ok - auroc_bare_move_ok) >= max(%.3f, %.1f*sd)"
                      % (THRESH_C2_ABS_FLOOR, THRESH_C2_SD_MULT)},
        {"name": "C3_discrimination_is_action_conditioned", "load_bearing": False,
         "passed": c3_pass, "mean": c3_mean, "sd": c3_sd,
         "requirement": THRESH_C3_MARGIN,
         "statistic": "mean(auroc_move_ok - auroc_shuffled_move_ok) >= %.3f"
                      % THRESH_C3_MARGIN},
    ]

    non_degenerate = {
        "C1_comparator_discriminates_move_ok": bool(n_ok >= 2 and c1_sd > 0.0),
        "C2_beats_bare_change_magnitude": bool(
            n_ok >= 2 and any(v is not None and abs(v) > 1e-9 for v in c2_vals)),
        "C3_discrimination_is_action_conditioned": bool(
            n_ok >= 2 and any(v is not None and abs(v) > 1e-9 for v in c3_vals)),
    }

    if not gate_green:
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
        overall = False
    else:
        overall = c1_pass and c2_pass
        if overall:
            # C3 is not verdict-bearing (it cannot flip PASS to FAIL), but it MUST be
            # able to qualify the label: C2 passing says the forward model beats raw
            # change magnitude, while C3 failing says it is not the ACTION that supplies
            # the advantage. That combination is state-conditioned predictability, not
            # the agency signal comparator_residual claims (e2_world.py:285-296), and an
            # unqualified "causal_signature_present" would overstate it downstream.
            label = ("causal_signature_present" if c3_pass
                     else "causal_signature_present_action_conditioning_unconfirmed")
        elif c1_pass and not c2_pass:
            label = "causal_signature_absent_change_detector_only"
        else:
            label = "causal_signature_not_demonstrated"
        direction = "supports" if overall else "weakens"

    return {
        "outcome": "PASS" if overall else "FAIL",
        "evidence_direction": direction,
        "criteria": criteria,
        "claim_scope": {
            "claim_id": "EXT-005",
            "registered_subject": "llm.causal_attribution",
            "what_this_run_measures": (
                "the REE-side remedy EXT-005 names in its ree_mechanism/notes -- "
                "specifically SD-031's single-pass comparator (SD-003's two-pass "
                "counterfactual form, which the claim notes actually cite, is "
                "SUPERSEDED). It does NOT and cannot test the registered subject, "
                "llm.causal_attribution: no V3 run observes an LLM."),
            "attribution_caveat": (
                "READ THE DIRECTION AS SCOPED TO THE REMEDY, NOT THE ASSERTION. A PASS "
                "shows REE computes a causal signature of the kind EXT-005 says LLMs "
                "lack; it is not independent evidence that LLMs lack one. A FAIL shows "
                "REE's comparator is also only a change detector -- which does NOT "
                "weaken the assertion that LLMs lack the mechanism (if anything it is "
                "consistent with the gap being general); what it weakens is the claim's "
                "remedy annotation. Governance should weight this run against the "
                "remedy clause, and should not read a FAIL as evidence against the "
                "LLM-side assertion."),
            "raised_by": "Step 4.5 adversarial red-team (fable), CONTESTED finding 3",
        },
        "requeue_semantics": (
            "the substrate_not_ready_requeue label is the skill-mandated self-route for "
            "a below-floor readiness precondition. It is a SIGNAL for governance "
            "(pending_review -> /failure-autopsy -> requeue under a new letter), NOT an "
            "automated requeue: a FAIL outcome removes the item from the queue, as it "
            "does for every experiment."),
        "combination_rule": (
            "overall PASS = C1 AND C2, both load-bearing, both read from the BASE arm "
            "in the MOVE_OK stratum only. C3 is supporting and NOT verdict-bearing -- "
            "it cannot flip the outcome, but a failed C3 DOES qualify the label to "
            "causal_signature_present_action_conditioning_unconfirmed. "
            "All criteria are additionally gated on every readiness precondition being "
            "met; a failed precondition self-routes to substrate_not_ready_requeue "
            "(non_contributory), never to a claim verdict."),
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": non_degenerate,
            "gate_green": gate_green,
        },
        "summary": {
            "base_auroc_move_ok_mean": c1_mean, "base_auroc_move_ok_sd": c1_sd,
            "base_c2_gap_vs_bare_mean": c2_mean, "base_c2_gap_vs_bare_sd": c2_sd,
            "base_c3_gap_vs_shuffled_mean": c3_mean, "base_c3_gap_vs_shuffled_sd": c3_sd,
            "n_seeds_scored": n_ok,
        },
        "degeneracy": check_degeneracy({
            "auroc_move_ok_base": [v for v in c1_vals if v is not None],
            "c2_gap_vs_bare_base": [v for v in c2_vals if v is not None],
        }),
    }


# --------------------------------------------------------------------------- main
def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = SEEDS[:2] if dry_run else SEEDS
    zg = ZGoalStreamAccumulator()
    seed_rows = [_run_seed(s, dry_run, zg) for s in seeds]
    adj = _adjudicate(seed_rows)

    run_id = "%s_%s_v3" % (EXPERIMENT_TYPE, datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": adj["outcome"],
        "evidence_direction": adj["evidence_direction"],
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "criteria": adj["criteria"],
        "claim_scope": adj["claim_scope"],
        "requeue_semantics": adj["requeue_semantics"],
        "combination_rule": adj["combination_rule"],
        "interpretation": adj["interpretation"],
        "summary": adj["summary"],
        "per_seed_results": seed_rows,
        "arm_results": [a for r in seed_rows for a in r["arms"]],
        "pre_registered_thresholds": {
            "C1_abs_floor": THRESH_C1_ABS_FLOOR, "C1_sd_mult": THRESH_C1_SD_MULT,
            "C2_abs_floor": THRESH_C2_ABS_FLOOR, "C2_sd_mult": THRESH_C2_SD_MULT,
            "C3_margin": THRESH_C3_MARGIN,
            "floor_stay_bare_auroc": FLOOR_STAY_BARE_AUROC,
            "band_move_ok_bare": [BAND_MOVE_OK_BARE_LOW, BAND_MOVE_OK_BARE_HIGH],
            "floor_move_ok_class_n": FLOOR_MOVE_OK_CLASS_N,
            "floor_action_pathway": FLOOR_ACTION_PATHWAY,
        },
        "authoring_pilot": PILOT_RECORD,
        "label_provenance": (
            "exogenous-change label taken from hazard COORDINATES before/after each "
            "step; info['env_drift_occurred'] is recorded for audit only because it is "
            "set on every drift TICK regardless of whether a hazard actually moved "
            "(causal_grid_world.py:3063-3065 discards _drift_hazards' own result)"),
        "known_substrate_limitations": [
            "SD-018 (degrading): overlaps latent/stack.py + latent/zworld_p0.py; both "
            "resource heads absent here (use_resource_proximity_head=False, "
            "use_resource_field_head=False), so neither SD-018 leg runs",
            "mech357-freeze-incompatible-pressure-mechanism (degrading): overlaps "
            "causal_grid_world.py; reef_enabled False, hazard_agent_pursuit 0.0",
            "SD-MECH303-THRESHOLD-SOURCING (degrading): overlaps causal_grid_world.py "
            "and utils/config.py",
        ],
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
        "dry_run": bool(dry_run),
    }
    manifest.update(adj["degeneracy"])

    stamp_recording_core(
        manifest,
        config={"env_kwargs": dict(ENV_KWARGS), "world_dim": WORLD_DIM,
                "alpha_world": ALPHA_WORLD, "episodes_per_phase": EPISODES_PER_PHASE,
                "p0a_steps_per_episode": P0A_STEPS_PER_EPISODE,
                "n_transitions": N_TRANSITIONS, "heldout_fraction": HELDOUT_FRACTION,
                "head_epochs": HEAD_EPOCHS, "head_batch": HEAD_BATCH, "head_lr": HEAD_LR,
                "interventional_fraction": INTERVENTIONAL_FRACTION,
                "interventional_margin": INTERVENTIONAL_MARGIN, "dry_run": bool(dry_run)},
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=zg.stats(),
    )
    return manifest


def main() -> Tuple[Dict[str, Any], Path, bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="tiny smoke run (2 seeds, 4 episodes/blocks)")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    manifest = run_experiment(dry_run=args.dry_run)
    # out_dir omitted on purpose -> resolve_evidence_experiments_dir(script_path),
    # which is worktree-aware. stamp=False: run_experiment already called
    # stamp_recording_core AFTER arm_results was assembled, so substrate_hash hoists
    # from the per-cell fingerprints instead of being recomputed.
    out_path = write_flat_manifest(manifest, dry_run=args.dry_run,
                                   script_path=Path(__file__), stamp=False)

    print("Outcome: %s" % manifest["outcome"], flush=True)
    print("label:   %s" % manifest["interpretation"]["label"], flush=True)
    for c in manifest["criteria"]:
        print("  %-42s load_bearing=%-5s passed=%-5s mean=%s"
              % (c["name"], c["load_bearing"], c["passed"],
                 ("%.4f" % c["mean"]) if c["mean"] == c["mean"] else "nan"), flush=True)
    for p in manifest["interpretation"]["preconditions"]:
        print("  precondition %-34s met=%-5s measured=%s"
              % (p["name"], p.get("met"), p.get("measured")), flush=True)
    print("Result pack written to: %s" % out_path, flush=True)
    return manifest, out_path, bool(args.dry_run)


if __name__ == "__main__":
    _manifest, _out_path, _dry = main()
    _outcome = str(_manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        run_id=_manifest["run_id"],
        queue_id=QUEUE_ID,
        dry_run=_dry,
    )
