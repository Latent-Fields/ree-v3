#!/opt/local/bin/python3
"""
V3-EXQ-1063 -- INV-063 leg B: which frozen-battery readout moves in the direction
the claim asserts, and at what base convergence?

SLEEP DRIVER: manual-cycle-loop (run_sleep_cycle() called once per cycle in a
              dedicated N_CYCLES wake-sleep-test loop)

WHY THIS RUN (user decision, 2026-09-19T09:25:51Z, option B)
-------------------------------------------------------------
Authoring of INV-063's four-arm intake-ladder falsifier was REFUSED at
/queue-experiment Step 2.5a. The refusal and its measurements are staged at
REE_assembly/evidence/planning/inv063_legb_dv_readability_staged_20260919.md
(REE_assembly e66187df0a). In one line: on a CONVERGED base a sleep cycle made the
frozen held-out world-forward MSE WORSE in 9 of 9 probe cells, so leg B's
across-sleep IMPROVEMENT DV was negative before any intake manipulation was applied.

The user's decision was to run THIS diagnostic first and to take the four-arm
falsifier's go/no-go back to the user with its numbers. Verbatim scope:

  "lever ON/OFF x converged/unconverged base x >= 3 seeds, recording BOTH
   frozen-battery readouts (MSE and InfoNCE) across sleep; EXPERIMENT_PURPOSE
   diagnostic; no claim verdict. Its job is to establish which readout moves in
   the direction INV-063 asserts and at what base convergence."

That is exactly what this script does and all it does.

WHAT INV-063 ASSERTS, AND WHAT "DIRECTION" MEANS HERE
------------------------------------------------------
INV-063 C1 leg B reads "the ACROSS-SLEEP IMPROVEMENT in world-forward prediction
error, measured on a FROZEN held-out battery with the V3-EXQ-701b/701c frozen-probe
instrument (pre-sleep PE minus post-sleep PE on the same frozen battery, so the DV
is what sleep ADDED, not how hard the waking period was)". The asserted direction
is therefore POSITIVE: sleep reduces held-out world-forward prediction error.
delta = pre - post, and "moves in the direction INV-063 asserts" means delta > 0.

THE MECHANISM UNDER TEST, which is an INSTRUMENT question, not a claim question
-------------------------------------------------------------------------------
V3-EXQ-1060's own docstring states it verbatim:

  "compute_e2_world_loss minimises the SD-056 InfoNCE CONTRASTIVE loss, while the
   701b frozen-probe DV is per-element MSE RECONSTRUCTION error. InfoNCE is
   insensitive to a global scale/shift of the prediction, so it can improve while
   frozen-battery MSE does not move."

1060 measured that gap on an UNCONVERGED head, where both fall together (MSE rel
improvement 0.2504, InfoNCE rel improvement 0.0023 -- already a 100x divergence, in
favour of the readout an untrained head improves for free). Nobody had measured it
PAST convergence, where the two objectives no longer share a descent direction. That
is the gap this run closes, and it is why BOTH readouts are recorded on the SAME
frozen battery rather than one being picked at authoring time.

FOUR ARMS (2 levers x 2 base regimes), seed-matched, >= 3 seeds
----------------------------------------------------------------
  ARM_CONV_ON   base CONVERGED   lever use_sleep_world_forward_consolidation=True
  ARM_CONV_OFF  base CONVERGED   lever False
  ARM_FRESH_ON  base UNCONVERGED lever True
  ARM_FRESH_OFF base UNCONVERGED lever False

THE TWO BASE REGIMES ARE MATCHED ON EVERYTHING EXCEPT THE OPTIMISER. Both run the
IDENTICAL P0 rollout -- same env, same seed, same P0_STEPS, same random-action
policy, same transition buffering. The CONVERGED arms additionally take one Adam
step per rollout step on agent.e2.parameters(); the UNCONVERGED arms take none.
This is a deliberate improvement on the staged probe, whose fresh cells had ~7x
less env exposure than its converged cells and so differed in buffer content and
state distribution as well as in convergence. Here the ONLY difference is whether
world_forward was trained before the measurement phase.

Both arms of a lever pair are bit-identical up to the cycle itself: the OFF arm is
STRUCTURALLY ABSENT from the interleaved schedule -- no key, no closure, no RNG draw
(phase_manager.py:700-708) -- which is what makes C4 a sharp attribution control
rather than a "nothing ran" tautology (V3-EXQ-1026's OFF arm disabled consolidation
entirely; this one keeps the full MECH-423 pass running in every arm).

THE FROZEN BATTERY IS CAPTURED ONCE, BEFORE P0, AND IS THE SAME IN ALL FOUR ARMS
--------------------------------------------------------------------------------
701b's / 1060's _sample_probe_battery form: a HELD-OUT env instance (seed + 9973), a
FIXED action policy independent of training, pure read (it never calls _e1_tick, so
it appends nothing to the replay buffers the trainer draws from). It is captured with
the agent's OWN encoder, and NOTHING in this script ever trains the encoder -- P0's
optimiser is scoped to agent.e2.parameters() and the sleep pass builds LOCAL per-module
Adam optimizers over the parameter lists it is handed. So z0/z1 live in one fixed
latent space for the whole cell and are identical across arms at a given seed, which
is what makes the four arms' readouts directly comparable rather than merely parallel.
Capturing it BEFORE P0 additionally makes conv_rel_drop a measurement on the SAME
battery the DV is later read on.

WHAT ROUTES THE VERDICT, AND WHAT DOES NOT -- read this before reading the manifest
-----------------------------------------------------------------------------------
This is a DIAGNOSTIC whose job is to MEASURE a direction, not to assert one. So:

  PASS/FAIL turns ONLY on VALIDITY -- C1..C6, plain AND. They ask whether the run
  measured what it set out to measure: were the two base regimes actually what they
  are labelled, did the lever move the world heads, did the negative control hold,
  were the readouts populated and finite, did sleep do work in every cell.

  THE DIRECTION FINDINGS D1..D4 ROUTE NO PASS/FAIL. They are pre-registered sign
  tests, they pick interpretation.label, and they are the run's actual output. A
  NEGATIVE D1 is a RESULT, not a failure -- it is, on the staged probe's evidence,
  the likeliest outcome, and reading it as a FAIL would be exactly the confusion
  this split exists to prevent.

If any validity criterion or any readiness precondition is unmet the run self-routes
to substrate_not_ready_requeue and NO direction label is emitted -- a direction read
off an invalid measurement is worth less than no reading.

NO CLAIM VERDICT. claim_ids=["INV-063"] is for traceability; experiment_purpose is
"diagnostic" (excluded from governance confidence/conflict scoring) and
evidence_direction is "non_contributory". This run adjudicates an INSTRUMENT. The
four-arm falsifier's go/no-go returns to the USER with these numbers; options C
(re-point leg B's DV to InfoNCE) and D (convert INV-063 to substrate_conditional)
remain open and are /governance's, not this script's.

DV-SYMMETRY / per-arm declaration (mandatory)
----------------------------------------------
All four arms share one DV family: a SIGNED DIFFERENCE of two scalar readouts
computed on a FIXED, pre-captured battery before and after one sleep cycle. It is
not an argmax/rank statistic (so the monotone-rescaling class cannot apply), and not
a set-aggregate over interchangeable units (so the permutation class cannot apply).

  ARM_CONV_ON / ARM_FRESH_ON -- the manipulation is a BINARY CODE-PATH GATE (whether
    a third loss closure and optimiser group exist at all), not a value transform of
    the DV, so no symmetry of the DV is invariant under it.
  ARM_CONV_OFF / ARM_FRESH_OFF -- same gate, negative side. Their DV is EXACTLY 0 by
    structural identity (see below), which is the CONTROL, not a measured null.
  The CONV-vs-FRESH contrast is a TRAINING-BUDGET difference in the base (P0 Adam
    steps 3600 vs 0). MSE is a squared error and is NOT invariant to any additive or
    multiplicative transform of the prediction, so the budget cannot cancel in it.

  DECLARED, because it is the point of the run rather than a defect to hide:
  the InfoNCE readout IS invariant to a global scale/shift of the prediction
  (SD-056 contrastive). A manipulation that only rescaled or shifted predictions
  would be invisible to D2/D4 while visible to D1/D3. The manipulation here is a
  code-path gate rather than a value transform, so it is not such a manipulation --
  but the asymmetry is exactly why both readouts are carried, and any reader
  comparing D1 against D2 must hold it in mind.

OFF-ARM ZERO IS A STRUCTURAL IDENTITY, NOT A MEASURED NULL
-----------------------------------------------------------
world_forward depends only on the two world heads (e2_fast.py:201-221); the OFF arm
applies no gradient to them; the SAME captured battery tensors are re-evaluated. So
pre and post are bitwise identical and every OFF delta is exactly 0.0. Do NOT read it
as a measured null effect, and do NOT use it as a difference-in-differences baseline
against which the ON arm's change is a "difference". Its job is attribution: it proves
the ON arm's movement is the lever's and not another offline writer's, because the
SAME MECH-423 consolidation pass runs in both.

WHAT THIS RUN CANNOT SETTLE
----------------------------
- It says nothing about whether the DV ORDERS WITH INTAKE. No intake ladder is run
  here (world_rule_shift is disabled in every arm). That is the four-arm falsifier's
  question and it is deliberately not posed.
- It says nothing about P1's MEL-ladder monotonicity on a converged base;
  V3-EXQ-798a's landed 3/3 stands and is not re-litigated here.
- "At what base convergence" is answered at the resolution this design affords: two
  levels, plus the per-cell P0 convergence TRAJECTORY recorded as ungated telemetry
  (battery MSE and InfoNCE at 0 / 900 / 1800 / 2700 / 3600 P0 steps), which is what
  lets a later reader locate a crossover between them without another run.

PRIOR-ART POINTERS the reader will want
-----------------------------------------
- V3-EXQ-1060 (PASS, 2026-09-19): the lever moves E2's world heads at all.
- V3-EXQ-798a: the SD-MEL-PRODUCER ladder on a converged base; also the origin of the
  recon-only P0 form used here (_e2_train_step, :604-626) and of MIN_REL_CONV_DROP.
- V3-EXQ-701b/701c: the frozen-probe instrument INV-063 names by name.
- GFLAG-0359: three drifted pointers in INV-063's own what_would_answer, including
  that P2's cumulative_sws_writes / cumulative_rem_rollouts do not exist -- the real
  merged-cycle keys are sws_n_writes / rem_n_rollouts, which C6 asserts here.
- GFLAG-0355: leg B's E1 -> E2 label correction, open with /governance. This script
  takes leg B to be E2.world_forward throughout, per 1060's documented chain.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_1063_inv063_legb_dv_direction.py --dry-run
  /opt/local/bin/python3 experiments/v3_exq_1063_inv063_legb_dv_direction.py
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments.pack_writer import write_flat_manifest

EXPERIMENT_TYPE = "v3_exq_1063_inv063_legb_dv_direction"
QUEUE_ID = "V3-EXQ-1063"
CLAIM_IDS: List[str] = ["INV-063"]
EXPERIMENT_PURPOSE = "diagnostic"

# -- arms --------------------------------------------------------------------
ARM_CONV_ON = "ARM_CONV_ON"
ARM_CONV_OFF = "ARM_CONV_OFF"
ARM_FRESH_ON = "ARM_FRESH_ON"
ARM_FRESH_OFF = "ARM_FRESH_OFF"
ARMS: Tuple[str, ...] = (ARM_CONV_ON, ARM_CONV_OFF, ARM_FRESH_ON, ARM_FRESH_OFF)
ARM_SPEC: Dict[str, Dict[str, Any]] = {
    ARM_CONV_ON: {"base": "converged", "lever": True},
    ARM_CONV_OFF: {"base": "converged", "lever": False},
    ARM_FRESH_ON: {"base": "unconverged", "lever": True},
    ARM_FRESH_OFF: {"base": "unconverged", "lever": False},
}
ON_ARMS = (ARM_CONV_ON, ARM_FRESH_ON)
OFF_ARMS = (ARM_CONV_OFF, ARM_FRESH_OFF)

# Seed 44 is deliberately absent (recurring reef-config early-death instability,
# EXQ-539/540, V3-EXQ-538a). This is V3-EXQ-798a's and V3-EXQ-1060's seed set.
SEEDS: Tuple[int, ...] = (42, 123, 456)

# -- substrate constants, carried from V3-EXQ-1060 (:258-262) ----------------
GRID_SIZE = 5
N_HAZARDS = 1
N_RESOURCES = 1
SELF_DIM = 16
WORLD_DIM = 16
STEPS_PER_EPISODE = 90          # V3-EXQ-798a's step budget

# -- P0: recon-only world-forward convergence, V3-EXQ-798a's form ------------
# 798a used CONV_EPISODES=60 x 90 = 5400 steps and reached conv_rel_drop ~0.99.
# 3600 reached 0.9967-0.9992 in the staged probe, i.e. the plateau is already
# well inside this budget; the shorter budget is chosen so the FRESH arms (which
# pay the same rollout cost with no optimiser) stay cheap.
P0_STEPS = 3600
P0_EPISODE_EQUIV = P0_STEPS // STEPS_PER_EPISODE          # 40
P0_TRAJECTORY_AT = (0, 30, 90, 180, 360, 720, 1440, 2400, 3600)  # ungated telemetry
E2_LR = 1e-3
BATCH_K = 8
BUF_MAX = 256
MIN_BUF_BEFORE_TRAIN = 16
MAX_GRAD_NORM = 1.0

# -- P1: the measurement phase ------------------------------------------------
N_CYCLES = 3
WAKE_EPS_PER_CYCLE = 2
# The [train] denominator M, and the queue entry's episodes_per_run. One number
# for every arm BECAUSE both base regimes run the identical rollout length.
EPISODES_PER_RUN = P0_EPISODE_EQUIV + N_CYCLES * WAKE_EPS_PER_CYCLE   # 46

CMC_STEPS = 8
CMC_LR = 1e-3
CMC_BATCH = 16

# Frozen held-out probe battery (701b:178 PROBE_BATTERY_SIZE = 64).
BATTERY_SIZE = 64
# Floor for a readable per-element MSE. 1060 used 2.0 (its criteria were parameter
# deltas, not battery statistics); this run READS the battery, so the floor is
# raised to half the requested size.
BATTERY_FLOOR = 32.0

# n_pairs = min(len(wbuf), len(abuf)) - 1 must clear CMC_BATCH for the trainer to
# draw a full batch. heartbeat.e1_steps_per_tick == 1 in this config, so the buffers
# grow 1:1 with P1 env steps: N_CYCLES(3) x WAKE_EPS(2) x STEPS(90) ~= 540 entries,
# which clears the floor >30x. Reachable by construction, not a tuned definition.
WORLD_BUFFER_FLOOR = float(CMC_BATCH + 1)
# The SD-056 rollout clamp is deliberately NOT enabled. Three reasons, in order of
# weight:
# (1) REACH. The lint's named failure mode is an UNBOUNDED IMAGINATION ROLLOUT
#     diverging to 1e16-1e18 (V3-EXQ-569e, 936). This driver runs no imagination
#     rollout at all: world_forward_contrastive_loss is called ONLY as a read-only
#     readout inside torch.no_grad() on a fixed battery, and the only contrastive
#     TRAINING is the sleep pass's 8 steps at lr 1e-3 with grads scoped to the two
#     world heads. Divergence is additionally DETECTED, not assumed away -- every
#     e1/e2 parameter is asserted finite after every cycle and a non-finite cell
#     routes the run to substrate_not_ready_requeue naming DIVERGENCE explicitly.
# (2) COMPARABILITY. This run's whole purpose is to extend V3-EXQ-1060's
#     frozen-battery numbers PAST convergence. 1060 declined the same clamp for the
#     same reason; enabling it here would change E2's rollout behaviour relative to
#     the only run these readouts can be compared against.
# (3) SCOPE. An unratified substrate-behaviour change inside a diagnostic whose job
#     is to measure an existing instrument is exactly the kind of drift that makes a
#     later reader unable to attribute a difference.
SD056_ROLLOUT_CLAMP_EXEMPT = (
    "No imagination rollout is run: world_forward_contrastive_loss is a read-only "
    "no_grad readout on a fixed battery, and the only contrastive training is the "
    "sleep pass's 8 steps at lr 1e-3 scoped to the two world heads. Divergence is "
    "detected (all e1/e2 params asserted finite after every cycle) rather than "
    "assumed absent. Enabling the clamp would also break comparability with "
    "V3-EXQ-1060, whose readouts this run extends past convergence and which "
    "declined the same clamp."
)

ANCHOR_REACHABILITY_EXEMPT = (
    "world_experience_buffer floor (17) is >30x cleared by the fixed P1 rollout "
    "(~540 entries/cell at e1_steps_per_tick=1); not a hand-tuned degeneracy "
    "definition."
)

# -- pre-registered thresholds (constants; NOT derived from run statistics) ---
MIN_CONV_REL_DROP = 0.90        # C1: a CONVERGED arm must have converged
MAX_UNCONV_REL_DROP = 0.02      # C2: an UNCONVERGED arm must not have
SEEDS_REQUIRED_FOR_DIRECTION = 2  # D1..D4: >= 2 of 3 seeds
EPS = 1e-12

WORLD_HEAD_MODULES = ("world_transition", "world_action_encoder")

ETHICS_PREFLIGHT = {
    "involves_negative_valence": False,
    "involves_suffering_like_state": False,
    "involves_self_model": False,
    "involves_inescapability_or_helplessness": False,
    "involves_offline_replay_over_harm": False,
    "involves_social_mind_or_language": False,
    "involves_human_data_or_clinical_context": False,
    "decision": "allow",
}


# ---------------------------------------------------------------------------
# preconditions -- regime-conditioned (experiments/_lib/precondition_gate.py).
# Every one declares the regimes it is meaningful for; NO cell's gate may vacate
# another cell's (failure_autopsy_V3-EXQ-785_2026-07-19 sections 2a/8).
# ---------------------------------------------------------------------------
def _is_converged(ctx: Dict[str, Any]) -> bool:
    return ctx["base"] == "converged"


def _is_unconverged(ctx: Dict[str, Any]) -> bool:
    return ctx["base"] == "unconverged"


def _is_lever_on(ctx: Dict[str, Any]) -> bool:
    return bool(ctx["lever"])


PRECONDITIONS: Tuple[PreconditionSpec, ...] = (
    PreconditionSpec(
        name="world_replay_pairs_populated",
        description="replay pairs available to the world trainer at cycle time",
        control="real P1 waking rollout via _e1_tick, not synthetic buffers",
        threshold=WORLD_BUFFER_FLOOR, direction="lower"),
    PreconditionSpec(
        name="world_loss_non_sentinel",
        description=("compute_e2_world_loss returns an exactly-zero graph-anchored "
                     "sentinel when n_pairs < 2, and consolidate()'s contract says "
                     "such a step does NOT count as touching the module"),
        control="probed on the same real buffers the sleep cycle draws from",
        threshold=0.0, direction="lower"),
    PreconditionSpec(
        name="world_grad_nonzero",
        description="a non-zero gradient reaches the two world heads",
        control=("backward() from the realised world loss; run in BOTH lever arms "
                 "so the arms stay RNG-matched entering the cycle"),
        threshold=0.0, direction="lower"),
    PreconditionSpec(
        name="action_buffer_non_vacuous",
        description=("fraction of replay action entries that are not all-zero; "
                     "world_action_encoder is nn.Linear so a zero input gives "
                     "dL/dW = 0 EXACTLY and the channel is untrainable"),
        control=("agent.e1_action_buffer_stats(), the substrate's own non-vacuity "
                 "detector for this failure mode (V3-EXQ-1060 red-team F1)"),
        threshold=0.5, direction="lower"),
    PreconditionSpec(
        name="frozen_battery_populated",
        description="held-out one-step transitions backing the per-element MSE",
        control="held-out env (seed+9973) + fixed action policy, pure read",
        threshold=BATTERY_FLOOR, direction="lower",
        structural_max=lambda ctx: float(BATTERY_SIZE)),
    PreconditionSpec(
        name="sleep_cycles_fired",
        description=("cycles whose merged metrics carry post_sleep_z_goal_retention, "
                     "which _run_cycle merges unconditionally at the end of every "
                     "completed cycle"),
        control="a key no internal gate can suppress on a completed cycle",
        threshold=0.5, direction="lower",
        structural_max=lambda ctx: float(N_CYCLES)),
    PreconditionSpec(
        name="base_converged",
        description=("frozen-battery MSE drop across P0 on the SAME battery the DV "
                     "is later read on"),
        control=("recon-only Adam on agent.e2.parameters() over buffered one-step "
                 "transitions from the no-shift env -- V3-EXQ-798a's P0 form"),
        threshold=MIN_CONV_REL_DROP, direction="lower",
        applies_to=_is_converged,
        applies_note=("an UNCONVERGED arm is DEFINED by running no optimiser, so "
                      "asserting convergence there would be structurally "
                      "un-passable and would collapse the two-regime design")),
    PreconditionSpec(
        name="base_unconverged",
        description="the same drop, bounded ABOVE, for the arms that must not train",
        control=("identical P0 rollout with the optimiser withheld; the bound is a "
                 "positive check that the regime label is true, not an assumption"),
        threshold=MAX_UNCONV_REL_DROP, direction="upper",
        applies_to=_is_unconverged,
        applies_note="a CONVERGED arm is required to exceed this bound, not stay under it",
        # A CEILING's satisfiability bound is the MINIMUM attainable value
        # (_spec_unsatisfiable, precondition_gate.py:213-221 -- it ignores
        # structural_max on a ceiling). With the optimiser withheld the battery
        # tensors are fixed and world_forward is untouched, so pre- and post-P0
        # readouts are bitwise identical and the drop is EXACTLY 0.
        structural_min=lambda ctx: 0.0),
    PreconditionSpec(
        name="battery_mse_delta_not_pinned",
        description=("|summed across-sleep MSE delta| -- the SAME statistic D1/D3 "
                     "take the SIGN of. A zero here means the instrument never "
                     "moved, which is 'not ready', not 'no effect'"),
        control=("measured on the lever-ON arm, where the world heads provably "
                 "moved (C3); it certifies the readout can move, and says nothing "
                 "about which way"),
        threshold=0.0, direction="lower", applies_to=_is_lever_on,
        applies_note=("an OFF arm's delta is EXACTLY 0 by structural identity -- "
                      "that is the control, and asserting non-zero there would be "
                      "structurally un-passable"),
        structural_max=None),
    PreconditionSpec(
        name="battery_infonce_delta_not_pinned",
        description=("|summed across-sleep InfoNCE delta| -- the SAME statistic "
                     "D2/D4 take the SIGN of"),
        control="same positive control as the MSE leg, on the same battery",
        threshold=0.0, direction="lower", applies_to=_is_lever_on,
        applies_note="same structural-identity reason as the MSE leg",
        structural_max=None),
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _finite_or_none(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _mean(xs: List[float]) -> float:
    vals = [float(x) for x in xs if _finite_or_none(x) is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def _sd(xs: List[float]) -> float:
    vals = [float(x) for x in xs if _finite_or_none(x) is not None]
    if len(vals) < 2:
        return float("nan")
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _to_batched(x: Any, device: Any) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x = x.to(device)
    return x.unsqueeze(0) if x.dim() == 1 else x


def _make_env(seed: int) -> CausalGridWorldV2:
    """No world_rule_shift in ANY arm -- this run poses no intake question."""
    return CausalGridWorldV2(
        seed=seed, size=GRID_SIZE, num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES, use_proxy_fields=True)


def _make_agent(env: CausalGridWorldV2, lever_on: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        sws_enabled=True,
        rem_enabled=True,
        use_sleep_loop=True,
        # A huge K keeps notify_episode_end's automatic cadence from ever firing,
        # so the N_CYCLES deliberate force_cycle() calls are the only sleep cycles.
        sleep_loop_episodes_K=1_000_000,
        use_sleep_aggregation_cluster=True,
        # IDENTICAL IN ALL FOUR ARMS -- only the lever below differs.
        use_cross_module_consolidation=True,
        cross_module_consolidation_schedule="interleaved",
        cross_module_consolidation_steps=CMC_STEPS,
        cross_module_consolidation_lr=CMC_LR,
        cross_module_consolidation_batch=CMC_BATCH,
        use_sleep_world_forward_consolidation=lever_on,
        # MECH-205 instrument, live in every arm so the waking phase is one
        # configuration. It routes no criterion here; leg A is the falsifier's.
        surprise_gated_replay=True,
        pe_ema_alpha=0.02,
        # P2 of INV-063's own preconditions: the MEL CONSUMER stays absent, so the
        # offline budget is scheduler-pinned and identical across arms. Asserted
        # from OUTPUT below (sws_n_writes / rem_n_rollouts), not from this config.
        use_mel_consumer=False,
        use_entry_pressure=False,
        use_within_life_sleep_trigger=False,
    )
    return REEAgent(cfg)


def _world_head_params(agent: REEAgent) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for name in WORLD_HEAD_MODULES:
        mod = getattr(agent.e2, name, None)
        if mod is not None:
            out.extend(mod.parameters())
    return out


def _snapshot(params: List[torch.Tensor]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _max_abs_delta(before: List[torch.Tensor], after: List[torch.Tensor]) -> float:
    worst = 0.0
    for b, a in zip(before, after):
        if b.numel() == 0:
            continue
        worst = max(worst, float((a - b).abs().max().item()))
    return worst


def _all_finite(params: List[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(p).all().item()) for p in params)


def _sense(agent: REEAgent, obs_dict: Dict[str, Any]):
    device = agent.device
    obs_harm = obs_dict.get("harm_obs", None)
    return agent.sense(
        _to_batched(obs_dict["body_state"], device),
        _to_batched(obs_dict["world_state"], device),
        obs_harm=_to_batched(obs_harm, device) if obs_harm is not None else None,
    )


# ---------------------------------------------------------------------------
# the frozen held-out battery and its two readouts
# ---------------------------------------------------------------------------
def _sample_probe_battery(agent: REEAgent, seed: int, n_transitions: int
                          ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """701b:489-531 / V3-EXQ-1060:410-450 form. HELD OUT: a distinct env instance
    and a FIXED action policy independent of training. PURE READ: senses and steps
    only; it never calls _e1_tick, so it appends nothing to the replay buffers."""
    env = _make_env(seed + 9973)
    _, obs_dict = env.reset()
    act_rng = random.Random(seed + 9973)
    battery: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    guard = 0
    max_guard = max(n_transitions, 1) * 8
    with torch.no_grad():
        while len(battery) < n_transitions and guard < max_guard:
            guard += 1
            z_now = _sense(agent, obs_dict).z_world.detach().reshape(1, -1).clone()
            if not bool(torch.isfinite(z_now).all().item()):
                break
            if prev is not None:
                battery.append((prev[0], prev[1], z_now))
            idx = act_rng.randrange(env.action_dim)
            action = torch.zeros(1, env.action_dim, device=agent.device)
            action[0, idx] = 1.0
            _, _, done, _, obs_dict = env.step(action)
            prev = (z_now, action)
            if done:
                _, obs_dict = env.reset()
                prev = None
    return battery


def _battery_tensors(battery: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                     device: Any
                     ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if len(battery) < 2:
        return None
    z0 = torch.cat([b[0] for b in battery], dim=0).to(device)
    acts = torch.cat([b[1] for b in battery], dim=0).to(device)
    z1 = torch.cat([b[2] for b in battery], dim=0).to(device)
    return z0, acts, z1


def _battery_readouts(agent: REEAgent,
                      tensors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
                      ) -> Dict[str, float]:
    """BOTH readouts on the SAME frozen battery -- the whole point of this run.

    mse     -- per-element reconstruction error, 701b:534-552 _frozen_probe_pe.
               This is the quantity INV-063's what_would_answer names.
    infonce -- the SD-056 objective compute_e2_world_loss actually minimises,
               called exactly as the trainer's call site does (min_batch_classes=1).
    """
    out = {"mse": float("nan"), "infonce": float("nan")}
    if tensors is None:
        return out
    z0, acts, z1 = tensors
    with torch.no_grad():
        pred = agent.e2.world_forward(z0, acts)
        out["mse"] = float((pred - z1).pow(2).mean().item())
        try:
            loss = agent.e2.world_forward_contrastive_loss(
                z_world_0=z0, actions=acts, z_world_1_targets=z1,
                min_batch_classes=1, simulation_mode=False,
            )
            out["infonce"] = (float(loss.detach().item())
                              if torch.is_tensor(loss) else float("nan"))
        except (RuntimeError, ValueError):
            out["infonce"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# P0 -- identical rollout in every arm; the optimiser is the ONLY difference
# ---------------------------------------------------------------------------
def _p0_train_step(agent: REEAgent, buf: Deque, opt: torch.optim.Optimizer,
                   rng: random.Random) -> Optional[float]:
    """One world-forward training step. RECON-ONLY: reconstruction MSE on buffered
    one-step transitions. The SD-056 contrastive auxiliary is omitted -- it is a
    CONFIRMED P0 destabiliser (V3-EXQ-701b ablation, carried by 798a:606-609)."""
    if len(buf) < MIN_BUF_BEFORE_TRAIN:
        return None
    pool = list(buf)
    batch = pool if len(pool) <= BATCH_K else rng.sample(pool, BATCH_K)
    z0 = torch.stack([t[0] for t in batch]).to(agent.device)
    acts = torch.stack([t[1] for t in batch]).to(agent.device)
    z1 = torch.stack([t[2] for t in batch]).to(agent.device)
    opt.zero_grad(set_to_none=True)
    loss = F.mse_loss(agent.e2.world_forward(z0, acts), z1)
    val = float(loss.detach().item())
    if not math.isfinite(val):
        return val
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    opt.step()
    return val


def _run_p0(agent: REEAgent, seed: int, arm: str, train: bool,
            battery_tensors: Any, p0_steps: int) -> Dict[str, Any]:
    """The rollout is IDENTICAL whether or not `train` -- same env, same seed, same
    action stream, same buffering. Only the optimiser step is withheld."""
    env = _make_env(seed)
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_LR)
    buf: Deque = deque(maxlen=BUF_MAX)
    rng = random.Random(seed)
    device = agent.device
    prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    trajectory: List[Dict[str, float]] = []
    losses: List[float] = []

    for step in range(p0_steps + 1):
        if step in P0_TRAJECTORY_AT or step == p0_steps:
            r = _battery_readouts(agent, battery_tensors)
            trajectory.append({"p0_step": float(step), "mse": r["mse"],
                               "infonce": r["infonce"]})
        if step == p0_steps:
            break
        if step % STEPS_PER_EPISODE == 0:
            ep = step // STEPS_PER_EPISODE
            print(f"  [train] {arm} seed={seed} ep {ep + 1}/{EPISODES_PER_RUN} "
                  f"phase=P0 train={int(train)}", flush=True)
        z_now = _sense(agent, obs_dict).z_world.detach().reshape(-1).clone()
        if prev is not None and bool(torch.isfinite(z_now).all().item()):
            buf.append((prev[0], prev[1], z_now))
        idx = rng.randrange(env.action_dim)
        a_vec = torch.zeros(env.action_dim, dtype=torch.float32)
        a_vec[idx] = 1.0
        prev = (z_now, a_vec)
        _, _, done, _, obs_dict = env.step(a_vec.unsqueeze(0).to(device))
        if train:
            lv = _p0_train_step(agent, buf, opt, rng)
            if lv is not None:
                losses.append(lv)
        if done:
            _, obs_dict = env.reset()
            agent.e1.reset_hidden_state()
            prev = None

    return {"trajectory": trajectory, "p0_loss_mean": _mean(losses),
            "p0_n_train_steps": float(len(losses))}


# ---------------------------------------------------------------------------
# one (arm, seed) cell
# ---------------------------------------------------------------------------
def run_cell(arm: str, seed: int, p0_steps: int, n_cycles: int,
             wake_eps: int, steps: int, battery_size: int) -> Dict[str, Any]:
    spec = ARM_SPEC[arm]
    lever_on = bool(spec["lever"])
    train_p0 = spec["base"] == "converged"
    print(f"Seed {seed} Condition {arm}", flush=True)

    env0 = _make_env(seed)
    agent = _make_agent(env0, lever_on)
    device = agent.device
    assert agent.sleep_loop is not None, "use_sleep_loop=True must build sleep_loop"

    world_params = _world_head_params(agent)

    # ---- frozen battery, captured ONCE, BEFORE P0 --------------------------
    # Nothing in this script trains the encoder, so these tensors are a fixed
    # latent-space reference for the whole cell (and identical across arms at a
    # given seed). Capturing pre-P0 makes conv_rel_drop a measurement on the SAME
    # battery the DV is later read on.
    battery = _sample_probe_battery(agent, seed, battery_size)
    battery_tensors = _battery_tensors(battery, device)
    n_battery = float(len(battery))
    agent.reset()
    agent.e1.reset_hidden_state()

    # ---- P0 -----------------------------------------------------------------
    p0 = _run_p0(agent, seed, arm, train_p0, battery_tensors, p0_steps)
    traj = p0["trajectory"]
    mse_before_p0 = traj[0]["mse"]
    mse_after_p0 = traj[-1]["mse"]
    conv_rel_drop = (((mse_before_p0 - mse_after_p0) / mse_before_p0)
                     if _finite_or_none(mse_before_p0) is not None
                     and mse_before_p0 > EPS else 0.0)

    # ---- P1: waking rollout + sleep cycles ---------------------------------
    env = _make_env(seed + 1)
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    act_rng = torch.Generator(device="cpu").manual_seed(seed + 7717)

    per_cycle: List[Dict[str, float]] = []
    cycles_fired = 0
    sws_writes: List[float] = []
    rem_rollouts: List[float] = []
    e2_world_key_seen: List[bool] = []
    updates_e2: List[float] = []
    updates_e2_world: List[float] = []
    world_delta_total = 0.0
    wenc_delta_total = 0.0
    finite_ok = True

    for cyc in range(n_cycles):
        for ep in range(wake_eps):
            ep_global = P0_EPISODE_EQUIV + cyc * wake_eps + ep
            print(f"  [train] {arm} seed={seed} ep {ep_global + 1}/{EPISODES_PER_RUN} "
                  f"phase=P1 cycle={cyc + 1}/{n_cycles}", flush=True)
            for _step in range(steps):
                latent = _sense(agent, obs_dict)
                ticks = agent.clock.advance()
                wdim = latent.z_world.shape[-1]
                e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick", False)
                            else torch.zeros(1, wdim, device=device))
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                if action is None:
                    idx = int(torch.randint(0, env.action_dim, (1,),
                                            generator=act_rng).item())
                    action = torch.zeros(1, env.action_dim, device=device)
                    action[0, idx] = 1.0
                    agent._last_action = action
                if not bool(torch.isfinite(action).all().item()):
                    finite_ok = False
                    break
                # Without this the replay action buffer is all zeros and
                # world_action_encoder is structurally untrainable (nn.Linear, zero
                # input -> dL/dW = 0 EXACTLY). V3-EXQ-1060 red-team Finding 1.
                # Called in EVERY arm, so the arms differ by the lever alone.
                agent.record_executed_action(action)
                _, harm_signal, done, _info, obs_dict = env.step(action)
                with torch.no_grad():
                    agent.update_residue(harm_signal=float(harm_signal),
                                         world_delta=None, hypothesis_tag=False,
                                         owned=True)
                if done:
                    _, obs_dict = env.reset()
                    agent.e1.reset_hidden_state()
            if not finite_ok:
                break
        if not finite_ok:
            break

        pre = _battery_readouts(agent, battery_tensors)
        before = _snapshot(world_params)
        metrics = agent.sleep_loop.force_cycle(agent) or {}
        after = _snapshot(world_params)
        post = _battery_readouts(agent, battery_tensors)

        fired = "post_sleep_z_goal_retention" in metrics
        if fired:
            cycles_fired += 1
        sws_writes.append(float(metrics.get("sws_n_writes", 0.0)))
        rem_rollouts.append(float(metrics.get("rem_n_rollouts", 0.0)))
        e2_world_key_seen.append(
            "cross_module_consolidation_updates_e2_world" in metrics)
        updates_e2.append(
            float(metrics.get("cross_module_consolidation_updates_e2", 0.0)))
        updates_e2_world.append(
            float(metrics.get("cross_module_consolidation_updates_e2_world", 0.0)))
        world_delta_total = max(world_delta_total, _max_abs_delta(before, after))
        wenc = getattr(agent.e2, "world_action_encoder", None)
        if wenc is not None:
            idxs = [i for i, p in enumerate(world_params)
                    if any(p is q for q in wenc.parameters())]
            for i in idxs:
                wenc_delta_total = max(
                    wenc_delta_total, _max_abs_delta([before[i]], [after[i]]))

        per_cycle.append({
            "cycle": float(cyc + 1),
            "mse_pre": pre["mse"], "mse_post": post["mse"],
            "mse_delta": (pre["mse"] - post["mse"]),
            "infonce_pre": pre["infonce"], "infonce_post": post["infonce"],
            "infonce_delta": (pre["infonce"] - post["infonce"]),
            "sleep_cycle_fired": float(fired),
        })

    # ---- the trainer's own loss and gradient, probed from output -----------
    # Run in BOTH lever arms so the global-RNG state is matched; the OFF arm simply
    # never uses the closure. Grads are cleared after, and
    # cross_module_consolidation.py:178 zero_grad()s before its own backward, so
    # this probe cannot leak an update into either arm. Probed AFTER the cycles so
    # the buffers it reads are the ones the cycles actually drew from.
    world_loss_probe = float("nan")
    world_grad_probe = float("nan")
    try:
        loss = agent.compute_e2_world_loss(batch_size=CMC_BATCH)
        world_loss_probe = float(loss.detach().item())
        for p in world_params:
            p.grad = None
        if loss.requires_grad and world_loss_probe != 0.0:
            loss.backward()
            world_grad_probe = max(
                (float(p.grad.abs().max().item()) if p.grad is not None else 0.0)
                for p in world_params) if world_params else 0.0
        else:
            world_grad_probe = 0.0
    except (RuntimeError, ValueError):
        world_loss_probe = 0.0
        world_grad_probe = 0.0
    finally:
        for p in world_params:
            p.grad = None

    n_world_buffer = len(getattr(agent, "_world_experience_buffer", []))
    n_action_buffer = len(getattr(agent, "_action_experience_buffer", []))
    n_pairs = float(min(n_world_buffer, n_action_buffer) - 1)
    try:
        action_nonzero_fraction = float(
            agent.e1_action_buffer_stats().get("nonzero_fraction", 0.0))
    except (AttributeError, RuntimeError, ValueError):
        action_nonzero_fraction = float("nan")

    mse_deltas = [c["mse_delta"] for c in per_cycle]
    infonce_deltas = [c["infonce_delta"] for c in per_cycle]
    mse_delta_sum = sum(d for d in mse_deltas if _finite_or_none(d) is not None)
    infonce_delta_sum = sum(d for d in infonce_deltas
                            if _finite_or_none(d) is not None)
    readouts_finite = bool(
        per_cycle
        and all(_finite_or_none(c[k]) is not None
                for c in per_cycle
                for k in ("mse_pre", "mse_post", "infonce_pre", "infonce_post")))

    params_finite = bool(
        _all_finite(world_params)
        and _all_finite(list(agent.e2.parameters()))
        and _all_finite(list(agent.e1.parameters())))

    # ---- per-cell regime-conditioned gate ----------------------------------
    cell_id = f"{arm}|seed{seed}"
    arm_ctx = {"id": cell_id, "arm": arm, "seed": seed,
               "base": spec["base"], "lever": lever_on}
    measured = {
        "world_replay_pairs_populated": n_pairs,
        "world_loss_non_sentinel": world_loss_probe,
        "world_grad_nonzero": world_grad_probe,
        "action_buffer_non_vacuous": action_nonzero_fraction,
        "frozen_battery_populated": n_battery,
        "sleep_cycles_fired": float(cycles_fired),
    }
    if spec["base"] == "converged":
        measured["base_converged"] = float(conv_rel_drop)
    else:
        measured["base_unconverged"] = float(conv_rel_drop)
    if lever_on:
        measured["battery_mse_delta_not_pinned"] = abs(mse_delta_sum)
        measured["battery_infonce_delta_not_pinned"] = abs(infonce_delta_sum)
    gate = evaluate_arm_gate(cell_id, arm_ctx, PRECONDITIONS, measured)

    all_cycles_fired = bool(cycles_fired == n_cycles)
    cell_ok = bool(gate["gate_green"] and params_finite and readouts_finite
                   and all_cycles_fired)
    print(
        f"  {arm} seed={seed} conv_rel_drop={conv_rel_drop:.4f} "
        f"mse_p0 {mse_before_p0:.6g} -> {mse_after_p0:.6g} "
        f"cycles={cycles_fired}/{n_cycles} n_pairs={n_pairs:.0f} "
        f"act_nonzero={action_nonzero_fraction:.3g} "
        f"world_delta={world_delta_total:.6g} wenc_delta={wenc_delta_total:.6g} "
        f"sws_writes={_mean(sws_writes):.3g} rem_rollouts={_mean(rem_rollouts):.3g}",
        flush=True)
    print(
        f"  {arm} seed={seed} [DV, routes no PASS/FAIL] "
        f"mse_delta_sum={mse_delta_sum:.6g} per_cycle="
        f"{['%.4g' % d for d in mse_deltas]}  "
        f"infonce_delta_sum={infonce_delta_sum:.6g} per_cycle="
        f"{['%.4g' % d for d in infonce_deltas]}",
        flush=True)
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)

    return {
        "arm": arm,
        "seed": seed,
        "base": spec["base"],
        "lever_on": lever_on,
        "cell_ok": cell_ok,
        "gate": gate,
        "params_finite": params_finite,
        "readouts_finite": readouts_finite,
        "all_cycles_fired": all_cycles_fired,
        "conv_rel_drop": float(conv_rel_drop),
        "p0_battery_mse_before": mse_before_p0,
        "p0_battery_mse_after": mse_after_p0,
        "p0_convergence_trajectory": traj,
        "p0_loss_mean": p0["p0_loss_mean"],
        "p0_n_train_steps": p0["p0_n_train_steps"],
        "n_battery": n_battery,
        "n_pairs": n_pairs,
        "world_loss_probe": world_loss_probe,
        "world_grad_probe": world_grad_probe,
        "action_buffer_nonzero_fraction": action_nonzero_fraction,
        "cycles_fired": float(cycles_fired),
        "sws_n_writes_per_cycle": sws_writes,
        "rem_n_rollouts_per_cycle": rem_rollouts,
        "sws_n_writes_min": (min(sws_writes) if sws_writes else 0.0),
        "rem_n_rollouts_min": (min(rem_rollouts) if rem_rollouts else 0.0),
        "e2_world_key_present": bool(e2_world_key_seen and all(e2_world_key_seen)),
        "updates_e2_min": (min(updates_e2) if updates_e2 else 0.0),
        "updates_e2_world_min": (min(updates_e2_world) if updates_e2_world else 0.0),
        "world_head_max_abs_delta": world_delta_total,
        "world_action_encoder_weight_delta": wenc_delta_total,
        "per_cycle_readouts": per_cycle,
        "mse_delta_sum": mse_delta_sum,
        "infonce_delta_sum": infonce_delta_sum,
        "agent": agent,
    }


# ---------------------------------------------------------------------------
# direction routing -- D1..D4 pick the label and route NO PASS/FAIL
# ---------------------------------------------------------------------------
def _direction_label(d1: bool, d2: bool, d3: bool, d4: bool) -> str:
    """Deterministic, total over the 16 outcomes. Named where a name carries
    information; bit-encoded otherwise so no outcome falls through unlabelled."""
    if d1 and d2 and d3 and d4:
        return "legb_dv_positive_both_readouts_both_regimes"
    if (not d1) and (not d2) and (not d3) and (not d4):
        return "legb_dv_negative_no_readout_moves_as_asserted"
    if d3 and d4 and (not d1) and (not d2):
        return "legb_dv_positive_only_on_unconverged_base"
    if d2 and d4 and (not d1) and (not d3):
        return "legb_dv_positive_only_in_infonce_readout"
    if d1 and d3 and (not d2) and (not d4):
        return "legb_dv_positive_only_in_mse_readout"
    if (not d1) and d2:
        return "legb_dv_converged_base_infonce_positive_mse_not"
    if d1 and (not d2):
        return "legb_dv_converged_base_mse_positive_infonce_not"
    return (f"legb_dv_mixed_convmse{int(d1)}_convinfonce{int(d2)}"
            f"_freshmse{int(d3)}_freshinfonce{int(d4)}")


def _worst_cell(rows: List[Dict[str, Any]], key: str, want: str
                ) -> Tuple[float, str]:
    """Return (worst value, offending cell id). `want` is "min" or "max" -- the
    direction that would BREAK the criterion, so the returned number is the one
    the indexer should recompute `met` from."""
    pairs = [(float(r[key]), f"{r['arm']}|seed{r['seed']}") for r in rows]
    return (min(pairs, key=lambda p: p[0]) if want == "min"
            else max(pairs, key=lambda p: p[0]))


def _seeds_positive(rows: List[Dict[str, Any]], key: str) -> Tuple[int, List[float]]:
    vals = [float(r[key]) for r in rows]
    return sum(1 for v in vals if _finite_or_none(v) is not None and v > 0.0), vals


# ---------------------------------------------------------------------------
def main(dry_run: bool = False) -> Tuple[str, Optional[str]]:
    seeds = list(SEEDS[:1]) if dry_run else list(SEEDS)
    p0_steps = 180 if dry_run else P0_STEPS
    n_cycles = 1 if dry_run else N_CYCLES
    wake_eps = 1 if dry_run else WAKE_EPS_PER_CYCLE
    steps = 30 if dry_run else STEPS_PER_EPISODE
    battery = BATTERY_SIZE   # never shortened -- see FIX note at BATTERY_FLOOR

    arm_contexts = [{"id": a, "arm": a, "base": ARM_SPEC[a]["base"],
                     "lever": ARM_SPEC[a]["lever"]} for a in ARMS]
    # Refuses the run BEFORE compute if any arm carries a structurally
    # unsatisfiable precondition. A pre-registered value that provably fails a
    # gate is a design-time proof -- never lower the threshold to resolve it.
    assert_no_structurally_unsatisfiable_gate(PRECONDITIONS, arm_contexts)

    t0 = time.time()
    rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    arm_results: List[Dict[str, Any]] = []
    agents_seen: List[REEAgent] = []

    for arm in ARMS:
        for seed in seeds:
            cfg_slice = {
                "experiment": EXPERIMENT_TYPE,
                "arm": arm,
                "base": ARM_SPEC[arm]["base"],
                "lever_on": ARM_SPEC[arm]["lever"],
                "p0_steps": p0_steps,
                "n_cycles": n_cycles,
                "wake_eps_per_cycle": wake_eps,
                "steps_per_episode": steps,
                "battery_size": battery,
                "e2_lr": E2_LR,
                "cmc_steps": CMC_STEPS,
                "cmc_lr": CMC_LR,
                "cmc_batch": CMC_BATCH,
                "grid_size": GRID_SIZE,
                "num_hazards": N_HAZARDS,
                "num_resources": N_RESOURCES,
                "self_dim": SELF_DIM,
                "world_dim": WORLD_DIM,
            }
            with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__),
                          extra_ineligible_reasons=[
                              "diagnostic_instrument_validation_no_reuse"]) as cell:
                row = run_cell(arm, seed, p0_steps, n_cycles, wake_eps, steps,
                               battery)
                agents_seen.append(row.pop("agent"))
                cell.stamp(row)
            rows[(arm, seed)] = row
            arm_results.append(row)

    elapsed = time.time() - t0
    all_rows = [rows[(a, s)] for a in ARMS for s in seeds]
    by_arm = {a: [rows[(a, s)] for s in seeds] for a in ARMS}
    on_rows = [r for a in ON_ARMS for r in by_arm[a]]
    off_rows = [r for a in OFF_ARMS for r in by_arm[a]]
    conv_rows = [r for r in all_rows if r["base"] == "converged"]
    fresh_rows = [r for r in all_rows if r["base"] == "unconverged"]

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    aggregate = aggregate_arm_gates([r["gate"] for r in all_rows])
    unready = [f"{r['arm']}|seed{r['seed']}" for r in all_rows
               if not r["cell_ok"]]

    run_config = {
        "arms": list(ARMS),
        "arm_spec": {a: dict(ARM_SPEC[a]) for a in ARMS},
        "seeds": list(seeds),
        "p0_steps": p0_steps,
        "p0_episode_equiv": P0_EPISODE_EQUIV,
        "p0_trajectory_at": list(P0_TRAJECTORY_AT),
        "p0_trajectory_sampling_note": (
            "Front-loaded on purpose: the authoring smoke measured conv_rel_drop "
            "0.9976 after only 180 P0 steps, so an evenly-spaced ladder would be "
            "flat past its first sample and could not locate a crossover."),
        "n_cycles": n_cycles,
        "wake_eps_per_cycle": wake_eps,
        "steps_per_episode": steps,
        "episodes_per_run": EPISODES_PER_RUN,
        "battery_size": battery,
        "battery_floor": BATTERY_FLOOR,
        "e2_lr": E2_LR,
        "batch_k": BATCH_K,
        "max_grad_norm": MAX_GRAD_NORM,
        "cmc_steps": CMC_STEPS,
        "cmc_lr": CMC_LR,
        "cmc_batch": CMC_BATCH,
        "cmc_schedule": "interleaved",
        "grid_size": GRID_SIZE,
        "num_hazards": N_HAZARDS,
        "num_resources": N_RESOURCES,
        "self_dim": SELF_DIM,
        "world_dim": WORLD_DIM,
        "world_buffer_floor": WORLD_BUFFER_FLOOR,
        "min_conv_rel_drop": MIN_CONV_REL_DROP,
        "max_unconv_rel_drop": MAX_UNCONV_REL_DROP,
        "seeds_required_for_direction": SEEDS_REQUIRED_FOR_DIRECTION,
        "lever": "use_sleep_world_forward_consolidation",
        "world_rule_shift": "DISABLED in every arm -- no intake question is posed",
    }

    base_manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "sleep_driver_pattern": (
            "manual-cycle-loop (run_sleep_cycle() called once per cycle in a "
            "dedicated N_CYCLES wake-sleep-test loop)"
        ),
        "timestamp_utc": ts,
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "arm_results": arm_results,
        "per_arm_gate": aggregate["per_arm_gate"],
        "ethics_preflight": dict(ETHICS_PREFLIGHT),
        "inv063_leg_b_dv_identity": (
            "E2, not E1. E1DeepPredictor has no world_forward head (every hit in "
            "e1_deep.py is a comment about E2's), and the V3-EXQ-701b/701c "
            "frozen-probe instrument INV-063's what_would_answer names computes "
            "agent.e2.world_forward (701b:547, 701c:554). The claims.yaml label "
            "'Leg B (E1 world-model updating)' is prose inherited from the "
            "four-function description taxonomy; the correction is open with "
            "/governance as GFLAG-0355 and does not affect this run."
        ),
        "scope_note": (
            "NO CLAIM VERDICT. This run adjudicates an INSTRUMENT: which frozen-"
            "battery readout moves in the direction INV-063 asserts, and at what "
            "base convergence. It poses NO intake question (world_rule_shift is "
            "disabled in every arm) and does not adjudicate INV-063. The four-arm "
            "falsifier's go/no-go returns to the USER with these numbers."
        ),
    }

    def _write(manifest: Dict[str, Any]) -> Optional[str]:
        if dry_run:
            print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.")
            return None
        out_path = write_flat_manifest(
            manifest, dry_run=False, config=manifest.get("config"),
            seeds=list(seeds), script_path=Path(__file__), agent=agents_seen)
        print(f"Result written to: {out_path}")
        return str(out_path)

    # ---- not-ready route ----------------------------------------------------
    if unready:
        diverged = [f"{r['arm']}|seed{r['seed']}" for r in all_rows
                    if not r["params_finite"]]
        reason = f"gate unmet in cell(s): {unready}"
        print(f"\n[{EXPERIMENT_TYPE}] -> substrate_not_ready_requeue: {reason}",
              flush=True)
        manifest = dict(base_manifest)
        manifest.update({
            "outcome": "FAIL",
            "result": "FAIL",
            "evidence_direction": "non_contributory",
            "evidence_direction_note": (
                "Diagnostic readiness unmet; NO direction reading is emitted. A "
                "direction read off an invalid measurement is worth less than no "
                "reading. This is not an INV-063 finding in either direction. "
                + reason),
            "non_degenerate": False,
            "degeneracy_reason": (
                "substrate_not_ready: " + reason + (
                    " NOTE: cell(s) " + ", ".join(diverged) + " carried a "
                    "NON-FINITE parameter after a sleep cycle -- that is numerical "
                    "DIVERGENCE, not merely an unmet readiness precondition."
                    if diverged else "")),
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": aggregate["adjudication_preconditions"],
                "criteria_non_degenerate": {},
            },
            "readout": {"substrate_ready": 0, "overall_pass": 0,
                        "n_unready_cells": len(unready)},
            "config": run_config,
            "elapsed_seconds": elapsed,
        })
        return "FAIL", _write(manifest)

    # ---- C1..C6 -- VALIDITY. These and only these route PASS/FAIL ----------
    min_conv_drop, cell_c1 = _worst_cell(conv_rows, "conv_rel_drop", "min")
    max_fresh_drop, cell_c2 = _worst_cell(fresh_rows, "conv_rel_drop", "max")
    min_world_delta_on, cell_c3 = _worst_cell(
        on_rows, "world_head_max_abs_delta", "min")
    max_world_delta_off, cell_c4 = _worst_cell(
        off_rows, "world_head_max_abs_delta", "max")
    min_battery, cell_c5 = _worst_cell(all_rows, "n_battery", "min")
    all_readouts_finite = all(r["readouts_finite"] for r in all_rows)
    min_sws, cell_sws = _worst_cell(all_rows, "sws_n_writes_min", "min")
    min_rem, cell_rem = _worst_cell(all_rows, "rem_n_rollouts_min", "min")
    cell_c6 = cell_sws if min_sws <= min_rem else cell_rem

    c1 = bool(min_conv_drop > MIN_CONV_REL_DROP)
    c2 = bool(max_fresh_drop < MAX_UNCONV_REL_DROP)
    c3 = bool(min_world_delta_on > 0.0)
    c4 = bool(max_world_delta_off == 0.0)
    c5 = bool(min_battery >= BATTERY_FLOOR and all_readouts_finite)
    c6 = bool(min_sws > 0.0 and min_rem > 0.0)

    criteria = [
        {"name": "C1_converged_arms_converged", "load_bearing": True, "passed": c1,
         "measured": float(min_conv_drop), "threshold": MIN_CONV_REL_DROP,
         "comparator": ">", "routes_verdict": True, "offending_cell": cell_c1},
        {"name": "C2_unconverged_arms_did_not_converge", "load_bearing": True,
         "passed": c2, "measured": float(max_fresh_drop),
         "threshold": MAX_UNCONV_REL_DROP, "comparator": "<", "routes_verdict": True,
         "offending_cell": cell_c2},
        {"name": "C3_lever_moves_world_heads_on", "load_bearing": True, "passed": c3,
         "measured": float(min_world_delta_on), "threshold": 0.0, "comparator": ">",
         "routes_verdict": True, "offending_cell": cell_c3},
        {"name": "C4_off_arm_world_heads_bit_identical", "load_bearing": True,
         "passed": c4, "measured": float(max_world_delta_off), "threshold": 0.0,
         "comparator": "<=", "routes_verdict": True, "offending_cell": cell_c4},
        {"name": "C5_battery_populated_and_readouts_finite", "load_bearing": True,
         "passed": c5, "measured": float(min_battery), "threshold": BATTERY_FLOOR,
         "comparator": ">=", "routes_verdict": True, "offending_cell": cell_c5},
        {"name": "C6_sleep_did_work_in_every_cell", "load_bearing": True, "passed": c6,
         "measured": float(min(min_sws, min_rem)), "threshold": 0.0, "comparator": ">",
         "routes_verdict": True, "offending_cell": cell_c6},
    ]

    # ---- D1..D4 -- the DIRECTION findings. They route NO PASS/FAIL ---------
    n_d1, v_d1 = _seeds_positive(by_arm[ARM_CONV_ON], "mse_delta_sum")
    n_d2, v_d2 = _seeds_positive(by_arm[ARM_CONV_ON], "infonce_delta_sum")
    n_d3, v_d3 = _seeds_positive(by_arm[ARM_FRESH_ON], "mse_delta_sum")
    n_d4, v_d4 = _seeds_positive(by_arm[ARM_FRESH_ON], "infonce_delta_sum")
    req = min(SEEDS_REQUIRED_FOR_DIRECTION, len(seeds))
    d1, d2, d3, d4 = (n_d1 >= req, n_d2 >= req, n_d3 >= req, n_d4 >= req)

    directions = [
        {"name": "D1_converged_base_mse_improves", "load_bearing": False,
         "passed": d1, "measured": float(n_d1), "threshold": float(req),
         "comparator": ">=", "routes_verdict": False, "per_seed": v_d1,
         "mean": _finite_or_none(_mean(v_d1)), "sd": _finite_or_none(_sd(v_d1))},
        {"name": "D2_converged_base_infonce_improves", "load_bearing": False,
         "passed": d2, "measured": float(n_d2), "threshold": float(req),
         "comparator": ">=", "routes_verdict": False, "per_seed": v_d2,
         "mean": _finite_or_none(_mean(v_d2)), "sd": _finite_or_none(_sd(v_d2))},
        {"name": "D3_unconverged_base_mse_improves", "load_bearing": False,
         "passed": d3, "measured": float(n_d3), "threshold": float(req),
         "comparator": ">=", "routes_verdict": False, "per_seed": v_d3,
         "mean": _finite_or_none(_mean(v_d3)), "sd": _finite_or_none(_sd(v_d3))},
        {"name": "D4_unconverged_base_infonce_improves", "load_bearing": False,
         "passed": d4, "measured": float(n_d4), "threshold": float(req),
         "comparator": ">=", "routes_verdict": False, "per_seed": v_d4,
         "mean": _finite_or_none(_mean(v_d4)), "sd": _finite_or_none(_sd(v_d4))},
    ]

    validity_pass = all(c["passed"] for c in criteria)
    outcome = "PASS" if validity_pass else "FAIL"
    label = (_direction_label(d1, d2, d3, d4) if validity_pass
             else "legb_dv_direction_measurement_invalid")

    # C4's value is non-degenerate exactly because the SAME MECH-423 pass ran in
    # both lever arms (updates_e2 >= 1 on every OFF cell) and the e2_world key is
    # present on ON and absent on OFF -- a genuine mechanistic absence, not a
    # coincidental tie and not the trivial "nothing ran" case.
    off_pass_really_ran = all(r["updates_e2_min"] >= 1.0 for r in off_rows)
    key_asymmetric = (all(r["e2_world_key_present"] for r in on_rows)
                      and not any(r["e2_world_key_present"] for r in off_rows))
    # A D finding is non-degenerate only if ALL of its arm's cells passed their own
    # gate -- a direction read off a red cell is not a reading. Computed here rather
    # than via arm_criteria_non_degenerate because the gate ids are per CELL while
    # each D aggregates the three cells of one arm.
    def _arm_all_green(arm: str) -> bool:
        return all(r["gate"]["gate_green"] for r in by_arm[arm])

    criteria_non_degenerate = {
        "C1_converged_arms_converged": bool(
            c1 and all(r["p0_n_train_steps"] > 0 for r in conv_rows)),
        "C2_unconverged_arms_did_not_converge": bool(
            all(r["p0_n_train_steps"] == 0 for r in fresh_rows)),
        "C3_lever_moves_world_heads_on": bool(
            c3 and all(r["world_loss_probe"] > 0.0 and r["world_grad_probe"] > 0.0
                       for r in on_rows)),
        "C4_off_arm_world_heads_bit_identical": bool(
            off_pass_really_ran and key_asymmetric),
        "C5_battery_populated_and_readouts_finite": bool(c5),
        "C6_sleep_did_work_in_every_cell": bool(c6),
        "D1_converged_base_mse_improves": _arm_all_green(ARM_CONV_ON),
        "D2_converged_base_infonce_improves": _arm_all_green(ARM_CONV_ON),
        "D3_unconverged_base_mse_improves": _arm_all_green(ARM_FRESH_ON),
        "D4_unconverged_base_infonce_improves": _arm_all_green(ARM_FRESH_ON),
    }

    note = (
        f"DIRECTION READING (routes no claim verdict). On a CONVERGED base "
        f"(min conv_rel_drop {min_conv_drop:.4f}) the across-sleep frozen-battery "
        f"MSE delta was positive on {n_d1}/{len(seeds)} seeds "
        f"(mean {_mean(v_d1):.6g}, sd {_sd(v_d1):.6g}) and the InfoNCE delta on "
        f"{n_d2}/{len(seeds)} (mean {_mean(v_d2):.6g}, sd {_sd(v_d2):.6g}). On an "
        f"UNCONVERGED base (max conv_rel_drop {max_fresh_drop:.4f}) the same two "
        f"were positive on {n_d3}/{len(seeds)} (mean {_mean(v_d3):.6g}) and "
        f"{n_d4}/{len(seeds)} (mean {_mean(v_d4):.6g}). INV-063 asserts POSITIVE "
        f"(sleep reduces held-out world-forward prediction error). Validity held: "
        f"the lever moved the world heads (min max|delta| {min_world_delta_on:.6g}) "
        f"while the OFF arms stayed bit-identical ({max_world_delta_off:.6g}) with "
        f"the SAME consolidation pass running. EXPERIMENT_PURPOSE=diagnostic; this "
        f"adjudicates an INSTRUMENT, not INV-063, and poses no intake question."
    ) if validity_pass else (
        f"Validity criteria failed: "
        f"{[c['name'] for c in criteria if not c['passed']]}. NO direction reading "
        f"is emitted -- see criteria[] for measured/threshold detail."
    )

    print(f"\n[{EXPERIMENT_TYPE}] validity (routes PASS/FAIL):", flush=True)
    for c in criteria:
        print(f"  {c['name']}: passed={c['passed']} measured={c['measured']:.6g} "
              f"{c['comparator']} {c['threshold']}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] direction (routes NO PASS/FAIL):", flush=True)
    for d in directions:
        print(f"  {d['name']}: {int(d['measured'])}/{len(seeds)} seeds positive "
              f"mean={_mean(d['per_seed']):.6g} sd={_sd(d['per_seed']):.6g} per_seed="
              f"{['%.4g' % v for v in d['per_seed']]}", flush=True)
    print(f"  -> {label} ({outcome}); elapsed={elapsed:.1f}s", flush=True)

    flat: Dict[str, float] = {
        "c1_converged_arms_converged": int(c1),
        "c2_unconverged_arms_did_not_converge": int(c2),
        "c3_lever_moves_world_heads_on": int(c3),
        "c4_off_arm_world_heads_bit_identical": int(c4),
        "c5_battery_populated_and_readouts_finite": int(c5),
        "c6_sleep_did_work_in_every_cell": int(c6),
        "overall_pass": int(validity_pass),
        "min_conv_rel_drop_converged_arms": float(min_conv_drop),
        "max_conv_rel_drop_unconverged_arms": float(max_fresh_drop),
        "min_world_head_max_abs_delta_on": float(min_world_delta_on),
        "max_world_head_max_abs_delta_off": float(max_world_delta_off),
        "min_sws_n_writes": float(min_sws),
        "min_rem_n_rollouts": float(min_rem),
        "d1_conv_mse_seeds_positive": float(n_d1),
        "d2_conv_infonce_seeds_positive": float(n_d2),
        "d3_fresh_mse_seeds_positive": float(n_d3),
        "d4_fresh_infonce_seeds_positive": float(n_d4),
        "d_seeds_required": float(req),
    }
    for key, vals in (
        ("conv_on_mse_delta", v_d1), ("conv_on_infonce_delta", v_d2),
        ("fresh_on_mse_delta", v_d3), ("fresh_on_infonce_delta", v_d4)):
        m, s = _finite_or_none(_mean(vals)), _finite_or_none(_sd(vals))
        if m is not None:
            flat[f"{key}_mean"] = m
        if s is not None:
            flat[f"{key}_sd"] = s

    manifest = dict(base_manifest)
    manifest.update({
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "NON_CONTRIBUTORY BY DESIGN, not by failure. experiment_purpose is "
            "diagnostic and this run routes no verdict on INV-063: it measures "
            "which frozen-battery readout moves in the direction the claim asserts "
            "and at what base convergence, so that the four-arm intake-ladder "
            "falsifier's go/no-go can be taken to the user with numbers. "
        ) + note,
        "non_degenerate": bool(all(criteria_non_degenerate.values())),
        "degeneracy_reason": (
            None if all(criteria_non_degenerate.values())
            else "degenerate criteria: " + ", ".join(
                k for k, v in criteria_non_degenerate.items() if not v)),
        "interpretation": {
            "label": label,
            "criteria": criteria + directions,
            "combination_rule": (
                "PASS iff ALL of C1..C6 pass (plain AND, no OR/any() branching). "
                "C1..C6 are VALIDITY criteria only. D1..D4 are pre-registered "
                "DIRECTION findings: they carry routes_verdict=false, they route NO "
                "PASS/FAIL, and they select interpretation.label. A NEGATIVE D is a "
                "RESULT, not a failure -- it is the measurement this run exists to "
                "make. If any validity criterion or readiness precondition is unmet "
                "the run self-routes substrate_not_ready_requeue and emits no "
                "direction label at all."
            ),
            "criteria_non_degenerate": criteria_non_degenerate,
            "preconditions": aggregate["adjudication_preconditions"],
        },
        "readout": flat,
        "direction_findings": {
            "asserted_direction": (
                "POSITIVE: delta = pre_sleep - post_sleep on the frozen held-out "
                "battery, so a positive delta means sleep REDUCED held-out "
                "world-forward prediction error, which is what INV-063 C1 leg B "
                "asserts ('the ACROSS-SLEEP IMPROVEMENT ...')."),
            "per_arm": {
                arm: {
                    "base": ARM_SPEC[arm]["base"],
                    "lever_on": ARM_SPEC[arm]["lever"],
                    "mse_delta_sum_per_seed": [r["mse_delta_sum"] for r in by_arm[arm]],
                    "infonce_delta_sum_per_seed": [
                        r["infonce_delta_sum"] for r in by_arm[arm]],
                    "mse_delta_sum_mean": _mean(
                        [r["mse_delta_sum"] for r in by_arm[arm]]),
                    "infonce_delta_sum_mean": _mean(
                        [r["infonce_delta_sum"] for r in by_arm[arm]]),
                    "conv_rel_drop_per_seed": [r["conv_rel_drop"] for r in by_arm[arm]],
                    "seeds": [r["seed"] for r in by_arm[arm]],
                } for arm in ARMS
            },
            "off_arm_zero_is_structural": True,
            "off_arm_note": (
                "Every OFF-arm delta is EXACTLY 0 by STRUCTURAL IDENTITY, not by "
                "measurement: world_forward depends only on the two world heads "
                "(e2_fast.py:201-221), the OFF arm applies no gradient to them, and "
                "the SAME captured battery tensors are re-evaluated -- so pre and "
                "post are bitwise identical. Do NOT read it as a measured null "
                "effect, and do NOT use it as a difference-in-differences baseline."),
            "infonce_invariance_note": (
                "The InfoNCE readout is invariant to a global scale/shift of the "
                "prediction (SD-056 contrastive) while the MSE readout is not. A "
                "manipulation that only rescaled or shifted predictions would be "
                "invisible to D2/D4 and visible to D1/D3. The manipulation here is a "
                "binary code-path gate, not a value transform, so it is not such a "
                "manipulation -- but any reader comparing D1 against D2 must hold "
                "this asymmetry in mind. It is the reason both readouts are carried."),
            "p0_convergence_trajectory_note": (
                "Each cell records the frozen-battery MSE and InfoNCE at P0 steps "
                f"{list(P0_TRAJECTORY_AT)} (arm_results[].p0_convergence_trajectory). "
                "That is what answers 'at what base convergence' at finer resolution "
                "than the two-level contrast, without another run."),
        },
        "config": run_config,
        "elapsed_seconds": elapsed,
    })
    return outcome, _write(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Smoke run: 1 seed, short P0, 1 cycle, no manifest.")
    args = parser.parse_args()
    _outcome, _manifest_path = main(dry_run=args.dry_run)
    _clean = str(_outcome).upper() if str(_outcome).upper() in ("PASS", "FAIL") else "FAIL"
    emit_outcome(outcome=_clean, manifest_path=_manifest_path, dry_run=args.dry_run)
    sys.exit(0)
