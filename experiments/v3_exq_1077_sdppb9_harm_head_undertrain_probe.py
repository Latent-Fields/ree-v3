#!/opt/local/bin/python3
"""
V3-EXQ-1077: SD-PP-B9 H-harm-head-undertrained probe -- is the e2_harm_a
forward head below the z(t-1) persistence baseline because it never converged?

experiment_purpose: diagnostic
SLEEP DRIVER: not_applicable (no sleep loop, SWS, REM or aggregation cluster is enabled)
red-team (Step 4.5): see RED_TEAM_VERDICT below -- recorded in the queue entry note as well.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
The cheapest leg of the GOV-FANOUT-1 four-leg discrimination on
SD-PP-B9-harm-forward-below-persistence-baseline (REE_assembly
evidence/planning/substrate_queue.json), from
failure_autopsy_V3-EXQ-1062a_2026-09-23 (confirmed; ratified by governance
cycle governance-20260923-0717, rec-20260923-10bca6b5). Design = Option A of
the user decision rec-20260923-533fb67e (chip
chip-20260923-mech055-probe-decision-design; full options in
ree-v3/.scratch/mech055_probe/DECISION.md).

It is NOT a re-run of MECH-055's falsifier. The re-derive brake FIRED on
MECH-055 at N=2 and refuses any V3-EXQ-1062b; the autopsy's
refused_requeue_scope exempts this probe explicitly: new EXQ number, different
mechanism (head convergence), different DV (skill vs persistence), no claim tag.
claim_ids = [] ; bears_on = ["mech055_harm_pe_source_validity"].

THE FINDING UNDER TEST
----------------------
In V3-EXQ-1062a the e2_harm_a head scored BELOW the persistence predictor
z_pred = z(t-1) in 6/6 cells (skill -71.7/-4.0/-23.5 stationary). A residual
head z + delta(z, a) has delta == 0 inside its hypothesis space, so at the MSE
optimum it cannot do worse than persistence on its training distribution. A
skill of -72 means the head was ACTIVELY adding error. 1062a trained it for
60 eps x 90 steps, ONLINE (one correlated transition per update), batch 1,
lr 5e-4, at epsilon 0.1, and evaluated at epsilon 0.0 with no P2 training.
The manifest recorded no loss curve, so convergence could not be checked.

DESIGN (Option A, as approved)
------------------------------
Stationary world only. Seeds 42 / 137 / 2026 (1062a's). Per seed, ONE shared
collection:
  P0 (30 eps)  encoder warmup exactly as 1062a (E1 + world-forward only).
  P1 (60 eps)  1062a's P1 EXACTLY: online, batch 1, lr 5e-4, epsilon 0.1,
               pairing e2_harm_a(z(t-1), a(t-1)) -> z(t). Every P1 transition
               is ALSO banked (z(t-1), a(t-1), z(t)) into a buffer, and every
               online loss is recorded (the curve 1062a never wrote).
Two heads then exist per seed, from the SAME initial weights and the SAME data:
  ARM_ONLINE_B1  -- the head as P1 left it. The reproduction control: it must
                    come out persistence_dominated at epsilon 0.0 (1062a's
                    condition) on today's substrate, or this run is not about
                    1062a's defect.
  ARM_CONVERGED  -- re-initialised to the saved PRE-P1 weights, then trained
                    offline on iid minibatches from the buffer: batch 64,
                    Adam from lr 5e-4, 10% of transitions held out. AMENDED
                    by user decision rec-20260923-f787f416 (Option 1, after a
                    BLOCKING red-team: fixed-lr Adam plateaued 1.18x ABOVE
                    persistence on its own training rows): halve lr each time
                    1000 steps pass without a >1% improvement on the BEST-so-far
                    held-out loss (a rise can no longer register as a plateau),
                    reload the best checkpoint at each halving, stop when
                    lr < 5e-4/256; cap 40000 steps. converged = schedule
                    exhausted AND best held-out loss <= persistence on the
                    same held-out rows (reachability: delta == 0 is inside the
                    residual head's hypothesis space). Train / held-out / lr
                    curves recorded (Recording Standard 3b/3c). Otherwise that
                    seed is `not_converged` (precondition unmet), never a
                    verdict.
Evaluation: each head is installed in a deep copy of the post-P1 agent + env
and run for a fresh no-grad 1800-step rollout (1062a's P2 budget) at
epsilon 0.1 (MATCHED to training) and at epsilon 0.0 (1062a's mismatched
condition) -- four rollouts per seed, RNG re-seeded identically at each
rollout start. Pairs are built exactly as 1062a's r2_pairs (fresh E3 ticks;
pred = e2_harm_a(z(t-1), a(t-1)), target z(t), prev z(t-1)).

THE INSTRUMENT
--------------
experiments/_lib/persistence_skill_gate.persistence_verdict (SD-PP-B9 piece 1).
It routes on d = (SSE_per - SSE_model) / (SSE_per + SSE_model), whose control
value is exactly 0 by construction, with a paired-bootstrap CI (N=2000, 95%,
pinned seed -- fixed by the instrument, not by this driver):
  CI low > 0 -> ready ; CI high < 0 -> persistence_dominated ;
  CI straddles 0 -> cannot_determine.
NO absolute R2 floor and NO absolute skill floor is used anywhere (the handover
measures why: a 16x denominator spread across seeds at an identical config).

PRE-REGISTERED ROUTING (quorum >= 2 of 3 seeds; per seed on ARM_CONVERGED at
epsilon 0.1, counted only for seeds that are ELIGIBLE = reproduced AND converged)
-------------------------------------------------------------------------------
  reproduction control fails (ARM_ONLINE_B1 @ eps 0.0 persistence_dominated on
    < 2 seeds)                          -> substrate_not_ready_requeue
                                           (non-attributable: the defect did
                                           not reproduce on today's substrate)
  fewer than 2 eligible seeds           -> substrate_not_ready_requeue
                                           (NOT_CONVERGED. This includes
                                           not_converged on ALL seeds. It is an
                                           INSTRUMENT outcome, NOT a structural
                                           verdict: the optimiser failed to reach
                                           the zero-delta point the head CAN
                                           represent. It says nothing about head
                                           capacity, the representation or the
                                           PE source.)
  ready >= 2                            -> h_undertrained_confirmed
                                           (PASS: training alone gives real
                                           skill; SD-PP-B9's substrate leg is a
                                           training-protocol fix, not a build)
  persistence_dominated >= 2            -> h_undertrained_refuted_declared_null
                                           (the head adds error EVEN AT
                                           CONVERGENCE -- points at pairing /
                                           eval distribution / PE source)
  cannot_determine >= 2                 -> active_error_removed_AMBIGUOUS
                                           (training removes the active error
                                           but no learnable delta shows. This is
                                           ALIASED: under-training-then-nothing-
                                           to-learn, a representation ceiling,
                                           and a structurally wrong PE source
                                           ALL predict it. It is NOT a ceiling
                                           verdict.)
  otherwise                             -> inconclusive_no_quorum
Only h_undertrained_confirmed is outcome PASS; every other label is FAIL with
its label carrying the meaning. evidence_direction is non_contributory
(claim-free diagnostic).

RECORDED, NON-GATING
--------------------
  * both heads' verdicts at BOTH epsilons (mismatch attribution: the
    converged head at eps 0.0 vs 0.1),
  * cross-scoring: each rollout's pairs scored by the OTHER head too,
  * an in-sample best-fit bound: a fresh head (same init) fitted on the eval
    rows themselves with the same decayed fitter, PLUS a permuted-delta twin
    (same rows, per-row target delta shuffled, so nothing is learnable from
    (z, a) by construction). The bound is read as "learnable" only when the
    real fit separates from the twin (real CI low > twin CI high), so a
    cannot_determine can be read as "nothing learnable" vs "learnable but
    not learned",
  * the full online and offline loss curves, buffer size, held-out size,
    stop step, the 1062a driver's sha256 (this driver imports its builders).

WHY NOT REANALYSIS (Step 2.4, GOV-REUSE-1): no manifest in
REE_assembly/evidence/experiments carries an e2_harm_a training curve or a
head trained to a held-out plateau (reanalysis_query: 0 manifests carry the
readout). The question needs a new training manipulation, so it is not
recoverable.

KNOWN LIMITATIONS (Step 2.5c, all severity degrading, WARN not BLOCK):
SD-PP-B9 itself (ree_core/latent/stack.py::ResidualHarmForward,
ree_core/predictors/e2_harm_a.py::E2HarmAForward -- the subject of this run),
SD-018, SD-106, SD-091, SD-ZWORLD-SENSE-PATH-PARITY, SD-PP-B4,
sd061-resume-progress-ecology, SD-MECH303-THRESHOLD-SOURCING,
mech005-betagate-decommit-counter-and-commit-ceiling.
"""

import argparse
import copy
import hashlib
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.fresh_select import FreshSelectCounter, FreshSelectProbe
from experiments._lib.persistence_skill_gate import (
    BOOTSTRAP_SEED, CI_LEVEL, MIN_ROWS, N_BOOTSTRAP, check_canary,
    format_verdict, persistence_verdict)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator

# The 1062a builders are IMPORTED, not copied, so the reproduction control's
# config and env are 1062a's verbatim BY CONSTRUCTION. The file is folded into
# the arm fingerprint via extra_substrate_paths and its sha256 is recorded.
from experiments import v3_exq_1062a_mech055_affect_channel_separation_postshift as B

DRIVER_1062A = Path(B.__file__).resolve()

EXPERIMENT_TYPE = "v3_exq_1077_sdppb9_harm_head_undertrain_probe"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
QUEUE_ID = "V3-EXQ-1077"
CLAIM_IDS: List[str] = []
BEARS_ON = ["mech055_harm_pe_source_validity"]
VALIDATES_SUBSTRATE = "SD-PP-B9-harm-forward-below-persistence-baseline"
RED_TEAM_VERDICT = (
    "red-team (fable) 2026-09-23: BLOCKING (F1: fixed-lr plateau certified a head "
    "1.18x ABOVE persistence on its own training rows; rule fired on a rise; "
    "in-sample bound identical for a permuted target) -> FIXED by user decision "
    "rec-20260923-f787f416 (Option 1: lr halving vs best-so-far, best checkpoint, "
    "reachability clause, permuted twin). Confirmers re-run on the driver's own "
    "fitter (seed 42, 1359-row buffer): best checkpoint = curve minimum; "
    "reachability now REFUSES to certify (held-out 1.0016x -> not_converged); "
    "in-sample real ready d=+0.034 vs permuted persistence_dominated d=-0.013, "
    "separated. Minor F3 (precondition counted converged not eligible) FIXED.")

# Both anchor preconditions are computed by SHIPPED instrument code, never a
# re-implementation: instrument_canary_... IS persistence_skill_gate.check_canary
# (the degeneracy definition, pinned from the landed 1062a manifest), and
# reproduction_control_... counts persistence_verdict statuses -- the same path
# the canary proves classifies 1062a's recorded cells persistence_dominated 6/6,
# so the gate is reachable by that control by construction.
ANCHOR_REACHABILITY_EXEMPT = (
    "anchors are the shipped check_canary / persistence_verdict path, not a "
    "re-implementation; the canary pins that path to persistence_dominated on "
    "1062a's recorded control cells, so the reproduction anchor is reachable")

# ---- schedule: 1062a's, unchanged ------------------------------------------
SEEDS = [42, 137, 2026]
P0_EPS = B.P0_EPS                    # 30
P1_EPS = B.P1_EPS                    # 60
TOTAL_TRAINING_EPS = P0_EPS + P1_EPS  # the [train] ep N/M denominator (90)
STEPS_PER_EPISODE = B.STEPS_PER_EPISODE  # 90
EVAL_STEP_BUDGET = B.P2_STEP_BUDGET   # 1800, per rollout
EPSILON_TRAIN = B.EPSILON_TRAIN       # 0.1
EVAL_EPSILONS = [0.1, 0.0]            # matched ; 1062a's condition
E2_HARM_A_LR = B.E2_HARM_A_LR         # 5e-4 -- held fixed for BOTH arms

# ---- ARM_CONVERGED offline training (Option A, pre-registered) ---------------
CONV_BATCH = 64
CONV_LR = E2_HARM_A_LR
HELDOUT_FRACTION = 0.10
PLATEAU_WINDOW = 1000        # steps without a >1% improvement on BEST-so-far
PLATEAU_REL_IMPROVEMENT = 0.01
LR_DECAY_FACTOR = 0.5        # halve lr on each plateau (reload best checkpoint)
MIN_LR = CONV_LR / 256.0     # schedule exhausted when lr falls below this
CONV_CAP = 40000             # steps; hitting it = not_converged
EVAL_EVERY = 100             # steps between held-out evaluations
PERMUTE_SEED_OFFSET = 104729 # generator for the permuted-delta twin
SPLIT_SEED_OFFSET = 7919     # generator for the split + minibatch draws

# ---- routing ----------------------------------------------------------------
SEEDS_REQUIRED = 2           # quorum: >= 2 of 3 seeds

ARM_ONLINE = "ARM_ONLINE_B1"
ARM_CONV = "ARM_CONVERGED"

_ZG = ZGoalStreamAccumulator()


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _config_slice() -> Dict[str, Any]:
    return {
        "base_config": "v3_exq_1062a make_config/_make_env (imported)",
        "base_driver_sha256": _sha256(DRIVER_1062A),
        "env": dict(B.ENV_KWARGS),
        "world_rule_shift": "disabled_throughout (stationary only)",
        "schedule": {"p0": P0_EPS, "p1": P1_EPS, "steps": STEPS_PER_EPISODE,
                     "epsilon_train": EPSILON_TRAIN,
                     "eval_step_budget": EVAL_STEP_BUDGET,
                     "eval_epsilons": EVAL_EPSILONS},
        "online_arm": {"batch": 1, "lr": E2_HARM_A_LR, "online": True},
        "converged_arm": {"batch": CONV_BATCH, "lr": CONV_LR,
                          "heldout_fraction": HELDOUT_FRACTION,
                          "plateau_window": PLATEAU_WINDOW,
                          "plateau_rel_improvement": PLATEAU_REL_IMPROVEMENT,
                          "lr_decay_factor": LR_DECAY_FACTOR, "min_lr": MIN_LR,
                          "decay_reference": "best_so_far_heldout",
                          "checkpoint": "best_heldout_reloaded_at_each_decay",
                          "reachability": "best_heldout_loss <= persistence_heldout_loss",
                          "permute_seed_offset": PERMUTE_SEED_OFFSET,
                          "cap": CONV_CAP, "eval_every": EVAL_EVERY,
                          "split_seed_offset": SPLIT_SEED_OFFSET},
        "seeds_required": SEEDS_REQUIRED,
        "instrument": {"n_bootstrap": N_BOOTSTRAP, "ci_level": CI_LEVEL,
                       "bootstrap_seed": BOOTSTRAP_SEED, "min_rows": MIN_ROWS},
        "fresh_select_namespace": B.FRESH_SELECT_NAMESPACE,
    }


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# P0 + P1: 1062a's loop, verbatim in behaviour, plus banking + loss recording  #
# --------------------------------------------------------------------------- #

def _collect(agent, env, seed: int, p0_eps: int, p1_eps: int,
             steps_per_ep: int) -> Dict[str, Any]:
    total_training = p0_eps + p1_eps
    e2a_opt = optim.Adam(agent.e2_harm_a.parameters(), lr=E2_HARM_A_LR)
    e1_opt = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
    wf_params = (list(agent.e2.world_transition.parameters())
                 + list(agent.e2.world_action_encoder.parameters()))
    wf_opt = optim.Adam(wf_params, lr=1e-3)
    wf_buf: List[Tuple] = []
    max_buf = 2000

    buf_prev: List[torch.Tensor] = []
    buf_act: List[torch.Tensor] = []
    buf_next: List[torch.Tensor] = []
    online_losses: List[float] = []
    online_ep_mean: List[float] = []

    for ep in range(total_training):
        is_p0 = ep < p0_eps
        agent.reset()
        _obs, od = env.reset()
        prev_zha: Optional[torch.Tensor] = None
        prev_zw: Optional[torch.Tensor] = None
        prev_action: Optional[torch.Tensor] = None
        ep_losses: List[float] = []

        if (ep + 1) % 10 == 0 or ep == 0:
            print("  [train] %s seed=%d ep %d/%d"
                  % ("SHARED_COLLECTION", seed, ep + 1, total_training), flush=True)

        for _step in range(steps_per_ep):
            body, world, harm, harm_a, hh = B._obs_tensors(od)
            latent = agent.sense(obs_body=body, obs_world=world, obs_harm=harm,
                                 obs_harm_a=harm_a, obs_harm_history=hh)
            ticks = agent.clock.advance()
            e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick")
                        else torch.zeros(1, B.WORLD_DIM, device=agent.device))
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            if EPSILON_TRAIN > 0.0 and random.random() < EPSILON_TRAIN:
                ai = random.randint(0, env.action_dim - 1)
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, ai] = 1.0

            zw_curr = latent.z_world.detach()

            if is_p0:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_opt.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                    e1_opt.step()
                if prev_zw is not None and prev_action is not None:
                    wf_buf.append((prev_zw.cpu(), prev_action.cpu(), zw_curr.cpu()))
                    if len(wf_buf) > max_buf:
                        wf_buf = wf_buf[-max_buf:]
                if len(wf_buf) >= 16:
                    k = min(32, len(wf_buf))
                    idx = torch.randperm(len(wf_buf))[:k].tolist()
                    zb = torch.cat([wf_buf[i][0] for i in idx]).to(agent.device)
                    ab = torch.cat([wf_buf[i][1] for i in idx]).to(agent.device)
                    zn = torch.cat([wf_buf[i][2] for i in idx]).to(agent.device)
                    wf_l = F.mse_loss(agent.e2.world_forward(zb, ab), zn)
                    if wf_l.requires_grad:
                        wf_opt.zero_grad()
                        wf_l.backward()
                        torch.nn.utils.clip_grad_norm_(wf_params, 1.0)
                        wf_opt.step()

            # P1: 1062a's online update, same causal pairing (prev_action).
            if ((not is_p0) and prev_zha is not None
                    and prev_action is not None and latent.z_harm_a is not None):
                z_pred = agent.e2_harm_a(prev_zha.detach(), prev_action.detach())
                loss = agent.e2_harm_a.compute_loss(z_pred, latent.z_harm_a.detach())
                if loss.requires_grad:
                    e2a_opt.zero_grad()
                    loss.backward()
                    e2a_opt.step()
                lv = float(loss.item())
                online_losses.append(lv)
                ep_losses.append(lv)
                # Bank the transition. No RNG is consumed here, so the P1
                # trajectory is identical to 1062a's at the same seed.
                buf_prev.append(prev_zha.detach().cpu().clone())
                buf_act.append(prev_action.detach().cpu().clone())
                buf_next.append(latent.z_harm_a.detach().cpu().clone())

            B._drive_valence_write_paths(agent, body)
            _obs, harm_signal, done, _info, od = env.step(
                int(action.argmax(dim=-1).item()))
            agent.update_residue(float(harm_signal) if harm_signal is not None else 0.0)

            prev_zha = (latent.z_harm_a.detach().clone()
                        if latent.z_harm_a is not None else None)
            prev_zw = zw_curr
            prev_action = action.detach()
            if done:
                break
        if not is_p0:
            online_ep_mean.append(float(np.mean(ep_losses)) if ep_losses else float("nan"))

    return {
        "prev": torch.cat(buf_prev) if buf_prev else torch.zeros(0, B.HARM_A_DIM),
        "act": torch.cat(buf_act) if buf_act else torch.zeros(0, env.action_dim),
        "next": torch.cat(buf_next) if buf_next else torch.zeros(0, B.HARM_A_DIM),
        "online_losses": online_losses,
        "online_ep_mean": online_ep_mean,
    }


# --------------------------------------------------------------------------- #
# Offline training to a held-out plateau (ARM_CONVERGED and the oracle bound)  #
# --------------------------------------------------------------------------- #

def _fit_to_plateau(head: torch.nn.Module, prev: torch.Tensor, act: torch.Tensor,
                    nxt: torch.Tensor, gen: torch.Generator, heldout_fraction: float,
                    cap: int, window: int, eval_every: int) -> Dict[str, Any]:
    """Decayed-lr fit to a best-checkpoint plateau (Option 1, rec-20260923-f787f416).

    Adam from CONV_LR on iid minibatches of CONV_BATCH. Every `eval_every` steps the
    evaluation loss is compared against the BEST-so-far value (not the value
    `window` steps back, so a RISE can never register as a plateau -- red-team
    finding F1b). When `window` steps pass without a > PLATEAU_REL_IMPROVEMENT
    relative improvement on the best, the lr is multiplied by LR_DECAY_FACTOR and
    the best checkpoint is reloaded. The schedule is EXHAUSTED when lr < MIN_LR.
    The head returned is always the best checkpoint.

    converged = schedule exhausted before `cap` AND the best evaluation loss is
    <= the persistence predictor's loss on the SAME rows. The second clause is
    reachability, not a new bar: the head is residual (z + delta), so delta == 0
    -- persistence exactly -- is inside its hypothesis space. A fit that ends
    above it did not reach a point it can represent (red-team F1a).

    heldout_fraction == 0 -> in-sample fit; the evaluation rows ARE the training
    rows (used only for the recorded, non-gating in-sample bound)."""
    n = int(prev.shape[0])
    perm = torch.randperm(n, generator=gen)
    n_ho = int(round(heldout_fraction * n)) if heldout_fraction > 0 else 0
    ho_idx = perm[:n_ho]
    tr_idx = perm[n_ho:] if n_ho > 0 else perm
    ev_idx = ho_idx if n_ho > 0 else tr_idx
    lr = CONV_LR
    opt = optim.Adam(head.parameters(), lr=lr)

    def _loss(idx: torch.Tensor) -> float:
        with torch.no_grad():
            return float(F.mse_loss(head(prev[idx], act[idx]), nxt[idx]).item())

    with torch.no_grad():
        persistence_eval_loss = float(F.mse_loss(prev[ev_idx], nxt[ev_idx]).item())
        persistence_train_loss = float(F.mse_loss(prev[tr_idx], nxt[tr_idx]).item())

    best = _loss(ev_idx)
    best_step = 0
    best_state = copy.deepcopy(head.state_dict())
    since = 0
    curve_step: List[int] = [0]
    curve_eval: List[float] = [best]
    curve_train: List[float] = [_loss(tr_idx)]
    curve_lr: List[float] = [lr]
    decay_steps: List[int] = []
    recent: List[float] = []
    exhausted = False
    step = 0
    while step < cap:
        step += 1
        bi = tr_idx[torch.randint(0, int(tr_idx.shape[0]), (CONV_BATCH,), generator=gen)]
        loss = F.mse_loss(head(prev[bi], act[bi]), nxt[bi])
        opt.zero_grad()
        loss.backward()
        opt.step()
        recent.append(float(loss.item()))
        if step % eval_every == 0:
            cur = _loss(ev_idx)
            curve_step.append(step)
            curve_eval.append(cur)
            curve_train.append(float(np.mean(recent)))
            curve_lr.append(lr)
            recent = []
            # The plateau counter resets only on a > PLATEAU_REL_IMPROVEMENT
            # gain over the best-so-far; the CHECKPOINT tracks the true best
            # (any improvement), so the head returned is the curve minimum.
            if cur < best * (1.0 - PLATEAU_REL_IMPROVEMENT):
                since = 0
            else:
                since += eval_every
            if cur < best:
                best, best_step = cur, step
                best_state = copy.deepcopy(head.state_dict())
            if since >= window:
                lr *= LR_DECAY_FACTOR
                since = 0
                decay_steps.append(step)
                head.load_state_dict(best_state)
                for g in opt.param_groups:
                    g["lr"] = lr
                if lr < MIN_LR:
                    exhausted = True
                    break
    head.load_state_dict(best_state)
    best_train = _loss(tr_idx)
    reachable = best <= persistence_eval_loss
    return {
        "converged": bool(exhausted and reachable),
        "schedule_exhausted": bool(exhausted),
        "cap_hit": bool(not exhausted),
        "reachable_zero_delta_point": bool(reachable),
        "stop_step": int(step), "best_step": int(best_step),
        "n_lr_decays": len(decay_steps), "decay_steps": decay_steps,
        "final_lr": lr,
        "n_train": int(tr_idx.shape[0]), "n_eval": int(ev_idx.shape[0]),
        "curve_step": curve_step, "curve_eval_loss": curve_eval,
        "curve_train_loss": curve_train, "curve_lr": curve_lr,
        "final_eval_loss": best,               # best checkpoint (the head returned)
        "persistence_eval_loss": persistence_eval_loss,
        "eval_loss_over_persistence": (best / persistence_eval_loss
                                       if persistence_eval_loss > 0 else float("nan")),
        "best_train_loss": best_train,
        "persistence_train_loss": persistence_train_loss,
        "train_loss_over_persistence": (best_train / persistence_train_loss
                                        if persistence_train_loss > 0 else float("nan")),
    }


def _detach_nonleaf(obj: Any, seen: Optional[set] = None, depth: int = 0) -> Any:
    """Replace every NON-LEAF tensor reachable from `obj` with its detached copy,
    in place, so the post-P1 agent can be deep-copied (torch refuses to deepcopy
    a tensor that carries a grad_fn). Training is over and every evaluation runs
    under torch.no_grad, so this is numerically a no-op: the values are unchanged,
    only the autograd history is dropped. Parameters are leaves and untouched."""
    if seen is None:
        seen = set()
    if depth > 12 or id(obj) in seen:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.detach() if obj.grad_fn is not None else obj
    if isinstance(obj, (str, bytes, int, float, bool, type(None))):
        return obj
    seen.add(id(obj))
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            obj[k] = _detach_nonleaf(obj[k], seen, depth + 1)
        return obj
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = _detach_nonleaf(obj[i], seen, depth + 1)
        return obj
    if isinstance(obj, tuple):
        new = tuple(_detach_nonleaf(x, seen, depth + 1) for x in obj)
        if all(a is b for a, b in zip(new, obj)):
            return obj          # nothing detached: keep the original object
        return type(obj)(*new) if hasattr(obj, "_fields") else new
    if isinstance(obj, torch.nn.Module):
        for m in obj.modules():
            if id(m) in seen and m is not obj:
                continue
            seen.add(id(m))
            for k, v in list(vars(m).items()):
                if k in ("_parameters", "_modules"):
                    continue
                vars(m)[k] = _detach_nonleaf(v, seen, depth + 1)
        return obj
    # Plain objects: only REE's own classes (agent sub-objects such as the
    # residue field or goal state). Never walk into stdlib / third-party state.
    if not str(type(obj).__module__).startswith("ree_core"):
        return obj
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        for k in list(d.keys()):
            d[k] = _detach_nonleaf(d[k], seen, depth + 1)
    return obj


def _eval_rollout(agent0, env0, head_state: Dict[str, torch.Tensor], epsilon: float,
                  seed: int, budget: int) -> Dict[str, Any]:
    # hippocampal._rng defaults to the `random` MODULE itself (the global
    # stream; ree_core/hippocampal/module.py:248). The copy must SHARE it, as the
    # original does -- and every rollout re-seeds the global stream below.
    memo = {id(random): random, id(np.random): np.random}
    agent = copy.deepcopy(agent0, memo)
    env = copy.deepcopy(env0, dict(memo))
    agent.e2_harm_a.load_state_dict(head_state)
    _seed_all(seed + 100003)   # identical stream for every rollout of this seed
    probe = FreshSelectProbe(B.FRESH_SELECT_NAMESPACE)
    counter = FreshSelectCounter()
    preds: List[torch.Tensor] = []
    tgts: List[torch.Tensor] = []
    prevs: List[torch.Tensor] = []
    acts: List[torch.Tensor] = []
    total = 0
    n_eps = 0
    n_not_fresh = 0
    with torch.no_grad():
        while total < budget:
            agent.reset()
            _obs, od = env.reset()
            counter.flush()
            n_eps += 1
            prev_zha = None
            prev_action = None
            while total < budget:
                body, world, harm, harm_a, hh = B._obs_tensors(od)
                latent = agent.sense(obs_body=body, obs_world=world, obs_harm=harm,
                                     obs_harm_a=harm_a, obs_harm_history=hh)
                ticks = agent.clock.advance()
                e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick")
                            else torch.zeros(1, B.WORLD_DIM, device=agent.device))
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                with probe.watch(agent) as fresh:
                    action = agent.select_action(candidates, ticks)
                is_fresh = bool(fresh)
                counter.record(is_fresh)
                if epsilon > 0.0 and random.random() < epsilon:
                    ai = random.randint(0, env.action_dim - 1)
                    action = torch.zeros(1, env.action_dim, device=agent.device)
                    action[0, ai] = 1.0
                if not is_fresh:
                    n_not_fresh += 1
                elif (prev_zha is not None and prev_action is not None
                        and latent.z_harm_a is not None):
                    z_pred = agent.e2_harm_a(prev_zha, prev_action)
                    preds.append(z_pred.detach().cpu())
                    tgts.append(latent.z_harm_a.detach().cpu())
                    prevs.append(prev_zha.detach().cpu())
                    acts.append(prev_action.detach().cpu())
                B._drive_valence_write_paths(agent, body)
                _obs, harm_signal, done, _info, od = env.step(
                    int(action.argmax(dim=-1).item()))
                agent.update_residue(float(harm_signal) if harm_signal is not None else 0.0)
                prev_zha = (latent.z_harm_a.detach().clone()
                            if latent.z_harm_a is not None else None)
                prev_action = action.detach()
                total += 1
                if done:
                    break
    counter.flush()
    _ZG.observe(agent)
    cat = (lambda xs, d: torch.cat(xs) if xs else torch.zeros(0, d))
    return {
        "pred": cat(preds, B.HARM_A_DIM), "target": cat(tgts, B.HARM_A_DIM),
        "prev": cat(prevs, B.HARM_A_DIM), "act": cat(acts, env.action_dim),
        "n_steps": total, "n_episodes": n_eps, "n_latched_ticks": n_not_fresh,
    }


def _verdict_dict(v) -> Dict[str, Any]:
    d = v.to_dict()
    d.pop("extra", None)
    return d


# --------------------------------------------------------------------------- #
# One seed = one cell (both arms share the collection, so they share a cell)   #
# --------------------------------------------------------------------------- #

def _run_seed(seed: int, dry_run: bool) -> List[Dict[str, Any]]:
    print("Seed %d Condition %s" % (seed, "ONLINE_B1+CONVERGED"), flush=True)
    p0 = 2 if dry_run else P0_EPS
    p1 = 2 if dry_run else P1_EPS
    spe = 12 if dry_run else STEPS_PER_EPISODE
    budget = 80 if dry_run else EVAL_STEP_BUDGET
    cap = 600 if dry_run else CONV_CAP
    window = 100 if dry_run else PLATEAU_WINDOW
    every = 20 if dry_run else EVAL_EVERY

    rows: List[Dict[str, Any]] = []
    with arm_cell(
        seed, config_slice=_config_slice(), script_path=Path(__file__),
        config_slice_declared=True, include_driver_script_in_hash=False,
        extra_substrate_paths=[DRIVER_1062A],
        extra_ineligible_reasons=(
            ["both_arms_share_one_collection_cell"]
            + (["dry_run"] if dry_run else [])),
    ) as cell:
        random.seed(seed)   # as 1062a, after arm_cell's full reset
        env = B._make_env(seed)
        agent = REEAgent(B.make_config(env))
        init_state = copy.deepcopy(agent.e2_harm_a.state_dict())

        col = _collect(agent, env, seed, p0, p1, spe)
        online_state = copy.deepcopy(agent.e2_harm_a.state_dict())
        # Training is over: drop autograd history so the post-P1 snapshot can
        # be deep-copied for each evaluation rollout (see _detach_nonleaf).
        _detach_nonleaf(agent)
        _detach_nonleaf(env)
        n_buf = int(col["prev"].shape[0])
        print("  [collect] seed=%d banked=%d online_updates=%d"
              % (seed, n_buf, len(col["online_losses"])), flush=True)

        # ARM_CONVERGED: same init, same data, offline to a held-out plateau.
        conv_head = copy.deepcopy(agent.e2_harm_a)
        conv_head.load_state_dict(init_state)
        gen = torch.Generator().manual_seed(seed + SPLIT_SEED_OFFSET)
        fit = _fit_to_plateau(conv_head, col["prev"], col["act"], col["next"], gen,
                              HELDOUT_FRACTION, cap, window, every)
        conv_state = copy.deepcopy(conv_head.state_dict())
        print("  [converge] seed=%d converged=%s exhausted=%s reachable=%s stop=%d "
              "decays=%d heldout/persistence=%.4f train/persistence=%.4f"
              % (seed, fit["converged"], fit["schedule_exhausted"],
                 fit["reachable_zero_delta_point"], fit["stop_step"],
                 fit["n_lr_decays"], fit["eval_loss_over_persistence"],
                 fit["train_loss_over_persistence"]), flush=True)

        # Online-arm held-out loss on the SAME held-out rows (comparability).
        with torch.no_grad():
            gen2 = torch.Generator().manual_seed(seed + SPLIT_SEED_OFFSET)
            perm = torch.randperm(n_buf, generator=gen2)
            ho = perm[:int(round(HELDOUT_FRACTION * n_buf))]
            on_head = copy.deepcopy(agent.e2_harm_a)
            on_head.load_state_dict(online_state)
            online_heldout_loss = (float(F.mse_loss(
                on_head(col["prev"][ho], col["act"][ho]), col["next"][ho]).item())
                if int(ho.shape[0]) > 0 else float("nan"))
            persistence_heldout_loss = (float(F.mse_loss(
                col["prev"][ho], col["next"][ho]).item())
                if int(ho.shape[0]) > 0 else float("nan"))

        other = {ARM_ONLINE: conv_state, ARM_CONV: online_state}
        states = {ARM_ONLINE: online_state, ARM_CONV: conv_state}
        for arm in (ARM_ONLINE, ARM_CONV):
            row: Dict[str, Any] = {
                "arm_id": arm, "seed": seed, "cell_id": "%s/seed%d" % (arm, seed),
                "n_banked_transitions": n_buf,
            }
            if arm == ARM_ONLINE:
                row["training"] = {
                    "mode": "online_batch1", "n_updates": len(col["online_losses"]),
                    "loss_curve_per_update": col["online_losses"],
                    "loss_curve_p1_episode_mean": col["online_ep_mean"],
                    # IN-SAMPLE for this arm: the online head trained on every P1
                    # transition, including the rows the converged arm holds out.
                    "in_sample_loss_on_converged_heldout_rows": online_heldout_loss,
                }
            else:
                row["training"] = dict(fit, mode="offline_minibatch_plateau",
                                       batch=CONV_BATCH, lr=CONV_LR)
            row["persistence_heldout_loss"] = persistence_heldout_loss
            for eps in EVAL_EPSILONS:
                tag = "eps%03d" % int(round(eps * 100))
                ro = _eval_rollout(agent, env, states[arm], eps, seed, budget)
                v = persistence_verdict(ro["pred"], ro["target"], ro["prev"])
                print("  [eval] %s seed=%d %s %s" % (arm, seed, tag,
                                                    format_verdict(v)), flush=True)
                # Cross-score: the OTHER head on these same rows (non-gating).
                xh = copy.deepcopy(agent.e2_harm_a)
                xh.load_state_dict(other[arm])
                with torch.no_grad():
                    xp = (xh(ro["prev"], ro["act"]) if ro["prev"].shape[0] > 0
                          else ro["prev"])
                xv = persistence_verdict(xp, ro["target"], ro["prev"])
                ev: Dict[str, Any] = {
                    "epsilon": eps, "n_steps": ro["n_steps"],
                    "n_episodes": ro["n_episodes"],
                    "n_latched_ticks": ro["n_latched_ticks"],
                    "verdict": _verdict_dict(v),
                    "cross_scored_other_head": _verdict_dict(xv),
                }
                # In-sample best-fit bound (converged arm only, non-gating).
                if arm == ARM_CONV and ro["prev"].shape[0] >= MIN_ROWS:
                    # Twin: the SAME rows with the per-row target delta permuted
                    # across rows, so (z, a) carries no information about it by
                    # construction. The bound is read as "learnable" only when the
                    # real fit separates from this twin (red-team F1c).
                    pg = torch.Generator().manual_seed(seed + PERMUTE_SEED_OFFSET)
                    pidx = torch.randperm(int(ro["prev"].shape[0]), generator=pg)
                    t_perm = ro["prev"] + (ro["target"] - ro["prev"])[pidx]
                    bound: Dict[str, Any] = {}
                    for bname, tgt in (("real", ro["target"]), ("permuted_delta", t_perm)):
                        oh = copy.deepcopy(agent.e2_harm_a)
                        oh.load_state_dict(init_state)
                        og = torch.Generator().manual_seed(seed + SPLIT_SEED_OFFSET + 1)
                        ofit = _fit_to_plateau(oh, ro["prev"], ro["act"], tgt,
                                               og, 0.0, cap, window, every)
                        with torch.no_grad():
                            op = oh(ro["prev"], ro["act"])
                        ov = persistence_verdict(op, tgt, ro["prev"])
                        bound[bname] = {
                            "schedule_exhausted": ofit["schedule_exhausted"],
                            "stop_step": ofit["stop_step"],
                            "loss_over_persistence": ofit["eval_loss_over_persistence"],
                            "verdict": _verdict_dict(ov)}
                    rv, pv = bound["real"]["verdict"], bound["permuted_delta"]["verdict"]
                    bound["real_separates_from_permuted"] = bool(
                        rv.get("ci_low") is not None and pv.get("ci_high") is not None
                        and rv["ci_low"] > pv["ci_high"])
                    ev["in_sample_bound"] = bound
                row[tag] = ev
            cell.stamp(row)
            rows.append(row)

        rep = rows[0]["eps000"]["verdict"]["status"]
        cstat = rows[1]["eps010"]["verdict"]["status"]
        ok = (rep == "persistence_dominated" and fit["converged"]
              and cstat == "ready")
        print("verdict: %s" % ("PASS" if ok else "FAIL"), flush=True)
    return rows


# --------------------------------------------------------------------------- #
# Routing                                                                      #
# --------------------------------------------------------------------------- #

def _route(rows: List[Dict[str, Any]], seeds: List[int]) -> Dict[str, Any]:
    by = {(r["arm_id"], r["seed"]): r for r in rows}
    per_seed: Dict[str, Dict[str, Any]] = {}
    for s in seeds:
        on = by[(ARM_ONLINE, s)]
        cv = by[(ARM_CONV, s)]
        reproduced = on["eps000"]["verdict"]["status"] == "persistence_dominated"
        converged = bool(cv["training"]["converged"])
        per_seed[str(s)] = {
            "reproduced": reproduced, "converged": converged,
            "eligible": reproduced and converged,
            "converged_eps010_status": cv["eps010"]["verdict"]["status"],
            "converged_eps010_d": cv["eps010"]["verdict"]["relative_skill"],
            "converged_eps000_status": cv["eps000"]["verdict"]["status"],
            "online_eps000_d": on["eps000"]["verdict"]["relative_skill"],
            "online_eps010_status": on["eps010"]["verdict"]["status"],
        }
    n_rep = sum(1 for v in per_seed.values() if v["reproduced"])
    n_conv = sum(1 for v in per_seed.values() if v["converged"])
    n_elig = sum(1 for v in per_seed.values() if v["eligible"])

    def _count(st: str) -> int:
        return sum(1 for v in per_seed.values()
                   if v["eligible"] and v["converged_eps010_status"] == st)

    n_ready = _count("ready")
    n_dom = _count("persistence_dominated")
    n_cd = _count("cannot_determine")
    if n_rep < SEEDS_REQUIRED:
        label = "substrate_not_ready_requeue"
        why = ("reproduction control failed: online head persistence_dominated at eps 0.0 "
               "on %d/%d seeds, quorum needs >= %d (non-attributable: the 1062a defect "
               "did not reproduce)" % (n_rep, len(seeds), SEEDS_REQUIRED))
    elif n_elig < SEEDS_REQUIRED:
        label = "substrate_not_ready_requeue"
        why = ("NOT_CONVERGED -- an INSTRUMENT outcome, NOT a structural verdict: "
               "fewer than %d eligible (reproduced AND converged) seeds: %d. The "
               "decayed optimiser did not reach the zero-delta point the residual "
               "head can represent (or hit the cap). This says nothing about the "
               "harm head's capacity, the z_harm_a representation or the PE source; "
               "read the recorded per-seed eval_loss_over_persistence and the "
               "online-vs-converged gap instead." % (SEEDS_REQUIRED, n_elig))
    elif n_ready >= SEEDS_REQUIRED:
        label = "h_undertrained_confirmed"
        why = "converged head ready on %d eligible seeds" % n_ready
    elif n_dom >= SEEDS_REQUIRED:
        label = "h_undertrained_refuted_declared_null"
        why = "converged head still persistence_dominated on %d eligible seeds" % n_dom
    elif n_cd >= SEEDS_REQUIRED:
        label = "active_error_removed_AMBIGUOUS"
        why = ("converged head cannot_determine on %d eligible seeds -- aliased "
               "across under-training, representation ceiling and PE-source; NOT "
               "a ceiling verdict" % n_cd)
    else:
        label = "inconclusive_no_quorum"
        why = "no status reached the quorum among eligible seeds"
    return {"label": label, "why": why, "per_seed": per_seed,
            "n_reproduced": n_rep, "n_converged": n_conv, "n_eligible": n_elig,
            "n_ready": n_ready, "n_persistence_dominated": n_dom,
            "n_cannot_determine": n_cd}


def run_experiment(dry_run: bool) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    canary = check_canary()
    seeds = SEEDS[:1] if dry_run else SEEDS
    rows: List[Dict[str, Any]] = []
    for s in seeds:
        rows.extend(_run_seed(s, dry_run))
    routed = _route(rows, seeds)
    label = routed["label"]
    outcome = "PASS" if label == "h_undertrained_confirmed" else "FAIL"

    def _worst(key_fn) -> float:
        vals = [key_fn(r) for r in rows]
        vals = [float(v) for v in vals if v is not None]
        return min(vals) if vals else 0.0

    worst_rows = _worst(lambda r: min(r["eps010"]["verdict"]["n_rows"],
                                      r["eps000"]["verdict"]["n_rows"]))
    preconditions = [
        {"name": "eligible_seeds_reproduced_and_converged",
         "description": "seeds that are BOTH reproduced (control persistence_dominated at "
                        "eps 0.0) AND converged -- the count the routing actually gates on",
         "measured": float(routed["n_eligible"]), "threshold": float(SEEDS_REQUIRED),
         "direction": "lower", "met": routed["n_eligible"] >= SEEDS_REQUIRED},
        {"name": "instrument_canary_reproduces_1062a",
         "description": "persistence_skill_gate.check_canary: the shipped verdict path "
                        "classifies the 6 landed 1062a cells persistence_dominated",
         "measured": 1.0 if canary.get("ok") else 0.0, "threshold": 1.0,
         "control": "pinned 1062a manifest cells (known persistence_dominated)",
         "direction": "lower", "met": bool(canary.get("ok"))},
        {"name": "reproduction_control_persistence_dominated_seeds",
         "description": "ARM_ONLINE_B1 (1062a's P1 exactly) is persistence_dominated at "
                        "eps 0.0 -- the defect reproduces on today's substrate",
         "measured": float(routed["n_reproduced"]),
         "threshold": float(SEEDS_REQUIRED),
         "control": "known-negative: 1062a measured persistence_dominated 3/3 stationary",
         "direction": "lower", "met": routed["n_reproduced"] >= SEEDS_REQUIRED},
        {"name": "converged_arm_reached_plateau_seeds",
         "description": "ARM_CONVERGED: lr-halving schedule exhausted (lr < %.2e) before "
                        "the %d-step cap AND best held-out loss <= persistence on the "
                        "same held-out rows (reachability of the zero-delta point)"
                        % (MIN_LR, CONV_CAP),
         "measured": float(routed["n_converged"]), "threshold": float(SEEDS_REQUIRED),
         "direction": "lower", "met": routed["n_converged"] >= SEEDS_REQUIRED},
        {"name": "eval_battery_rows_worst_cell",
         "description": "fewest paired rows in any evaluation rollout (instrument MIN_ROWS)",
         "measured": worst_rows, "threshold": float(MIN_ROWS),
         "direction": "lower", "met": worst_rows >= MIN_ROWS},
    ]
    criteria = [
        {"name": "C1_converged_head_ready_eps_matched", "load_bearing": True,
         "measured": float(routed["n_ready"]), "threshold": float(SEEDS_REQUIRED),
         "description": "eligible seeds where ARM_CONVERGED @ eps 0.1 is ready "
                        "(paired-bootstrap CI on d entirely > 0)",
         "passed": label == "h_undertrained_confirmed"},
        {"name": "N1_declared_null_converged_head_persistence_dominated",
         "load_bearing": False,
         "measured": float(routed["n_persistence_dominated"]),
         "threshold": float(SEEDS_REQUIRED),
         "description": "eligible seeds where the converged head is STILL "
                        "persistence_dominated (declared null: under-training refuted)",
         "passed": label == "h_undertrained_refuted_declared_null"},
        {"name": "A1_aliased_cannot_determine", "load_bearing": False,
         "measured": float(routed["n_cannot_determine"]),
         "threshold": float(SEEDS_REQUIRED),
         "description": "eligible seeds where the converged head is cannot_determine "
                        "(AMBIGUOUS: aliased across three hypotheses)",
         "passed": label == "active_error_removed_AMBIGUOUS"},
    ]
    combination_rule = (
        "Preconditions first: reproduction control (ARM_ONLINE_B1 @ eps 0.0 "
        "persistence_dominated) on >= 2/3 seeds, else substrate_not_ready_requeue; "
        ">= 2 ELIGIBLE (reproduced AND converged) seeds, else "
        "substrate_not_ready_requeue. Then on ARM_CONVERGED @ eps 0.1 among eligible "
        "seeds, first quorum (>= 2) wins in the order ready -> "
        "h_undertrained_confirmed (PASS); persistence_dominated -> "
        "h_undertrained_refuted_declared_null (FAIL); cannot_determine -> "
        "active_error_removed_AMBIGUOUS (FAIL, not a ceiling verdict); else "
        "inconclusive_no_quorum (FAIL). Only C1 is load-bearing for PASS.")
    all_pre_met = all(p["met"] for p in preconditions)
    ds = [r[t]["verdict"]["relative_skill"] for r in rows for t in ("eps010", "eps000")]
    ds = [round(float(x), 9) for x in ds if x is not None and math.isfinite(float(x))]
    non_degenerate = all_pre_met and len(set(ds)) > 1

    flat: Dict[str, float] = {
        "n_seeds": float(len(seeds)),
        "n_reproduced": float(routed["n_reproduced"]),
        "n_converged": float(routed["n_converged"]),
        "n_eligible": float(routed["n_eligible"]),
        "n_ready": float(routed["n_ready"]),
        "n_persistence_dominated": float(routed["n_persistence_dominated"]),
        "n_cannot_determine": float(routed["n_cannot_determine"]),
        "canary_ok": 1.0 if canary.get("ok") else 0.0,
        "all_preconditions_met": 1.0 if all_pre_met else 0.0,
        "worst_eval_rows": float(worst_rows),
    }
    for r in rows:
        a = "on" if r["arm_id"] == ARM_ONLINE else "conv"
        for t in ("eps010", "eps000"):
            v = r[t]["verdict"]
            for k, key in (("d", "relative_skill"), ("skill", "skill"),
                           ("model_r2", "model_r2"), ("persistence_r2", "persistence_r2")):
                x = v.get(key)
                if x is not None and math.isfinite(float(x)):
                    flat["%s_%s_s%d_%s" % (a, t, r["seed"], k)] = float(x)
        if a == "conv":
            flat["conv_s%d_stop_step" % r["seed"]] = float(r["training"]["stop_step"])
            flat["conv_s%d_converged" % r["seed"]] = 1.0 if r["training"]["converged"] else 0.0
    flat = {k: v for k, v in flat.items() if v is not None and math.isfinite(v)}

    manifest: Dict[str, Any] = {
        "run_id": "%s_%s_v3" % (EXPERIMENT_TYPE, time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())),
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "bears_on": BEARS_ON,
        "validates_substrate": VALIDATES_SUBSTRATE,
        "evidence_direction": "non_contributory",
        "outcome": outcome,
        "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "sleep_driver_pattern": "not_applicable",
        "decision_record": "rec-20260923-533fb67e (Option A)",
        "red_team": RED_TEAM_VERDICT,
        "base_driver": {"path": str(DRIVER_1062A.name),
                        "sha256": _sha256(DRIVER_1062A)},
        "non_degenerate": bool(non_degenerate),
        "degeneracy_reason": (None if non_degenerate else
                              "a precondition is unmet or d is identical in every rollout"),
        "readout": flat,
        "arm_results": rows,
        "routing": routed,
        "canary": canary,
        "interpretation": {
            "label": label,
            "why": routed["why"],
            "combination_rule": combination_rule,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": {
                c["name"]: bool(non_degenerate) for c in criteria},
        },
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow"},
    }
    return manifest, t0


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--dry-run", action="store_true")
    _args = _ap.parse_args()

    _manifest, _t0 = run_experiment(dry_run=_args.dry_run)
    _out_path = write_flat_manifest(
        _manifest, dry_run=_args.dry_run, config=_config_slice(),
        seeds=(SEEDS[:1] if _args.dry_run else SEEDS), script_path=Path(__file__),
        started_at=_t0, z_goal_stream_stats=_ZG.stats())
    print("[%s] outcome=%s label=%s" % (EXPERIMENT_TYPE, _manifest["outcome"],
                                         _manifest["interpretation"]["label"]), flush=True)
    _o = str(_manifest["outcome"]).upper()
    emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                 manifest_path=_out_path, dry_run=_args.dry_run)
