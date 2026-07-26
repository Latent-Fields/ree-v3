"""V3-EXQ-817a -- SD-080 world-effect-grounding CAPABILITY falsifier.

Successor to V3-EXQ-817. Same scientific question: if E2.action_object_head O
were GROUNDED in world-effects (not a frozen random projection, SD-080), would the
hippocampal CEM -- which searches and refits in O -- plan better (SD-004)? The
SD-080 frozen-projection defect is already CONFIRMED by V3-EXQ-809; this is the
orthogonal CAPABILITY test (does fixing O change BEHAVIOUR).

WHY 817a (what the 817 autopsy found -- failure_autopsy_batch-793a-817-819_2026-07-26).
817's self-route "substrate_not_ready_requeue" was a MISNOMER: the head DID train
(M0_action_object_head_param_delta_l2_from_frozen ~1.3-1.47; grounding loss
converged ~1e-5), but the delta-prediction objective did NOT induce the targeted
geometry -- M5 r2 fell only 0.994->0.982 (needed <=0.95) and M6 within-pair
spearman rose only to 0.085 (needed >=0.15), with high inter-seed variance
(0.011/0.251/0.088; seed-1 alone cleared the floor). grounding_took=false blocked
the behavioural read -> non_contributory. Two faults, two fixes -- and this script's
own pre-queue calibration (2026-07-26, seeds 0/1) drove the final design:

  FIX 1 -- STRONGER / DIFFERENT grounding objective (calibration-selected). 817
  trained o_t -> TRAINABLE readout -> z_world delta (MSE), letting o hide the signal
  in a low-variance subspace the readout amplified; M5/M6 barely moved. Two further
  objectives were CALIBRATED and rejected: (b) a relational/soft-contrastive metric
  loss is SCALE-INVARIANT, so it reshapes only o's coarse (action-cluster) DIRECTION
  and cannot grow the state-dependent RESIDUAL MAGNITUDE M5 measures (head_delta ~2.0
  yet M5 ~0.987, M6 ~0.03); (c) regressing o onto the DENSE ground-truth consequence
  (dpos, harm, obs) DID move M5 (to ~0.76-0.91) but not M6, because that target is
  only ~20% predictable from z_world (MSE stuck ~0.8). The SELECTED objective (train_
  head_grounding): regress o_t, with NO trainable readout, onto a fixed near-isometric
  projection of the REALISED next world-state z_world_{t+1} -- the achievable latent
  world-effect world_forward itself predicts. It is LEARNABLE (world_forward solves
  the same regression) and magnitude-bearing: calibration gave ARM_1 M5 0.70-0.72
  (from a frozen ~0.99) with a real fit (MSE 0.64-0.77), while the SHUFFLED control
  stayed ~0.99 (unfittable). This is the STRONGEST (oracle-quality) world-effect
  grounding: a behavioural null under it means SD-004's efficiency rationale is not
  capability-load-bearing.

  FIX 2 -- RECALIBRATED gate, OFF the noise-dominated M6. The autopsy flagged the
  M6 floor as possibly mis-set; calibration showed the deeper truth: M6 (within-pair
  consequence spearman) is NOT a usable gate on the 16-dim ao head. The FROZEN ao
  head's M6 ranges -0.18..+0.13 by seed and world_forward's OWN M6 is 0.07-0.22 --
  signal is buried in seed-to-seed noise, and even the SHUFFLED control hit M6=0.21
  on one seed. So the gate reads the ROBUST, low-noise manipulation signals instead:
  (1) ARM_1 fits the world-effect target (grounding MSE <= GATE_FIT_CEIL; else
  z_world is under-informative -> SUBSTRATE label, not objective failure); (2) ARM_1
  becomes materially state-dependent (M5 <= GATE_M5_CEIL AND paired ARM_0-ARM_1 drop
  >= GATE_M5_PAIRED_DROP); (3) TRUE-effect grounding is state-dependent while the
  SHUFFLED control is not (ARM_2-ARM_1 M5 margin >= GATE_STATE_DEP_MARGIN -> content,
  not gradient traffic). M5/M6 (ao + world_forward, all arms, per seed) are still
  RECORDED richly for governance; M6 is just not gated. This keeps the "the self-route
  is a hypothesis, not a verdict" discipline (V3-EXQ-642): a poor world-effect FIT
  self-routes to a distinct SUBSTRATE label, separable from an objective failure.

DESIGN (unchanged from 817). 3 arms x matched seeds, CausalGridWorldV2,
EXQ-003-comparable behavioural DVs. All arms share ONE P0 substrate per seed
(P0 trained once, deepcopied into 3 arms), so the only difference across arms is
the P1 head treatment -- isolating O's content as the manipulation.

  ARM_0  baseline           : P0 only; action_object_head frozen (SD-080 defect).
  ARM_1  world-effect-grounded: P0 + P1 regresses action_object_head so o_t is a
                               faithful low-dim code of the REALISED next world-state
                               z_world_{t+1} (the world-effect of the taken action);
                               encoder frozen, on .detach()ed latents.
  ARM_2  content control     : P0 + P1 identical to ARM_1 but the next-state targets
                               are SHUFFLED across the set. Same objective + gradient
                               traffic, NO real world-effect content. Note the shuffled
                               target is largely UNFITTABLE, so ARM_2's O stays close to
                               frozen (M5 ~0.99) -- it controls for gradient traffic;
                               the primary behavioural comparison is ARM_1 vs ARM_0.

PRIMARY GATE (substrate-side, must fire first; ARM_1). Robust, low-noise signals:
  (1) worldeffect_groundable   : ARM_1 median grounding MSE <= GATE_FIT_CEIL (else
                                 z_world under-informative -> SUBSTRATE label);
  (2) state_dependence_acquired: ARM_1 median M5 <= GATE_M5_CEIL AND paired ARM_0-ARM_1
                                 M5 drop >= GATE_M5_PAIRED_DROP;
  (3) content_discriminated    : ARM_2-ARM_1 M5 margin >= GATE_STATE_DEP_MARGIN.
  If the gate does not fire the behavioural read is uninterpretable -> self-route
  (worldeffect_ungroundable_from_zworld OR grounding_objective_ineffective_requeue),
  non_contributory, non_degenerate=false (scoring-excluded). M6 is recorded, not gated.

SECONDARY / LOAD-BEARING (behavioural, only interpreted if the gate fires).
harm_rate and survival under TERRAIN planning (hippocampal.propose_trajectories,
the CEM in O -- the DVs SD-004's EXQ-003/EXQ-114 PASS rests on):
  (a) DEFECT CONFIRMED LOAD-BEARING -> outcome PASS: ARM_1 beats ARM_0 AND ARM_2
      behaviourally. O's semantic grounding is load-bearing and was absent;
      SD-080 promotes, SD-004's mechanism is repaired.
  (b) DEFECT REAL BUT NOT LOAD-BEARING -> outcome FAIL (informative, not a null):
      ARM_1 moves the substrate DVs (gate fired) but NOT behaviour. SD-004's
      "semantic grounding / planning-horizon" efficiency rationale is demoted;
      SD-080 stays candidate as a correctness-not-capability finding.

DV-SYMMETRY DECLARATION (skill-mandated, per arm). DV = harm_rate/survival under
the TERRAIN CEM. Manipulation = action_object_head content. o_t enters via the
CEM's per-candidate action-object sequences -> E3 scoring -> selected action ->
env transition -> harm. A change in head content changes o_t NON-uniformly across
(state, action), so it is neither a broadcast additive constant (invisible to
argmax/softmax selection), nor a monotone rescaling (preserves selection rank),
nor a permutation of interchangeable units. harm_rate/survival are therefore NOT
invariant under the head-content manipulation for ARM_1 (whose O is substantially
rewritten toward the world-effect: calibration head_delta ~17, M5 0.99->0.70). A null
ARM_1-vs-ARM_0 behavioural result is thus a MEASURED null (reading (b)), not an
arithmetic identity. (ARM_2's shuffled target is largely unfittable so its O barely
moves; it is a gradient-traffic control, not the primary behavioural comparator.)

EXPERIMENT_PURPOSE = evidence (both directions pre-declared and scientifically
load-bearing). The substrate-side gate is recorded as interpretation.preconditions
and, when it fails, the run marks non_degenerate=false so it is scoring-excluded
rather than counted as a spurious behavioural weaken.

No sleep subsystem is used, so no SLEEP DRIVER line is required.

Provenance: substrate-DV measurement functions (m1/m2/m3/m5/m6, _spearman,
collect_states, consequence_vector, gather) are vendored VERBATIM from
REE_assembly/evidence/planning/action_object_invariance_spike_2026-07-22_probe.py
(session epic-burnell-995d28) so the ARM_1 gate reads the SAME statistics the spike
reported; behavioural harness modelled on experiments/v3_exq_003_sd004_action_objects.py.
supersedes: V3-EXQ-817 / v3_exq_817_sd080_consequence_grounding_falsifier.

ASCII-only output.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng
from experiments.pack_writer import write_flat_manifest

EXPERIMENT_TYPE = "v3_exq_817a_sd080_worldeffect_grounding_falsifier"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["SD-080", "SD-004"]
SUPERSEDES_RUN = "v3_exq_817_sd080_consequence_grounding_falsifier"

EVIDENCE_DIR = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"

# ------------------------------------------------------------------ config
WORLD_DIM = SELF_DIM = 32
ALPHA_WORLD = 0.9
ENV_SIZE = 6
N_STATES = 150          # substrate-DV state sample (M5 is stable here; M6 is diagnostic-only)
SEEDS = [0, 1, 2, 3, 4]
ARMS = ["ARM_0", "ARM_1", "ARM_2"]

WARMUP_EPISODES = 40
EVAL_EPISODES = 20
STEPS_PER_EPISODE = 200
# 817a: bigger set + larger batch + more steps -- a relational/contrastive loss
# needs richer pairwise structure per batch than a per-sample MSE.
GROUNDING_TRANSITIONS = 3000
GROUNDING_STEPS = 600
GROUNDING_BATCH = 128
GROUNDING_LR = 1e-3
WARMUP_LR = 1e-3

# ------------------------------------------------------ pre-registered thresholds
# Primary substrate-side readiness gate (ARM_1). RECALIBRATED off the noise-
# dominated absolute M6 floor (calibration 2026-07-26: the FROZEN ao head's M6
# ranges -0.18..+0.13 by seed, and world_forward's own M6 is 0.07-0.22 -- signal
# is buried in seed-to-seed noise, so M6>=0.15 is not a usable gate on the 16-dim
# ao head). The gate now reads the ROBUST, low-noise manipulation signals:
#   (1) ARM_1 becomes materially state-dependent (M5 r2 drops far below the frozen
#       ~0.99), (2) ARM_1 actually FITS the world-effect target (grounding MSE well
#       below the unit-variance no-fit level 1.0), and (3) TRUE-effect grounding is
#       state-dependent while the SHUFFLED control is NOT (paired M5 margin).
# M5/M6 (ao and world_forward, per seed, all arms) are still RECORDED richly for
# governance -- just not gated on the noisy M6.
GATE_M5_CEIL = 0.90             # ARM_1 median r2_explained_by_action_alone must DROP to <= this
GATE_M5_PAIRED_DROP = 0.10      # median(ARM_0 M5 - ARM_1 M5) must be >= this (grounding made O state-dependent)
GATE_FIT_CEIL = 0.85            # ARM_1 median grounding MSE must be <= this (target is unit-variance;
                                #   <=0.85 => O explains >=15% of the world-effect it was grounded to.
                                #   >0.85 => z_world cannot support world-effect grounding = SUBSTRATE issue)
GATE_STATE_DEP_MARGIN = 0.10    # median(ARM_2 M5 - ARM_1 M5) must be >= this (true-effect grounding is
                                #   state-dependent; the shuffled control is not -> content, not traffic)
# Secondary behavioural verdict (only read if the gate fires).
BEHAV_REL_MARGIN = 0.15         # ARM_1 harm_rate must be <= (1-margin) * comparator harm_rate
BEHAV_SEED_CONSISTENCY = 0.8    # fraction of seeds ARM_1 must beat ARM_0 in

# ============================================================================
# Vendored substrate-DV measurement functions (VERBATIM from the spike probe;
# see module docstring provenance). Kept byte-faithful so the gate reads the
# same statistics the spike reported.
# ============================================================================

def collect_states(agent, env, n_states):
    """Roll out; capture (z_world, env snapshot) at each visited state."""
    from _harness import StepHarness

    harness = StepHarness(agent, env, train_mode=False)
    out = []
    _, obs_dict = env.reset()
    agent.reset()
    harness.reset()
    steps = 0
    reset_every = max(5, n_states // 12)
    while len(out) < n_states and steps < n_states * 6:
        snapshot = copy.deepcopy(env)
        res = harness.step(obs_dict)
        z_world = res.latent.z_world.detach().clone()
        out.append((z_world, snapshot))
        obs_dict = res.next_obs_dict
        steps += 1
        if getattr(res, "done", False) or steps % reset_every == 0:
            _, obs_dict = env.reset()
            agent.reset()
            harness.reset()
    return out[:n_states]


def consequence_vector(snapshot, a, action_dim):
    """Ground-truth consequence of taking action a from this exact env state."""
    env2 = copy.deepcopy(snapshot)
    pos0 = (float(env2.agent_x), float(env2.agent_y))
    onehot = torch.zeros(action_dim)
    onehot[a] = 1.0
    flat, harm, done, info, obs_dict = env2.step(onehot)
    pos1 = (float(env2.agent_x), float(env2.agent_y))
    dpos = (pos1[0] - pos0[0], pos1[1] - pos0[1])

    dense = np.concatenate([
        np.asarray(dpos, dtype=np.float64),
        np.asarray([float(harm)], dtype=np.float64),
        np.asarray(flat, dtype=np.float64).ravel(),
    ])
    key = (dpos, str(info.get("transition_type", "")))
    return dense, key


def gather(agent, env, n_states):
    action_dim = env.action_dim
    states = collect_states(agent, env, n_states)
    AO, WF, CONS, KEYS, ZW = [], [], [], [], []
    with torch.no_grad():
        for z_world, snap in states:
            ZW.append(z_world[0].numpy())
            ao_s, wf_s, cons_s, key_s = [], [], [], []
            for a in range(action_dim):
                onehot = torch.zeros(1, action_dim)
                onehot[0, a] = 1.0
                ao_s.append(agent.e2.action_object(z_world, onehot)[0].numpy())
                wf_s.append(agent.e2.world_forward(z_world, onehot)[0].numpy())
                dense, key = consequence_vector(snap, a, action_dim)
                cons_s.append(dense)
                key_s.append(key)
            AO.append(np.stack(ao_s))
            WF.append(np.stack(wf_s))
            CONS.append(np.stack(cons_s))
            KEYS.append(key_s)
    return np.stack(AO), np.stack(WF), np.stack(CONS), KEYS, np.stack(ZW)


def m1_variance_decomposition(X):
    """X: [S, A, D]. Between-state vs within-state(action) variance."""
    per_state_mean = X.mean(axis=1, keepdims=True)
    between = per_state_mean[:, 0, :].var(axis=0).sum()
    within = (X - per_state_mean).var(axis=(0, 1)).sum()
    grand_mean = X.mean(axis=(0, 1))
    dev = X - per_state_mean
    return {
        "between_state_var": float(between),
        "within_state_action_var": float(within),
        "action_var_fraction": float(within / (within + between + 1e-30)),
        "mean_per_dim_action_std": float(dev.std(axis=(0, 1)).mean()),
        "grand_mean_norm": float(np.linalg.norm(grand_mean)),
        "mean_action_deviation_norm": float(np.linalg.norm(dev, axis=-1).mean()),
        "action_signal_to_offset_ratio": float(
            np.linalg.norm(dev, axis=-1).mean() / (np.linalg.norm(grand_mean) + 1e-30)
        ),
    }


def m2_linear_probe(X, seed=0):
    """Can a LINEAR probe recover the action class from state-centred ao?"""
    S, A, D = X.shape
    Xc = (X - X.mean(axis=1, keepdims=True)).reshape(S * A, D)
    y = np.tile(np.arange(A), S)
    sd = Xc.std(axis=0)
    Xc = Xc / np.where(sd > 1e-12, sd, 1.0)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(S)
    n_tr = int(S * 0.7)
    tr = np.concatenate([np.arange(i * A, (i + 1) * A) for i in idx[:n_tr]])
    te = np.concatenate([np.arange(i * A, (i + 1) * A) for i in idx[n_tr:]])

    Xt = torch.tensor(Xc[tr], dtype=torch.float32)
    yt = torch.tensor(y[tr], dtype=torch.long)
    Xe = torch.tensor(Xc[te], dtype=torch.float32)
    ye = torch.tensor(y[te], dtype=torch.long)
    W = torch.zeros(D, A, requires_grad=True)
    b = torch.zeros(A, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(600):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float(((Xe @ W + b).argmax(dim=1) == ye).float().mean())
    return {"linear_probe_test_acc": acc, "chance": 1.0 / A}


def _spearman(u, v):
    def rank(x):
        order = np.argsort(x)
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(x))
        return r
    ru, rv = rank(u), rank(v)
    ru -= ru.mean(); rv -= rv.mean()
    den = np.sqrt((ru ** 2).sum() * (rv ** 2).sum())
    return float((ru * rv).sum() / den) if den > 0 else 0.0


def m3_consequence_structure(X, CONS, KEYS):
    """DECISIVE trap: does embedding distance track consequence distance?

    Retained only as the documented spike trap (Sec 3.2); READ WITH M6, never
    alone -- pooled it gives a false positive because the same/different labels
    redistribute over a fixed distance matrix.
    """
    S, A, _ = X.shape
    d_emb, d_con, same_flag = [], [], []
    for s in range(S):
        for i in range(A):
            for j in range(i + 1, A):
                d_emb.append(np.linalg.norm(X[s, i] - X[s, j]))
                d_con.append(np.linalg.norm(CONS[s, i] - CONS[s, j]))
                same_flag.append(KEYS[s][i] == KEYS[s][j])
    d_emb = np.asarray(d_emb); d_con = np.asarray(d_con)
    same = np.asarray(same_flag, dtype=bool)
    diff = ~same

    out = {
        "n_pairs": int(len(d_emb)),
        "n_same_consequence_pairs": int(same.sum()),
        "n_diff_consequence_pairs": int(diff.sum()),
        "spearman_d_emb_vs_d_consequence": _spearman(d_emb, d_con),
    }
    if same.sum() > 0 and diff.sum() > 0:
        ms, md = float(d_emb[same].mean()), float(d_emb[diff].mean())
        pooled = float(d_emb.std() + 1e-30)
        out.update({
            "mean_d_emb_same_consequence": ms,
            "mean_d_emb_diff_consequence": md,
            "separation_ratio_diff_over_same": float(md / (ms + 1e-30)),
            "cohens_d_diff_minus_same": float((md - ms) / pooled),
        })
    return out


def m5_state_dependence(X):
    """Is the embedding a function of the ACTION ALONE? (control for M3)

    R2 = fraction of ao variance explained by a per-action-class mean lookup.
    High r2 (~0.99) = state-invariant (the SD-080 defect); grounding must DROP it.
    """
    S, A, D = X.shape
    per_action_mean = X.mean(axis=0)
    resid = X - per_action_mean[None, :, :]
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((X - X.mean(axis=(0, 1))) ** 2).sum())
    r2_action_only = 1.0 - ss_res / (ss_tot + 1e-30)

    dmats = []
    for s in range(S):
        dmats.append([np.linalg.norm(X[s, i] - X[s, j])
                      for i in range(A) for j in range(i + 1, A)])
    dmats = np.asarray(dmats)
    cv = dmats.std(axis=0) / (dmats.mean(axis=0) + 1e-30)
    return {
        "r2_explained_by_action_alone": float(r2_action_only),
        "pairdist_across_state_cv_mean": float(cv.mean()),
        "pairdist_across_state_cv_max": float(cv.max()),
    }


def m6_within_pair_consequence_corr(X, CONS):
    """M3 with action-pair identity partialled out (the decisive consequence read).

    For each FIXED action pair (i,j), correlate d_emb vs d_consequence ACROSS
    states. Baseline ~0; grounding must RAISE mean_within_pair_spearman.
    """
    S, A, _ = X.shape
    per_pair = {}
    for i in range(A):
        for j in range(i + 1, A):
            de = np.array([np.linalg.norm(X[s, i] - X[s, j]) for s in range(S)])
            dc = np.array([np.linalg.norm(CONS[s, i] - CONS[s, j]) for s in range(S)])
            per_pair["%d-%d" % (i, j)] = {
                "spearman": _spearman(de, dc),
                "d_emb_cv": float(de.std() / (de.mean() + 1e-30)),
            }
    vals = [v["spearman"] for v in per_pair.values()]
    return {
        "per_pair": per_pair,
        "mean_within_pair_spearman": float(np.mean(vals)),
        "max_abs_within_pair_spearman": float(np.max(np.abs(vals))),
        "mean_d_emb_cv_within_pair": float(
            np.mean([v["d_emb_cv"] for v in per_pair.values()])),
    }


# ============================================================================
# Substrate construction
# ============================================================================

def make_env(seed):
    return CausalGridWorldV2(size=ENV_SIZE, seed=seed)


def make_agent(env):
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=ALPHA_WORLD,
        world_dim=WORLD_DIM,
        self_dim=SELF_DIM,
    )
    return REEAgent(cfg)


# ============================================================================
# P0 -- shared substrate warmup (modelled on EXQ-003 _train_episodes).
# Trains E1 + E2(self + world); accumulates residue via update_residue on harm
# events (so the TERRAIN planner has a shaped terrain). action_object_head
# receives ZERO gradient (no loss touches it) -- this IS the frozen baseline.
# ============================================================================

def run_p0(agent, env, num_episodes, steps_per_episode, seed, progress_denom):
    agent.train()
    optimizer = optim.Adam(agent.parameters(), lr=WARMUP_LR)
    world_transition_buffer: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_events = 0
    total_harm = 0.0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_self_prev = None
        z_world_prev = None
        action_prev = None

        for step in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            if z_world_prev is not None and action_prev is not None:
                world_transition_buffer.append((
                    z_world_prev.detach(), action_prev.detach(), latent.z_world.detach(),
                ))
                if len(world_transition_buffer) > 500:
                    world_transition_buffer = world_transition_buffer[-500:]
            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            with torch.no_grad():
                e1_prior = torch.zeros(1, agent.config.latent.world_dim, device=agent.device)
                candidates = agent.hippocampal.propose_trajectories(
                    z_world=latent.z_world.detach(),
                    z_self=latent.z_self.detach(),
                    num_candidates=4,
                    e1_prior=e1_prior,
                )
                result = agent.e3.select(candidates, temperature=1.5)
                action = result.selected_action
                agent._last_action = action

            z_self_prev = latent.z_self.detach()
            z_world_prev = latent.z_world.detach()
            action_prev = action.detach()

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            if harm_signal < 0:
                harm_events += 1
                total_harm += abs(harm_signal)
                agent.update_residue(
                    harm_signal=harm_signal,
                    hypothesis_tag=False,
                    owned=(info["transition_type"] == "agent_caused_hazard"),
                )

            e1_loss = agent.compute_prediction_loss()
            e2_self_loss = agent.compute_e2_loss()
            e2_world_loss = e1_loss.new_zeros(())
            if len(world_transition_buffer) >= 4:
                n = min(16, len(world_transition_buffer))
                idxs = torch.randperm(len(world_transition_buffer))[:n].tolist()
                batch = [world_transition_buffer[i] for i in idxs]
                zw_t, acts, zw_t1 = zip(*batch)
                zw_t = torch.cat(zw_t, dim=0)
                acts = torch.cat(acts, dim=0)
                zw_t1 = torch.cat(zw_t1, dim=0)
                e2_world_loss = F.mse_loss(agent.e2.world_forward(zw_t, acts), zw_t1)

            loss = e1_loss + e2_self_loss + e2_world_loss
            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if done:
                break

        if (ep + 1) % 10 == 0 or ep == num_episodes - 1:
            print(f"  [train] p0 seed={seed} ep {ep + 1}/{progress_denom}  harm={total_harm:.2f}",
                  flush=True)

    return {"warmup_harm_events": harm_events, "warmup_total_harm": total_harm}


# ============================================================================
# P1 -- consequence grounding of action_object_head (encoder FROZEN).
#
# 817a objective: shape the DISTANCE GEOMETRY of o_t directly toward the true
# experienced world-effect (consequence) geometry, via a relational / soft-
# contrastive loss (NO linear readout -- that was 817's escape hatch into a
# low-variance subspace). Collect (z_world_t, a_t, consequence_t) transitions
# under the P0 agent; consequence_t is the SAME dense vector M3/M6 measure
# against (dpos, harm, next-obs), reconstructed from the realised step (a
# deterministic transition, so == consequence_vector for the taken action).
# ARM_2 SHUFFLES the consequence targets across the set (same params + gradient
# traffic + state-dependence, no real consequence structure). All latents are
# .detach()ed so no gradient reaches the encoder (phased P1).
# ============================================================================

def collect_grounding_transitions(agent, env, n_transitions, steps_per_episode, seed):
    """Collect (z_world_t, a_t, z_world_{t+1}) under the P0 agent.

    The grounding TARGET is the REALISED next world-state z_world_{t+1} -- the
    achievable latent world-effect world_forward itself predicts (and whose
    geometry tracks the dense ground-truth consequence M6 measures at ~0.2, the
    substrate ceiling). This is what makes the fit LEARNABLE from (z_world_t, a_t):
    world_forward solves exactly this regression, so the ao head can too. The
    dense ground-truth consequence vector (dpos, harm, obs) is only ~20%
    predictable from z_world (MSE stayed ~0.8 in calibration) -> using it as the
    training target failed to move M6; the realised next-state does not.
    """
    agent.eval()
    torch.manual_seed(seed + 7000)
    Zt: List[torch.Tensor] = []
    Aoh: List[torch.Tensor] = []
    Znext: List[torch.Tensor] = []
    steps = 0
    while len(Zt) < n_transitions and steps < n_transitions * 4:
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev = None
        action_prev = None
        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world = latent.z_world.detach()
                e1_prior = torch.zeros(1, agent.config.latent.world_dim, device=agent.device)
                candidates = agent.hippocampal.propose_trajectories(
                    z_world=z_world, z_self=latent.z_self.detach(),
                    num_candidates=4, e1_prior=e1_prior,
                )
                result = agent.e3.select(candidates, temperature=1.5)
                action = result.selected_action
                agent._last_action = action
            # Pair the PREVIOUS (z_world, action) with the REALISED next world-state
            # (this iteration's z_world), exactly like the Delta pairing in 817 but
            # keeping the absolute next-state rather than the difference.
            if z_world_prev is not None and action_prev is not None:
                Zt.append(z_world_prev.detach().clone())
                Aoh.append(action_prev.detach().clone())
                Znext.append(z_world.detach().clone())   # z_world_{t+1} = effect of action_prev
            z_world_prev = z_world
            action_prev = action.detach()
            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            steps += 1
            if done:
                break
    if not Zt:
        return None
    return (torch.cat(Zt[:n_transitions], dim=0),
            torch.cat(Aoh[:n_transitions], dim=0),
            torch.cat(Znext[:n_transitions], dim=0))


def _consequence_projection(c_dim, ao_dim, device):
    """Fixed, seed-stable near-isometric projection consequence_space -> ao_space.

    Orthonormal columns (QR of a fixed random matrix) so the projection preserves
    pairwise distances (Johnson-Lindenstrauss) -- the target ao code carries the
    consequence DISTANCE GEOMETRY that M6 reads. Fixed and SHARED across arms so
    ARM_1 and ARM_2 differ only in whether the consequence content is real or
    shuffled, never in the projection.
    """
    g = torch.Generator().manual_seed(4242)
    R = torch.randn(c_dim, max(ao_dim, c_dim), generator=g)[:, :ao_dim] if c_dim < ao_dim \
        else torch.randn(c_dim, ao_dim, generator=g)
    Q, _ = torch.linalg.qr(R)            # Q: [c_dim, ao_dim] with orthonormal columns
    return Q[:, :ao_dim].to(device)


def train_head_grounding(agent, transitions, shuffle_target, n_steps, batch, lr, seed, label,
                         progress_denom):
    """WORLD-EFFECT grounding: regress o_t onto a fixed near-isometric projection
    of the REALISED next world-state z_world_{t+1} (the achievable latent
    world-effect), by MSE, with NO trainable readout.

    Why THIS target/objective, and not the three that empirically FAILED on this
    substrate (all calibrated 2026-07-26):
      * 817's o_t -> TRAINABLE readout -> z_world delta (MSE): the free readout let
        o hide the signal in a low-variance subspace it then amplified; M5/M6
        (which read o's OWN variance geometry) barely moved (M6 ~0.085).
      * a relational/soft-contrastive loss is SCALE-INVARIANT, so it reshapes only
        o's coarse (action-cluster) DIRECTION and cannot grow the state-dependent
        RESIDUAL MAGNITUDE M5 measures (head_delta ~2.0 yet M5 ~0.987, M6 ~0.03).
      * regressing o onto the DENSE ground-truth consequence (dpos, harm, obs) is
        magnitude-bearing and DID move M5 (to ~0.76-0.91) but NOT M6 (~0.0): that
        target is only ~20% predictable from z_world (MSE stuck ~0.8), so the fitted
        part does not carry the within-action consequence geometry.
    The REALISED next world-state z_world_{t+1} is the target world_forward itself
    predicts (M6 ~0.2, the substrate ceiling) and is LEARNABLE from (z_world_t, a_t)
    -- world_forward solves exactly this regression, so the ao head can too. Pinning
    every dim of o_t to a fixed near-isometric projection of it (a) removes the
    hiding freedom, (b) is magnitude-bearing (grows state-dependence -> M5 drops),
    and (c) makes o a faithful low-dim code of the next world-state, so o inherits
    that state's distance geometry -> M6 rises toward world_forward's ceiling. This
    is the STRONGEST (oracle-quality) world-effect grounding: a behavioural null
    under it means SD-004's semantic-grounding efficiency rationale is not
    capability-load-bearing.

    Target scale standardised per-dim (unit variance) so o_t's scale stays
    controlled for the CEM; ARM_2 uses the SAME projection + scaling on SHUFFLED
    next-states (state-dependent structured code, no TRUE world-effect content) ->
    the control that separates "o is now a structured code" from "o is grounded in
    the TRUE world-effect". The DV metric (M6, measured against the dense
    consequence vector) is INDEPENDENT of this latent training target, so a rise in
    M6 is not teaching-to-the-test.
    """
    Zt, Aoh, Znext = transitions
    device = agent.device
    n = Zt.shape[0]

    # Standardise the next-state target per-dim (controlled code scale). Guard
    # constant dims.
    z_mean = Znext.mean(dim=0, keepdim=True)
    z_std = Znext.std(dim=0, keepdim=True)
    z_std = torch.where(z_std > 1e-8, z_std, torch.ones_like(z_std))
    Zstd = (Znext - z_mean) / z_std

    if shuffle_target:
        g = torch.Generator().manual_seed(seed + 9000)
        perm = torch.randperm(n, generator=g)
        Zstd = Zstd[perm].clone()   # scramble world-effect content vs (state, action)

    ao_dim = agent.e2.action_object_head[-1].out_features   # robust to config path
    src_dim = int(Zstd.shape[1])
    Q = _consequence_projection(src_dim, ao_dim, device)    # [src_dim, ao_dim]
    Target = Zstd.to(device) @ Q                            # [n, ao_dim] world-effect code
    # Per-dim unit-variance target -> controlled o scale (ARM_2 identical).
    t_mean = Target.mean(dim=0, keepdim=True)
    t_std = Target.std(dim=0, keepdim=True)
    t_std = torch.where(t_std > 1e-8, t_std, torch.ones_like(t_std))
    Target = (Target - t_mean) / t_std

    params = list(agent.e2.action_object_head.parameters())
    opt = optim.Adam(params, lr=lr)
    g2 = torch.Generator().manual_seed(seed + 11000)

    agent.train()
    final_loss = float("nan")
    first_loss = float("nan")
    for it in range(n_steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=g2)
        zt_b = Zt[idx].to(device).detach()
        a_b = Aoh[idx].to(device).detach()
        y_b = Target[idx].detach()
        o_t = agent.e2.action_object(zt_b, a_b)   # grad flows into action_object_head only
        loss = F.mse_loss(o_t, y_b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().item())
        if it == 0:
            first_loss = final_loss
        # NB: deliberately NOT the "ep N/M" pattern -- P0 owns the runner progress
        # denominator (episodes_per_run); a second denominator here would flap it.
        if (it + 1) % max(1, n_steps // 4) == 0 or it == n_steps - 1:
            print(f"  [ground] {label} step {it + 1}/{progress_denom}  mse={final_loss:.5f}",
                  flush=True)
    agent.eval()
    return {
        "grounding_objective": "worldeffect_nextstate_code_regression",
        "grounding_final_loss": final_loss,
        "grounding_first_loss": first_loss,
        "grounding_shuffled_target": bool(shuffle_target),
        "n_grounding_transitions": int(n),
        "worldeffect_target_dim": int(src_dim),
        "ao_code_dim": int(ao_dim),
    }


# ============================================================================
# P2a -- behavioural eval (TERRAIN planner only; modelled on EXQ-003 _eval_condition).
# ============================================================================

def eval_terrain(agent, env, num_episodes, steps_per_episode, seed):
    agent.eval()
    torch.manual_seed(seed + 2000)
    harm_events = 0
    resource_events = 0
    survival_steps: List[int] = []
    fatal_errors = 0
    try:
        for ep in range(num_episodes):
            flat_obs, obs_dict = env.reset()
            agent.reset()
            for step in range(steps_per_episode):
                obs_body = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world)
                    agent.clock.advance()
                    z_world = latent.z_world.detach()
                    z_self = latent.z_self.detach()
                    e1_prior = torch.zeros(1, agent.config.latent.world_dim, device=agent.device)
                    candidates = agent.hippocampal.propose_trajectories(
                        z_world=z_world, z_self=z_self, num_candidates=8, e1_prior=e1_prior,
                    )
                    result = agent.e3.select(candidates, temperature=1.5)
                    action = result.selected_action
                    agent._last_action = action
                flat_obs, harm_signal, done, info, obs_dict = env.step(action)
                if harm_signal < 0:
                    harm_events += 1
                    agent.update_residue(
                        harm_signal=harm_signal, hypothesis_tag=False,
                        owned=(info["transition_type"] == "agent_caused_hazard"),
                    )
                elif harm_signal > 0:
                    resource_events += 1
                if done:
                    break
            survival_steps.append(env.steps)
    except Exception:
        import traceback
        fatal_errors += 1
        print(f"  FATAL in eval seed={seed}:\n{traceback.format_exc()}", flush=True)

    total_steps = sum(survival_steps)
    harm_rate = harm_events / max(1, total_steps)
    mean_survival = float(sum(survival_steps) / max(1, len(survival_steps)))
    return {
        "harm_events": int(harm_events),
        "resource_events": int(resource_events),
        "total_eval_steps": int(total_steps),
        "harm_rate": float(harm_rate),
        "mean_survival_steps": float(mean_survival),
        "fatal_errors": int(fatal_errors),
    }


# ============================================================================
# P2b -- substrate DVs (vendored probe measures) on ao + world_forward reference.
# ============================================================================

def measure_substrate(agent, env, n_states, frozen_head_snapshot):
    AO, WF, CONS, KEYS, ZW = gather(agent, env, n_states)
    head_after = [p.detach().clone() for p in agent.e2.action_object_head.parameters()]
    head_delta = float(sum(((b - a) ** 2).sum()
                           for b, a in zip(frozen_head_snapshot, head_after)) ** 0.5)
    return {
        "M0_action_object_head_param_delta_l2_from_frozen": head_delta,
        "n_states": int(AO.shape[0]),
        "zworld_control": {
            "zworld_total_var_across_states": float(ZW.var(axis=0).sum()),
            "zworld_mean_per_dim_std": float(ZW.std(axis=0).mean()),
            "zworld_mean_norm": float(np.linalg.norm(ZW, axis=1).mean()),
        },
        "ao": {
            "M1": m1_variance_decomposition(AO),
            "M2": m2_linear_probe(AO),
            "M3": m3_consequence_structure(AO, CONS, KEYS),
            "M5": m5_state_dependence(AO),
            "M6": m6_within_pair_consequence_corr(AO, CONS),
        },
        "world_forward_reference": {
            "M1": m1_variance_decomposition(WF),
            "M5": m5_state_dependence(WF),
            "M6": m6_within_pair_consequence_corr(WF, CONS),
        },
    }


# ============================================================================
# Per-(seed, arm) cell.
# ============================================================================

def run_seed_arm(seed, arm, agent_p0, frozen_head_snapshot, env, args, config_slice):
    print(f"Seed {seed} Condition {arm}", flush=True)
    agent = copy.deepcopy(agent_p0)

    # Cell RNG reset (arm_cell) makes P1 + eval reproducible per cell; the P0
    # substrate is trained OUTSIDE this fingerprinted scope (shared deepcopy),
    # so these cells are marked reuse-ineligible (honest: not a pure function of
    # (config, seed) alone).
    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=Path(__file__),
        config_slice_declared=True,
        extra_ineligible_reasons=["p0_substrate_trained_outside_fingerprinted_cell_scope"],
    ) as cell:
        p1_info: Dict[str, Any] = {"grounding_applied": False}
        if arm in ("ARM_1", "ARM_2"):
            transitions = collect_grounding_transitions(
                agent, env, args.grounding_transitions, args.steps, seed)
            if transitions is None:
                p1_info = {"grounding_applied": False, "grounding_error": "no_transitions"}
            else:
                p1_info = train_head_grounding(
                    agent, transitions, shuffle_target=(arm == "ARM_2"),
                    n_steps=args.grounding_steps, batch=GROUNDING_BATCH, lr=GROUNDING_LR,
                    seed=seed, label=f"{arm}-ground seed={seed}",
                    progress_denom=args.grounding_steps)
                p1_info["grounding_applied"] = True

        substrate = measure_substrate(agent, env, args.n_states, frozen_head_snapshot)
        behav = eval_terrain(agent, env, args.eval_episodes, args.steps, seed)

        m5 = substrate["ao"]["M5"]["r2_explained_by_action_alone"]
        m6 = substrate["ao"]["M6"]["mean_within_pair_spearman"]
        wf_m5 = substrate["world_forward_reference"]["M5"]["r2_explained_by_action_alone"]
        wf_m6 = substrate["world_forward_reference"]["M6"]["mean_within_pair_spearman"]
        print(f"  [{arm} seed={seed}] M5_r2={m5:.4f} M6_spearman={m6:.4f} "
              f"wf_M6={wf_m6:.4f} harm_rate={behav['harm_rate']:.4f} "
              f"survival={behav['mean_survival_steps']:.1f} "
              f"head_delta={substrate['M0_action_object_head_param_delta_l2_from_frozen']:.4f}",
              flush=True)
        print(f"verdict: {'PASS' if behav['fatal_errors'] == 0 else 'FAIL'}", flush=True)

        row = {
            "arm_id": arm,
            "seed": int(seed),
            "p1": p1_info,
            "substrate": substrate,
            "behavioural": behav,
            "ao_M5_r2_explained_by_action_alone": float(m5),
            "ao_M6_mean_within_pair_spearman": float(m6),
            "wf_M5_r2_explained_by_action_alone": float(wf_m5),
            "wf_M6_mean_within_pair_spearman": float(wf_m6),
            "grounding_final_loss": (float(p1_info["grounding_final_loss"])
                                     if p1_info.get("grounding_applied") else None),
            "harm_rate": float(behav["harm_rate"]),
            "mean_survival_steps": float(behav["mean_survival_steps"]),
        }
        cell.stamp(row)
    return row


# ============================================================================
# Adjudication.
# ============================================================================

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _median(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.median(xs)) if xs else float("nan")


# self-route labels that carry NO behavioural read (gate never fired).
_NON_CONTRIBUTORY_LABELS = {
    "worldeffect_ungroundable_from_zworld",
    "grounding_objective_ineffective_requeue",
}


def adjudicate(rows_by_arm):
    """Robust grounding gate (M5 state-dependence + world-effect fit + ARM_1-vs-ARM_2
    discrimination) -> behavioural verdict. M6 is recorded, NOT gated (noisy).
    """
    a0 = rows_by_arm["ARM_0"]
    a1 = rows_by_arm["ARM_1"]
    a2 = rows_by_arm["ARM_2"]

    # ---- gate statistics (all low-noise, reproducible across seeds) ----
    a0_m5 = [r["ao_M5_r2_explained_by_action_alone"] for r in a0]
    a1_m5 = [r["ao_M5_r2_explained_by_action_alone"] for r in a1]
    a2_m5 = [r["ao_M5_r2_explained_by_action_alone"] for r in a2]
    a1_fit = [r["grounding_final_loss"] for r in a1 if r.get("grounding_final_loss") is not None]

    median_a1_m5 = _median(a1_m5)
    median_a1_fit = _median(a1_fit) if a1_fit else float("nan")
    paired_m5_drop = _median([m0 - m1 for m0, m1 in zip(a0_m5, a1_m5)])
    state_dep_margin = _median([m2 - m1 for m2, m1 in zip(a2_m5, a1_m5)])

    # (1) world-effect must be GROUNDABLE from z_world at all: if ARM_1 cannot fit
    #     the next-state target (MSE ~ the unit-variance no-fit level), z_world does
    #     not carry enough world-effect information -> SUBSTRATE issue, not objective.
    worldeffect_groundable = bool((median_a1_fit == median_a1_fit)  # not NaN
                                  and median_a1_fit <= GATE_FIT_CEIL)
    # (2) grounding actually made O state-dependent (robust; noise-free).
    state_dependence_acquired = bool(median_a1_m5 <= GATE_M5_CEIL
                                     and paired_m5_drop >= GATE_M5_PAIRED_DROP)
    # (3) it is the CONTENT (true effect), not gradient traffic: TRUE-effect grounding
    #     is state-dependent while the SHUFFLED control is not.
    content_discriminated = bool(state_dep_margin >= GATE_STATE_DEP_MARGIN)

    grounding_took = bool(worldeffect_groundable and state_dependence_acquired
                          and content_discriminated)

    # M6 (ao + world_forward, all arms) -- RECORDED for governance, NOT gated (noisy).
    a0_m6 = [r["ao_M6_mean_within_pair_spearman"] for r in a0]
    a1_m6 = [r["ao_M6_mean_within_pair_spearman"] for r in a1]
    a2_m6 = [r["ao_M6_mean_within_pair_spearman"] for r in a2]
    wf_m6 = [r["wf_M6_mean_within_pair_spearman"] for r in a1]
    wf_m5 = [r["wf_M5_r2_explained_by_action_alone"] for r in a1]

    preconditions = [
        {"name": "worldeffect_groundable_from_zworld",
         "description": "ARM_1 median grounding MSE onto the realised next-world-state "
                        "(unit-variance target; 1.0 = no fit). <= ceiling => z_world carries "
                        "enough world-effect information for O to be grounded; > ceiling => a "
                        "SUBSTRATE limit (z_world under-informative), not an objective failure.",
         "measured": median_a1_fit, "threshold": GATE_FIT_CEIL, "direction": "upper",
         "control": "ARM_2 (shuffled target) fits at ~1.0 -- the no-real-structure baseline",
         "met": worldeffect_groundable},
        {"name": "arm1_state_dependence_acquired",
         "description": "ARM_1 median r2_explained_by_action_alone (M5, ao) DROPS to <= ceiling "
                        "(frozen baseline ~0.99); grounding made O materially state-dependent. "
                        "Low-noise and reproducible (unlike M6).",
         "measured": median_a1_m5, "threshold": GATE_M5_CEIL, "direction": "upper",
         "control": "ARM_0 (frozen) stays ~0.99; ARM_2 (shuffled, unfittable) also stays ~0.99",
         "met": bool(median_a1_m5 <= GATE_M5_CEIL)},
        {"name": "arm1_state_dependence_paired_drop",
         "description": "median(ARM_0 M5 - ARM_1 M5) -- paired within-seed state-dependence gain "
                        "from grounding, removing the seed baseline.",
         "measured": paired_m5_drop, "threshold": GATE_M5_PAIRED_DROP, "direction": "lower",
         "control": "ARM_0 is the same substrate with a frozen head",
         "met": bool(paired_m5_drop >= GATE_M5_PAIRED_DROP)},
        {"name": "content_not_traffic_discriminated",
         "description": "median(ARM_2 M5 - ARM_1 M5) -- TRUE-effect grounding is state-dependent "
                        "while the SHUFFLED control is not (its target is unfittable). Confirms "
                        "the manipulation is world-effect CONTENT, not gradient traffic.",
         "measured": state_dep_margin, "threshold": GATE_STATE_DEP_MARGIN, "direction": "lower",
         "control": "ARM_2 = same objective + traffic on shuffled targets",
         "met": content_discriminated},
    ]

    # ---- behavioural statistics (read only if the gate fires) ----
    a0_hr = _mean([r["harm_rate"] for r in a0])
    a1_hr = _mean([r["harm_rate"] for r in a1])
    a2_hr = _mean([r["harm_rate"] for r in a2])
    a0_sv = _mean([r["mean_survival_steps"] for r in a0])
    a1_sv = _mean([r["mean_survival_steps"] for r in a1])
    a2_sv = _mean([r["mean_survival_steps"] for r in a2])

    n = len(a1)
    per_seed_beats_a0 = sum(
        1 for r1, r0 in zip(a1, a0) if r1["harm_rate"] < r0["harm_rate"])
    consistency = per_seed_beats_a0 / max(1, n)

    beats_a0 = a1_hr <= (1.0 - BEHAV_REL_MARGIN) * a0_hr
    beats_a2 = a1_hr <= (1.0 - BEHAV_REL_MARGIN) * a2_hr
    survival_not_worse = a1_sv >= a0_sv
    consistent = consistency >= BEHAV_SEED_CONSISTENCY
    load_bearing = bool(grounding_took and beats_a0 and beats_a2 and survival_not_worse and consistent)

    behav = {
        "arm0_mean_harm_rate": a0_hr, "arm1_mean_harm_rate": a1_hr, "arm2_mean_harm_rate": a2_hr,
        "arm0_mean_survival": a0_sv, "arm1_mean_survival": a1_sv, "arm2_mean_survival": a2_sv,
        "arm1_beats_arm0_by_margin": bool(beats_a0),
        "arm1_beats_arm2_by_margin": bool(beats_a2),
        "arm1_survival_not_worse_than_arm0": bool(survival_not_worse),
        "per_seed_arm1_beats_arm0_fraction": float(consistency),
    }

    gate_detail = {
        "worldeffect_groundable_from_zworld": worldeffect_groundable,
        "state_dependence_acquired": state_dependence_acquired,
        "content_not_traffic_discriminated": content_discriminated,
        "arm1_fit_per_seed": [float(x) for x in a1_fit],
        "arm1_fit_median": median_a1_fit,
        "arm0_m5_per_seed": [float(x) for x in a0_m5],
        "arm1_m5_per_seed": [float(x) for x in a1_m5],
        "arm2_m5_per_seed": [float(x) for x in a2_m5],
        "arm1_m5_median": median_a1_m5,
        "paired_m5_drop_arm0_minus_arm1": paired_m5_drop,
        "state_dep_margin_arm2_minus_arm1": state_dep_margin,
        # M6 recorded but NOT gated (noise-dominated on the ao head; see thresholds note).
        "arm0_m6_per_seed": [float(x) for x in a0_m6],
        "arm1_m6_per_seed": [float(x) for x in a1_m6],
        "arm2_m6_per_seed": [float(x) for x in a2_m6],
        "world_forward_m5_per_seed": [float(x) for x in wf_m5],
        "world_forward_m6_per_seed": [float(x) for x in wf_m6],
        "m6_is_gated": False,
        "m6_note": ("M6 (within-pair consequence spearman) is recorded but NOT gated: "
                    "calibration showed the frozen ao head's M6 varies -0.18..+0.13 by seed and "
                    "world_forward's own M6 is 0.07-0.22, so an absolute M6 floor is not usable "
                    "on the 16-dim ao head."),
    }

    if not worldeffect_groundable:
        label = "worldeffect_ungroundable_from_zworld"
        outcome = "FAIL"
        summary = ("Substrate limit: ARM_1 could not fit the realised next-world-state "
                   "(median grounding MSE=%.4f > ceiling %.2f; unit-variance target). z_world "
                   "does not carry enough world-effect information to ground O, so the ARM_1 "
                   "manipulation is not established -- a substrate issue, not an objective failure."
                   % (median_a1_fit, GATE_FIT_CEIL))
        edir = {"SD-080": "non_contributory", "SD-004": "non_contributory"}
        non_degenerate = False
        degeneracy_reason = ("ARM_1 could not fit the world-effect target (z_world "
                             "under-informative): the grounded-O manipulation is not established, "
                             "so the behavioural comparison cannot test whether grounding helps.")
    elif not grounding_took:
        label = "grounding_objective_ineffective_requeue"
        outcome = "FAIL"
        summary = ("ARM_1 fit the world-effect (MSE=%.4f) but the grounding gate did not fully "
                   "fire (M5 median=%.4f, paired M5 drop=%.4f, ARM_2-ARM_1 M5 margin=%.4f). "
                   "O did not become sufficiently/discriminably state-dependent; strengthen the "
                   "objective." % (median_a1_fit, median_a1_m5, paired_m5_drop, state_dep_margin))
        edir = {"SD-080": "non_contributory", "SD-004": "non_contributory"}
        non_degenerate = False
        degeneracy_reason = ("ARM_1 grounding gate did not fully fire (state-dependence / content "
                             "discrimination): the behavioural comparison cannot test whether "
                             "grounding is load-bearing.")
    elif load_bearing:
        label = "grounding_load_bearing_defect_confirmed"
        outcome = "PASS"
        summary = ("Grounding O in the world-effect improves planning: ARM_1 harm_rate=%.4f beats "
                   "ARM_0=%.4f and ARM_2=%.4f by the pre-registered margin. SD-080's defect is "
                   "behaviourally load-bearing; SD-004's mechanism is repaired."
                   % (a1_hr, a0_hr, a2_hr))
        edir = {"SD-080": "supports", "SD-004": "mixed"}
        non_degenerate = True
        degeneracy_reason = ""
    else:
        label = "grounding_real_but_not_load_bearing"
        outcome = "FAIL"
        summary = ("Grounding O in the world-effect took (M5 median=%.4f, fit MSE=%.4f, "
                   "ARM_2-ARM_1 M5 margin=%.4f) but did NOT change behaviour (ARM_1 harm_rate=%.4f "
                   "vs ARM_0=%.4f, ARM_2=%.4f). SD-004's semantic-grounding efficiency rationale "
                   "is demoted; SD-080 stays candidate as a correctness-not-capability finding."
                   % (median_a1_m5, median_a1_fit, state_dep_margin, a1_hr, a0_hr, a2_hr))
        edir = {"SD-080": "mixed", "SD-004": "weakens"}
        non_degenerate = True
        degeneracy_reason = ""

    return {
        "outcome": outcome,
        "interpretation": {
            "label": label,
            "grounding_took": grounding_took,
            "preconditions": preconditions,
            "gate_detail": gate_detail,
            "criteria_non_degenerate": {
                "worldeffect_groundable": worldeffect_groundable,
                "state_dependence_acquired": state_dependence_acquired,
                "content_discriminated": content_discriminated,
                "grounding_gate": bool(grounding_took),
                "behavioural_load_bearing": bool(load_bearing),
            },
            "summary": summary,
        },
        "criteria": [
            {"name": "worldeffect_groundable", "load_bearing": False,
             "passed": worldeffect_groundable},
            {"name": "grounding_gate_arm1", "load_bearing": False, "passed": bool(grounding_took)},
            {"name": "behavioural_load_bearing", "load_bearing": True, "passed": bool(load_bearing)},
        ],
        "behavioural_verdict": behav,
        "evidence_direction_per_claim": edir,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
    }


# ============================================================================
# Driver.
# ============================================================================

def run_experiment(args):
    t0 = time.perf_counter()
    seeds = args.seeds
    rows: List[Dict[str, Any]] = []
    rows_by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ARMS}

    full_config = {
        "env": "CausalGridWorldV2", "env_size": ENV_SIZE, "alpha_world": ALPHA_WORLD,
        "world_dim": WORLD_DIM, "self_dim": SELF_DIM, "action_object_dim": 16,
        "warmup_episodes": args.warmup_episodes, "eval_episodes": args.eval_episodes,
        "steps_per_episode": args.steps, "n_states": args.n_states,
        "grounding_objective": "worldeffect_nextstate_code_regression",
        "grounding_transitions": args.grounding_transitions,
        "grounding_steps": args.grounding_steps, "grounding_batch": GROUNDING_BATCH,
        "grounding_lr": GROUNDING_LR, "warmup_lr": WARMUP_LR,
        "gate_m5_ceil": GATE_M5_CEIL, "gate_m5_paired_drop": GATE_M5_PAIRED_DROP,
        "gate_fit_ceil": GATE_FIT_CEIL, "gate_state_dep_margin": GATE_STATE_DEP_MARGIN,
        "m6_gated": False,
        "behav_rel_margin": BEHAV_REL_MARGIN, "behav_seed_consistency": BEHAV_SEED_CONSISTENCY,
        "seeds": seeds, "arms": ARMS, "supersedes": SUPERSEDES_RUN,
    }
    # config_slice for the OFF (P0) path fingerprint: only what P0 reads.
    config_slice = {
        "env": "CausalGridWorldV2", "env_size": ENV_SIZE, "alpha_world": ALPHA_WORLD,
        "world_dim": WORLD_DIM, "self_dim": SELF_DIM,
        "warmup_episodes": args.warmup_episodes, "steps_per_episode": args.steps,
        "warmup_lr": WARMUP_LR,
    }

    warmup_by_seed: Dict[int, Dict[str, float]] = {}
    for seed in seeds:
        # -------- P0: shared substrate, trained once per seed --------
        reset_all_rng(seed)
        env = make_env(seed)
        agent_p0 = make_agent(env)
        frozen_head_snapshot = [p.detach().clone()
                                for p in agent_p0.e2.action_object_head.parameters()]
        warmup_by_seed[seed] = run_p0(
            agent_p0, env, args.warmup_episodes, args.steps, seed, args.warmup_episodes)

        # -------- P1/P2 per arm on a deepcopy of the shared P0 substrate --------
        for arm in ARMS:
            arm_env = make_env(seed)   # fresh env instance per cell (no shared mutable env)
            row = run_seed_arm(seed, arm, agent_p0, frozen_head_snapshot, arm_env, args, config_slice)
            row["warmup"] = warmup_by_seed[seed]
            rows.append(row)
            rows_by_arm[arm].append(row)

    adj = adjudicate(rows_by_arm)

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES_RUN,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "outcome": adj["outcome"],
        "evidence_direction": ("supports" if adj["outcome"] == "PASS"
                               else ("non_contributory"
                                     if adj["interpretation"]["label"] in _NON_CONTRIBUTORY_LABELS
                                     else "mixed")),
        "evidence_direction_per_claim": adj["evidence_direction_per_claim"],
        "interpretation": adj["interpretation"],
        "criteria": adj["criteria"],
        "behavioural_verdict": adj["behavioural_verdict"],
        "non_degenerate": adj["non_degenerate"],
        "degeneracy_reason": adj["degeneracy_reason"],
        "warmup_by_seed": {str(k): v for k, v in warmup_by_seed.items()},
        "arm_results": rows,
    }
    # Always-core is stamped by the single sanctioned writer (write_flat_manifest,
    # stamp=True) at emission time -- after arm_results is assembled, so the
    # multi-arm substrate_hash HOISTS from the per-cell fingerprints.
    return manifest, full_config, t0


def main(args):
    manifest, full_config, t0 = run_experiment(args)
    out_path = write_flat_manifest(
        manifest,
        out_dir=None,                 # worktree-aware auto-resolve via script_path
        dry_run=args.dry_run,
        config=full_config,
        seeds=args.seeds,
        script_path=Path(__file__),
        started_at=t0,
    )
    print(f"\nV3-EXQ-817a outcome: {manifest['outcome']}", flush=True)
    print(f"  label: {manifest['interpretation']['label']}", flush=True)
    print(f"  {manifest['interpretation']['summary']}", flush=True)
    if args.dry_run:
        print("[dry-run] manifest relocated out of evidence/ by emit_outcome", flush=True)
    else:
        print(f"  wrote {out_path}", flush=True)
    return manifest["outcome"], out_path


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--warmup-episodes", type=int, default=WARMUP_EPISODES)
    ap.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    ap.add_argument("--steps", type=int, default=STEPS_PER_EPISODE)
    ap.add_argument("--n-states", type=int, default=N_STATES)
    ap.add_argument("--grounding-transitions", type=int, default=GROUNDING_TRANSITIONS)
    ap.add_argument("--grounding-steps", type=int, default=GROUNDING_STEPS)
    args = ap.parse_args()

    if args.dry_run:
        args.seeds = [0]
        args.warmup_episodes = 2
        args.eval_episodes = 1
        args.steps = 20
        args.n_states = 12
        args.grounding_transitions = 80
        args.grounding_steps = 20
    elif args.seeds is None:
        args.seeds = SEEDS
    return args


if __name__ == "__main__":
    _args = _parse_args()
    _outcome, _out_path = main(_args)
    _outcome_raw = str(_outcome).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=_args.dry_run,
    )
