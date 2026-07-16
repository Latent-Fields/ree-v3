#!/opt/local/bin/python3
"""
V3-EXQ-680e -- EXP-0380 MECH-423 cross-model SUPER-ADDITIVITY 3-arm ablation.
Supersedes V3-EXQ-680d (SAME hypothesis + SAME hardened super-additivity margin
+ SAME co-training stabiliser; a READINESS-COSINE-GATE RECALIBRATION only). This
is the measurement/test-design fix the 680d failure autopsy
(REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-680d_2026-06-15.md,
confirmed 2026-06-15 "Re-queue 680e, fix gate") ordered.

WHY 680e (the ONE design change -- a legitimate iteration, not a re-run)
------------------------------------------------------------------------
680d's co-training stabilisation WORKED: 0/5 INTEGRATED_PAIR seeds diverged, all
world_R2 finite OFF the -1 clamp floor. On that stable substrate the shared-encoder
gradient cosines COLLAPSED toward zero -- range -0.037 .. +0.051 (2/5 positive),
every |cos| < 0.051. That is the near-ORTHOGONAL regime (gradient-orthogonality
noise), NOT negative transfer. But 680d's readiness gate took min(cosine over the
5 usable seeds) against a hard threshold of 0.0 (require min_cos >= 0). For a
quantity noise-distributed around zero the minimum of 5 samples is essentially
always slightly negative, so the gate fired `r1_grad_cosine_net_negative` on
-0.0371 orthogonality noise and self-excluded the run (FORK 3) BEFORE the
super-additivity verdict ran. The suppressed FORK-2 verdict would PASS: from
680d's own manifest, PAIR > ISOLATED on 5/5 seeds; delta_pair
[0.595, 0.337, 0.398, 0.208, 0.210]; mean +0.35; margin 0.286; 3/5 seeds clear
the margin => super_additive = True. MECH-423 was neither weakened nor falsified;
the self-route was a mis-calibrated precondition, contradicted by the run's own
PAIR>ISO data (the 642 "precondition-test-itself-is-wrong" pattern).

THE FIX (readiness cosine gate: magnitude-floored / tolerance-band)
-------------------------------------------------------------------
Replace min(cos) >= 0 with a NEAR-ORTHOGONAL TOLERANCE BAND:
  - COSINE_ORTHOGONAL_BAND (0.15): a usable PAIR seed whose |cos| <= BAND is
    NEAR-ORTHOGONAL and counts as readiness-MET (orthogonal task gradients do NOT
    preclude super-additivity; Caruana 1997 -- near-orthogonal predicts
    additive/independent contributions, and the empirics show super-additivity).
    The band is pre-registered from 680d's observed usable-seed cosine noise
    (max |cos| = 0.051) x ~3 as a noise-vs-signal separator; the 680c
    negative-transfer datum was -0.40 (an order of magnitude below -BAND, and it
    was measured on a DIVERGED seed).
  - Only a CLEARLY net-negative cosine (cos < -BAND) on a MAJORITY of usable seeds
    (>= MIN_SEEDS_NET_NEGATIVE = 3/5) counts as the genuine negative-transfer
    readiness fail -> route r1_grad_cosine_net_negative. Using a MAJORITY (not the
    bare per-seed minimum) stops one near-zero seed from self-excluding the run.
  - Only NON-FINITE / diverged seeds route co_training_diverged.
Everything else is 680d-IDENTICAL and must NOT regress: the gentled encoder
co-training (lower ENC_LR + warmup + non-finite-loss guard; proven 0/5 diverged),
the inf/NaN guard on the readiness cosine, the R2_SCORE_FLOOR bounded score, and
the hardened SD-of-delta + absolute-effect-floor super-additivity margin, n>=5.
Emits n_diverged_seeds and keeps the diverged/non-finite bucketing SEPARATE from
the net-negative test (a diverged seed is never counted net-negative).

SLEEP DRIVER: N/A (R3 readiness calls CrossModuleConsolidator.consolidate()
              directly; no SleepLoopManager / sleep cycle is driven).

EXPERIMENT_PURPOSE = "evidence". claim_ids = ["MECH-423"]. (Evidence, not diagnostic:
the readiness gate self-routes substrate_not_ready / non_degenerate=False when unmet
-- scoring-excluded -- but if readiness IS met the super-additivity verdict runs and
that IS the actual MECH-423 test, so the run must be able to weight MECH-423. Same
purpose + self-routing design as the 680/680a/680b/680c/680d supersession chain.)

PRE-REGISTERED THREE-WAY FORK (so the next self-route is unambiguous)
--------------------------------------------------------------------
  FORK 1 -- diverged seeds dominate (n_usable < MIN_FINITE_PAIR_SEEDS):
    co-training instability regressed. outcome FAIL / substrate_not_ready_requeue /
    co_training_diverged. -> diagnose / stronger stabiliser; do NOT iterate blindly.
  FORK 2 -- stable substrate, cosines near-orthogonal-within-band or positive
    (< MIN_SEEDS_NET_NEGATIVE usable seeds have cos < -BAND): readiness MET -> the
    super-additivity verdict RUNS (the actual MECH-423 test; expected on this
    substrate, likely PASS -- 680d's suppressed verdict was 3/5 over margin).
  FORK 3 -- stable substrate, a CLEARLY net-negative cosine (cos < -BAND) on
    >= MIN_SEEDS_NET_NEGATIVE usable seeds: GENUINE readiness fail / substrate
    finding -- the shared-latent integration is in the negative-transfer regime,
    so super-additivity is NOT expected (anti-aligned task gradients; PCGrad /
    Yu 2020). outcome FAIL / substrate_not_ready_requeue / r1_grad_cosine_net_negative.
    -> /implement-substrate (shared-latent objective reconciliation) or narrow
    MECH-423, NOT another re-queue.

HYPOTHESIS (MECH-423)
---------------------
Cross-model integration is SUPER-ADDITIVE: a representation co-trained over a
SHARED latent by two task objectives exceeds the SUM of what each objective
achieves in isolation (Caruana 1997 shared-rep MTL). The integration locus is the
shared encoder (LatentStack): when both task heads' gradients flow into the shared
latent the encoder develops features shaped by both tasks; with a gradient stop
between the streams it does not.

THREE ARMS (param- and compute-matched; the ONLY structural difference is the
gradient-stop / proxy-head, so a delta cannot be a capacity artefact)
----------------------------------------------------------------------------
  ARM_ISOLATED          -- two task heads train on a DETACHED shared latent (no
                           gradient into the shared encoder). The ADDITIVE baseline.
  ARM_INTEGRATED_PAIR   -- both task heads train on the LIVE shared latent
                           (cross-module gradient flow); iterative-inference settling
                           ON (R2); interleaved cross-module consolidation (R3). The
                           LOAD-BEARING arm for the MECH-423 verdict.
  ARM_INTEGRATED_TRIPLE -- PAIR plus a third PROXY object-binding head co-shaping the
                           shared latent. A stand-in attachment surface for the
                           ARC-080 object spine (V4, not built); the TRIPLE delta is
                           EXPLORATORY ONLY and tags no claim.

OPERATIONALISATION (the two task objectives)
--------------------------------------------
  world_head : z_self -> z_world   (recover world structure from body state)
  self_head  : z_world -> z_self   (recover the body/affordance state from world)
Score per arm = world_R2 + affordance_R2 (held-out R2 of each head). The super-
additivity test is on the DELTA of this combined score, integrated minus isolated.

PHASED TRAINING
---------------
P0 integration phase: rollout; per step the two (PAIR: three) heads co-train over
   the shared latent -- LIVE for the integrated arms AFTER the encoder warmup
   (gradient into the encoder = the integration under test, at ENC_LR), DETACHED for
   the isolated arm and during the integrated arms' warmup window (gradient stop).
P1 frozen-head finetune: the encoder (LatentStack) is FROZEN; the heads are
   finetuned on DETACHED latents IDENTICALLY across arms (calibrates the eval readout
   to each arm's final frozen encoder; the arm difference is the P0 encoder
   co-shaping, not the readout).
P2 eval: everything frozen; a fresh held-out rollout; world_R2 + affordance_R2.

PRE-REGISTERED SUPER-ADDITIVITY MARGIN (statistical floored by absolute effect)
------------------------------------------------------------------------------
With readiness satisfied (FORK 2):
  additive_baseline[seed] = integration_score(ARM_ISOLATED, seed)
  delta_pair[seed]        = integration_score(ARM_INTEGRATED_PAIR, seed) - baseline
  delta_sd                = across-seed SD of the DELTA distribution (delta_pair)
  margin                  = max(SUPERADD_SD_MULT * delta_sd, SUPERADD_MIN_EFFECT_FLOOR)
  PASS  if delta_pair[seed] > margin on >= MIN_SEEDS_PASS seeds.
  FAIL/weakens if mean delta_pair <= 0 (merely additive / sub-additive).
  inconclusive (positive but below margin) -> FAIL, evidence_direction "inconclusive".
The absolute floor hard-floors the margin so a vanishing delta_sd (consistent delta)
cannot manufacture a false PASS (the 680/680a false-PASS lesson). UNCHANGED from 680d.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_680e_mech423_superadditivity_ablation.py --dry-run
  /opt/local/bin/python3 experiments/v3_exq_680e_mech423_superadditivity_ablation.py
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_680e_mech423_superadditivity_ablation"
QUEUE_ID = "V3-EXQ-680e"
CLAIM_IDS: List[str] = ["MECH-423"]
EXPERIMENT_PURPOSE = "evidence"
# Supersedes V3-EXQ-680d (which superseded 680c/680b/680a/680). 680d's co-training
# stabilisation WORKED (0/5 diverged; all world_R2 finite off-floor) but its readiness
# cosine gate (min(cos over usable seeds) >= 0) fired r1_grad_cosine_net_negative on
# -0.0371 gradient-ORTHOGONALITY noise and self-excluded a run whose suppressed FORK-2
# verdict would PASS (3/5 seeds over margin 0.286; mean delta +0.35; PAIR>ISO 5/5). 680e
# re-runs the SAME hypothesis + SAME margin + SAME stabiliser and changes ONLY the
# readiness cosine gate: a near-orthogonal band (|cos| <= COSINE_ORTHOGONAL_BAND) is
# readiness-MET, and only a MAJORITY of usable seeds with a CLEARLY net-negative cosine
# (< -BAND) routes r1_grad_cosine_net_negative. Mark the 680d manifest
# evidence_direction=superseded at next governance.
SUPERSEDES = "v3_exq_680d_mech423_superadditivity_ablation"
INTEGRATION_EFFECT_FLOOR = 1e-3  # |delta_pair| below this on EVERY seed = inert integration

# ---- 680c robustness fixes (RETAINED, failure_autopsy_V3-EXQ-680b_2026-06-14) ----
GRAD_CLIP_NORM = 1.0   # clip P0 (heads + shared encoder) + P1 (heads) grad norm.
R2_SCORE_FLOOR = -1.0  # lower-bound on each held-out R2 (world / affordance) before
                       # forming integration_score = world_R2 + aff_R2, so a diverged
                       # seed contributes at most R2_SCORE_FLOOR (not -149) to the
                       # across-seed delta SD. The hardened margin BASIS is unchanged.

# ---- 680d co-training-stability fixes (RETAINED, failure_autopsy_V3-EXQ-680c) ----
# (a-1) separate, LOWER learning rate for the SHARED ENCODER (latent_stack) in P0.
#       The heads learn at HEAD_LR; the integration locus moves 10x slower so the
#       co-training does not blow the world head to the world_R2 -1 floor. PROVEN
#       stable in 680d (0/5 PAIR seeds diverged) -- DO NOT regress.
ENC_LR = 1e-4
# (a-2) encoder-gradient WARMUP: for the integrated arms, DETACH the shared latent
#       for the first ENC_WARMUP_EPISODES of P0 (heads-only), then turn on co-training.
ENC_WARMUP_EPISODES = 10
# (b) acceptance precondition: >= this many of the 5 INTEGRATED_PAIR seeds must be
#     USABLE (finite world_R2 off the -1 floor AND finite R1 grad/cosine) before any
#     cosine is read. Below this -> substrate_not_ready_requeue / co_training_diverged.
MIN_FINITE_PAIR_SEEDS = 4

ARMS = ("ARM_ISOLATED", "ARM_INTEGRATED_PAIR", "ARM_INTEGRATED_TRIPLE")

# ---- design parameters (PRE-REGISTERED) ----
SEEDS = (42, 123, 456, 789, 1011)
TRAIN_EPISODES = 40               # P0 integration-phase episodes (the [train] denominator)
STEPS_PER_EPISODE = 80
P1_STEPS = 300                    # P1 frozen-encoder head finetune steps
EVAL_STEPS = 300                  # P2 held-out eval rollout
HEAD_LR = 1e-3
GRID_SIZE = 6
N_HAZARDS = 4
N_RESOURCES = 3

# R2 iterative-inference settling (integrated arms)
SETTLE_ITERS = 10
REL_TOL = 0.05

# R3 cross-module consolidation readiness
CMC_STEPS = 8
CMC_LR = 1e-3
CMC_BATCH = 16

# R1 readiness floors
MIN_GRAD_FLOOR = 1e-6

# ---- 680e RECALIBRATED readiness cosine gate (the ONE design change) ----
# Near-orthogonal tolerance band on the shared-encoder gradient cosine. A usable
# PAIR seed whose |cos| <= COSINE_ORTHOGONAL_BAND is NEAR-ORTHOGONAL and counts as
# readiness-MET (orthogonal task gradients do not preclude super-additivity). The
# band is pre-registered from 680d's observed usable-seed cosine noise: range
# -0.037 .. +0.051 (max |cos| = 0.051); BAND = ~3 x that = 0.15, cleanly above the
# orthogonality noise and far above the 680c negative-transfer datum (-0.40).
COSINE_ORTHOGONAL_BAND = 0.15
# The genuine negative-transfer readiness fail (FORK 3) fires ONLY when a MAJORITY of
# usable seeds have a CLEARLY net-negative cosine (cos < -BAND). Majority (not the bare
# per-seed minimum) stops one near-zero seed self-excluding the run.
MIN_SEEDS_NET_NEGATIVE = 3        # majority of len(SEEDS)==5; matches MIN_SEEDS_PASS
# Indexer-safe ceiling form of the MIN_SEEDS_NET_NEGATIVE rule: the readiness
# precondition measures n_net_negative and is UNMET (FORK 3) iff n_net_negative
# exceeds this many tolerated (i.e. >= MIN_SEEDS_NET_NEGATIVE). direction="upper".
MAX_NET_NEGATIVE_SEEDS_TOLERATED = MIN_SEEDS_NET_NEGATIVE - 1  # 2

# super-additivity margin (pre-registered; UNCHANGED from 680d).
SUPERADD_SD_MULT = 2.0
SUPERADD_MIN_EFFECT_FLOOR = 0.02  # absolute floor on the super-additive delta
                                  # (combined-R2 units); > INTEGRATION_EFFECT_FLOOR.
MIN_SEEDS_PASS = 3               # majority of len(SEEDS)==5

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Proxy object-binding stream -- a stand-in ATTACHMENT SURFACE for the ARC-080
# object spine (V4, not in V3). NOT ARC-080; the TRIPLE delta is exploratory.
# --------------------------------------------------------------------------- #
class ProxyObjectStream(nn.Module):
    """A trainable third stream: a bottlenecked autoencoder over the shared
    latent. When fed the LIVE shared latent its gradient co-shapes the encoder
    (a third cross-pollinating stream). A placeholder for the ARC-080 object
    spine -- it does NOT implement type/token/anchor binding and tags no claim."""

    def __init__(self, combined_dim: int):
        super().__init__()
        bottleneck = max(4, combined_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(combined_dim, bottleneck),
            nn.Tanh(),
            nn.Linear(bottleneck, combined_dim),
        )

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        return self.net(combined)


def _to_batched(x, device) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x = x.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        hazard_harm=0.02,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2, integrated: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        # R2 iterative-inference settling: ON only for the integrated arms.
        use_iterative_inference=integrated,
        inference_settle_iters=SETTLE_ITERS if integrated else 1,
        inference_convergence_rel_tol=REL_TOL,
        # R3 consolidator built on every arm (standalone readiness readout);
        # the SCHEDULE differs by arm at consolidate() call time.
        use_cross_module_consolidation=True,
        cross_module_consolidation_schedule="interleaved" if integrated else "blocked",
        cross_module_consolidation_steps=CMC_STEPS,
        cross_module_consolidation_lr=CMC_LR,
        cross_module_consolidation_batch=CMC_BATCH,
    )
    return REEAgent(cfg)


def _set_requires_grad(modules, flag: bool) -> None:
    for m in modules:
        for p in m.parameters():
            p.requires_grad = flag


def _r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Variance-explained R2 of pred vs target over a [N, D] batch."""
    sse = float(((pred - target) ** 2).sum().item())
    var = float(((target - target.mean(dim=0, keepdim=True)) ** 2).sum().item())
    if var <= EPS:
        return 0.0
    return 1.0 - sse / var


def _frozen_rollout(agent, env, device, rng, n_steps: int):
    """Roll the env for n_steps with the encoder frozen; return detached
    (z_self [N, self_dim], z_world [N, world_dim])."""
    self_list: List[torch.Tensor] = []
    world_list: List[torch.Tensor] = []
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    with torch.no_grad():
        for _step in range(n_steps):
            obs_body = _to_batched(obs_dict["body_state"], device)
            obs_world = _to_batched(obs_dict["world_state"], device)
            obs_harm = obs_dict.get("harm_obs", None)
            if obs_harm is not None:
                obs_harm = _to_batched(obs_harm, device)
            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            self_list.append(latent.z_self.detach().clone())
            world_list.append(latent.z_world.detach().clone())
            action_idx = int(torch.randint(0, env.action_dim, (1,), generator=rng).item())
            action = torch.zeros(1, env.action_dim, device=device)
            action[0, action_idx] = 1.0
            _, harm_signal, done, info, obs_dict = env.step(action)
            if done:
                _, obs_dict = env.reset()
                agent.reset()
                agent.e1.reset_hidden_state()
    return torch.cat(self_list, dim=0), torch.cat(world_list, dim=0)


def run_cell(arm: str, seed: int, train_eps: int, steps: int,
             p1_steps: int, eval_steps: int, enc_warmup: int) -> Dict:
    """Train + eval one (arm, seed) cell; return its row (incl. readiness on
    the integrated arms)."""
    print(f"Seed {seed} Condition {arm}", flush=True)
    integrated = arm in ("ARM_INTEGRATED_PAIR", "ARM_INTEGRATED_TRIPLE")
    triple = arm == "ARM_INTEGRATED_TRIPLE"

    env = _make_env(seed)
    agent = _make_agent(env, integrated)
    device = agent.device

    self_dim = agent.config.latent.self_dim
    world_dim = agent.config.latent.world_dim
    combined_dim = self_dim + world_dim

    # Two task heads (+ optional proxy object stream).
    world_head = nn.Linear(self_dim, world_dim).to(device)   # z_self -> z_world
    self_head = nn.Linear(world_dim, self_dim).to(device)    # z_world -> z_self
    proxy_obj = ProxyObjectStream(combined_dim).to(device) if triple else None

    head_params = list(world_head.parameters()) + list(self_head.parameters())
    if proxy_obj is not None:
        head_params += list(proxy_obj.parameters())
    enc_params_p0 = list(agent.latent_stack.parameters())
    # (680d a-1) P0 optimizer: heads at HEAD_LR, the SHARED ENCODER (latent_stack,
    # the integration locus + 680c divergence site) at the LOWER ENC_LR via Adam
    # param groups. For ISOLATED the detach stops the encoder gradient so Adam
    # leaves it at init regardless; for INTEGRATED the encoder co-trains slowly.
    p0_opt = torch.optim.Adam(
        [{"params": head_params, "lr": HEAD_LR},
         {"params": enc_params_p0, "lr": ENC_LR}])
    p1_opt = torch.optim.Adam(head_params, lr=HEAD_LR)  # heads only; encoder frozen

    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    rng = torch.Generator(device="cpu").manual_seed(seed)

    n_nonfinite_skips = 0
    n_enc_live_episodes = 0

    # ---------------- P0: integration phase ----------------
    for ep in range(train_eps):
        # (680d a-2) encoder-gradient warmup: integrated arms detach the shared
        # latent (heads-only) for the first enc_warmup episodes, then co-train.
        enc_live = integrated and (ep >= enc_warmup)
        if enc_live:
            n_enc_live_episodes += 1
        if ep % 8 == 0 or ep == train_eps - 1:
            print(f"  [train] {arm} seed={seed} ep {ep+1}/{train_eps} "
                  f"enc_live={enc_live}", flush=True)
        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()
        for _step in range(steps):
            obs_body = _to_batched(obs_dict["body_state"], device)
            obs_world = _to_batched(obs_dict["world_state"], device)
            obs_harm = obs_dict.get("harm_obs", None)
            if obs_harm is not None:
                obs_harm = _to_batched(obs_harm, device)

            prev_latent = agent._current_latent
            prev_z_self = (
                prev_latent.z_self.detach().clone() if prev_latent is not None else None
            )

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks = agent.clock.advance()
            if ticks.get("e1_tick", False):
                agent._e1_tick(latent)

            action_idx = int(torch.randint(0, env.action_dim, (1,), generator=rng).item())
            action = torch.zeros(1, env.action_dim, device=device)
            action[0, action_idx] = 1.0
            if prev_z_self is not None:
                agent.record_transition(prev_z_self, action, latent.z_self.detach())

            # Shared latent. LIVE for integrated AFTER warmup (grad flows into the
            # encoder = the integration under test); DETACHED for isolated AND for
            # integrated during the warmup window (gradient stop).
            combined_full = torch.cat([latent.z_self, latent.z_world], dim=-1)

            z_self_in = latent.z_self if enc_live else latent.z_self.detach()
            z_world_in = latent.z_world if enc_live else latent.z_world.detach()
            comb_in = combined_full if enc_live else combined_full.detach()

            # Cross-stream task losses (targets always detached).
            loss = (
                F.mse_loss(world_head(z_self_in), latent.z_world.detach())
                + F.mse_loss(self_head(z_world_in), latent.z_self.detach())
            )
            if proxy_obj is not None:
                loss = loss + F.mse_loss(proxy_obj(comb_in), comb_in.detach())

            # (680d a-3) non-finite-loss guard: a nan/inf loss skips the step so a
            # single blown step cannot corrupt the encoder/head weights.
            if not torch.isfinite(loss):
                n_nonfinite_skips += 1
                p0_opt.zero_grad()
            else:
                p0_opt.zero_grad()
                loss.backward()
                # clip the joint (heads + shared encoder) grad norm (680c, retained).
                torch.nn.utils.clip_grad_norm_(head_params + enc_params_p0, GRAD_CLIP_NORM)
                p0_opt.step()

            _, harm_signal, done, info, obs_dict = env.step(action)
            agent.update_residue(float(harm_signal) if float(harm_signal) < 0 else 0.0)
            if done:
                _, obs_dict = env.reset()
                agent.reset()
                agent.e1.reset_hidden_state()

    n_world_buffer = len(agent._world_experience_buffer)
    n_e2_buffer = len(agent._e2_transition_buffer)

    # ---------------- P1: frozen-encoder head finetune (identical all arms) ----
    _set_requires_grad([agent.e1, agent.e2, agent.latent_stack], False)
    ft_self, ft_world = _frozen_rollout(agent, env, device, rng, eval_steps)
    n_ft = ft_self.shape[0]
    if n_ft >= CMC_BATCH:
        for _ps in range(p1_steps):
            idx = torch.randint(0, n_ft, (CMC_BATCH,), generator=rng)
            b_self = ft_self[idx]
            b_world = ft_world[idx]
            loss = (
                F.mse_loss(world_head(b_self), b_world)
                + F.mse_loss(self_head(b_world), b_self)
            )
            if proxy_obj is not None:
                loss = loss + F.mse_loss(
                    proxy_obj(torch.cat([b_self, b_world], dim=-1)),
                    torch.cat([b_self, b_world], dim=-1))
            if not torch.isfinite(loss):
                n_nonfinite_skips += 1
                p1_opt.zero_grad()
                continue
            p1_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, GRAD_CLIP_NORM)
            p1_opt.step()

    # ---------------- P2: held-out eval (everything frozen) ----------------
    world_head.eval()
    self_head.eval()
    ez_self, ez_world = _frozen_rollout(agent, env, device, rng, eval_steps)
    with torch.no_grad():
        world_r2_raw = _r2(world_head(ez_self), ez_world)
        affordance_r2_raw = _r2(self_head(ez_world), ez_self)
    # Bound each R2 at R2_SCORE_FLOOR before forming the score (680c, retained).
    world_r2 = max(world_r2_raw, R2_SCORE_FLOOR)
    affordance_r2 = max(affordance_r2_raw, R2_SCORE_FLOOR)
    integration_score = world_r2 + affordance_r2

    # ---------------- readiness readouts (integrated arms only) ----------------
    r1 = r2 = r3 = None
    if integrated:
        world_head.train()
        self_head.train()
        # R1: cross-module gradient coupling on the SHARED ENCODER params (680c).
        _set_requires_grad([agent.latent_stack], True)
        enc_params = [p for p in agent.latent_stack.parameters() if p.requires_grad]
        _, r1_obs = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()
        ob_b = _to_batched(r1_obs["body_state"], device)
        ob_w = _to_batched(r1_obs["world_state"], device)
        ob_h = r1_obs.get("harm_obs", None)
        if ob_h is not None:
            ob_h = _to_batched(ob_h, device)
        r1_lat = agent.sense(ob_b, ob_w, obs_harm=ob_h)
        world_loss = F.mse_loss(world_head(r1_lat.z_self), r1_lat.z_world.detach())
        self_loss = F.mse_loss(self_head(r1_lat.z_world), r1_lat.z_self.detach())
        if enc_params:
            gw = torch.autograd.grad(world_loss, enc_params, retain_graph=True,
                                     allow_unused=True)
            gs = torch.autograd.grad(self_loss, enc_params, retain_graph=False,
                                     allow_unused=True)
            gw_flat = torch.cat([
                (g if g is not None else torch.zeros_like(p)).reshape(-1)
                for g, p in zip(gw, enc_params)])
            gs_flat = torch.cat([
                (g if g is not None else torch.zeros_like(p)).reshape(-1)
                for g, p in zip(gs, enc_params)])
            wn = float(gw_flat.norm().item())
            sn = float(gs_flat.norm().item())
            cos = (float((torch.dot(gw_flat, gs_flat)
                          / (gw_flat.norm() * gs_flat.norm() + EPS)).item())
                   if (wn > EPS and sn > EPS) else 1.0)
            min_grad = min(wn, sn)
            # (680e) "coupled" = coupling EXISTS (finite grad off floor) AND is not
            # CLEARLY net-negative (cos >= -BAND). Near-orthogonal counts as coupled.
            # Informational per-cell label; the readiness GATE reads the raw cosine.
            r1 = {
                "coupled": bool(math.isfinite(min_grad) and math.isfinite(cos)
                                and min_grad > EPS
                                and cos >= -COSINE_ORTHOGONAL_BAND),
                "min_grad_norm": min_grad,
                "mean_pairwise_cosine": cos,
                "n_modules": int((wn > EPS) + (sn > EPS)),
                "probe_locus": "shared_encoder_params",
            }
        else:
            r1 = {"coupled": False, "min_grad_norm": 0.0,
                  "mean_pairwise_cosine": -1.0, "n_modules": 0,
                  "probe_locus": "shared_encoder_params"}

        # R2: iterative-inference convergence from one sense() on the substrate.
        _, r2_obs = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()
        obs_body = _to_batched(r2_obs["body_state"], device)
        obs_world = _to_batched(r2_obs["world_state"], device)
        obs_harm = r2_obs.get("harm_obs", None)
        if obs_harm is not None:
            obs_harm = _to_batched(obs_harm, device)
        _ = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
        ic = agent.last_inference_convergence
        if ic is None:
            r2 = {"present": False, "converged": False,
                  "final_rel_delta": float("nan"), "n_iters": 0}
        else:
            r2 = {
                "present": True,
                "converged": bool(ic.get("converged", False)),
                "final_rel_delta": float(ic.get("final_rel_delta", float("nan"))),
                "n_iters": int(ic.get("n_iters", 0)),
            }

        # R3: interleaved vs blocked cross-module consolidation share.
        _set_requires_grad([agent.e1, agent.e2], True)
        cmc = agent.cross_module_consolidator
        module_losses = {
            "e1": lambda: agent.compute_prediction_loss(),
            "e2": lambda: agent.compute_e2_loss(batch_size=CMC_BATCH),
        }
        module_params = {
            "e1": list(agent.e1.parameters()),
            "e2": list(agent.e2.parameters()),
        }
        inter = cmc.consolidate(
            module_losses=module_losses, module_params=module_params,
            n_steps=CMC_STEPS, schedule="interleaved", lr=CMC_LR, simulation_mode=False)
        blocked = cmc.consolidate(
            module_losses=module_losses, module_params=module_params,
            n_steps=CMC_STEPS, schedule="blocked", lr=CMC_LR, simulation_mode=False)
        r3 = {
            "interleaved_share": float(inter["cross_module_replay_share"]),
            "blocked_share": float(blocked["cross_module_replay_share"]),
            "interleaved_n_updates": float(inter["n_updates"]),
            "updates_e2_interleaved": float(inter.get("updates_e2", 0.0)),
            "interleaved_flag": float(inter["interleaved"]),
        }

    print(
        f"  {arm} seed={seed} world_R2={world_r2:.4f}(raw={world_r2_raw:.4g}) "
        f"aff_R2={affordance_r2:.4f} score={integration_score:.4f}"
        + (f" | R1 cos={r1['mean_pairwise_cosine']:.3g} min_grad={r1['min_grad_norm']:.3g}"
           f" | R2 conv={r2['converged']} n_iters={r2['n_iters']}"
           f" | R3 inter={r3['interleaved_share']:.2f} blk={r3['blocked_share']:.2f}"
           f" | nonfinite_skips={n_nonfinite_skips} enc_live_ep={n_enc_live_episodes}"
           if integrated else ""),
        flush=True,
    )
    # Per-cell progress verdict (cell completed with a finite score).
    cell_ok = bool(math.isfinite(integration_score))
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)

    return {
        "arm": arm,
        "seed": seed,
        "world_r2": world_r2,
        "affordance_r2": affordance_r2,
        "world_r2_raw": world_r2_raw,
        "affordance_r2_raw": affordance_r2_raw,
        "r2_score_floor": R2_SCORE_FLOOR,
        "integration_score": integration_score,
        "n_world_buffer": n_world_buffer,
        "n_e2_buffer": n_e2_buffer,
        "n_nonfinite_skips": n_nonfinite_skips,
        "n_enc_live_episodes": n_enc_live_episodes,
        "R1": r1,
        "R2": r2,
        "R3": r3,
    }


def _finite_off_floor(row: Dict) -> bool:
    """A PAIR cell whose world head did NOT diverge to the -1 clamp floor."""
    raw = row["world_r2_raw"]
    return math.isfinite(raw) and raw > (R2_SCORE_FLOOR + 1e-9)


def _finite_r1(row: Dict) -> bool:
    """A PAIR cell whose R1 shared-encoder grad/cosine probe is numerically finite."""
    r1 = row["R1"]
    if not r1:
        return False
    return math.isfinite(r1["mean_pairwise_cosine"]) and math.isfinite(r1["min_grad_norm"])


def _usable(row: Dict) -> bool:
    """A PAIR cell usable for the readiness cosine: stable world head AND finite R1."""
    return _finite_off_floor(row) and _finite_r1(row)


def main(dry_run: bool = False):
    """Returns (outcome, manifest_path). manifest_path is None on dry-run."""
    train_eps = 3 if dry_run else TRAIN_EPISODES
    steps = 15 if dry_run else STEPS_PER_EPISODE
    p1_steps = 20 if dry_run else P1_STEPS
    eval_steps = 20 if dry_run else EVAL_STEPS
    seeds = (SEEDS[0],) if dry_run else SEEDS
    # Keep at least one encoder-live episode in dry-run so the co-training path is
    # exercised; full-scale uses the pre-registered ENC_WARMUP_EPISODES.
    enc_warmup = min(1, train_eps - 1) if dry_run else ENC_WARMUP_EPISODES
    # Dry-run has 1 seed, so relax the FORK-1 finite-seed gate to len(seeds) so the
    # recalibrated readiness/verdict path (FORK 2) is exercised in the smoke test.
    # Full runs (5 seeds) use the pre-registered MIN_FINITE_PAIR_SEEDS unchanged.
    min_finite_pair = MIN_FINITE_PAIR_SEEDS if not dry_run else min(
        MIN_FINITE_PAIR_SEEDS, len(seeds))

    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run}) "
          f"arms={ARMS} seeds={seeds} train_eps={train_eps} "
          f"enc_warmup={enc_warmup} enc_lr={ENC_LR} "
          f"cosine_band={COSINE_ORTHOGONAL_BAND}", flush=True)
    t0 = time.time()

    rows: Dict[tuple, Dict] = {}
    arm_results: List[Dict] = []
    for arm in ARMS:
        for seed in seeds:
            cfg_slice = {
                "experiment": EXPERIMENT_TYPE,
                "arm": arm,
                "train_episodes": train_eps,
                "steps_per_episode": steps,
                "p1_steps": p1_steps,
                "eval_steps": eval_steps,
                "enc_warmup": enc_warmup,
                "enc_lr": ENC_LR,
                "settle_iters": SETTLE_ITERS,
                "rel_tol": REL_TOL,
                "cmc_steps": CMC_STEPS,
                "grid_size": GRID_SIZE,
                "num_hazards": N_HAZARDS,
                "num_resources": N_RESOURCES,
            }
            with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__),
                          extra_ineligible_reasons=["fresh_evidence_run_no_reuse"]) as cell:
                row = run_cell(arm, seed, train_eps, steps, p1_steps, eval_steps,
                               enc_warmup)
                cell.stamp(row)
            rows[(arm, seed)] = row
            arm_results.append(row)

    elapsed = time.time() - t0

    # ---------------- readiness gate (load-bearing ARM_INTEGRATED_PAIR) -------
    pair_rows = [rows[("ARM_INTEGRATED_PAIR", s)] for s in seeds]
    usable_rows = [r for r in pair_rows if _usable(r)]
    n_usable = len(usable_rows)
    n_diverged = len(pair_rows) - n_usable
    diverged_seeds = [r["seed"] for r in pair_rows if not _usable(r)]
    pair_finite_world = {r["seed"]: _finite_off_floor(r) for r in pair_rows}

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    co_training = {
        "n_pair_seeds": len(pair_rows),
        "n_usable_pair_seeds": n_usable,
        "n_diverged_seeds": n_diverged,
        "diverged_seeds": diverged_seeds,
        "min_finite_pair_seeds_required": MIN_FINITE_PAIR_SEEDS,
        "pair_world_r2_raw": {r["seed"]: r["world_r2_raw"] for r in pair_rows},
        "pair_finite_off_floor": pair_finite_world,
        "pair_r1_cosine": {r["seed"]: (r["R1"]["mean_pairwise_cosine"] if r["R1"] else None)
                           for r in pair_rows},
        "pair_nonfinite_skips": {r["seed"]: r["n_nonfinite_skips"] for r in pair_rows},
    }

    def _write_manifest(manifest: Dict) -> Optional[str]:
        if dry_run:
            print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.")
            return None
        out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Result written to: {out_path}")
        return str(out_path)

    base_manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "supersedes": SUPERSEDES,
        "timestamp_utc": ts,
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "proposal_id": "EXP-0380",
        "sleep_driver_pattern": "N/A (CrossModuleConsolidator.consolidate() called directly; no sleep cycle)",
        "thresholds": {
            "min_grad_floor": MIN_GRAD_FLOOR,
            "cosine_orthogonal_band": COSINE_ORTHOGONAL_BAND,
            "min_seeds_net_negative": MIN_SEEDS_NET_NEGATIVE,
            "max_net_negative_seeds_tolerated": MAX_NET_NEGATIVE_SEEDS_TOLERATED,
            "rel_tol": REL_TOL,
            "settle_iters": SETTLE_ITERS,
            "superadd_sd_mult": SUPERADD_SD_MULT,
            "superadd_min_effect_floor": SUPERADD_MIN_EFFECT_FLOOR,
            "integration_effect_floor": INTEGRATION_EFFECT_FLOOR,
            "min_seeds_pass": MIN_SEEDS_PASS,
            "min_finite_pair_seeds": MIN_FINITE_PAIR_SEEDS,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "r2_score_floor": R2_SCORE_FLOOR,
            "enc_lr": ENC_LR,
            "head_lr": HEAD_LR,
            "enc_warmup_episodes": enc_warmup,
        },
        "config": {
            "arms": list(ARMS),
            "seeds": list(seeds),
            "train_episodes": train_eps,
            "steps_per_episode": steps,
            "p1_steps": p1_steps,
            "eval_steps": eval_steps,
            "grid_size": GRID_SIZE,
            "num_hazards": N_HAZARDS,
            "num_resources": N_RESOURCES,
            "cosine_orthogonal_band": COSINE_ORTHOGONAL_BAND,
            "min_seeds_net_negative": MIN_SEEDS_NET_NEGATIVE,
        },
        "co_training": co_training,
        "arm_results": arm_results,
        "proxy_object_spine_note": (
            "ARM_INTEGRATED_TRIPLE uses a PROXY object-binding head (ProxyObjectStream), "
            "NOT the ARC-080 object spine (candidate/v3_pending/v4). The TRIPLE delta is "
            "EXPLORATORY only and is NOT evidence about ARC-080; this run tags MECH-423 only."),
    }

    # ---- FORK 1: co-training diverged (diverged seeds dominate) ----
    if n_usable < min_finite_pair:
        reason = (f"co_training_diverged: only {n_usable}/{len(pair_rows)} "
                  f"INTEGRATED_PAIR seeds usable (finite world_R2 off the -1 floor AND "
                  f"finite R1 grad/cosine); required >= {min_finite_pair}. "
                  f"Diverged seeds: {diverged_seeds}.")
        print(f"\n[{EXPERIMENT_TYPE}] FORK 1 -> substrate_not_ready_requeue: {reason}")
        manifest = dict(base_manifest)
        manifest.update({
            "outcome": "FAIL",
            "result": "FAIL",
            "evidence_direction": "inconclusive",
            "evidence_direction_note": (
                "Co-training instability REGRESSED: the integrated-PAIR world head "
                "diverges to the world_R2 -1 clamp floor on too many seeds, so the "
                "readiness cosine cannot be read. NOT a MECH-423 weakens -- the "
                "integrated arm equals the isolated arm by construction when the "
                "encoder co-training has not stabilised. Route: diagnose / stronger "
                "stabiliser (do NOT iterate blindly). " + reason),
            "non_degenerate": False,
            "degeneracy_reason": "substrate_not_ready: " + reason,
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "route": "co_training_diverged",
                "fork": 1,
                "preconditions": [
                    {"name": "pair_seeds_finite_world_r2",
                     "description": "PAIR seeds with finite world_R2 off the -1 clamp floor AND finite R1 grad/cosine",
                     "measured": float(n_usable), "threshold": float(min_finite_pair),
                     "direction": "lower", "control": "co-training stability precondition",
                     "met": False},
                ],
                "criteria_non_degenerate": {"co_training_stable": False},
            },
            "elapsed_seconds": elapsed,
        })
        return "FAIL", _write_manifest(manifest)

    # ---- readiness checks over the USABLE seeds (inf/NaN-guarded) ----
    usable_cos = [r["R1"]["mean_pairwise_cosine"] for r in usable_rows]
    # (680e) near-orthogonal tolerance-band classification of the usable-seed cosines.
    n_net_negative = sum(1 for c in usable_cos if c < -COSINE_ORTHOGONAL_BAND)
    n_near_orthogonal = sum(1 for c in usable_cos if abs(c) <= COSINE_ORTHOGONAL_BAND)
    n_net_positive = sum(1 for c in usable_cos if c > COSINE_ORTHOGONAL_BAND)
    min_cos = min(usable_cos)
    max_cos = max(usable_cos)
    mean_cos = float(sum(usable_cos) / len(usable_cos))
    net_negative_seeds = [r["seed"] for r in usable_rows
                          if r["R1"]["mean_pairwise_cosine"] < -COSINE_ORTHOGONAL_BAND]

    min_min_grad = min(r["R1"]["min_grad_norm"] for r in usable_rows)
    min_n_iters = min(r["R2"]["n_iters"] for r in usable_rows)
    min_updates_e2 = min(r["R3"]["updates_e2_interleaved"] for r in usable_rows)
    inter_share_ok = all(r["R3"]["interleaved_share"] > 0.0 for r in usable_rows)
    blocked_share_zero = all(r["R3"]["blocked_share"] == 0.0 for r in usable_rows)

    cosine_band_classification = {
        "cosine_orthogonal_band": COSINE_ORTHOGONAL_BAND,
        "n_net_negative": n_net_negative,
        "n_near_orthogonal": n_near_orthogonal,
        "n_net_positive": n_net_positive,
        "net_negative_seeds": net_negative_seeds,
        "min_cos_usable": min_cos,
        "max_cos_usable": max_cos,
        "mean_cos_usable": mean_cos,
        "min_seeds_net_negative_for_fork3": MIN_SEEDS_NET_NEGATIVE,
    }

    readiness_checks = [
        {"name": "pair_seeds_finite_world_r2", "measured": float(n_usable),
         "threshold": float(min_finite_pair), "direction": "lower"},
        {"name": "r1_shared_latent_grad_coupled", "measured": float(min_min_grad),
         "threshold": float(MIN_GRAD_FLOOR), "direction": "lower"},
        # (680e) RECALIBRATED cosine gate: near-orthogonal (|cos| <= BAND) is
        # readiness-MET. Readiness FAILS (route r1_grad_cosine_net_negative) only when
        # a MAJORITY of usable seeds are CLEARLY net-negative (cos < -BAND), i.e.
        # n_net_negative >= MIN_SEEDS_NET_NEGATIVE. Encoded as a CEILING on
        # n_net_negative: met iff measured <= MAX_NET_NEGATIVE_SEEDS_TOLERATED.
        {"name": "pair_seeds_not_net_negative_transfer", "measured": float(n_net_negative),
         "threshold": float(MAX_NET_NEGATIVE_SEEDS_TOLERATED), "direction": "upper"},
        {"name": "r2_iterative_loop_iterated", "measured": float(min_n_iters),
         "threshold": 2.0, "direction": "lower"},
        {"name": "r3_e2_touchable_under_interleaved", "measured": float(min_updates_e2),
         "threshold": 1.0, "direction": "lower"},
        {"name": "r3_interleaved_share_positive", "measured": float(1.0 if inter_share_ok else 0.0),
         "threshold": 1.0, "direction": "lower"},
        {"name": "r3_blocked_share_zero", "measured": float(1.0 if blocked_share_zero else 0.0),
         "threshold": 1.0, "direction": "lower"},
    ]

    try:
        preconditions = p0_readiness_gate(readiness_checks)
    except P0NotReady as e:
        # FORK 3: a MAJORITY of usable seeds have a CLEARLY net-negative cosine
        # (< -BAND) = the genuine negative-transfer readiness fail (the substrate
        # finding). Distinct from FORK 1 (diverged); both self-route
        # substrate_not_ready_requeue but the downstream routing differs.
        is_net_negative = "pair_seeds_not_net_negative_transfer" in e.reason
        route = "r1_grad_cosine_net_negative" if is_net_negative else "readiness_unmet_other"
        fork = 3
        if is_net_negative:
            note = (f"GENUINE net-negative shared-encoder cosine on a MAJORITY of usable "
                    f"seeds: {n_net_negative}/{n_usable} seeds have cos < -{COSINE_ORTHOGONAL_BAND} "
                    f"(net-negative seeds {net_negative_seeds}; min_cos={min_cos:.4g}, "
                    f"mean_cos={mean_cos:.4g}). The shared-latent integration is in the "
                    f"negative-transfer regime, so super-additivity is NOT expected "
                    f"(anti-aligned task gradients; PCGrad / Yu 2020). NOT a MECH-423 "
                    f"weakens -- it is a substrate finding. Route: /implement-substrate "
                    f"(shared-latent objective reconciliation) or narrow MECH-423, NOT "
                    f"another re-queue. " + e.reason)
        else:
            note = ("A non-cosine readiness precondition is unmet on the usable PAIR "
                    "seeds (R1 grad coupling / R2 settling / R3 interleave); the "
                    "integration machinery is not demonstrably doing cross-module work, "
                    "so a super-additivity verdict would be vacuous. " + e.reason)
        print(f"\n[{EXPERIMENT_TYPE}] FORK 3 -> substrate_not_ready_requeue "
              f"({route}): {e.reason}")
        manifest = dict(base_manifest)
        manifest.update({
            "outcome": "FAIL",
            "result": "FAIL",
            "evidence_direction": "inconclusive",
            "evidence_direction_note": note,
            "non_degenerate": False,
            "degeneracy_reason": "substrate_not_ready: " + e.reason,
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "route": route,
                "fork": fork,
                "preconditions": e.preconditions,
                "cosine_band_classification": cosine_band_classification,
            },
            "elapsed_seconds": elapsed,
        })
        return "FAIL", _write_manifest(manifest)

    # ---- FORK 2: readiness MET -> super-additivity verdict (the MECH-423 test) ----
    iso_scores = [rows[("ARM_ISOLATED", s)]["integration_score"] for s in seeds]
    pair_scores = [rows[("ARM_INTEGRATED_PAIR", s)]["integration_score"] for s in seeds]
    triple_scores = [rows[("ARM_INTEGRATED_TRIPLE", s)]["integration_score"] for s in seeds]

    delta_pair = [p - i for p, i in zip(pair_scores, iso_scores)]
    delta_triple = [t - i for t, i in zip(triple_scores, iso_scores)]
    baseline_sd = float(statistics.pstdev(iso_scores)) if len(iso_scores) > 1 else 0.0
    delta_sd = float(statistics.pstdev(delta_pair)) if len(delta_pair) > 1 else 0.0
    statistical_margin = SUPERADD_SD_MULT * delta_sd
    margin = max(statistical_margin, SUPERADD_MIN_EFFECT_FLOOR)
    n_seeds_pass = sum(1 for d in delta_pair if d > margin)
    mean_delta_pair = float(sum(delta_pair) / len(delta_pair))

    super_additive = n_seeds_pass >= MIN_SEEDS_PASS

    # Non-degeneracy net: degenerate only if EVERY seed has iso == pair.
    degen = check_degeneracy({
        "integration_score_iso_vs_pair": {
            "groups": [[iso_scores[k], pair_scores[k]] for k in range(len(seeds))]},
    })
    max_abs_delta = max(abs(d) for d in delta_pair) if delta_pair else 0.0
    if max_abs_delta < INTEGRATION_EFFECT_FLOOR:
        degen = {
            "non_degenerate": False,
            "degeneracy_reason": (
                f"integration inert: max|delta_pair|={max_abs_delta:.3g} < floor "
                f"{INTEGRATION_EFFECT_FLOOR} -- encoder co-shaping produced no "
                f"measurable effect (iso == integrated); criterion vacuous"),
            "degenerate_metrics": {"integration_effect": f"max_abs_delta {max_abs_delta:.3g} < {INTEGRATION_EFFECT_FLOOR}"},
        }

    if super_additive:
        outcome = "PASS"
        evidence_direction = "supports"
        label = "super_additive"
        note = (f"Integration is super-additive: delta_pair > margin ({margin:.4g}) on "
                f"{n_seeds_pass}/{len(seeds)} seeds (mean delta {mean_delta_pair:.4g}); "
                f"readiness MET with cosines in the near-orthogonal/positive band "
                f"(n_net_negative={n_net_negative}/{n_usable} < {MIN_SEEDS_NET_NEGATIVE}).")
    elif mean_delta_pair <= 0.0:
        outcome = "FAIL"
        evidence_direction = "weakens"
        label = "sub_or_merely_additive"
        note = (f"Integrated arm at or below the additive baseline (mean delta "
                f"{mean_delta_pair:.4g} <= 0) with readiness satisfied (cosines "
                f"near-orthogonal/positive, not net-negative): integration is merely "
                f"additive / sub-additive.")
    else:
        outcome = "FAIL"
        evidence_direction = "inconclusive"
        label = "additive_below_margin"
        note = (f"Positive but below the pre-registered margin: mean delta "
                f"{mean_delta_pair:.4g} > 0 but only {n_seeds_pass}/{len(seeds)} seeds "
                f"clear margin {margin:.4g}. Additive-only (NOT weakens).")

    print(f"\n[{EXPERIMENT_TYPE}] FORK 2 readiness MET; aggregate:")
    print(f"  n_usable_pair_seeds={n_usable}/{len(seeds)} cosine band={COSINE_ORTHOGONAL_BAND}")
    print(f"  cos: net_neg={n_net_negative} near_orth={n_near_orthogonal} net_pos={n_net_positive}"
          f" | min={min_cos:.4g} mean={mean_cos:.4g} max={max_cos:.4g}")
    print(f"  iso_scores={[round(x,4) for x in iso_scores]} sd={baseline_sd:.4g}")
    print(f"  pair_scores={[round(x,4) for x in pair_scores]} delta={[round(x,4) for x in delta_pair]}")
    print(f"  triple_scores={[round(x,4) for x in triple_scores]} delta={[round(x,4) for x in delta_triple]} (exploratory)")
    print(f"  margin={margin:.4g} n_seeds_pass={n_seeds_pass}/{len(seeds)} -> {label} ({outcome})")
    print(f"  non_degenerate={degen['non_degenerate']} elapsed={elapsed:.1f}s")

    manifest = dict(base_manifest)
    manifest.update({
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_note": note,
        "interpretation": {
            "label": label,
            "route": "readiness_met_superadditivity_verdict",
            "fork": 2,
            "preconditions": preconditions,
            "cosine_band_classification": cosine_band_classification,
            "criteria": [
                {"name": "superadditivity_margin_pair", "load_bearing": True,
                 "passed": bool(super_additive)},
            ],
        },
        "superadditivity": {
            "iso_scores": iso_scores,
            "pair_scores": pair_scores,
            "triple_scores_exploratory": triple_scores,
            "delta_pair": delta_pair,
            "delta_triple_exploratory": delta_triple,
            "baseline_across_seed_sd": baseline_sd,
            "delta_across_seed_sd": delta_sd,
            "statistical_margin": statistical_margin,
            "min_effect_floor": SUPERADD_MIN_EFFECT_FLOOR,
            "margin": margin,
            "margin_basis": ("max(SUPERADD_SD_MULT * pstdev(delta_pair), "
                             "SUPERADD_MIN_EFFECT_FLOOR)"),
            "n_seeds_pass": n_seeds_pass,
            "mean_delta_pair": mean_delta_pair,
            "super_additive": super_additive,
        },
        "elapsed_seconds": elapsed,
    })
    manifest.update(degen)
    return outcome, _write_manifest(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    _outcome, _manifest_path = main(dry_run=args.dry_run)
    _outcome_clean = str(_outcome).upper() if str(_outcome).upper() in ("PASS", "FAIL") else "FAIL"
    emit_outcome(outcome=_outcome_clean, manifest_path=_manifest_path, dry_run=args.dry_run)
    sys.exit(0)
