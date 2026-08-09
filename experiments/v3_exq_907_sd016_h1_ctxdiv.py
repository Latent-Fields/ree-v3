#!/opt/local/bin/python3
"""
V3-EXQ-907 -- SD-016 GOV-FANOUT-1 leg H1 (DRIVE axis): explicit context-divergence objective

EXPERIMENT_PURPOSE: diagnostic

QUESTION:
  Three independently-designed SD-016 selection mechanisms (418d/e/i q.k attention +
  population-diversity loss; 418m Path-3 tagger, untrained encoder; 898 Path-3 tagger,
  SD-070-TRAINED encoder) have all hit the identical uniform-softmax saddle
  (sel_entropy_mean -> ln(16)=2.7726). V3-EXQ-898's readiness gate proved the SD-070
  encoder fix genuinely takes on this apparatus (z_world varies across contexts,
  balanced-accuracy 0.95-0.99 hazard/resource decode), so its FAIL cleanly attributes
  to the SELECTION MECHANISM, not to "nothing to select on."

  The single unifying gap the three attempts share (per failure_autopsy_V3-EXQ-898):
  NO tested mechanism has ever been given a training signal that specifically rewards
  context-CONDITIONED divergence -- the safe-vs-dangerous selection distributions
  differing FROM EACH OTHER -- as distinct from population-level slot diversity
  (differing from uniform in aggregate, which Path 1 achieved and which did NOT help).

  H1 hypothesis (DRIVE axis): the tagger CAN represent context-selective slot
  distributions but the loss landscape never rewards them (terrain_loss is satisfied
  equally well by uniform mixing -- the ln(16) attractor). Add an auxiliary objective
  that DIRECTLY rewards sel_context_divergence during P1 and the tagger leaves the
  saddle.

  This is one leg of the three-axis GOV-FANOUT-1 discrimination portfolio designed in
  sd016_selection_fanout_portfolio_scope_staged_20260809.md (H1 drive / H3 algorithm /
  H2 representation). H1 and H3 attack orthogonal design axes; a null on both with
  readiness met would elevate H2 (representation) to the remaining live hypothesis.

SUBSTRATE REQUIREMENT -- NONE (driver-only probe, no /implement-substrate dependency):
  The obvious hook, E1DeepPredictor._last_cue_slot_weights, is .detach()ed at
  e1_deep.py:450 (a read-only diagnostic cache) so it carries NO gradient. But the
  driver does not need that cache: agent.e1.cue_slot_tagger is a PUBLIC nn.Sequential
  (world_dim -> hidden -> num_slots), so the driver calls it DIRECTLY on a batched
  safe/dangerous z_world pair, builds differentiable softmax weights itself, and adds
    -lambda * |mean_w_safe - mean_w_dang|_1
  (a batched safe/dangerous pair per P1 step) to the E1 loss. Empirically confirmed
  (Step 2.5a probe, 2026-08-09): tagger reachable, the divergence scalar has
  requires_grad=True, and backward() puts non-zero gradient into all 4 tagger tensors,
  while _last_cue_slot_weights.requires_grad is False. If H1 confirms the mechanism
  helps, THEN promote it to a real substrate knob (sd016_context_divergence_weight +
  E1.compute_context_divergence_loss) in a follow-on /implement-substrate -- NOT part
  of this probe.

DESIGN (inherits the V3-EXQ-898 harness; scope doc section 2 shared scaffold):
  Five arms x three seeds [42, 43, 44]:

    A0_OFF            sd016_cue_slot_tagger=False (legacy q.k attention -- C2 control,
                      reproduces ln(16)).
    A1_tagger         sd016_cue_slot_tagger=True, NO divergence loss = MATCHED-BUDGET
                      baseline (= the 898 ON arm, same P1 budget). This is what makes an
                      H1 positive attributable to the OBJECTIVE, not to any budget change.
    A2_ctxdiv_l0p1    tagger + auxiliary context-divergence loss, lambda=0.1
    A2_ctxdiv_l0p5    tagger + auxiliary context-divergence loss, lambda=0.5
    A2_ctxdiv_l2p0    tagger + auxiliary context-divergence loss, lambda=2.0

  LAMBDA SWEEP {0.1, 0.5, 2.0}, NOT the scope's illustrative {0.5, 1.0, 2.0}: a Step-2.5a
  foreground probe (2026-08-09, seed 43, P0a=30/P1=24) showed the divergence objective
  SATURATES by lambda=0.5 -- lambda=0.5 and lambda=2.0 gave bit-near-identical eval
  context-divergence (0.26868 vs 0.26872) while both left selection entropy at the saddle
  (2.7370 vs ln(16)=2.7726). So {0.5,1.0,2.0} would have been three saturated points;
  {0.1, 0.5, 2.0} spans the informative pre-saturation region (0.1) and confirms
  saturation at the top (2.0). The manipulation is confirmed LIVE (lambda=0.0 baseline
  eval_div 0.0159 vs lambda=0.5 eval_div 0.2687), so a flat sweep here is real saturation,
  not a clamp absorbing the manipulation (the multi-point-sweep guard, /queue-experiment
  Step 3.5).

  Per (arm, seed) CELL (wrapped in experiments._lib.arm_fingerprint.arm_cell, which
  resets ALL RNG on entry):

    P0a (encoder warmup, SD-070, world_dim=128): a dedicated warmup_env under
        RandomPolicy(seed) for P0A_EPISODES; run_zworld_p0 trains
        split_encoder.world_encoder + world_precision_logit via the validated recipe.
        RNG-neutral by construction.

    READINESS PRECONDITIONS (measured on THIS cell's trained encoder, before P1 -- the
    898 "does the fix actually take" check, load-bearing for validity):
      world_encoder_weights_moved   >=1 world-path tensor changed across P0a.
      z_world_spread_lift           mean per-dim std of z_world across an env-diverse
                                    batch, TRAINED/UNTRAINED on the SAME apparatus.
                                    Positive control: V3-EXQ-783 measured this recipe
                                    raise contrast_ratio ~1.7-1.9x.
    A cell whose readiness is unmet self-routes to substrate_not_ready_requeue for that
    SEED; fewer than a seed-majority ready self-routes the WHOLE RUN. All five arms of a
    seed share the SD-070 encoder recipe (the tagger flag changes only the post-encoder
    selection path), so readiness is arm-symmetric within a seed by construction -- there
    is no clean arm for an unready sibling to vacate (the 785 multi-regime hazard does
    not arise here; the preconditions are identical across arms of a seed).

    P1 (E1 selection training on DETACHED z_world -- phased-training compliant, the
        encoder optimiser is never stepped in P1): P1_EPISODES episodes alternating
        safe/dangerous env every CONTEXT_SWITCH_EVERY episodes, training ONLY
        agent.e1.parameters() (which includes cue_slot_tagger). Per step:
          terrain_loss (+ cue_action_loss in P1_late), matching 898 exactly, PLUS
          for A2 arms: -lambda * context_divergence, computed by calling
          agent.e1.cue_slot_tagger DIRECTLY on FIXED, DETACHED safe/dangerous z_world
          batches (collected once before P1 from the frozen encoder). Because the
          encoder is frozen in P1, those batches are stable; only the tagger changes, so
          the term is a clean differentiable pressure on the tagger. The DIV training
          batch (DIV_BATCH_SIZE each context) is SEPARATE from the eval batch, so C1b is
          a GENERALISATION readout, not the trained quantity (verdict-aliasing guard).

  Selection entropy / context-divergence measured identically to 898/418m: extract_cue_
  context over a fresh eval batch, read _last_cue_slot_weights. FULL per-context slot
  histograms (mean_w_safe[16], mean_w_dang[16]) are recorded per cell (scope doc section
  5 verdict-aliasing guard: the scalar L1-of-means could cancel structure living in
  higher moments, so a null can be inspected rather than assumed absent).

PER-ARM METRICS:
  sel_entropy_mean          mean slot-selection entropy over eval batch; uniform ln(16).
  sel_context_divergence    L1 distance between mean safe-vs-dangerous selection
                            distributions on the EVAL batch (C1b anti-degeneracy metric).
  sel_w_safe_hist[16] / sel_w_dang_hist[16]   full per-context selection histograms.
  train_ctx_div_final       (A2 arms) the context-divergence achieved on the TRAINING
                            batch after P1 -- distinguishes "objective never moved it"
                            from "moved it on train, failed to generalise to eval".
  action_bias_per_channel_std / action_bias_div  reported, not gated.

WHICH CRITERION IS LOAD-BEARING (a Step-2.5a probe finding, important for reading a FAIL):
  C1b (context-divergence) is the DIRECTLY-OPTIMISED quantity -- the auxiliary loss IS
  -lambda*|mean_w_safe - mean_w_dang|_1, so an A2 C1b pass is a MANIPULATION CHECK ("the
  drive took"), not independent evidence of selectivity. It is measured on a SEPARATE
  held-out eval batch, so it does test GENERALISATION (not a bit-identity), but it is not
  the surprising readout. C1 (entropy) is the FREE, non-optimised, scientifically
  LOAD-BEARING readout: does rewarding context-conditioning ALSO induce the sparse/peaky
  pattern-separation the architecture needs? The probe predicts C1b-PASS / C1-FAIL (the
  drive conditions selection on context but does NOT sparsify it), which is a DISCRIMINATING
  result, not a flat null -- see the FAIL sub-typing below.

ACCEPTANCE CRITERIA (computed over READY seeds only, majority = (n_ready//2)+1):
  C1  (PRIMARY scientific, load-bearing)  any A2 lambda arm sel_entropy_mean < 2.5 on a
                                 majority of ready seeds (sparse/selective retrieval).
  C1b (MANIPULATION CHECK + anti-degeneracy, load-bearing for the full-PASS gate) the SAME
                                 A2 lambda arm sel_context_divergence > 0.1 on a majority
                                 of ready seeds. In the full-PASS gate C1b guards against
                                 "constant-peaky" (C1 pass by collapsing to one slot for
                                 BOTH contexts); on its own it is the directly-optimised
                                 quantity and only confirms the drive took.
  C2  (CONTROL, load-bearing)   A0_OFF sel_entropy_mean > 2.65 on a majority of ready
                                 seeds (reproduces the saddle -- confirms the ablation
                                 isolates the selection change).

  COMBINATION RULE (recorded explicitly in the manifest):
    overall_pass = C2_pass AND (ANY A2 lambda arm has BOTH C1 and C1b on a majority of
    ready seeds). The C1/C1b legs are OR-ed across the three lambda sub-cells; the AND
    of C1 and C1b is enforced WITHIN a single lambda arm (breaks_saddle), never across
    arms. Attribution is additionally reported (non-gating): attribution_clean requires
    the A1_tagger matched-budget baseline NOT to break the saddle, so a positive
    attributes to the OBJECTIVE and not to the extra P1 budget.

  FOUR DISCRIMINATING OUTCOME REGIONS (interpretation.label; a FAIL is sub-typed, never a
  flat "insufficient"):
  - breaks_saddle (C1 AND C1b in one arm, C2 on saddle) -> PASS. H1 confirmed: the drive
    objective induces BOTH sparse AND context-conditioned selection. Route /failure-autopsy
    (PASS target) then a follow-on /implement-substrate promoting lambda to a real knob
    (sd016_context_divergence_weight).
  - context_conditioned_not_sparse (any A2 C1b pass, no A2 C1 pass, C2 on saddle) -> FAIL,
    but INFORMATIVE (the probe-predicted outcome). The drive induces generalising context-
    conditioned divergence but NOT sparsity. Rules OUT "the tagger merely lacked the right
    gradient"; implicates the ALGORITHM axis (H3 hard/competitive selection) for sparsity,
    or pairing the divergence term with an explicit entropy floor.
  - sparse_not_context_conditioned (any A2 C1 pass, no A2 C1b pass, C2 on saddle) -> FAIL.
    The scope's named degeneracy (constant-peaky -- sparse but the same slot for both
    contexts). Route: pair the divergence term with an entropy floor.
  - objective_insufficient (no A2 C1 and no A2 C1b, readiness met, C2 on saddle) -> FAIL.
    The true null: even a directly-rewarded context-divergence objective produced nothing
    generalising. Does NOT rule out H2 (wrong retrieval unit) or H3.
  - substrate_inconsistent_off_arm_off_saddle (C2 fail) -> OFF arm not on the saddle;
    investigate before interpreting C1/C1b.

DV-SYMMETRY CHECK (per /queue-experiment Step 3 -- one sentence per arm; name the DV's
  symmetry group and confirm the manipulation is not invariant under it):
  - A0_OFF: the tagger flag changes WHICH function maps z_world -> slot logits (q.k
    dot-product vs an independent feedforward MLP with its own random init) -- not a
    broadcast constant, monotone rescaling, or permutation of the ON arm's logits -- so
    sel_entropy_mean / sel_context_divergence are free to differ across arms.
  - A1_tagger: a distinct MLP with its own init trained by terrain_loss only; its DV is
    not structurally pinned to the OFF arm's.
  - A2_ctxdiv_l*: the manipulation ADDS -lambda*|mean_w_safe - mean_w_dang|_1, a term
    designed to MOVE the C1b DV, so it is deliberately NOT invariant under it (the opposite
    of the DV-symmetry failure mode). Two honesty caveats follow and are load-bearing for
    reading the result: (1) because C1b is the DIRECTLY-OPTIMISED quantity, an A2 C1b pass
    is a MANIPULATION CHECK, not independent evidence of selectivity -- it is measured on a
    SEPARATE held-out eval batch (so it tests GENERALISATION, not a bit-identity fixed
    before the run) but it is not surprising and does not on its own establish "selection
    became context-selective". (2) C1 (entropy) is NOT targeted by the loss at all, so it
    is the free, non-optimised, scientifically decisive readout -- and it is the criterion
    the interpretation sub-typing routes on. Neither DV is a broadcast-constant / monotone /
    permutation image of the OFF arm's DV, so cross-arm comparisons are well posed.

claim_ids: []  (diagnostic -- weights no SD-016 claim-confidence directly, matching the
  V3-EXQ-418g/418m/898 precedent for this exact acceptance-test shape; a PASS/FAIL here
  requires /failure-autopsy adjudication before any claims.yaml action).
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id: ends _v3
"""

import sys
import argparse
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.capability_eval import RandomPolicy
from experiments._lib.zworld_p0_warmup import run_zworld_p0
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator


EXPERIMENT_TYPE = "v3_exq_907_sd016_h1_ctxdiv"
CLAIM_IDS: List[str] = []          # diagnostic: substrate/objective probe, weights no claim
EXPERIMENT_PURPOSE = "diagnostic"

# The readiness preconditions below are the same 898 metrics (world_encoder_weights_moved
# binary tensor-changed count; z_world_spread_lift spread ratio) -- neither has a frozen
# reference_cells literal for a script that has not executed once. Thresholds are grounded
# in V3-EXQ-783's measured SD-070 lift (~1.7-1.9x) and the V3-EXQ-737a/728 wiring-failure
# signature (0 tensors moved), exactly as in V3-EXQ-898.
ANCHOR_REACHABILITY_EXEMPT = (
    "no frozen reference_cells literal exists for this script's own metrics pre-run; "
    "readiness thresholds are grounded in V3-EXQ-783's measured SD-070 lift (~1.7-1.9x) "
    "and the V3-EXQ-737a/728 wiring-failure signature (0 tensors moved), inherited from "
    "V3-EXQ-898 -- see module docstring READINESS PRECONDITIONS section"
)

WORLD_DIM            = 128         # the V3-EXQ-783 D128_TRAINED axis
P0A_EPISODES         = 60          # SD-070's validated operating point
P1_EPISODES          = 40          # unchanged from V3-EXQ-898/418m
STEPS_PER_EPISODE    = 150
CONTEXT_SWITCH_EVERY = 5
LAMBDA_TERRAIN       = 0.1
LAMBDA_CUE_ACTION    = 0.5
EVAL_BATCH_SIZE      = 32
READINESS_BATCH_SIZE = 16          # cheap positive-control probe batch, pre-P1
DIV_BATCH_SIZE       = 64          # per-context training batch for the divergence loss
LR                   = 1e-4
SEEDS                = [42, 43, 44]

# Arm spec: (label, cue_slot_tagger, ctxdiv_lambda). lambda None => no tagger (OFF);
# lambda 0.0 => tagger, no divergence loss (matched-budget baseline); lambda > 0 => A2.
# LAMBDA SWEEP {0.1, 0.5, 2.0} (NOT the scope's illustrative {0.5, 1.0, 2.0}): a Step-2.5a
# foreground probe (2026-08-09, seed 43, P0a=30/P1=24) showed the divergence objective
# SATURATES by lambda=0.5 -- lambda=0.5 and lambda=2.0 gave bit-near-identical eval
# context-divergence (0.26868 vs 0.26872) while both left selection entropy at the saddle
# (2.7370 vs ln(16)=2.7726). So {0.5,1.0,2.0} would have been three saturated points;
# {0.1, 0.5, 2.0} spans the informative pre-saturation region (0.1) and confirms
# saturation at the top (2.0). lambda=0.0 baseline (A1_tagger) gave eval_div 0.0159.
ARMS: List[Tuple[str, bool, Optional[float]]] = [
    ("A0_OFF",         False, None),
    ("A1_tagger",      True,  0.0),
    ("A2_ctxdiv_l0p1", True,  0.1),
    ("A2_ctxdiv_l0p5", True,  0.5),
    ("A2_ctxdiv_l2p0", True,  2.0),
]
A2_LABELS = [lab for lab, _, lam in ARMS if lam is not None and lam > 0.0]

SEL_ENTROPY_C1_THRESHOLD  = 2.5
SEL_CONTEXT_DIV_THRESHOLD = 0.1
SEL_ENTROPY_C2_FLOOR      = 2.65
UNIFORM_REFERENCE         = None   # filled at runtime = ln(num_slots)

WORLD_ENCODER_WEIGHT_MOVE_FLOOR = 1     # >=1 tensor changed
Z_WORLD_SPREAD_LIFT_FLOOR       = 1.3   # trained/untrained spread ratio (783 measured ~1.7-1.9x)


# ---------------------------------------------------------------------------
# Env + agent helpers (mirror V3-EXQ-898 exactly).
# ---------------------------------------------------------------------------

def _make_env_safe(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed, size=8, num_hazards=1, num_resources=3,
        hazard_harm=0.02, use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_env_dangerous(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed + 1000, size=8, num_hazards=5, num_resources=3,
        hazard_harm=0.04, use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_warmup_env(seed: int) -> CausalGridWorldV2:
    """A DEDICATED env for P0a -- never the training env (run_zworld_p0's own contract)."""
    return CausalGridWorldV2(
        seed=seed + 5000, size=8, num_hazards=1, num_resources=3,
        hazard_harm=0.02, use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2, cue_slot_tagger: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        sd016_enabled=True,
        sd016_writepath_mode="off",
        sd016_cue_slot_tagger=cue_slot_tagger,
        sws_enabled=False,
        rem_enabled=False,
        shy_enabled=False,
    )
    return REEAgent(cfg)


def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def get_hazard_max(obs_dict, world_obs):
    if "harm_obs" in obs_dict:
        harm_obs = obs_dict["harm_obs"]
        if hasattr(harm_obs, "shape") and harm_obs.shape[-1] >= 26:
            return float(harm_obs[..., :25].max().item())
    if "hazard_field_view" in obs_dict:
        hfv = obs_dict["hazard_field_view"]
        if hasattr(hfv, "shape"):
            return float(hfv.max().item())
    if world_obs is not None and world_obs.shape[-1] >= 225:
        return float(world_obs[..., 200:225].max().item())
    return 0.0


def compute_terrain_loss(agent, z_world_detached, hazard_max):
    _, terrain_weight = agent.e1.extract_cue_context(z_world_detached)
    w_harm_target = 0.8 if hazard_max > 0.3 else 0.2
    w_goal_target = 0.8 if hazard_max < 0.1 else 0.3
    target = torch.tensor([[w_harm_target, w_goal_target]],
                          dtype=terrain_weight.dtype,
                          device=terrain_weight.device)
    return F.mse_loss(terrain_weight, target)


def compute_cue_action_loss(agent, z_world_detached, action):
    action_bias, _ = agent.e1.extract_cue_context(z_world_detached)
    with torch.no_grad():
        ao_target = agent.e2.action_object(z_world_detached, action.detach())
    return F.mse_loss(action_bias, ao_target.detach())


# ---------------------------------------------------------------------------
# H1 auxiliary objective: differentiable context-divergence on the tagger.
# Calls agent.e1.cue_slot_tagger DIRECTLY (a public nn.Sequential) -- the
# _last_cue_slot_weights cache is .detach()ed at e1_deep.py:450 and cannot be
# used. z_safe/z_dang are FIXED detached batches from the frozen encoder, so the
# only differentiable path is through the tagger (which is in agent.e1.parameters()).
# ---------------------------------------------------------------------------

def _context_divergence(agent, z_safe_div, z_dang_div, temp: float) -> torch.Tensor:
    logits_safe = agent.e1.cue_slot_tagger(z_safe_div)   # [B, num_slots], differentiable
    logits_dang = agent.e1.cue_slot_tagger(z_dang_div)
    w_safe = F.softmax(logits_safe / temp, dim=-1).mean(dim=0)   # [num_slots]
    w_dang = F.softmax(logits_dang / temp, dim=-1).mean(dim=0)
    return (w_safe - w_dang).abs().sum()


# ---------------------------------------------------------------------------
# Selection / output probes (unchanged from V3-EXQ-898/418m).
# ---------------------------------------------------------------------------

def _selection_entropy_mean(agent, z_world_batch) -> float:
    with torch.no_grad():
        agent.e1.extract_cue_context(z_world_batch)
        w = agent.e1._last_cue_slot_weights.clamp(min=1e-12)
        return float(-(w * w.log()).sum(dim=-1).mean().item())


def _action_bias_per_channel_std(agent, z_world_batch) -> float:
    with torch.no_grad():
        action_bias, _ = agent.e1.extract_cue_context(z_world_batch)
        return float(action_bias.std(dim=0).mean().item())


def _action_bias_divergence(agent, z_safe, z_dang) -> float:
    with torch.no_grad():
        ab_safe, _ = agent.e1.extract_cue_context(z_safe)
        ab_dang, _ = agent.e1.extract_cue_context(z_dang)
        return float((ab_safe.mean(dim=0) - ab_dang.mean(dim=0)).norm().item())


def _selection_histograms(agent, z_safe, z_dang) -> Tuple[List[float], List[float], float]:
    """Full per-context selection histograms (mean over the eval batch) + the L1
    context-divergence derived from them, so the scalar and the histograms are
    guaranteed consistent (scope doc section 5 verdict-aliasing guard)."""
    with torch.no_grad():
        agent.e1.extract_cue_context(z_safe)
        w_safe = agent.e1._last_cue_slot_weights.mean(dim=0)   # [num_slots]
        agent.e1.extract_cue_context(z_dang)
        w_dang = agent.e1._last_cue_slot_weights.mean(dim=0)
        ctx_div = float((w_safe - w_dang).abs().sum().item())
        return w_safe.tolist(), w_dang.tolist(), ctx_div


def _z_world_spread(agent, env, n_samples: int) -> float:
    """Mean per-dimension std of z_world across a diverse observation batch (the
    readiness-precondition statistic -- spread, matching what C1/C1b depend on
    upstream), measured identically before and after P0a for a positive control."""
    batch = _collect_eval_batch(agent, env, n_samples)
    return float(batch.std(dim=0).mean().item())


# ---------------------------------------------------------------------------
# Eval batch collector (mirrors V3-EXQ-898/418m).
# ---------------------------------------------------------------------------

def _collect_eval_batch(agent, env, n_samples: int) -> torch.Tensor:
    device = agent.device
    z_world_list: List[torch.Tensor] = []
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    for _ in range(STEPS_PER_EPISODE * 4):
        ob = obs_dict["body_state"]
        ow = obs_dict["world_state"]
        ob = ob.to(device) if torch.is_tensor(ob) else torch.tensor(ob, dtype=torch.float32, device=device)
        ow = ow.to(device) if torch.is_tensor(ow) else torch.tensor(ow, dtype=torch.float32, device=device)
        if ob.dim() == 1:
            ob = ob.unsqueeze(0)
        if ow.dim() == 1:
            ow = ow.unsqueeze(0)
        with torch.no_grad():
            latent = agent.sense(ob, ow)
        z_world_list.append(latent.z_world.detach().clone().squeeze(0))
        if len(z_world_list) >= n_samples:
            break
        action_idx = random.randint(0, env.action_dim - 1)
        action = _onehot(action_idx, env.action_dim, device)
        with torch.no_grad():
            _, _h, done, _i, obs_dict = env.step(action)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            agent.e1.reset_hidden_state()
    return torch.stack(z_world_list[:n_samples], dim=0)


# ---------------------------------------------------------------------------
# P1 training episode -- E1 ONLY, on DETACHED z_world (phased-training compliant).
# For A2 arms adds -ctxdiv_lambda * context_divergence each step, computed on the
# FIXED detached safe/dangerous divergence batches (encoder frozen in P1).
# ---------------------------------------------------------------------------

def _run_training_episode(agent, env, optimizer, phase: str,
                          ctxdiv_lambda: Optional[float],
                          z_safe_div: Optional[torch.Tensor],
                          z_dang_div: Optional[torch.Tensor],
                          div_temp: float) -> int:
    device = agent.device
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    ep_steps = 0

    for _step in range(STEPS_PER_EPISODE):
        ob = obs_dict["body_state"]
        ow = obs_dict["world_state"]
        ob = ob.to(device) if torch.is_tensor(ob) else torch.tensor(ob, dtype=torch.float32, device=device)
        ow = ow.to(device) if torch.is_tensor(ow) else torch.tensor(ow, dtype=torch.float32, device=device)
        if ob.dim() == 1:
            ob = ob.unsqueeze(0)
        if ow.dim() == 1:
            ow = ow.unsqueeze(0)

        with torch.no_grad():
            latent = agent.sense(ob, ow)
        z_world_det = latent.z_world.detach()
        agent.clock.advance()

        hazard_max = get_hazard_max(obs_dict, ow)
        action_idx = random.randint(0, env.action_dim - 1)
        action = _onehot(action_idx, env.action_dim, device)

        with torch.no_grad():
            _, harm_signal, done, _info, obs_dict = env.step(action)
        ep_steps += 1

        t_loss = compute_terrain_loss(agent, z_world_det, hazard_max)
        total_loss = LAMBDA_TERRAIN * t_loss

        if phase == "P1_late":
            ca_loss = compute_cue_action_loss(agent, z_world_det, action)
            total_loss = total_loss + LAMBDA_CUE_ACTION * ca_loss

        # H1 DRIVE term: reward context-conditioned divergence (maximise -> subtract).
        if ctxdiv_lambda and ctxdiv_lambda > 0.0:
            div = _context_divergence(agent, z_safe_div, z_dang_div, div_temp)
            total_loss = total_loss - ctxdiv_lambda * div

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
        optimizer.step()

        agent.update_residue(float(harm_signal) if float(harm_signal) < 0 else 0.0)

        if done:
            break

    return ep_steps


# ---------------------------------------------------------------------------
# Per-cell run (one arm x one seed).
# ---------------------------------------------------------------------------

def _run_one_cell(arm_label: str, tagger: bool, ctxdiv_lambda: Optional[float], seed: int,
                   p0a: int, p1: int, eval_n: int, readiness_n: int, div_n: int,
                   dry_run: bool, zg_acc: ZGoalStreamAccumulator) -> Dict[str, Any]:
    config_slice = {
        "lineage": "v3_exq_907_sd016_h1_ctxdiv",
        "arm": arm_label,
        "cue_slot_tagger": tagger,
        "ctxdiv_lambda": ctxdiv_lambda,
        "world_dim": WORLD_DIM,
        "p0a_recipe": "sd070",
        "p0a_episodes": p0a,
        "p1_episodes": p1,
        "steps_per_episode": STEPS_PER_EPISODE,
        "div_batch_size": div_n,
    }

    with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
        env_safe = _make_env_safe(seed)
        env_dang = _make_env_dangerous(seed)
        warmup_env = _make_warmup_env(seed)
        agent = _make_agent(env_safe, tagger)

        print(f"Seed {seed} Condition {arm_label}", flush=True)
        print(f"  [arm {arm_label} seed {seed}] cue_slot_tagger={tagger} "
              f"ctxdiv_lambda={ctxdiv_lambda} world_dim={WORLD_DIM} p0a={p0a} p1={p1}",
              flush=True)

        # -- readiness positive control: BEFORE P0a -----------------------------------
        z_spread_untrained = _z_world_spread(agent, env_safe, readiness_n)

        # -- P0a: SD-070 encoder warmup (RNG-neutral) ----------------------------------
        world_path_before = {
            n: p.detach().clone()
            for n, p in agent.latent_stack.named_parameters()
            if "world_encoder" in n or "world_precision_logit" in n
        }
        policy = RandomPolicy(seed=seed)
        p0a_stats = run_zworld_p0(
            agent, warmup_env, seed, p0a, STEPS_PER_EPISODE, policy,
            label=arm_label, dry_run=dry_run,
        )
        n_moved = 0
        for n, p in agent.latent_stack.named_parameters():
            if n in world_path_before:
                if float((p.detach() - world_path_before[n]).abs().max().item()) > 0.0:
                    n_moved += 1

        # -- readiness positive control: AFTER P0a -------------------------------------
        z_spread_trained = _z_world_spread(agent, env_safe, readiness_n)
        spread_lift = (
            z_spread_trained / z_spread_untrained
            if z_spread_untrained > 1e-12 else 0.0
        )

        preconditions = [
            {
                "name": "world_encoder_weights_moved",
                "description": (
                    "world_encoder/world_precision_logit tensors changed across P0a -- "
                    "the exact V3-EXQ-737a/728 wiring-failure signature (0 moved) this "
                    "recipe exists to fix."
                ),
                "control": "world-path tensors, this cell",
                "measured": float(n_moved),
                "threshold": float(WORLD_ENCODER_WEIGHT_MOVE_FLOOR),
                "direction": "lower",
                "met": n_moved >= WORLD_ENCODER_WEIGHT_MOVE_FLOOR,
            },
            {
                "name": "z_world_spread_lift",
                "description": (
                    "z_world batch spread (mean per-dim std), trained/untrained ratio on "
                    "the SAME apparatus -- the same kind of statistic (spread, not "
                    "magnitude) the C1/C1b selection criteria depend on upstream. "
                    "Positive control: V3-EXQ-783 measured this recipe raise "
                    "contrast_ratio ~1.7-1.9x."
                ),
                "control": "same agent/env, before vs after P0a",
                "measured": float(spread_lift),
                "threshold": float(Z_WORLD_SPREAD_LIFT_FLOOR),
                "direction": "lower",
                "met": spread_lift >= Z_WORLD_SPREAD_LIFT_FLOOR,
            },
        ]
        cell_ready = all(p["met"] for p in preconditions)

        result: Dict[str, Any] = {
            "arm": arm_label,
            "cue_slot_tagger": tagger,
            "ctxdiv_lambda": ctxdiv_lambda,
            "seed": seed,
            "ready": cell_ready,
            "preconditions": preconditions,
            "p0a_stats": p0a_stats,
            "z_spread_untrained": z_spread_untrained,
            "z_spread_trained": z_spread_trained,
        }

        if not cell_ready:
            print(f"  [READINESS-UNMET] {arm_label} seed={seed} "
                  f"moved={n_moved} spread_lift={spread_lift:.4f}", flush=True)
            print("verdict: FAIL", flush=True)
            zg_acc.observe(agent)
            cell.stamp(result)
            return result

        # -- divergence training batches (fixed, detached, from the frozen encoder) ----
        div_temp = max(float(getattr(agent.e1, "_sd016_cue_slot_tagger_temperature", 1.0)), 1e-3)
        z_safe_div = z_dang_div = None
        if ctxdiv_lambda and ctxdiv_lambda > 0.0:
            z_safe_div = _collect_eval_batch(agent, env_safe, div_n).detach()
            z_dang_div = _collect_eval_batch(agent, env_dang, div_n).detach()

        # -- P1: E1 selection training on DETACHED z_world -----------------------------
        optimizer = optim.Adam(agent.e1.parameters(), lr=LR)
        half = p1 // 2
        for ep in range(p1):
            phase = "P1_early" if ep < half else "P1_late"
            env = env_dang if (ep // CONTEXT_SWITCH_EVERY) % 2 == 1 else env_safe
            _run_training_episode(agent, env, optimizer, phase,
                                  ctxdiv_lambda, z_safe_div, z_dang_div, div_temp)
            if (ep + 1) % 50 == 0 or (ep + 1) == p1:
                print(f"  [train] {arm_label} seed={seed} ep {ep+1}/{p1}", flush=True)

        # -- final training-batch divergence (diagnostic: train vs eval generalisation) -
        train_ctx_div_final = None
        if ctxdiv_lambda and ctxdiv_lambda > 0.0:
            with torch.no_grad():
                train_ctx_div_final = float(
                    _context_divergence(agent, z_safe_div, z_dang_div, div_temp).item())

        z_safe = _collect_eval_batch(agent, env_safe, eval_n)
        z_dang = _collect_eval_batch(agent, env_dang, eval_n)

        sel_ent = _selection_entropy_mean(agent, z_safe)
        w_safe_hist, w_dang_hist, sel_ctx = _selection_histograms(agent, z_safe, z_dang)
        ab_std = _action_bias_per_channel_std(agent, z_safe)
        ab_div = _action_bias_divergence(agent, z_safe, z_dang)

        print(f"  [arm {arm_label} seed {seed}] "
              f"sel_entropy={sel_ent:.4f} sel_ctx_div={sel_ctx:.4f} "
              f"train_ctx_div={train_ctx_div_final if train_ctx_div_final is None else round(train_ctx_div_final,4)} "
              f"action_bias_std={ab_std:.6f} action_bias_div={ab_div:.4f}", flush=True)
        print("verdict: PASS", flush=True)

        result.update({
            "sel_entropy_mean": sel_ent,
            "sel_context_divergence": sel_ctx,
            "sel_w_safe_hist": w_safe_hist,
            "sel_w_dang_hist": w_dang_hist,
            "train_ctx_div_final": train_ctx_div_final,
            "action_bias_per_channel_std": ab_std,
            "action_bias_div": ab_div,
        })
        zg_acc.observe(agent)
        cell.stamp(result)
        return result


# ---------------------------------------------------------------------------
# Aggregation + acceptance.
# ---------------------------------------------------------------------------

def _summarise(ready_results: List[Dict]) -> Dict:
    n = len(ready_results)
    if n == 0:
        return {"n_seeds": 0}
    return {
        "n_seeds":                          n,
        "sel_entropy_mean_mean":            sum(r["sel_entropy_mean"] for r in ready_results) / n,
        "sel_entropy_mean_min":             min(r["sel_entropy_mean"] for r in ready_results),
        "sel_entropy_mean_max":             max(r["sel_entropy_mean"] for r in ready_results),
        "sel_context_divergence_mean":      sum(r["sel_context_divergence"] for r in ready_results) / n,
        "sel_context_divergence_min":       min(r["sel_context_divergence"] for r in ready_results),
        "sel_context_divergence_max":       max(r["sel_context_divergence"] for r in ready_results),
        "action_bias_per_channel_std_mean": sum(r["action_bias_per_channel_std"] for r in ready_results) / n,
        "action_bias_div_mean":             sum(r["action_bias_div"] for r in ready_results) / n,
    }


def _evaluate(per_arm_ready: Dict[str, List[Dict]], n_ready_seeds: int) -> Dict:
    majority = (n_ready_seeds // 2) + 1 if n_ready_seeds > 0 else 0

    off = per_arm_ready.get("A0_OFF", [])
    c2_seeds_pass = sum(1 for r in off if r["sel_entropy_mean"] > SEL_ENTROPY_C2_FLOOR)
    c2_pass = n_ready_seeds > 0 and c2_seeds_pass >= majority

    # Per A2 lambda arm: C1 AND C1b enforced WITHIN the arm (breaks_saddle).
    a2_arms: Dict[str, Dict] = {}
    for lab in A2_LABELS:
        rs = per_arm_ready.get(lab, [])
        c1_seeds = sum(1 for r in rs if r["sel_entropy_mean"] < SEL_ENTROPY_C1_THRESHOLD)
        c1b_seeds = sum(1 for r in rs if r["sel_context_divergence"] > SEL_CONTEXT_DIV_THRESHOLD)
        c1_arm = n_ready_seeds > 0 and c1_seeds >= majority
        c1b_arm = n_ready_seeds > 0 and c1b_seeds >= majority
        a2_arms[lab] = {
            "c1_seeds_pass": c1_seeds, "c1b_seeds_pass": c1b_seeds,
            "c1_pass": c1_arm, "c1b_pass": c1b_arm,
            "breaks_saddle": c1_arm and c1b_arm,
        }
    breaking = [lab for lab, v in a2_arms.items() if v["breaks_saddle"]]
    c1_c1b_pass = len(breaking) > 0
    # Sub-typing signals (a FAIL is discriminating, not a flat "insufficient"):
    #   C1b (context-divergence) is the DIRECTLY-OPTIMISED quantity -- an A2 pass here is a
    #   MANIPULATION CHECK ("the drive took"), not independent evidence of selectivity; it is
    #   measured on a held-out eval batch so it does test GENERALISATION, not a bit-identity.
    #   C1 (entropy) is the FREE, non-optimised, scientifically load-bearing readout: does
    #   rewarding context-conditioning ALSO induce the sparse/peaky pattern-separation the
    #   architecture needs? The Step-2.5a probe predicts C1b-pass / C1-fail (drive conditions
    #   but does not sparsify) -> implicates the ALGORITHM axis (H3), not "no gradient".
    any_c1b_pass = any(v["c1b_pass"] for v in a2_arms.values())
    any_c1_pass = any(v["c1_pass"] for v in a2_arms.values())

    # A1 matched-budget baseline (attribution, non-gating).
    a1 = per_arm_ready.get("A1_tagger", [])
    a1_c1_seeds = sum(1 for r in a1 if r["sel_entropy_mean"] < SEL_ENTROPY_C1_THRESHOLD)
    a1_c1b_seeds = sum(1 for r in a1 if r["sel_context_divergence"] > SEL_CONTEXT_DIV_THRESHOLD)
    a1_breaks = (n_ready_seeds > 0 and a1_c1_seeds >= majority and a1_c1b_seeds >= majority)
    attribution_clean = c1_c1b_pass and not a1_breaks

    overall = c1_c1b_pass and c2_pass

    return {
        "combination_rule": (
            "overall_pass = C2_pass AND (any A2 lambda arm has C1[entropy<2.5] AND "
            "C1b[ctxdiv>0.1] on a majority of ready seeds; the C1&C1b AND is enforced "
            "WITHIN one lambda arm, OR-ed across the three lambda sub-cells). "
            "attribution_clean additionally requires A1_tagger (matched budget) NOT to "
            "break the saddle."
        ),
        "C1_C1b_any_lambda_breaks_saddle": {
            "pass": c1_c1b_pass, "breaking_lambda_arms": breaking,
            "majority": majority, "load_bearing": True,
            "c1_threshold": SEL_ENTROPY_C1_THRESHOLD,
            "c1b_threshold": SEL_CONTEXT_DIV_THRESHOLD,
        },
        "C2_off_arm_on_saddle": {
            "pass": c2_pass, "seeds_pass": c2_seeds_pass, "majority": majority,
            "floor": SEL_ENTROPY_C2_FLOOR, "load_bearing": True,
        },
        "per_a2_arm": a2_arms,
        "any_c1b_pass": any_c1b_pass,
        "any_c1_pass": any_c1_pass,
        "a1_matched_budget_baseline": {
            "c1_seeds_pass": a1_c1_seeds, "c1b_seeds_pass": a1_c1b_seeds,
            "breaks_saddle": a1_breaks,
        },
        "attribution_clean": attribution_clean,
        "overall_pass": overall,
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> Dict:
    global UNIFORM_REFERENCE
    UNIFORM_REFERENCE = math.log(16)
    t0 = time.perf_counter()

    p0a        = P0A_EPISODES        if not dry_run else 2
    p1         = P1_EPISODES         if not dry_run else 3
    eval_n     = EVAL_BATCH_SIZE     if not dry_run else 4
    readiness_n = READINESS_BATCH_SIZE if not dry_run else 4
    div_n      = DIV_BATCH_SIZE      if not dry_run else 4

    print(f"V3-EXQ-907 SD-016 H1 context-divergence objective "
          f"(seeds={SEEDS} arms={[a[0] for a in ARMS]} dry_run={dry_run} "
          f"uniform_ref={UNIFORM_REFERENCE:.4f} world_dim={WORLD_DIM})", flush=True)

    zg_acc = ZGoalStreamAccumulator()
    all_cells: List[Dict] = []
    per_arm_all: Dict[str, List[Dict]] = {lab: [] for lab, _, _ in ARMS}

    for seed in SEEDS:
        for arm_label, tagger, ctxdiv_lambda in ARMS:
            r = _run_one_cell(arm_label, tagger, ctxdiv_lambda, seed,
                              p0a, p1, eval_n, readiness_n, div_n, dry_run, zg_acc)
            all_cells.append(r)
            per_arm_all[arm_label].append(r)

    n_seeds_total = len(SEEDS)
    ready_seeds = [
        s for s in SEEDS
        if all(
            next(r for r in per_arm_all[lab] if r["seed"] == s)["ready"]
            for lab, _, _ in ARMS
        )
    ]
    n_ready = len(ready_seeds)
    seed_majority = (n_seeds_total // 2) + 1

    flat_preconditions = []
    for r in all_cells:
        for p in r["preconditions"]:
            flat_preconditions.append({**p, "name": f"{r['arm']}::seed{r['seed']}::{p['name']}"})

    if n_ready < seed_majority:
        outcome = "FAIL"
        acceptance = None
        label = "substrate_not_ready_requeue"
        criteria_non_degenerate = {}
        print(f"  [summary] READINESS UNMET: {n_ready}/{n_seeds_total} seeds ready "
              f"(need >= {seed_majority}) -> {label}", flush=True)
    else:
        per_arm_ready = {
            lab: [r for r in per_arm_all[lab] if r["ready"]]
            for lab, _, _ in ARMS
        }
        acceptance = _evaluate(per_arm_ready, n_ready)
        outcome = "PASS" if acceptance["overall_pass"] else "FAIL"
        # Sub-type the verdict so a FAIL is discriminating, not a flat "insufficient".
        # C1 (entropy, free readout) is the load-bearing scientific criterion; C1b
        # (context-divergence, directly optimised) is a generalising manipulation check.
        if outcome == "PASS":
            # C1 AND C1b within one lambda arm, AND C2 control -> full success.
            label = "sd016_h1_ctxdiv_breaks_saddle"
        elif not acceptance["C2_off_arm_on_saddle"]["pass"]:
            label = "sd016_substrate_inconsistent_off_arm_off_saddle"
        elif acceptance["any_c1b_pass"] and not acceptance["any_c1_pass"]:
            # THE PROBE-PREDICTED OUTCOME: the drive induces generalising context-
            # conditioned divergence but NOT sparse/low-entropy selection. Rules OUT
            # "the tagger merely lacked the right gradient"; implicates the ALGORITHM
            # axis (H3 hard/competitive selection) for sparsity, or pairing the
            # divergence term with an explicit entropy floor.
            label = "sd016_h1_ctxdiv_context_conditioned_not_sparse"
        elif acceptance["any_c1_pass"] and not acceptance["any_c1b_pass"]:
            # The scope's named degeneracy (constant-peaky): sparse but the same slot for
            # both contexts. Route: pair the divergence term with an entropy floor.
            label = "sd016_h1_ctxdiv_sparse_not_context_conditioned"
        else:
            # Both fail with readiness met and C2 on the saddle: the true null -- even a
            # directly-rewarded context-divergence objective produced nothing generalising.
            label = "sd016_h1_ctxdiv_objective_insufficient_under_trained_encoder"
        # Non-degeneracy is established by the readiness gate above (same argument as
        # V3-EXQ-898): the selection statistics are measured on an encoder already proven
        # (per-cell preconditions) to vary across observations, so a criterion cannot pass
        # here for the "z_world is constant" reason that would make it vacuous.
        criteria_non_degenerate = {
            "C1_C1b_any_lambda_breaks_saddle": True,
            "C2_off_arm_on_saddle": True,
        }
        print(f"  [summary] ready={n_ready}/{n_seeds_total} "
              f"C1C1b={acceptance['C1_C1b_any_lambda_breaks_saddle']['pass']} "
              f"(lambda={acceptance['C1_C1b_any_lambda_breaks_saddle']['breaking_lambda_arms']}) "
              f"C2={acceptance['C2_off_arm_on_saddle']['pass']} "
              f"attribution_clean={acceptance['attribution_clean']} -> outcome={outcome}",
              flush=True)

    per_arm_ready_for_summary = {
        lab: [r for r in per_arm_all[lab] if r["ready"]]
        for lab, _, _ in ARMS
    }
    summaries = {arm: _summarise(rs) for arm, rs in per_arm_ready_for_summary.items()}

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    output: Dict[str, Any] = {
        "experiment_type":    EXPERIMENT_TYPE,
        "run_id":             run_id,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids":          CLAIM_IDS,
        "claim_ids_tested":   CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome":            outcome,
        "timestamp_utc":      ts,
        "evidence_direction": "diagnostic",
        "evidence_direction_note": (
            "SD-016 GOV-FANOUT-1 leg H1 (DRIVE axis): does an explicit context-CONDITIONED "
            "divergence objective break the uniform-selection saddle three prior SD-016 "
            "selection mechanisms all hit? Driver-only probe -- calls the public "
            "agent.e1.cue_slot_tagger directly (the _last_cue_slot_weights cache is "
            "detached) and adds -lambda*|mean_w_safe - mean_w_dang|_1 to the P1 E1 loss on "
            "a fixed detached safe/dangerous batch. 5 arms x 3 seeds: A0_OFF (legacy q.k, "
            "C2 control), A1_tagger (matched-budget baseline = the 898 ON arm), and three "
            "A2_ctxdiv arms sweeping lambda in {0.1,0.5,2.0} (a Step-2.5a probe showed the "
            "objective saturates by lambda=0.5, so this spans the pre-saturation region "
            "rather than the scope's illustrative {0.5,1,2}). claim_ids=[] (weights no "
            "claim; /failure-autopsy adjudication required before any claims.yaml action, "
            "matching the 418g/418m/898 diagnostic precedent). C1 (entropy<2.5) is the FREE, "
            "load-bearing SCIENTIFIC readout (does the drive induce sparse selection?); C1b "
            "(context-divergence>0.1) is the DIRECTLY-OPTIMISED manipulation check measured "
            "held-out (did the drive take, generalising?). Full PASS = any A2 lambda arm has "
            "C1 AND C1b on a majority of READY seeds AND C2 CONTROL (OFF>2.65, on saddle). A "
            "FAIL is sub-typed into four discriminating regions: context_conditioned_not_"
            "sparse (C1b pass, C1 fail -- the probe-predicted outcome; implicates H3 / an "
            "entropy floor), sparse_not_context_conditioned (constant-peaky), "
            "objective_insufficient (true null), substrate_inconsistent (C2 fail). A seed "
            "not clearing readiness is excluded; fewer than a seed-majority ready self-routes "
            "substrate_not_ready_requeue. A FAIL does NOT rule out H2/H3; a non-insufficient "
            "FAIL rules OUT 'the tagger merely lacked the right gradient.'"
        ),
        "interpretation": {
            "label": label,
            "preconditions": flat_preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "acceptance_checks": acceptance,
        "n_seeds_total": n_seeds_total,
        "n_seeds_ready": n_ready,
        "ready_seeds": ready_seeds,
        "uniform_reference": UNIFORM_REFERENCE,
        "per_arm_summaries": summaries,
        # NAMED "arm_results" deliberately -- the multi-arm substrate_hash hoist keys on
        # this exact field name (arm_results[i].arm_fingerprint.substrate_hash).
        "arm_results": all_cells,
        "thresholds": {
            "sel_entropy_c1_threshold":        SEL_ENTROPY_C1_THRESHOLD,
            "sel_context_div_threshold":       SEL_CONTEXT_DIV_THRESHOLD,
            "sel_entropy_c2_floor":            SEL_ENTROPY_C2_FLOOR,
            "world_encoder_weight_move_floor": WORLD_ENCODER_WEIGHT_MOVE_FLOOR,
            "z_world_spread_lift_floor":       Z_WORLD_SPREAD_LIFT_FLOOR,
            "uniform_reference":               UNIFORM_REFERENCE,
            "lambdas":                         [lam for _, _, lam in ARMS if lam],
        },
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
        "params": {
            "seeds":                SEEDS,
            "world_dim":            WORLD_DIM,
            "p0a_episodes":         p0a,
            "p1_episodes":          p1,
            "steps_per_episode":    STEPS_PER_EPISODE,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "lambda_terrain":       LAMBDA_TERRAIN,
            "lambda_cue_action":    LAMBDA_CUE_ACTION,
            "ctxdiv_lambdas":       [lam for _, _, lam in ARMS if lam],
            "eval_batch_size":      eval_n,
            "readiness_batch_size": readiness_n,
            "div_batch_size":       div_n,
            "lr":                   LR,
            "arms": [{"label": lab, "cue_slot_tagger": tg, "ctxdiv_lambda": lam}
                     for lab, tg, lam in ARMS],
            "sd016_enabled":               True,
            "sd016_writepath_mode":        "off",
            "p0a_recipe":                  "sd070",
            "sws_enabled":                 False,
            "rem_enabled":                 False,
            "shy_enabled":                 False,
            "dry_run":                     dry_run,
        },
    }

    out_path = None
    if not dry_run:
        out_path = write_flat_manifest(
            output,
            dry_run=False,
            config=output.get("params"),
            seeds=SEEDS,
            script_path=Path(__file__),
            started_at=t0,
            z_goal_stream_stats=zg_acc.stats(),
        )
        print(f"Results written to {out_path}", flush=True)
    else:
        print(f"[DRY RUN] run_id={run_id} outcome={outcome}", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    output["_manifest_path"] = out_path
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)

    _manifest_path = result.get("_manifest_path")
    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_manifest_path,
        dry_run=bool(args.dry_run),
    )
