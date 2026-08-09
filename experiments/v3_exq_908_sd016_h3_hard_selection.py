#!/opt/local/bin/python3
"""
V3-EXQ-908 -- SD-016 H3 leg: hard/competitive selection operator discrimination

EXPERIMENT_PURPOSE: diagnostic

QUESTION:
  GOV-FANOUT-1 / V3-EXQ-898 failure autopsy (2026-08-08): the soft, end-to-end
  differentiable Path 3 tagger (V3-EXQ-898's A1_ON arm) reproduced the EXACT
  uniform ln(16)=2.7726 selection-entropy attractor under C2's control even
  when trained on a SD-070-fixed, genuinely-varying z_world encoder --
  ruling out "z_world doesn't vary" (898's own motivating diagnosis) as the
  cause and raising hypothesis H3: a soft, end-to-end differentiable softmax
  gate may not be able to HOLD a sparse, context-selective optimum even when
  correctly rewarded by terrain_loss -- gradient descent relaxes it back
  toward the uniform attractor regardless of what the loss demands.

  This is the ALGORITHM axis of the SD-016 selection-mechanism discrimination
  portfolio (H1=drive, H2=representation [deferred, gated on unbuilt deps +
  an absent lit base -- see the scope doc], H3=algorithm). H3 tests whether a
  STRUCTURALLY competitive selector -- independent of what terrain_loss
  demands -- breaks the saddle where the soft gate could not.

  Full design of record: REE_assembly/evidence/planning/
  sd016_selection_fanout_portfolio_scope_staged_20260809.md, section 3
  "Leg H3". Substrate this leg depends on: SD-016 H3 (ree-v3 CLAUDE.md,
  "SD-016 H3: Hard/competitive selection operator", 2026-08-09; the
  sd016_cue_slot_tagger_selection: "soft"|"gumbel"|"topk" knob on
  E1DeepPredictor.extract_cue_context, landed ree-v3 commit eeb999a).

DESIGN:
  Five arms x three seeds [42, 43, 44] (same seed pair convention as
  V3-EXQ-898/418-family; 45 not added, matching 898's own note).

    A0_OFF          sd016_cue_slot_tagger=False  (legacy q.k attention; C2 control)
    A1_tagger_soft  selection="soft"    (Path 3 baseline == V3-EXQ-898's A1_ON,
                                          matched-budget: same P0a/P1 recipe)
    A2_tagger_gumbel selection="gumbel" (straight-through annealed Gumbel-softmax)
    A3a_topk_k1     selection="topk", topk_k=1  (straight-through top-1)
    A3b_topk_k2     selection="topk", topk_k=2  (straight-through top-2 sub-sweep)

  Per (arm, seed) CELL: IDENTICAL scaffold to V3-EXQ-898 -- SD-070 D128 encoder
  warmup (P0a, RNG-neutral), the mandatory readiness gate (world-encoder
  weights moved + z_world spread lift on a matched positive control, measured
  per cell), then phased P1 E1-only training on DETACHED z_world (terrain_loss
  + cue_action_loss, unchanged LAMBDA_TERRAIN/LAMBDA_CUE_ACTION). Only the
  E1Config.sd016_cue_slot_tagger_selection (+ sd016_cue_slot_tagger_topk_k for
  the topk arms) knob differs across arms -- everything else (env, P0a/P1
  episode budgets, optimizer, LR, eval batches) is held fixed so a difference
  in outcome attributes to the SELECTION OPERATOR, not a confounded budget or
  encoder difference. A0_OFF's readiness gate is measured identically (its
  own agent also runs P0a since cue_slot_tagger's encoder path is independent
  of the E1 tagger) purely for the seed-level joint-readiness comparison
  V3-EXQ-898 established (see READINESS below).

  READINESS (identical mechanism to V3-EXQ-898, extended to 5 arms): a seed
  is READY only if ALL 5 arms' per-cell readiness preconditions
  (world_encoder_weights_moved, z_world_spread_lift) are met. This is NOT the
  V3-EXQ-785 regime-conditioning anti-pattern (a precondition that is
  structurally impossible for one regime silently vacating another regime's
  valid result) -- P0a trains the SHARED world encoder (split_encoder.
  world_encoder + world_precision_logit) via a recipe that does not read the
  E1 tagger's selection-mode flag at all, so per-arm readiness failure is
  expected to correlate strongly across arms for the same seed (either the
  SD-070 recipe took for that seed's RNG draw or it didn't) rather than
  differ systematically BY DESIGN across arms, which is what would make the
  joint-AND gate unsafe. A seed failing readiness self-routes
  substrate_not_ready_requeue for that seed across ALL arms (never a
  per-arm/per-mechanism verdict on an untrained encoder).

PER-ARM METRICS (extends V3-EXQ-898 -- adds full per-context selection
  histograms per the scope doc's aliasing guard, section 5(ii), so a null can
  be inspected for hidden structure rather than assumed absent from the
  scalar alone):
  sel_entropy_mean          mean slot-selection entropy over eval batch;
                            uniform reference ln(16) ~= 2.7726.
  sel_context_divergence    L1 distance between mean safe-vs-dangerous
                            selection distributions (scalar summary).
  mean_w_safe / mean_w_dang FULL per-slot [16] mean selection-weight vectors
                            under the safe / dangerous eval batches --
                            recorded so a "sel_context_divergence" near zero
                            can be inspected for hidden structure the L1
                            scalar could mask (aliasing guard).
  action_bias_per_channel_std / action_bias_div  reported, not gated
                            (unchanged from 898; full propagation depends on
                            cue_action_proj / SD-055, out of scope here).

ACCEPTANCE CRITERIA (per scope doc section 3 H3 -- computed over READY seeds
  only, majority = (n_ready//2)+1):
  C2  (CONTROL, load-bearing)     A0_OFF sel_entropy_mean > 2.65 on a
                                  majority of ready seeds (reproduces the
                                  saddle; isolates the ablation).
  Per ON arm (A1_tagger_soft, A2_tagger_gumbel, A3a_topk_k1, A3b_topk_k2):
    C1  (load-bearing)   sel_entropy_mean < 2.5 on a majority of ready seeds.
    C1b (load-bearing)   sel_context_divergence > 0.1 on a majority of
                         ready seeds.

  Per-arm verdict:
    PASS               C1 AND C1b.
    CONSTANT_PEAKY      C1 PASS, C1b FAIL -- the "constant-peaky" degeneracy
                         C1b exists to catch: the arm sparsifies (as its
                         operator structurally must) but selects the SAME
                         slot regardless of context, i.e. sparsifies WITHOUT
                         context-conditioning. This is a distinct, genuinely
                         informative outcome, not a plain FAIL -- it says the
                         operator's hard selection is real but blind to
                         context, which the scope doc names explicitly.
    FAIL                C1 FAIL (any C1b outcome) -- hard selection did not
                         even sparsify below the entropy floor for this arm.

  Top-level outcome: PASS iff C2 passes AND at least one ON arm achieves a
  PASS verdict (i.e. the portfolio finds ANY selection mechanism -- soft or
  hard -- that breaks the saddle discriminatively). FAIL iff C2 passes but
  EVERY ON arm is FAIL or CONSTANT_PEAKY (the null: no mechanism in this
  leg's coverage context-conditions). C2 failing routes to a substrate-
  inconsistency label, matching 898's own routing, since neither C1/C1b nor
  the constant-peaky read is interpretable if the control itself is off the
  saddle for the wrong reason.

  Null declaration (per scope doc section 3 H3, verbatim): if the ON arms
  fail C1/C1b with readiness met, hard selection alone does not break the
  saddle -- implicating the drive (H1, tested in parallel by V3-EXQ-907) or
  the representation (H2, deferred), not the softness of the gate.

DV-SYMMETRY CHECK (per /queue-experiment Step 3 -- one sentence per arm;
  MANDATORY declaration for a portfolio where TWO of five arms manipulate the
  operator, not just the tagger's learned content):
  A0_OFF          reference/control; carries no manipulation under test.
  A1_tagger_soft  identical DV-symmetry statement to V3-EXQ-898's A1_ON: the
                  tagger flag changes WHICH function maps z_world -> slot
                  logits (independent feedforward MLP vs q.k dot-product),
                  not a uniform-broadcast/monotone/permutation transform of
                  the OFF arm's logits -- sel_entropy_mean and
                  sel_context_divergence are free to differ, not structurally
                  pinned.
  A2_tagger_gumbel / A3a/A3b_topk  THE CRITICAL CAVEAT: swapping the soft
                  softmax for a straight-through hard estimator is NOT a
                  broadcast/monotone/permutation transform of the tagger's
                  logits in the classic sense, BUT it DOES structurally
                  compress the FORWARD-VALUE entropy toward the operator's
                  own structural floor (topk k=1 -> exactly 0 every context;
                  topk k=2 -> at most ln(2)=0.693; annealed gumbel -> near 0
                  once temperature anneals down) REGARDLESS of whether the
                  tagger has learned anything context-discriminative. So
                  **C1 (entropy) for these three arms is expected to pass
                  near-trivially BY CONSTRUCTION of the selection operator,
                  independent of genuine context-conditioning** -- it is a
                  necessary sanity check, not evidence the arm discriminates.
                  C1b (context divergence: does a DIFFERENT slot get picked
                  for a different z_world input) is NOT fixed by the operator
                  -- an untrained or poorly-trained tagger can pick the SAME
                  single slot every time regardless of input, which the
                  straight-through estimator neither forces nor forbids.
                  C1b is therefore the true discriminator for the three hard
                  arms; this is exactly why the scope doc's "constant-peaky"
                  degeneracy (C1-pass/C1b-fail) is called out as a distinct,
                  expected-to-occur, and genuinely informative outcome rather
                  than folded into a generic FAIL. Recorded structurally
                  below via criteria_non_degenerate (C1 marked non-
                  discriminating for the three hard arms; C1b marked
                  discriminating for all four ON arms).

claim_ids: []  (diagnostic -- substrate-readiness/mechanism-discrimination,
  weights no SD-016 claim-confidence directly, matching the V3-EXQ-898/
  418g/418m precedent for this acceptance-test family; any PASS/FAIL here
  still requires /failure-autopsy adjudication before any claims.yaml action
  per the Diagnostic adjudication gate).
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id: ends _v3
"""

import sys
import argparse
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


EXPERIMENT_TYPE = "v3_exq_908_sd016_h3_hard_selection"
CLAIM_IDS: List[str] = []          # diagnostic: mechanism discrimination, weights no claim
EXPERIMENT_PURPOSE = "diagnostic"

# Same anchor-reachability exemption rationale as V3-EXQ-898 -- this script's own
# readiness metrics have no frozen reference_cells literal from a prior run of THIS
# script. Thresholds are grounded identically: world_encoder_weights_moved>=1 is the
# V3-EXQ-737a/728 wiring-failure signature; z_world_spread_lift>=1.3 sits inside the
# V3-EXQ-783-measured ~1.7-1.9x lift for the identical SD-070 recipe this script
# inherits verbatim from V3-EXQ-898.
ANCHOR_REACHABILITY_EXEMPT = (
    "no frozen reference_cells literal exists for this script's own metrics pre-run; "
    "thresholds are grounded in V3-EXQ-783's measured SD-070 lift (~1.7-1.9x) and the "
    "V3-EXQ-737a/728 wiring-failure signature (0 tensors moved), inherited verbatim "
    "from V3-EXQ-898 -- see module docstring"
)

WORLD_DIM            = 128         # the V3-EXQ-783/898 D128_TRAINED axis
P0A_EPISODES         = 60          # SD-070's validated operating point (unchanged from 898)
P1_EPISODES          = 40          # unchanged from V3-EXQ-898/418-family
STEPS_PER_EPISODE    = 150
CONTEXT_SWITCH_EVERY = 5
LAMBDA_TERRAIN       = 0.1
LAMBDA_CUE_ACTION    = 0.5
EVAL_BATCH_SIZE      = 32
READINESS_BATCH_SIZE = 16
LR                   = 1e-4
SEEDS                = [42, 43, 44]

# (arm_label, cue_slot_tagger, selection, topk_k)
ARMS: List[Tuple[str, bool, str, int]] = [
    ("A0_OFF",          False, "soft",   1),
    ("A1_tagger_soft",  True,  "soft",   1),
    ("A2_tagger_gumbel", True, "gumbel", 1),
    ("A3a_topk_k1",     True,  "topk",   1),
    ("A3b_topk_k2",     True,  "topk",   2),
]
ON_ARM_LABELS = [lab for lab, tagger, _, _ in ARMS if tagger]

SEL_ENTROPY_C1_THRESHOLD  = 2.5
SEL_CONTEXT_DIV_THRESHOLD = 0.1
SEL_ENTROPY_C2_FLOOR      = 2.65
UNIFORM_REFERENCE         = None   # filled at runtime = ln(num_slots)

WORLD_ENCODER_WEIGHT_MOVE_FLOOR = 1     # >=1 tensor changed
Z_WORLD_SPREAD_LIFT_FLOOR       = 1.3   # trained/untrained spread ratio (783 measured ~1.7-1.9x)


# ---------------------------------------------------------------------------
# Env + agent helpers (env identical to V3-EXQ-898/418m; agent adds the H3
# selection-mode knob).
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


def _make_agent(env: CausalGridWorldV2, cue_slot_tagger: bool,
                 selection: str, topk_k: int) -> REEAgent:
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
        sd016_cue_slot_tagger_selection=selection,
        sd016_cue_slot_tagger_topk_k=topk_k,
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
# Selection / output probes (extends V3-EXQ-898 with full per-slot histograms).
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


def _selection_context_histograms(agent, z_safe, z_dang) -> Dict[str, Any]:
    """Full per-slot selection histograms + the scalar L1 divergence -- the
    scope doc's aliasing guard (section 5(ii)): a near-zero scalar can hide
    structure that only the full [16]-vectors reveal."""
    with torch.no_grad():
        agent.e1.extract_cue_context(z_safe)
        w_safe = agent.e1._last_cue_slot_weights.mean(dim=0)
        agent.e1.extract_cue_context(z_dang)
        w_dang = agent.e1._last_cue_slot_weights.mean(dim=0)
        divergence = float((w_safe - w_dang).abs().sum().item())
        return {
            "sel_context_divergence": divergence,
            "mean_w_safe": [float(x) for x in w_safe.tolist()],
            "mean_w_dang": [float(x) for x in w_dang.tolist()],
        }


def _z_world_spread(agent, env, n_samples: int) -> float:
    """Mean per-dimension std of z_world across a diverse observation batch --
    identical to V3-EXQ-898's readiness-precondition statistic."""
    batch = _collect_eval_batch(agent, env, n_samples)
    return float(batch.std(dim=0).mean().item())


# ---------------------------------------------------------------------------
# Eval batch collector (identical to V3-EXQ-898).
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
# P1 training episode -- E1 ONLY, on DETACHED z_world (identical to V3-EXQ-898).
# ---------------------------------------------------------------------------

def _run_training_episode(agent, env, optimizer, phase: str) -> int:
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

def _run_one_cell(arm_label: str, tagger: bool, selection: str, topk_k: int, seed: int,
                   p0a: int, p1: int, eval_n: int, readiness_n: int,
                   dry_run: bool) -> Dict[str, Any]:
    config_slice = {
        "lineage": "v3_exq_908_sd016_h3",
        "arm": arm_label,
        "cue_slot_tagger": tagger,
        "selection": selection,
        "topk_k": topk_k,
        "world_dim": WORLD_DIM,
        "p0a_recipe": "sd070",
        "p0a_episodes": p0a,
        "p1_episodes": p1,
        "steps_per_episode": STEPS_PER_EPISODE,
    }

    with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
        env_safe = _make_env_safe(seed)
        env_dang = _make_env_dangerous(seed)
        warmup_env = _make_warmup_env(seed)
        agent = _make_agent(env_safe, tagger, selection, topk_k)

        print(f"Seed {seed} Condition {arm_label}", flush=True)
        print(f"  [arm {arm_label} seed {seed}] cue_slot_tagger={tagger} "
              f"selection={selection} topk_k={topk_k} "
              f"world_dim={WORLD_DIM} p0a={p0a} p1={p1}", flush=True)

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
            "selection": selection,
            "topk_k": topk_k,
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
            cell.stamp(result)
            return result

        # -- P1: E1 selection training on DETACHED z_world -----------------------------
        optimizer = optim.Adam(agent.e1.parameters(), lr=LR)
        half = p1 // 2
        for ep in range(p1):
            phase = "P1_early" if ep < half else "P1_late"
            env = env_dang if (ep // CONTEXT_SWITCH_EVERY) % 2 == 1 else env_safe
            _run_training_episode(agent, env, optimizer, phase)
            if (ep + 1) % 50 == 0 or (ep + 1) == p1:
                print(f"  [train] {arm_label} seed={seed} ep {ep+1}/{p1}", flush=True)

        z_safe = _collect_eval_batch(agent, env_safe, eval_n)
        z_dang = _collect_eval_batch(agent, env_dang, eval_n)

        sel_ent = _selection_entropy_mean(agent, z_safe)
        hist = _selection_context_histograms(agent, z_safe, z_dang)
        ab_std = _action_bias_per_channel_std(agent, z_safe)
        ab_div = _action_bias_divergence(agent, z_safe, z_dang)

        print(f"  [arm {arm_label} seed {seed}] "
              f"sel_entropy={sel_ent:.4f} sel_ctx_div={hist['sel_context_divergence']:.4f} "
              f"action_bias_std={ab_std:.6f} action_bias_div={ab_div:.4f}", flush=True)
        print("verdict: PASS", flush=True)

        result.update({
            "sel_entropy_mean": sel_ent,
            "sel_context_divergence": hist["sel_context_divergence"],
            "mean_w_safe": hist["mean_w_safe"],
            "mean_w_dang": hist["mean_w_dang"],
            "action_bias_per_channel_std": ab_std,
            "action_bias_div": ab_div,
        })
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
        "action_bias_per_channel_std_mean": sum(r["action_bias_per_channel_std"] for r in ready_results) / n,
        "action_bias_div_mean":             sum(r["action_bias_div"] for r in ready_results) / n,
    }


def _evaluate(per_arm_ready: Dict[str, List[Dict]], n_ready_seeds: int) -> Dict:
    """Per-ON-arm C1/C1b + the shared C2 control, plus a per-arm verdict that
    distinguishes PASS / CONSTANT_PEAKY (the scope doc's named degeneracy) /
    FAIL. Overall PASS iff C2 passes AND at least one ON arm PASSes."""
    majority = (n_ready_seeds // 2) + 1 if n_ready_seeds > 0 else 0

    off = per_arm_ready.get("A0_OFF", [])
    c2_seeds_pass = sum(1 for r in off if r["sel_entropy_mean"] > SEL_ENTROPY_C2_FLOOR)
    c2_pass = n_ready_seeds > 0 and c2_seeds_pass >= majority

    per_arm: Dict[str, Any] = {}
    any_on_pass = False
    for lab in ON_ARM_LABELS:
        rows = per_arm_ready.get(lab, [])
        c1_seeds_pass = sum(1 for r in rows if r["sel_entropy_mean"] < SEL_ENTROPY_C1_THRESHOLD)
        c1_pass = n_ready_seeds > 0 and c1_seeds_pass >= majority
        c1b_seeds_pass = sum(1 for r in rows if r["sel_context_divergence"] > SEL_CONTEXT_DIV_THRESHOLD)
        c1b_pass = n_ready_seeds > 0 and c1b_seeds_pass >= majority

        if c1_pass and c1b_pass:
            verdict = "pass"
            any_on_pass = True
        elif c1_pass and not c1b_pass:
            verdict = "constant_peaky_degenerate"
        else:
            verdict = "fail"

        per_arm[lab] = {
            "C1_breaks_saddle": {
                "pass": c1_pass, "seeds_pass": c1_seeds_pass, "majority": majority,
                "threshold": SEL_ENTROPY_C1_THRESHOLD, "load_bearing": True,
            },
            "C1b_context_dependent": {
                "pass": c1b_pass, "seeds_pass": c1b_seeds_pass, "majority": majority,
                "threshold": SEL_CONTEXT_DIV_THRESHOLD, "load_bearing": True,
            },
            "verdict": verdict,
        }

    return {
        "combination_rule": (
            "overall_pass = C2_off_arm_on_saddle.pass AND any(per_arm[lab].verdict == "
            "'pass' for lab in ON_ARM_LABELS) -- NOT a plain AND across all arms. A "
            "portfolio finding ANY hard-or-soft mechanism that breaks the saddle is a "
            "PASS; per-arm detail (including constant_peaky_degenerate) is preserved "
            "in per_arm regardless of the top-level outcome."
        ),
        "C2_off_arm_on_saddle": {
            "pass": c2_pass, "seeds_pass": c2_seeds_pass, "majority": majority,
            "floor": SEL_ENTROPY_C2_FLOOR, "load_bearing": True,
        },
        "per_arm": per_arm,
        "any_on_arm_pass": any_on_pass,
        "overall_pass": c2_pass and any_on_pass,
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> Dict:
    import math
    global UNIFORM_REFERENCE
    UNIFORM_REFERENCE = math.log(16)
    t0 = time.perf_counter()

    p0a        = P0A_EPISODES        if not dry_run else 2
    p1         = P1_EPISODES         if not dry_run else 3
    eval_n     = EVAL_BATCH_SIZE     if not dry_run else 4
    readiness_n = READINESS_BATCH_SIZE if not dry_run else 4

    print(f"V3-EXQ-908 SD-016 H3 hard-selection discrimination "
          f"(seeds={SEEDS} arms={[a[0] for a in ARMS]} dry_run={dry_run} "
          f"uniform_ref={UNIFORM_REFERENCE:.4f} world_dim={WORLD_DIM})", flush=True)

    all_cells: List[Dict] = []
    per_arm_all: Dict[str, List[Dict]] = {lab: [] for lab, _, _, _ in ARMS}

    for seed in SEEDS:
        for arm_label, tagger, selection, topk_k in ARMS:
            r = _run_one_cell(arm_label, tagger, selection, topk_k, seed,
                               p0a, p1, eval_n, readiness_n, dry_run)
            all_cells.append(r)
            per_arm_all[arm_label].append(r)

    n_seeds_total = len(SEEDS)
    ready_seeds = [
        s for s in SEEDS
        if all(
            next(r for r in per_arm_all[lab] if r["seed"] == s)["ready"]
            for lab, _, _, _ in ARMS
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
            for lab, _, _, _ in ARMS
        }
        acceptance = _evaluate(per_arm_ready, n_ready)
        outcome = "PASS" if acceptance["overall_pass"] else "FAIL"

        if not acceptance["C2_off_arm_on_saddle"]["pass"]:
            label = "sd016_h3_control_off_saddle_inconsistent"
        elif acceptance["any_on_arm_pass"]:
            passing = [lab for lab, v in acceptance["per_arm"].items() if v["verdict"] == "pass"]
            label = "sd016_h3_hard_selection_breaks_saddle:" + ",".join(passing)
        else:
            degenerate = [lab for lab, v in acceptance["per_arm"].items()
                          if v["verdict"] == "constant_peaky_degenerate"]
            if degenerate:
                label = "sd016_h3_constant_peaky_degenerate:" + ",".join(degenerate)
            else:
                label = "sd016_h3_no_mechanism_breaks_saddle"

        # C1 for the three hard arms (gumbel/topk) is EXPECTED to pass near-trivially
        # by construction of the straight-through estimator (structural entropy floor),
        # independent of genuine context-conditioning -- see the DV-SYMMETRY CHECK in
        # the module docstring. So it is marked non-discriminating for those arms.
        # C1b is the true discriminator for all four ON arms and for the soft arm
        # (A1_tagger_soft) BOTH C1 and C1b are meaningful, matching V3-EXQ-898.
        criteria_non_degenerate = {"C2_off_arm_on_saddle": True}
        for lab in ON_ARM_LABELS:
            hard = lab in ("A2_tagger_gumbel", "A3a_topk_k1", "A3b_topk_k2")
            criteria_non_degenerate[f"{lab}::C1_breaks_saddle"] = not hard
            criteria_non_degenerate[f"{lab}::C1b_context_dependent"] = True

        print(f"  [summary] ready={n_ready}/{n_seeds_total} C2={acceptance['C2_off_arm_on_saddle']['pass']} "
              f"any_on_pass={acceptance['any_on_arm_pass']} -> outcome={outcome} label={label}",
              flush=True)
        for lab, v in acceptance["per_arm"].items():
            print(f"    [arm-verdict] {lab}: C1={v['C1_breaks_saddle']['pass']} "
                  f"C1b={v['C1b_context_dependent']['pass']} verdict={v['verdict']}", flush=True)

    per_arm_ready_for_summary = {
        lab: [r for r in per_arm_all[lab] if r["ready"]]
        for lab, _, _, _ in ARMS
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
            "SD-016 H3 leg (algorithm axis) of the GOV-FANOUT-1 selection-mechanism "
            "discrimination portfolio, following the V3-EXQ-898 failure autopsy "
            "(2026-08-08): the soft Path 3 tagger reproduced the uniform ln(16) "
            "attractor even on a SD-070-fixed, genuinely-varying z_world encoder. "
            "claim_ids=[] (weights no claim; diagnostic adjudication via "
            "/failure-autopsy required before any claims.yaml action). 5-arm ablation: "
            "A0_OFF (legacy q.k, C2 control) vs A1_tagger_soft (matched-budget == "
            "V3-EXQ-898 A1_ON) vs A2_tagger_gumbel (annealed straight-through "
            "Gumbel-softmax) vs A3a/A3b_topk (straight-through top-1/top-2), all on "
            "the IDENTICAL SD-070 encoder recipe and P0a/P1 budgets -- only the "
            "selection OPERATOR differs. C2 CONTROL (load-bearing): OFF stays > 2.65. "
            "Per ON arm, C1 (load-bearing): entropy < 2.5; C1b (load-bearing, the true "
            "discriminator for the hard arms per the module docstring's DV-SYMMETRY "
            "note): context-divergence > 0.1. A C1-pass/C1b-fail arm is labelled "
            "constant_peaky_degenerate, not a plain FAIL -- it means the operator "
            "sparsifies (as it structurally must) but is blind to context. Overall "
            "PASS iff C2 passes AND >=1 ON arm achieves C1 AND C1b. A seed not "
            "clearing its readiness preconditions (identical mechanism to V3-EXQ-898, "
            "extended to all 5 arms) is excluded from the acceptance count; fewer "
            "than a seed-majority ready self-routes substrate_not_ready_requeue."
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
        # NAMED "arm_results" (not "per_cell_results") deliberately -- manifest_core.
        # stamp_recording_core's multi-arm hoist path (top-level substrate_hash <-
        # arm_results[i].arm_fingerprint.substrate_hash) keys on this exact field name.
        "arm_results": all_cells,
        "thresholds": {
            "sel_entropy_c1_threshold":        SEL_ENTROPY_C1_THRESHOLD,
            "sel_context_div_threshold":       SEL_CONTEXT_DIV_THRESHOLD,
            "sel_entropy_c2_floor":            SEL_ENTROPY_C2_FLOOR,
            "world_encoder_weight_move_floor": WORLD_ENCODER_WEIGHT_MOVE_FLOOR,
            "z_world_spread_lift_floor":       Z_WORLD_SPREAD_LIFT_FLOOR,
            "uniform_reference":               UNIFORM_REFERENCE,
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
            "eval_batch_size":      eval_n,
            "readiness_batch_size": readiness_n,
            "lr":                   LR,
            "arms": [
                {"label": lab, "cue_slot_tagger": tg, "selection": sel, "topk_k": k}
                for lab, tg, sel, k in ARMS
            ],
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
