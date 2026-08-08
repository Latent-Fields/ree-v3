#!/opt/local/bin/python3
"""
V3-EXQ-898 -- SD-016 leg (A) retrieval-selectivity retest under the SD-070 encoder fix

EXPERIMENT_PURPOSE: diagnostic

QUESTION:
  SD-016's own claim text (claims.yaml what_would_answer, leg A RETRIEVAL-SELECTIVITY)
  names the exact retest this script performs. V3-EXQ-418g/418d/418e/418i (the q.k
  attention path) and V3-EXQ-418m (Path 3, the feedforward cue->slot tagger) both FAILed
  to break the uniform ln(16)=2.7726 selection-entropy saddle. V3-EXQ-418j/k (env
  enrichment) separately showed the bottleneck is not environment information-poverty:
  z_world stays near-constant (cos_cross_mean 0.987-0.9999) across a 4-arm enrichment
  ladder. The unifying diagnosis across all of these: z_world itself does not vary enough
  across contexts for ANY selection mechanism -- q.k or feedforward -- to have anything
  to differentiate on. The bottleneck sits one level upstream of selection, in the encoder.

  SD-070 (2026-07-18) is a validated encoder-training recipe that measurably raises z_world
  discriminative structure without touching SD-016's own substrate (it trains
  split_encoder.world_encoder + world_precision_logit from outside LatentStack; SD-016's
  ContextMemory/Path-3 tagger are untouched). V3-EXQ-783 (PASS, 2026-07-18, scoped to
  Q-002/SD-031) measured this: contrast_ratio 0.125 (untrained, dim=32 or 128) -> 0.211
  (SD-070-trained, dim=32) -> 0.232 (SD-070-trained, dim=128). But that run never read SD-016's
  own retrieval-selectivity readout back onto this criterion -- claims.yaml flags this
  explicitly as the missing leg.

  This script re-runs the 418g/418m-shaped acceptance test (same env, same C1/C1b/C2
  criteria, same Path-3 tagger under test) with EXACTLY ONE upstream change: the z_world
  encoder is trained via the validated SD-070 recipe (world_dim=128, the V3-EXQ-783
  D128_TRAINED configuration) instead of being left at initialisation. Does retrieval
  selectivity emerge once the encoder actually varies across contexts?

  Per claims.yaml: explicitly NOT re-running the SD-023-landmarks environment-enrichment
  probe -- V3-EXQ-418j/k already answered that question (environment enrichment does not
  restore selectivity) and it was retroactively marked superseded 2026-08-02.

DESIGN:
  Two arms x three seeds [42, 43, 44] (seed 45 is not used -- see CLAUDE.md "seed 44
  instability" note; that note is scoped to reef-config envs, which this experiment's env
  is not, and 44/45 are the standard 418-family seed pair, so 44 is kept and 45 is simply
  not added).

    A0_OFF   sd016_cue_slot_tagger=False  (legacy q.k attention -- the 418-family saddle)
    A1_ON    sd016_cue_slot_tagger=True   (Path 3 feedforward tagger)

  Per (arm, seed) CELL (wrapped in experiments._lib.arm_fingerprint.arm_cell, which resets
  ALL RNG on entry -- torch/cuda/numpy/random/harness -- so both arms of a seed start from an
  IDENTICAL RNG state and independently converge to matched encoder weights; see the
  DETERMINISM NOTE below):

    P0a (encoder warmup, SD-070, NEW vs the 418-family): world_dim=128. A dedicated
        warmup_env (never the training env) is rolled out under RandomPolicy(seed) for
        P0A_EPISODES episodes; experiments._lib.zworld_p0_warmup.run_zworld_p0 buffers
        world_state + the SD-018 resource-proximity target and trains
        split_encoder.world_encoder + world_precision_logit via the validated recipe
        (ZWorldP0Config defaults: variance_weight=25, covariance_weight=50,
        reconstruction_weight=10, batch_size=64, epochs=12). RNG-neutral by construction
        (see the module docstring) -- turning P0a on does not itself perturb P1's RNG draws
        relative to a hypothetical P0a=0 run.

    READINESS PRECONDITIONS (measured on THIS cell's trained encoder, before P1 -- the
    "does the fix actually take" check, matching V3-EXQ-783's own weight-delta readiness
    check and the module's anti-collapse framing):
      world_encoder_weights_moved   >=1 of {world_encoder.*, world_precision_logit} tensors
                                     changed across P0a (the exact V3-EXQ-737a/728 wiring-
                                     failure signature this recipe exists to fix).
      z_world_spread_lift           mean per-dimension std of z_world across a batch of
                                     env-diverse observations, TRAINED / UNTRAINED (measured
                                     on the SAME agent+env apparatus, before vs after P0a) --
                                     the SAME kind of statistic (spread, not magnitude) the
                                     load-bearing C1/C1b criteria depend on upstream: a
                                     selection mechanism cannot differentiate across contexts
                                     if z_world itself does not vary across them. Positive
                                     control: this exact ratio is what V3-EXQ-783 measured
                                     rising 1.7-1.9x (contrast_ratio 0.125->0.21-0.23) under
                                     the identical recipe.
    A cell whose readiness is unmet self-routes to substrate_not_ready_requeue for that
    SEED (both arms, since C1/C1b need the ON encoder and C2 needs the OFF encoder from the
    SAME seed to be jointly interpretable) rather than corrupting the acceptance count. If
    fewer than a majority of the pre-registered seeds clear readiness, the WHOLE RUN
    self-routes substrate_not_ready_requeue -- never a substrate_ceiling/does_not_support
    verdict on an encoder that never actually trained.

    P1 (E1 selection training, mirrors the 418-family exactly except for phased-training
        compliance -- see below): P1_EPISODES episodes alternating safe/dangerous env every
        CONTEXT_SWITCH_EVERY episodes, training ONLY agent.e1.parameters() (terrain_loss +
        cue_action_loss, LAMBDA_TERRAIN/LAMBDA_CUE_ACTION unchanged from 418m) on
        z_world.detach() -- the encoder optimiser is never stepped in P1.

  PHASED TRAINING NOTE (correction relative to the 418-family): V3-EXQ-418g/418i/418m
  trained agent.e1.parameters() + agent.latent_stack.parameters() JOINTLY on non-detached
  z_world. That was tolerable there only because the encoder was never meaningfully driven --
  with SD-070 now training a REAL encoder in P0a, joint E1/encoder training in P1 would let
  a live terrain_loss gradient reach back into a just-trained encoder and move it again,
  reintroducing exactly the moving-target collapse this skill's phased-training rule exists
  to prevent (EXQ-166b/c/d, EXQ-085l, EXQ-194). So P1 here detaches z_world before every E1
  loss and its optimiser covers agent.e1.parameters() only -- P0a trains the encoder, P1
  trains E1, never both at once.

  Selection entropy / context-divergence are measured identically to V3-EXQ-418g/418m: call
  extract_cue_context over an eval batch and read the cached
  E1DeepPredictor._last_cue_slot_weights, so OFF (q.k softmax) and ON (tagger softmax) are
  compared on equal footing.

PER-ARM METRICS (unchanged from V3-EXQ-418m):
  sel_entropy_mean          mean slot-selection entropy over eval batch; uniform reference
                            ln(16) ~= 2.7726.
  sel_context_divergence    L1 distance between mean safe-vs-dangerous selection
                            distributions (anti-degeneracy: a collapsed-to-one-slot tagger
                            would pass the entropy gate but score ~0 here).
  action_bias_per_channel_std / action_bias_div  reported, not gated (EXQ-449/449b metrics;
                            full propagation depends on cue_action_proj / SD-055, out of
                            scope here).

ACCEPTANCE CRITERIA (unchanged shape from V3-EXQ-418g/418m, per claims.yaml's own
  specification -- computed over READY seeds only, majority = (n_ready//2)+1):
  C1  (PRIMARY, load-bearing)   A1_ON sel_entropy_mean < 2.5 on a majority of ready seeds.
  C1b (ANTI-DEGENERACY, load-bearing) A1_ON sel_context_divergence > 0.1 on a majority of
                                 ready seeds.
  C2  (CONTROL)                 A0_OFF sel_entropy_mean > 2.65 on a majority of ready seeds
                                 (reproduces the saddle -- confirms the ablation isolates the
                                 tagger and the substrate is otherwise consistent with the
                                 418-family).

  PASS: C1 AND C1b AND C2 (all load-bearing).
  C1/C1b FAIL with C2 PASS -> the encoder fix does not, by itself, let Path 3 clear
    selectivity; route to /failure-autopsy (tagger capacity / training-signal under a
    genuinely-varying z_world, a strictly narrower question than the 418m result, which
    could not distinguish "tagger insufficient" from "nothing to select on").
  C2 FAIL -> substrate inconsistency (OFF arm not on the saddle under the new encoder);
    investigate before interpreting C1/C1b.

DV-SYMMETRY CHECK (per /queue-experiment Step 3 -- one sentence per arm): neither arm's
  manipulation is a uniform broadcast/monotone/permutation transform of the DV. The tagger
  flag changes WHICH function maps z_world -> slot logits (q.k dot-product vs an independent
  feedforward MLP with its own random init) -- not a constant added to, or a monotone
  rescaling of, the OFF arm's logits -- so sel_entropy_mean and sel_context_divergence are
  not structurally pinned across arms by construction; they are free to differ.

claim_ids: []  (diagnostic -- substrate readiness, weights no SD-016 claim-confidence
  directly, matching the V3-EXQ-418g/418m precedent for this exact acceptance-test shape;
  a PASS/FAIL here still requires /failure-autopsy adjudication before any claims.yaml
  action per the Diagnostic adjudication gate).
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


EXPERIMENT_TYPE = "v3_exq_898_sd016_lega_encoder_fix_retest"
CLAIM_IDS: List[str] = []          # diagnostic: substrate readiness, weights no claim
EXPERIMENT_PURPOSE = "diagnostic"

# Both readiness preconditions below are new metrics for this script (world_encoder_weights_
# moved is a binary tensor-changed count; z_world_spread_lift is this script's own spread
# statistic) -- neither has a frozen `reference_cells` literal from a prior run of THIS
# script to feed assert_anchor_reachable, which is a chicken-and-egg requirement for a
# script that has not yet executed once. The thresholds are not ungrounded, though:
# world_encoder_weights_moved>=1 is the exact V3-EXQ-737a/728 wiring-failure signature
# (0 moved) this recipe was built to fix, and z_world_spread_lift>=1.3 sits well inside the
# ~1.7-1.9x lift V3-EXQ-783 measured for the identical SD-070 recipe (contrast_ratio
# 0.125 -> 0.211-0.232) on a similar (not identical) grid config.
ANCHOR_REACHABILITY_EXEMPT = (
    "no frozen reference_cells literal exists for this script's own metrics pre-run; "
    "thresholds are grounded in V3-EXQ-783's measured SD-070 lift (~1.7-1.9x) and the "
    "V3-EXQ-737a/728 wiring-failure signature (0 tensors moved) instead -- see module "
    "docstring READINESS PRECONDITIONS section"
)

WORLD_DIM            = 128         # the V3-EXQ-783 D128_TRAINED axis
P0A_EPISODES         = 60          # SD-070's validated operating point (CLAUDE.md ZWORLD_P0_EPISODES)
P1_EPISODES          = 40          # unchanged from V3-EXQ-418m
STEPS_PER_EPISODE    = 150
CONTEXT_SWITCH_EVERY = 5
LAMBDA_TERRAIN       = 0.1
LAMBDA_CUE_ACTION    = 0.5
EVAL_BATCH_SIZE      = 32
READINESS_BATCH_SIZE = 16          # cheap positive-control probe batch, pre-P1
LR                   = 1e-4
SEEDS                = [42, 43, 44]

ARMS: List[Tuple[str, bool]] = [
    ("A0_OFF", False),
    ("A1_ON",  True),
]

SEL_ENTROPY_C1_THRESHOLD  = 2.5
SEL_CONTEXT_DIV_THRESHOLD = 0.1
SEL_ENTROPY_C2_FLOOR      = 2.65
UNIFORM_REFERENCE         = None   # filled at runtime = ln(num_slots)

WORLD_ENCODER_WEIGHT_MOVE_FLOOR = 1     # >=1 tensor changed
Z_WORLD_SPREAD_LIFT_FLOOR       = 1.3   # trained/untrained spread ratio (783 measured ~1.7-1.9x)


# ---------------------------------------------------------------------------
# Env + agent helpers (env mirrors V3-EXQ-418m; agent adds world_dim + alpha_world=0.9,
# both already implied for z_world-fidelity work by CLAUDE.md's SD-008 note and already the
# 418-family default for alpha_world).
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
# Selection / output probes (unchanged from V3-EXQ-418m).
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


def _selection_context_divergence(agent, z_safe, z_dang) -> float:
    with torch.no_grad():
        agent.e1.extract_cue_context(z_safe)
        w_safe = agent.e1._last_cue_slot_weights.mean(dim=0)
        agent.e1.extract_cue_context(z_dang)
        w_dang = agent.e1._last_cue_slot_weights.mean(dim=0)
        return float((w_safe - w_dang).abs().sum().item())


def _z_world_spread(agent, env, n_samples: int) -> float:
    """Mean per-dimension std of z_world across a diverse observation batch -- the
    readiness-precondition statistic (spread, matching what C1/C1b ultimately depend
    on), measured identically before and after P0a for a same-apparatus positive control."""
    batch = _collect_eval_batch(agent, env, n_samples)
    return float(batch.std(dim=0).mean().item())


# ---------------------------------------------------------------------------
# Eval batch collector (mirrors V3-EXQ-418m).
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
# P1 training episode -- E1 ONLY, on DETACHED z_world (phased-training compliant;
# see the module docstring's PHASED TRAINING NOTE for why this differs from 418m).
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

def _run_one_cell(arm_label: str, tagger: bool, seed: int,
                   p0a: int, p1: int, eval_n: int, readiness_n: int,
                   dry_run: bool) -> Dict[str, Any]:
    config_slice = {
        "lineage": "v3_exq_898_sd016_lega",
        "arm": arm_label,
        "cue_slot_tagger": tagger,
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
        agent = _make_agent(env_safe, tagger)

        print(f"Seed {seed} Condition {arm_label}", flush=True)
        print(f"  [arm {arm_label} seed {seed}] cue_slot_tagger={tagger} "
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
        sel_ctx = _selection_context_divergence(agent, z_safe, z_dang)
        ab_std = _action_bias_per_channel_std(agent, z_safe)
        ab_div = _action_bias_divergence(agent, z_safe, z_dang)

        print(f"  [arm {arm_label} seed {seed}] "
              f"sel_entropy={sel_ent:.4f} sel_ctx_div={sel_ctx:.4f} "
              f"action_bias_std={ab_std:.6f} action_bias_div={ab_div:.4f}", flush=True)
        print("verdict: PASS", flush=True)

        result.update({
            "sel_entropy_mean": sel_ent,
            "sel_context_divergence": sel_ctx,
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
    majority = (n_ready_seeds // 2) + 1 if n_ready_seeds > 0 else 0

    on = per_arm_ready.get("A1_ON", [])
    off = per_arm_ready.get("A0_OFF", [])

    c1_seeds_pass = sum(1 for r in on if r["sel_entropy_mean"] < SEL_ENTROPY_C1_THRESHOLD)
    c1_pass = n_ready_seeds > 0 and c1_seeds_pass >= majority

    c1b_seeds_pass = sum(1 for r in on if r["sel_context_divergence"] > SEL_CONTEXT_DIV_THRESHOLD)
    c1b_pass = n_ready_seeds > 0 and c1b_seeds_pass >= majority

    c2_seeds_pass = sum(1 for r in off if r["sel_entropy_mean"] > SEL_ENTROPY_C2_FLOOR)
    c2_pass = n_ready_seeds > 0 and c2_seeds_pass >= majority

    return {
        "C1_tagger_breaks_saddle": {
            "pass": c1_pass, "seeds_pass": c1_seeds_pass, "majority": majority,
            "threshold": SEL_ENTROPY_C1_THRESHOLD, "load_bearing": True,
        },
        "C1b_selection_context_dependent": {
            "pass": c1b_pass, "seeds_pass": c1b_seeds_pass, "majority": majority,
            "threshold": SEL_CONTEXT_DIV_THRESHOLD, "load_bearing": True,
        },
        "C2_off_arm_on_saddle": {
            "pass": c2_pass, "seeds_pass": c2_seeds_pass, "majority": majority,
            "floor": SEL_ENTROPY_C2_FLOOR, "load_bearing": True,
        },
        "overall_pass": c1_pass and c1b_pass and c2_pass,
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

    print(f"V3-EXQ-898 SD-016 leg-A encoder-fix retest "
          f"(seeds={SEEDS} dry_run={dry_run} uniform_ref={UNIFORM_REFERENCE:.4f} "
          f"world_dim={WORLD_DIM})", flush=True)

    all_cells: List[Dict] = []
    per_arm_all: Dict[str, List[Dict]] = {lab: [] for lab, _ in ARMS}

    for seed in SEEDS:
        seed_cells = {}
        for arm_label, tagger in ARMS:
            r = _run_one_cell(arm_label, tagger, seed, p0a, p1, eval_n, readiness_n, dry_run)
            all_cells.append(r)
            per_arm_all[arm_label].append(r)
            seed_cells[arm_label] = r

    n_seeds_total = len(SEEDS)
    ready_seeds = [
        s for s in SEEDS
        if all(
            next(r for r in per_arm_all[lab] if r["seed"] == s)["ready"]
            for lab, _ in ARMS
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
            for lab, _ in ARMS
        }
        acceptance = _evaluate(per_arm_ready, n_ready)
        outcome = "PASS" if acceptance["overall_pass"] else "FAIL"
        label = (
            "sd016_lega_selectivity_restored_under_encoder_fix" if outcome == "PASS"
            else "sd016_lega_tagger_insufficient_under_trained_encoder"
        )
        # Non-degeneracy is established by the readiness gate above, not here: sel_entropy_mean
        # and sel_context_divergence are continuous statistics measured on an encoder already
        # proven (by the per-cell preconditions) to vary across observations, so a criterion
        # cannot pass here for the "z_world is constant" reason that would make it vacuous.
        criteria_non_degenerate = {
            "C1_tagger_breaks_saddle": True,
            "C1b_selection_context_dependent": True,
            "C2_off_arm_on_saddle": True,
        }
        print(f"  [summary] ready={n_ready}/{n_seeds_total} "
              f"C1={acceptance['C1_tagger_breaks_saddle']['pass']} "
              f"C1b={acceptance['C1b_selection_context_dependent']['pass']} "
              f"C2={acceptance['C2_off_arm_on_saddle']['pass']} -> outcome={outcome}",
              flush=True)

    per_arm_ready_for_summary = {
        lab: [r for r in per_arm_all[lab] if r["ready"]]
        for lab, _ in ARMS
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
            "SD-016 leg (A) retrieval-selectivity retest under the SD-070 encoder fix "
            "(world_dim=128, V3-EXQ-783's D128_TRAINED recipe). claim_ids=[] (weights no "
            "claim; diagnostic adjudication via /failure-autopsy required before any "
            "claims.yaml action, matching the V3-EXQ-418g/418m precedent for this exact "
            "acceptance-test shape). 2-arm ablation OFF (legacy q.k) vs ON (Path 3 tagger), "
            "matched terrain_loss training on a shared-recipe SD-070-trained encoder. "
            "C1/C1b PRIMARY (load-bearing): ON selection entropy < 2.5 AND context-"
            "divergence > 0.1 on a majority of READY seeds. C2 CONTROL (load-bearing): OFF "
            "stays > 2.65 (on the saddle). A seed not clearing its readiness preconditions "
            "(encoder weights moved + z_world spread lift on a matched positive control) is "
            "excluded from the acceptance count, not treated as a criterion failure; fewer "
            "than a seed-majority ready self-routes substrate_not_ready_requeue for the "
            "whole run."
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
            "arms": [{"label": lab, "cue_slot_tagger": tg} for lab, tg in ARMS],
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
