#!/opt/local/bin/python3
"""
V3-EXQ-922 -- SD-016 production-combination validation: MECH-150/151/152/ARC-041

EXPERIMENT_PURPOSE: evidence

QUESTION:
  SD-016's selection-mechanism gap was resolved 2026-08-09/10 by the GOV-FANOUT-1
  portfolio: V3-EXQ-907 confirmed an explicit context-divergence auxiliary loss
  (H1, DRIVE axis) breaks the uniform-softmax saddle, and V3-EXQ-908 confirmed
  annealed Gumbel-softmax hard selection (H3, ALGORITHM axis) does too (straight-
  through top-k was eliminated: constant_peaky_degenerate). Both legs were
  promoted into substrate 2026-08-11 (ree-v3 110a2785b6): the recommended
  PRODUCTION COMBINATION is

      sd016_cue_slot_tagger=True,
      sd016_cue_slot_tagger_selection="gumbel",   # NOT "topk" -- eliminated
      sd016_context_divergence_weight=0.5,

  (see E1Config docstring, ree_core/utils/config.py ~line 494). This exact
  three-knob combination has NEVER been run together in one experiment -- 907
  and 908 each tested one axis in isolation, holding the other at its baseline.

  This experiment runs the combination for the first time and asks four
  questions that were all blocked on the identical selection-mechanism gap
  (substrate_queue.json SD-016 entry unblocks_claims):

    MECH-150 (confirmatory): does the COMBINED config produce genuinely
      selective, context-conditioned retrieval (sel_entropy_mean < 2.5 AND
      sel_context_divergence > 0.1), not just either axis alone? This is
      simultaneously the scientific MECH-150 confirmatory result AND a
      readiness gate for MECH-151/152/ARC-041 below -- an untrained or
      degenerate retrieval mechanism would otherwise masquerade as a
      MECH-151/152 falsification, so a FAIL here self-routes the whole run
      to substrate_not_ready_requeue rather than scoring the downstream
      claims on a premise that never held.

    MECH-151 (E2 leg): does E1's cue-indexed action_bias, added to
      E2.action_object() outputs, become CONTEXT-CONDITIONED (differ between
      safe and dangerous cue-contexts) now that MECH-150's retrieval carries
      genuine content? Run with use_differentiable_cem=True armed throughout
      (SD-055, HippocampalConfig) -- SD-055 fixes a documented dead-gradient
      defect in the legacy argsort/indexed-mean CEM elite-selection step that
      otherwise blocks any gradient from reaching cue_action_proj.weight when
      action_bias participates in real trajectory search (V3-EXQ-640a found
      mean_cue_action_bias_norm NULL in all 6 cells under the legacy path).

      HONEST SCOPE CAVEAT (read before citing this as an SD-055 CEM-gradient
      test): this driver trains cue_action_proj via the SAME phased,
      DIRECT-SUPERVISION recipe V3-EXQ-907/908 validated for terrain_loss
      (compute_cue_action_loss: MSE against agent.e2.action_object(z,
      action).detach(), computed by calling agent.e1.extract_cue_context()
      directly on a detached z_world). That path never calls
      agent.hippocampal.propose_trajectories() and never routes gradient
      through the CEM elite-refit HippocampalModule.propose_trajectories
      implements -- it reaches cue_action_proj by a route the legacy
      argsort/indexed-mean defect cannot touch at all, whether or not
      use_differentiable_cem is armed. use_differentiable_cem=True is set
      here because the task specification requires it armed (matching the
      production recommendation) and because it is a harmless no-op under
      this training path (HippocampalModule.propose_trajectories is never
      invoked by this driver), NOT because this driver exercises the
      specific CEM-gradient mechanism SD-055 names. A driver that actually
      calls propose_trajectories() with a live (non-detached) action_bias
      and backprops through the differentiable elite refit would be a
      materially different, higher-engineering-risk training loop with no
      existing validated recipe (V3-EXQ-568 was a diagnostic grad_max smoke
      test, not a training protocol) -- out of scope for a single-session
      driver build; flagged as a follow-on in the queue entry note. What
      THIS driver validly tests: does a validly-trained action_bias (via the
      907/908-precedent phased recipe) become context-conditioned once fed
      by MECH-150's now-genuinely-selective retrieval -- the MECH-151
      mechanism claim itself, independent of the SD-055 gradient-route
      question.

    MECH-152 (E3 leg): does terrain_weight=[w_harm, w_goal], now produced by
      the REAL end-to-end retrieval path (z_world -> ContextMemory ->
      cue_context -> cue_terrain_proj -> terrain_weight, via
      E1DeepPredictor.extract_cue_context), replicate V3-EXQ-194a's
      correlation findings (r_w_harm=0.70 PASS, r_w_goal=-0.007 FAIL)? 194a
      measured this through a STANDALONE TerrainHead trained directly on
      z_world -- bypassing ContextMemory / the retrieval question entirely.
      Pre-registered question: does r_w_harm still hold, and does r_w_goal
      IMPROVE, now that cue_context itself is genuinely context-differentiated
      (unlike every prior SD-016 retrieval test, all of which hit the uniform
      saddle)?

      DEVIATION FROM V3-EXQ-907/908's compute_terrain_loss, FLAGGED: 907/908's
      w_goal_target formula (`0.8 if hazard_max < 0.1 else 0.3`) is EXQ-194's
      ORIGINAL, since-corrected threshold -- 194a's own module docstring
      documents that with num_hazards=1/use_proxy_fields=True the hazard
      field floor is ~0.22, so `< 0.1` nearly never fires and w_goal never
      gets a real high-target training signal (this is the exact mechanism
      194's own C2 FAIL, r_w_goal=-0.007, is attributed to). 907/908 never
      noticed because they only ever gated on selection entropy/context-
      divergence, not on terrain_weight's regression quality against hazard
      proximity -- so the stale threshold was inert for their purposes.  It
      is NOT inert here: MECH-152's own acceptance criterion is exactly
      r_w_goal, so silently inheriting 907/908's threshold would replay
      194's known defect and bias the driver toward a spurious FAIL that has
      nothing to do with the retrieval question under test. THIS DRIVER USES
      194a's validated correction (`0.8 if hazard_max < 0.33 else 0.2`) in
      its own compute_terrain_loss, giving w_goal a fair (194a-precedented)
      chance to train.

    ARC-041 (dual-pathway dissociation): MECH-151 and MECH-152 measured on
      the SAME arms/seeds (already satisfied by the two questions above
      sharing one run). Per ARC-041's own what_would_answer (claims.yaml):
      CONFIRMING signature = BOTH pathways independently track cue-context
      content; FALSIFYING signature (specifically ARC-041, distinct from
      falsifying either sub-mechanism alone) = only ONE of the two pathways
      shows content-tracking while the other stays flat -- which falsifies
      the DUAL-pathway architecture even though it leaves the surviving
      pathway's own MECH-151/MECH-152 claim intact. Both-flat is a genuine
      null on BOTH sub-mechanisms under confirmed non-degenerate retrieval,
      not the dissociation-falsification signature (claims.yaml's own
      NON-DEGENERACY PRECONDITION note: the caveat about entropy pinned at
      the uniform ceiling does not apply once MECH-150 is confirmed).

  INV-040 (temporal precedence) -- EXPLICITLY DEFERRED, not tested here.
    Checked before deciding: (a) the env DOES support an early-visible hazard
    gradient without needing SD-023 landmarks -- CausalGridWorldV2's
    hazard_field_view [25] (present whenever use_proxy_fields=True, which
    V3-EXQ-907/908 and this driver already use) is documented as "a
    proximity gradient... continuous, smooth, autocorrelated, agent-
    independent" (causal_grid_world.py), i.e. exactly the kind of at-a-
    distance sensory cue INV-040's testable prediction needs; and (b)
    latent.z_harm_a (the EMA harm accumulator, tau~20) is already returned
    by agent.sense() every step, so both signals INV-040's precedence
    statistic needs are technically available within the rollouts this
    driver already runs. So the "env does not support it" escape does NOT
    apply. It is deferred anyway because the MEASUREMENT DESIGN is
    materially different from anything MECH-150/151/152/ARC-041 need: those
    are measured on static eval batches / random-action rollouts exactly
    like V3-EXQ-907/908's already-probed harness, whereas INV-040's
    precedence-timing statistic ("step-index where w_harm first crosses a
    threshold vs step-index where z_harm_a first crosses a threshold,
    averaged over episodes with hazard visible early") requires classifying
    which random-walk episodes actually contain a "hazard visible, agent not
    yet adjacent" window, and picking first-crossing thresholds for two
    differently-scaled, differently-behaved signals -- none of which has had
    its own Step-2.5a empirical probe under this config. Grafting an
    under-designed new measurement onto an already-substantial four-claim
    combined driver, with no prior probe of z_harm_a's actual rise dynamics
    here, is exactly the "would compromise the core design" case the task
    brief names for deferral. claim_ids does NOT include INV-040; a
    dedicated follow-on (its own driver, with its own Step-2.5a probe of
    approach-episode classification and threshold choice) is recommended but
    intentionally not queued or chipped by this session.

  SD-017 is explicitly OUT OF SCOPE (separate, unrelated blocker -- a
  slot_cosine_sim occupancy-mask DV defect -- per the orchestrating session's
  own investigation; not touched here).

DESIGN:
  Two arms x three seeds [42, 43, 44]:

    A0_OFF          sd016_cue_slot_tagger=False (legacy q.k attention -- C2
                    control for the MECH-150 saddle check). use_differentiable_
                    cem=True (armed identically to A1 so it is not a confound;
                    inert either way since HippocampalModule.propose_
                    trajectories is never invoked by this driver -- see the
                    MECH-151 caveat above).
    A1_PRODUCTION   sd016_enabled=True, sd016_cue_slot_tagger=True,
                    sd016_cue_slot_tagger_selection="gumbel",
                    sd016_context_divergence_weight=0.5,
                    use_differentiable_cem=True -- the exact recommended
                    combination, run together for the first time.

  Per (arm, seed) CELL (V3-EXQ-907/908 harness pattern -- SD-070 P0a warmup,
  readiness preconditions, phased P1 E1-only training on DETACHED z_world):

    P0a (encoder warmup, SD-070, world_dim=128): a dedicated warmup_env under
        RandomPolicy(seed) for P0A_EPISODES=60 (the validated operating
        point); run_zworld_p0 trains split_encoder.world_encoder +
        world_precision_logit.

    READINESS PRECONDITIONS (measured on THIS cell's trained encoder, before
      P1 -- gates P1/collection entirely for an unready cell):
      world_encoder_weights_moved   >=1 world-path tensor changed across P0a.
      z_world_spread_lift           trained/untrained z_world spread ratio,
                                    >=1.3 (V3-EXQ-783 measured ~1.7-1.9x).
    A seed whose readiness is unmet on EITHER arm self-routes to
    substrate_not_ready_requeue for that seed; fewer than a seed-majority
    ready self-routes the WHOLE RUN this way (encoder-level gate, distinct
    from the MECH-150-confirmatory gate below).

    P1 (E1-only training on DETACHED z_world, phased-training compliant --
        the encoder optimiser is never stepped in P1): P1_EPISODES=40
        episodes alternating safe/dangerous env every CONTEXT_SWITCH_EVERY=5
        episodes, training agent.e1.parameters() (tagger, cue_action_proj,
        cue_terrain_proj, world_query_proj) on:
          terrain_loss (LAMBDA_TERRAIN, w_goal_target FIXED per the
            deviation note above) always;
          cue_action_loss (LAMBDA_CUE_ACTION) in P1_late (direct MSE
            distillation against E2.action_object(), matching 907/908);
          agent.e1.compute_context_divergence_loss(z_safe_div, z_dang_div)
            for A1_PRODUCTION only (the promoted substrate method -- already
            weighted by sd016_context_divergence_weight and sign-flipped to
            REWARD divergence; ADD directly. Self-gating to a zero tensor
            with no graph when the tagger is absent or the weight is 0, so
            it would be safe to call unconditionally, but this driver only
            calls it -- and only collects the fixed safe/dangerous
            divergence-training batches -- for A1_PRODUCTION, matching
            V3-EXQ-907's economy).

    COLLECTION (MECH-152, post-P1, eval mode, no_grad): COLLECTION_EPISODES
        =20 (10 safe-context + 10 dangerous-context) episodes x up to
        STEPS_PER_EPISODE steps, random actions, logging
        (w_harm, w_goal, hazard_max) at every step via
        agent.e1.extract_cue_context(z_world) on the LIVE z_world from
        agent.sense() -- the real end-to-end path, replacing 194a's
        standalone TerrainHead. r_w_harm/r_w_goal computed via Pearson
        correlation over the pooled per-step samples, matching 194a's own
        metric exactly (thresholds inherited: r_w_harm>0.5, r_w_goal<-0.3).

PER-ARM METRICS:
  sel_entropy_mean, sel_context_divergence, sel_w_safe_hist[16],
    sel_w_dang_hist[16]        MECH-150 (same computation as V3-EXQ-907/908).
  action_bias_per_channel_std, action_bias_div, action_bias_mean_norm
                                MECH-151 (action_bias_mean_norm added: an
                                explicit non-null check against the
                                V3-EXQ-640a "mean_cue_action_bias_norm NULL"
                                signature; informative, not gating).
  r_w_harm, r_w_goal, final_terrain_loss, n_context_A_steps, n_context_B_steps
                                MECH-152 (matches V3-EXQ-194a's own metric set).
  train_ctx_div_final           (A1_PRODUCTION only) training-batch
                                divergence after P1 -- train-vs-eval
                                generalisation diagnostic, non-gating.

ACCEPTANCE CRITERIA (computed over READY seeds only, majority=(n_ready//2)+1):
  MECH-150 gate (load-bearing, ALSO the readiness precondition for 151/152/
    ARC-041 below):
    C1  A1_PRODUCTION sel_entropy_mean < 2.5 on a seed-majority.
    C1b A1_PRODUCTION sel_context_divergence > 0.1 on a seed-majority.
    C2  A0_OFF sel_entropy_mean > 2.65 on a seed-majority (control).
    mech150_confirmed = C1 AND C1b AND C2. If UNMET: the whole run self-
    routes to substrate_not_ready_requeue -- MECH-151/152/ARC-041 are NOT
    scored as pass/fail against a premise (genuine selective retrieval) that
    never held; evidence_direction_per_claim marks them non_contributory.

  MECH-151 (gated on mech150_confirmed):
    action_bias_div(A1_PRODUCTION) > action_bias_div(A0_OFF), PAIRED PER
    SEED, on a seed-majority -- a RELATIVE, DV-symmetric-safe comparison
    (no prior run has ever measured this quantity under a genuinely-
    selective retrieval, so no absolute floor is grounded; every prior
    907/908/898/418m report explicitly marks action_bias_div "non-gating"
    for exactly this reason).

  MECH-152 (gated on mech150_confirmed), inherited from V3-EXQ-194a:
    C1  r_w_harm > 0.5 on a seed-majority.
    C2  r_w_goal < -0.3 on a seed-majority.
    mech152_pass = C1 AND C2.

  ARC-041 (gated on mech150_confirmed):
    confirmed_dual_pathway        mech151_pass AND mech152_pass.
    falsified_dissociation_*      exactly one of {mech151_pass, mech152_pass}
                                  (names which pathway survives).
    neither_pathway_confirmed     neither -- a genuine null under confirmed
                                  non-degenerate retrieval (claims.yaml's own
                                  NON-DEGENERACY PRECONDITION caveat does not
                                  excuse this case, since retrieval itself is
                                  confirmed selective here).

  TOP-LEVEL outcome: PASS iff mech150_confirmed AND mech151_pass AND
    mech152_pass (full dual-pathway confirmation under the production
    combination). Every other reachable state is FAIL, sub-typed via
    interpretation.label (never a flat "insufficient"):
      substrate_not_ready_requeue                 encoder-level readiness
                                                    unmet (pre-P1 gate).
      substrate_not_ready_requeue                 (requeue_reason=
                                                    "mech150_combo_retrieval_
                                                    not_selective") MECH-150
                                                    gate unmet post-P1.
      sd016_combo_dissociation_mech151_only        ARC-041 falsified, MECH-151
                                                    stands, MECH-152 falls.
      sd016_combo_dissociation_mech152_only        ARC-041 falsified, MECH-152
                                                    stands (replicates 194a's
                                                    own C1-only pattern), MECH-
                                                    151 falls.
      sd016_combo_neither_pathway_confirmed        true null on both pathways.

DV-SYMMETRY CHECK (per /queue-experiment Step 3):
  A0_OFF          reference/control; the tagger flag changes WHICH function
                  maps z_world -> slot logits (independent feedforward MLP vs
                  q.k dot-product with its own random init), not a
                  broadcast/monotone/permutation transform of the ON arm's
                  logits -- sel_entropy_mean, sel_context_divergence,
                  action_bias_div, r_w_harm/r_w_goal are all free to differ.
  A1_PRODUCTION   compute_context_divergence_loss is DESIGNED to move
                  sel_context_divergence (V3-EXQ-907's own honesty caveat
                  applies identically here: a C1b pass is confirmatory of
                  the drive taking, not surprising on its own; C1, measured
                  on a held-out eval batch and NOT directly optimised, is
                  the free discriminating readout). action_bias_div and
                  r_w_harm/r_w_goal are NOT directly optimised by any P1
                  loss term (cue_action_loss trains action_bias toward
                  E2.action_object(), a per-sample target, not toward
                  context-divergence itself; terrain_loss trains
                  terrain_weight toward hazard_max-derived targets, not
                  toward Pearson correlation directly) -- both are free,
                  non-trivial readouts of whether training on a genuinely
                  selective cue produces context-tracking downstream
                  signals, not manipulation checks.

claim_ids: ["MECH-150", "MECH-151", "MECH-152", "ARC-041"]
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

import numpy as np
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


EXPERIMENT_TYPE = "v3_exq_922_sd016_mech151_152_arc041_production_combo"
CLAIM_IDS: List[str] = ["MECH-150", "MECH-151", "MECH-152", "ARC-041"]
EXPERIMENT_PURPOSE = "evidence"

# No frozen reference_cells literal exists for this script's own metrics pre-run
# (first run of this exact combined config). Readiness thresholds are grounded
# identically to V3-EXQ-907/908: V3-EXQ-783's measured SD-070 lift (~1.7-1.9x)
# and the V3-EXQ-737a/728 wiring-failure signature (0 tensors moved). The
# MECH-150/151/152 acceptance thresholds are grounded in V3-EXQ-907/908 (entropy/
# context-divergence) and V3-EXQ-194a (r_w_harm/r_w_goal), both prior CONFIRMED
# or precedent-setting runs on the same apparatus family.
ANCHOR_REACHABILITY_EXEMPT = (
    "no frozen reference_cells literal exists for this script's own metrics "
    "pre-run (first run of the combined production config); readiness "
    "thresholds are grounded in V3-EXQ-783's measured SD-070 lift (~1.7-1.9x) "
    "and the V3-EXQ-737a/728 wiring-failure signature (0 tensors moved); "
    "MECH-150 thresholds inherited from V3-EXQ-907/908; MECH-152 thresholds "
    "inherited from V3-EXQ-194a -- see module docstring"
)

WORLD_DIM             = 128        # the V3-EXQ-783/898/907/908 D128_TRAINED axis
P0A_EPISODES           = 60        # SD-070's validated operating point
P1_EPISODES             = 40       # unchanged from V3-EXQ-907/908
STEPS_PER_EPISODE       = 150
CONTEXT_SWITCH_EVERY    = 5
LAMBDA_TERRAIN          = 0.1
LAMBDA_CUE_ACTION       = 0.5
EVAL_BATCH_SIZE         = 32
READINESS_BATCH_SIZE    = 16
DIV_BATCH_SIZE          = 64
COLLECTION_EPISODES     = 20       # 10 safe + 10 dangerous, MECH-152 pearson series
LR                      = 1e-4
SEEDS                   = [42, 43, 44]

# (arm_label, cue_slot_tagger, selection, ctxdiv_weight, use_differentiable_cem)
ARMS: List[Tuple[str, bool, str, float, bool]] = [
    ("A0_OFF",        False, "soft",   0.0, True),
    ("A1_PRODUCTION", True,  "gumbel", 0.5, True),
]

SEL_ENTROPY_C1_THRESHOLD  = 2.5
SEL_CONTEXT_DIV_THRESHOLD = 0.1
SEL_ENTROPY_C2_FLOOR      = 2.65
UNIFORM_REFERENCE         = None   # filled at runtime = ln(16)

WORLD_ENCODER_WEIGHT_MOVE_FLOOR = 1     # >=1 tensor changed
Z_WORLD_SPREAD_LIFT_FLOOR       = 1.3   # trained/untrained spread ratio (783 ~1.7-1.9x)

R_W_HARM_THRESHOLD = 0.5    # V3-EXQ-194a C1
R_W_GOAL_THRESHOLD = -0.3   # V3-EXQ-194a C2


# ---------------------------------------------------------------------------
# Env + agent helpers (env identical to V3-EXQ-907/908).
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


def _make_agent(env: CausalGridWorldV2, cue_slot_tagger: bool, selection: str,
                 ctxdiv_weight: float, use_diff_cem: bool) -> REEAgent:
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
        sd016_context_divergence_weight=ctxdiv_weight,
        use_differentiable_cem=use_diff_cem,
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


def pearson_r(x: List[float], y: List[float]) -> float:
    """Pearson correlation between two lists (matches V3-EXQ-194a's own helper)."""
    if len(x) < 4:
        return 0.0
    xa = np.array(x, dtype=np.float64)
    ya = np.array(y, dtype=np.float64)
    xa_mean = xa - xa.mean()
    ya_mean = ya - ya.mean()
    num = float(np.dot(xa_mean, ya_mean))
    denom = float(np.sqrt(np.dot(xa_mean, xa_mean) * np.dot(ya_mean, ya_mean)))
    if denom < 1e-12:
        return 0.0
    return num / denom


def compute_terrain_loss(agent, z_world_detached, hazard_max):
    """DEVIATION FROM V3-EXQ-907/908 (flagged in module docstring): uses
    V3-EXQ-194a's CORRECTED w_goal_target threshold (<0.33), not 907/908's
    inherited-from-broken-EXQ-194 threshold (<0.1, which nearly never fires
    at this env's hazard-field floor of ~0.22 and was the documented cause
    of EXQ-194's own C2 FAIL)."""
    _, terrain_weight = agent.e1.extract_cue_context(z_world_detached)
    w_harm_target = 0.8 if hazard_max > 0.3 else 0.2
    w_goal_target = 0.8 if hazard_max < 0.33 else 0.2
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
# Selection / output probes (MECH-150/151, unchanged from V3-EXQ-907/908).
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


def _action_bias_mean_norm(agent, z_world_batch) -> float:
    """Non-null check against the V3-EXQ-640a 'mean_cue_action_bias_norm NULL'
    signature (informative, not gating)."""
    with torch.no_grad():
        action_bias, _ = agent.e1.extract_cue_context(z_world_batch)
        return float(action_bias.mean(dim=0).norm().item())


def _action_bias_divergence(agent, z_safe, z_dang) -> float:
    with torch.no_grad():
        ab_safe, _ = agent.e1.extract_cue_context(z_safe)
        ab_dang, _ = agent.e1.extract_cue_context(z_dang)
        return float((ab_safe.mean(dim=0) - ab_dang.mean(dim=0)).norm().item())


def _selection_histograms(agent, z_safe, z_dang) -> Tuple[List[float], List[float], float]:
    with torch.no_grad():
        agent.e1.extract_cue_context(z_safe)
        w_safe = agent.e1._last_cue_slot_weights.mean(dim=0)
        agent.e1.extract_cue_context(z_dang)
        w_dang = agent.e1._last_cue_slot_weights.mean(dim=0)
        ctx_div = float((w_safe - w_dang).abs().sum().item())
        return w_safe.tolist(), w_dang.tolist(), ctx_div


def _z_world_spread(agent, env, n_samples: int) -> float:
    batch = _collect_eval_batch(agent, env, n_samples)
    return float(batch.std(dim=0).mean().item())


# ---------------------------------------------------------------------------
# Eval batch collector (static snapshot, mirrors V3-EXQ-907/908).
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
# MECH-152 continuous collection: real end-to-end terrain_weight vs hazard_max,
# logged per step across actual episodes (not static batches) -- replicates
# V3-EXQ-194a's own collection methodology but through the SD-016 retrieval
# path instead of a standalone TerrainHead.
# ---------------------------------------------------------------------------

def _collect_terrain_series(agent, env_safe, env_dang, n_episodes: int,
                             steps_per_episode: int) -> Tuple[List[float], List[float], List[float]]:
    device = agent.device
    w_harm_vals: List[float] = []
    w_goal_vals: List[float] = []
    hazard_vals: List[float] = []
    half = max(1, n_episodes // 2)

    for ep in range(n_episodes):
        env = env_dang if ep >= half else env_safe
        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()

        for _step in range(steps_per_episode):
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
            agent.clock.advance()

            hazard_max = get_hazard_max(obs_dict, ow)
            with torch.no_grad():
                _, terrain_weight = agent.e1.extract_cue_context(latent.z_world.detach())

            w_harm_vals.append(float(terrain_weight[0, 0].item()))
            w_goal_vals.append(float(terrain_weight[0, 1].item()))
            hazard_vals.append(hazard_max)

            action_idx = random.randint(0, env.action_dim - 1)
            action = _onehot(action_idx, env.action_dim, device)
            with torch.no_grad():
                _, _h, done, _i, obs_dict = env.step(action)
            if done:
                break

    return w_harm_vals, w_goal_vals, hazard_vals


# ---------------------------------------------------------------------------
# P1 training episode -- E1 ONLY, on DETACHED z_world (phased-training
# compliant, matching V3-EXQ-907/908). For A1_PRODUCTION adds
# agent.e1.compute_context_divergence_loss (the promoted V3-EXQ-907 substrate
# method) on the fixed detached safe/dangerous divergence batches.
# ---------------------------------------------------------------------------

def _run_training_episode(agent, env, optimizer, phase: str, ctxdiv_active: bool,
                          z_safe_div: Optional[torch.Tensor],
                          z_dang_div: Optional[torch.Tensor]) -> int:
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

        if ctxdiv_active:
            div_loss = agent.e1.compute_context_divergence_loss(z_safe_div, z_dang_div)
            total_loss = total_loss + div_loss

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

def _run_one_cell(arm_label: str, tagger: bool, selection: str, ctxdiv_weight: float,
                   use_diff_cem: bool, seed: int, p0a: int, p1: int, eval_n: int,
                   readiness_n: int, div_n: int, coll_episodes: int, dry_run: bool,
                   zg_acc: ZGoalStreamAccumulator) -> Dict[str, Any]:
    config_slice = {
        "lineage": "v3_exq_922_sd016_production_combo",
        "arm": arm_label,
        "cue_slot_tagger": tagger,
        "selection": selection,
        "ctxdiv_weight": ctxdiv_weight,
        "use_differentiable_cem": use_diff_cem,
        "world_dim": WORLD_DIM,
        "p0a_recipe": "sd070",
        "p0a_episodes": p0a,
        "p1_episodes": p1,
        "steps_per_episode": STEPS_PER_EPISODE,
        "div_batch_size": div_n,
        "collection_episodes": coll_episodes,
    }

    with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
        env_safe = _make_env_safe(seed)
        env_dang = _make_env_dangerous(seed)
        warmup_env = _make_warmup_env(seed)
        agent = _make_agent(env_safe, tagger, selection, ctxdiv_weight, use_diff_cem)

        print(f"Seed {seed} Condition {arm_label}", flush=True)
        print(f"  [arm {arm_label} seed {seed}] cue_slot_tagger={tagger} "
              f"selection={selection} ctxdiv_weight={ctxdiv_weight} "
              f"use_differentiable_cem={use_diff_cem} world_dim={WORLD_DIM} "
              f"p0a={p0a} p1={p1}", flush=True)

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
                    "z_world batch spread (mean per-dim std), trained/untrained ratio "
                    "on the SAME apparatus. Positive control: V3-EXQ-783 measured this "
                    "recipe raise contrast_ratio ~1.7-1.9x."
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
            "ctxdiv_weight": ctxdiv_weight,
            "use_differentiable_cem": use_diff_cem,
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

        # -- divergence training batches (A1_PRODUCTION only) --------------------------
        ctxdiv_active = tagger and ctxdiv_weight > 0.0
        z_safe_div = z_dang_div = None
        if ctxdiv_active:
            z_safe_div = _collect_eval_batch(agent, env_safe, div_n).detach()
            z_dang_div = _collect_eval_batch(agent, env_dang, div_n).detach()

        # -- P1: E1 selection/action-bias/terrain-weight training -----------------------
        optimizer = optim.Adam(agent.e1.parameters(), lr=LR)
        half = p1 // 2
        for ep in range(p1):
            phase = "P1_early" if ep < half else "P1_late"
            env = env_dang if (ep // CONTEXT_SWITCH_EVERY) % 2 == 1 else env_safe
            _run_training_episode(agent, env, optimizer, phase, ctxdiv_active,
                                  z_safe_div, z_dang_div)
            if (ep + 1) % 10 == 0 or (ep + 1) == p1:
                print(f"  [train] {arm_label} seed={seed} ep {ep+1}/{p1}", flush=True)

        train_ctx_div_final = None
        if ctxdiv_active:
            with torch.no_grad():
                # compute_context_divergence_loss returns -weight*divergence (already
                # weighted and sign-flipped to REWARD divergence). Recover the raw,
                # unweighted divergence for this diagnostic so it is directly
                # comparable to sel_context_divergence (both are |mean_w_safe -
                # mean_w_dang|_1 on the same tagger, just different batches).
                _div_loss = float(
                    agent.e1.compute_context_divergence_loss(z_safe_div, z_dang_div).item()
                )
                train_ctx_div_final = (-_div_loss / ctxdiv_weight) if ctxdiv_weight > 0 else 0.0

        # -- MECH-150/151 eval (static batches, matching V3-EXQ-907/908) ---------------
        z_safe = _collect_eval_batch(agent, env_safe, eval_n)
        z_dang = _collect_eval_batch(agent, env_dang, eval_n)

        sel_ent = _selection_entropy_mean(agent, z_safe)
        w_safe_hist, w_dang_hist, sel_ctx = _selection_histograms(agent, z_safe, z_dang)
        ab_std = _action_bias_per_channel_std(agent, z_safe)
        ab_norm = _action_bias_mean_norm(agent, z_safe)
        ab_div = _action_bias_divergence(agent, z_safe, z_dang)

        # -- MECH-152 collection (continuous, real end-to-end retrieval path) ----------
        w_harm_series, w_goal_series, hazard_series = _collect_terrain_series(
            agent, env_safe, env_dang, coll_episodes, STEPS_PER_EPISODE,
        )
        r_w_harm = pearson_r(w_harm_series, hazard_series)
        r_w_goal = pearson_r(w_goal_series, hazard_series)
        n_context_A = sum(1 for h in hazard_series if h > 0.7)
        n_context_B = sum(1 for h in hazard_series if h < 0.33)
        # Mean terrain-target MSE over the COLLECTION series itself (each step's
        # own w_harm/w_goal against the target its own hazard_max implies) --
        # a genuine post-hoc convergence diagnostic (V3-EXQ-194a's C3 spirit),
        # computed directly from already-collected scalars rather than re-running
        # the network against a mismatched batch/target pairing.
        final_terrain_loss = None
        if hazard_series:
            sq_errs = []
            for wh, wg, hz in zip(w_harm_series, w_goal_series, hazard_series):
                wh_t = 0.8 if hz > 0.3 else 0.2
                wg_t = 0.8 if hz < 0.33 else 0.2
                sq_errs.append(((wh - wh_t) ** 2 + (wg - wg_t) ** 2) / 2.0)
            final_terrain_loss = sum(sq_errs) / len(sq_errs)

        print(f"  [arm {arm_label} seed {seed}] "
              f"sel_entropy={sel_ent:.4f} sel_ctx_div={sel_ctx:.4f} "
              f"action_bias_std={ab_std:.6f} action_bias_div={ab_div:.4f} "
              f"r_w_harm={r_w_harm:.4f} r_w_goal={r_w_goal:.4f}", flush=True)
        print("verdict: PASS", flush=True)

        result.update({
            "sel_entropy_mean": sel_ent,
            "sel_context_divergence": sel_ctx,
            "sel_w_safe_hist": w_safe_hist,
            "sel_w_dang_hist": w_dang_hist,
            "train_ctx_div_final": train_ctx_div_final,
            "action_bias_per_channel_std": ab_std,
            "action_bias_mean_norm": ab_norm,
            "action_bias_div": ab_div,
            "r_w_harm": r_w_harm,
            "r_w_goal": r_w_goal,
            "final_terrain_loss": final_terrain_loss,
            "n_context_A_steps": n_context_A,
            "n_context_B_steps": n_context_B,
            "n_collection_steps": len(hazard_series),
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
        "sel_context_divergence_mean":      sum(r["sel_context_divergence"] for r in ready_results) / n,
        "action_bias_div_mean":             sum(r["action_bias_div"] for r in ready_results) / n,
        "action_bias_mean_norm_mean":       sum(r["action_bias_mean_norm"] for r in ready_results) / n,
        "r_w_harm_mean":                    sum(r["r_w_harm"] for r in ready_results) / n,
        "r_w_goal_mean":                    sum(r["r_w_goal"] for r in ready_results) / n,
    }


def _evaluate(per_arm_ready: Dict[str, List[Dict]], n_ready_seeds: int) -> Dict:
    majority = (n_ready_seeds // 2) + 1 if n_ready_seeds > 0 else 0

    off = per_arm_ready.get("A0_OFF", [])
    prod = per_arm_ready.get("A1_PRODUCTION", [])

    # -- MECH-150 gate (C1/C1b on A1_PRODUCTION, C2 control on A0_OFF) -------------
    c2_seeds_pass = sum(1 for r in off if r["sel_entropy_mean"] > SEL_ENTROPY_C2_FLOOR)
    c2_pass = n_ready_seeds > 0 and c2_seeds_pass >= majority
    c1_seeds_pass = sum(1 for r in prod if r["sel_entropy_mean"] < SEL_ENTROPY_C1_THRESHOLD)
    c1_pass = n_ready_seeds > 0 and c1_seeds_pass >= majority
    c1b_seeds_pass = sum(1 for r in prod if r["sel_context_divergence"] > SEL_CONTEXT_DIV_THRESHOLD)
    c1b_pass = n_ready_seeds > 0 and c1b_seeds_pass >= majority
    mech150_confirmed = c1_pass and c1b_pass and c2_pass

    mech150 = {
        "C1_entropy_breaks_saddle": {
            "pass": c1_pass, "seeds_pass": c1_seeds_pass, "majority": majority,
            "threshold": SEL_ENTROPY_C1_THRESHOLD, "load_bearing": True,
        },
        "C1b_context_divergence": {
            "pass": c1b_pass, "seeds_pass": c1b_seeds_pass, "majority": majority,
            "threshold": SEL_CONTEXT_DIV_THRESHOLD, "load_bearing": True,
        },
        "C2_off_arm_on_saddle": {
            "pass": c2_pass, "seeds_pass": c2_seeds_pass, "majority": majority,
            "floor": SEL_ENTROPY_C2_FLOOR, "load_bearing": True,
        },
        "confirmed": mech150_confirmed,
    }

    if not mech150_confirmed:
        return {
            "mech150": mech150,
            "mech150_confirmed": False,
            "mech151": None,
            "mech152": None,
            "arc041_label": "not_interpretable_mech150_not_confirmed",
            "overall_pass": False,
        }

    # -- MECH-151: paired-per-seed relative action_bias_div comparison -------------
    ab_div_seeds_pass = 0
    ab_div_pairs = []
    for r_prod in prod:
        r_off = next((r for r in off if r["seed"] == r_prod["seed"]), None)
        if r_off is None:
            continue
        beats_off = r_prod["action_bias_div"] > r_off["action_bias_div"]
        ab_div_pairs.append({
            "seed": r_prod["seed"],
            "action_bias_div_A1": r_prod["action_bias_div"],
            "action_bias_div_A0": r_off["action_bias_div"],
            "A1_beats_A0": beats_off,
        })
        if beats_off:
            ab_div_seeds_pass += 1
    mech151_pass = n_ready_seeds > 0 and ab_div_seeds_pass >= majority
    mech151 = {
        "action_bias_div_beats_off_control": {
            "pass": mech151_pass, "seeds_pass": ab_div_seeds_pass, "majority": majority,
            "load_bearing": True,
        },
        "per_seed_pairs": ab_div_pairs,
    }

    # -- MECH-152: inherited V3-EXQ-194a thresholds ---------------------------------
    r_harm_seeds_pass = sum(1 for r in prod if r["r_w_harm"] > R_W_HARM_THRESHOLD)
    r_goal_seeds_pass = sum(1 for r in prod if r["r_w_goal"] < R_W_GOAL_THRESHOLD)
    mech152_c1_pass = n_ready_seeds > 0 and r_harm_seeds_pass >= majority
    mech152_c2_pass = n_ready_seeds > 0 and r_goal_seeds_pass >= majority
    mech152_pass = mech152_c1_pass and mech152_c2_pass
    mech152 = {
        "C1_r_w_harm": {
            "pass": mech152_c1_pass, "seeds_pass": r_harm_seeds_pass, "majority": majority,
            "threshold": R_W_HARM_THRESHOLD, "load_bearing": True,
        },
        "C2_r_w_goal": {
            "pass": mech152_c2_pass, "seeds_pass": r_goal_seeds_pass, "majority": majority,
            "threshold": R_W_GOAL_THRESHOLD, "load_bearing": True,
        },
        "pass": mech152_pass,
    }

    # -- ARC-041 dual-pathway dissociation ------------------------------------------
    if mech151_pass and mech152_pass:
        arc041_label = "confirmed_dual_pathway"
    elif mech151_pass and not mech152_pass:
        arc041_label = "falsified_dissociation_mech151_only"
    elif mech152_pass and not mech151_pass:
        arc041_label = "falsified_dissociation_mech152_only"
    else:
        arc041_label = "neither_pathway_confirmed"

    overall_pass = mech150_confirmed and mech151_pass and mech152_pass

    return {
        "mech150": mech150,
        "mech150_confirmed": True,
        "mech151": mech151,
        "mech151_pass": mech151_pass,
        "mech152": mech152,
        "mech152_pass": mech152_pass,
        "arc041_label": arc041_label,
        "overall_pass": overall_pass,
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> Dict:
    global UNIFORM_REFERENCE
    UNIFORM_REFERENCE = math.log(16)
    t0 = time.perf_counter()

    p0a         = P0A_EPISODES        if not dry_run else 2
    p1          = P1_EPISODES         if not dry_run else 3
    eval_n      = EVAL_BATCH_SIZE     if not dry_run else 4
    readiness_n = READINESS_BATCH_SIZE if not dry_run else 4
    div_n       = DIV_BATCH_SIZE      if not dry_run else 4
    coll_eps    = COLLECTION_EPISODES if not dry_run else 2

    print(f"V3-EXQ-922 SD-016 production-combination validation "
          f"(seeds={SEEDS} arms={[a[0] for a in ARMS]} dry_run={dry_run} "
          f"uniform_ref={UNIFORM_REFERENCE:.4f} world_dim={WORLD_DIM})", flush=True)

    zg_acc = ZGoalStreamAccumulator()
    all_cells: List[Dict] = []
    per_arm_all: Dict[str, List[Dict]] = {lab: [] for lab, _, _, _, _ in ARMS}

    for seed in SEEDS:
        for arm_label, tagger, selection, ctxdiv_weight, use_diff_cem in ARMS:
            r = _run_one_cell(arm_label, tagger, selection, ctxdiv_weight, use_diff_cem,
                              seed, p0a, p1, eval_n, readiness_n, div_n, coll_eps,
                              dry_run, zg_acc)
            all_cells.append(r)
            per_arm_all[arm_label].append(r)

    n_seeds_total = len(SEEDS)
    ready_seeds = [
        s for s in SEEDS
        if all(
            next(r for r in per_arm_all[lab] if r["seed"] == s)["ready"]
            for lab, _, _, _, _ in ARMS
        )
    ]
    n_ready = len(ready_seeds)
    seed_majority = (n_seeds_total // 2) + 1

    flat_preconditions = []
    for r in all_cells:
        for p in r["preconditions"]:
            flat_preconditions.append({**p, "name": f"{r['arm']}::seed{r['seed']}::{p['name']}"})

    requeue_reason = None
    if n_ready < seed_majority:
        outcome = "FAIL"
        acceptance = None
        label = "substrate_not_ready_requeue"
        requeue_reason = "encoder_readiness_unmet"
        criteria_non_degenerate = {}
        print(f"  [summary] READINESS UNMET: {n_ready}/{n_seeds_total} seeds ready "
              f"(need >= {seed_majority}) -> {label}", flush=True)
        evidence_direction = "non_contributory"
        evidence_direction_per_claim = {
            "MECH-150": "non_contributory", "MECH-151": "non_contributory",
            "MECH-152": "non_contributory", "ARC-041": "non_contributory",
        }
    else:
        per_arm_ready = {
            lab: [r for r in per_arm_all[lab] if r["ready"]]
            for lab, _, _, _, _ in ARMS
        }
        acceptance = _evaluate(per_arm_ready, n_ready)

        if not acceptance["mech150_confirmed"]:
            outcome = "FAIL"
            label = "substrate_not_ready_requeue"
            requeue_reason = "mech150_combo_retrieval_not_selective"
            criteria_non_degenerate = {"mech150_confirmed_gate": True}
            evidence_direction = "non_contributory"
            evidence_direction_per_claim = {
                "MECH-150": "does_not_support",
                "MECH-151": "non_contributory",
                "MECH-152": "non_contributory",
                "ARC-041": "non_contributory",
            }
            print(f"  [summary] MECH-150 GATE UNMET (combo retrieval not "
                  f"selective) -> {label} (requeue_reason={requeue_reason})",
                  flush=True)
        else:
            outcome = "PASS" if acceptance["overall_pass"] else "FAIL"
            arc041_label = acceptance["arc041_label"]
            if outcome == "PASS":
                label = "sd016_combo_full_confirmation_mech150_151_152_arc041"
                evidence_direction = "supports"
                evidence_direction_per_claim = {
                    "MECH-150": "supports", "MECH-151": "supports",
                    "MECH-152": "supports", "ARC-041": "supports",
                }
            elif arc041_label == "falsified_dissociation_mech151_only":
                label = "sd016_combo_dissociation_mech151_only"
                evidence_direction = "weakens"
                evidence_direction_per_claim = {
                    "MECH-150": "supports", "MECH-151": "supports",
                    "MECH-152": "does_not_support", "ARC-041": "does_not_support",
                }
            elif arc041_label == "falsified_dissociation_mech152_only":
                label = "sd016_combo_dissociation_mech152_only"
                evidence_direction = "weakens"
                evidence_direction_per_claim = {
                    "MECH-150": "supports", "MECH-151": "does_not_support",
                    "MECH-152": "supports", "ARC-041": "does_not_support",
                }
            else:
                label = "sd016_combo_neither_pathway_confirmed"
                evidence_direction = "does_not_support"
                evidence_direction_per_claim = {
                    "MECH-150": "supports", "MECH-151": "does_not_support",
                    "MECH-152": "does_not_support", "ARC-041": "does_not_support",
                }
            criteria_non_degenerate = {
                "mech150_confirmed_gate": True,
                "C1_entropy_breaks_saddle": True,
                "C2_off_arm_on_saddle": True,
            }
            print(f"  [summary] ready={n_ready}/{n_seeds_total} "
                  f"mech150_confirmed={acceptance['mech150_confirmed']} "
                  f"mech151_pass={acceptance.get('mech151_pass')} "
                  f"mech152_pass={acceptance.get('mech152_pass')} "
                  f"arc041={arc041_label} -> outcome={outcome} label={label}",
                  flush=True)

    per_arm_ready_for_summary = {
        lab: [r for r in per_arm_all[lab] if r["ready"]]
        for lab, _, _, _, _ in ARMS
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
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "evidence_direction_note": (
            "First run of the SD-016 production combination (sd016_cue_slot_tagger="
            "True, selection='gumbel', sd016_context_divergence_weight=0.5), "
            "recommended by V3-EXQ-907 (H1 ctxdiv-loss, CONFIRMED) + V3-EXQ-908 "
            "(H3 gumbel-selection, CONFIRMED; topk ELIMINATED), promoted into "
            "substrate ree-v3 110a2785b6. MECH-150 (confirmatory + readiness gate "
            "for the rest): does the COMBINED config break the uniform-softmax "
            "saddle (sel_entropy_mean<2.5 AND sel_context_divergence>0.1 vs an "
            "A0_OFF control that stays on the saddle, entropy>2.65)? If not, the "
            "run self-routes substrate_not_ready_requeue -- MECH-151/152/ARC-041 "
            "are not interpretable on a premise (genuinely selective retrieval) "
            "that never held. MECH-151 (E2 leg): does action_bias_div (paired "
            "per-seed vs the A0_OFF control) show cue-context content-tracking "
            "under use_differentiable_cem=True (SD-055)? HONEST CAVEAT (see "
            "module docstring): trained via the same phased direct-supervision "
            "recipe V3-EXQ-907/908 validated, which never routes gradient through "
            "HippocampalModule.propose_trajectories' CEM elite-refit -- so this "
            "does not specifically test SD-055's gradient-restoration mechanism, "
            "only whether action_bias becomes context-conditioned once fed by a "
            "genuinely selective retrieval. MECH-152 (E3 leg): does terrain_weight "
            "from the REAL retrieval path (not V3-EXQ-194a's standalone TerrainHead "
            "bypass) replicate 194a's r_w_harm PASS (>0.5) and, now with 194a's own "
            "corrected w_goal_target threshold (<0.33, not 907/908's inherited "
            "stale <0.1), IMPROVE on 194a's r_w_goal FAIL (was -0.007, threshold "
            "<-0.3)? ARC-041: dual-pathway dissociation across MECH-151/MECH-152 on "
            "the same arms/seeds, per claims.yaml's own confirming/falsifying "
            "signature (both track content = confirmed; exactly one = falsified "
            "dissociation, naming which pathway survives; neither = a genuine null "
            "under confirmed non-degenerate retrieval). INV-040 (temporal "
            "precedence) is explicitly DEFERRED -- see module docstring for the "
            "env-support finding (hazard_field_view does support it) and the "
            "measurement-design reasoning for deferral anyway (no Step-2.5a probe "
            "of the required episode-classification/first-crossing-threshold "
            "design existed). SD-017 is out of scope (separate, unrelated blocker)."
        ),
        "interpretation": {
            "label": label,
            "requeue_reason": requeue_reason,
            "preconditions": flat_preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "acceptance_checks": acceptance,
        "n_seeds_total": n_seeds_total,
        "n_seeds_ready": n_ready,
        "ready_seeds": ready_seeds,
        "uniform_reference": UNIFORM_REFERENCE,
        "per_arm_summaries": summaries,
        # NAMED "arm_results" deliberately -- the multi-arm substrate_hash hoist
        # (manifest_core.stamp_recording_core) keys on this exact field name.
        "arm_results": all_cells,
        "thresholds": {
            "sel_entropy_c1_threshold":        SEL_ENTROPY_C1_THRESHOLD,
            "sel_context_div_threshold":       SEL_CONTEXT_DIV_THRESHOLD,
            "sel_entropy_c2_floor":            SEL_ENTROPY_C2_FLOOR,
            "world_encoder_weight_move_floor": WORLD_ENCODER_WEIGHT_MOVE_FLOOR,
            "z_world_spread_lift_floor":       Z_WORLD_SPREAD_LIFT_FLOOR,
            "r_w_harm_threshold":              R_W_HARM_THRESHOLD,
            "r_w_goal_threshold":              R_W_GOAL_THRESHOLD,
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
            "div_batch_size":       div_n,
            "collection_episodes":  coll_eps,
            "lr":                   LR,
            "arms": [
                {"label": lab, "cue_slot_tagger": tg, "selection": sel,
                 "ctxdiv_weight": cd, "use_differentiable_cem": udc}
                for lab, tg, sel, cd, udc in ARMS
            ],
            "sd016_enabled":               True,
            "sd016_writepath_mode":        "off",
            "p0a_recipe":                  "sd070",
            "w_goal_target_threshold":     "0.33 (V3-EXQ-194a corrected, NOT 907/908's inherited 0.1)",
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
