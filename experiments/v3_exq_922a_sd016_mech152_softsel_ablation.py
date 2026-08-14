#!/opt/local/bin/python3
"""
V3-EXQ-922a -- SD-016 MECH-152 soft-selection ablation: discriminate selection-
hardness interaction from mechanism absence.

EXPERIMENT_PURPOSE: diagnostic

LINEAGE / SUPERSESSION:
  This is a follow-on DIAGNOSTIC of V3-EXQ-922, NOT a correction of it. It does
  NOT supersede 922 -- 922's production-combination evidence stands verbatim
  (MECH-150 CONFIRMED, MECH-151 SUPPORTS scoped, MECH-152 FAIL 0/3 under the
  production hard-Gumbel combo, ARC-041 dissociation signature met but HELD).
  922a asks a strictly narrower, orthogonal question: WHY did MECH-152 collapse?
  Confirmed autopsy: failure_autopsy_V3-EXQ-922_2026-08-13 (.md/.json).

QUESTION:
  V3-EXQ-922's MECH-152 leg FAILed cleanly under the production combination
  (sd016_cue_slot_tagger=True, selection="gumbel", sd016_context_divergence_
  weight=0.5): A1_PRODUCTION scored r_w_harm_mean=0.0013 (threshold >0.5) and
  r_w_goal_mean=-0.0077 (threshold <-0.3), both 0/3. But the A0_OFF arm (legacy
  q.k soft-selection, no ctxdiv) in the SAME run scored MUCH closer to the 194a
  thresholds (r_w_harm_mean=0.356, r_w_goal_mean=-0.355 -- the latter would even
  individually clear the <-0.3 bar). A0_OFF differs from A1_PRODUCTION in THREE
  knobs at once (tagger off, soft-not-gumbel, ctxdiv off), so its proximity is a
  live but confounded interpretive lead. This experiment cleanly DISCRIMINATES
  the selection-hardness axis by holding tagger and ctxdiv fixed and flipping
  only the selection operator:

    Is MECH-152's terrain-weight content-tracking collapse SELECTION-HARDNESS-
    SPECIFIC (hard-Gumbel one-hot selection suppresses the graded correlational
    terrain signal, while soft-weighted selection preserves it), or a GENUINE
    ABSENCE of the MECH-152 mechanism through the end-to-end retrieval path
    (neither soft nor hard selection produces terrain content-tracking)?

  Biological plausibility of the selection-hardness hypothesis (autopsy sec 3):
  MECH-152's terrain_weight is a CA3-completion-like graded correlational
  readout; the Gumbel-softmax categorical WTA-like hardness MECH-150 needs for
  selective retrieval may not coexist well with that graded output from the same
  single E1 mechanism -- soft selection would then be the regime in which the
  graded terrain signal survives.

  Why this is NOT already answered by V3-EXQ-907's A2_ctxdiv_l0p5 arm (which is
  config-close: tagger=True, selection default "soft", ctxdiv=0.5): 907 measured
  only the MECH-150 selection metrics (entropy / context-divergence) via its
  inline pre-promotion ctxdiv term. It NEVER ran the MECH-152 terrain collection
  (r_w_harm/r_w_goal via the real end-to-end retrieval path) and NEVER used
  194a's corrected w_goal_target threshold. The decisive readout for this
  discrimination is therefore recorded in no prior manifest (GOV-REUSE-1 checked:
  907's MECH-150-only metrics do not carry r_w_harm/r_w_goal; 922 ran soft only
  on the tagger-OFF A0 arm, not on a tagger+ctxdiv soft arm) -> run.

DESIGN:
  Three arms x three seeds [42, 43, 44]. The MECH-152 terrain-collection
  machinery, the 194a-corrected w_goal_target threshold (<0.33), the promoted
  compute_context_divergence_loss, the phased P0a/P1 training, the readiness
  preconditions and the MECH-150 selectivity gate are all IDENTICAL to
  V3-EXQ-922 -- only the arm set changes, so A2_SOFTSEL is a true 1-knob
  ablation of A1_PRODUCTION.

    A0_OFF          sd016_cue_slot_tagger=False, selection="soft", ctxdiv=0.0.
                    Legacy q.k attention. Serves TWO roles: (i) the MECH-150 C2
                    on-saddle control (entropy > 2.65), and (ii) reproduces
                    922's closest-to-threshold arm as an in-run anchor.
    A1_PRODUCTION   tagger=True, selection="gumbel", ctxdiv=0.5. The exact 922
                    production combo -- reproduces the r_w collapse in-run so the
                    A2-vs-A1 contrast is measured on this run's own apparatus,
                    not across runs.
    A2_SOFTSEL      tagger=True, selection="soft", ctxdiv=0.5. THE DISCRIMINATING
                    ABLATION. Differs from A1_PRODUCTION in the selection
                    operator ONLY (gumbel -> soft); tagger and ctxdiv held
                    identical. This is the arm whose MECH-152 terrain result
                    answers the question.

  use_differentiable_cem=True on all arms (armed identically to 922; a
  documented no-op under this training path -- this driver never calls
  HippocampalModule.propose_trajectories, so it does not exercise SD-055's
  CEM-gradient mechanism; see V3-EXQ-922's MECH-151 caveat).

  Per (arm, seed) cell = V3-EXQ-907/908/922 harness: P0a SD-070 encoder warmup
  (world_dim=128, 60 episodes), encoder-readiness preconditions gating P1,
  phased P1 E1-only training on DETACHED z_world (terrain_loss always +
  cue_action_loss in P1_late + compute_context_divergence_loss for the two
  tagger+ctxdiv arms A1/A2 only), then the MECH-152 continuous terrain collection
  (real end-to-end path z_world -> ContextMemory -> extract_cue_context ->
  terrain_weight, 20 episodes, random actions, Pearson r_w_harm/r_w_goal vs
  hazard_max over pooled per-step samples, thresholds inherited from 194a:
  r_w_harm>0.5, r_w_goal<-0.3).

DISCRIMINATION (this is a diagnostic -- the interpretation.label IS the verdict;
  governance adjudicates, this run does NOT finalize MECH-152 or ARC-041):

  READINESS PREMISE (interpretation.preconditions, load-bearing). The A2 terrain
  result is only interpretable if A2_SOFTSEL's retrieval is genuinely selective
  (else a terrain collapse means "retrieval never selective", not "mechanism
  absent"). So the load-bearing preconditions are:
    - A2_SOFTSEL sel_entropy_mean < 2.5 (soft+ctxdiv breaks the uniform saddle),
    - A2_SOFTSEL sel_context_divergence > 0.1,
    - A2_SOFTSEL terrain series NON-DEGENERATE (hazard varies AND terrain_weight
      varies, so Pearson r is a real measurement not a pearson_r()=0 degeneracy).
  If A2 retrieval is not selective (either C1/C1b unmet) the WHOLE run self-
  routes substrate_not_ready_requeue (requeue_reason=
  "a2_softsel_retrieval_not_selective") -- NEVER a mechanism-verdict label.
  A0_OFF C2 (entropy>2.65 control) and A1_PRODUCTION selectivity + A1 r_w
  reproduce-922 are reported in acceptance_checks as CONTEXT, not as adjudication-
  gating preconditions (an A1 that fails to reproduce is a flagged confound, but
  must not vacate the A2 discrimination -- 785-class regime conditioning).

  VERDICT LABELS (pre-registered), computed over READY seeds, majority=
  (n_ready//2)+1, only when the A2 readiness premise holds and A2's terrain
  series is non-degenerate:

    selection_hardness_confirmed_full     H_selection_hardness (strong). A2_
                                          SOFTSEL clears BOTH 194a thresholds on
                                          a seed-majority while A1_PRODUCTION does
                                          not. Soft selection recovers the terrain
                                          signal; hard-Gumbel suppresses it. The
                                          MECH-152 mechanism is PRESENT but
                                          selection-hardness-sensitive.
    selection_hardness_partial_recovery   H_selection_hardness (graded). A2 does
                                          not fully clear the 194a bar but
                                          SUBSTANTIALLY recovers toward it vs A1:
                                          (r_w_harm_mean_A2 - r_w_harm_mean_A1) >
                                          RECOVERY_MARGIN AND (r_w_goal_mean_A1 -
                                          r_w_goal_mean_A2) > RECOVERY_MARGIN
                                          (RECOVERY_MARGIN=0.15, grounded in 922's
                                          ~0.35 A0-vs-A1 gap). Soft selection
                                          materially recovers terrain-precision
                                          tracking but not to the full bar.
    mechanism_absence_indicated           H_mechanism_absence. A2 clears neither
                                          194a threshold AND does not substantially
                                          recover vs A1 (A2 ~ A1, both flat). The
                                          collapse is NOT selection-hardness-
                                          specific -- terrain content-tracking
                                          genuinely does not survive the end-to-end
                                          retrieval path regardless of selection
                                          operator.
    both_selections_recover_a1_not_reproduced   CONFOUND FLAG. A2 AND A1 both
                                          clear the 194a bar -- 922's A1 collapse
                                          did not reproduce in-run, so the contrast
                                          is uninterpretable this run. Route to
                                          re-examination, not a mechanism verdict.
    inverted_hard_better_than_soft        UNEXPECTED FLAG. A1 clears the bar and
                                          A2 does not / does not recover -- soft
                                          worse than hard, contradicting the
                                          interpretive lead. Route to re-examination.

  outcome PASS = a real discrimination was produced (A2 readiness premise held,
  terrain non-degenerate, a verdict label assigned). outcome FAIL = probe could
  not discriminate (substrate_not_ready_requeue, or A2 terrain series degenerate
  -> label "a2_terrain_series_degenerate"). A diagnostic PASS/FAIL is about
  whether the probe RAN meaningfully; the scientific reading lives in the label.

DV-SYMMETRY CHECK (per /queue-experiment Step 3, per arm):
  A0_OFF          reference/control. The tagger flag changes WHICH function maps
                  z_world -> slot logits (feedforward tagger vs q.k dot-product),
                  not a broadcast/monotone/permutation transform -- r_w_harm/
                  r_w_goal free to differ. Manipulation NOT invariant under the
                  Pearson-correlation DV.
  A1 vs A2        the manipulated variable is the selection operator (hard-Gumbel
                  near-one-hot vs soft-weighted mixture). This changes which slots
                  are retrieved and therefore cue_context as a POINT IN SPACE (not
                  a scalar multiple of A1's), so terrain_weight = cue_terrain_proj
                  (cue_context) differs NON-monotonically per step. Pearson r is
                  invariant under a positive-affine rescale of terrain_weight, but
                  the selection swap is NOT an affine rescale of A1's terrain
                  series -- it changes the retrieved content -- so the per-step
                  (w_harm, w_goal) series, and hence r_w_harm/r_w_goal, are free
                  to differ. The manipulation is NOT invariant under the DV's
                  symmetry group; the delta is a genuine measurement, not an
                  arithmetic identity.
  Readiness-gate scope note (604c-class): the A2 MECH-150 selectivity gate
  certifies the A2 retrieval channel specifically; it does not speak for A0/A1.

INV-040 / SD-017 / MECH-151 / ARC-041: not tested here. MECH-151's action_bias_
  div is recorded per arm as informative context (it lets a later reader see
  whether soft selection also affects the MECH-151 leg, feeding ARC-041's HELD
  dual-pathway disposition) but is NOT a scored claim in this diagnostic. ARC-041
  is HELD (2026-08-13 Step 8 gate) and is only INFORMED, never finalized, by this
  run -- that remains /governance's call once this ablation lands.

claim_ids: ["MECH-152"]
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


EXPERIMENT_TYPE = "v3_exq_922a_sd016_mech152_softsel_ablation"
CLAIM_IDS: List[str] = ["MECH-152"]
EXPERIMENT_PURPOSE = "diagnostic"

# No frozen reference_cells literal exists for this script's own metrics pre-run
# (first run of this soft-selection ablation config). Readiness thresholds are
# grounded identically to V3-EXQ-907/908/922: V3-EXQ-783's measured SD-070 lift
# (~1.7-1.9x) and the V3-EXQ-737a/728 wiring-failure signature (0 tensors moved).
# The MECH-150 selectivity thresholds are grounded in V3-EXQ-907/908 (entropy/
# context-divergence) and the MECH-152 r_w_harm/r_w_goal thresholds in
# V3-EXQ-194a; the RECOVERY_MARGIN is grounded in V3-EXQ-922's measured ~0.35
# A0_OFF-vs-A1_PRODUCTION r_w gap. All prior CONFIRMED / precedent-setting runs
# on the same apparatus family.
ANCHOR_REACHABILITY_EXEMPT = (
    "no frozen reference_cells literal exists for this script's own metrics "
    "pre-run (first run of the soft-selection ablation config); readiness "
    "thresholds are grounded in V3-EXQ-783's measured SD-070 lift (~1.7-1.9x) "
    "and the V3-EXQ-737a/728 wiring-failure signature (0 tensors moved); "
    "MECH-150 selectivity thresholds inherited from V3-EXQ-907/908; MECH-152 "
    "r_w thresholds inherited from V3-EXQ-194a; RECOVERY_MARGIN grounded in "
    "V3-EXQ-922's A0-vs-A1 gap -- see module docstring"
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
# A2_SOFTSEL differs from A1_PRODUCTION in the selection operator ONLY
# (gumbel -> soft); cue_slot_tagger and ctxdiv_weight are held identical, so it
# is a true 1-knob selection-hardness ablation.
ARMS: List[Tuple[str, bool, str, float, bool]] = [
    ("A0_OFF",        False, "soft",   0.0, True),
    ("A1_PRODUCTION", True,  "gumbel", 0.5, True),
    ("A2_SOFTSEL",    True,  "soft",   0.5, True),
]

SEL_ENTROPY_C1_THRESHOLD  = 2.5
SEL_CONTEXT_DIV_THRESHOLD = 0.1
SEL_ENTROPY_C2_FLOOR      = 2.65
UNIFORM_REFERENCE         = None   # filled at runtime = ln(16)

WORLD_ENCODER_WEIGHT_MOVE_FLOOR = 1     # >=1 tensor changed
Z_WORLD_SPREAD_LIFT_FLOOR       = 1.3   # trained/untrained spread ratio (783 ~1.7-1.9x)

R_W_HARM_THRESHOLD = 0.5    # V3-EXQ-194a C1
R_W_GOAL_THRESHOLD = -0.3   # V3-EXQ-194a C2

# Pre-registered "substantial recovery" bar for the partial-recovery verdict:
# A2_SOFTSEL must move toward each 194a threshold vs A1_PRODUCTION by at least
# this margin. Grounded in V3-EXQ-922's measured A0_OFF-vs-A1 gap (r_w_harm
# 0.356 vs 0.001 = 0.355; r_w_goal -0.355 vs -0.008 = -0.347) -- ~0.15 is
# roughly half that gap, a materially-non-trivial recovery that is not full pass.
RECOVERY_MARGIN = 0.15

# Terrain-series non-degeneracy floors (so a Pearson r is a real measurement,
# not a pearson_r()=0 artifact of a constant hazard or a pinned terrain_weight).
HAZARD_STD_FLOOR        = 1e-6
TERRAIN_WEIGHT_STD_FLOOR = 1e-9


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
        "lineage": "v3_exq_922a_sd016_mech152_softsel_ablation",
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
        # Terrain-series non-degeneracy inputs (so a later reader / _evaluate can
        # confirm a Pearson r is a real measurement, not a pearson_r()=0 artifact
        # of a constant hazard field or a pinned terrain_weight).
        hazard_std = float(np.std(np.asarray(hazard_series, dtype=np.float64))) if hazard_series else 0.0
        w_harm_std = float(np.std(np.asarray(w_harm_series, dtype=np.float64))) if w_harm_series else 0.0
        w_goal_std = float(np.std(np.asarray(w_goal_series, dtype=np.float64))) if w_goal_series else 0.0
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
            "hazard_std": hazard_std,
            "w_harm_std": w_harm_std,
            "w_goal_std": w_goal_std,
            "terrain_series_nondegenerate": (
                len(hazard_series) >= 4
                and hazard_std > HAZARD_STD_FLOOR
                and w_harm_std > TERRAIN_WEIGHT_STD_FLOOR
                and w_goal_std > TERRAIN_WEIGHT_STD_FLOOR
            ),
        })
        zg_acc.observe(agent)
        cell.stamp(result)
        return result


# ---------------------------------------------------------------------------
# Aggregation + acceptance.
# ---------------------------------------------------------------------------

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _majority_order_stat(values: List[float], majority: int, direction: str) -> float:
    """The single value whose comparison to a threshold EQUALS the majority test,
    so an indexer that recomputes `met` from (measured, threshold) agrees exactly
    with the seed-majority `met` we assert (the skill's same-statistic rule).

    - direction 'upper' (ceiling; met when value < threshold; "at least `majority`
      seeds below ceiling"): the majority-th SMALLEST value < threshold iff at
      least `majority` are below. Return ascending[majority-1].
    - direction 'lower' (floor; met when value > threshold; "at least `majority`
      seeds above floor"): the majority-th LARGEST value > threshold iff at least
      `majority` are above. Return descending[majority-1].
    """
    if not values or majority <= 0:
        return 0.0
    k = min(majority, len(values)) - 1
    if direction == "upper":
        return sorted(values)[k]                 # majority-th smallest
    return sorted(values, reverse=True)[k]       # majority-th largest


def _summarise(ready_results: List[Dict]) -> Dict:
    n = len(ready_results)
    if n == 0:
        return {"n_seeds": 0}
    return {
        "n_seeds":                          n,
        "sel_entropy_mean_mean":            _mean([r["sel_entropy_mean"] for r in ready_results]),
        "sel_context_divergence_mean":      _mean([r["sel_context_divergence"] for r in ready_results]),
        "action_bias_div_mean":             _mean([r["action_bias_div"] for r in ready_results]),
        "action_bias_mean_norm_mean":       _mean([r["action_bias_mean_norm"] for r in ready_results]),
        "r_w_harm_mean":                    _mean([r["r_w_harm"] for r in ready_results]),
        "r_w_goal_mean":                    _mean([r["r_w_goal"] for r in ready_results]),
        "terrain_series_nondegenerate_n":   sum(1 for r in ready_results if r.get("terrain_series_nondegenerate")),
    }


def _evaluate(per_arm_ready: Dict[str, List[Dict]], n_ready_seeds: int) -> Dict:
    """MECH-152 selection-hardness discrimination (diagnostic -- the label IS the
    verdict; governance adjudicates). Load-bearing premise: A2_SOFTSEL retrieval
    selective AND its terrain series non-degenerate. A0_OFF C2 control and
    A1_PRODUCTION selectivity / r_w reproduce-922 are CONTEXT (reported, never
    adjudication-gating), so a non-reproducing A1 does not vacate the A2 read
    (785-class regime conditioning)."""
    majority = (n_ready_seeds // 2) + 1 if n_ready_seeds > 0 else 0

    off  = per_arm_ready.get("A0_OFF", [])
    prod = per_arm_ready.get("A1_PRODUCTION", [])
    soft = per_arm_ready.get("A2_SOFTSEL", [])

    def _selectivity(rows: List[Dict]) -> Dict:
        c1 = sum(1 for r in rows if r["sel_entropy_mean"] < SEL_ENTROPY_C1_THRESHOLD)
        c1b = sum(1 for r in rows if r["sel_context_divergence"] > SEL_CONTEXT_DIV_THRESHOLD)
        return {
            "c1_seeds_pass": c1, "c1_pass": n_ready_seeds > 0 and c1 >= majority,
            "c1b_seeds_pass": c1b, "c1b_pass": n_ready_seeds > 0 and c1b >= majority,
            "sel_entropy_mean": _mean([r["sel_entropy_mean"] for r in rows]),
            "sel_context_divergence_mean": _mean([r["sel_context_divergence"] for r in rows]),
            "sel_entropy_majority_stat": _majority_order_stat(
                [r["sel_entropy_mean"] for r in rows], majority, "upper"),
            "sel_ctxdiv_majority_stat": _majority_order_stat(
                [r["sel_context_divergence"] for r in rows], majority, "lower"),
        }

    def _mech152(rows: List[Dict]) -> Dict:
        rh = sum(1 for r in rows if r["r_w_harm"] > R_W_HARM_THRESHOLD)
        rg = sum(1 for r in rows if r["r_w_goal"] < R_W_GOAL_THRESHOLD)
        c1 = n_ready_seeds > 0 and rh >= majority
        c2 = n_ready_seeds > 0 and rg >= majority
        return {
            "C1_r_w_harm": {"pass": c1, "seeds_pass": rh, "majority": majority,
                            "threshold": R_W_HARM_THRESHOLD},
            "C2_r_w_goal": {"pass": c2, "seeds_pass": rg, "majority": majority,
                            "threshold": R_W_GOAL_THRESHOLD},
            "pass": c1 and c2,
            "r_w_harm_mean": _mean([r["r_w_harm"] for r in rows]),
            "r_w_goal_mean": _mean([r["r_w_goal"] for r in rows]),
            "r_w_harm_per_seed": [{"seed": r["seed"], "r_w_harm": r["r_w_harm"]} for r in rows],
            "r_w_goal_per_seed": [{"seed": r["seed"], "r_w_goal": r["r_w_goal"]} for r in rows],
            "terrain_nondegenerate_seeds": sum(1 for r in rows if r.get("terrain_series_nondegenerate")),
        }

    sel_a0_c2 = sum(1 for r in off if r["sel_entropy_mean"] > SEL_ENTROPY_C2_FLOOR)
    selectivity = {
        "A0_OFF_C2_on_saddle": {
            "seeds_pass": sel_a0_c2, "pass": n_ready_seeds > 0 and sel_a0_c2 >= majority,
            "floor": SEL_ENTROPY_C2_FLOOR, "role": "control",
            "sel_entropy_mean": _mean([r["sel_entropy_mean"] for r in off]),
        },
        "A1_PRODUCTION": {**_selectivity(prod), "role": "reproduce_922_context"},
        "A2_SOFTSEL": {**_selectivity(soft), "role": "load_bearing_premise"},
    }
    a2_retrieval_selective = selectivity["A2_SOFTSEL"]["c1_pass"] and selectivity["A2_SOFTSEL"]["c1b_pass"]

    mech152_a1 = _mech152(prod)
    mech152_a2 = _mech152(soft)
    a2_terrain_nondegenerate = n_ready_seeds > 0 and mech152_a2["terrain_nondegenerate_seeds"] >= majority
    a1_terrain_nondegenerate = n_ready_seeds > 0 and mech152_a1["terrain_nondegenerate_seeds"] >= majority

    # -- recovery magnitude A2 vs A1 (the graded discrimination readout) ----------
    harm_recovery = mech152_a2["r_w_harm_mean"] - mech152_a1["r_w_harm_mean"]   # >0 = A2 toward >0.5
    goal_recovery = mech152_a1["r_w_goal_mean"] - mech152_a2["r_w_goal_mean"]   # >0 = A2 toward <-0.3
    substantial_recovery = harm_recovery > RECOVERY_MARGIN and goal_recovery > RECOVERY_MARGIN

    # -- MECH-151 action_bias context-tracking (INFORMATIVE, NOT a scored claim) --
    ab_div_context = {
        "A0_OFF_mean":        _mean([r["action_bias_div"] for r in off]),
        "A1_PRODUCTION_mean": _mean([r["action_bias_div"] for r in prod]),
        "A2_SOFTSEL_mean":    _mean([r["action_bias_div"] for r in soft]),
        "note": ("informative only; MECH-151 is NOT a scored claim in this "
                 "diagnostic -- recorded so a later reader can see whether soft "
                 "selection also moves the MECH-151 leg (feeds ARC-041's HELD "
                 "disposition, which /governance adjudicates)"),
    }

    return {
        "n_ready_seeds": n_ready_seeds,
        "majority": majority,
        "selectivity": selectivity,
        "a2_retrieval_selective": a2_retrieval_selective,
        "mech152_A1_PRODUCTION": mech152_a1,
        "mech152_A2_SOFTSEL": mech152_a2,
        "a2_terrain_nondegenerate": a2_terrain_nondegenerate,
        "a1_terrain_nondegenerate": a1_terrain_nondegenerate,
        "recovery": {
            "harm_recovery_A2_minus_A1": harm_recovery,
            "goal_recovery_A1_minus_A2": goal_recovery,
            "recovery_margin": RECOVERY_MARGIN,
            "substantial_recovery": substantial_recovery,
        },
        "mech151_action_bias_div_informative": ab_div_context,
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

    print(f"V3-EXQ-922a SD-016 MECH-152 soft-selection ablation "
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

    # interpretation.preconditions carries ONLY the aggregate load-bearing
    # premises (recomputable, arm-scoped to the discriminating A2 arm), NOT the
    # flat per-cell encoder preconditions -- a single unready A0/A1 cell must not
    # vacate the whole diagnostic at the indexer's arm-blind adjudication (785).
    # The per-cell encoder preconditions are preserved in `cell_readiness` below.
    preconditions: List[Dict[str, Any]] = [
        {
            "name": "seed_readiness_majority",
            "description": ("a seed-majority of seeds cleared encoder P0a readiness "
                            "(world_encoder_weights_moved + z_world_spread_lift) on "
                            "ALL arms -- the pre-P1 encoder-level gate"),
            "control": "encoder P0a readiness across all arms, per seed",
            "measured": float(n_ready), "threshold": float(seed_majority),
            "direction": "lower", "met": n_ready >= seed_majority,
        },
    ]
    criteria: List[Dict[str, Any]] = []
    criteria_non_degenerate: Dict[str, Any] = {}
    requeue_reason = None

    if n_ready < seed_majority:
        outcome = "FAIL"
        acceptance = None
        label = "substrate_not_ready_requeue"
        requeue_reason = "encoder_readiness_unmet"
        evidence_direction = "non_contributory"
        print(f"  [summary] READINESS UNMET: {n_ready}/{n_seeds_total} seeds ready "
              f"(need >= {seed_majority}) -> {label}", flush=True)
    else:
        per_arm_ready = {
            lab: [r for r in per_arm_all[lab] if r["ready"]]
            for lab, _, _, _, _ in ARMS
        }
        acceptance = _evaluate(per_arm_ready, n_ready)
        sel_a2 = acceptance["selectivity"]["A2_SOFTSEL"]
        m152_a1 = acceptance["mech152_A1_PRODUCTION"]
        m152_a2 = acceptance["mech152_A2_SOFTSEL"]
        a2_selective = acceptance["a2_retrieval_selective"]
        a2_nondegen = acceptance["a2_terrain_nondegenerate"]
        subst = acceptance["recovery"]["substantial_recovery"]

        # -- load-bearing premises for the A2 discrimination (recomputable) --------
        preconditions.append({
            "name": "a2_softsel_retrieval_entropy_selective",
            "description": ("A2_SOFTSEL sel_entropy_mean below the saddle-break "
                            "ceiling on a seed-majority (soft+ctxdiv yields selective "
                            "retrieval) -- premise for interpreting A2's terrain result"),
            "control": "A2_SOFTSEL retrieval channel; majority-order statistic",
            "measured": float(sel_a2["sel_entropy_majority_stat"]),
            "threshold": float(SEL_ENTROPY_C1_THRESHOLD),
            "direction": "upper", "met": bool(sel_a2["c1_pass"]),
        })
        preconditions.append({
            "name": "a2_softsel_retrieval_ctxdiv_selective",
            "description": ("A2_SOFTSEL sel_context_divergence above floor on a "
                            "seed-majority -- retrieval is genuinely context-conditioned"),
            "control": "A2_SOFTSEL retrieval channel; majority-order statistic",
            "measured": float(sel_a2["sel_ctxdiv_majority_stat"]),
            "threshold": float(SEL_CONTEXT_DIV_THRESHOLD),
            "direction": "lower", "met": bool(sel_a2["c1b_pass"]),
        })
        preconditions.append({
            "name": "a2_softsel_terrain_series_nondegenerate",
            "description": ("a seed-majority of A2_SOFTSEL cells have a non-degenerate "
                            "terrain collection (hazard varies AND terrain_weight varies) "
                            "so Pearson r_w is a real measurement, not a pearson_r()=0 "
                            "artifact"),
            "control": "A2_SOFTSEL terrain series non-degeneracy count",
            "measured": float(m152_a2["terrain_nondegenerate_seeds"]),
            "threshold": float(acceptance["majority"]),
            "direction": "lower", "met": bool(a2_nondegen),
        })

        criteria = [
            {"name": "a2_retrieval_selective_premise", "load_bearing": True,
             "passed": bool(a2_selective)},
            {"name": "a2_terrain_nondegenerate_premise", "load_bearing": True,
             "passed": bool(a2_nondegen)},
        ]
        criteria_non_degenerate = {
            "a2_retrieval_selective": bool(a2_selective),
            "a2_terrain_series_nondegenerate": bool(a2_nondegen),
            "a1_terrain_series_nondegenerate": bool(acceptance["a1_terrain_nondegenerate"]),
        }

        if not a2_selective:
            # A2's terrain result is uninterpretable -- retrieval never selective.
            outcome = "FAIL"
            label = "substrate_not_ready_requeue"
            requeue_reason = "a2_softsel_retrieval_not_selective"
            evidence_direction = "non_contributory"
        elif not a2_nondegen:
            # Terrain series degenerate -> a "collapse" reading is an artifact.
            outcome = "FAIL"
            label = "a2_terrain_series_degenerate"
            requeue_reason = "a2_terrain_series_degenerate"
            evidence_direction = "non_contributory"
        else:
            # Premise holds -> the discrimination is interpretable. A diagnostic
            # that produces a verdict is a PASS (the probe RAN meaningfully); the
            # scientific reading lives in the label.
            a1p = m152_a1["pass"]
            a2p = m152_a2["pass"]
            outcome = "PASS"
            if a2p and not a1p:
                label = "selection_hardness_confirmed_full"
                evidence_direction = "mixed"
            elif a2p and a1p:
                label = "both_selections_recover_a1_not_reproduced"
                evidence_direction = "unknown"
            elif (not a2p) and subst:
                label = "selection_hardness_partial_recovery"
                evidence_direction = "mixed"
            elif (not a2p) and (not subst) and a1p:
                label = "inverted_hard_better_than_soft"
                evidence_direction = "unknown"
            else:
                label = "mechanism_absence_indicated"
                evidence_direction = "does_not_support"
            print(f"  [summary] ready={n_ready}/{n_seeds_total} "
                  f"a2_selective={a2_selective} a2_nondegen={a2_nondegen} "
                  f"mech152_A1_pass={a1p} mech152_A2_pass={a2p} "
                  f"harm_recov={acceptance['recovery']['harm_recovery_A2_minus_A1']:.4f} "
                  f"goal_recov={acceptance['recovery']['goal_recovery_A1_minus_A2']:.4f} "
                  f"substantial={subst} -> outcome={outcome} label={label}",
                  flush=True)

    if requeue_reason and label != "a2_terrain_series_degenerate":
        print(f"  [summary] {label} (requeue_reason={requeue_reason})", flush=True)

    # Per-cell encoder readiness preserved for the record (NOT in the adjudication
    # preconditions list, per the arm-blind vacating hazard above).
    cell_readiness = []
    for r in all_cells:
        for p in r["preconditions"]:
            cell_readiness.append({**p, "name": f"{r['arm']}::seed{r['seed']}::{p['name']}"})

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
        "evidence_direction_note": (
            "DIAGNOSTIC (excluded from governance confidence/conflict scoring). "
            "Follow-on of V3-EXQ-922 (does NOT supersede it -- 922's production-"
            "combination evidence stands). Confirmed autopsy: "
            "failure_autopsy_V3-EXQ-922_2026-08-13. V3-EXQ-922's MECH-152 leg "
            "FAILed under the production combo (A1_PRODUCTION r_w_harm_mean=0.0013, "
            "r_w_goal_mean=-0.0077, both 0/3), while its A0_OFF arm (legacy soft "
            "selection) scored much closer to the 194a thresholds (0.356/-0.355). "
            "This diagnostic DISCRIMINATES the selection-hardness axis by a clean "
            "1-knob ablation: A2_SOFTSEL (tagger=True, selection='soft', "
            "ctxdiv=0.5) vs A1_PRODUCTION (identical but selection='gumbel'), "
            "tagger and ctxdiv held. Load-bearing premise: A2_SOFTSEL retrieval "
            "must be selective (sel_entropy<2.5 AND sel_context_divergence>0.1) "
            "and its terrain series non-degenerate, else the run self-routes "
            "substrate_not_ready_requeue / a2_terrain_series_degenerate rather "
            "than a mechanism verdict. VERDICT (interpretation.label): "
            "selection_hardness_confirmed_full (A2 clears BOTH 194a thresholds, "
            "A1 does not -> mechanism PRESENT but hard-Gumbel-suppressed); "
            "selection_hardness_partial_recovery (A2 recovers toward the "
            "thresholds vs A1 by >RECOVERY_MARGIN on both, without full pass); "
            "mechanism_absence_indicated (A2 also flat -> NOT selection-specific); "
            "both_selections_recover_a1_not_reproduced / "
            "inverted_hard_better_than_soft (confound / unexpected -> re-examine). "
            "A0_OFF is the MECH-150 C2 saddle control + reproduces 922's closest-"
            "to-threshold arm. MECH-151 action_bias_div is recorded per arm as "
            "informative context (feeds ARC-041's HELD dual-pathway disposition) "
            "but is NOT a scored claim here. This run RESOLVES MECH-152's "
            "pending_retest_after_substrate flag and INFORMS ARC-041's HELD "
            "disposition; it does NOT finalize either -- that is /governance's "
            "call once this ablation lands (2026-08-13 Step 8 gate held ARC-041)."
        ),
        "interpretation": {
            "label": label,
            "requeue_reason": requeue_reason,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": criteria_non_degenerate,
            "cell_readiness": cell_readiness,
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
            "recovery_margin":                 RECOVERY_MARGIN,
            "hazard_std_floor":                HAZARD_STD_FLOOR,
            "terrain_weight_std_floor":        TERRAIN_WEIGHT_STD_FLOOR,
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
