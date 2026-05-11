#!/opt/local/bin/python3
"""V3-EXQ-543c: ARC-062 Phase 3 Monomodal-Collapse Falsifier on bipartite SD-054.

Supersedes V3-EXQ-543b. Per arc_062_rule_apprehension_plan.md decision-log entry
2026-05-11 (diagnose-errors session diagnose-v3-exq-543c-2026-05-11T0635Z plus
SD-054 bipartite-layout extension landed same day in ree-v3 commit 73fdd7e).

Why this iteration was needed
-----------------------------
V3-EXQ-543b ran 2026-05-10 (240 min on Mac) and registered `non_contributory`
on all 3 seeds with the `inert_gating_detected` short-circuit firing at
MID_TRAINING_EP=30. The behavioural-divergence probe recorded
`mean_tv_distance = max_tv_distance = min_tv_distance = 0.0` (exact float
zero) across 3 seeds x 12 probe windows x 32 probe states. Exact zero
across that variation is not numerical noise -- it is structural.

The 2026-05-11 diagnose-errors session traced the failure to TWO distinct
causes:

  Cause 1 (script-level bug): probe-buffer construction in
    `_p0_warmup_train` used `candidates[i].world_states[0]` to build
    `candidate_features`. By E2FastPredictor convention
    (ree_core/predictors/e2_fast.py:316), `world_states[0] =
    initial_z_world` -- the PRE-action z_world, IDENTICAL across all K
    candidates of a single CEM proposal step. So `candidate_features` was
    a [K, world_dim] tensor with K copies of the same row -> heads
    emitted identical scalars across K -> `gated_score_bias` constant in
    K -> `softmax(-constant_vector)` uniform -> TV vs bypass-uniform =
    exactly 0 by arithmetic. The diversification loss
    `(head_0_bias - head_1_bias)^2.mean()` was non-zero only because of
    the symmetry-broken init at gated_policy.py:217 (head_0.bias=+0.05,
    head_1.bias=-0.05); gradients flowed only into the heads' last-Linear
    bias terms (clamped at +/-0.1 by bias_scale), never into feature-
    conditional weights -- so the heads could never learn per-state
    specialisation regardless of episode count.

  Cause 2 (substrate-level finding): even with the Cause-1 fix applied
    (smoke-tested 2026-05-11 in the diagnose-errors session), direct
    numerical probe confirmed: pairwise max-diff across 8 candidates'
    world_states[1] is only ~1.2e-5 (growing to ~2.8e-3 by t=30) because
    the CEM proposer at init produces 8 candidates all sharing
    argmax-first-action=3 with continuous-action vectors differing only
    ~1e-4. E2 world_forward is a small-residual model, so post-action
    z_world spread stays microscopic. The ARC-062 head reads z_world-only
    inputs that are structurally near-indistinguishable -- a real
    substrate-design gap that the script-level fix alone cannot bridge.

V3-EXQ-543c corrections
-----------------------
Two changes from V3-EXQ-543b (all other 543b config preserved verbatim --
P0=40 / P1=60 / P2=8 phased training; gated_policy in dedicated Adam at
LR_GATED_POLICY=5e-4; hardened C3 with C3_TRANSIT_RATE_FLOOR=0.05 +
np.nanmean + min 2 valid seeds; behavioral-divergence probe with
INERT_GATING_THRESHOLD=0.05 mid-P1 short-circuit at MID_TRAINING_EP=30):

  CORRECTION 1 (Cause-1 fix): probe-buffer construction now stacks
    `candidates[i].world_states[1]` (first POST-action predicted z_world
    per candidate, produced by `e2.world_forward(z_world, action_first)`)
    rather than `world_states[0]`. The probe-collection guard is
    tightened from `len(world_states) >= 1` to `>= 2` so the new index is
    always safe; in practice the CEM proposer's trajectory horizon is
    well above 1, so the guard is defensive only. The two downstream
    consumers (`_compute_scaffolding_loss` in P1 training and
    `_run_behavioral_divergence_probe` in mid-P1 measurement) read
    `snap["candidate_features"]` from the buffer unchanged -- they
    automatically receive the corrected post-action z_world per candidate.

  CORRECTION 2 (Cause-2 substrate fix): ENV_KWARGS now sets
    `reef_bipartite_layout=True`, `reef_bipartite_axis="horizontal"`,
    `reef_bipartite_agent_band_radius=1`. This activates the SD-054
    bipartite-layout extension landed 2026-05-11 (ree-v3 commit 73fdd7e;
    REE_assembly commit 5b96cd3e33). Reef cells are now clustered in
    the bottom half of the grid (rows > midline + radius), food / hazards
    spawn exclusively in the top half (rows < midline - radius), and the
    agent always spawns in the midline band (rows midline +/- 1).
    Reef-approach and forage-approach trajectories therefore have
    categorically opposite first-action argmaxes on the row axis by
    construction (action 1 = down toward reef, action 0 = up toward
    food). V3-EXQ-548 substrate-readiness PASS confirmed structural
    divergence uplift 0.633 -> 0.807 (1.27x over legacy SD-054).

Honest scope note
-----------------
Two changes (script-level world_states[1] + substrate-level bipartite
layout) are applied jointly because they were both diagnosed in the same
2026-05-11 session and the substrate-level bipartite is a strict
prerequisite for testing whether the script-level fix is sufficient.
Running the Cause-1 fix alone would reproduce the 543b inert-gating
result (verified: direct numerical probe under legacy SD-054 with
world_states[1] gave TV ~9e-8, still well below the 0.05 threshold).

Three possible outcomes for V3-EXQ-543c:
  - **PASS** (probe-gate PASS + >=2 of C2/C3/C4 + no F1/F2): ARC-062
    substrate is testable AND the gated-policy heads produce per-
    candidate bimodality given the structural opportunity SD-054
    bipartite layout creates. Closes arc_062 GAP-B. Routes to GAP-C/D
    Phase 3 wiring (discriminator -> SD-033a LateralPFCAnalog
    rule_state + bias-head into E3 optimizer; commitment_closure GAP-1
    unblocks).
  - **FAIL with probe-gate FAIL (inert gating STILL detected)**:
    column-axis ties in the bipartite layout dilute the row-axis signal
    too much for the ARC-062 head reading z_world-only inputs. Routes
    to option 2 from the 2026-05-11 decision-log: augment GatedPolicy
    head input with first-action one-hot. This is a substrate change to
    ARC-062 (modifies the contract registered in CLAUDE.md), so belongs
    under /implement-substrate rather than /diagnose-errors.
  - **FAIL with probe-gate PASS but C2/C3/C4 criteria violated** OR
    **F1 monomodal-collapse signature**: substrate works, gating
    produces non-trivial discrimination, but the trained policy still
    collapses to monomodal -- this IS the MECH-309 substrate-level
    monomodal-collapse signature the falsifier is designed to detect.
    Strong evidence for MECH-309. Routes to ARC-063 V4 strong-reading
    design.

V3-EXQ-543b corrections (preserved here for audit)
--------------------------------------------------
V3-EXQ-543 declared PASS but was reclassified `non_contributory` for all three
tagged claims (MECH-309, ARC-062, SD-029) on review. Three independent issues:

  (1) C3 risk-type dissociation was a divide-by-near-zero artifact: ARM_1c
      seed 0 had `transit_hazard_rate = 0.0` (zero transit-regime hazards),
      so `risk_type_ratio = forage_rate / ~0 = 74883`. The arm-mean (24961)
      was dominated by that single seed. Two of three seeds were *below*
      ARM_0's mean of 0.082. C3 measured degenerate-trajectory geometry,
      not behavioral dissociation.
  (2) ARM_1c seed 2 produced byte-identical metrics to ARM_0 seed 2: with
      gated_policy parameters intentionally NOT in any optimizer (Phase 2a
      tested structural-sufficiency-at-init) and disc_init_scale=0.1 keeping
      the sigmoid output near 0.5, the gating layer was inert for that seed
      -- the same RNG stream produced identical trajectories. The "random
      init breaks symmetry" assumption was too weak.
  (3) C2 state-dependence -- the criterion most directly tied to MECH-309's
      rule-apprehension prediction -- FAILED: ARM_1c mean_abs_rho=0.111 vs
      ARM_0=0.291 (gated arm shows LESS state-dependent reef behavior, not
      more). With untrained gating + sigmoid-near-0.5 init, the only two
      outcomes available were inert clones of baseline (seed 2) or random-
      init noise disrupting baseline (seeds 0/1). Neither falsifies MECH-309.

V3-EXQ-543b corrections (Phase 3 design)
----------------------------------------
Four changes from V3-EXQ-543, each addressing one of the issues above:

  CORRECTION A (gated_policy params in an optimizer, fixes issue 2):
      Phase 1 of arc_062 plan landed gated_policy as an nn.Module with
      symmetry-broken init, but its parameters were NOT in any optimizer.
      Untrained, they remained near init for the entire run, producing
      bit-identical-to-baseline behavior on any seed where the random init
      noise happened to cancel. V3-EXQ-543b adds gated_policy.parameters()
      to a dedicated Adam optimizer in P1 (LR_GATED_POLICY=5e-4) so the
      heads + discriminator move under non-trivial gradient pressure.

  CORRECTION B (phased training schedule, supports CORRECTION A):
      Per CLAUDE.md Phased Training Protocol: any experiment training a
      head on z_world / z_harm / encoder output requires phased training to
      avoid moving-target collapse. P0 warmup (40 episodes; encoder + E1 +
      E2_world_forward + E3 harm_eval; gated_policy params stay at init).
      P1 training (60 episodes; encoder frozen via .requires_grad_(False);
      gated_policy params now in optimizer; scaffolding loss creates non-
      trivial gradient pressure -- see CORRECTION A note below). P2 eval
      (8 episodes per arm; gated_policy.eval()).

      P1 training-pressure honest scope. The scaffolding loss is a
      diversification regularizer:
          L_scaff = -lambda_div * mean(||h0(features) - h1(features)||^2)
                  - lambda_disc * variance(w over recent batch)
      This is NOT rigorous REINFORCE on environmental reward. It is
      "scaffolding pressure" that ensures gated_policy parameters MOVE
      under any non-trivial gradient. The architecturally important
      question this experiment can falsify is *substrate-level*: when the
      parameters are not pinned to init, can the substrate produce
      context-conditional behavioral divergence on SD-054 at all? If yes,
      the C2/C3/C4 acceptance grid applies. If no, the substrate is
      unable to escape monomodal collapse even when its parameters are
      free to move -- which is the MECH-309 prediction at the substrate
      level. Full REINFORCE-based training (where gated_policy is trained
      jointly with E3 via score-aggregation gradient through environmental
      reward) is deferred to Phase 3 GAP-C/GAP-D when the discriminator is
      wired into SD-033a LateralPFCAnalog and the bias head joins the
      composite E3 optimizer (commitment_closure GAP-1).

  CORRECTION C (behavioral-divergence probe with mid-training inert-gating
                short-circuit, fixes issue 2 detection latency):
      A static probe set of (z_world, z_self, z_harm_a, candidate_features)
      tuples is sampled once at the end of P0 from diverse buffer states.
      Every PROBE_INTERVAL_P1_EPS=5 P1 episodes, the script computes the
      mean per-candidate softmax-distribution divergence between
      use_gated_policy=True (with the current gated_policy bias) and the
      bypass-baseline (zero bias). Mean total-variation distance below
      INERT_GATING_THRESHOLD=0.05 at MID_TRAINING_EP=30 short-circuits
      the run for that seed: per-claim direction set to non_contributory
      with diagnostic note `inert_gating_detected_during_training`. This
      catches the seed-2 byte-identical pattern within ~25 minutes rather
      than ~50, and routes the result correctly rather than letting
      downstream C2/C3/C4 metrics interpret pure noise.

  CORRECTION D (hardened C3 calculation, fixes issue 1):
      Per-seed `risk_type_ratio` is included in the arm-mean only if
      `transit_hazard_rate > C3_TRANSIT_RATE_FLOOR (0.05)`. Seeds below
      the floor contribute `np.nan`; the arm-mean uses `np.nanmean`. The
      manifest reports `n_valid_seeds` per arm so reviewers see the
      filter's bite. ARM_0 vs ARM_1c comparison only fires when both
      arms have at least 2 valid seeds; otherwise C3 is reported as
      `non_contributory_insufficient_transit_events`.

Substrate (matches V3-EXQ-543 baseline; only manipulated variable changes)
--------------------------------------------------------------------------
  SD-007 reafference perspective correction
  SD-008 alpha_world=0.9 encoder correction
  SD-010 use_harm_stream=True sensory-discriminative z_harm_s
  SD-011 use_affective_harm_stream=True + harm_history_len=10 affective z_harm_a
  SD-012 z_goal_enabled=True + drive_weight=2.0 homeostatic drive modulation
  SD-018 use_resource_proximity_head=True resource proximity supervision
  SD-021 harm_descending_mod_enabled=True descending pain modulation
  SD-054 reef_enabled=True + hazard_food_attraction=0.7 + n_reef_patches=3
  MECH-090 beta_gate_bistable=True bistable commitment latch
  ARC-062 use_gated_policy=True (ARM_1c only); OFF in ARM_0
  Phase 3 additions:
    P1 freezes encoder via latent_stack.parameters().requires_grad_(False)
    Dedicated Adam over gated_policy.parameters() with LR_GATED_POLICY=5e-4
    Scaffolding diversification loss in P1 only

Acceptance criteria (per arc_062_rule_apprehension_plan.md decision-log
2026-05-10, gated by behavioral-divergence probe per CORRECTION C). PASS rule:
>=2 of {C2, C3, C4} hold across seeds with no contradictory signal AND probe
gate passes (mean per-seed mid-P1 mismatch >= 0.05). C1 is NOT MEASURABLE in
this single-density Phase 2a-corrected run (deferred to Phase 2b).

  Probe gate: mean total-variation distance between gated and bypass action
     distributions on the probe set, evaluated at P1 episode 30 (mid-training).
     Mismatch < 0.05 in ANY ARM_1c seed -> that seed marked
     non_contributory_inert_gating; arm-level non_contributory if >=2/3 seeds
     marked.

  C2 state-dependence (Balaban-Feld 2019): same as 543. PASS = ARM_1c
     n_rho_above_threshold >= 2 AND ARM_1c mean_abs_rho > ARM_0 mean_abs_rho.

  C3 risk-type dissociation (Eccard 2020) HARDENED per CORRECTION D: PASS =
     |ARM_1c mean_risk_ratio - ARM_0 mean_risk_ratio| / max(|ARM_0|, eps) >= 0.5
     AND both arms have n_valid_seeds >= 2 (where valid = transit_rate > 0.05).
     If either arm has n_valid_seeds < 2, C3 reports
     non_contributory_insufficient_transit_events.

  C4 cross-seed variation (Eccard 2020 + Crowell 2016): same as 543. PASS =
     ARM_1c CoV >= 0.10.

Unambiguous FAIL signatures (any one is unambiguous; flips per-claim
direction to weakens regardless of C2/C3/C4 PASS counts):
  F1 total invariance (gated_policy free-to-move STILL produces no behavioral
     divergence): all 4 of (reef_diff, both arms abs_rho, C3 hardened delta,
     both arms CoV) below their F1 thresholds AND the probe gate passed
     (rules out the "inert gating" alternative explanation). This is the
     unambiguous monomodal-collapse signature MECH-309 predicts at the
     substrate level: even with parameter freedom, the substrate cannot
     produce relative-risk-pattern-dependent policy.
  F2 biologically inverted: ARM_1c rho_drive_vs_reef monotonically positive
     (Spearman > +0.4); naive "always-flee-when-hazard-present" rather than
     state-dependent.

Tagging
-------
  claim_ids: [ARC-062, MECH-309]
    SD-029 dropped from V3-EXQ-543b's tag list per the claim_ids accuracy rule:
    this experiment runs on the SD-054 substrate (now with bipartite layout
    on), not SD-029's scheduled_external_hazard substrate. SD-029's C2 / C3
    monomodal-balance measurement is downstream of MECH-269 V_s monostrategy
    landing per WORKSPACE_STATE.md, not this experiment's question.
  evidence_direction_per_claim:
    ARC-062 (V3 weak-reading architectural validation; primary):
      "supports" if PASS (probe gate + >=2 of C2/C3/C4 + no F1/F2)
      "weakens" if F1 or F2
      "non_contributory" if probe gate failed OR PASS rule unmet without F1/F2
    MECH-309 (logical-necessity falsifier; primary):
      "supports" if PASS
      "weakens" if F1 (substrate cannot escape monomodal even when free) or F2
      "non_contributory" if probe gate failed (inert gating) OR
        PASS rule unmet but no F1/F2 (mixed signal)
  experiment_purpose: "evidence" (architecturally load-bearing falsifier)
  supersedes: V3-EXQ-543b

Plan-of-record: REE_assembly/evidence/planning/arc_062_rule_apprehension_plan.md
GAP-B owner_exq -> V3-EXQ-543c on PASS; on FAIL routes through the three-
outcome branch table above (probe-gate FAIL -> option 2 substrate change to
GatedPolicy head input; probe-gate PASS + criteria-FAIL or F1 -> MECH-309
substrate-level monomodal-collapse evidence -> ARC-063 V4 design).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_543b_arc062_phase3_optimized_falsifier.py [--dry-run]

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome


EXPERIMENT_TYPE = "v3_exq_543c_arc062_phase3_bipartite_falsifier"
EXPERIMENT_PURPOSE = "evidence"
QUEUE_ID = "V3-EXQ-543c"
SUPERSEDES = "V3-EXQ-543b"
CLAIM_IDS = ["ARC-062", "MECH-309"]

# Env: ARM_1_med density (matches V3-EXQ-543 + V3-EXQ-522 baseline).
# V3-EXQ-543c CORRECTION 2: SD-054 bipartite-layout extension enabled
# (reef_bipartite_layout=True + horizontal axis + agent_band_radius=1).
# Validated structurally by V3-EXQ-548 substrate-readiness PASS 2026-05-11
# (ARM_0 div=0.633 -> ARM_1 div=0.807, 1.27x uplift). With this on, reef
# cells cluster in rows > midline+radius (bottom half); hazards / food
# spawn exclusively in rows < midline-radius (top half); agent spawns in
# the midline band [midline-radius, midline+radius]. Reef-approach and
# forage-approach trajectories have categorically opposite first-action
# argmaxes on the row axis (action 1 down toward reef vs action 0 up
# toward food). This creates the structural condition required for the
# ARC-062 head -- reading z_world-only inputs -- to potentially produce
# per-candidate first-action argmax diversity at probe states.
ENV_KWARGS = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    # SD-054 bipartite-layout extension (V3-EXQ-543c CORRECTION 2).
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)

# Phased training schedule.
P0_WARMUP_EPISODES = 40
P1_TRAIN_EPISODES = 60
P2_EVAL_EPISODES = 8
STEPS_PER_EPISODE = 200

# Latent dims (match V3-EXQ-543 / 522 baseline).
WORLD_DIM = 32
SELF_DIM = 32
HARM_DIM = 32
HARM_A_DIM = 16
HARM_HISTORY_LEN = 10

# Buffer + batch.
WF_BUF_MAX = 2000
HARM_EVAL_BUF_MAX = 2000
PROBE_BUF_MAX = 256
BATCH_SIZE = 32

# Learning rates (match 543).
LR_E1 = 1e-4
LR_E2_WF = 3e-4
LR_E3_HARM = 1e-3
LR_ENC_AUX = 5e-4
# CORRECTION A: dedicated optimizer over gated_policy.parameters() in P1.
LR_GATED_POLICY = 5e-4

# Scaffolding diversification loss weights (P1 only, ARM_1c only).
LAMBDA_HEAD_DIV = 1.0
LAMBDA_DISC_VAR = 0.5

# CORRECTION C: behavioral-divergence probe.
PROBE_INTERVAL_P1_EPS = 5
MID_TRAINING_EP = 30  # P1 episode index at which inert-gating short-circuit fires
INERT_GATING_THRESHOLD = 0.05  # mean per-state TV-distance threshold
N_PROBE_STATES = 32
N_PROBE_CANDIDATES = 8
SOFTMAX_TEMPERATURE_PROBE = 1.0

# Drive-bin breakpoints for C2 state-dependence Spearman (3 bins).
DRIVE_BINS = (0.33, 0.67)

# Acceptance thresholds (pre-registered, match 543 except CORRECTION D).
C2_RHO_THRESHOLD = 0.20
C2_MIN_PASS_SEEDS = 2
C3_RELATIVE_DELTA_THRESHOLD = 0.50
C4_COV_THRESHOLD = 0.10
F1_REEF_DIFF_THRESHOLD = 0.02
F1_C2_RHO_THRESHOLD = 0.05
F1_C3_DELTA_THRESHOLD = 0.05
F1_C4_COV_THRESHOLD = 0.02
F2_INVERTED_RHO_THRESHOLD = 0.40

# CORRECTION D: hardened C3.
C3_TRANSIT_RATE_FLOOR = 0.05
C3_MIN_VALID_SEEDS_PER_ARM = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _obs_harm(obs_dict):
    return obs_dict.get("harm_obs")


def _obs_harm_a(obs_dict):
    return obs_dict.get("harm_obs_a")


def _obs_harm_history(obs_dict):
    return obs_dict.get("harm_history")


def _obs_accum(obs_dict) -> float:
    v = obs_dict.get("accumulated_harm")
    return float(v) if v is not None else 0.0


def _obs_resource_prox(obs_dict) -> float:
    rv = obs_dict.get("resource_field_view")
    if rv is None:
        return 0.0
    return float(rv.max().item()) if isinstance(rv, torch.Tensor) else float(np.max(rv))


def _spearman_rho(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation. Returns 0.0 on degenerate (constant) input."""
    if len(x) < 4 or len(y) < 4:
        return 0.0
    rx = np.argsort(np.argsort(np.asarray(x, dtype=np.float64)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=np.float64)))
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _make_agent_and_env(seed: int, use_gated_policy: bool) -> Tuple[REEAgent, CausalGridWorldV2]:
    """Build agent + env. Manipulated variable: use_gated_policy (matches 543)."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        harm_dim=HARM_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        use_harm_stream=True,
        z_harm_dim=HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        # ARC-062 manipulated variable.
        use_gated_policy=use_gated_policy,
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    config.harm_descending_mod_enabled = True
    config.descending_attenuation_factor = 0.5

    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Phase P0: encoder warmup (gated_policy params stay at init)
# ---------------------------------------------------------------------------

def _p0_warmup_train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    total_train_episodes: int,
    arm_label: str,
    probe_buf: List[Dict],
) -> Dict:
    """Phase P0: standard 543 training; gated_policy params NOT in any optimizer.

    Also collects probe-state snapshots for the CORRECTION C probe in P1.
    """
    device = agent.device
    action_dim = env.action_dim

    e1_optimizer = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    e2_wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    harm_eval_optimizer = optim.Adam(
        agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM,
    )
    aux_params = list(agent.latent_stack.parameters())
    aux_optimizer = optim.Adam(aux_params, lr=LR_ENC_AUX)

    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_eval_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []
    reward_log: List[float] = []

    # Probe snapshot frequency: roughly N_PROBE_STATES samples spread across P0.
    probe_snapshot_every = max(1, (num_episodes * steps_per_episode) // (N_PROBE_STATES * 3))
    probe_step_counter = 0

    agent.train()

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        ep_reward = 0.0

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h = _obs_harm(obs_dict)
            obs_h_a = _obs_harm_a(obs_dict)
            obs_h_h = _obs_harm_history(obs_dict)
            prox_t = _obs_resource_prox(obs_dict)
            accum_t = _obs_accum(obs_dict)

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
            )
            z_world_curr = latent.z_world.detach()

            aux_terms: List[torch.Tensor] = []
            prox_target_t = torch.tensor([[prox_t]], device=device)
            prox_loss = agent.compute_resource_proximity_loss(prox_target_t, latent)
            if prox_loss is not None and prox_loss.requires_grad:
                aux_terms.append(prox_loss)
            accum_target_t = torch.tensor([[accum_t]], device=device)
            harm_accum_loss = agent.compute_harm_accum_loss(accum_target_t, latent)
            if harm_accum_loss is not None and harm_accum_loss.requires_grad:
                aux_terms.append(harm_accum_loss)
            if aux_terms:
                aux_loss = sum(aux_terms)
                aux_optimizer.zero_grad()
                aux_loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(aux_params, 1.0)
                aux_optimizer.step()

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
            )

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            drive_level = REEAgent.compute_drive_level(obs_body)
            benefit_exposure = max(0.0, float(obs_dict.get("benefit_exposure", 0.0)))
            agent.update_z_goal(
                benefit_exposure=benefit_exposure,
                drive_level=drive_level,
            )

            # CORRECTION C: probe-state snapshot (cache pre-action latents +
            # candidate features; only when ARM_1c so probe set is built from
            # gated arm state distribution, but probe inputs are arm-agnostic).
            # candidates is list[Trajectory]; each Trajectory.world_states is
            # list[Tensor[1, world_dim]]. First-step summary stacks
            # candidate.world_states[1] (POST first-action predicted z_world)
            # across the first N candidates. V3-EXQ-543c CORRECTION 1: switched
            # from world_states[0] (= initial_z_world, identical across all K
            # by E2FastPredictor convention) to world_states[1] so per-candidate
            # features are distinguishable across K. world_states[1] differs
            # per candidate because each candidate's first action differs;
            # E2.world_forward propagates the action-conditioned next state
            # via the trained residual model. Guard tightened from >= 1 to
            # >= 2 so the new index is always safe.
            if (agent.gated_policy is not None
                    and probe_step_counter % probe_snapshot_every == 0
                    and len(probe_buf) < PROBE_BUF_MAX
                    and isinstance(candidates, list)
                    and len(candidates) >= N_PROBE_CANDIDATES
                    and getattr(candidates[0], "world_states", None) is not None
                    and len(candidates[0].world_states) >= 2):
                first_step_world = torch.cat([
                    c.world_states[1].detach().clone()
                    for c in candidates[:N_PROBE_CANDIDATES]
                ], dim=0)
                probe_buf.append({
                    "z_world": latent.z_world.detach().clone(),
                    "z_self": latent.z_self.detach().clone(),
                    "z_harm_a": (
                        latent.z_harm_a.detach().clone()
                        if latent.z_harm_a is not None else None
                    ),
                    "candidate_features": first_step_world,
                })
            probe_step_counter += 1

            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, action_dim - 1), action_dim, device,
                )
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ep_reward += float(harm_signal)

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > WF_BUF_MAX:
                    wf_buf = wf_buf[-WF_BUF_MAX:]

            harm_target = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0
            harm_eval_buf.append((z_world_curr.cpu(), torch.tensor([harm_target])))
            if len(harm_eval_buf) > HARM_EVAL_BUF_MAX:
                harm_eval_buf = harm_eval_buf[-HARM_EVAL_BUF_MAX:]

            if len(wf_buf) >= BATCH_SIZE:
                idxs = torch.randperm(len(wf_buf))[:BATCH_SIZE].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    e2_wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters())
                        + list(agent.e2.world_action_encoder.parameters()), 1.0,
                    )
                    e2_wf_optimizer.step()
                with torch.no_grad():
                    agent.e3.update_running_variance(
                        (wf_pred.detach() - zw1_b).detach()
                    )

            if len(harm_eval_buf) >= BATCH_SIZE:
                idxs = torch.randperm(len(harm_eval_buf))[:BATCH_SIZE].tolist()
                zw_b = torch.cat([harm_eval_buf[i][0] for i in idxs]).to(device)
                ht_b = torch.cat([harm_eval_buf[i][1] for i in idxs]).to(device)
                hp = agent.e3.harm_eval(zw_b)
                he_loss = F.mse_loss(hp.squeeze(), ht_b.squeeze())
                if he_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    he_loss.backward()
                    harm_eval_optimizer.step()

            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_optimizer.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                    e1_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

        reward_log.append(ep_reward)

        if (ep + 1) % 10 == 0 or ep == num_episodes - 1:
            print(
                f"  [train] {arm_label} ep {ep+1}/{total_train_episodes}"
                f"  phase=P0  rv={agent.e3._running_variance:.4f}"
                f"  ep_reward={ep_reward:.4f}",
                flush=True,
            )

    return {
        "p0_final_running_variance": agent.e3._running_variance,
        "p0_first10_reward": float(np.mean(reward_log[:10])) if len(reward_log) >= 10 else float(np.mean(reward_log)),
        "p0_last10_reward": float(np.mean(reward_log[-10:])) if len(reward_log) >= 10 else float(np.mean(reward_log)),
        "p0_n_probe_states_collected": len(probe_buf),
    }


# ---------------------------------------------------------------------------
# Phase P1: training under scaffolding pressure with optional gated_policy
# ---------------------------------------------------------------------------

def _compute_scaffolding_loss(
    agent: REEAgent,
    probe_buf: List[Dict],
    n_samples: int,
) -> Tuple[torch.Tensor, float, float]:
    """Diversification regularizer over a minibatch from probe_buf.

    Returns (loss_tensor, head_div_value, disc_var_value).
    """
    if agent.gated_policy is None or len(probe_buf) < 2:
        zero = torch.zeros(1, device=agent.device, requires_grad=False)
        return zero, 0.0, 0.0

    n = min(n_samples, len(probe_buf))
    idxs = np.random.choice(len(probe_buf), size=n, replace=False)

    head_div_terms: List[torch.Tensor] = []
    disc_w_values: List[torch.Tensor] = []
    for i in idxs:
        snap = probe_buf[int(i)]
        out = agent.gated_policy.forward(
            z_world=snap["z_world"],
            z_self=snap["z_self"],
            z_harm_a=snap.get("z_harm_a"),
            candidate_features=snap["candidate_features"],
            simulation_mode=False,
        )
        # Head divergence: L2 between raw head outputs across candidates.
        diff = (out.head_0_bias - out.head_1_bias).flatten()
        head_div_terms.append((diff * diff).mean())
        # Build discriminator scalar with grad: re-run discriminator forward
        # via the cached out; gating_weight is a float, but we need a Tensor.
        # Use head_0/head_1 difference statistics through w as a proxy:
        # variance(w across batch) is computed below via the scalar w detached
        # from the float; since we need grad, capture the raw w_tensor by
        # rebuilding it from the discriminator path.
        # Simpler approach: read disc_input again and run discriminator.
        zw = snap["z_world"]
        zs = snap["z_self"]
        za = snap.get("z_harm_a")
        if za is None:
            za = torch.zeros(
                zw.shape[0] if zw.dim() == 2 else 1,
                agent.gated_policy.harm_a_dim,
                device=agent.device,
            )
        zw2 = zw if zw.dim() == 2 else zw.unsqueeze(0)
        zs2 = zs if zs.dim() == 2 else zs.unsqueeze(0)
        za2 = za if za.dim() == 2 else za.unsqueeze(0)
        disc_input = torch.cat(
            [zw2.mean(dim=0, keepdim=True),
             zs2.mean(dim=0, keepdim=True),
             za2.mean(dim=0, keepdim=True)],
            dim=-1,
        )
        w_tensor = agent.gated_policy.discriminator(disc_input).squeeze()
        disc_w_values.append(w_tensor)

    head_div_term = torch.stack(head_div_terms).mean()
    disc_w_stack = torch.stack(disc_w_values)
    disc_var_term = disc_w_stack.var(unbiased=False)

    # We MAXIMIZE head_div and disc_var, so loss = NEGATIVE of those.
    loss = -(LAMBDA_HEAD_DIV * head_div_term + LAMBDA_DISC_VAR * disc_var_term)
    return loss, float(head_div_term.detach().item()), float(disc_var_term.detach().item())


def _run_behavioral_divergence_probe(
    agent: REEAgent,
    probe_buf: List[Dict],
) -> Dict:
    """CORRECTION C: per-state TV-distance between gated and bypass policies.

    For each probe state, computes softmax(-(zero_bias)/T) and softmax(-(gated_bias)/T)
    over candidate scores. Reports mean total-variation distance and per-claim probe
    pass/fail. ARM_0 (no gated_policy) returns N/A.
    """
    if agent.gated_policy is None or len(probe_buf) == 0:
        return {
            "n_probe_states": 0,
            "mean_tv_distance": 0.0,
            "max_tv_distance": 0.0,
            "min_tv_distance": 0.0,
            "applicable": False,
        }
    tv_distances: List[float] = []
    with torch.no_grad():
        for snap in probe_buf[:N_PROBE_STATES]:
            out = agent.gated_policy.forward(
                z_world=snap["z_world"],
                z_self=snap["z_self"],
                z_harm_a=snap.get("z_harm_a"),
                candidate_features=snap["candidate_features"],
                simulation_mode=False,
            )
            gated_bias = out.gated_score_bias  # [K]
            T = SOFTMAX_TEMPERATURE_PROBE
            # Use bias alone as the differentiating signal: pi_gated and pi_bypass
            # treat raw scores as identical baseline; the marginal contribution of
            # gated_policy is exactly its bias term.
            pi_gated = F.softmax(-gated_bias / T, dim=0)
            pi_bypass = F.softmax(torch.zeros_like(gated_bias) / T, dim=0)
            tv = 0.5 * (pi_gated - pi_bypass).abs().sum().item()
            tv_distances.append(tv)
    return {
        "n_probe_states": len(tv_distances),
        "mean_tv_distance": float(np.mean(tv_distances)) if tv_distances else 0.0,
        "max_tv_distance": float(np.max(tv_distances)) if tv_distances else 0.0,
        "min_tv_distance": float(np.min(tv_distances)) if tv_distances else 0.0,
        "applicable": True,
    }


def _p1_train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    total_train_episodes: int,
    p0_episodes: int,
    arm_label: str,
    probe_buf: List[Dict],
    use_gated_policy: bool,
    dry_run: bool,
) -> Dict:
    """Phase P1: encoder frozen; gated_policy params in dedicated optimizer.

    Continues E2 world_forward and harm_eval training (encoder downstream
    consumers); freezes encoder via latent_stack.parameters().requires_grad_(False).
    Adds gated_policy.parameters() to a dedicated Adam.
    """
    device = agent.device
    action_dim = env.action_dim

    # Freeze encoder (CORRECTION B P1 phased training).
    for p in agent.latent_stack.parameters():
        p.requires_grad_(False)

    e2_wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    harm_eval_optimizer = optim.Adam(
        agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM,
    )

    # CORRECTION A: gated_policy params in a dedicated Adam (ARM_1c only).
    gated_optimizer: Optional[optim.Optimizer] = None
    if use_gated_policy and agent.gated_policy is not None:
        gated_optimizer = optim.Adam(
            agent.gated_policy.parameters(), lr=LR_GATED_POLICY,
        )

    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_eval_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []
    reward_log: List[float] = []

    # CORRECTION C: probe-gate state.
    probe_log: List[Dict] = []
    inert_gating_detected = False

    agent.train()

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        ep_reward = 0.0

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h = _obs_harm(obs_dict)
            obs_h_a = _obs_harm_a(obs_dict)
            obs_h_h = _obs_harm_history(obs_dict)

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
            )
            z_world_curr = latent.z_world.detach()

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
            )

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            drive_level = REEAgent.compute_drive_level(obs_body)
            benefit_exposure = max(0.0, float(obs_dict.get("benefit_exposure", 0.0)))
            agent.update_z_goal(
                benefit_exposure=benefit_exposure,
                drive_level=drive_level,
            )

            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, action_dim - 1), action_dim, device,
                )
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ep_reward += float(harm_signal)

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > WF_BUF_MAX:
                    wf_buf = wf_buf[-WF_BUF_MAX:]
            harm_target = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0
            harm_eval_buf.append((z_world_curr.cpu(), torch.tensor([harm_target])))
            if len(harm_eval_buf) > HARM_EVAL_BUF_MAX:
                harm_eval_buf = harm_eval_buf[-HARM_EVAL_BUF_MAX:]

            if len(wf_buf) >= BATCH_SIZE:
                idxs = torch.randperm(len(wf_buf))[:BATCH_SIZE].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    e2_wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters())
                        + list(agent.e2.world_action_encoder.parameters()), 1.0,
                    )
                    e2_wf_optimizer.step()
                with torch.no_grad():
                    agent.e3.update_running_variance(
                        (wf_pred.detach() - zw1_b).detach()
                    )

            if len(harm_eval_buf) >= BATCH_SIZE:
                idxs = torch.randperm(len(harm_eval_buf))[:BATCH_SIZE].tolist()
                zw_b = torch.cat([harm_eval_buf[i][0] for i in idxs]).to(device)
                ht_b = torch.cat([harm_eval_buf[i][1] for i in idxs]).to(device)
                hp = agent.e3.harm_eval(zw_b)
                he_loss = F.mse_loss(hp.squeeze(), ht_b.squeeze())
                if he_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    he_loss.backward()
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

        reward_log.append(ep_reward)

        # End-of-episode: scaffolding gradient pressure on gated_policy (ARM_1c only).
        if gated_optimizer is not None and len(probe_buf) >= 2:
            n_steps_per_ep = 4 if not dry_run else 1
            for _ in range(n_steps_per_ep):
                loss, head_div_val, disc_var_val = _compute_scaffolding_loss(
                    agent, probe_buf, n_samples=min(BATCH_SIZE, len(probe_buf)),
                )
                if loss.requires_grad:
                    gated_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.gated_policy.parameters(), 1.0,
                    )
                    gated_optimizer.step()

        # CORRECTION C: behavioral-divergence probe every PROBE_INTERVAL_P1_EPS.
        if (ep + 1) % PROBE_INTERVAL_P1_EPS == 0 or ep == num_episodes - 1:
            probe = _run_behavioral_divergence_probe(agent, probe_buf)
            probe["p1_ep"] = ep + 1
            probe_log.append(probe)
            print(
                f"  [probe] {arm_label} P1 ep {ep+1}/{num_episodes}"
                f"  applicable={probe['applicable']}"
                f"  mean_tv={probe['mean_tv_distance']:.4f}"
                f"  max_tv={probe['max_tv_distance']:.4f}",
                flush=True,
            )
            # Mid-training inert-gating short-circuit.
            if (probe["applicable"] and (ep + 1) >= MID_TRAINING_EP
                    and probe["mean_tv_distance"] < INERT_GATING_THRESHOLD
                    and not inert_gating_detected):
                inert_gating_detected = True
                print(
                    f"  [probe] {arm_label} INERT-GATING SHORT-CIRCUIT at "
                    f"P1 ep {ep+1}: mean_tv={probe['mean_tv_distance']:.4f} "
                    f"< {INERT_GATING_THRESHOLD}",
                    flush=True,
                )

        if (ep + 1) % 10 == 0 or ep == num_episodes - 1:
            cur_total_ep = p0_episodes + ep + 1
            print(
                f"  [train] {arm_label} ep {cur_total_ep}/{total_train_episodes}"
                f"  phase=P1  rv={agent.e3._running_variance:.4f}"
                f"  ep_reward={ep_reward:.4f}"
                f"  inert_gating={inert_gating_detected}",
                flush=True,
            )

    return {
        "p1_first10_reward": float(np.mean(reward_log[:10])) if len(reward_log) >= 10 else float(np.mean(reward_log)),
        "p1_last10_reward": float(np.mean(reward_log[-10:])) if len(reward_log) >= 10 else float(np.mean(reward_log)),
        "p1_inert_gating_detected": bool(inert_gating_detected),
        "p1_probe_log": probe_log,
        "p1_final_running_variance": agent.e3._running_variance,
    }


# ---------------------------------------------------------------------------
# Phase P2: eval (matches 543's _eval_collect_metrics)
# ---------------------------------------------------------------------------

def _p2_eval_collect_metrics(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    total_train_episodes: int,
    arm_label: str,
) -> Dict:
    """Phase P2: eval with behavioural metric collection (matches 543)."""
    device = agent.device
    action_dim = env.action_dim
    agent.eval()

    per_episode_reef_fractions: List[float] = []
    per_episode_drives: List[List[float]] = []
    per_episode_in_reef: List[List[bool]] = []
    forage_hazard_events = 0
    forage_total_steps = 0
    transit_hazard_events = 0
    transit_total_steps = 0

    for ep_idx in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        reef_cells_set = getattr(env, "_reef_cells", set())
        prev_in_reef = False
        ep_drive_log: List[float] = []
        ep_in_reef_log: List[bool] = []

        for step_idx in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h = _obs_harm(obs_dict)
            obs_h_a = _obs_harm_a(obs_dict)
            obs_h_h = _obs_harm_history(obs_dict)

            with torch.no_grad():
                latent = agent.sense(
                    obs_body, obs_world,
                    obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
                )
                if z_self_prev is not None and action_prev is not None:
                    agent.record_transition(
                        z_self_prev, action_prev, latent.z_self.detach(),
                    )
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, WORLD_DIM, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

                drive_level = REEAgent.compute_drive_level(obs_body)
                benefit_exposure = max(0.0, float(obs_dict.get("benefit_exposure", 0.0)))
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure,
                    drive_level=drive_level,
                )

                action = agent.select_action(candidates, ticks, temperature=1.0)
                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, action_dim - 1), action_dim, device,
                    )
                    agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            agent_pos = (int(env.agent_x), int(env.agent_y))
            in_reef = agent_pos in reef_cells_set
            transition_event = (in_reef != prev_in_reef)
            harm_event = float(harm_signal) < 0

            ep_drive_log.append(float(drive_level))
            ep_in_reef_log.append(bool(in_reef))

            if transition_event:
                transit_total_steps += 1
                if harm_event:
                    transit_hazard_events += 1
            elif not in_reef:
                forage_total_steps += 1
                if harm_event:
                    forage_hazard_events += 1

            prev_in_reef = in_reef
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

        reef_frac = (
            sum(ep_in_reef_log) / max(len(ep_in_reef_log), 1)
        )
        per_episode_reef_fractions.append(reef_frac)
        per_episode_drives.append(ep_drive_log)
        per_episode_in_reef.append(ep_in_reef_log)

        if (ep_idx + 1) % 4 == 0 or ep_idx == num_episodes - 1:
            cur_ep = total_train_episodes - num_episodes + ep_idx + 1
            print(
                f"  [train] {arm_label} ep {cur_ep}/{total_train_episodes}"
                f"  phase=P2  reef_frac={reef_frac:.3f}"
                f"  steps={len(ep_in_reef_log)}",
                flush=True,
            )

    flat_drives: List[float] = []
    flat_in_reef: List[float] = []
    for d_log, r_log in zip(per_episode_drives, per_episode_in_reef):
        flat_drives.extend(d_log)
        flat_in_reef.extend([1.0 if r else 0.0 for r in r_log])
    rho_drive_reef = _spearman_rho(flat_drives, flat_in_reef)

    forage_hazard_rate = forage_hazard_events / max(forage_total_steps, 1)
    transit_hazard_rate = transit_hazard_events / max(transit_total_steps, 1)
    risk_type_ratio = forage_hazard_rate / max(transit_hazard_rate, 1e-6)

    return {
        "per_episode_reef_fractions": per_episode_reef_fractions,
        "mean_reef_fraction": float(np.mean(per_episode_reef_fractions)),
        "rho_drive_vs_reef": float(rho_drive_reef),
        "forage_hazard_rate": float(forage_hazard_rate),
        "transit_hazard_rate": float(transit_hazard_rate),
        "risk_type_ratio": float(risk_type_ratio),
        "n_forage_steps": int(forage_total_steps),
        "n_transit_steps": int(transit_total_steps),
        "n_forage_hazards": int(forage_hazard_events),
        "n_transit_hazards": int(transit_hazard_events),
    }


# ---------------------------------------------------------------------------
# Per-arm/seed run
# ---------------------------------------------------------------------------

def run_arm_seed(
    arm_label: str,
    use_gated_policy: bool,
    seed: int,
    dry_run: bool,
) -> Dict:
    p0_eps = 3 if dry_run else P0_WARMUP_EPISODES
    p1_eps = 4 if dry_run else P1_TRAIN_EPISODES
    p2_eps = 2 if dry_run else P2_EVAL_EPISODES
    steps_per_ep = 30 if dry_run else STEPS_PER_EPISODE
    total_train_eps = p0_eps + p1_eps + p2_eps

    print(f"\nSeed {seed} Condition {arm_label}", flush=True)
    print(
        f"  use_gated_policy={use_gated_policy}"
        f"  P0={p0_eps}  P1={p1_eps}  P2={p2_eps}  steps/ep={steps_per_ep}",
        flush=True,
    )

    agent, env = _make_agent_and_env(seed, use_gated_policy)
    print(
        f"  world_obs_dim={env.world_obs_dim}"
        f"  agent.gated_policy={'on' if agent.gated_policy is not None else 'off'}",
        flush=True,
    )

    probe_buf: List[Dict] = []

    p0_metrics = _p0_warmup_train(
        agent, env, p0_eps, steps_per_ep,
        total_train_episodes=total_train_eps, arm_label=arm_label,
        probe_buf=probe_buf,
    )

    p1_metrics = _p1_train(
        agent, env, p1_eps, steps_per_ep,
        total_train_episodes=total_train_eps,
        p0_episodes=p0_eps,
        arm_label=arm_label,
        probe_buf=probe_buf,
        use_gated_policy=use_gated_policy,
        dry_run=dry_run,
    )

    eval_metrics = _p2_eval_collect_metrics(
        agent, env, p2_eps, steps_per_ep,
        total_train_episodes=total_train_eps,
        arm_label=arm_label,
    )

    seed_summary = {
        "arm_label": arm_label,
        "seed": seed,
        "use_gated_policy": use_gated_policy,
        **p0_metrics,
        **p1_metrics,
        **eval_metrics,
    }

    print(
        f"  seed={seed} arm={arm_label}"
        f"  reef_frac={eval_metrics['mean_reef_fraction']:.3f}"
        f"  rho={eval_metrics['rho_drive_vs_reef']:+.3f}"
        f"  forage_hr={eval_metrics['forage_hazard_rate']:.4f}"
        f"  transit_hr={eval_metrics['transit_hazard_rate']:.4f}"
        f"  ratio={eval_metrics['risk_type_ratio']:.3f}"
        f"  inert_gating={p1_metrics['p1_inert_gating_detected']}",
        flush=True,
    )

    seed_pass = (
        eval_metrics["mean_reef_fraction"] > 0.0
        and eval_metrics["n_forage_steps"] >= 5
        and not p1_metrics["p1_inert_gating_detected"]
    )
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'}", flush=True)
    return seed_summary


# ---------------------------------------------------------------------------
# Acceptance computation
# ---------------------------------------------------------------------------

def _aggregate_arm(seed_results: List[Dict]) -> Dict:
    rfs = [r["mean_reef_fraction"] for r in seed_results]
    rhos = [r["rho_drive_vs_reef"] for r in seed_results]
    forage_hrs = [r["forage_hazard_rate"] for r in seed_results]
    transit_hrs = [r["transit_hazard_rate"] for r in seed_results]
    inert_flags = [bool(r.get("p1_inert_gating_detected", False)) for r in seed_results]

    # CORRECTION D: hardened C3 -- per-seed ratio enters arm-mean only when
    # transit_rate > C3_TRANSIT_RATE_FLOOR; else np.nan.
    ratios_filtered: List[float] = []
    for r in seed_results:
        if r["transit_hazard_rate"] > C3_TRANSIT_RATE_FLOOR:
            ratios_filtered.append(r["risk_type_ratio"])
        else:
            ratios_filtered.append(float("nan"))
    n_valid_ratios = int(np.sum(~np.isnan(ratios_filtered)))
    mean_ratio_hardened = float(np.nanmean(ratios_filtered)) if n_valid_ratios >= 1 else float("nan")

    cov = (
        float(np.std(rfs) / max(abs(float(np.mean(rfs))), 1e-9))
        if len(rfs) > 1 else 0.0
    )
    return {
        "mean_reef_fraction": float(np.mean(rfs)),
        "std_reef_fraction": float(np.std(rfs)),
        "cov_reef_fraction": cov,
        "rhos_per_seed": rhos,
        "abs_rho_per_seed": [abs(r) for r in rhos],
        "n_rho_above_threshold": sum(1 for r in rhos if abs(r) >= C2_RHO_THRESHOLD),
        "mean_abs_rho": float(np.mean([abs(r) for r in rhos])),
        "mean_forage_hazard_rate": float(np.mean(forage_hrs)),
        "mean_transit_hazard_rate": float(np.mean(transit_hrs)),
        "mean_risk_type_ratio_legacy": float(np.mean([r["risk_type_ratio"] for r in seed_results])),
        "ratios_per_seed_hardened": ratios_filtered,
        "n_valid_seeds_for_c3": n_valid_ratios,
        "mean_risk_type_ratio_hardened": mean_ratio_hardened,
        "inert_gating_per_seed": inert_flags,
        "n_inert_gating_seeds": int(sum(inert_flags)),
    }


def _compute_acceptance(arm_summaries: Dict[str, Dict]) -> Dict:
    a0 = arm_summaries["ARM_0_baseline"]
    a1 = arm_summaries["ARM_1c_full_3stream"]

    # CORRECTION C arm-level probe gate: ARM_1c arm-level non_contributory if
    # >=2/3 seeds short-circuited as inert.
    arm_probe_failed = (a1["n_inert_gating_seeds"] >= 2)

    # C2 state-dependence
    c2_a1_pass = (a1["n_rho_above_threshold"] >= C2_MIN_PASS_SEEDS)
    c2_a1_beats_a0 = (a1["mean_abs_rho"] > a0["mean_abs_rho"])
    c2_pass = bool(c2_a1_pass and c2_a1_beats_a0)

    # CORRECTION D: hardened C3.
    c3_arms_have_enough_valid = (
        a0["n_valid_seeds_for_c3"] >= C3_MIN_VALID_SEEDS_PER_ARM
        and a1["n_valid_seeds_for_c3"] >= C3_MIN_VALID_SEEDS_PER_ARM
    )
    c3_relative_delta_hardened = float("nan")
    c3_pass = False
    if c3_arms_have_enough_valid:
        a0_ratio = a0["mean_risk_type_ratio_hardened"]
        a1_ratio = a1["mean_risk_type_ratio_hardened"]
        if not (np.isnan(a0_ratio) or np.isnan(a1_ratio)):
            c3_relative_delta_hardened = abs(a1_ratio - a0_ratio) / max(abs(a0_ratio), 1e-6)
            c3_pass = bool(c3_relative_delta_hardened >= C3_RELATIVE_DELTA_THRESHOLD)

    # C4 cross-seed variation
    c4_pass = bool(a1["cov_reef_fraction"] >= C4_COV_THRESHOLD)

    n_criteria_passed = int(c2_pass) + int(c3_pass) + int(c4_pass)
    pass_rule_met = (n_criteria_passed >= 2) and (not arm_probe_failed)

    # F1 unambiguous monomodal-collapse (gated_policy free-to-move STILL produces
    # no behavioral divergence; probe gate must have passed to rule out inert
    # explanation).
    reef_diff = abs(a1["mean_reef_fraction"] - a0["mean_reef_fraction"])
    f1_signature = bool(
        not arm_probe_failed
        and reef_diff < F1_REEF_DIFF_THRESHOLD
        and a0["mean_abs_rho"] < F1_C2_RHO_THRESHOLD
        and a1["mean_abs_rho"] < F1_C2_RHO_THRESHOLD
        and (
            (not np.isnan(c3_relative_delta_hardened))
            and c3_relative_delta_hardened < F1_C3_DELTA_THRESHOLD
        )
        and a0["cov_reef_fraction"] < F1_C4_COV_THRESHOLD
        and a1["cov_reef_fraction"] < F1_C4_COV_THRESHOLD
    )

    a1_mean_rho_signed = float(np.mean(a1["rhos_per_seed"])) if a1["rhos_per_seed"] else 0.0
    f2_inverted = bool(a1_mean_rho_signed > F2_INVERTED_RHO_THRESHOLD)

    overall_pass = pass_rule_met and not f1_signature and not f2_inverted

    return {
        "C1_density_tracking": "non_contributory_phase2a_corrected_single_density",
        "C2_state_dependence_pass": c2_pass,
        "C2_a1_seeds_above_threshold": a1["n_rho_above_threshold"],
        "C2_a1_mean_abs_rho": a1["mean_abs_rho"],
        "C2_a0_mean_abs_rho": a0["mean_abs_rho"],
        "C3_risk_type_dissociation_pass": c3_pass,
        "C3_relative_delta_hardened": c3_relative_delta_hardened,
        "C3_a0_ratio_hardened": a0["mean_risk_type_ratio_hardened"],
        "C3_a1_ratio_hardened": a1["mean_risk_type_ratio_hardened"],
        "C3_a0_n_valid_seeds": a0["n_valid_seeds_for_c3"],
        "C3_a1_n_valid_seeds": a1["n_valid_seeds_for_c3"],
        "C3_arms_have_enough_valid_seeds": c3_arms_have_enough_valid,
        "C4_cross_seed_variation_pass": c4_pass,
        "C4_a1_cov_reef_fraction": a1["cov_reef_fraction"],
        "C4_a0_cov_reef_fraction": a0["cov_reef_fraction"],
        "n_criteria_passed": n_criteria_passed,
        "pass_rule_met": pass_rule_met,
        "F1_monomodal_collapse_signature": f1_signature,
        "F2_biologically_inverted_signature": f2_inverted,
        "probe_gate_arm_failed": arm_probe_failed,
        "n_inert_gating_seeds_arm1c": a1["n_inert_gating_seeds"],
        "overall_pass": overall_pass,
    }


def _compute_per_claim_direction(acceptance: Dict) -> Tuple[str, Dict[str, str]]:
    """Per-claim direction grid (CORRECTION C extends 543 to handle inert-gating)."""
    pass_ = acceptance["overall_pass"]
    f1 = acceptance["F1_monomodal_collapse_signature"]
    f2 = acceptance["F2_biologically_inverted_signature"]
    probe_failed = acceptance["probe_gate_arm_failed"]

    if probe_failed:
        outcome = "FAIL"
        per_claim = {
            "ARC-062": "non_contributory",
            "MECH-309": "non_contributory",
        }
    elif pass_:
        outcome = "PASS"
        per_claim = {
            "ARC-062": "supports",
            "MECH-309": "supports",
        }
    elif f1 or f2:
        outcome = "FAIL"
        per_claim = {
            "ARC-062": "weakens",
            "MECH-309": "weakens",
        }
    else:
        outcome = "FAIL"
        per_claim = {
            "ARC-062": "non_contributory",
            "MECH-309": "non_contributory",
        }
    return outcome, per_claim


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(seeds: Optional[List[int]] = None, dry_run: bool = False) -> Dict:
    if seeds is None:
        seeds = [0, 1, 2]

    arms = [
        ("ARM_0_baseline", False),
        ("ARM_1c_full_3stream", True),
    ]

    print(
        f"[V3-EXQ-543c] ARC-062 Phase 3 Monomodal-Collapse Falsifier"
        f" (bipartite SD-054 + world_states[1] fix)"
        f"  seeds={seeds}  dry_run={dry_run}",
        flush=True,
    )

    seed_results_by_arm: Dict[str, List[Dict]] = {a[0]: [] for a in arms}
    for seed in seeds:
        for arm_label, use_gated in arms:
            r = run_arm_seed(arm_label, use_gated, seed, dry_run=dry_run)
            seed_results_by_arm[arm_label].append(r)

    arm_summaries = {
        arm_label: _aggregate_arm(seed_results_by_arm[arm_label])
        for arm_label, _ in arms
    }
    acceptance = _compute_acceptance(arm_summaries)

    return {
        "arm_summaries": arm_summaries,
        "seed_results_by_arm": seed_results_by_arm,
        "acceptance": acceptance,
    }


def write_manifest(result: Dict, dry_run: bool, elapsed: float) -> Tuple[Path, str]:
    acceptance = result["acceptance"]
    outcome, per_claim_direction = _compute_per_claim_direction(acceptance)
    if acceptance["probe_gate_arm_failed"]:
        overall_direction = "non_contributory"
    elif outcome == "PASS":
        overall_direction = "supports"
    elif (acceptance["F1_monomodal_collapse_signature"]
          or acceptance["F2_biologically_inverted_signature"]):
        overall_direction = "weakens"
    else:
        overall_direction = "non_contributory"

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "evidence_direction": overall_direction,
        "evidence_direction_per_claim": per_claim_direction,
        "metrics": {
            "arm_summaries": result["arm_summaries"],
            "acceptance": acceptance,
            "per_seed_per_arm": {
                k: [{kk: vv for kk, vv in s.items() if kk != "per_episode_reef_fractions"}
                    | {"per_episode_reef_fractions": s["per_episode_reef_fractions"]}
                    for s in v]
                for k, v in result["seed_results_by_arm"].items()
            },
        },
        "elapsed_seconds": elapsed,
        "dry_run": dry_run,
        "notes": (
            "ARC-062 Phase 3-corrected single-density 2-arm contrast on SD-054 "
            "(ARM_0 baseline vs ARM_1c full 3-stream gated_policy). Supersedes "
            "V3-EXQ-543. Four corrections vs V3-EXQ-543: (A) gated_policy params "
            "in dedicated Adam during P1, (B) phased training P0 encoder warmup "
            "-> P1 frozen encoder + scaffolding diversification loss on "
            "gated_policy params -> P2 eval, (C) behavioral-divergence probe with "
            "mid-P1 inert-gating short-circuit, (D) hardened C3 with "
            "transit-rate floor 0.05 + np.nanmean. P1 training pressure is "
            "scaffolding (head-pair output divergence + discriminator output "
            "variance), NOT REINFORCE on environmental reward. Honest scope: "
            "this experiment falsifies whether the ARC-062 substrate, when its "
            "parameters are free to move under any non-trivial gradient, can "
            "produce context-conditional behavioral divergence on SD-054. Full "
            "REINFORCE is deferred to Phase 3 GAP-C/D (commitment_closure GAP-1) "
            "when the discriminator is wired into SD-033a LateralPFCAnalog. "
            "Plan-of-record: REE_assembly/evidence/planning/"
            "arc_062_rule_apprehension_plan.md decision-log entry 2026-05-10 "
            "('GAP-B reclassified non_contributory; jump to Phase 3 design'). "
            "PASS unblocks Phase 3 substrate work; FAIL routes through the "
            "falsification chain in Phase 2 deliverable 5."
        ),
    }
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return out_path, outcome


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with reduced episodes/steps for smoke testing.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Override seed list (default [0, 1, 2]).")
    args = parser.parse_args()

    t0 = time.time()
    result = run(seeds=args.seeds, dry_run=args.dry_run)
    elapsed = time.time() - t0

    out_path, outcome = write_manifest(result, args.dry_run, elapsed)

    acc = result["acceptance"]
    print("\n=== V3-EXQ-543c SUMMARY ===", flush=True)
    print(
        f"  C2 state_dependence: pass={acc['C2_state_dependence_pass']}"
        f"  a1_mean_abs_rho={acc['C2_a1_mean_abs_rho']:.3f}"
        f"  a0_mean_abs_rho={acc['C2_a0_mean_abs_rho']:.3f}",
        flush=True,
    )
    print(
        f"  C3 risk_type (hardened): pass={acc['C3_risk_type_dissociation_pass']}"
        f"  delta={acc['C3_relative_delta_hardened']:.3f}"
        f"  a0_ratio={acc['C3_a0_ratio_hardened']}"
        f"  a1_ratio={acc['C3_a1_ratio_hardened']}"
        f"  valid_a0={acc['C3_a0_n_valid_seeds']}"
        f"  valid_a1={acc['C3_a1_n_valid_seeds']}",
        flush=True,
    )
    print(
        f"  C4 cross_seed_cov:  pass={acc['C4_cross_seed_variation_pass']}"
        f"  a1_cov={acc['C4_a1_cov_reef_fraction']:.3f}"
        f"  a0_cov={acc['C4_a0_cov_reef_fraction']:.3f}",
        flush=True,
    )
    print(
        f"  probe_gate_arm_failed: {acc['probe_gate_arm_failed']}"
        f"  n_inert_seeds_arm1c={acc['n_inert_gating_seeds_arm1c']}",
        flush=True,
    )
    print(
        f"  n_criteria_passed: {acc['n_criteria_passed']} (rule = >=2)"
        f"  F1={acc['F1_monomodal_collapse_signature']}"
        f"  F2={acc['F2_biologically_inverted_signature']}",
        flush=True,
    )
    print(f"  outcome:            {outcome}", flush=True)
    print(f"  elapsed:            {elapsed:.1f}s", flush=True)
    print(f"Result written to: {out_path}", flush=True)

    _outcome_raw = str(outcome).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
    )
