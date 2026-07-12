#!/opt/local/bin/python3
"""V3-EXQ-543h: ARC-062 x MECH-309 falsifier with Phase-3 crystallization factor.

Supersedes V3-EXQ-543g. The canonical 543d/g 2x2 SP-CEM factorial + GAP-B option-2
first-action one-hot head input + outcome-coupled REINFORCE P1 loss, UNCHANGED. The
ONLY addition is a third binary factor: crystallize_at_phase3 (INV-074 / MECH-333 /
MECH-334), making this a 2x2x2 = 8-arm factorial.

  V3-EXQ-543g RESULT (2026-05-17, FAIL / weakens ARC-062): ARM_2_gated_only=0.444
    (best), ARM_3_both=0.243 (regresses BELOW gated-only). D2 (ARM_3-ARM_2 >= +0.10)
    FAILED with delta=-0.200. Interpretation: the dACC training signal corrupts the
    gated-policy discrimination via shared-return gradient flow (the heterosynaptic-
    depression analog, O'Dell 2025). The gated policy is never crystallized before
    dACC's perturbation is introduced, so dACC destroys the discrimination ARM_2 had
    established. This is the substrate signature of the missing critical-period closure
    mechanism (MECH-334).

  543h MECHANISM: at a pre-registered point partway through P1 (the open-window-closes
    boundary, CRYSTALLIZE_P1_OPEN_FRACTION of P1), crystallize-ON arms fire the
    INV-074/MECH-334 closure: agent.gated_policy.crystallize() freezes head_0/head_1/
    discriminator and adds a fresh zero-init plastic expansion MLP (Nikishin 2023
    plasticity injection; forward = frozen_gated(x) + expansion(x.detach())); and
    agent.residue_field.snapshot_ewc_anchor() captures the Phase-3 residue checkpoint.
    The P1 REINFORCE optimizer is then rebuilt to target gated_policy.
    expansion_parameters() ONLY (the frozen heads are protected), and
    agent.residue_field.ewc_penalty() (Kirkpatrick 2017, lambda=RESIDUE_EWC_LAMBDA)
    is added to the P1 loss. Diversity signals (dACC/MECH-260, MECH-313, MECH-314a,
    MECH-320) are non-learned arithmetic biases; their gradient influence reaches the
    policy only via the REINFORCE return -- post-crystallize that gradient sinks into
    the expansion layer, NOT the crystallized discrimination.

  SCHEDULER-GATE ADAPTATION (transparent design deviation): the production driver for
    the closure is InfantCurriculumScheduler(on_phase3_entry=...). Its hard episode-min
    gate is PHASE_EP_MIN[3]=2000; this short GAP-B falsifier runs ~108 episodes/arm, so
    the production scheduler cannot organically reach Phase 3 here. 543h instantiates
    the scheduler with the IDENTICAL closure (validates the production wiring contract)
    but fires that closure at the falsifier's open-window-closes boundary. The
    crystallization SUBSTRATE PRIMITIVE (gated_policy.crystallize + residue EWC +
    on_phase3_entry hook) is the scientific object under test; the scheduler episode
    gate is a deployment driver, not testable in a compressed falsifier. Recorded in
    the GAP-B decision log 2026-05-17.

  RESIDUE-EWC SCOPING (honest): in the 543 falsifier the residue field is accumulated
    via .data writes (add_residue / update_valence), NOT gradient-trained, so the
    policy-side plasticity injection is the LOAD-BEARING crystallization factor for the
    D2_xtal hypothesis. The residue EWC term is included per MECH-334, is non-inert
    (lambda=RESIDUE_EWC_LAMBDA, anchored at the crystallize boundary), and its value is
    logged in the manifest; its behavioural effect in this specific falsifier is limited.

  PRE-CHECK ROUTED-SIGNAL F-ERROR DEPENDENCE (encoded in
    REE_assembly/docs/architecture/critical_period_crystallization.md): MECH-314b
    (uncertainty, reads e3._running_variance) and MECH-314c (learning-progress, EMA of
    |PE_t-PE_{t-K}| fed e3._running_variance) are forward-model-error-dependent (314c is
    the canonical Pathak 2017 ICM self-defeat) and decay to ~0 before Phase-3
    crystallization fires; they cannot establish competitive weight on the expansion
    layer. MECH-313 (constant temperature), MECH-314a (residue-RBF novelty, Wittmann
    2008 RPE-independent), MECH-320-primary (avg-reward-rate EWMA, Niv 2007) and
    dACC/MECH-260 (state-dependent recency) are F-robust. 543h therefore holds MECH-314
    in NOVELTY-ONLY config (314a ON, 314b/314c OFF) on the crystallize-ON arms so the
    test is not diluted by a confounded null channel.

Design: 2x2x2 factorial across (use_gated_policy, use_dacc, crystallize_at_phase3)
--------------------------------------------------------
8 arms x 3 seeds = 24 runs. crystallize OFF arms are an EXACT 543g reproduction
(secondary check); crystallize ON arms add the full closure regime (plasticity
injection + residue EWC + MECH-314 novelty-only routing):
  ARM_0_baseline       : gated OFF, dacc OFF, xtal OFF   (= 543g ARM_0)
  ARM_1_dacc_only      : gated OFF, dacc ON,  xtal OFF   (= 543g ARM_1)
  ARM_2_gated_only     : gated ON,  dacc OFF, xtal OFF   (= 543g ARM_2)
  ARM_3_both           : gated ON,  dacc ON,  xtal OFF   (= 543g ARM_3)
  ARM_4_baseline_xtal  : gated OFF, dacc OFF, xtal ON
  ARM_5_dacc_only_xtal : gated OFF, dacc ON,  xtal ON
  ARM_6_gated_only_xtal: gated ON,  dacc OFF, xtal ON
  ARM_7_both_xtal      : gated ON,  dacc ON,  xtal ON

Pre-registered acceptance
-------------------------
  PRIMARY  D2_xtal : ARM_7_both_xtal.reef - ARM_6_gated_only_xtal.reef >= +0.10
                     (with crystallization, dACC-on-top-of-gated NO LONGER regresses
                      -- D2 now PASSes where 543g had D2 FAIL delta=-0.200).
  SECONDARY repro  : ARM_2_gated_only.reef > ARM_3_both.reef (the 543g dACC-regression
                     signature is reproduced on the crystallize-OFF arms).
  CONTEXT (non-gating, reported): D1/D3/D4 on the OFF arms (543g grid), D2_off.

PASS rule: D2_xtal AND repro.

Interpretation grid (one row per outcome -> next action):
  (a) D2_xtal PASS + repro reproduced
      -> crystallization rescues the gradient interference; Q-052 resolves toward
         (B) structural; close GAP-B. {ARC-062 supports, MECH-309 supports,
         INV-074 supports, MECH-334 supports}.
  (b) D2_xtal FAIL + repro reproduced
      -> problem isolated to dACC's own architecture, NOT gradient interference
         (narrower diagnostic); GAP-B stays in-progress, route to dACC-architecture
         review. {INV-074 weakens, MECH-334 weakens, ARC-062 non_contributory,
         MECH-309 non_contributory}.
  (c) repro NOT reproduced (OFF arms do not show the 543g signature)
      -> substrate / seed drift; treat run non_contributory across all claims.
  (d) crystallize ON degrades gated-only below OFF gated-only
      (ARM_6 < ARM_2 by > D2_XTAL_DELTA)
      -> plasticity injection itself harmful; escalate /diagnose-errors.
         {INV-074 weakens, MECH-334 weakens, ARC-062/MECH-309 non_contributory}.

SLEEP DRIVER: not applicable (no sleep loop in this experiment).
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
# INV-074 / MECH-334 production driver for the crystallization closure. Imported
# to instantiate + validate the on_phase3_entry wiring contract; see the
# SCHEDULER-GATE ADAPTATION note in the module docstring.
from infant_curriculum import InfantCurriculumScheduler
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_543h_arc062_crystallization_falsifier"
EXPERIMENT_PURPOSE = "evidence"
QUEUE_ID = "V3-EXQ-543h"
SUPERSEDES = "V3-EXQ-543g"
CLAIM_IDS = ["ARC-062", "MECH-309", "INV-074", "MECH-334"]

# Pre-registered cross-arm attribution thresholds (unchanged from 543d/543e/543f/g).
D1_DACC_ALONE_DELTA = 0.10
D2_DACC_ADDS_TO_GATED_DELTA = 0.10
D3_GATED_ADDS_TO_DACC_DELTA = 0.05
D4_GATED_ALONE_REPLICATION_DELTA = 0.05

# INV-074 / MECH-334 crystallization factor (pre-registered).
# Fraction of P1 after which the open window closes and the crystallization
# closure fires (gated heads established discrimination during [0, this);
# dACC's REINFORCE perturbation thereafter sinks into the expansion layer).
CRYSTALLIZE_P1_OPEN_FRACTION = 0.5
# MECH-334 residue-field EWC penalty weight on crystallize-ON arms (non-inert).
RESIDUE_EWC_LAMBDA = 0.1
# PRIMARY criterion threshold: ARM_7_both_xtal - ARM_6_gated_only_xtal >= this.
D2_XTAL_DELTA = 0.10

# MECH-260 anti-recency suppression weight when use_dacc=True.
DACC_SUPPRESSION_WEIGHT = 0.5
# dacc_weight must be > 0 to activate DACCtoE3Adapter.
DACC_WEIGHT = 1.0

# Env: matches 543f exactly (SD-054 bipartite layout enabled).
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
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)

# Phased training schedule (unchanged from 543f).
P0_WARMUP_EPISODES = 40
P1_TRAIN_EPISODES = 60
P2_EVAL_EPISODES = 8
STEPS_PER_EPISODE = 200

# Latent dims (unchanged from 543f).
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

# Learning rates (unchanged from 543f).
LR_E1 = 1e-4
LR_E2_WF = 3e-4
LR_E3_HARM = 1e-3
LR_ENC_AUX = 5e-4
LR_GATED_POLICY = 5e-4

# Outcome-coupled P1 loss weights.
# head_div_term REMOVED (causes symmetric-cancellation training design error; see 543f autopsy).
# disc_var retained as secondary regularizer at reduced weight.
LAMBDA_DISC_VAR = 0.1

# REINFORCE outcome-buffer config.
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
RECORD_EVERY_N_STEPS = 4
EMA_DECAY = 0.9

# Behavioral-divergence probe config (unchanged from 543f).
PROBE_INTERVAL_P1_EPS = 5
MID_TRAINING_EP = 30
INERT_GATING_THRESHOLD = 0.05
N_PROBE_STATES = 32
N_PROBE_CANDIDATES = 8
SOFTMAX_TEMPERATURE_PROBE = 1.0

# Drive-bin breakpoints for C2 state-dependence Spearman.
DRIVE_BINS = (0.33, 0.67)

# Acceptance thresholds (pre-registered, unchanged from 543f).
C2_RHO_THRESHOLD = 0.20
C2_MIN_PASS_SEEDS = 2
C3_RELATIVE_DELTA_THRESHOLD = 0.50
C4_COV_THRESHOLD = 0.10
F1_REEF_DIFF_THRESHOLD = 0.02
F1_C2_RHO_THRESHOLD = 0.05
F1_C3_DELTA_THRESHOLD = 0.05
F1_C4_COV_THRESHOLD = 0.02
F2_INVERTED_RHO_THRESHOLD = 0.40

# Hardened C3.
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


def _make_agent_and_env(
    seed: int,
    use_gated_policy: bool,
    use_dacc: bool,
    dacc_suppression_weight: float,
    crystallize: bool = False,
) -> Tuple[REEAgent, CausalGridWorldV2]:
    """Build agent + env for the 2x2x2 factorial.

    Inherited from 543g with all fixes active:
    - gated_policy_use_first_action_onehot=use_gated_policy (FIX 1 from 543f)
    - dacc_weight=DACC_WEIGHT (1.0) when use_dacc=True (FIX 2 from 543f)

    When crystallize=True (INV-074 / MECH-333 / MECH-334), the substrate is ARMED:
    REEConfig.crystallize_at_phase3=True (GatedPolicy.crystallize_enabled +
    ResidueConfig.ewc_enabled), residue_ewc_lambda=RESIDUE_EWC_LAMBDA, and MECH-314
    is held in NOVELTY-ONLY config (314a ON, 314b/314c OFF) per the routed-signal
    F-error-dependence pre-check. The crystallize() / snapshot_ewc_anchor() calls
    still only fire at the P1 open-window-closes boundary via the closure in
    _p1_train; arming alone is bit-identical until then.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    _xtal_kwargs = {}
    if crystallize:
        _xtal_kwargs = dict(
            crystallize_at_phase3=True,
            residue_ewc_lambda=RESIDUE_EWC_LAMBDA,
            gated_policy_crystallize_expansion_hidden=32,
            # MECH-314 novelty-only routing (pre-check: 314b/314c are
            # forward-model-error-dependent and ~0 by Phase 3).
            use_structured_curiosity=True,
            use_curiosity_novelty=True,
            use_curiosity_uncertainty=False,
            use_curiosity_learning_progress=False,
        )
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
        # ARC-062 factorial axis 1.
        use_gated_policy=use_gated_policy,
        # one-hot augmentation on gated arms (FIX 1 from 543f).
        gated_policy_use_first_action_onehot=use_gated_policy,
        # MECH-260 / SD-032b factorial axis 2.
        use_dacc=use_dacc,
        # dacc_weight=1.0 activates DACCtoE3Adapter (FIX 2 from 543f).
        dacc_weight=(DACC_WEIGHT if use_dacc else 0.0),
        dacc_suppression_weight=(dacc_suppression_weight if use_dacc else 0.0),
        # SP-CEM on all four arms.
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # INV-074 / MECH-334 crystallization arming + MECH-314 novelty-only
        # (empty dict on crystallize-OFF arms -> exact 543g reproduction).
        **_xtal_kwargs,
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    config.harm_descending_mod_enabled = True
    config.descending_attenuation_factor = 0.5

    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Pre-flight non-degeneracy check (inherited from 543f)
# ---------------------------------------------------------------------------

def _preflight_check() -> None:
    """Verify dACC-on and gated-on configs are non-degenerate before main run."""
    agent_arm1, _env1 = _make_agent_and_env(
        0, use_gated_policy=False, use_dacc=True,
        dacc_suppression_weight=DACC_SUPPRESSION_WEIGHT,
    )
    assert agent_arm1.dacc_adapter is not None, (
        "PREFLIGHT FAIL: ARM_1 dacc_adapter is None (use_dacc=True but adapter not built)"
    )
    actual_dacc_weight = agent_arm1.dacc_adapter.config.dacc_weight
    assert actual_dacc_weight > 0.0, (
        "PREFLIGHT FAIL: ARM_1 dacc_adapter.config.dacc_weight={} (should be > 0). "
        "DACCtoE3Adapter returns zeros when dacc_weight==0.".format(actual_dacc_weight)
    )

    agent_arm2, _env2 = _make_agent_and_env(
        0, use_gated_policy=True, use_dacc=False,
        dacc_suppression_weight=0.0,
    )
    assert agent_arm2.gated_policy is not None, (
        "PREFLIGHT FAIL: ARM_2 gated_policy is None (use_gated_policy=True but not built)"
    )
    assert agent_arm2.gated_policy.config.use_first_action_onehot, (
        "PREFLIGHT FAIL: ARM_2 gated_policy.config.use_first_action_onehot=False."
    )
    fa_dim = agent_arm2.gated_policy.config.first_action_dim
    assert fa_dim > 0, (
        "PREFLIGHT FAIL: ARM_2 gated_policy.config.first_action_dim={} (should be > 0)".format(fa_dim)
    )

    # INV-074 / MECH-334: verify the crystallize factor ARMS the substrate
    # (gated_policy.crystallize_enabled + residue EWC) but does NOT pre-fire.
    agent_xtal, _env_x = _make_agent_and_env(
        0, use_gated_policy=True, use_dacc=True,
        dacc_suppression_weight=DACC_SUPPRESSION_WEIGHT,
        crystallize=True,
    )
    assert agent_xtal.gated_policy is not None, (
        "PREFLIGHT FAIL: xtal arm gated_policy is None"
    )
    assert agent_xtal.gated_policy.config.crystallize_enabled, (
        "PREFLIGHT FAIL: crystallize=True did not arm gated_policy.config."
        "crystallize_enabled"
    )
    assert not agent_xtal.gated_policy.crystallized, (
        "PREFLIGHT FAIL: gated_policy crystallized at construction (must only "
        "fire at the P1 open-window-closes boundary)"
    )
    assert bool(getattr(agent_xtal.residue_field, "ewc_enabled", False)), (
        "PREFLIGHT FAIL: crystallize=True did not arm residue_field.ewc_enabled"
    )
    assert not agent_xtal.residue_field.ewc_anchored, (
        "PREFLIGHT FAIL: residue EWC anchored at construction (must only "
        "snapshot at the P1 open-window-closes boundary)"
    )
    # Novelty-only MECH-314 routing on xtal arms (pre-check).
    assert getattr(agent_xtal.config, "use_structured_curiosity", False), (
        "PREFLIGHT FAIL: xtal arm did not enable MECH-314 (novelty-only)"
    )
    assert getattr(agent_xtal.config, "use_curiosity_novelty", False), (
        "PREFLIGHT FAIL: xtal arm MECH-314a novelty OFF"
    )
    assert not getattr(agent_xtal.config, "use_curiosity_uncertainty", True), (
        "PREFLIGHT FAIL: xtal arm MECH-314b uncertainty ON (pre-check says OFF)"
    )
    assert not getattr(agent_xtal.config, "use_curiosity_learning_progress", True), (
        "PREFLIGHT FAIL: xtal arm MECH-314c learning-progress ON (pre-check OFF)"
    )

    del agent_arm1, agent_arm2, _env1, _env2, agent_xtal, _env_x
    print(
        "Preflight non-degeneracy check PASS: "
        "dacc_weight={} active, first_action_onehot=True (action_dim={}); "
        "crystallize factor arms gated.crystallize_enabled + residue.ewc "
        "without pre-firing; MECH-314 novelty-only on xtal arms.".format(
            DACC_WEIGHT, fa_dim,
        ),
        flush=True,
    )


# ---------------------------------------------------------------------------
# Phase P0: encoder warmup (identical to 543f)
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
    """Phase P0: standard training; gated_policy params NOT in any optimizer.

    Collects probe-state snapshots for TV-distance diagnostic in P1.
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

    probe_snapshot_every = max(1, (num_episodes * steps_per_episode) // (N_PROBE_STATES * 3))
    probe_step_counter = 0

    capture_fa_onehot = (
        agent.gated_policy is not None
        and getattr(agent.gated_policy.config, "use_first_action_onehot", False)
    )

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
                fa_onehots: Optional[torch.Tensor] = None
                if capture_fa_onehot:
                    fa_list = []
                    ok = True
                    for c in candidates[:N_PROBE_CANDIDATES]:
                        if (getattr(c, "actions", None) is not None
                                and c.actions.shape[1] >= 1):
                            fa_list.append(c.actions[:, 0, :][0].detach().float())
                        else:
                            ok = False
                            break
                    if ok and fa_list:
                        fa_onehots = torch.stack(fa_list, dim=0).clone()
                probe_buf.append({
                    "z_world": latent.z_world.detach().clone(),
                    "z_self": latent.z_self.detach().clone(),
                    "z_harm_a": (
                        latent.z_harm_a.detach().clone()
                        if latent.z_harm_a is not None else None
                    ),
                    "candidate_features": first_step_world,
                    "first_action_onehots": fa_onehots,
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
        "p0_mean_reward": float(np.mean(reward_log)) if reward_log else 0.0,
    }


# ---------------------------------------------------------------------------
# Phase P1: outcome-coupled GatedPolicy training
# ---------------------------------------------------------------------------

def _compute_outcome_coupled_loss(
    agent: REEAgent,
    outcome_buf: List[Tuple[Dict, int, float]],
    arm_baseline: float,
    n_samples: int,
    device,
) -> Tuple[torch.Tensor, float, float]:
    """REINFORCE loss over a minibatch from outcome_buf.

    Uses advantage = ep_return - arm_baseline for credit assignment.
    Higher gated_score_bias = more preferred by agent (additive bonus to CEM scores),
    so log_softmax uses positive sign: log_softmax(bias / T, dim=0).
    head_div_term removed (symmetric cancellation: pushing head_0->+X, head_1->-X
    leaves composed output w*X + (1-w)*(-X) = (2w-1)*X ~ 0 for w~0.5).
    disc_var retained as secondary regularizer at reduced weight (LAMBDA_DISC_VAR=0.1).

    Returns (loss_tensor, reinforce_loss_value, disc_var_value).
    """
    if agent.gated_policy is None or len(outcome_buf) < 2:
        zero = torch.zeros(1, device=device, requires_grad=False)
        return zero, 0.0, 0.0

    n = min(n_samples, len(outcome_buf))
    idxs = np.random.choice(len(outcome_buf), size=n, replace=False)

    reinforce_terms: List[torch.Tensor] = []
    disc_w_values: List[torch.Tensor] = []

    for i in idxs:
        snap, sel_idx, ep_return = outcome_buf[int(i)]
        advantage = ep_return - arm_baseline
        if abs(advantage) < ADV_MIN_THRESHOLD:
            continue

        out = agent.gated_policy.forward(
            z_world=snap["z_world"],
            z_self=snap["z_self"],
            z_harm_a=snap.get("z_harm_a"),
            candidate_features=snap["candidate_features"],
            first_action_onehots=snap.get("first_action_onehots"),
            simulation_mode=False,
        )
        K = out.gated_score_bias.shape[0]
        capped_idx = min(sel_idx, K - 1)

        # Correct sign: higher gated_bias -> more preferred by agent.
        log_probs = F.log_softmax(out.gated_score_bias / POLICY_TEMPERATURE, dim=0)
        reinforce_terms.append(-advantage * log_probs[capped_idx])

        # Secondary disc_var: encourages discriminator to vary across states.
        zw = snap["z_world"]
        zs = snap["z_self"]
        za = snap.get("z_harm_a")
        if za is None:
            za = torch.zeros(
                zw.shape[0] if zw.dim() == 2 else 1,
                agent.gated_policy.harm_a_dim,
                device=device,
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

    if not reinforce_terms:
        zero = torch.zeros(1, device=device, requires_grad=False)
        return zero, 0.0, 0.0

    reinforce_loss = torch.stack(reinforce_terms).mean()

    disc_var_term = (
        torch.stack(disc_w_values).var(unbiased=False)
        if len(disc_w_values) > 1
        else torch.zeros(1, device=device)
    )

    # Maximize disc_var (encourages discriminator variance) while minimizing REINFORCE loss.
    loss = reinforce_loss - LAMBDA_DISC_VAR * disc_var_term
    return loss, float(reinforce_loss.detach().item()), float(disc_var_term.detach().item())


def _run_behavioral_divergence_probe(
    agent: REEAgent,
    probe_buf: List[Dict],
) -> Dict:
    """Per-state TV-distance between gated and bypass policies (diagnostic only).

    TV measures deviation from uniform regardless of sign convention.
    ARM_0/ARM_1 (no gated_policy) returns N/A.
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
                first_action_onehots=snap.get("first_action_onehots"),
                simulation_mode=False,
            )
            gated_bias = out.gated_score_bias  # [K]
            T = SOFTMAX_TEMPERATURE_PROBE
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
    p0_mean_reward: float,
    crystallize: bool = False,
) -> Dict:
    """Phase P1: encoder frozen; GatedPolicy trained with outcome-coupled REINFORCE.

    Inherited from 543g: advantage-weighted REINFORCE on GatedPolicy parameters;
    outcome buffer accumulates (snap, sel_idx, ep_return); after each episode a
    minibatch gradient step pulls gated scoring toward higher-return selections.

    INV-074 / MECH-334 (crystallize=True only): at the open-window-closes boundary
    (CRYSTALLIZE_P1_OPEN_FRACTION of num_episodes) the closure fires once --
    agent.gated_policy.crystallize() freezes head_0/head_1/discriminator and adds a
    zero-init plastic expansion MLP; agent.residue_field.snapshot_ewc_anchor()
    captures the Phase-3 residue checkpoint. The REINFORCE optimizer is rebuilt to
    target gated_policy.expansion_parameters() ONLY (frozen heads protected) and
    agent.residue_field.ewc_penalty() is added to the loss. The closure is the
    IDENTICAL callable an InfantCurriculumScheduler(on_phase3_entry=...) would
    invoke in production (instantiated below to validate that wiring contract);
    see the SCHEDULER-GATE ADAPTATION note in the module docstring.
    """
    device = agent.device
    action_dim = env.action_dim

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

    gated_optimizer: Optional[optim.Optimizer] = None
    if use_gated_policy and agent.gated_policy is not None:
        gated_optimizer = optim.Adam(
            agent.gated_policy.parameters(), lr=LR_GATED_POLICY,
        )

    # INV-074 / MECH-334 crystallization closure + state.
    crystallized_active = False
    crystallize_info: Dict = {}
    ewc_snapshot_info: Dict = {}
    last_ewc_penalty = 0.0
    # Boundary: end of the open window. Episodes [0, boundary) train the gated
    # heads; at episode == boundary the closure fires, then [boundary, end)
    # trains only the expansion (frozen heads protected).
    crystallize_p1_ep = (
        int(round(num_episodes * CRYSTALLIZE_P1_OPEN_FRACTION))
        if crystallize else -1
    )

    def crystallize_closure() -> Dict:
        """The exact callable InfantCurriculumScheduler.on_phase3_entry invokes."""
        info = {"gated": None, "residue": None}
        if agent.gated_policy is not None:
            info["gated"] = agent.gated_policy.crystallize()
        if getattr(agent, "residue_field", None) is not None and hasattr(
            agent.residue_field, "snapshot_ewc_anchor"
        ):
            info["residue"] = agent.residue_field.snapshot_ewc_anchor()
        return info

    # Instantiate the production driver to validate the on_phase3_entry wiring
    # contract (its hard episode-min gate is incompatible with this short
    # falsifier -- see SCHEDULER-GATE ADAPTATION; 543h fires the same closure
    # directly at crystallize_p1_ep).
    _xtal_sched: Optional[InfantCurriculumScheduler] = None
    if crystallize:
        _xtal_sched = InfantCurriculumScheduler(
            grid_size=ENV_KWARGS["size"], on_phase3_entry=crystallize_closure,
        )

    capture_fa_onehot = (
        agent.gated_policy is not None
        and getattr(agent.gated_policy.config, "use_first_action_onehot", False)
    )

    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_eval_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []
    reward_log: List[float] = []

    # Outcome buffer for REINFORCE: (snap_dict, sel_idx, ep_return).
    outcome_buf: List[Tuple[Dict, int, float]] = []
    # Baseline initialized from P0 mean reward (warm start avoids cold-start advantage explosion).
    arm_baseline_ema: float = p0_mean_reward

    probe_log: List[Dict] = []
    inert_gating_detected = False

    agent.train()

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()  # GatedPolicy.reset() does NOT un-crystallize (developmental)

        # INV-074 / MECH-334: fire the crystallization closure once at the
        # open-window-closes boundary, then rebuild the REINFORCE optimizer to
        # target ONLY the fresh plastic expansion (frozen heads protected; the
        # dACC-perturbed REINFORCE gradient now sinks into the expansion).
        if crystallize and not crystallized_active and ep == crystallize_p1_ep:
            _info = crystallize_closure()
            crystallize_info = _info.get("gated") or {}
            ewc_snapshot_info = _info.get("residue") or {}
            if (
                agent.gated_policy is not None
                and agent.gated_policy.crystallized
            ):
                _exp_params = list(agent.gated_policy.expansion_parameters())
                if _exp_params:
                    gated_optimizer = optim.Adam(
                        _exp_params, lr=LR_GATED_POLICY,
                    )
            crystallized_active = True
            print(
                f"  [crystallize] {arm_label} P1 ep {ep}/{num_episodes}"
                f"  gated_frozen={crystallize_info.get('n_frozen_params', 0)}"
                f"  expansion_params={crystallize_info.get('n_expansion_params', 0)}"
                f"  ewc_anchored={ewc_snapshot_info.get('anchored', False)}"
                f"  ewc_active_centers={ewc_snapshot_info.get('n_active_centers', 0)}"
                f"  (closure == InfantCurriculumScheduler.on_phase3_entry)",
                flush=True,
            )

        z_world_prev: Optional[torch.Tensor] = None
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        ep_reward = 0.0

        # Per-episode buffer of (snap, sel_idx) recorded every RECORD_EVERY_N_STEPS steps.
        ep_step_buf: List[Tuple[Dict, int]] = []
        step_counter = 0

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

            # Build outcome snap before select_action (candidates still available).
            snap_this_step: Optional[Dict] = None
            n_cands_recorded = 0
            if (gated_optimizer is not None
                    and step_counter % RECORD_EVERY_N_STEPS == 0
                    and isinstance(candidates, list)
                    and len(candidates) >= N_PROBE_CANDIDATES
                    and getattr(candidates[0], "world_states", None) is not None
                    and len(candidates[0].world_states) >= 2):
                n_c = min(len(candidates), N_PROBE_CANDIDATES)
                first_step_world = torch.cat([
                    c.world_states[1].detach().clone()
                    for c in candidates[:n_c]
                ], dim=0)
                fa_onehots: Optional[torch.Tensor] = None
                if capture_fa_onehot:
                    fa_list = []
                    ok = True
                    for c in candidates[:n_c]:
                        if (getattr(c, "actions", None) is not None
                                and c.actions.shape[1] >= 1):
                            fa_list.append(c.actions[:, 0, :][0].detach().float())
                        else:
                            ok = False
                            break
                    if ok and fa_list:
                        fa_onehots = torch.stack(fa_list, dim=0).clone()
                snap_this_step = {
                    "z_world": latent.z_world.detach().clone(),
                    "z_self": latent.z_self.detach().clone(),
                    "z_harm_a": (
                        latent.z_harm_a.detach().clone()
                        if latent.z_harm_a is not None else None
                    ),
                    "candidate_features": first_step_world,
                    "first_action_onehots": fa_onehots,
                }
                n_cands_recorded = n_c

            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, action_dim - 1), action_dim, device,
                )
                agent._last_action = action

            # Recover selected candidate index by matching first-action argmax.
            if snap_this_step is not None and action is not None and n_cands_recorded > 0:
                action_argmax = int(action.argmax(-1).item())
                sel_idx = 0
                for ci, c in enumerate(candidates[:n_cands_recorded]):
                    if (getattr(c, "actions", None) is not None
                            and c.actions.shape[1] >= 1):
                        fa_argmax = int(c.actions[:, 0, :].argmax(-1).item())
                        if fa_argmax == action_argmax:
                            sel_idx = ci
                            break
                ep_step_buf.append((snap_this_step, sel_idx))

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
            step_counter += 1
            if done:
                break

        reward_log.append(ep_reward)

        # Update EMA baseline and add episode tuples to outcome buffer.
        arm_baseline_ema = EMA_DECAY * arm_baseline_ema + (1.0 - EMA_DECAY) * ep_reward
        for snap, sel_idx in ep_step_buf:
            outcome_buf.append((snap, sel_idx, ep_reward))
        if len(outcome_buf) > OUTCOME_BUF_MAX:
            outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]

        # REINFORCE gradient step from outcome buffer.
        if gated_optimizer is not None and len(outcome_buf) >= 4:
            n_steps = 1 if not dry_run else 1
            for _ in range(n_steps):
                loss, reinforce_val, disc_var_val = _compute_outcome_coupled_loss(
                    agent, outcome_buf, arm_baseline_ema,
                    n_samples=min(BATCH_SIZE, len(outcome_buf)),
                    device=device,
                )
                # INV-074 / MECH-334: residue EWC write-protect term.
                # ewc_penalty() returns a 0.0 scalar pre-anchor / when
                # disabled. Recorded for the manifest on every crystallized
                # path; added to the loss ONLY inside a genuine REINFORCE step
                # (riding the same backward) so it cannot flip the
                # reinforce-only requires_grad gate. Honest scoping: the
                # residue field is accumulated via .data writes (not in
                # gated_optimizer, which post-crystallize targets expansion
                # params only), so the load-bearing crystallization factor is
                # the policy-side plasticity injection; the residue EWC grad
                # is cleared after the step to avoid cross-iteration
                # accumulation. See module docstring RESIDUE-EWC SCOPING.
                _ewc = None
                if crystallized_active and getattr(
                    agent, "residue_field", None
                ) is not None and hasattr(agent.residue_field, "ewc_penalty"):
                    _ewc = agent.residue_field.ewc_penalty()
                    last_ewc_penalty = float(_ewc.detach().item())
                if loss.requires_grad:
                    if _ewc is not None:
                        loss = loss + _ewc
                    gated_optimizer.zero_grad()
                    loss.backward()
                    # Post-crystallize the frozen heads carry no grad; clip
                    # over gated_policy.parameters() therefore clips only the
                    # plastic expansion (params-with-grad filtered internally).
                    torch.nn.utils.clip_grad_norm_(
                        agent.gated_policy.parameters(), 1.0,
                    )
                    gated_optimizer.step()
                    # Clear residue param grads from the EWC term so they do
                    # not accumulate across iterations (no optimizer steps
                    # them in this falsifier).
                    if _ewc is not None and hasattr(
                        agent.residue_field, "zero_grad"
                    ):
                        agent.residue_field.zero_grad(set_to_none=True)

        if (ep + 1) % PROBE_INTERVAL_P1_EPS == 0 or ep == num_episodes - 1:
            probe = _run_behavioral_divergence_probe(agent, probe_buf)
            probe["p1_ep"] = ep + 1
            probe_log.append(probe)
            print(
                f"  [probe] {arm_label} P1 ep {ep+1}/{num_episodes}"
                f"  applicable={probe['applicable']}"
                f"  mean_tv={probe['mean_tv_distance']:.4f}"
                f"  max_tv={probe['max_tv_distance']:.4f}"
                f"  buf={len(outcome_buf)}",
                flush=True,
            )
            if (probe["applicable"] and (ep + 1) >= MID_TRAINING_EP
                    and probe["mean_tv_distance"] < INERT_GATING_THRESHOLD
                    and not inert_gating_detected):
                inert_gating_detected = True
                print(
                    f"  [probe] {arm_label} INERT-GATING at "
                    f"P1 ep {ep+1}: mean_tv={probe['mean_tv_distance']:.4f} "
                    f"< {INERT_GATING_THRESHOLD} (REINFORCE outcome-coupling active)",
                    flush=True,
                )

        if (ep + 1) % 10 == 0 or ep == num_episodes - 1:
            cur_total_ep = p0_episodes + ep + 1
            print(
                f"  [train] {arm_label} ep {cur_total_ep}/{total_train_episodes}"
                f"  phase=P1  rv={agent.e3._running_variance:.4f}"
                f"  ep_reward={ep_reward:.4f}"
                f"  baseline={arm_baseline_ema:.4f}"
                f"  buf={len(outcome_buf)}"
                f"  inert_gating={inert_gating_detected}",
                flush=True,
            )

    return {
        "p1_first10_reward": float(np.mean(reward_log[:10])) if len(reward_log) >= 10 else float(np.mean(reward_log)),
        "p1_last10_reward": float(np.mean(reward_log[-10:])) if len(reward_log) >= 10 else float(np.mean(reward_log)),
        "p1_inert_gating_detected": bool(inert_gating_detected),
        "p1_probe_log": probe_log,
        "p1_final_running_variance": agent.e3._running_variance,
        "p1_final_baseline": float(arm_baseline_ema),
        "p1_outcome_buf_size": len(outcome_buf),
        # INV-074 / MECH-334 crystallization telemetry.
        "p1_crystallize_requested": bool(crystallize),
        "p1_crystallized_active": bool(crystallized_active),
        "p1_crystallize_p1_ep": int(crystallize_p1_ep),
        "p1_crystallize_n_frozen_params": int(
            crystallize_info.get("n_frozen_params", 0)
        ),
        "p1_crystallize_n_expansion_params": int(
            crystallize_info.get("n_expansion_params", 0)
        ),
        "p1_residue_ewc_anchored": bool(
            ewc_snapshot_info.get("anchored", False)
        ),
        "p1_residue_ewc_n_active_centers": int(
            ewc_snapshot_info.get("n_active_centers", 0)
        ),
        "p1_residue_ewc_fisher_sum": float(
            ewc_snapshot_info.get("fisher_sum", 0.0)
        ),
        "p1_last_ewc_penalty": float(last_ewc_penalty),
        "p1_xtal_scheduler_wired": bool(_xtal_sched is not None),
    }


# ---------------------------------------------------------------------------
# Phase P2: eval (identical to 543f)
# ---------------------------------------------------------------------------

def _p2_eval_collect_metrics(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    total_train_episodes: int,
    arm_label: str,
) -> Dict:
    """Phase P2: eval with behavioural metric collection."""
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
    use_dacc: bool,
    seed: int,
    dry_run: bool,
    crystallize: bool = False,
) -> Dict:
    p0_eps = 3 if dry_run else P0_WARMUP_EPISODES
    p1_eps = 4 if dry_run else P1_TRAIN_EPISODES
    p2_eps = 2 if dry_run else P2_EVAL_EPISODES
    steps_per_ep = 30 if dry_run else STEPS_PER_EPISODE
    total_train_eps = p0_eps + p1_eps + p2_eps

    print(f"\nSeed {seed} Condition {arm_label}", flush=True)
    print(
        f"  use_gated_policy={use_gated_policy}  use_dacc={use_dacc}"
        f"  dacc_weight={DACC_WEIGHT if use_dacc else 0.0}"
        f"  dacc_suppression_weight={DACC_SUPPRESSION_WEIGHT if use_dacc else 0.0}"
        f"  first_action_onehot={use_gated_policy}"
        f"  crystallize={crystallize}"
        f"  P0={p0_eps}  P1={p1_eps}  P2={p2_eps}  steps/ep={steps_per_ep}",
        flush=True,
    )

    agent, env = _make_agent_and_env(
        seed,
        use_gated_policy=use_gated_policy,
        use_dacc=use_dacc,
        dacc_suppression_weight=DACC_SUPPRESSION_WEIGHT,
        crystallize=crystallize,
    )
    gp_status = "off"
    if agent.gated_policy is not None:
        fa_on = getattr(agent.gated_policy.config, "use_first_action_onehot", False)
        fa_dim = getattr(agent.gated_policy.config, "first_action_dim", 0)
        gp_status = f"on(fa_onehot={fa_on},fa_dim={fa_dim})"
    dacc_status = "off"
    if getattr(agent, "dacc_adapter", None) is not None:
        dacc_status = f"on(dw={agent.dacc_adapter.config.dacc_weight},sw={agent.dacc_adapter.config.dacc_suppression_weight})"
    print(
        f"  world_obs_dim={env.world_obs_dim}"
        f"  agent.gated_policy={gp_status}"
        f"  agent.dacc={dacc_status}",
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
        p0_mean_reward=p0_metrics["p0_mean_reward"],
        crystallize=crystallize,
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
        f"  inert_gating={p1_metrics['p1_inert_gating_detected']}"
        f"  outcome_buf={p1_metrics['p1_outcome_buf_size']}",
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
    """2x2x2 acceptance.

    PRIMARY  D2_xtal : ARM_7_both_xtal - ARM_6_gated_only_xtal >= D2_XTAL_DELTA
                       (crystallization rescues the 543g dACC regression).
    SECONDARY repro  : ARM_2_gated_only > ARM_3_both (543g signature reproduced
                       on the crystallize-OFF arms).
    PASS rule: D2_xtal AND repro.
    Context (non-gating, reported): D1/D2_off/D3/D4 on the OFF arms (543g grid)
    + legacy C2/C3/C4/F1/F2 on OFF ARM_3 vs ARM_0.
    """
    a0 = arm_summaries["ARM_0_baseline"]
    a1 = arm_summaries["ARM_1_dacc_only"]
    a2 = arm_summaries["ARM_2_gated_only"]
    a3 = arm_summaries["ARM_3_both"]
    a6 = arm_summaries["ARM_6_gated_only_xtal"]
    a7 = arm_summaries["ARM_7_both_xtal"]

    arm2_probe_failed = (a2["n_inert_gating_seeds"] >= 2)
    arm3_probe_failed = (a3["n_inert_gating_seeds"] >= 2)
    arm6_probe_failed = (a6["n_inert_gating_seeds"] >= 2)
    arm7_probe_failed = (a7["n_inert_gating_seeds"] >= 2)

    d1_delta = a1["mean_reef_fraction"] - a0["mean_reef_fraction"]
    d1_pass = bool(d1_delta >= D1_DACC_ALONE_DELTA)

    # D2_off: the 543g comparison on the crystallize-OFF arms (expected < 0).
    d2_off_delta = a3["mean_reef_fraction"] - a2["mean_reef_fraction"]
    d2_off_pass = bool(d2_off_delta >= D2_DACC_ADDS_TO_GATED_DELTA)

    # PRIMARY: D2_xtal -- with crystallization, dACC-on-top-of-gated no longer
    # regresses below gated-only.
    d2_xtal_delta = a7["mean_reef_fraction"] - a6["mean_reef_fraction"]
    d2_xtal_pass = bool(d2_xtal_delta >= D2_XTAL_DELTA)

    # SECONDARY repro: the 543g dACC regression is reproduced on OFF arms
    # (gated-only strictly beats both).
    repro_543g_signature = bool(
        a2["mean_reef_fraction"] > a3["mean_reef_fraction"]
    )

    # Branch (d) guard: crystallization must not itself degrade gated-only
    # below the OFF gated-only baseline by more than the primary threshold.
    xtal_harms_gated_only = bool(
        (a2["mean_reef_fraction"] - a6["mean_reef_fraction"]) > D2_XTAL_DELTA
    )

    d3_delta = a3["mean_reef_fraction"] - a1["mean_reef_fraction"]
    d3_pass = bool(d3_delta >= D3_GATED_ADDS_TO_DACC_DELTA)

    d4_delta_abs = abs(a2["mean_reef_fraction"] - a0["mean_reef_fraction"])
    d4_pass = bool(d4_delta_abs < D4_GATED_ALONE_REPLICATION_DELTA)

    overall_pass = bool(d2_xtal_pass and repro_543g_signature
                        and not xtal_harms_gated_only)

    # Legacy 543b/c grid on ARM_3 vs ARM_0 (secondary; reported only).
    a1c_legacy = a3
    c2_a1_pass = (a1c_legacy["n_rho_above_threshold"] >= C2_MIN_PASS_SEEDS)
    c2_a1_beats_a0 = (a1c_legacy["mean_abs_rho"] > a0["mean_abs_rho"])
    c2_pass = bool(c2_a1_pass and c2_a1_beats_a0)

    c3_arms_have_enough_valid = (
        a0["n_valid_seeds_for_c3"] >= C3_MIN_VALID_SEEDS_PER_ARM
        and a1c_legacy["n_valid_seeds_for_c3"] >= C3_MIN_VALID_SEEDS_PER_ARM
    )
    c3_relative_delta_hardened = float("nan")
    c3_pass = False
    if c3_arms_have_enough_valid:
        a0_ratio = a0["mean_risk_type_ratio_hardened"]
        a1_ratio = a1c_legacy["mean_risk_type_ratio_hardened"]
        if not (np.isnan(a0_ratio) or np.isnan(a1_ratio)):
            c3_relative_delta_hardened = abs(a1_ratio - a0_ratio) / max(abs(a0_ratio), 1e-6)
            c3_pass = bool(c3_relative_delta_hardened >= C3_RELATIVE_DELTA_THRESHOLD)

    c4_pass = bool(a1c_legacy["cov_reef_fraction"] >= C4_COV_THRESHOLD)

    n_criteria_passed_legacy = int(c2_pass) + int(c3_pass) + int(c4_pass)
    legacy_pass_rule_met = (n_criteria_passed_legacy >= 2) and (not arm3_probe_failed)

    reef_diff_legacy = abs(a1c_legacy["mean_reef_fraction"] - a0["mean_reef_fraction"])
    f1_signature = bool(
        not arm3_probe_failed
        and reef_diff_legacy < F1_REEF_DIFF_THRESHOLD
        and a0["mean_abs_rho"] < F1_C2_RHO_THRESHOLD
        and a1c_legacy["mean_abs_rho"] < F1_C2_RHO_THRESHOLD
        and (
            (not np.isnan(c3_relative_delta_hardened))
            and c3_relative_delta_hardened < F1_C3_DELTA_THRESHOLD
        )
        and a0["cov_reef_fraction"] < F1_C4_COV_THRESHOLD
        and a1c_legacy["cov_reef_fraction"] < F1_C4_COV_THRESHOLD
    )

    a3_mean_rho_signed = float(np.mean(a3["rhos_per_seed"])) if a3["rhos_per_seed"] else 0.0
    f2_inverted = bool(a3_mean_rho_signed > F2_INVERTED_RHO_THRESHOLD)

    return {
        # PRIMARY + secondary (pre-registered).
        "D2_xtal_pass": d2_xtal_pass,
        "D2_xtal_delta_arm7_minus_arm6": float(d2_xtal_delta),
        "repro_543g_signature": repro_543g_signature,
        "xtal_harms_gated_only": xtal_harms_gated_only,
        "overall_pass": overall_pass,
        # Context (non-gating).
        "D1_dacc_alone_pass": d1_pass,
        "D1_delta_arm1_minus_arm0": float(d1_delta),
        "D2_off_dacc_adds_to_gated_pass": d2_off_pass,
        "D2_off_delta_arm3_minus_arm2": float(d2_off_delta),
        "D3_gated_adds_to_dacc_pass": d3_pass,
        "D3_delta_arm3_minus_arm1": float(d3_delta),
        "D4_replication_543c_pass": d4_pass,
        "D4_delta_abs_arm2_minus_arm0": float(d4_delta_abs),
        "reef_fraction_per_arm": {
            "ARM_0_baseline": float(a0["mean_reef_fraction"]),
            "ARM_1_dacc_only": float(a1["mean_reef_fraction"]),
            "ARM_2_gated_only": float(a2["mean_reef_fraction"]),
            "ARM_3_both": float(a3["mean_reef_fraction"]),
            "ARM_6_gated_only_xtal": float(a6["mean_reef_fraction"]),
            "ARM_7_both_xtal": float(a7["mean_reef_fraction"]),
        },
        "probe_gate_arm2_failed": arm2_probe_failed,
        "probe_gate_arm3_failed": arm3_probe_failed,
        "probe_gate_arm6_failed": arm6_probe_failed,
        "probe_gate_arm7_failed": arm7_probe_failed,
        "n_inert_gating_seeds_arm2": a2["n_inert_gating_seeds"],
        "n_inert_gating_seeds_arm3": a3["n_inert_gating_seeds"],
        "n_inert_gating_seeds_arm6": a6["n_inert_gating_seeds"],
        "n_inert_gating_seeds_arm7": a7["n_inert_gating_seeds"],
        # Legacy 543b/c grid on OFF ARM_3 vs ARM_0 (context only, NOT pass-gating).
        "C1_density_tracking": "non_contributory_phase2a_corrected_single_density",
        "C2_state_dependence_pass": c2_pass,
        "C3_risk_type_dissociation_pass": c3_pass,
        "C3_relative_delta_hardened": c3_relative_delta_hardened,
        "C4_cross_seed_variation_pass": c4_pass,
        "n_criteria_passed_legacy": n_criteria_passed_legacy,
        "legacy_pass_rule_met": legacy_pass_rule_met,
        "F1_monomodal_collapse_signature": f1_signature,
        "F2_biologically_inverted_signature": f2_inverted,
        "C1_density_tracking": "non_contributory_phase2a_corrected_single_density",
        "C2_state_dependence_pass": c2_pass,
        "C2_a3_seeds_above_threshold": a3["n_rho_above_threshold"],
        "C2_a3_mean_abs_rho": a3["mean_abs_rho"],
        "C2_a0_mean_abs_rho": a0["mean_abs_rho"],
        "C3_risk_type_dissociation_pass": c3_pass,
        "C3_relative_delta_hardened": c3_relative_delta_hardened,
        "C3_a0_ratio_hardened": a0["mean_risk_type_ratio_hardened"],
        "C3_a3_ratio_hardened": a3["mean_risk_type_ratio_hardened"],
        "C3_a0_n_valid_seeds": a0["n_valid_seeds_for_c3"],
        "C3_a3_n_valid_seeds": a3["n_valid_seeds_for_c3"],
        "C3_arms_have_enough_valid_seeds": c3_arms_have_enough_valid,
        "C4_cross_seed_variation_pass": c4_pass,
        "C4_a3_cov_reef_fraction": a3["cov_reef_fraction"],
        "C4_a0_cov_reef_fraction": a0["cov_reef_fraction"],
        "n_criteria_passed_legacy": n_criteria_passed_legacy,
        "legacy_pass_rule_met": legacy_pass_rule_met,
        "F1_monomodal_collapse_signature": f1_signature,
        "F2_biologically_inverted_signature": f2_inverted,
    }


def _compute_per_claim_direction(acceptance: Dict) -> Tuple[str, Dict[str, str], str]:
    """Pre-registered 2x2x2 interpretation grid (branches a-d in the docstring).

    Returns (outcome, per_claim, branch_label). 4 claims:
    ARC-062, MECH-309 (the falsifier lineage) + INV-074, MECH-334 (the
    crystallization hypothesis).
    """
    d2_xtal = acceptance["D2_xtal_pass"]
    repro = acceptance["repro_543g_signature"]
    xtal_harms = acceptance["xtal_harms_gated_only"]

    if xtal_harms:
        # (d) crystallization itself degraded gated-only -> plasticity
        # injection harmful; escalate /diagnose-errors.
        outcome = "FAIL"
        per_claim = {
            "INV-074": "weakens", "MECH-334": "weakens",
            "ARC-062": "non_contributory", "MECH-309": "non_contributory",
        }
        branch = "d_xtal_harms_gated_only_escalate_diagnose_errors"
    elif not repro:
        # (c) the 543g signature was not reproduced on the OFF arms ->
        # substrate / seed drift; nothing is cleanly attributable.
        outcome = "FAIL"
        per_claim = {
            "INV-074": "non_contributory", "MECH-334": "non_contributory",
            "ARC-062": "non_contributory", "MECH-309": "non_contributory",
        }
        branch = "c_543g_signature_not_reproduced_substrate_drift"
    elif d2_xtal:
        # (a) crystallization rescues the gradient interference; Q-052
        # resolves toward (B) structural; close GAP-B.
        outcome = "PASS"
        per_claim = {
            "INV-074": "supports", "MECH-334": "supports",
            "ARC-062": "supports", "MECH-309": "supports",
        }
        branch = "a_crystallization_rescues_close_GAP-B_Q052_structural"
    else:
        # (b) D2_xtal FAIL with repro reproduced -> the defect is in dACC's
        # own architecture, not gradient interference; narrower diagnostic.
        outcome = "FAIL"
        per_claim = {
            "INV-074": "weakens", "MECH-334": "weakens",
            "ARC-062": "non_contributory", "MECH-309": "non_contributory",
        }
        branch = "b_dacc_architecture_isolated_GAP-B_in_progress"
    return outcome, per_claim, branch


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(seeds: Optional[List[int]] = None, dry_run: bool = False) -> Dict:
    if seeds is None:
        seeds = [0, 1, 2]

    _preflight_check()

    # 2x2x2 = 8 arms (use_gated_policy, use_dacc, crystallize). xtal OFF arms
    # are an exact 543g reproduction; xtal ON arms add the full closure regime.
    arms = [
        ("ARM_0_baseline",       False, False, False),
        ("ARM_1_dacc_only",      False, True,  False),
        ("ARM_2_gated_only",     True,  False, False),
        ("ARM_3_both",           True,  True,  False),
        ("ARM_4_baseline_xtal",  False, False, True),
        ("ARM_5_dacc_only_xtal", False, True,  True),
        ("ARM_6_gated_only_xtal", True, False, True),
        ("ARM_7_both_xtal",      True,  True,  True),
    ]

    print(
        f"[V3-EXQ-543h] ARC-062 x MECH-309 x crystallize_at_phase3 Falsifier"
        f" (INV-074/MECH-334; outcome-coupled REINFORCE P1 loss; 2x2x2)"
        f"  seeds={seeds}  dry_run={dry_run}",
        flush=True,
    )

    seed_results_by_arm: Dict[str, List[Dict]] = {a[0]: [] for a in arms}
    for seed in seeds:
        for arm_label, use_gated, use_dacc, use_xtal in arms:
            r = run_arm_seed(
                arm_label,
                use_gated_policy=use_gated,
                use_dacc=use_dacc,
                seed=seed,
                dry_run=dry_run,
                crystallize=use_xtal,
            )
            seed_results_by_arm[arm_label].append(r)

    arm_summaries = {
        arm_label: _aggregate_arm(seed_results_by_arm[arm_label])
        for arm_label, *_ in arms
    }
    acceptance = _compute_acceptance(arm_summaries)

    return {
        "arm_summaries": arm_summaries,
        "seed_results_by_arm": seed_results_by_arm,
        "acceptance": acceptance,
    }


def write_manifest(result: Dict, dry_run: bool, elapsed: float) -> Tuple[Path, str]:
    acceptance = result["acceptance"]
    outcome, per_claim_direction, branch = _compute_per_claim_direction(acceptance)
    acceptance["interpretation_branch"] = branch

    # Overall direction is a summary of the per-claim grid (branches a-d).
    if outcome == "PASS":
        overall_direction = "supports"
    elif branch.startswith("c_") or branch.startswith("b_"):
        overall_direction = "non_contributory"
    else:
        overall_direction = "weakens"

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
            "V3-EXQ-543h: ARC-062 x MECH-309 x crystallize_at_phase3 falsifier "
            "(INV-074 / MECH-333 / MECH-334). Supersedes V3-EXQ-543g. The 543d/g "
            "2x2 SP-CEM factorial + GAP-B option-2 one-hot head input + outcome-"
            "coupled REINFORCE P1 loss is UNCHANGED; the only addition is a third "
            "binary factor crystallize_at_phase3 (2x2x2 = 8 arms). On crystallize-"
            "ON arms, at CRYSTALLIZE_P1_OPEN_FRACTION={} of P1 the closure fires "
            "(agent.gated_policy.crystallize() freezes head_0/head_1/discriminator "
            "and adds a zero-init plastic expansion MLP -- Nikishin 2023 plasticity "
            "injection, forward = frozen_gated(x) + expansion(x.detach()); "
            "agent.residue_field.snapshot_ewc_anchor() captures the Phase-3 residue "
            "checkpoint). The REINFORCE optimizer is rebuilt to target "
            "expansion_parameters() only; residue_field.ewc_penalty() "
            "(lambda={}) is added to the P1 loss. The closure is the IDENTICAL "
            "callable InfantCurriculumScheduler(on_phase3_entry=...) invokes in "
            "production; the scheduler's hard ep-min gate (2000) is incompatible "
            "with this short falsifier so 543h fires the same closure at the "
            "open-window-closes boundary (SCHEDULER-GATE ADAPTATION; recorded in "
            "arc_062 GAP-B decision log 2026-05-17). Pre-check: MECH-314 held in "
            "novelty-only config on xtal arms (314b/314c are Pathak-2017 ICM-self-"
            "defeat, ~0 by Phase 3). PRIMARY D2_xtal = ARM_7-ARM_6 >= {}; "
            "SECONDARY repro = ARM_2 > ARM_3 (543g signature). PASS = D2_xtal AND "
            "repro AND not xtal_harms_gated_only. Interpretation branch recorded "
            "in metrics.acceptance.interpretation_branch. SD-054 bipartite layout; "
            "thresholds/metrics otherwise inherited from 543c-g.".format(
                CRYSTALLIZE_P1_OPEN_FRACTION, RESIDUE_EWC_LAMBDA, D2_XTAL_DELTA,
            )
        ),
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )
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
    print("\n=== V3-EXQ-543h SUMMARY ===", flush=True)
    rfa = acc["reef_fraction_per_arm"]
    print(
        f"  reef_fraction: "
        f"ARM_0={rfa['ARM_0_baseline']:.3f}  "
        f"ARM_1={rfa['ARM_1_dacc_only']:.3f}  "
        f"ARM_2={rfa['ARM_2_gated_only']:.3f}  "
        f"ARM_3={rfa['ARM_3_both']:.3f}  "
        f"ARM_6_xtal={rfa['ARM_6_gated_only_xtal']:.3f}  "
        f"ARM_7_xtal={rfa['ARM_7_both_xtal']:.3f}",
        flush=True,
    )
    print(
        f"  [PRIMARY] D2_xtal (ARM_7-ARM_6): pass={acc['D2_xtal_pass']}"
        f"  delta={acc['D2_xtal_delta_arm7_minus_arm6']:+.3f}"
        f"  (threshold {D2_XTAL_DELTA})",
        flush=True,
    )
    print(
        f"  [SECONDARY] repro_543g (ARM_2 > ARM_3): {acc['repro_543g_signature']}"
        f"  D2_off delta={acc['D2_off_delta_arm3_minus_arm2']:+.3f}"
        f"  xtal_harms_gated_only={acc['xtal_harms_gated_only']}",
        flush=True,
    )
    print(
        f"  [context OFF arms] D1={acc['D1_dacc_alone_pass']}"
        f"({acc['D1_delta_arm1_minus_arm0']:+.3f}) "
        f"D3={acc['D3_gated_adds_to_dacc_pass']}"
        f"({acc['D3_delta_arm3_minus_arm1']:+.3f}) "
        f"D4={acc['D4_replication_543c_pass']}"
        f"({acc['D4_delta_abs_arm2_minus_arm0']:.3f})",
        flush=True,
    )
    print(
        f"  probe_gate inert: ARM_2={acc['probe_gate_arm2_failed']}"
        f"  ARM_3={acc['probe_gate_arm3_failed']}"
        f"  ARM_6={acc['probe_gate_arm6_failed']}"
        f"  ARM_7={acc['probe_gate_arm7_failed']}",
        flush=True,
    )
    print(
        f"  interpretation_branch: {acc.get('interpretation_branch')}",
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
