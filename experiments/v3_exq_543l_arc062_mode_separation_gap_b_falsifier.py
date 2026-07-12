#!/opt/local/bin/python3
"""V3-EXQ-543l: ARC-062 GAP-B mode-separation floor falsifier (escalated). Supersedes 543k.

EXPERIMENT_PURPOSE = evidence. Post-543k retest with two parameter escalations
on the gated arms: MODE_SEPARATION_FLOOR 0.25 -> 0.5 and
P1_W_DEVIATION_AUX_WEIGHT 0.1 -> 0.3. Everything else (12-arm grid, env,
K_IDENTICAL_RUNS, acceptance criteria, supersession-chain logic, manifest shape)
inherited byte-for-byte from V3-EXQ-543k.

WHY (V3-EXQ-543k 2026-05-22T091714Z FAIL/mixed, ARC-062=weakens): the 0.25/0.1
combination cleared neither basin-stability nor diff_on_escape. The escalated
floor adds 2x stronger non-cancellable mode contrast at w~0.5; the escalated
aux puts 3x stronger pressure on the discriminator to move w off 0.5 during
outcome-coupled REINFORCE. Both are within the GAP-B design envelope per
arc_062_rule_apprehension_plan.md decision-log 2026-05-20.

Design: same 12-arm grid as 543k (ARM_0..ARM_11; 2x2x2 x diff_on/off).
Gated arms: floor=0.50, p1_w_deviation_aux=0.30.
K_IDENTICAL_RUNS=3 per (arm, seed) basin-stability gate; hostname in manifest.

PASS = basin_stable AND diff_on_escape AND diff_off_reproduced_collapse AND c2c3_on_pass.
No per-claim direction unless basin_stable.

SLEEP DRIVER: not applicable (no sleep loop in this experiment).
"""

from __future__ import annotations

import json
import os
import random
import socket
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


EXPERIMENT_TYPE = "v3_exq_543l_arc062_mode_separation_gap_b_falsifier"
EXPERIMENT_PURPOSE = "evidence"
QUEUE_ID = "V3-EXQ-543l"
# Direct successor to V3-EXQ-543k (FAIL/mixed 2026-05-22T091714Z; ARC-062 weakens
# at the 0.25 floor / 0.1 aux combination). Inherits the supersession chain
# from 543k: 543k itself superseded 543i, which superseded 543h+543g per the
# cross-machine autopsy (failure_autopsy_V3-EXQ-543h_2026-05-18).
SUPERSEDES = "V3-EXQ-543k"
SUPERSEDES_CHAIN = ["V3-EXQ-543k", "V3-EXQ-543i", "V3-EXQ-543h", "V3-EXQ-543g"]
CLAIM_IDS = ["ARC-062", "MECH-309", "INV-074", "MECH-334"]
# V3-EXQ-543i differential-heads escape criterion: a diff-ON gated arm
# ESCAPES collapse iff zero of its seeds are inert.
DIFF_ON_ESCAPE_MAX_INERT_SEEDS = 0
# A diff-OFF gated arm must REPRODUCE the 543g/h collapse (sanity/repro):
# at least this many seeds inert. If diff-OFF is NOT inert -> substrate/
# seed drift -> non_contributory (543h branch-(c) logic).
DIFF_OFF_REPRO_MIN_INERT_SEEDS = 2

MODE_SEPARATION_FLOOR = 0.5
P1_W_DEVIATION_AUX_WEIGHT = 0.3
K_IDENTICAL_RUNS = 3

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
    differential_heads: bool = False,
) -> Tuple[REEAgent, CausalGridWorldV2]:
    """Build agent + env for the 2x2x2 (use_gated x use_dacc x crystallize)
    factorial, plus the V3-EXQ-543i use_differential_heads factor on the
    gated arms.

    differential_heads=True enables the ARC-062 base + candidate-axis-norm-
    pinned differential reparameterization (GatedPolicyConfig.
    use_differential_heads; landed 2026-05-18 from the V3-EXQ-543h failure
    autopsy + cross-machine 543g replication). It is wired via the flat
    REEConfig attr config.gated_policy_use_differential_heads, which
    agent.py reads with a getattr fallback (bit-identical when absent /
    when use_gated_policy=False). This is the ONLY factor that differs from
    543h; the P1 outcome-coupled REINFORCE loss is byte-identical, so 543i
    is a clean single-variable test of whether STRUCTURE alone escapes the
    monomodal-collapse equilibrium MECH-309 predicts.

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

    # V3-EXQ-543i factor: ARC-062 differential-heads reparameterization.
    # Flat REEConfig attr read by agent.py GatedPolicyConfig construction
    # via getattr (bit-identical when False / when use_gated_policy=False).
    # Set BEFORE REEAgent(config) so GatedPolicy reads it at construction
    # (single clean build -- no rebuild, no RNG offset vs the diff-OFF arm).
    config.gated_policy_use_differential_heads = bool(differential_heads)
    config.gated_policy_differential_bias_scale = 0.1
    if use_gated_policy:
        config.gated_policy_mode_separation_floor = MODE_SEPARATION_FLOOR
        config.gated_policy_p1_w_deviation_aux_weight = P1_W_DEVIATION_AUX_WEIGHT

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

    # V3-EXQ-543i: verify the use_differential_heads factor actually wires
    # through (agent.py getattr passthrough). diff-OFF must be the legacy
    # two-independent-head path; diff-ON must be the base+delta reparam.
    agent_diff_off, _env_do = _make_agent_and_env(
        0, use_gated_policy=True, use_dacc=False,
        dacc_suppression_weight=0.0, differential_heads=False,
    )
    assert agent_diff_off.gated_policy is not None
    assert agent_diff_off.gated_policy._use_diff is False, (
        "PREFLIGHT FAIL: differential_heads=False did not yield the legacy "
        "two-independent-head path (gated_policy._use_diff is True)"
    )
    assert agent_diff_off.gated_policy.head_0 is not None, (
        "PREFLIGHT FAIL: diff-OFF gated_policy.head_0 is None"
    )
    agent_diff_on, _env_di = _make_agent_and_env(
        0, use_gated_policy=True, use_dacc=False,
        dacc_suppression_weight=0.0, differential_heads=True,
    )
    assert agent_diff_on.gated_policy is not None
    assert agent_diff_on.gated_policy._use_diff is True, (
        "PREFLIGHT FAIL: differential_heads=True did NOT reach GatedPolicy "
        "(agent.py getattr passthrough broken -- the substrate wiring point)"
    )
    assert (agent_diff_on.gated_policy.head_0 is None
            and agent_diff_on.gated_policy.base is not None
            and agent_diff_on.gated_policy.delta is not None), (
        "PREFLIGHT FAIL: diff-ON gated_policy did not build base+delta "
        "(head_0 should be None, base/delta not None)"
    )
    _dbs = agent_diff_on.gated_policy.config.differential_bias_scale
    assert abs(_dbs - 0.1) < 1e-9, (
        "PREFLIGHT FAIL: differential_bias_scale={} (expected 0.1)".format(_dbs)
    )


    agent_floor, _env_fl = _make_agent_and_env(
        0, use_gated_policy=True, use_dacc=False,
        dacc_suppression_weight=0.0, differential_heads=True,
    )
    assert abs(agent_floor.gated_policy.config.mode_separation_floor - MODE_SEPARATION_FLOOR) < 1e-9
    assert abs(agent_floor.gated_policy.config.p1_w_deviation_aux_weight - P1_W_DEVIATION_AUX_WEIGHT) < 1e-9
    del agent_floor, _env_fl
    del agent_arm1, agent_arm2, _env1, _env2, agent_xtal, _env_x
    del agent_diff_off, _env_do, agent_diff_on, _env_di
    print(
        "Preflight non-degeneracy check PASS: "
        "dacc_weight={} active, first_action_onehot=True (action_dim={}); "
        "crystallize factor arms gated.crystallize_enabled + residue.ewc "
        "without pre-firing; MECH-314 novelty-only on xtal arms; "
        "use_differential_heads wires (OFF=two-head, ON=base+delta).".format(
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
    if (
        float(getattr(agent.gated_policy.config, "p1_w_deviation_aux_weight", 0.0)) > 0.0
        and len(disc_w_values) > 0
    ):
        loss = loss + agent.gated_policy.p1_training_auxiliary_loss(disc_w_values[-1])
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
    differential_heads: bool = False,
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
        f"  differential_heads={differential_heads}"
        f"  P0={p0_eps}  P1={p1_eps}  P2={p2_eps}  steps/ep={steps_per_ep}",
        flush=True,
    )

    agent, env = _make_agent_and_env(
        seed,
        use_gated_policy=use_gated_policy,
        use_dacc=use_dacc,
        dacc_suppression_weight=DACC_SUPPRESSION_WEIGHT,
        crystallize=crystallize,
        differential_heads=differential_heads,
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


def _consensus_seed_result(k_runs: List[Dict]) -> Dict:
    inert_flags = [bool(r.get("p1_inert_gating_detected", False)) for r in k_runs]
    consensus_inert = inert_flags[0] if len(set(inert_flags)) == 1 else None
    out = dict(k_runs[-1])
    if consensus_inert is not None:
        out["p1_inert_gating_detected"] = consensus_inert
    out["k_identical_runs"] = int(len(k_runs))
    out["k_inert_flags"] = inert_flags
    out["k_basin_unanimous"] = bool(len(set(inert_flags)) == 1)
    return out


def _basin_stable_all_gated(seed_results_by_arm: Dict[str, List[Dict]]) -> bool:
    gated = (
        "ARM_2_gated_only", "ARM_3_both", "ARM_6_gated_only_xtal", "ARM_7_both_xtal",
        "ARM_8_gated_only_diff", "ARM_9_both_diff",
        "ARM_10_gated_only_xtal_diff", "ARM_11_both_xtal_diff",
    )
    return all(
        all(bool(r.get("k_basin_unanimous", True)) for r in seed_results_by_arm.get(lbl, []))
        for lbl in gated
    )


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
    """V3-EXQ-543i acceptance.

    PRIMARY (543i) : diff_on_escape AND diff_off_reproduced_collapse AND
                     c2c3_on_pass -- does the base+norm-pinned-differential
                     STRUCTURE escape the monomodal-collapse equilibrium?
                     (overall_pass IS this.)
    CONTEXT (543h legacy, supersession continuity only, NOT pass-gating):
                     D2_xtal = ARM_7_both_xtal - ARM_6_gated_only_xtal,
                     repro_543g = ARM_2 > ARM_3, D1/D2_off/D3/D4 on the OFF
                     arms, legacy C2/C3/C4/F1/F2 on OFF ARM_3 vs ARM_0.
    """
    a0 = arm_summaries["ARM_0_baseline"]
    a1 = arm_summaries["ARM_1_dacc_only"]
    a2 = arm_summaries["ARM_2_gated_only"]
    a3 = arm_summaries["ARM_3_both"]
    a6 = arm_summaries["ARM_6_gated_only_xtal"]
    a7 = arm_summaries["ARM_7_both_xtal"]
    # V3-EXQ-543i diff-ON gated arms.
    a8 = arm_summaries["ARM_8_gated_only_diff"]
    a9 = arm_summaries["ARM_9_both_diff"]
    a10 = arm_summaries["ARM_10_gated_only_xtal_diff"]
    a11 = arm_summaries["ARM_11_both_xtal_diff"]

    # ---- V3-EXQ-543i PRIMARY: does the differential STRUCTURE escape the
    # monomodal-collapse equilibrium? Matched diff-OFF/diff-ON gated pairs:
    #   ARM_2  <-> ARM_8   (gated_only,        xtal OFF)
    #   ARM_3  <-> ARM_9   (both,              xtal OFF)
    #   ARM_6  <-> ARM_10  (gated_only_xtal)
    #   ARM_7  <-> ARM_11  (both_xtal)
    diff_pairs = [
        ("ARM_2_gated_only", a2, "ARM_8_gated_only_diff", a8),
        ("ARM_3_both", a3, "ARM_9_both_diff", a9),
        ("ARM_6_gated_only_xtal", a6, "ARM_10_gated_only_xtal_diff", a10),
        ("ARM_7_both_xtal", a7, "ARM_11_both_xtal_diff", a11),
    ]
    diff_on_n_inert = {p[2]: int(p[3]["n_inert_gating_seeds"]) for p in diff_pairs}
    diff_off_n_inert = {p[0]: int(p[1]["n_inert_gating_seeds"]) for p in diff_pairs}
    # (1) diff-ON escapes: ALL 4 diff-ON gated arms have 0 inert seeds.
    diff_on_escape = bool(
        all(v <= DIFF_ON_ESCAPE_MAX_INERT_SEEDS for v in diff_on_n_inert.values())
    )
    # (2) diff-OFF reproduces the collapse (sanity/repro): ALL 4 matched
    # diff-OFF gated arms are inert. If NOT -> substrate/seed drift ->
    # non_contributory (543h branch-(c) logic), the run cannot attribute.
    diff_off_reproduced_collapse = bool(
        all(v >= DIFF_OFF_REPRO_MIN_INERT_SEEDS for v in diff_off_n_inert.values())
    )
    # (3) C2 state-dependence AND C3 risk-type dissociation on the diff-ON
    # gated arms (guards forced-but-misrouted differentiation: the norm pin
    # forces SOME modal split, so we must verify the discriminator routes w
    # by context, not merely that the heads differ). Evaluated on the
    # primary diff-ON arm (ARM_8, gated_only_diff, no xtal confound) vs the
    # baseline ARM_0, mirroring the legacy C2/C3 ARM_3-vs-ARM_0 contract.
    c2_on_seeds_ok = (a8["n_rho_above_threshold"] >= C2_MIN_PASS_SEEDS)
    c2_on_beats_a0 = (a8["mean_abs_rho"] > a0["mean_abs_rho"])
    c2_on_pass = bool(c2_on_seeds_ok and c2_on_beats_a0)
    c3_on_enough = (
        a0["n_valid_seeds_for_c3"] >= C3_MIN_VALID_SEEDS_PER_ARM
        and a8["n_valid_seeds_for_c3"] >= C3_MIN_VALID_SEEDS_PER_ARM
    )
    c3_on_rel_delta = float("nan")
    c3_on_pass = False
    if c3_on_enough:
        _a0r = a0["mean_risk_type_ratio_hardened"]
        _a8r = a8["mean_risk_type_ratio_hardened"]
        if not (np.isnan(_a0r) or np.isnan(_a8r)):
            c3_on_rel_delta = abs(_a8r - _a0r) / max(abs(_a0r), 1e-6)
            c3_on_pass = bool(c3_on_rel_delta >= C3_RELATIVE_DELTA_THRESHOLD)
    c2c3_on_pass = bool(c2_on_pass and c3_on_pass)

    # PRIMARY 543i pass rule.
    diff_primary_pass = bool(
        diff_on_escape and diff_off_reproduced_collapse and c2c3_on_pass
    )

    arm2_probe_failed = (a2["n_inert_gating_seeds"] >= 2)
    arm3_probe_failed = (a3["n_inert_gating_seeds"] >= 2)
    arm6_probe_failed = (a6["n_inert_gating_seeds"] >= 2)
    arm7_probe_failed = (a7["n_inert_gating_seeds"] >= 2)

    d1_delta = a1["mean_reef_fraction"] - a0["mean_reef_fraction"]
    d1_pass = bool(d1_delta >= D1_DACC_ALONE_DELTA)

    # D2_off: the 543g comparison on the crystallize-OFF arms (expected < 0).
    d2_off_delta = a3["mean_reef_fraction"] - a2["mean_reef_fraction"]
    d2_off_pass = bool(d2_off_delta >= D2_DACC_ADDS_TO_GATED_DELTA)

    # CONTEXT (543h legacy, NOT the 543i pass gate): D2_xtal.
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

    # 543h legacy rule (kept for supersession continuity / context only).
    legacy_543h_pass = bool(d2_xtal_pass and repro_543g_signature
                            and not xtal_harms_gated_only)
    # V3-EXQ-543i overall_pass IS the differential-heads PRIMARY.
    overall_pass = diff_primary_pass

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
        # ---- V3-EXQ-543i PRIMARY (pre-registered): differential-heads escape.
        "diff_primary_pass": diff_primary_pass,
        "diff_on_escape": diff_on_escape,
        "diff_off_reproduced_collapse": diff_off_reproduced_collapse,
        "diff_on_n_inert_per_arm": diff_on_n_inert,
        "diff_off_n_inert_per_arm": diff_off_n_inert,
        "c2c3_on_pass": c2c3_on_pass,
        "c2_on_pass": c2_on_pass,
        "c3_on_pass": c3_on_pass,
        "c3_on_relative_delta_hardened": c3_on_rel_delta,
        "overall_pass": overall_pass,
        # ---- 543h legacy grid (supersession continuity / context only;
        # NOT the 543i pass gate).
        "legacy_543h_pass": legacy_543h_pass,
        "D2_xtal_pass": d2_xtal_pass,
        "D2_xtal_delta_arm7_minus_arm6": float(d2_xtal_delta),
        "repro_543g_signature": repro_543g_signature,
        "xtal_harms_gated_only": xtal_harms_gated_only,
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
            "ARM_8_gated_only_diff": float(a8["mean_reef_fraction"]),
            "ARM_9_both_diff": float(a9["mean_reef_fraction"]),
            "ARM_10_gated_only_xtal_diff": float(a10["mean_reef_fraction"]),
            "ARM_11_both_xtal_diff": float(a11["mean_reef_fraction"]),
        },
        "probe_gate_arm2_failed": arm2_probe_failed,
        "probe_gate_arm3_failed": arm3_probe_failed,
        "probe_gate_arm6_failed": arm6_probe_failed,
        "probe_gate_arm7_failed": arm7_probe_failed,
        "n_inert_gating_seeds_arm2": a2["n_inert_gating_seeds"],
        "n_inert_gating_seeds_arm3": a3["n_inert_gating_seeds"],
        "n_inert_gating_seeds_arm6": a6["n_inert_gating_seeds"],
        "n_inert_gating_seeds_arm7": a7["n_inert_gating_seeds"],
        "n_inert_gating_seeds_arm8": a8["n_inert_gating_seeds"],
        "n_inert_gating_seeds_arm9": a9["n_inert_gating_seeds"],
        "n_inert_gating_seeds_arm10": a10["n_inert_gating_seeds"],
        "n_inert_gating_seeds_arm11": a11["n_inert_gating_seeds"],
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
    """Pre-registered V3-EXQ-543i interpretation grid (single-variable test
    of whether the differential STRUCTURE escapes the monomodal-collapse
    equilibrium MECH-309 predicts).

    Returns (outcome, per_claim, branch_label). 4 claims:
    ARC-062, MECH-309 (the weak-reading falsifier lineage) + INV-074,
    MECH-334 (crystallization, only meaningfully testable once the gated
    policy is functional, i.e. on the diff-ON arms).

    Branches:
      (c) diff-OFF gated arms did NOT reproduce the collapse -> substrate /
          seed drift; nothing cleanly attributable; non_contributory all
          (identical logic to the 543h autopsy branch (c)).
      (a) diff-ON ESCAPES (all diff-ON gated arms 0 inert seeds) AND C2/C3
          route by context -> the differential structure is a sufficient
          rule-apprehension inductive bias: ARC-062 weak reading viable,
          MECH-309 holds only for UNSTRUCTURED parametric policies (bounded
          -> weakens its universal reading), and INV-074/MECH-334 become
          testable on a functional policy (the diff-ON xtal arms ARM_10/11
          carry a non-collapsed policy for the closure to crystallize).
      (e) diff-ON still COLLAPSES despite the structural non-equilibrium ->
          MECH-309 strong confirmation (collapse survives a structural
          inductive bias -> a genuine rule-apprehender, ARC-063/V4, is
          required); ARC-062 weak reading falsified; crystallization still
          untestable (non_contributory).
    """
    repro = acceptance["diff_off_reproduced_collapse"]
    escape = acceptance["diff_on_escape"]
    c2c3 = acceptance["c2c3_on_pass"]

    basin_stable = bool(acceptance.get("basin_stable", False))
    if not basin_stable:
        outcome = "FAIL"
        per_claim = {
            "ARC-062": "non_contributory", "MECH-309": "non_contributory",
            "INV-074": "non_contributory", "MECH-334": "non_contributory",
        }
        branch = "b_basin_unstable_nondeterministic_no_directional_read"
    elif not repro:
        outcome = "FAIL"
        per_claim = {
            "ARC-062": "non_contributory", "MECH-309": "non_contributory",
            "INV-074": "non_contributory", "MECH-334": "non_contributory",
        }
        branch = "c_diff_off_collapse_not_reproduced_substrate_drift"
    elif escape and c2c3:
        outcome = "PASS"
        per_claim = {
            "ARC-062": "supports",      # weak-reading structural slot works
            "MECH-309": "weakens",      # bounded: collapse not universal
            "INV-074": "supports",      # plasticity-injection now testable
            "MECH-334": "supports",     # closure on a functional policy
        }
        branch = "a_differential_structure_escapes_collapse_ARC062_viable"
    else:
        # escape failed (still collapses) OR escaped-but-misrouted
        # (forced differentiation without context routing -> not a genuine
        # rule-apprehender). Either way the weak-reading structural bias is
        # insufficient -> MECH-309 strong confirmation.
        outcome = "FAIL"
        per_claim = {
            "ARC-062": "weakens",       # weak reading insufficient
            "MECH-309": "supports",     # collapse survives the structure
            "INV-074": "non_contributory",
            "MECH-334": "non_contributory",
        }
        branch = (
            "e_collapse_survives_structure_MECH309_strong_ARC063_required"
            if not escape else
            "e_escaped_but_misrouted_c2c3_fail_MECH309_strong"
        )
    return outcome, per_claim, branch


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(seeds: Optional[List[int]] = None, dry_run: bool = False) -> Dict:
    if seeds is None:
        seeds = [0, 1, 2]

    _preflight_check()

    # 12-arm pruned design (tuple = label, use_gated, use_dacc, use_xtal,
    # use_diff). ARM_0..ARM_7 = the exact 543h 2x2x2 (use_differential_heads
    # OFF) -- full supersession + the diff-OFF repro/sanity baseline (the
    # gated arms ARM_2/3/6/7 MUST reproduce the cross-machine inert collapse).
    # ARM_8..ARM_11 = the 4 gated arms with use_differential_heads ON -- the
    # V3-EXQ-543i PRIMARY test of whether the base+norm-pinned-differential
    # STRUCTURE alone escapes the monomodal-collapse equilibrium MECH-309
    # predicts. The 4 non-gated diff-ON permutations are omitted (diff is a
    # structural no-op without a GatedPolicy -- they would be bit-identical
    # duplicates of ARM_0/1/4/5). P1 loss byte-identical to 543g/h on every
    # arm -- the ONLY varied factor across the diff-OFF/diff-ON gated pairs
    # is use_differential_heads (clean single-variable test).
    arms = [
        ("ARM_0_baseline",          False, False, False, False),
        ("ARM_1_dacc_only",         False, True,  False, False),
        ("ARM_2_gated_only",        True,  False, False, False),
        ("ARM_3_both",              True,  True,  False, False),
        ("ARM_4_baseline_xtal",     False, False, True,  False),
        ("ARM_5_dacc_only_xtal",    False, True,  True,  False),
        ("ARM_6_gated_only_xtal",   True,  False, True,  False),
        ("ARM_7_both_xtal",         True,  True,  True,  False),
        ("ARM_8_gated_only_diff",   True,  False, False, True),
        ("ARM_9_both_diff",         True,  True,  False, True),
        ("ARM_10_gated_only_xtal_diff", True, False, True, True),
        ("ARM_11_both_xtal_diff",   True,  True,  True,  True),
    ]

    print(
        f"[V3-EXQ-543l] ARC-062 differential-heads Falsifier"
        f" (escalated floor={MODE_SEPARATION_FLOOR} aux={P1_W_DEVIATION_AUX_WEIGHT};"
        f" 12-arm grid + REINFORCE P1 loss inherited from 543k; supersedes 543k)"
        f"  seeds={seeds}  dry_run={dry_run}",
        flush=True,
    )

    seed_results_by_arm: Dict[str, List[Dict]] = {a[0]: [] for a in arms}
    for seed in seeds:
        for arm_label, use_gated, use_dacc, use_xtal, use_diff in arms:
            k_runs: List[Dict] = []
            for _k in range(K_IDENTICAL_RUNS):
                k_runs.append(
                    run_arm_seed(
                        arm_label,
                        use_gated_policy=use_gated,
                        use_dacc=use_dacc,
                        seed=seed,
                        dry_run=dry_run,
                        crystallize=use_xtal,
                        differential_heads=use_diff,
                    )
                )
            seed_results_by_arm[arm_label].append(_consensus_seed_result(k_runs))

    arm_summaries = {
        arm_label: _aggregate_arm(seed_results_by_arm[arm_label])
        for arm_label, *_ in arms
    }
    acceptance = _compute_acceptance(arm_summaries)

    acceptance["basin_stable"] = _basin_stable_all_gated(seed_results_by_arm)
    acceptance["overall_pass"] = bool(
        acceptance.get("diff_primary_pass")
        and acceptance["basin_stable"]
    )
    acceptance["diff_primary_pass"] = acceptance["overall_pass"]

    return {
        "arm_summaries": arm_summaries,
        "seed_results_by_arm": seed_results_by_arm,
        "acceptance": acceptance,
    }


def write_manifest(result: Dict, dry_run: bool, elapsed: float) -> Tuple[Path, str]:
    acceptance = result["acceptance"]
    outcome, per_claim_direction, branch = _compute_per_claim_direction(acceptance)
    acceptance["interpretation_branch"] = branch

    # Overall direction summarises the per-claim grid. Branch a -> supports
    # (ARC-062). Branch c -> non_contributory (substrate/seed drift). Branch
    # e -> mixed (MECH-309 supports while ARC-062 weakens); per_claim is the
    # authoritative split.
    if outcome == "PASS":
        overall_direction = "supports"
    elif branch.startswith("c_"):
        overall_direction = "non_contributory"
    else:
        overall_direction = "mixed"

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
        "supersedes_chain": SUPERSEDES_CHAIN,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "hostname": socket.gethostname(),
        "mode_separation_floor": MODE_SEPARATION_FLOOR,
        "p1_w_deviation_aux_weight": P1_W_DEVIATION_AUX_WEIGHT,
        "k_identical_runs": K_IDENTICAL_RUNS,
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
            "V3-EXQ-543l: ARC-062 GAP-B falsifier with ESCALATED gated-arm "
            "parameters (MODE_SEPARATION_FLOOR 0.25 -> 0.50, "
            "P1_W_DEVIATION_AUX_WEIGHT 0.1 -> 0.3). SUPERSEDES V3-EXQ-543k "
            "(FAIL/mixed 2026-05-22T091714Z, ARC-062=weakens) -- the 0.25/0.1 "
            "combination cleared neither basin-stability nor diff_on_escape. "
            "12-arm grid, env config, acceptance criteria, supersession-chain "
            "logic, manifest shape, and P1 outcome-coupled REINFORCE loss are "
            "byte-identical to 543k; the ONLY varied factors are the two "
            "scalar thresholds applied on the gated arms. This is the GAP-B "
            "unblock path for the arc_062_rule_apprehension cluster (downstream "
            "IGW items V3-EXQ-606b for ARC-064 GAP-I, V3-EXQ-598/598a/598b "
            "for commitment_closure GAP-1 are HELD pending a contributory PASS "
            "from this run). LEGACY (inherited from 543k): "
            "V3-EXQ-543i: ARC-062 differential-heads falsifier. SUPERSEDES "
            "BOTH V3-EXQ-543h and V3-EXQ-543g (543h itself superseded 543g). "
            "Motivated by failure_autopsy_V3-EXQ-543h_2026-05-18: the same "
            "543g config landed gating-ACTIVE on host-A but INERT (n_inert=3, "
            "TV<0.05) on cloud-3 AND cloud-4 -- head_0==head_1 collapse is the "
            "cross-machine COMMON attractor; 543g _144716Z 'weakens' is a 1/3 "
            "minority-basin artifact; 543h non_contributory (crystallization "
            "froze an already-collapsed policy). 12-arm pruned design: "
            "ARM_0..ARM_7 = the exact 543h 2x2x2 with use_differential_heads "
            "OFF (full supersession + diff-OFF repro/sanity baseline); "
            "ARM_8..ARM_11 = the 4 gated arms with use_differential_heads ON "
            "(ARC-062 GatedPolicyConfig.use_differential_heads, landed "
            "2026-05-18: heads synthesized as base +/- delta_hat with "
            "delta_hat candidate-axis L2 norm pinned to "
            "differential_bias_scale=0.1, making delta==0 a structural "
            "non-equilibrium). The P1 outcome-coupled REINFORCE loss is "
            "BYTE-IDENTICAL across every arm -- the ONLY varied factor on the "
            "matched diff-OFF/diff-ON gated pairs is use_differential_heads "
            "(clean single-variable test of structure vs the MECH-309 "
            "monomodal-collapse equilibrium). The crystallize_at_phase3 "
            "third factor is retained (CRYSTALLIZE_P1_OPEN_FRACTION={}, "
            "residue EWC lambda={}); on diff-ON arms it is now testable "
            "because the gated policy is functional. PRIMARY = diff_on_escape "
            "(ALL diff-ON gated arms 0 inert seeds) AND "
            "diff_off_reproduced_collapse (matched diff-OFF gated arms inert) "
            "AND c2c3_on_pass (diff-ON discriminator routes w by context, "
            "guarding forced-but-misrouted differentiation). Branch a -> "
            "ARC-062 supports / MECH-309 weakens (bounded) / INV-074+MECH-334 "
            "testable. Branch e -> MECH-309 strong confirmation, ARC-062 weak "
            "reading falsified, ARC-063/V4 required. Branch c -> "
            "non_contributory (substrate/seed drift). Cross-machine: a "
            "single-machine pass is PROVISIONAL -- governance requires "
            "n_inert==0 confirmation on >=2 machines before this counts as "
            "escape (the 543g cross-machine bistability is exactly the "
            "failure mode under test). Interpretation branch in "
            "metrics.acceptance.interpretation_branch. SD-054 bipartite "
            "layout; thresholds/metrics otherwise inherited from 543c-h."
            .format(CRYSTALLIZE_P1_OPEN_FRACTION, RESIDUE_EWC_LAMBDA)
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
    print("\n=== V3-EXQ-543l SUMMARY ===", flush=True)
    rfa = acc["reef_fraction_per_arm"]
    print(
        f"  reef_fraction OFF: "
        f"ARM_2={rfa['ARM_2_gated_only']:.3f}  "
        f"ARM_3={rfa['ARM_3_both']:.3f}  "
        f"ARM_6={rfa['ARM_6_gated_only_xtal']:.3f}  "
        f"ARM_7={rfa['ARM_7_both_xtal']:.3f}",
        flush=True,
    )
    print(
        f"  reef_fraction ON : "
        f"ARM_8={rfa['ARM_8_gated_only_diff']:.3f}  "
        f"ARM_9={rfa['ARM_9_both_diff']:.3f}  "
        f"ARM_10={rfa['ARM_10_gated_only_xtal_diff']:.3f}  "
        f"ARM_11={rfa['ARM_11_both_xtal_diff']:.3f}",
        flush=True,
    )
    print(
        f"  basin_stable={acc.get('basin_stable')}"
        f"  [PRIMARY] diff_primary_pass={acc['diff_primary_pass']}"
        f"  | diff_on_escape={acc['diff_on_escape']}"
        f"  diff_off_reproduced_collapse={acc['diff_off_reproduced_collapse']}"
        f"  c2c3_on_pass={acc['c2c3_on_pass']}",
        flush=True,
    )
    print(
        f"  diff-ON  n_inert/arm: {acc['diff_on_n_inert_per_arm']}",
        flush=True,
    )
    print(
        f"  diff-OFF n_inert/arm: {acc['diff_off_n_inert_per_arm']}",
        flush=True,
    )
    print(
        f"  [context/supersession 543h legacy] legacy_543h_pass="
        f"{acc['legacy_543h_pass']}  D2_xtal={acc['D2_xtal_pass']}"
        f"({acc['D2_xtal_delta_arm7_minus_arm6']:+.3f})"
        f"  repro_543g={acc['repro_543g_signature']}",
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
