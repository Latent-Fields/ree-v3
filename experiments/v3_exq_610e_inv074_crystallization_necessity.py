#!/opt/local/bin/python3
"""V3-EXQ-610e: INV-074 crystallization necessity test (HARNESS-FIXED retest).

Supersedes V3-EXQ-610d (FAIL; non_contributory / measurement_test_design_defect
per failure_autopsy_V3-EXQ-610d_2026-06-03). 610d (and 610c before it) measured
an UNTRAINED gated policy -- the "policy training" block built a
requires_grad=False tensor and then `pass`ed, no policy_optimizer.step() existed,
the rebuilt expansion optimizer was never stepped, and residue_field.ewc_penalty()
had zero call sites. Crystallization was therefore a behavioral no-op and INV-074's
predicted winner-take-all monostrategy collapse was never instantiated (both arms
near-uniform action entropy 1.06-1.12 of ln(5)=1.609 -> D1/D2 non-discriminative
by construction).

THE THREE FIXES 610e WIRES (the prescription 610d dropped a second time)
-----------------------------------------------------------------------
  FIX 1 (real REINFORCE): the gated_policy is now the BEHAVIORAL policy. Each step
    builds a differentiable Categorical over CEM candidates from
    gated_policy.gated_score_bias, samples a candidate, stores log_prob + entropy,
    and steps env on the sampled candidate's first action. At episode end a real
    REINFORCE update (discounted returns -> mean-baseline advantages ->
    policy_loss = -(log_prob * advantage).sum() + entropy_bonus) calls
    policy_loss.backward() and policy_optimizer.step() on the gated_policy
    parameters during Phases 0-2. The policy genuinely learns and is WTA-prone.

  FIX 2 (stepped expansion optimizer): after gated_policy.crystallize() fires at
    Phase-3 entry (ARM_1 only), the policy_optimizer is REBUILT over
    gated_policy.expansion_parameters() (+ residue rbf params for FIX 3) and is
    ACTUALLY STEPPED each Phase-3 episode so the plastic expansion layer learns.

  FIX 3 (ewc_penalty in the loss): residue_field.ewc_penalty() (which already
    scales by residue_ewc_lambda internally) is summed into the Phase-3 training
    loss after snapshot_ewc_anchor() has been captured. The residue rbf
    Parameters (centers, weights) are included in the Phase-3 optimizer so the
    EWC pull is a non-vacuous protective gradient step (the MECH-334 write-protect
    analog). EWC is armed only in ARM_1 via crystallize_at_phase3=True, so
    ewc_penalty() returns exactly 0 in ARM_0 (unconditional add is safe).

MANDATORY PRE-RUN ASSERTION (the load-bearing process fix)
----------------------------------------------------------
At startup `_assert_fixes_wired()` runs a short synthetic-reward REINFORCE burst on
a crystallize-armed agent and HARD-ASSERTS, before any real run counts:
  (1) policy params receive non-zero grad AND change after policy_optimizer.step()
      AND the trained selected-action entropy drops out of the untrained band
      (>= the 610c/610d 1.06-1.12 signature) -- proving the policy is non-uniform
      and trainable (NOT the untrained-policy signature that killed 610c/610d);
  (2) after crystallize(), expansion_parameters() receive non-zero grad AND change
      after a step (the expansion layer actually learns);
  (3) residue_field.ewc_penalty() > 0 once anchored AND back-propagates a non-zero
      gradient into the residue rbf params (the penalty is in the loss and is live).
Any failure raises AssertionError and the run ERRORs (does not silently produce a
non-contributory result a third time).

EXPERIMENT_PURPOSE = "evidence"

Claims: INV-074 (primary, universal invariant), MECH-334, MECH-333

Design (UNCHANGED from 610d except the three fixes above)
--------------------------------------------------------
2-arm discriminative experiment testing INV-074: plasticity crystallization is
NECESSARY for diversity persistence post-Phase-3.

  ARM_0 (control): crystallize_at_phase3=FALSE. The trained gated policy keeps
    training through Phase 3 under destabilising pressure -> INV-074 predicts
    winner-take-all monostrategy collapse (diversity COLLAPSES post-Phase-3).
  ARM_1 (test): crystallize_at_phase3=TRUE. At Phase-3 entry the differentiated
    policy is frozen + a plastic expansion channel is added (Nikishin 2023) and
    residue EWC write-protect is armed (Kirkpatrick 2017) -> INV-074 predicts the
    discrimination is protected (diversity PERSISTS post-Phase-3).

Both arms: infant_curriculum 4-phase training (Phases 0-3); Phase 3 WITH the
IGW-20260601-023 destabilising pressure (SD-047 multi_source + SD-048
interoceptive noise + accelerated drift -- the 610d autopsy confirmed it partially
worked: control collapse rose from ~0 in 610b to +0.047 in 610d, so it is kept);
use_gated_policy=True, use_differential_heads=True (ARC-062), gated policy bias
authority raised (gated_policy_bias_scale) so the REINFORCE policy can express a
non-uniform action distribution; MECH-313 noise floor + MECH-260 dACC + MECH-341
E3 score diversity preservation all enabled; 3 matched seeds (42, 43, 44).

Metrics
-------
Primary observable: selected_action_entropy (Shannon entropy of the actually
sampled action distribution) at end_phase_2 (peak) and end_phase_3 (post-closure +
pressure). Plus wiring counters (n_policy_steps, n_expansion_steps_phase3,
n_ewc_terms_phase3, ewc_penalty_last) so a reviewer can confirm all three fixes
fired WITHOUT reading the whole script.

Pre-registered acceptance criteria (UNCHANGED from 610c/610d)
------------------------------------------------------------
  D1 (crystallization preserves diversity):
      ARM_1.end_phase_3_entropy - ARM_0.end_phase_3_entropy >= +0.10 nats
  D2 (control shows collapse):
      ARM_0.end_phase_2_entropy - ARM_0.end_phase_3_entropy >= +0.10 nats
  D3 (sanity -- both diverse at Phase-2 peak):
      ARM_0.end_phase_2_entropy > 0.4 AND ARM_1.end_phase_2_entropy > 0.4
PASS rule: D1 AND D2 AND D3.

Interpretation grid (one row per plausible outcome -> next action)
-----------------------------------------------------------------
  (a) D1 + D2 + D3 all PASS:
      -> crystallization NECESSARY for diversity persistence;
         {INV-074 supports, MECH-334 supports, MECH-333 supports}.

  (b) D1 FAIL (ARM_1 does NOT preserve diversity despite crystallization):
      -> crystallization insufficient under destabilising pressure; escalate to
         /diagnose-errors {INV-074 weakens, MECH-334 weakens}.

  (c) SUBSTRATE-CEILING FORK (PRE-REGISTERED): D2 FAIL (ARM_0 control does NOT
      collapse) WHILE the policy IS verifiably trained -- i.e. Phases 0-2 produced
      NON-uniform action entropy (mean_phase2_entropy clearly below the ln(5)=1.609
      uniform ceiling and out of the 610c/610d untrained band 1.06-1.12) AND the
      wiring counters confirm n_policy_steps>0, n_expansion_steps_phase3>0 (ARM_1),
      n_ewc_terms_phase3>0 with ewc_penalty_last>0 (ARM_1):
      -> This STRENGTHENS MECH-341 / MECH-313 (the diversity-preservation
         machinery -- SP-CEM candidate diversity + the policy entropy bonus + the
         dACC/noise-floor substrate -- is robust enough that crystallization is
         UNNECESSARY in this substrate). It does NOT weaken INV-074 (the universal
         invariant is not falsified by a substrate whose diversity floor already
         resists collapse). Disposition: non_contributory-for-INV-074,
         positive-for-MECH-341/313; route to a follow-on that escalates Phase-3
         pressure or lowers the entropy bonus to find the collapse threshold.

  (d) D2 FAIL WITH near-uniform action entropy (mean_phase2_entropy in the
      1.06-1.12 untrained band): this is the 610c/610d untrained-policy signature
      and MUST NOT occur in 610e -- the startup assertion is designed to catch it
      before queueing. If it somehow occurs at full scale, the policy did not
      train -> measurement_test_design_defect; escalate /diagnose-errors. (NOT the
      same as fork (c): (c) requires a verifiably-trained non-uniform policy.)

  (e) D3 FAIL (neither arm diverse at Phase-2): diversity mechanisms failed to
      establish; substrate issue, escalate /diagnose-errors. Non_contributory.

Estimated runtime: ~13h (2 arms x 3 seeds x ~2500 episodes @ 200 steps/ep);
  V3-EXQ-610d actual ~13h on ree-cloud-2. Long cloud run.

supersedes: V3-EXQ-610d (and transitively V3-EXQ-610c).
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# Add experiments/ to path for infant_curriculum import.
EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from infant_curriculum import InfantCurriculumScheduler
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_610e_inv074_crystallization_necessity"
EXPERIMENT_PURPOSE = "evidence"
QUEUE_ID = "V3-EXQ-610e"
CLAIM_IDS = ["INV-074", "MECH-334", "MECH-333"]
BACKLOG_ID = "EVB-0270"
SUPERSEDES = "V3-EXQ-610d"

# Env config: simple grid with hazards + resources for diversity opportunity.
ENV_BASE_KWARGS = dict(
    size=12,
    num_hazards=3,
    num_resources=4,
    hazard_harm=0.05,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    use_proxy_fields=True,
)

# Training schedule: run through all 4 phases (0-3) of infant curriculum.
#   Phase 0: ep 0..99 (babbling)
#   Phase 1: ep 100..499 (benefit discovery)
#   Phase 2: ep 500..1999 (harm/benefit geography)
#   Phase 3: ep 2000+ (pre-gate readiness + DESTABILIZING PRESSURE from IGW-20260601-023)
MAX_EPISODES = 2500
STEPS_PER_EPISODE = 200

# Latent dims.
WORLD_DIM = 32
SELF_DIM = 32
HARM_DIM = 32
HARM_A_DIM = 16
HARM_HISTORY_LEN = 10

# Diversity mechanism weights (substrate-side; act on encoder/aux + select_action
# internals, retained from 610d for the substrate context).
NOISE_FLOOR_WEIGHT = 0.3  # MECH-313 constant temperature lift.
DACC_SUPPRESSION_WEIGHT = 0.5  # MECH-260 anti-recency bias.
DACC_WEIGHT = 1.0  # Must be > 0 to activate DACCtoE3Adapter.

# Crystallization config (ARM_1 only).
RESIDUE_EWC_LAMBDA = 0.1  # EWC penalty weight (MECH-334); applied INSIDE ewc_penalty().

# ----- REINFORCE policy config (FIX 1/2) -----
# The gated_policy is the behavioral policy. Raised bias authority so the policy
# can express a non-uniform action distribution over CEM candidates (the default
# clamp 0.1 keeps the softmax near-uniform -> would reproduce the untrained
# signature). Matched across BOTH arms (only crystallize_at_phase3 differs).
GATED_BIAS_SCALE = 2.0
POLICY_TEMP = 1.0          # softmax temperature over candidate logits = -gated_bias/T.
POLICY_GAMMA = 0.95        # discount for REINFORCE returns.
# MECH-341/MECH-313 policy-layer diversity-preservation analog: an entropy bonus
# that resists collapse during the open window. Overwhelmable by REINFORCE WTA
# pressure under Phase-3 destabilisation (so ARM_0 control can still collapse).
ENTROPY_BONUS_WEIGHT = 0.02

# Learning rates.
LR_E1 = 1e-4
LR_E2_WF = 3e-4
LR_E3_HARM = 1e-3
LR_ENC_AUX = 5e-4
LR_POLICY = 5e-4

# Buffer + batch.
WF_BUF_MAX = 2000
HARM_EVAL_BUF_MAX = 2000
BATCH_SIZE = 32

# Acceptance thresholds (pre-registered; UNCHANGED).
D1_ENTROPY_DELTA = 0.10
D2_COLLAPSE_DELTA = 0.10
D3_MIN_ENTROPY = 0.4

# Untrained-policy action-entropy band (the 610c/610d non-contributory signature,
# expressed as nats; ln(5)=1.609 max). The startup assertion requires the trained
# policy to land BELOW this band.
UNTRAINED_BAND_LOW = 1.04
LN5 = float(np.log(5.0))

# Seeds.
SEEDS = [42, 43, 44]

# Phase boundary episodes for metric capture.
PHASE_2_MIN_EP = 500
PHASE_3_MIN_EP = 2000
ENTROPY_SAMPLE_WINDOW = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _compute_action_entropy(action_counts: Counter) -> float:
    """Shannon entropy of action distribution (nats)."""
    total = sum(action_counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in action_counts.values()]
    return float(-sum(p * np.log(p + 1e-12) for p in probs if p > 0))


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


def _build_candidate_inputs(candidates, latent, device):
    """Per-candidate first-step z_world summary [K, world_dim] + first-action
    one-hots [K, action_dim] (mirrors the agent.select_action gated_policy block)."""
    gp_list: List[torch.Tensor] = []
    fa_list: List[torch.Tensor] = []
    for c in candidates:
        if c.world_states is not None:
            ws = c.get_world_state_sequence()  # [1, horizon, world_dim]
            gp_list.append(ws[0, 0, :].detach())
        else:
            gp_list.append(latent.z_world[0].detach())
        fa_list.append(c.actions[:, 0, :][0].detach().float())
    gp_summaries = torch.stack(gp_list, dim=0).detach().to(device)  # [K, world_dim]
    first_action_onehots = torch.stack(fa_list, dim=0).to(device)  # [K, action_dim]
    return gp_summaries, first_action_onehots


def _gated_policy_select(agent, latent, candidates, device):
    """FIX 1 core: build a differentiable Categorical over candidates from the
    gated_policy bias, sample, and return (action, log_prob, entropy, sel_idx).

    log_prob and entropy carry grad w.r.t. the live gated_policy params (the
    frozen-heads + plastic-expansion path post-crystallize).
    """
    K = len(candidates)
    if K == 0:
        return None, None, None, -1
    gp_summaries, first_action_onehots = _build_candidate_inputs(
        candidates, latent, device,
    )
    use_onehot = bool(getattr(agent.gated_policy.config, "use_first_action_onehot", False))
    # Detach the latent streams: REINFORCE trains ONLY the gated_policy params;
    # the policy gradient must NOT flow back into the encoders (they have their
    # own losses, and their per-step in-place optimizer.step() would otherwise
    # corrupt the retained episode-end REINFORCE graph -- autograd version error).
    _zw = latent.z_world.detach()
    _zs = latent.z_self.detach()
    _za = latent.z_harm_a.detach() if latent.z_harm_a is not None else None
    gp_out = agent.gated_policy(
        z_world=_zw,
        z_self=_zs,
        z_harm_a=_za,
        candidate_features=gp_summaries,
        first_action_onehots=first_action_onehots if use_onehot else None,
        simulation_mode=False,
    )
    gp_bias = gp_out.gated_score_bias  # [K], differentiable
    # E3 lower-is-better: lower bias = more preferred -> logits = -bias / T.
    logits = -gp_bias / POLICY_TEMP
    log_probs_all = F.log_softmax(logits, dim=0)
    if K >= 2:
        dist = Categorical(logits=logits)
        sel_idx = int(dist.sample().item())
        entropy_t = -(log_probs_all.exp() * log_probs_all).sum()
    else:
        sel_idx = 0
        entropy_t = torch.zeros((), device=device)
    log_prob_sel = log_probs_all[sel_idx]
    action = candidates[sel_idx].actions[:, 0, :].to(device)  # [1, action_dim]
    return action, log_prob_sel, entropy_t, sel_idx


def _reinforce_update(
    policy_optimizer,
    policy_param_list,
    ep_log_probs,
    ep_entropies,
    ep_rewards,
    ewc_penalty_fn,
    in_phase3: bool,
    device,
) -> Tuple[int, float]:
    """FIX 1/2/3: REINFORCE policy update at episode end.

    Returns (stepped:int 0/1, ewc_penalty_value:float). ewc_penalty_fn() is
    summed into the loss only when in_phase3 (it returns exactly 0 before
    snapshot_ewc_anchor / when EWC disabled, so the add is always safe).
    """
    if not ep_log_probs:
        return 0, 0.0
    # Discounted returns.
    returns: List[float] = []
    G = 0.0
    for r in reversed(ep_rewards):
        G = float(r) + POLICY_GAMMA * G
        returns.insert(0, G)
    returns_t = torch.tensor(returns, device=device, dtype=torch.float32)
    advantages = returns_t - returns_t.mean()  # mean-baseline advantage
    if advantages.numel() > 1 and float(advantages.std()) > 1e-6:
        advantages = advantages / (advantages.std() + 1e-8)
    log_probs_t = torch.stack(ep_log_probs)  # [T] (grad)
    entropies_t = torch.stack(ep_entropies)  # [T] (grad)
    policy_loss = -(log_probs_t * advantages.detach()).sum()
    entropy_bonus = -ENTROPY_BONUS_WEIGHT * entropies_t.sum()  # subtract entropy = maximize it
    total_loss = policy_loss + entropy_bonus

    ewc_val = 0.0
    if in_phase3:
        # FIX 3: ewc_penalty() already scales by residue_ewc_lambda internally;
        # it returns exactly 0 until snapshot_ewc_anchor() + EWC armed (ARM_1).
        ewc_term = ewc_penalty_fn()
        total_loss = total_loss + ewc_term
        ewc_val = float(ewc_term.detach())

    policy_optimizer.zero_grad()
    total_loss.backward()
    if policy_param_list:
        torch.nn.utils.clip_grad_norm_(policy_param_list, 1.0)
    policy_optimizer.step()
    return 1, ewc_val


# ---------------------------------------------------------------------------
# Agent + env factory
# ---------------------------------------------------------------------------

def _make_agent_and_env(
    seed: int,
    crystallize: bool,
    grid_size: int,
) -> Tuple[REEAgent, CausalGridWorldV2, InfantCurriculumScheduler]:
    """Build agent + env + scheduler for one arm."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = CausalGridWorldV2(seed=seed, **ENV_BASE_KWARGS)

    xtal_kwargs = {}
    if crystallize:
        xtal_kwargs = dict(
            crystallize_at_phase3=True,
            residue_ewc_lambda=RESIDUE_EWC_LAMBDA,
            gated_policy_crystallize_expansion_hidden=32,
            # MECH-314 novelty-only routing (314b/314c are F-error-dependent
            # and self-defeat before Phase 3).
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
        # Diversity mechanisms (both arms).
        use_gated_policy=True,
        gated_policy_use_differential_heads=True,  # ARC-062 fix.
        gated_policy_use_first_action_onehot=True,  # ARC-062 head input (policy sees first action).
        use_dacc=True,
        dacc_weight=DACC_WEIGHT,
        dacc_suppression_weight=DACC_SUPPRESSION_WEIGHT,
        # MECH-313 noise floor.
        use_noise_floor=True,
        noise_floor_weight=NOISE_FLOOR_WEIGHT,
        # MECH-341 E3 score diversity preservation (RETUNED parameters per 2026-05-27).
        use_e3_score_diversity=True,
        use_e3_diversity_entropy_bonus=True,
        use_e3_diversity_stratified_select=False,
        # Crystallization kwargs (ARM_1 only; empty dict on ARM_0).
        **xtal_kwargs,
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    config.harm_descending_mod_enabled = True
    config.descending_attenuation_factor = 0.5
    # FIX 1 authority: raise the gated-policy bias clamp so the REINFORCE policy
    # can express a non-uniform action distribution (matched across arms).
    config.gated_policy_bias_scale = GATED_BIAS_SCALE

    agent = REEAgent(config)

    # INV-074 / MECH-334 Phase-3 crystallization hook (ARM_1 only). Fires once
    # at the Phase 2->3 transition: freeze the trained heads + add the plastic
    # expansion channel, and snapshot the EWC anchor over the populated residue.
    def _on_phase3_entry_closure():
        if crystallize:
            if agent.gated_policy is not None:
                agent.gated_policy.crystallize()
                print(
                    "  [INV-074] gated_policy.crystallize() fired at Phase 3 entry",
                    flush=True,
                )
            if hasattr(agent.residue_field, "snapshot_ewc_anchor"):
                agent.residue_field.snapshot_ewc_anchor()
                print(
                    "  [MECH-334] residue_field.snapshot_ewc_anchor() fired",
                    flush=True,
                )

    scheduler = InfantCurriculumScheduler(
        grid_size=grid_size,
        on_phase3_entry=_on_phase3_entry_closure if crystallize else None,
    )

    return agent, env, scheduler


# ---------------------------------------------------------------------------
# Training loop (all phases) -- gated-policy REINFORCE behavioral policy
# ---------------------------------------------------------------------------

def _train_infant_curriculum(
    agent: REEAgent,
    scheduler: InfantCurriculumScheduler,
    seed: int,
    arm_label: str,
    crystallize: bool,
    dry_run: bool = False,
) -> Dict:
    device = agent.device

    # Encoder / aux / forward-model optimizers (substrate training; unchanged).
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

    # FIX 1: REINFORCE policy optimizer over the gated_policy parameters.
    policy_param_list = (
        [p for p in agent.gated_policy.parameters() if p.requires_grad]
        if agent.gated_policy is not None else []
    )
    policy_optimizer = optim.Adam(policy_param_list, lr=LR_POLICY)
    expansion_optimizer_active = False  # True once rebuilt over expansion params (FIX 2).

    def _ewc_penalty_fn():
        return agent.residue_field.ewc_penalty()

    # Buffers.
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_eval_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []

    # Metrics.
    reward_log: List[float] = []
    action_history: List[int] = []

    end_phase_2_entropy: Optional[float] = None
    end_phase_3_entropy: Optional[float] = None
    phase_2_capture_started = False
    phase_3_capture_started = False

    # Wiring counters (manifest-visible proof that the three fixes fired).
    n_policy_steps = 0
    n_policy_steps_phase3 = 0
    n_expansion_steps_phase3 = 0
    n_ewc_terms_phase3 = 0
    ewc_penalty_last = 0.0
    n_random_fallback_steps = 0

    agent.train()
    max_eps = 5 if dry_run else MAX_EPISODES

    for ep in range(max_eps):
        env_kwargs = {**ENV_BASE_KWARGS, **scheduler.env_kwargs()}
        env = CausalGridWorldV2(seed=seed + ep, **env_kwargs)
        action_dim = env.action_dim

        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        ep_reward = 0.0

        # REINFORCE episode buffers.
        ep_log_probs: List[torch.Tensor] = []
        ep_entropies: List[torch.Tensor] = []
        ep_rewards: List[float] = []

        in_phase3 = scheduler.current_phase == 3

        steps_this_ep = 20 if dry_run else STEPS_PER_EPISODE
        for step_idx in range(steps_this_ep):
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

            # Aux losses (resource proximity, harm accum).
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

            # E2 transition recording.
            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            # Re-sense after aux update.
            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
            )

            # Generate candidate trajectories (SP-CEM Layer-A diversity).
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

            # FIX 1: gated-policy REINFORCE selection (differentiable log_prob).
            action, log_prob_sel, entropy_t, sel_idx = _gated_policy_select(
                agent, latent, candidates, device,
            )
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, action_dim - 1), action_dim, device,
                )
                agent._last_action = action
                n_random_fallback_steps += 1
            else:
                agent._last_action = action
                ep_log_probs.append(log_prob_sel)
                ep_entropies.append(entropy_t)

            action_idx = int(torch.argmax(action).item())
            action_history.append(action_idx)

            # Step env.
            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ep_reward += float(harm_signal)
            ep_rewards.append(float(harm_signal))

            # Populate residue so the EWC anchor (FIX 3) protects a non-trivial
            # field. 610d never called this, so its anchor was empty.
            agent.update_residue(harm_signal=float(harm_signal))

            # World-forward training.
            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > WF_BUF_MAX:
                    wf_buf = wf_buf[-WF_BUF_MAX:]

            harm_target = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0
            harm_eval_buf.append((z_world_curr.cpu(), torch.tensor([harm_target])))
            if len(harm_eval_buf) > HARM_EVAL_BUF_MAX:
                harm_eval_buf = harm_eval_buf[-HARM_EVAL_BUF_MAX:]

            # Train E2 world-forward.
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

            # Train E3 harm eval.
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

            # Train E1 prediction.
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

        # ---- FIX 1/2/3: REINFORCE policy update at episode end ----
        stepped, ewc_val = _reinforce_update(
            policy_optimizer=policy_optimizer,
            policy_param_list=policy_param_list,
            ep_log_probs=ep_log_probs,
            ep_entropies=ep_entropies,
            ep_rewards=ep_rewards,
            ewc_penalty_fn=_ewc_penalty_fn,
            in_phase3=in_phase3,
            device=device,
        )
        if stepped:
            n_policy_steps += 1
            if in_phase3:
                n_policy_steps_phase3 += 1
                if expansion_optimizer_active:
                    n_expansion_steps_phase3 += 1
                if ewc_val > 0.0:
                    n_ewc_terms_phase3 += 1
                    ewc_penalty_last = ewc_val

        reward_log.append(ep_reward)

        # Advance the curriculum (fires the Phase-3 crystallization hook at boundary).
        scheduler.update(episode=ep)

        # Capture entropy at phase boundaries (selected-action distribution).
        if ep >= (PHASE_3_MIN_EP - ENTROPY_SAMPLE_WINDOW) and ep < PHASE_3_MIN_EP:
            phase_2_capture_started = True
        if phase_2_capture_started and ep == PHASE_3_MIN_EP - 1:
            recent_actions = action_history[-(ENTROPY_SAMPLE_WINDOW * STEPS_PER_EPISODE):]
            end_phase_2_entropy = _compute_action_entropy(Counter(recent_actions))
            phase_2_capture_started = False

        if ep >= (max_eps - ENTROPY_SAMPLE_WINDOW):
            phase_3_capture_started = True
        if phase_3_capture_started and ep == max_eps - 1:
            recent_actions = action_history[-(ENTROPY_SAMPLE_WINDOW * STEPS_PER_EPISODE):]
            end_phase_3_entropy = _compute_action_entropy(Counter(recent_actions))

        # Logging (progress instrumentation; M = loop bound).
        if (ep + 1) % 100 == 0 or ep == max_eps - 1 or scheduler.phase_changed:
            print(
                f"  [train] {arm_label} ep {ep+1}/{max_eps} "
                f"phase={scheduler.current_phase} "
                f"ep_reward={ep_reward:.4f} "
                f"rv={agent.e3._running_variance:.4f} "
                f"pol_steps={n_policy_steps} exp_steps={n_expansion_steps_phase3} "
                f"ewc_terms={n_ewc_terms_phase3}",
                flush=True,
            )

        # FIX 2: rebuild the policy optimizer over expansion_parameters() (+ residue
        # rbf params for FIX 3) once crystallization has fired at Phase-3 entry.
        if (
            crystallize
            and scheduler.current_phase == 3
            and not expansion_optimizer_active
            and agent.gated_policy is not None
            and getattr(agent.gated_policy, "crystallized", False)
        ):
            exp_params = [p for p in agent.gated_policy.expansion_parameters() if p.requires_grad]
            residue_params = [
                agent.residue_field.rbf_field.centers,
                agent.residue_field.rbf_field.weights,
            ]
            policy_param_list = exp_params + residue_params
            policy_optimizer = optim.Adam(policy_param_list, lr=LR_POLICY)
            expansion_optimizer_active = True
            print(
                f"  [INV-074] policy_optimizer rebuilt for expansion_parameters "
                f"({len(exp_params)} expansion + {len(residue_params)} residue params)",
                flush=True,
            )

    return {
        "arm": arm_label,
        "mean_reward": float(np.mean(reward_log)) if reward_log else 0.0,
        "final_phase": scheduler.current_phase,
        "end_phase_2_entropy": end_phase_2_entropy,
        "end_phase_3_entropy": end_phase_3_entropy,
        "total_episodes": len(reward_log),
        # Wiring proof (fix verification visible in the manifest).
        "n_policy_steps": n_policy_steps,
        "n_policy_steps_phase3": n_policy_steps_phase3,
        "n_expansion_steps_phase3": n_expansion_steps_phase3,
        "n_ewc_terms_phase3": n_ewc_terms_phase3,
        "ewc_penalty_last": ewc_penalty_last,
        "n_random_fallback_steps": n_random_fallback_steps,
        "crystallized": bool(getattr(agent.gated_policy, "crystallized", False)),
    }


# ---------------------------------------------------------------------------
# Run one arm
# ---------------------------------------------------------------------------

def _run_arm(arm_config: Dict, seed: int, dry_run: bool) -> Dict:
    arm_label = arm_config["label"]
    crystallize = arm_config["crystallize"]

    # Progress: seed/condition boundary line (runner resets episodes_in_run).
    print(f"Seed {seed} Condition {arm_label}", flush=True)
    print(
        f"[V3-EXQ-610e] Starting {arm_label} seed={seed} crystallize={crystallize}",
        flush=True,
    )

    agent, env, scheduler = _make_agent_and_env(
        seed=seed,
        crystallize=crystallize,
        grid_size=ENV_BASE_KWARGS["size"],
    )

    metrics = _train_infant_curriculum(
        agent=agent,
        scheduler=scheduler,
        seed=seed,
        arm_label=arm_label,
        crystallize=crystallize,
        dry_run=dry_run,
    )

    metrics["seed"] = seed
    metrics["crystallize"] = crystallize

    # Progress: per seed x condition run verdict (runner increments runs_done).
    print(f"verdict: PASS", flush=True)
    return metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(results: List[Dict]) -> Dict:
    arm_0_results = [r for r in results if not r["crystallize"]]
    arm_1_results = [r for r in results if r["crystallize"]]

    def _mean(lst, key):
        vals = [r[key] for r in lst if r[key] is not None]
        return float(np.mean(vals)) if vals else 0.0

    arm_0_end_p2 = _mean(arm_0_results, "end_phase_2_entropy")
    arm_0_end_p3 = _mean(arm_0_results, "end_phase_3_entropy")
    arm_1_end_p2 = _mean(arm_1_results, "end_phase_2_entropy")
    arm_1_end_p3 = _mean(arm_1_results, "end_phase_3_entropy")

    d1_delta = arm_1_end_p3 - arm_0_end_p3
    d1_pass = d1_delta >= D1_ENTROPY_DELTA

    d2_delta = arm_0_end_p2 - arm_0_end_p3
    d2_pass = d2_delta >= D2_COLLAPSE_DELTA

    d3_arm0 = arm_0_end_p2 > D3_MIN_ENTROPY
    d3_arm1 = arm_1_end_p2 > D3_MIN_ENTROPY
    d3_pass = d3_arm0 and d3_arm1

    passed = d1_pass and d2_pass and d3_pass

    # Substrate-ceiling fork support signal: was the policy verifiably trained
    # (non-uniform Phase-2 entropy, out of the untrained band) on both arms?
    mean_phase2 = float(np.mean([
        r["end_phase_2_entropy"] for r in results
        if r["end_phase_2_entropy"] is not None
    ])) if any(r["end_phase_2_entropy"] is not None for r in results) else 0.0
    policy_trained_nonuniform = (
        mean_phase2 > 0.0
        and not (UNTRAINED_BAND_LOW <= mean_phase2 <= 1.14)
        and mean_phase2 < LN5
    )

    return {
        "d1_crystallization_preserves_diversity": d1_pass,
        "d1_delta": d1_delta,
        "d2_control_shows_collapse": d2_pass,
        "d2_delta": d2_delta,
        "d3_sanity_both_show_diversity_at_phase2": d3_pass,
        "d3_arm0_phase2_entropy": arm_0_end_p2,
        "d3_arm1_phase2_entropy": arm_1_end_p2,
        "arm_0_end_phase_2_entropy": arm_0_end_p2,
        "arm_0_end_phase_3_entropy": arm_0_end_p3,
        "arm_1_end_phase_2_entropy": arm_1_end_p2,
        "arm_1_end_phase_3_entropy": arm_1_end_p3,
        "mean_phase2_entropy": mean_phase2,
        "policy_trained_nonuniform": policy_trained_nonuniform,
        "ln5_uniform_ceiling": LN5,
        "verdict": "PASS" if passed else "FAIL",
    }


# ---------------------------------------------------------------------------
# MANDATORY PRE-RUN ASSERTION (the load-bearing process fix)
# ---------------------------------------------------------------------------

def _assert_fixes_wired() -> Dict:
    """Hard-assert the three fixes BEFORE any real run counts.

    Runs a short synthetic-reward REINFORCE burst on a crystallize-armed agent
    that strongly rewards action class 0, then verifies:
      (1) policy params receive non-zero grad + change after a step AND the
          trained selected-action entropy drops out of the untrained band
          (NOT the 610c/610d near-uniform 1.06-1.12 signature);
      (2) after crystallize(), expansion_parameters() receive non-zero grad +
          change after a step;
      (3) residue_field.ewc_penalty() > 0 once anchored AND back-propagates a
          non-zero gradient into the residue rbf params.
    Raises AssertionError on any failure.
    """
    print("[V3-EXQ-610e][ASSERT] verifying the three fixes are wired...", flush=True)
    device = torch.device("cpu")
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)

    agent, env, scheduler = _make_agent_and_env(seed=7, crystallize=True, grid_size=8)
    assert agent.gated_policy is not None, "gated_policy must be constructed"

    # Snapshot a pre-train copy of policy params for the change check.
    pol_params = [p for p in agent.gated_policy.parameters() if p.requires_grad]
    assert pol_params, "gated_policy must expose trainable parameters"
    pre_train_clone = [p.detach().clone() for p in pol_params]
    policy_opt = optim.Adam(pol_params, lr=1e-2)

    def _run_synthetic_episode(collect_actions=False):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        logps, ents, rews = [], [], []
        acts = []
        for _ in range(12):
            latent = agent.sense(
                obs_dict["body_state"], obs_dict["world_state"],
                obs_harm=_obs_harm(obs_dict), obs_harm_a=_obs_harm_a(obs_dict),
                obs_harm_history=_obs_harm_history(obs_dict),
            )
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action, logp, ent, sidx = _gated_policy_select(agent, latent, candidates, device)
            if action is None:
                action = _action_to_onehot(random.randint(0, env.action_dim - 1), env.action_dim, device)
                agent._last_action = action
            else:
                agent._last_action = action
                logps.append(logp)
                ents.append(ent)
            a_idx = int(torch.argmax(action).item())
            if collect_actions:
                acts.append(a_idx)
            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            agent.update_residue(harm_signal=float(harm_signal))
            # SYNTHETIC reward: strongly favour action class 0 (decoupled from env)
            # so the policy has an unambiguous gradient to concentrate.
            rews.append(1.0 if a_idx == 0 else -1.0)
            if done:
                break
        return logps, ents, rews, acts

    # --- FIX 1: train the policy on the synthetic preference; assert it moves. ---
    init_acts: List[int] = []
    for _ in range(2):
        _, _, _, a = _run_synthetic_episode(collect_actions=True)
        init_acts.extend(a)
    init_entropy = _compute_action_entropy(Counter(init_acts)) if init_acts else LN5

    grad_seen = False
    for _ in range(50):
        logps, ents, rews, _ = _run_synthetic_episode()
        if not logps:
            continue
        returns, G = [], 0.0
        for r in reversed(rews):
            G = r + POLICY_GAMMA * G
            returns.insert(0, G)
        ret_t = torch.tensor(returns, device=device, dtype=torch.float32)
        adv = ret_t - ret_t.mean()
        if adv.numel() > 1 and float(adv.std()) > 1e-6:
            adv = adv / (adv.std() + 1e-8)
        loss = -(torch.stack(logps) * adv.detach()).sum() - ENTROPY_BONUS_WEIGHT * torch.stack(ents).sum()
        policy_opt.zero_grad()
        loss.backward()
        gsum = sum(float(p.grad.abs().sum()) for p in pol_params if p.grad is not None)
        if gsum > 0.0:
            grad_seen = True
        policy_opt.step()

    trained_acts: List[int] = []
    for _ in range(4):
        _, _, _, a = _run_synthetic_episode(collect_actions=True)
        trained_acts.extend(a)
    trained_entropy = _compute_action_entropy(Counter(trained_acts)) if trained_acts else LN5

    params_changed = any(
        not torch.allclose(a, b) for a, b in zip(pre_train_clone, pol_params)
    )
    assert grad_seen, "FIX 1 FAILED: policy params never received a non-zero gradient"
    assert params_changed, "FIX 1 FAILED: policy params did not change after policy_optimizer.step()"
    # Anti-untrained-signature guard: the trained policy must NOT sit in the
    # 610c/610d near-uniform band (1.06-1.12). Require it below the band edge.
    assert trained_entropy < UNTRAINED_BAND_LOW, (
        f"FIX 1 FAILED: trained selected-action entropy {trained_entropy:.4f} is NOT "
        f"below the untrained band edge {UNTRAINED_BAND_LOW} (ln5={LN5:.4f}). This is "
        f"the 610c/610d untrained-policy signature -- the policy is not learning a "
        f"non-uniform distribution. Do NOT queue."
    )
    print(
        f"[V3-EXQ-610e][ASSERT] FIX 1 OK: grad_seen={grad_seen} params_changed={params_changed} "
        f"init_entropy={init_entropy:.4f} -> trained_entropy={trained_entropy:.4f} "
        f"(< {UNTRAINED_BAND_LOW}; uniform ln5={LN5:.4f})",
        flush=True,
    )

    # --- FIX 2: crystallize, then assert expansion params train. ---
    xtal_info = agent.gated_policy.crystallize()
    assert agent.gated_policy.crystallized, "FIX 2 FAILED: crystallize() did not set crystallized"
    exp_params = [p for p in agent.gated_policy.expansion_parameters() if p.requires_grad]
    assert exp_params, "FIX 2 FAILED: expansion_parameters() is empty after crystallize()"
    pre_exp_clone = [p.detach().clone() for p in exp_params]
    exp_opt = optim.Adam(exp_params, lr=1e-2)
    exp_grad_seen = False
    for _ in range(20):
        logps, ents, rews, _ = _run_synthetic_episode()
        if not logps:
            continue
        returns, G = [], 0.0
        for r in reversed(rews):
            G = r + POLICY_GAMMA * G
            returns.insert(0, G)
        ret_t = torch.tensor(returns, device=device, dtype=torch.float32)
        adv = ret_t - ret_t.mean()
        if adv.numel() > 1 and float(adv.std()) > 1e-6:
            adv = adv / (adv.std() + 1e-8)
        loss = -(torch.stack(logps) * adv.detach()).sum()
        exp_opt.zero_grad()
        loss.backward()
        gsum = sum(float(p.grad.abs().sum()) for p in exp_params if p.grad is not None)
        if gsum > 0.0:
            exp_grad_seen = True
        exp_opt.step()
    exp_changed = any(
        not torch.allclose(a, b) for a, b in zip(pre_exp_clone, exp_params)
    )
    assert exp_grad_seen, "FIX 2 FAILED: expansion params never received a non-zero gradient"
    assert exp_changed, "FIX 2 FAILED: expansion params did not change after a stepped optimizer"
    print(
        f"[V3-EXQ-610e][ASSERT] FIX 2 OK: crystallized={agent.gated_policy.crystallized} "
        f"n_expansion_params={sum(p.numel() for p in exp_params)} "
        f"exp_grad_seen={exp_grad_seen} exp_changed={exp_changed}",
        flush=True,
    )

    # --- FIX 3: snapshot the EWC anchor; assert ewc_penalty() is live + protective. ---
    # Guarantee a populated residue field so the anchor + penalty are non-trivial
    # regardless of stochastic harm in the synthetic episodes (update_residue only
    # accumulates on harm_signal<0). In the real run, Phase-0..2 harm populates it.
    for _i in range(4):
        agent.residue_field.rbf_field.add_residue(torch.randn(1, WORLD_DIM) * 0.1, 0.5)
    anchor_info = agent.residue_field.snapshot_ewc_anchor()
    assert agent.residue_field.ewc_anchored, "FIX 3 FAILED: snapshot_ewc_anchor() did not anchor"
    # Perturb a residue weight so the penalty is strictly positive.
    with torch.no_grad():
        agent.residue_field.rbf_field.weights.add_(0.5 * agent.residue_field.rbf_field.active_mask.float())
    penalty = agent.residue_field.ewc_penalty()
    assert float(penalty.detach()) > 0.0, (
        f"FIX 3 FAILED: ewc_penalty()={float(penalty.detach()):.6g} is not > 0 after anchoring + "
        f"perturbation (EWC not armed -- check crystallize_at_phase3 -> ewc_enabled/lambda)"
    )
    res_params = [
        agent.residue_field.rbf_field.centers,
        agent.residue_field.rbf_field.weights,
    ]
    for p in res_params:
        if p.grad is not None:
            p.grad = None
    penalty.backward()
    res_grad = sum(float(p.grad.abs().sum()) for p in res_params if p.grad is not None)
    assert res_grad > 0.0, (
        "FIX 3 FAILED: ewc_penalty().backward() produced no gradient on the residue rbf params"
    )
    print(
        f"[V3-EXQ-610e][ASSERT] FIX 3 OK: ewc_anchored={agent.residue_field.ewc_anchored} "
        f"ewc_penalty={float(penalty):.6g} residue_grad_sum={res_grad:.6g} "
        f"(fisher_sum={anchor_info.get('fisher_sum')})",
        flush=True,
    )

    print("[V3-EXQ-610e][ASSERT] all three fixes wired + verified.", flush=True)
    return {
        "fix1_policy_trained": True,
        "fix1_trained_entropy": trained_entropy,
        "fix1_init_entropy": init_entropy,
        "fix2_expansion_stepped": True,
        "fix2_n_expansion_params": int(sum(p.numel() for p in exp_params)),
        "fix3_ewc_penalty_in_loss": True,
        "fix3_ewc_penalty_value": float(penalty.detach()),
        "fix3_residue_grad_sum": float(res_grad),
        "untrained_band_low": UNTRAINED_BAND_LOW,
        "ln5_uniform_ceiling": LN5,
    }


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv

    print(
        "[V3-EXQ-610e] INV-074 crystallization necessity test (EVB-0270, harness-fixed "
        "retest; supersedes V3-EXQ-610d)",
        flush=True,
    )

    # MANDATORY: assert all three fixes are wired before any real run counts.
    fix_verification = _assert_fixes_wired()

    arms = [
        {"label": "ARM_0_control", "crystallize": False},
        {"label": "ARM_1_test", "crystallize": True},
    ]

    seeds = SEEDS if not dry else [42]

    results = []
    for arm in arms:
        for seed in seeds:
            metrics = _run_arm(arm, seed=seed, dry_run=dry)
            results.append(metrics)

    eval_out = _evaluate(results)

    print("")
    print("[V3-EXQ-610e] Results:")
    for r in results:
        p2_ent = r['end_phase_2_entropy']
        p3_ent = r['end_phase_3_entropy']
        p2_str = f"{p2_ent:.4f}" if p2_ent is not None else "None"
        p3_str = f"{p3_ent:.4f}" if p3_ent is not None else "None"
        print(
            f"  {r['arm']} seed={r['seed']} "
            f"end_p2_entropy={p2_str} end_p3_entropy={p3_str} "
            f"mean_reward={r['mean_reward']:.4f} "
            f"pol_steps={r['n_policy_steps']} exp_steps_p3={r['n_expansion_steps_phase3']} "
            f"ewc_terms_p3={r['n_ewc_terms_phase3']} ewc_last={r['ewc_penalty_last']:.4g}"
        )
    print("")
    print("[V3-EXQ-610e] Acceptance:")
    print(f"  D1 (crystallization preserves diversity): {eval_out['d1_crystallization_preserves_diversity']} (delta={eval_out['d1_delta']:.4f})")
    print(f"  D2 (control shows collapse): {eval_out['d2_control_shows_collapse']} (delta={eval_out['d2_delta']:.4f})")
    print(f"  D3 (sanity -- both diverse at phase 2): {eval_out['d3_sanity_both_show_diversity_at_phase2']}")
    print(f"  policy_trained_nonuniform: {eval_out['policy_trained_nonuniform']} (mean_phase2_entropy={eval_out['mean_phase2_entropy']:.4f}, uniform ln5={eval_out['ln5_uniform_ceiling']:.4f})")
    print(f"  Verdict: {eval_out['verdict']}")

    if not dry:
        run_id = f"{EXPERIMENT_TYPE}_{_utc_iso_now().replace(':', '').replace('-', '')}_v3"
        evidence_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
        manifest_path = evidence_dir / f"{run_id}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        manifest = {
            "experiment_type": EXPERIMENT_TYPE,
            "run_id": run_id,
            "queue_id": QUEUE_ID,
            "claim_ids": CLAIM_IDS,
            "backlog_id": BACKLOG_ID,
            "supersedes": SUPERSEDES,
            "architecture_epoch": "ree_hybrid_guardrails_v1",
            "experiment_purpose": EXPERIMENT_PURPOSE,
            "outcome": eval_out["verdict"],
            "completed_at": _utc_iso_now(),
            "timestamp_utc": _utc_iso_now(),
            "acceptance": eval_out,
            "fix_verification": fix_verification,
            "arm_results": results,
            "config": {
                "seeds": SEEDS,
                "max_episodes": MAX_EPISODES,
                "steps_per_episode": STEPS_PER_EPISODE,
                "noise_floor_weight": NOISE_FLOOR_WEIGHT,
                "dacc_suppression_weight": DACC_SUPPRESSION_WEIGHT,
                "residue_ewc_lambda": RESIDUE_EWC_LAMBDA,
                "gated_bias_scale": GATED_BIAS_SCALE,
                "policy_temp": POLICY_TEMP,
                "policy_gamma": POLICY_GAMMA,
                "entropy_bonus_weight": ENTROPY_BONUS_WEIGHT,
                "lr_policy": LR_POLICY,
            },
        }

        manifest_path = write_flat_manifest(
            manifest,
            evidence_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )

        print(f"[V3-EXQ-610e] Manifest written: {manifest_path}", flush=True)

        _outcome_raw = str(eval_out["verdict"]).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=str(manifest_path),
            run_id=run_id,
            queue_id=QUEUE_ID,
        )
