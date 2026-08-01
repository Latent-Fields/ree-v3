#!/opt/local/bin/python3
"""
V3-EXQ-436b -- MECH-166 wall-independent representation confirmer
SLEEP DRIVER: manual-multi (run_sleep_cycle() called directly every SLEEP_INTERVAL episodes in training loop)

experiment_purpose: evidence

Discharges IGW-20260801-223 (claim MECH-166, backlog EVB-0475, "Proposal for
MECH-166") and IGW-20260801-225 (GOV-CONFIRM-1, "Confirm evidence: MECH-166,
lit 0.88 exp ~0"). Prior drafting pass (session 6ac06193-...-worktrees-best-
practices-3e2efc, RELATIONSHIP.md + DESIGN.md, 2026-08-01) determined these
are the SAME underlying evidence gap for MECH-166 (genuine_exp_count: 0
despite literature_confidence 0.876), not two separate needs -- see those
docs for the full derivation. This session (igw-loop-223225) re-verified the
draft's load-bearing claims against LIVE source before writing (below) and
found one that needed fixing, not just confirming.

SUPERSEDES: V3-EXQ-436 -> V3-EXQ-436a (both non_contributory, FAIL,
n_seeds_passed=0/5 on the conjunctive C1-AND-C2 gate). 436a's own
evidence_direction_note / INTERPRETATION GRID named the exact requeue
condition this design satisfies ("C1 FAIL, C2 FAIL... Compare to 265a
per-seed heterogeneity to assess whether the failure is methodological or
architectural" -> diagnosed root cause: monomodal, bit-identical waking
trajectories across seeds/conditions starved the SWS-analog consolidation
pass of anything to differentiate). ARC-065 (behavioral_diversity_generation_
pathway), the diagnosed fix, is now status: stable, v3_pending: False as of
2026-07-11 (claims.yaml, re-confirmed live 2026-08-01). No V3-EXQ-436b has
ever been queued or run (confirmed 2026-08-01: no match in
ree-v3/experiments/, ree-v3/experiment_queue.json, ree-v3/runner_status.json,
or REE_assembly/evidence/experiments/; reanalysis_query.py's one
slot_cosine_sim match, 436a itself, is UNVERIFIABLE -- no substrate_hash,
pre-recording-standard -- so GOV-REUSE-1 correctly routes to a fresh run,
not a reanalysis).

CLAIM SUBSTRATE UNDER TEST (all three read the SAME wall-independent DV;
claims.yaml's own "Experimental implication" text for SD-017/ARC-045 already
specifies this exact metric and threshold -- not a redesign invention):
  SD-017    (sleep_phase.minimal_sleep_infrastructure_v3): "context
            representations remain globally undifferentiated" without the
            SWS/REM-analog phases -- slot_cosine_sim -> 1.0 without them.
  ARC-045   (hippocampus.bidirectional_information_flow): "an agent with
            bidirectional offline flow should show cosine_sim < 0.95
            (differentiated contexts) after sleep phases; one with only
            waking online encoding remains at cosine_sim -> 1.0 regardless
            of training duration" (claims.yaml notes, verbatim experimental
            implication -- this IS the C1 test below).
  MECH-166  (hippocampus.slot_formation_filling_temporal_separation): "Slot
            structure must be consolidated during an SWS-analog phase...
            Experimental implication: EXQ-239 (MECH-153) provides an
            indirect test. A direct test requires implementing the SWS-
            analog pass and comparing attribution map quality (context
            cosine_sim...) with vs without it." This experiment IS that
            direct test.

WHAT CHANGES vs 436a (two changes; both load-bearing, both re-verified
against live source 2026-08-01, not trusted from the draft):

1. DV PRIMACY INVERTED (per IGW-225's explicit mandate for a WALL-
   INDEPENDENT DV). 436a's PASS required C1 (slot_cosine_sim, representation,
   read under torch.no_grad() off agent.e1.context_memory.memory -- already
   wall-independent, confirmed unchanged at ree_core/predictors/e1_deep.py
   ContextMemory) AND C2 (harm_rate_dangerous, behavioural, wall-bound)
   CONJUNCTIVELY -- so a wall-bound behavioural collapse could sink a
   representation-level result even if C1 alone had signal. This design
   makes slot_cosine_sim the SOLE PASS/FAIL gate (C1 below), decoupled from
   behavioural metrics, which are recorded generously as secondary/
   exploratory (never gating). C1 is DIRECTIONAL (not the 436a signed-|diff|
   shape) because ARC-045's own pre-registered experimental implication
   (above) specifies a direction, not "either direction informative" --
   436a's bidirectional shape was appropriate when C1 was one of two joint
   claims; here it is the sole confirmer of a claim that predicts a specific
   sign.

2. UPSTREAM DIVERSITY FIX WIRED IN -- AND WIRED AT THE CORRECT CALL SITE.
   436a's FAIL was attributed to a monomodal waking policy (bit-identical
   trajectories across seeds/conditions), starving the SWS-analog
   consolidation pass. ARC-065's substrate (ree_core/policy/noise_floor.py
   MECH-313 stochastic_noise_floor) is built but default-off
   (REEConfig.use_noise_floor: bool = False, confirmed live at
   ree_core/utils/config.py:3238; threaded through REEConfig.from_dims at
   lines 5777-5779/6994-6996). This design sets use_noise_floor=True (with
   default noise_floor_alpha=0.1, noise_floor_min_temperature=1.0) on every
   condition's agent.

   THE OPEN ITEM THE DRAFT FLAGGED, RE-VERIFIED AND FOUND REAL: noise_floor.py's
   own docstring states its integration site precisely -- "REEAgent.
   select_action() reads noise_floor.compute_effective_temperature(...)
   BEFORE calling e3.select(..., temperature=effective_T, ...)" -- confirmed
   live at ree_core/agent.py:7438-7444 (the tonic lift feeds e3.select's
   softmax-then-multinomial sampling, ree_core/predictors/e3_selector.py:
   3104-3105 `probs = F.softmax(-scores/temperature); torch.multinomial(...)`).
   436a's own action-selection helpers (_select_action_baseline,
   _select_action_context_cond) NEVER called agent.select_action() -- they
   looped over discrete actions calling agent.e2.world_forward +
   agent.e3.harm_eval directly and took a deterministic argmin/filtered-
   argmin. Simply flipping use_noise_floor=True while keeping that
   deterministic argmin path would have had ZERO effect on trajectory
   diversity (agent.noise_floor is constructed but never consulted) --
   this would have silently reproduced 436a's exact failure while looking
   like the fix had been applied. FIX (this script): the harm-scoring loop
   is unchanged (still the causal/context-conditioned-threshold mechanism
   under test, DR-6 pathway), but the final action pick is now a
   temperature-graded softmax sample -- probs = softmax(-harms /
   effective_T), action = multinomial(probs, 1) -- using the SAME
   agent.noise_floor.compute_effective_temperature(baseline_temperature=1.0,
   simulation_mode=False) call the substrate's own select_action() makes,
   applied to this driver's own harm-based candidate scores. See
   _effective_temperature() / _select_action_baseline() /
   _select_action_context_cond() below. A defensive
   `assert agent.noise_floor is not None` immediately after agent
   construction catches any future wiring regression loudly rather than
   silently re-degenerating to argmax.

DV-SYMMETRY INVARIANCE DECLARATION (mandatory per 604c net): the
manipulation under test is "ran an SWS-analog + REM-analog consolidation
pass" vs "did not" -- run_sleep_cycle() directly mutates
agent.e1.context_memory.memory via ContextMemory.write() during sleep, and
slot_cosine_sim reads that same memory tensor. This is not a uniform
additive constant, not a monotone rescaling, and not a permutation of
interchangeable units -- it is a genuinely different memory-writing
procedure whose effect the DV reads directly, so the DV is not invariant
under any of the three flagged symmetry classes and the manipulation can
legitimately move it.

CONDITIONS (2, reduced from 436a's 3): WAKING_ONLY (baseline, no sleep) and
SWS_THEN_REM (full SWS-then-REM cycle every SLEEP_INTERVAL episodes, plus
the DR-6 context-conditioned harm threshold in action selection). 436a's
third arm, SWS_ONLY, is dropped -- it isolates SWS-alone vs SWS+REM
contribution, which is not required to answer the primary MECH-166 question
(does an SWS-analog pass differentiate slots at all) and dropping it lets
the full 5-seed budget go to the two-arm comparison the DV-symmetry
declaration above is scoped to. use_noise_floor=True on BOTH conditions
(the diversity fix is a waking-phase substrate property, not part of the
manipulation being tested).

ACCEPTANCE CRITERIA:
  C1 (PRIMARY, SOLE GATE -- SD-017 + ARC-045 + MECH-166, wall-independent):
      slot_cosine_sim(SWS_THEN_REM) < slot_cosine_sim(WAKING_ONLY) in >= 3/5
      seeds. Directional per ARC-045's own pre-registered experimental
      implication (claims.yaml, quoted above): SWS-then-REM should
      differentiate (orthogonalize) slots relative to the waking-only
      attractor; ARC-045's "< 0.95" absolute-value framing is recorded as a
      descriptive secondary readout (arc045_differentiated_abs_sim, per
      seed) but the paired comparison to WAKING_ONLY is the pre-registered
      gate, since 0.95 is a rule-of-thumb reference point, not a threshold
      ARC-045's notes formally pin.
  C4 (SECONDARY, ARC-045 slot_separation, non-gating, carried from 436a):
      slot_separation(SWS_THEN_REM) > 0.3 in >= 3/5 seeds -- confirms slots
      correspond to the env's actual SAFE/DANGEROUS context structure, not
      an arbitrary partition.
  Secondary / exploratory (recorded, NEVER gating, per IGW-225's mandate to
  decouple behaviour from the representation confirmer):
      harm_rate_dangerous, harm_rate_safe signed diffs (436a's C2/C3). A
      behavioural PASS here is a bonus (upgrades toward SD-017/ARC-045 end-
      to-end support); a behavioural null alongside a representation PASS is
      still a genuine confirmatory result for MECH-166 specifically, which
      is a claim about slot-formation, not about downstream harm-threshold
      use (claims.yaml MECH-166 notes: "Relationship to MECH-165" already
      factors this apart).

PASS: C1 alone (n_seeds_passed >= 3/5).

INTERPRETATION GRID:
  C1 PASS  -> SWS-analog consolidation differentiates context slots once the
              waking-phase input-diversity confound (436a's diagnosed root
              cause) is removed. Supports SD-017, ARC-045, MECH-166.
              Discharges IGW-223/225; MECH-166 experimental_confidence moves
              off 0.0. C4 / behavioural readouts reported alongside as
              corroborating-or-not context, non-gating.
  C1 FAIL  -> Either (a) the noise-floor diversity fix is insufficient on its
              own to unstick slot differentiation (a genuine finding: input
              diversity was necessary-but-not-sufficient), or (b) MECH-166's
              slot-formation/filling separation does not hold as stated on
              this substrate. Check per-seed non_degenerate flag first: if
              waking-phase action-class entropy (recorded per seed) is still
              ~0 despite use_noise_floor=True, that is (a) and should route
              to a /failure-autopsy on the noise-floor magnitude (Q-043
              calibration is still open -- default alpha=0.1 may be too
              weak); if waking-phase entropy is healthy but slot_cosine_sim
              still does not differentiate, that is a genuine (b) weakens
              reading for all three claims. Either way this is real evidence
              (the run is not vacuous), not a wiring failure.

claim_ids: ["SD-017", "ARC-045", "MECH-166"]
experiment_purpose: "evidence"

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow
"""

import argparse
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_436b_sd017_mech166_repr_confirmer"
QUEUE_ID = "V3-EXQ-436b"
SUPERSEDES = "V3-EXQ-436a"
CLAIM_IDS = ["SD-017", "ARC-045", "MECH-166"]
EXPERIMENT_PURPOSE = "evidence"

# Pre-registered thresholds (unchanged from 436a unless noted).
BASE_HARM_THRESHOLD = 0.05       # filter actions whose predicted harm exceeds this
CONTEXT_BETA = 0.8                # danger-score modulation strength
SLOT_DANGER_EMA_ALPHA = 0.05      # slot_danger_score EMA update rate

# Phase 2 substrate template (validated by V3-EXQ-265a PASS 2026-05-09T20:12Z;
# reused verbatim from 436a -- unchanged, not part of this redesign).
SD016_DIVERSIFICATION_WEIGHT = 0.5

# MECH-313 / ARC-065 noise floor -- the load-bearing config change vs 436a.
USE_NOISE_FLOOR = True
NOISE_FLOOR_ALPHA = 0.1
NOISE_FLOOR_MIN_TEMPERATURE = 1.0
BASELINE_TEMPERATURE = 1.0        # matches REEAgent.select_action's own default

# Acceptance thresholds.
C1_N_SEEDS_REQUIRED = 3           # >= 3/5 seeds, sole PASS/FAIL gate
C4_SLOT_SEPARATION_THRESHOLD = 0.3
C4_N_SEEDS_REQUIRED = 3
ARC045_ABS_COSINE_REFERENCE = 0.95  # descriptive only; claims.yaml's own reference point

SLEEP_INTERVAL = 10
CONTEXT_SWITCH_EVERY = 5
TRAINING_EPISODES = 200
STEPS_PER_EPISODE = 150
EVAL_EPISODES_EACH = 30
MAX_HARM_BUF = 4000
SEEDS = [42, 7, 13, 100, 200]      # unchanged from 436/436a for cross-lineage comparability

CONDITIONS = ["WAKING_ONLY", "SWS_THEN_REM"]  # 436a's SWS_ONLY dropped (see docstring)


# ------------------------------------------------------------------ #
# Env / agent helpers                                                  #
# ------------------------------------------------------------------ #

def _make_env_safe(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=1,
        num_resources=4,
        hazard_harm=0.02,
        env_drift_interval=50,
        env_drift_prob=0.05,
        proximity_harm_scale=0.10,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_env_dangerous(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed + 1000,
        size=10,
        num_hazards=8,
        num_resources=4,
        hazard_harm=0.05,
        env_drift_interval=50,
        env_drift_prob=0.05,
        proximity_harm_scale=0.15,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2, sws_enabled: bool, rem_enabled: bool,
                use_sleep_loop: bool) -> REEAgent:
    """Phase 2 substrate stack (unchanged from 436a) + MECH-313/ARC-065 noise
    floor (the load-bearing addition -- see module docstring point 2).
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
        # Phase 2 substrate template (5 flags, mechanically applied; constant
        # across both conditions in this experiment).
        sd016_writepath_mode="off",
        sd016_diversification_weight=SD016_DIVERSIFICATION_WEIGHT,
        use_per_stream_vs=True,
        use_anchor_sets=True,
        use_sd039_anchor_payload=True,
        # SD-017 sleep phases (toggle per condition).
        sws_enabled=sws_enabled,
        sws_consolidation_steps=8,
        sws_schema_weight=0.1,
        rem_enabled=rem_enabled,
        rem_attribution_steps=6,
        use_sleep_loop=use_sleep_loop,
        # MECH-313 / ARC-065 stochastic noise floor -- ON in every condition
        # (the diversity fix is a waking-phase property, not the manipulation
        # under test).
        use_noise_floor=USE_NOISE_FLOOR,
        noise_floor_alpha=NOISE_FLOOR_ALPHA,
        noise_floor_min_temperature=NOISE_FLOOR_MIN_TEMPERATURE,
    )
    agent = REEAgent(cfg)
    assert agent.noise_floor is not None, (
        "use_noise_floor=True did not construct agent.noise_floor -- "
        "REEConfig/REEAgent wiring regression; the diversity fix this "
        "experiment depends on is not live. See module docstring point 2."
    )
    return agent


def _action_onehot(a_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, a_idx] = 1.0
    return v


def _effective_temperature(agent: REEAgent) -> float:
    """The same tonic-lift computation REEAgent.select_action() applies
    before e3.select() (ree_core/agent.py:7438-7444), applied here so this
    driver's own harm-based action scores get the same MECH-313 noise-floor
    diversity injection the substrate's own selection path would give them.
    """
    if agent.noise_floor is not None:
        return agent.noise_floor.compute_effective_temperature(
            baseline_temperature=BASELINE_TEMPERATURE, simulation_mode=False,
        )
    return BASELINE_TEMPERATURE


# ------------------------------------------------------------------ #
# Context-slot detection (unchanged from 436a -- re-verified live against  #
# ree_core/predictors/e1_deep.py ContextMemory, 2026-08-01)                #
# ------------------------------------------------------------------ #

def _active_slot_idx(agent: REEAgent, z_self: torch.Tensor,
                     z_world: torch.Tensor) -> int:
    """Determine which ContextMemory slot is most strongly activated by
    (z_self, z_world). Argmax over slots vs ContextMemory.read()'s soft mix.
    """
    with torch.no_grad():
        cm = agent.e1.context_memory
        state = torch.cat([z_self, z_world], dim=-1)
        query = cm.query_proj(state)
        keys = cm.key_proj(cm.memory)
        scores = torch.mm(query, keys.t()) / (cm.memory_dim ** 0.5)
        idx = int(scores.argmax(dim=-1).item())
    return idx


def _compute_slot_cosine_sim(agent: REEAgent) -> float:
    with torch.no_grad():
        mem = agent.e1.context_memory.memory
        n = mem.shape[0]
        normed = F.normalize(mem, dim=-1)
        sim = torch.mm(normed, normed.t())
        mask = ~torch.eye(n, dtype=torch.bool, device=mem.device)
        return float(sim[mask].mean().item())


# ------------------------------------------------------------------ #
# Action selection -- STOCHASTIC (noise-floor temperature-graded sample,   #
# replacing 436a's deterministic argmin -- see module docstring point 2)   #
# ------------------------------------------------------------------ #

def _select_action_baseline(agent: REEAgent, z_world: torch.Tensor,
                             num_actions: int) -> Tuple[int, float]:
    """Temperature-graded softmax sample over predicted harm (low harm ->
    high selection probability), using the MECH-313 noise-floor effective
    temperature. Replaces 436a's argmin (which never varied across seeds).
    """
    with torch.no_grad():
        harms: List[float] = []
        for a in range(num_actions):
            a_oh = _action_onehot(a, num_actions, z_world.device)
            zw_next = agent.e2.world_forward(z_world, a_oh)
            h = agent.e3.harm_eval(zw_next).mean().item()
            harms.append(h)
        eff_t = _effective_temperature(agent)
        harms_t = torch.tensor(harms, dtype=torch.float32)
        probs = F.softmax(-harms_t / eff_t, dim=0)
        best_a = int(torch.multinomial(probs, 1).item())
        best_h = harms[best_a]
    return best_a, best_h


def _select_action_context_cond(agent: REEAgent, z_world: torch.Tensor,
                                 num_actions: int, slot_danger_score: float,
                                 base_thresh: float, context_beta: float
                                 ) -> Tuple[int, float, float]:
    """Context-conditioned harm threshold action selection (DR-6 pathway),
    unchanged causal structure from 436a: effective threshold =
    base_thresh * (1 - context_beta * slot_danger_score); higher danger ->
    lower threshold -> more candidates filtered -> more cautious. Selection
    WITHIN the filtered (or, on empty filter, the full) candidate set is now
    a noise-floor temperature-graded softmax sample rather than argmin --
    same fix as _select_action_baseline above, applied on both branches so
    the fallback path also carries the diversity injection.
    Returns (action_idx, chosen_harm, effective_threshold).
    """
    eff_thresh = base_thresh * max(0.1, 1.0 - context_beta * slot_danger_score)
    with torch.no_grad():
        harms: List[float] = []
        for a in range(num_actions):
            a_oh = _action_onehot(a, num_actions, z_world.device)
            zw_next = agent.e2.world_forward(z_world, a_oh)
            h = agent.e3.harm_eval(zw_next).mean().item()
            harms.append(h)
        eff_t = _effective_temperature(agent)
        filtered_idx = [a for a, h in enumerate(harms) if h <= eff_thresh]
        if filtered_idx:
            sub_harms = torch.tensor([harms[a] for a in filtered_idx], dtype=torch.float32)
            probs = F.softmax(-sub_harms / eff_t, dim=0)
            sel = int(torch.multinomial(probs, 1).item())
            best_a = filtered_idx[sel]
        else:
            harms_t = torch.tensor(harms, dtype=torch.float32)
            probs = F.softmax(-harms_t / eff_t, dim=0)
            best_a = int(torch.multinomial(probs, 1).item())
        best_h = harms[best_a]
    return best_a, float(best_h), float(eff_thresh)


# ------------------------------------------------------------------ #
# Episode runner (unchanged control flow from 436a; action selection calls #
# now route through the stochastic functions above)                        #
# ------------------------------------------------------------------ #

def _run_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    steps: int,
    train: bool,
    is_dangerous_ep: bool,
    optimizer,
    harm_eval_opt,
    harm_buf_pos: List,
    harm_buf_neg: List,
    slot_danger_ema: List[float],
    use_context_cond: bool,
) -> Tuple[float, List[torch.Tensor], List[int], List[int]]:
    """Run single episode. Returns (harm_sum, z_world_list, slot_visits,
    action_seq). action_seq is recorded to compute a per-episode action-class
    entropy diagnostic (evidence the noise-floor fix is actually engaging,
    independent of the slot_cosine_sim readout itself).
    Updates slot_danger_ema in place when train=True.
    """
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    ep_harm = 0.0
    z_world_list: List[torch.Tensor] = []
    slot_visits: List[int] = []
    action_seq: List[int] = []

    for _step in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        obs_harm = obs_dict.get("harm_obs", None)

        latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
        agent.clock.advance()
        z_self = latent.z_self.detach().clone()
        z_world = latent.z_world.detach().clone()
        z_world_list.append(z_world)

        slot_idx = _active_slot_idx(agent, z_self, z_world)
        slot_visits.append(slot_idx)

        if use_context_cond:
            danger = slot_danger_ema[slot_idx]
            action_idx, _, _ = _select_action_context_cond(
                agent, z_world, env.action_dim, danger,
                BASE_HARM_THRESHOLD, CONTEXT_BETA,
            )
        else:
            action_idx, _ = _select_action_baseline(agent, z_world, env.action_dim)
        action_seq.append(action_idx)

        action_oh = _action_onehot(action_idx, env.action_dim, agent.device)
        agent._last_action = action_oh

        _, harm_signal, done, _info, obs_dict = env.step(action_oh)
        is_harm = float(harm_signal) < 0
        if is_harm:
            ep_harm += abs(float(harm_signal))

        if train:
            target = 1.0 if is_dangerous_ep else 0.0
            slot_danger_ema[slot_idx] = (
                (1.0 - SLOT_DANGER_EMA_ALPHA) * slot_danger_ema[slot_idx]
                + SLOT_DANGER_EMA_ALPHA * target
            )

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total = e1_loss + e2_loss
            if total.requires_grad:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if is_harm:
                harm_buf_pos.append(z_world)
            else:
                harm_buf_neg.append(z_world)

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b = torch.cat([zw_pos, zw_neg], dim=0)
                target_t = torch.cat([
                    torch.ones(k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred = agent.e3.harm_eval_head(zw_b)
                h_loss = F.binary_cross_entropy_with_logits(pred, target_t)
                harm_eval_opt.zero_grad()
                h_loss.backward()
                harm_eval_opt.step()

        if done:
            break

    return ep_harm, z_world_list, slot_visits, action_seq


def _action_class_entropy(action_seq: List[int], num_actions: int) -> float:
    """Shannon entropy (nats) of the realized action-class distribution over
    one episode. 0.0 = fully degenerate (one action class every tick, the
    436a failure signature); ln(num_actions) = uniform. Diagnostic only --
    not gating -- but decisive for reading a C1 FAIL (see INTERPRETATION
    GRID): near-zero entropy despite use_noise_floor=True would mean the
    diversity fix did not engage, not that MECH-166 is refuted.
    """
    if not action_seq:
        return 0.0
    counts = [0] * num_actions
    for a in action_seq:
        counts[a % num_actions] += 1
    total = float(len(action_seq))
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log(p)
    return float(ent)


# ------------------------------------------------------------------ #
# Condition runner                                                     #
# ------------------------------------------------------------------ #

def _run_condition(
    seed: int,
    condition: str,
    training_episodes: int,
    steps_per_episode: int,
    eval_episodes_each: int,
    zg: ZGoalStreamAccumulator,
    verbose: bool = True,
) -> Dict:
    sws_en = condition == "SWS_THEN_REM"
    rem_en = condition == "SWS_THEN_REM"
    use_sleep_loop = sws_en or rem_en  # ON for SWS_THEN_REM; OFF for WAKING_ONLY
    use_context_cond = condition == "SWS_THEN_REM"   # DR-6 pathway only here

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)
    agent = _make_agent(env_safe, sws_en, rem_en, use_sleep_loop)

    standard_params = [p for n, p in agent.named_parameters()
                       if "harm_eval_head" not in n]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=1e-3)
    harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    num_slots = agent.e1.context_memory.num_slots
    slot_danger_ema: List[float] = [0.5] * num_slots

    per_ep_harm_safe: List[float] = []
    per_ep_harm_dang: List[float] = []
    per_ep_action_entropy: List[float] = []
    slot_visit_safe_count: List[int] = [0] * num_slots
    slot_visit_dang_count: List[int] = [0] * num_slots
    sleep_passes = 0
    cum_train_pos = 0  # cumulative harm_eval_head TRAINING label counts
    cum_train_neg = 0  # (pre-MAX_HARM_BUF-trim, so not diluted by the cap)

    agent.train()
    for ep in range(training_episodes):
        block = ep // CONTEXT_SWITCH_EVERY
        is_safe_ep = (block % 2 == 0)
        env = env_safe if is_safe_ep else env_dang

        _len_pos_before, _len_neg_before = len(harm_buf_pos), len(harm_buf_neg)
        ep_harm, _z_list, slot_visits, action_seq = _run_episode(
            agent, env, steps_per_episode,
            train=True,
            is_dangerous_ep=(not is_safe_ep),
            optimizer=optimizer,
            harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos,
            harm_buf_neg=harm_buf_neg,
            slot_danger_ema=slot_danger_ema,
            use_context_cond=use_context_cond,
        )
        harm_rate = ep_harm / steps_per_episode
        per_ep_action_entropy.append(_action_class_entropy(action_seq, env.action_dim))
        cum_train_pos += len(harm_buf_pos) - _len_pos_before
        cum_train_neg += len(harm_buf_neg) - _len_neg_before
        if is_safe_ep:
            per_ep_harm_safe.append(harm_rate)
            for s in slot_visits:
                slot_visit_safe_count[s] += 1
        else:
            per_ep_harm_dang.append(harm_rate)
            for s in slot_visits:
                slot_visit_dang_count[s] += 1

        if len(harm_buf_pos) > MAX_HARM_BUF:
            harm_buf_pos = harm_buf_pos[-MAX_HARM_BUF:]
        if len(harm_buf_neg) > MAX_HARM_BUF:
            harm_buf_neg = harm_buf_neg[-MAX_HARM_BUF:]

        if (sws_en or rem_en) and (ep + 1) % SLEEP_INTERVAL == 0 and ep > 0:
            if rem_en:
                _ = agent.run_sleep_cycle()
            else:
                _ = agent.run_sws_schema_pass()
            sleep_passes += 1

        if (ep + 1) % 50 == 0:
            print(f"  [train] label seed={seed} cond={condition} "
                  f"ep {ep+1}/{training_episodes} "
                  f"harm_safe_ema={(sum(per_ep_harm_safe[-10:])/max(len(per_ep_harm_safe[-10:]),1)):.4f} "
                  f"harm_dang_ema={(sum(per_ep_harm_dang[-10:])/max(len(per_ep_harm_dang[-10:]),1)):.4f} "
                  f"action_entropy_ema={(sum(per_ep_action_entropy[-10:])/max(len(per_ep_action_entropy[-10:]),1)):.4f}",
                  flush=True)

    safe_tot = float(sum(slot_visit_safe_count))
    dang_tot = float(sum(slot_visit_dang_count))
    if safe_tot > 0 and dang_tot > 0:
        safe_dist = [c / safe_tot for c in slot_visit_safe_count]
        dang_dist = [c / dang_tot for c in slot_visit_dang_count]
        slot_separation = float(sum(abs(s - d) for s, d in zip(safe_dist, dang_dist)))
    else:
        slot_separation = 0.0

    final_slot_sim = _compute_slot_cosine_sim(agent)
    train_action_entropy_mean = (
        sum(per_ep_action_entropy) / max(1, len(per_ep_action_entropy))
    )

    zg.observe(agent)

    agent.eval()
    eval_harm_safe: List[float] = []
    eval_harm_dang: List[float] = []
    eval_z_safe: List[torch.Tensor] = []
    eval_z_dang: List[torch.Tensor] = []

    for _ in range(eval_episodes_each):
        h_s, zs, _, _ = _run_episode(
            agent, env_safe, steps_per_episode,
            train=False, is_dangerous_ep=False,
            optimizer=optimizer, harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos, harm_buf_neg=harm_buf_neg,
            slot_danger_ema=slot_danger_ema,
            use_context_cond=use_context_cond,
        )
        eval_harm_safe.append(h_s / steps_per_episode)
        eval_z_safe.extend(zs)

    for _ in range(eval_episodes_each):
        h_d, zd, _, _ = _run_episode(
            agent, env_dang, steps_per_episode,
            train=False, is_dangerous_ep=True,
            optimizer=optimizer, harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos, harm_buf_neg=harm_buf_neg,
            slot_danger_ema=slot_danger_ema,
            use_context_cond=use_context_cond,
        )
        eval_harm_dang.append(h_d / steps_per_episode)
        eval_z_dang.extend(zd)

    with torch.no_grad():
        n_samp = min(len(eval_z_safe), len(eval_z_dang), 200)
        if n_samp > 0:
            zs_s = torch.cat(eval_z_safe[:n_samp], dim=0)
            zd_s = torch.cat(eval_z_dang[:n_samp], dim=0)
            he_safe = float(agent.e3.harm_eval(zs_s).mean().item())
            he_dang = float(agent.e3.harm_eval(zd_s).mean().item())
        else:
            he_safe = 0.0
            he_dang = 0.0
    harm_discrim = he_dang - he_safe

    harm_safe = sum(eval_harm_safe) / max(1, len(eval_harm_safe))
    harm_dang = sum(eval_harm_dang) / max(1, len(eval_harm_dang))

    if verbose:
        print(f"  [seed={seed} {condition}] "
              f"slot_sim={final_slot_sim:.4f} "
              f"slot_sep={slot_separation:.3f} "
              f"harm_safe={harm_safe:.4f} "
              f"harm_dang={harm_dang:.4f} "
              f"discrim={harm_discrim:.4f} "
              f"action_entropy_mean={train_action_entropy_mean:.4f} "
              f"sleep_passes={sleep_passes}",
              flush=True)

    # Per-condition verdict (progress-instrumentation / runner-ETA purposes
    # only, matches 436a's convention; the experiment-level PASS/FAIL is the
    # aggregate C1 gate computed once across all seeds in __main__).
    verdict = "PASS" if (harm_dang < 0.04 and harm_safe < 0.04) else "FAIL"
    print(f"verdict: {verdict}", flush=True)

    return {
        "seed": seed,
        "condition": condition,
        "slot_cosine_sim": float(final_slot_sim),
        "slot_separation": float(slot_separation),
        "harm_rate_safe": float(harm_safe),
        "harm_rate_dangerous": float(harm_dang),
        "harm_discrimination": float(harm_discrim),
        "harm_eval_safe": float(he_safe),
        "harm_eval_dangerous": float(he_dang),
        "slot_danger_ema": [float(x) for x in slot_danger_ema],
        "slot_visit_safe_count": slot_visit_safe_count,
        "slot_visit_dang_count": slot_visit_dang_count,
        "train_harm_safe_final": float(sum(per_ep_harm_safe[-20:]) / max(1, len(per_ep_harm_safe[-20:]))),
        "train_harm_dang_final": float(sum(per_ep_harm_dang[-20:]) / max(1, len(per_ep_harm_dang[-20:]))),
        "train_action_class_entropy_mean": float(train_action_entropy_mean),
        "sleep_passes": sleep_passes,
        "effective_temperature_last": float(_effective_temperature(agent)),
        "noise_floor_state": agent.noise_floor.get_state() if agent.noise_floor is not None else None,
        "label_balance": {
            "harm_eval_head_train_pos_frac": (
                float(cum_train_pos) / max(1, cum_train_pos + cum_train_neg)
            ),
            "harm_eval_head_train_n_pos": cum_train_pos,
            "harm_eval_head_train_n_neg": cum_train_neg,
        },
    }


# ------------------------------------------------------------------ #
# Run                                                                   #
# ------------------------------------------------------------------ #

def run(dry_run: bool = False) -> Tuple[dict, ZGoalStreamAccumulator]:
    zg = ZGoalStreamAccumulator()

    if dry_run:
        print("[DRY RUN] MECH-166 repr confirmer smoke "
              "(seed=42, 2 conditions, 3 train eps each)", flush=True)
        smoke_ok = True
        smoke_results = []
        try:
            for cond in CONDITIONS:
                print(f"Seed 42 Condition {cond}", flush=True)
                config_slice = {
                    "seed": 42, "condition": cond,
                    "training_episodes": 3, "steps_per_episode": 30,
                    "eval_episodes_each": 2,
                }
                with arm_cell(42, config_slice=config_slice, script_path=Path(__file__)) as cell:
                    r = _run_condition(
                        seed=42, condition=cond,
                        training_episodes=3,
                        steps_per_episode=30,
                        eval_episodes_each=2,
                        zg=zg,
                        verbose=False,
                    )
                    cell.stamp(r)
                smoke_results.append(r)
                print(f"  {cond}: slot_sim={r['slot_cosine_sim']:.4f} "
                      f"slot_sep={r['slot_separation']:.3f} "
                      f"harm_safe={r['harm_rate_safe']:.4f} "
                      f"harm_dang={r['harm_rate_dangerous']:.4f} "
                      f"action_entropy={r['train_action_class_entropy_mean']:.4f} "
                      f"sleep_passes={r['sleep_passes']}")

            required_keys = {"slot_cosine_sim", "slot_separation",
                             "harm_rate_safe", "harm_rate_dangerous",
                             "train_action_class_entropy_mean", "arm_fingerprint"}
            for r in smoke_results:
                missing = required_keys - set(r.keys())
                if missing:
                    print(f"  [SMOKE] FAIL: condition {r['condition']} missing keys {missing}")
                    smoke_ok = False

            if smoke_ok:
                print("[DRY RUN] PASS - noise-floor stochastic selection + both "
                      "conditions wire correctly; per-seed metrics populate as "
                      "expected")
            else:
                print("[DRY RUN] FAIL - check above for missing metric keys")
        except Exception as exc:
            print(f"[DRY RUN] FAIL - exception during smoke: {exc!r}")
            smoke_ok = False

        return {
            "outcome": "PASS" if smoke_ok else "FAIL",
            "status": "PASS" if smoke_ok else "FAIL",
        }, zg

    t0 = time.time()
    print(f"{QUEUE_ID} {EXPERIMENT_TYPE}", flush=True)

    arm_results: List[Dict] = []
    for seed in SEEDS:
        print(f"Seed {seed}")
        for cond in CONDITIONS:
            print(f"Seed {seed} Condition {cond}")
            config_slice = {
                "seed": seed, "condition": cond,
                "training_episodes": TRAINING_EPISODES,
                "steps_per_episode": STEPS_PER_EPISODE,
                "eval_episodes_each": EVAL_EPISODES_EACH,
                "use_noise_floor": USE_NOISE_FLOOR,
                "noise_floor_alpha": NOISE_FLOOR_ALPHA,
                "noise_floor_min_temperature": NOISE_FLOOR_MIN_TEMPERATURE,
                "sd016_diversification_weight": SD016_DIVERSIFICATION_WEIGHT,
            }
            with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
                r = _run_condition(
                    seed=seed, condition=cond,
                    training_episodes=TRAINING_EPISODES,
                    steps_per_episode=STEPS_PER_EPISODE,
                    eval_episodes_each=EVAL_EPISODES_EACH,
                    zg=zg,
                )
                cell.stamp(r)
            arm_results.append(r)

    elapsed = time.time() - t0

    def by_cond(c):
        return [r for r in arm_results if r["condition"] == c]

    waking = by_cond("WAKING_ONLY")
    sws_r = by_cond("SWS_THEN_REM")

    per_seed_diff: Dict[str, Dict] = {}
    for w_r, s_r in zip(waking, sws_r):
        seed = w_r["seed"]
        slot_sim_diff = s_r["slot_cosine_sim"] - w_r["slot_cosine_sim"]
        harm_dang_diff = s_r["harm_rate_dangerous"] - w_r["harm_rate_dangerous"]
        harm_safe_diff = s_r["harm_rate_safe"] - w_r["harm_rate_safe"]
        slot_sep_diff = s_r["slot_separation"] - w_r["slot_separation"]
        per_seed_diff[str(seed)] = {
            "seed": seed,
            "waking_slot_cosine_sim": w_r["slot_cosine_sim"],
            "sws_then_rem_slot_cosine_sim": s_r["slot_cosine_sim"],
            "slot_cosine_sim_signed_diff": slot_sim_diff,
            # C1: PRIMARY GATE. Directional (per ARC-045's own experimental
            # implication) -- SWS_THEN_REM strictly lower than WAKING_ONLY.
            "slot_cosine_sim_passes_C1": s_r["slot_cosine_sim"] < w_r["slot_cosine_sim"],
            "arc045_sws_then_rem_below_reference": s_r["slot_cosine_sim"] < ARC045_ABS_COSINE_REFERENCE,
            "waking_harm_rate_dangerous": w_r["harm_rate_dangerous"],
            "sws_then_rem_harm_rate_dangerous": s_r["harm_rate_dangerous"],
            "harm_rate_dangerous_signed_diff": harm_dang_diff,
            "waking_harm_rate_safe": w_r["harm_rate_safe"],
            "sws_then_rem_harm_rate_safe": s_r["harm_rate_safe"],
            "harm_rate_safe_signed_diff": harm_safe_diff,
            "waking_slot_separation": w_r["slot_separation"],
            "sws_then_rem_slot_separation": s_r["slot_separation"],
            "slot_separation_signed_diff": slot_sep_diff,
            "sws_then_rem_slot_separation_passes_C4": s_r["slot_separation"] > C4_SLOT_SEPARATION_THRESHOLD,
            "waking_action_class_entropy": w_r["train_action_class_entropy_mean"],
            "sws_then_rem_action_class_entropy": s_r["train_action_class_entropy_mean"],
        }

    c1_count = sum(1 for d in per_seed_diff.values() if d["slot_cosine_sim_passes_C1"])
    c1_pass = c1_count >= C1_N_SEEDS_REQUIRED

    c4_count = sum(1 for d in per_seed_diff.values() if d["sws_then_rem_slot_separation_passes_C4"])
    c4_pass = c4_count >= C4_N_SEEDS_REQUIRED

    # PASS = C1 alone. C4 and the harm-rate diffs are recorded but never gate.
    outcome = "PASS" if c1_pass else "FAIL"

    def _direction(passed: bool) -> str:
        return "supports" if passed else "weakens"

    # All three claims share the SAME pre-registered wall-independent test
    # (claims.yaml's own "Experimental implication" text for SD-017 and
    # ARC-045 both specify slot_cosine_sim differentiation after sleep; the
    # MECH-166 notes describe this exact comparison as "the direct test").
    evidence_direction_per_claim = {
        "SD-017": _direction(c1_pass),
        "ARC-045": _direction(c1_pass),
        "MECH-166": _direction(c1_pass),
    }
    evidence_direction = _direction(c1_pass)

    # Non-degeneracy self-report: guard against a silent repeat of the
    # original 436/436a failure mode (bit-identical trajectories -> zero
    # spread on the load-bearing metric) despite use_noise_floor=True.
    all_cosine_values = [r["slot_cosine_sim"] for r in arm_results]
    all_action_entropy = [r["train_action_class_entropy_mean"] for r in arm_results]
    degeneracy = check_degeneracy({
        "slot_cosine_sim": all_cosine_values,
        "train_action_class_entropy_mean": {"values": all_action_entropy, "floor": 1e-6},
    })

    summary = {
        "C1_primary_slot_cosine_sim_directional": {
            "n_seeds_required": C1_N_SEEDS_REQUIRED,
            "n_seeds_passed": c1_count,
            "pass": c1_pass,
            "desc": ("SOLE PASS/FAIL GATE. slot_cosine_sim(SWS_THEN_REM) < "
                     "slot_cosine_sim(WAKING_ONLY) in >= 3/5 seeds. "
                     "Wall-independent (read under torch.no_grad() off "
                     "agent.e1.context_memory.memory)."),
        },
        "C4_arc045_slot_separation_threshold": {
            "threshold": C4_SLOT_SEPARATION_THRESHOLD,
            "n_seeds_required": C4_N_SEEDS_REQUIRED,
            "n_seeds_passed": c4_count,
            "pass": c4_pass,
            "desc": ("SECONDARY, non-gating. slot_separation in SWS_THEN_REM "
                     "> 0.3 in >= 3/5 seeds."),
        },
    }

    print(f"\nOutcome: {outcome}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  Per-claim direction: {evidence_direction_per_claim}")
    print(f"  Non-degenerate: {degeneracy.get('non_degenerate')} "
          f"({degeneracy.get('degeneracy_reason', '')})")

    result = {
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "claim_ids": CLAIM_IDS,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "status": outcome,
        "result": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "pass_criteria_summary": summary,
        "aggregated": {
            "per_seed": per_seed_diff,
            "n_seeds_passed": {"C1": c1_count, "C4": c4_count},
        },
        "arm_results": arm_results,
        "per_seed_results": arm_results,
        "registered_thresholds": {
            "C1_N_SEEDS_REQUIRED": C1_N_SEEDS_REQUIRED,
            "C4_SLOT_SEPARATION_THRESHOLD": C4_SLOT_SEPARATION_THRESHOLD,
            "C4_N_SEEDS_REQUIRED": C4_N_SEEDS_REQUIRED,
            "ARC045_ABS_COSINE_REFERENCE": ARC045_ABS_COSINE_REFERENCE,
            "BASE_HARM_THRESHOLD": BASE_HARM_THRESHOLD,
            "CONTEXT_BETA": CONTEXT_BETA,
            "SLOT_DANGER_EMA_ALPHA": SLOT_DANGER_EMA_ALPHA,
            "USE_NOISE_FLOOR": USE_NOISE_FLOOR,
            "NOISE_FLOOR_ALPHA": NOISE_FLOOR_ALPHA,
            "NOISE_FLOOR_MIN_TEMPERATURE": NOISE_FLOOR_MIN_TEMPERATURE,
            "BASELINE_TEMPERATURE": BASELINE_TEMPERATURE,
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
        "elapsed_seconds": elapsed,
        "notes": (
            "MECH-166 wall-independent representation confirmer. Discharges "
            "IGW-20260801-223 (EVB-0475) and IGW-20260801-225 (GOV-CONFIRM-1). "
            "Successor to V3-EXQ-436/436a: slot_cosine_sim is now the SOLE "
            "PASS/FAIL gate (decoupled from behavioural metrics per IGW-225's "
            "mandate), and MECH-313/ARC-065 noise-floor diversity is wired "
            "into this driver's own action-selection loop (agent.noise_floor."
            "compute_effective_temperature -> softmax(-harms/T) -> "
            "multinomial), not merely enabled on the config -- 436a's "
            "deterministic argmin never consulted it, which is confirmed to "
            "be the actual reason the original run saw zero cross-seed "
            "variation. 2 conditions (WAKING_ONLY, SWS_THEN_REM); 436a's "
            "third SWS_ONLY arm dropped as not required for the primary "
            "MECH-166 question."
        ),
    }
    return result, zg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result, zg_accumulator = run(dry_run=args.dry_run)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["timestamp_utc"] = ts
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config={
            "conditions": CONDITIONS,
            "sleep_interval": SLEEP_INTERVAL,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "training_episodes": TRAINING_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "eval_episodes_each": EVAL_EPISODES_EACH,
            "seeds": SEEDS,
            "env_safe_num_hazards": 1,
            "env_dangerous_num_hazards": 8,
            "sd016_writepath_mode": "off",
            "sd016_diversification_weight": SD016_DIVERSIFICATION_WEIGHT,
            "use_per_stream_vs": True,
            "use_anchor_sets": True,
            "use_sd039_anchor_payload": True,
            "use_sleep_loop_in_sleep_arms": True,
            "use_noise_floor": USE_NOISE_FLOOR,
            "noise_floor_alpha": NOISE_FLOOR_ALPHA,
            "noise_floor_min_temperature": NOISE_FLOOR_MIN_TEMPERATURE,
        },
        seeds=SEEDS,
        script_path=__file__,
        started_at=t0,
        z_goal_stream_stats=zg_accumulator.stats(),
    )
    print(f"Output written to: {out_path}", flush=True)

    _outcome_clean = str(result.get("outcome", "FAIL")).upper()
    if _outcome_clean not in ("PASS", "FAIL"):
        _outcome_clean = "FAIL"
    emit_outcome(
        outcome=_outcome_clean,
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
