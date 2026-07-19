#!/opt/local/bin/python3
"""V3-EXQ-485i: SD-033b trained-OFC-head BEHAVIOURAL validation (MECH-263 a+b) ON
the MECH-448 (ARC-107) rank-preserving F->eligibility DEMOTION-ENABLED E3 selector.

commitment_closure:GAP-8 evidence-grade follow-on. SUPERSEDES V3-EXQ-485h, the
DISAMBIGUATING REDESIGN that RAN FAIL / non_contributory 2026-06-19 -- the
MECH-439 F-dominance conversion-ceiling signature: a genuinely-trained OFC head
produced a real cross-candidate bias RANGE (~0.50) with ZERO behavioural
conversion (devaluation_selection_shift ~0, between-context TV ~0).

WHY NOW (the ceiling 485h was stuck under is operationally LIFTED):
  The 2026-06-21 governance cycle promoted MECH-448 to provisional on the
  V3-EXQ-689d PASS (rank-preserving F->eligibility demotion converts committed-
  action diversity; all four load-bearing criteria met). MECH-448 demotes the
  primary harm/goal score F to ELIGIBILITY only (a graded, rank-preserving
  divisive-normalisation envelope) and lets the modulatory channel arbitrate the
  COMMITTED action WITHIN the F-eligible set with F REMOVED from the final argmin.
  This is the shared F-dominance conversion ceiling that 485h's behavioural DVs
  were drowned under: the trained OFC valuation reached the E3 accumulator but the
  F-monopolised argmin (V3-EXQ-571: F ~88-89%% of committed-selection variance)
  never moved. The substrate-ceiling audit (scripts/check_substrate_ceiling_audit.py)
  flags SD-033b + MECH-263 as "ceiling-may-have-lifted".

THE 485i QUESTION: routed through the demotion-enabled selector, does the trained
  OFC outcome-value channel CONVERT to committed-action change under devaluation
  (C1) and across task-role contexts (C2)? A PASS confirms the demotion lever
  converts the OFC valuation channel -> contributes to closing
  behavioral_diversity_isolation:GAP-I and takes SD-033b candidate -> provisional
  behavioural evidence.

THE KEY HARNESS CHANGE vs 485h: the PRIMARY DVs (names UNCHANGED:
  C1 devaluation_selection_shift, C2 between-context TV) are re-targeted from the
  485h ISOLATED OFC softmax (softmax(-ofc_bias), where F is not in the loop, so the
  demotion lever could not act) to the COMMITTED selection through the REAL
  E3.select() (SelectionResult.selected_index) with the OFC bias passed as
  score_bias. That is where F dominates and where MECH-448 demotes it -- "behavioural
  conversion of the trained OFC bias on the demotion-enabled selector". The eval is
  made deterministic by forcing the committed (argmin-within-eligible) path
  (EVAL_COMMIT_THRESHOLD), so each DV is the genuine committed action of the selector.

THREE ARMS (a clean dissociation that attributes conversion to the CONJUNCTION
trained-head AND demotion -- not to either alone):
  ARM_0_frozen_demotion_on   : frozen zeroed OFC head + demotion ON. Silence
                               control -- the demotion selector with an UNTRAINED
                               (~0) valuation channel converts nothing (C3).
  ARM_1_trained_demotion_off : trained OFC head + demotion OFF. The F-dominance
                               CEILING control -- the trained valuation reaches the
                               accumulator but F dominates the argmin -> ~0 conversion.
  ARM_2_trained_demotion_on  : trained OFC head + demotion ON. The TEST (ARM_ON) --
                               trained valuation + F demoted to eligibility -> the
                               OFC channel arbitrates within the F-eligible set.
  All three pass the OFC bias as score_bias and use the SAME settle/eval pipeline;
  the ONLY varied factors are (head trained?) x (demotion on?). MECH-448 flags
  (use_f_eligibility_demotion / f_eligibility_envelope_floor / f_eligibility_dn_sigma)
  are matched constants on the demotion-on arms.

KEPT FROM 485h: ofc_bias_scale=0.5, SD-056 e2_action_contrastive armed in P0, the
  PAIRED threat-conditioned head driver (outcome-coupled spread@high-threat-state_code
  + anti-range variance penalty@devalued-state_code, both via the SHARED
  _settle_state_code the eval uses), phased training (P0 encoder warmup with SD-056
  contrastive -> P1 head training on the frozen-encoder state_code -> P2 BEHAVIOURAL
  eval), the OFC-isolated SD-054 bipartite reef/forage env (ofc_harm_dim>0),
  LR_OFC_BIAS 2e-3 / P1 120 ep, and the readiness floor 0.05 (== DEVAL_SHIFT_MARGIN,
  the same-statistic non-vacuity gate).

PRE-REGISTERED ACCEPTANCE (per seed, then >= MIN_PASS_SEEDS of 3):
  READINESS (non-vacuity precondition, on the ARM_2 test arm; SAME statistic the
    load-bearing DVs route on):
      (a) trained-head OFC bias cross-candidate RANGE at the high-threat positive
          control > BIAS_RANGE_FLOOR (0.05) -- RANGE not magnitude.
      (b) head trained (state_bias_head weight-delta > HEAD_DELTA_MIN).
      (c) MECH-448 NON-DEGENERACY: f_eligibility_excluded_count > 0 AND
          f_eligibility_demotion_active -- the envelope actually EXCLUDED a candidate
          (F was genuinely demoted on a divergent F pool), not an all-admit no-op.
    Any below floor -> substrate_not_ready_requeue (non_contributory), NEVER a weakens.
  C1_devaluation_behavioural_shift (load-bearing): ARM_2 devaluation_selection_shift
    > DEVAL_SHIFT_MARGIN AND > ARM_1 (demotion-off ceiling) shift + DEVAL_SHIFT_MARGIN
    -- conversion attributable to the demotion lever, not to the trained head alone.
  C2_discrimination_behavioural_separation (load-bearing): ARM_2 separation_ratio
    >= SEPARATION_RATIO_MIN AND between_context_selection_tv >= BETWEEN_TV_FLOOR AND
    > ARM_1 between_tv + BETWEEN_TV_FLOOR.
  C3_silence_control (load-bearing): ARM_0 (frozen head, demotion on) shift < 1e-9
    AND between-context TV < 1e-9 -- the demotion selector alone (without a trained
    valuation) converts nothing.
  PASS (supports SD-033b + MECH-263) = READINESS AND C1 AND C2 AND C3 on a majority.

INTERPRETATION GRID (one row per plausible outcome -> next action):
  PASS / supports: the trained OFC outcome-value channel, routed through the
    demotion-enabled selector, produces devaluation-sensitive AND task-role-
    discriminative committed selection that neither the demotion-off ceiling arm nor
    the untrained-head silence arm produces. SD-033b candidate -> provisional
    behavioural evidence; contributes to closing behavioral_diversity_isolation:GAP-I.
  FAIL / weakens (readiness MET, DVs fail): the trained OFC channel has genuine
    cross-candidate range AND the demotion envelope genuinely excluded F-dominant
    candidates, yet committed selection still does not convert under devaluation /
    discrimination. Route /failure-autopsy (conversion-ceiling-persists-despite-
    demotion -> MECH-449 follow-on / V4) BEFORE stamping. An honest result.
  FAIL / substrate_not_ready_requeue (readiness UNMET): below the 0.05 bias-range
    floor, head untrained, OR the F pool was non-divergent so the envelope admitted
    all (excluded_count == 0) -> the behavioural test never ran. Re-queue with a
    stronger driver / budget / divergent F pool. NOT a weakens.

SLEEP DRIVER: not applicable (no sleep loop in this experiment).
"""

from __future__ import annotations

import argparse
import json
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

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_485i_sd033b_demotion_enabled_behavioural"
EXPERIMENT_PURPOSE = "evidence"
QUEUE_ID = "V3-EXQ-485i"
CLAIM_IDS: List[str] = ["SD-033b", "MECH-263"]

SELF_DIM = 8
WORLD_DIM = 32
HARM_DIM = 4
HARM_A_DIM = 4
HARM_HISTORY_LEN = 10

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

P0_EPISODES = 30
P1_EPISODES = 120                      # kept from 485h: give the driver budget
STEPS_PER_EPISODE = 100
TRAIN_EPS = P0_EPISODES + P1_EPISODES  # progress denominator M

LR_E1 = 1e-3
LR_E2_WF = 1e-3
LR_E3_HARM = 1e-3
LR_OFC_BIAS = 2e-3                     # kept from 485h
BATCH_SIZE = 32
WF_BUF_MAX = 256
N_PROBE_CANDIDATES = 8
RECORD_EVERY_N_STEPS = 4
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
EMA_DECAY = 0.9

# -- 485h paired threat-conditioned driver (kept verbatim) --
OUTCOME_COUPLE_GAIN = 4.0
ADV_MIN_THRESHOLD = 1e-4
DEVALUED_ANTIRANGE_WEIGHT = 0.5
DRIVER_SNAP_BATCH = 16

# -- 485f Fix 1 (kept): defeat OFC clamp-saturation --
OFC_BIAS_SCALE = 0.5

# -- 485f Fix 2 (kept): SD-056 e2_action_contrastive in P0 -> action-divergent bank --
E2_CONTRASTIVE_WEIGHT = 0.05
E2_CONTRASTIVE_TEMP = 0.1

# -- P2 behavioural-eval drive lengths (parallel to 485b/485c/485f/485g/485h) --
PRE_ONSET_TICKS = 25
POST_ONSET_TICKS = 40
CONTEXT_TICKS = 25

# -- MECH-448 (ARC-107) demotion lever constants (matched on the demotion-on arms) --
F_ELIG_ENVELOPE_FLOOR = 0.30   # absolute DN-share floor (substrate default)
F_ELIG_DN_SIGMA = 0.0          # DN semi-saturation / global tightness (substrate default)

# -- Deterministic committed readout: force the argmin-within-eligible path so each
#    DV is the genuine committed selection of the demotion-enabled selector (not a
#    multinomial sample). Eval-time only; training keeps commitment_threshold=0.5. --
EVAL_COMMIT_THRESHOLD = 1.0e6

# -- Pre-registered thresholds (NOT derived from the run's own statistics) --
HEAD_DELTA_MIN = 1e-3
BIAS_RANGE_FLOOR = 0.05      # == DEVAL_SHIFT_MARGIN (same-statistic non-vacuity gate)
DEVAL_SHIFT_MARGIN = 0.05
SEPARATION_RATIO_MIN = 3.0
BETWEEN_TV_FLOOR = 0.05
FROZEN_SILENCE_EPS = 1e-9
MIN_PASS_SEEDS = 2


def _kth_largest(values, k: int, default: float = 0.0) -> float:
    """The k-th LARGEST value, so `result > floor` is exactly "at least k values
    > floor".

    The readiness gates below are k-of-n COUNT-OF-SEEDS predicates
    (`... >= MIN_PASS_SEEDS`), but the manifest used to report `max(...)` over
    seeds for them. The REE_assembly indexer
    (build_experiment_indexes._precondition_unmet) RECOMPUTES every
    interpretation.preconditions[] entry's `met` from that entry's OWN reported
    (measured, threshold) pair and treats the recompute as AUTHORITATIVE over the
    author's value. A max-over-seeds statistic declared against a PER-SEED floor
    therefore recomputes as MET as soon as ONE seed clears the floor -- strictly
    LOOSER than the shipped k-of-n gate, i.e. it silently clears real premise
    failures (the dangerous MISSED_UNMET direction). The k-th largest is the exact
    statistic for a floor needing k of n.

    Fewer than k values -> `default` (0.0), which is below every floor here, so
    the entry recomputes UNMET -- matching the count predicate, which also cannot
    reach k.
    """
    vals = sorted((float(v) for v in values), reverse=True)
    return vals[k - 1] if len(vals) >= k else float(default)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _onehot_vec(idx: int, k: int) -> torch.Tensor:
    """[K] one-hot of the committed candidate index (the deterministic committed
    selection distribution under the demotion selector)."""
    v = torch.zeros(k)
    if 0 <= idx < k:
        v[idx] = 1.0
    return v


def _tv(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total-variation distance between two selection distributions."""
    return float(0.5 * (p - q).abs().sum().item())


def _bias_range(agent: REEAgent, bank: torch.Tensor) -> float:
    """Cross-candidate range (max - min) of the OFC bias over the bank -- the
    readiness statistic the load-bearing selection DVs route on."""
    bias = agent.ofc.compute_bias(bank).detach()
    if bias.numel() < 2:
        return 0.0
    return float((bias.max() - bias.min()).item())


def _bank_zworld_spread(bank: torch.Tensor) -> float:
    """Mean pairwise L2 across the K candidate-bank z_world summaries (diagnostic)."""
    if bank is None or bank.shape[0] < 2:
        return 0.0
    d = torch.cdist(bank, bank)
    K = bank.shape[0]
    eye = torch.eye(K, dtype=torch.bool, device=bank.device)
    return float(d[~eye].mean().item())


def _settle_state_code(
    agent: REEAgent,
    z_world_rep: torch.Tensor,
    z_harm: Optional[torch.Tensor],
    ticks: int,
    reset: bool,
) -> None:
    """Drive the OFC state_code toward a regime via gated EMA updates. SHARED by the
    P1 driver and the P2 eval so the trained and tested state_codes are constructed
    identically (kept from 485h)."""
    if reset:
        agent.ofc.reset()
    for _ in range(ticks):
        agent.ofc.update(z_world=z_world_rep, z_harm=z_harm, gate=1.0)


def _make_agent(seed: int, train_head: bool, demotion: bool) -> Tuple[REEAgent, CausalGridWorldV2]:
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
        # OFC isolated: no gated_policy / dACC / lateral_pfc bias channels.
        use_gated_policy=False,
        use_dacc=False,
        dacc_weight=0.0,
        use_lateral_pfc_analog=False,
        # SP-CEM main-path for first-action candidate diversity.
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # 485f Fix 2 (kept): SD-056 e2 action-conditional contrastive.
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=E2_CONTRASTIVE_WEIGHT,
        e2_action_contrastive_temperature=E2_CONTRASTIVE_TEMP,
        e2_action_contrastive_min_batch_classes=2,
        # SD-033b OFC analog under test. ofc_harm_dim>0 -> z_harm enters state_code.
        use_ofc_analog=True,
        ofc_state_dim=16,
        ofc_harm_dim=HARM_DIM,
        ofc_bias_scale=OFC_BIAS_SCALE,
        ofc_train_state_bias_head=train_head,
        # MECH-448 (ARC-107) rank-preserving F->eligibility demotion (the ARM_ON
        # variable). The OFC bias is passed as score_bias at eval -> _modulatory_accum
        # is non-None -> the demotion eligibility branch arbitrates the committed
        # action within the F-eligible set with F removed from the final argmin.
        use_f_eligibility_demotion=demotion,
        f_eligibility_envelope_floor=F_ELIG_ENVELOPE_FLOOR,
        f_eligibility_dn_sigma=F_ELIG_DN_SIGMA,
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    agent = REEAgent(config)
    return agent, env


def _preflight() -> None:
    frozen, _ = _make_agent(1, train_head=False, demotion=True)
    trainable, _ = _make_agent(2, train_head=True, demotion=True)
    ceiling, _ = _make_agent(3, train_head=True, demotion=False)
    assert frozen.ofc is not None and trainable.ofc is not None
    assert frozen.ofc.config.train_state_bias_head is False
    assert trainable.ofc.config.train_state_bias_head is True
    assert frozen.ofc.config.harm_dim == HARM_DIM
    assert abs(float(trainable.ofc.config.bias_scale) - OFC_BIAS_SCALE) < 1e-9
    assert hasattr(trainable.e2, "world_forward_contrastive_loss")
    assert bool(getattr(trainable.config.e2, "e2_action_contrastive_enabled", False))
    # readiness floor re-aligned to the DV floor (same-statistic gate).
    assert abs(BIAS_RANGE_FLOOR - DEVAL_SHIFT_MARGIN) < 1e-9
    fz = frozen.ofc.state_bias_head[-1]
    tz = trainable.ofc.state_bias_head[-1]
    assert bool(torch.all(fz.weight == 0)) and bool(torch.all(fz.bias == 0))
    assert not (bool(torch.all(tz.weight == 0)) and bool(torch.all(tz.bias == 0)))
    assert len(list(trainable.ofc.bias_head_parameters())) == 4
    # MECH-448 contract: E3Config carries the demotion lever flags; ARM_ON True.
    assert hasattr(trainable.e3.config, "use_f_eligibility_demotion")
    assert trainable.e3.config.use_f_eligibility_demotion is True
    assert ceiling.e3.config.use_f_eligibility_demotion is False
    assert abs(float(trainable.e3.config.f_eligibility_envelope_floor) - F_ELIG_ENVELOPE_FLOOR) < 1e-9
    del frozen, trainable, ceiling
    print(
        "Preflight PASS: OFC analog + harm_dim + trainable/frozen head + "
        "bias_head_parameters + ofc_bias_scale=0.5 + SD-056 contrastive + readiness "
        "floor 0.05 == DV floor + paired threat-conditioned driver + MECH-448 "
        "demotion flags wired (ARM_ON True / ARM_OFF False)",
        flush=True,
    )


def _build_snap(agent: REEAgent, candidates: List) -> Optional[torch.Tensor]:
    """First-step z_world summaries [K, world_dim] from the proposer candidates."""
    if not isinstance(candidates, list) or len(candidates) < N_PROBE_CANDIDATES:
        return None
    if getattr(candidates[0], "world_states", None) is None:
        return None
    if len(candidates[0].world_states) < 2:
        return None
    return torch.cat(
        [c.world_states[1].detach().clone() for c in candidates[:N_PROBE_CANDIDATES]],
        dim=0,
    )  # [K, world_dim]


def _head_weight_vector(agent: REEAgent) -> torch.Tensor:
    return torch.cat(
        [p.detach().reshape(-1).cpu() for p in agent.ofc.bias_head_parameters()]
    )


# --------------------------------------------------------------------------- #
# training (P0 encoder warmup + P1 paired threat-conditioned head driver)
# kept verbatim from 485h -- the demotion flag does NOT affect training, only the
# eval committed selection through E3.select().
# --------------------------------------------------------------------------- #
def _encoder_step(agent: REEAgent, env, steps: int, arm: str, ep_global: int) -> float:
    device = agent.device
    e1_opt = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    e2_opt = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    he_opt = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM)
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    he_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []
    ep_reward = 0.0
    z_wp = z_sp = act_p = None
    _, obs_dict = env.reset()
    agent.reset()
    for _ in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        latent = agent.sense(
            obs_body,
            obs_world,
            obs_harm=obs_dict.get("harm_obs"),
            obs_harm_a=obs_dict.get("harm_obs_a"),
            obs_harm_history=obs_dict.get("harm_history"),
        )
        z_w = latent.z_world.detach()
        if z_wp is not None and act_p is not None:
            agent.record_transition(z_sp, act_p, latent.z_self.detach())
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent)
            if ticks.get("e1_tick", False)
            else torch.zeros(1, WORLD_DIM, device=device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        drive = REEAgent.compute_drive_level(obs_body)
        agent.update_z_goal(
            benefit_exposure=max(0.0, float(obs_dict.get("benefit_exposure", 0.0))),
            drive_level=drive,
        )
        action = agent.select_action(candidates, ticks, temperature=1.0)
        if action is None:
            action = _action_to_onehot(
                random.randint(0, env.action_dim - 1), env.action_dim, device
            )
            agent._last_action = action
        _, harm, done, _, obs_dict = env.step(action)
        ep_reward += float(harm)
        if z_wp is not None and act_p is not None:
            wf_buf.append((z_wp.cpu(), act_p.cpu(), z_w.cpu()))
            if len(wf_buf) > WF_BUF_MAX:
                wf_buf = wf_buf[-WF_BUF_MAX:]
        he_buf.append(
            (z_w.cpu(), torch.tensor([abs(float(harm)) if float(harm) < 0 else 0.0]))
        )
        if len(wf_buf) >= BATCH_SIZE:
            idx = torch.randperm(len(wf_buf))[:BATCH_SIZE].tolist()
            zw = torch.cat([wf_buf[i][0] for i in idx]).to(device)
            a = torch.cat([wf_buf[i][1] for i in idx]).to(device)
            zw1 = torch.cat([wf_buf[i][2] for i in idx]).to(device)
            pred = agent.e2.world_forward(zw, a)
            loss = F.mse_loss(pred, zw1)
            # SD-056 action-contrastive auxiliary loss (kept from 485f/485g/485h).
            contrast = agent.e2.world_forward_contrastive_loss(
                zw, a, zw1,
                temperature=E2_CONTRASTIVE_TEMP,
                min_batch_classes=2,
            )
            loss = loss + E2_CONTRASTIVE_WEIGHT * contrast
            if loss.requires_grad:
                e2_opt.zero_grad()
                loss.backward()
                e2_opt.step()
            with torch.no_grad():
                agent.e3.update_running_variance((pred.detach() - zw1).detach())
        if len(he_buf) >= BATCH_SIZE:
            idx = torch.randperm(len(he_buf))[:BATCH_SIZE].tolist()
            zw = torch.cat([he_buf[i][0] for i in idx]).to(device)
            ht = torch.cat([he_buf[i][1] for i in idx]).to(device)
            loss = F.mse_loss(agent.e3.harm_eval(zw).squeeze(), ht.squeeze())
            if loss.requires_grad:
                he_opt.zero_grad()
                loss.backward()
                he_opt.step()
        if len(agent._world_experience_buffer) >= 2:
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                e1_opt.step()
        z_wp, z_sp, act_p = z_w, latent.z_self.detach(), action.detach()
        if done:
            break
    if ep_global == 1 or ep_global % 10 == 0:
        print(
            f"  [train] {arm} ep {ep_global}/{TRAIN_EPS}  phase=P0  reward={ep_reward:.3f}",
            flush=True,
        )
    return ep_reward


def _ofc_threat_conditioned_driver_loss(
    agent: REEAgent,
    snap_buf: List[Tuple[torch.Tensor, torch.Tensor]],
    z_harm_high: torch.Tensor,
    z_harm_low: torch.Tensor,
    device,
) -> Tuple[torch.Tensor, int, int]:
    """485h paired threat-conditioned, outcome-coupled, per-candidate driver (kept
    verbatim). Trains the trained-OFC head on BOTH the high-threat regime (spread
    bias to favour low predicted harm) AND the devalued regime (suppress bias range)
    using a SHARED state_code construction (_settle_state_code) identical to the eval.
    Returns (loss, n_spread, n_anti). Side effect: mutates ofc.state_code; the caller
    saves/restores it."""
    if agent.ofc is None or len(snap_buf) < 2:
        return torch.zeros(1, device=device), 0, 0
    threat_high = float(z_harm_high.detach().norm().item())
    idxs = np.random.choice(
        len(snap_buf), size=min(DRIVER_SNAP_BATCH, len(snap_buf)), replace=False
    )
    spread_terms: List[torch.Tensor] = []
    anti_terms: List[torch.Tensor] = []
    for i in idxs:
        bank, z_world_rep = snap_buf[int(i)]
        if bank.shape[0] < 2:
            continue
        # ---- high-threat regime: spread ----
        _settle_state_code(agent, z_world_rep, z_harm_high, PRE_ONSET_TICKS, reset=True)
        with torch.no_grad():
            harm = agent.e3.harm_eval(bank).reshape(-1)  # [K] per-candidate harm cost
            if harm.numel() != bank.shape[0]:
                continue
            adv = OUTCOME_COUPLE_GAIN * threat_high * (harm.mean() - harm)  # [K]
        if float(adv.abs().max()) >= ADV_MIN_THRESHOLD:
            bias_hi = agent.ofc.compute_bias(bank)  # [K], grad flows into the head
            log_p = F.log_softmax(-bias_hi / POLICY_TEMPERATURE, dim=0)
            spread_terms.append(-(adv * log_p).sum())
        # ---- devalued regime: anti-range (continue the high->devalued transition) ----
        _settle_state_code(agent, z_world_rep, z_harm_low, POST_ONSET_TICKS, reset=False)
        bias_dev = agent.ofc.compute_bias(bank)  # [K], grad flows into the head
        anti_terms.append((bias_dev - bias_dev.mean()).pow(2).mean())
    if not spread_terms and not anti_terms:
        return torch.zeros(1, device=device), 0, 0
    loss = torch.zeros(1, device=device)
    if spread_terms:
        loss = loss + torch.stack(spread_terms).mean()
    if anti_terms:
        loss = loss + DEVALUED_ANTIRANGE_WEIGHT * torch.stack(anti_terms).mean()
    return loss, len(spread_terms), len(anti_terms)


def _p1_train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    episodes: int,
    steps: int,
    arm: str,
    train_head: bool,
    z_harm_high: torch.Tensor,
    z_harm_low: torch.Tensor,
) -> Dict:
    device = agent.device
    bias_opt = (
        optim.Adam(list(agent.ofc.bias_head_parameters()), lr=LR_OFC_BIAS)
        if train_head
        else None
    )
    head_init = _head_weight_vector(agent)
    grad_nonzero_updates = 0
    n_spread_terms = 0
    n_anti_terms = 0
    snap_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []
    bias_samples: List[float] = []
    agent.train()
    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()
        step = 0
        z_wp = z_sp = act_p = None
        for _ in range(steps):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(
                obs_body,
                obs_world,
                obs_harm=obs_dict.get("harm_obs"),
                obs_harm_a=obs_dict.get("harm_obs_a"),
                obs_harm_history=obs_dict.get("harm_history"),
            )
            if z_wp is not None and act_p is not None:
                agent.record_transition(z_sp, act_p, latent.z_self.detach())
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            agent.update_z_goal(
                benefit_exposure=max(0.0, float(obs_dict.get("benefit_exposure", 0.0))),
                drive_level=REEAgent.compute_drive_level(obs_body),
            )
            if step % RECORD_EVERY_N_STEPS == 0:
                snap = _build_snap(agent, candidates)
                if snap is not None:
                    snap_buf.append((snap, latent.z_world.detach().clone()))
            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, device
                )
                agent._last_action = action
            if agent.ofc is not None:
                bias_samples.append(float(agent.ofc._last_bias_abs_mean))
            _, harm, done, _, obs_dict = env.step(action)
            z_wp, z_sp, act_p = (
                latent.z_world.detach(),
                latent.z_self.detach(),
                action.detach(),
            )
            step += 1
            if done:
                break
        if len(snap_buf) > OUTCOME_BUF_MAX:
            snap_buf = snap_buf[-OUTCOME_BUF_MAX:]
        if bias_opt is not None and len(snap_buf) >= 2:
            saved_state = agent.ofc.state_code.detach().clone()
            l_loss, n_sp, n_an = _ofc_threat_conditioned_driver_loss(
                agent, snap_buf, z_harm_high, z_harm_low, device
            )
            with torch.no_grad():
                agent.ofc.state_code.copy_(saved_state)
            n_spread_terms += n_sp
            n_anti_terms += n_an
            if l_loss.requires_grad:
                bias_opt.zero_grad()
                l_loss.backward()
                gsum = sum(
                    float(p.grad.abs().sum())
                    for p in agent.ofc.bias_head_parameters()
                    if p.grad is not None
                )
                if gsum > 0:
                    grad_nonzero_updates += 1
                torch.nn.utils.clip_grad_norm_(agent.ofc.bias_head_parameters(), 1.0)
                bias_opt.step()
        ep_global = P0_EPISODES + ep + 1
        if ep_global % 10 == 0:
            print(
                f"  [train] {arm} ep {ep_global}/{TRAIN_EPS}  phase=P1"
                f"  bias_abs={bias_samples[-1] if bias_samples else 0:.5f}"
                f"  grad_updates={grad_nonzero_updates}",
                flush=True,
            )
    head_final = _head_weight_vector(agent)
    head_delta = float(torch.norm(head_final - head_init).item())
    return {
        "p1_mean_abs_ofc_bias": float(np.mean(bias_samples)) if bias_samples else 0.0,
        "head_weight_delta_norm": head_delta,
        "n_grad_nonzero_updates": grad_nonzero_updates,
        "n_spread_loss_terms": n_spread_terms,
        "n_antirange_loss_terms": n_anti_terms,
    }


# --------------------------------------------------------------------------- #
# P2 behavioural eval on the demotion-enabled E3 selector
# --------------------------------------------------------------------------- #
def _collect_eval_candidates(
    agent: REEAgent, env: CausalGridWorldV2, steps: int
) -> Tuple[Optional[List], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run the trained agent briefly and return (candidates[:K], candidate_bank
    [K, world_dim], z_world_rep [1, world_dim]) -- a real candidate list aligned with
    its first-step z_world bank for the controlled devaluation / discrimination drives
    (kept from 485h)."""
    device = agent.device
    cands: Optional[List] = None
    bank: Optional[torch.Tensor] = None
    z_world_rep: Optional[torch.Tensor] = None
    agent.eval()
    with torch.no_grad():
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps):
            latent = agent.sense(
                obs_dict["body_state"],
                obs_dict["world_state"],
                obs_harm=obs_dict.get("harm_obs"),
                obs_harm_a=obs_dict.get("harm_obs_a"),
                obs_harm_history=obs_dict.get("harm_history"),
            )
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            snap = _build_snap(agent, candidates)
            if snap is not None and bank is None:
                bank = snap
                z_world_rep = latent.z_world.detach().clone()
                cands = list(candidates[:N_PROBE_CANDIDATES])
            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(0, env.action_dim, device)
            _, _, done, _, obs_dict = env.step(action)
            if bank is not None:
                break
            if done:
                _, obs_dict = env.reset()
                agent.reset()
    return cands, bank, z_world_rep


def _committed_select(
    agent: REEAgent, cands: List, score_bias: torch.Tensor, temperature: float
) -> Tuple[int, bool, Dict, float]:
    """Route the OFC bias through the REAL E3 selector as score_bias and return the
    COMMITTED candidate index + commit flag + diagnostics + raw (F-score) range.

    With use_f_eligibility_demotion on the agent's E3Config, the OFC bias (passed as
    score_bias -> _modulatory_accum) arbitrates the committed action WITHIN the
    F-eligible envelope, F removed from the final argmin. EVAL_COMMIT_THRESHOLD has
    forced the committed (argmin-within-eligible) path so selected_index is
    deterministic -- the genuine committed selection of the demotion-enabled selector."""
    res = agent.e3.select(
        cands,
        temperature=temperature,
        goal_state=getattr(agent, "goal_state", None),
        score_bias=score_bias,
    )
    diag = dict(agent.e3.last_score_diagnostics)
    raw = agent.e3.last_raw_scores
    raw_range = (
        float((raw.max() - raw.min()).item())
        if raw is not None and raw.numel() > 1
        else 0.0
    )
    return int(res.selected_index), bool(res.committed), diag, raw_range


def _demotion_committed_eval(
    agent: REEAgent, cands: List, bank: torch.Tensor, z_world_rep: torch.Tensor,
    z_harm_high: torch.Tensor, z_harm_low: torch.Tensor, seed: int,
) -> Dict:
    """PRIMARY behavioural eval: devaluation-selection shift + task-role discrimination
    measured on the COMMITTED selection through the demotion-enabled E3 selector.
    Records the readiness positive control (OFC bias range at the high-threat state),
    the devalued-state range (the 485h disambiguator), and the MECH-448 non-degeneracy
    signals (f_eligibility_excluded_count / envelope_size / demotion_active)."""
    K = bank.shape[0]
    agent.eval()
    prev_thr = float(agent.e3.config.commitment_threshold)
    agent.e3.config.commitment_threshold = EVAL_COMMIT_THRESHOLD
    excluded_count = -1
    envelope_size = -1
    demotion_active = False
    winner_neq_f = False
    raw_range_hi = 0.0
    try:
        with torch.no_grad():
            # ---- P2-a devaluation sensitivity ----
            _settle_state_code(agent, z_world_rep, z_harm_high, PRE_ONSET_TICKS, reset=True)
            bias_high = agent.ofc.compute_bias(bank).detach()
            bias_range_high = _bias_range(agent, bank)  # readiness positive control
            idx_high, comm_high, diag_hi, raw_range_hi = _committed_select(
                agent, cands, bias_high, POLICY_TEMPERATURE
            )
            excluded_count = int(diag_hi.get("f_eligibility_excluded_count", -1))
            envelope_size = int(diag_hi.get("f_eligibility_envelope_size", -1))
            demotion_active = bool(diag_hi.get("f_eligibility_demotion_active", False))
            winner_neq_f = bool(diag_hi.get("f_eligibility_winner_neq_f_argmin", False))

            _settle_state_code(agent, z_world_rep, z_harm_low, POST_ONSET_TICKS, reset=False)
            bias_low = agent.ofc.compute_bias(bank).detach()
            bias_range_devalued = _bias_range(agent, bank)  # 485h disambiguator
            idx_low, comm_low, _, _ = _committed_select(
                agent, cands, bias_low, POLICY_TEMPERATURE
            )
            deval_shift = _tv(_onehot_vec(idx_high, K), _onehot_vec(idx_low, K))

            # ---- P2-b task-role discrimination ----
            ctx = CONTEXT_TICKS
            g = torch.Generator().manual_seed(20000 + seed)
            hist_a = z_world_rep + 1.5 + 0.1 * torch.randn(ctx, WORLD_DIM, generator=g)
            hist_b = z_world_rep - 1.5 + 0.1 * torch.randn(ctx, WORLD_DIM, generator=g)
            hist_a2 = z_world_rep + 1.5 + 0.1 * torch.randn(ctx, WORLD_DIM, generator=g)

            def _drive_then_select(hist: torch.Tensor) -> Tuple[int, float]:
                agent.ofc.reset()
                for t in range(hist.shape[0]):
                    agent.ofc.update(z_world=hist[t:t + 1], z_harm=None, gate=1.0)
                agent.ofc.update(z_world=z_world_rep, z_harm=None, gate=1.0)
                b = agent.ofc.compute_bias(bank).detach()
                idx, _, _, _ = _committed_select(agent, cands, b, POLICY_TEMPERATURE)
                rng = _bias_range(agent, bank)
                return idx, rng

            idx_a, bias_range_context_a = _drive_then_select(hist_a)
            idx_b, bias_range_context_b = _drive_then_select(hist_b)
            idx_a2, _ = _drive_then_select(hist_a2)
            between = _tv(_onehot_vec(idx_a, K), _onehot_vec(idx_b, K))
            within = _tv(_onehot_vec(idx_a, K), _onehot_vec(idx_a2, K))
            separation_ratio = between / max(within, 1e-6)
    finally:
        agent.e3.config.commitment_threshold = prev_thr

    collapse_ratio = (
        bias_range_devalued / bias_range_high if bias_range_high > 1e-9 else 0.0
    )
    return {
        "devaluation_selection_shift": deval_shift,
        "bias_range_high_threat": bias_range_high,
        "bias_range_devalued": bias_range_devalued,
        "devaluation_range_collapse_ratio": collapse_ratio,
        "bias_range_context_a": bias_range_context_a,
        "bias_range_context_b": bias_range_context_b,
        "bank_zworld_spread": _bank_zworld_spread(bank),
        "between_context_selection_tv": between,
        "within_context_selection_jitter": within,
        "discrimination_separation_ratio": separation_ratio,
        "committed_high": comm_high,
        "committed_low": comm_low,
        "f_eligibility_demotion_active": demotion_active,
        "f_eligibility_excluded_count": excluded_count,
        "f_eligibility_envelope_size": envelope_size,
        "f_eligibility_winner_neq_f_argmin": winner_neq_f,
        "raw_score_range_high": raw_range_hi,
    }


def _empty_eval() -> Dict:
    return {
        "devaluation_selection_shift": 0.0,
        "bias_range_high_threat": 0.0,
        "bias_range_devalued": 0.0,
        "devaluation_range_collapse_ratio": 0.0,
        "bias_range_context_a": 0.0,
        "bias_range_context_b": 0.0,
        "bank_zworld_spread": 0.0,
        "between_context_selection_tv": 0.0,
        "within_context_selection_jitter": 0.0,
        "discrimination_separation_ratio": 0.0,
        "committed_high": False,
        "committed_low": False,
        "f_eligibility_demotion_active": False,
        "f_eligibility_excluded_count": -1,
        "f_eligibility_envelope_size": -1,
        "f_eligibility_winner_neq_f_argmin": False,
        "raw_score_range_high": 0.0,
        "eval_context_built": False,
    }


def run_arm_seed(arm: str, train_head: bool, demotion: bool, seed: int, dry_run: bool) -> Dict:
    p0 = 3 if dry_run else P0_EPISODES
    p1 = 4 if dry_run else P1_EPISODES
    steps = 30 if dry_run else STEPS_PER_EPISODE
    print(
        f"\nSeed {seed} Condition {arm} train_head={train_head} demotion={demotion}",
        flush=True,
    )
    full_config = {
        "arm": arm,
        "train_head": train_head,
        "demotion": demotion,
        "p0": p0,
        "p1": p1,
        "steps": steps,
        "ofc_bias_scale": OFC_BIAS_SCALE,
        "e2_contrastive_weight": E2_CONTRASTIVE_WEIGHT,
        "lr_ofc_bias": LR_OFC_BIAS,
        "outcome_couple_gain": OUTCOME_COUPLE_GAIN,
        "devalued_antirange_weight": DEVALUED_ANTIRANGE_WEIGHT,
        "bias_range_floor": BIAS_RANGE_FLOOR,
        "f_eligibility_envelope_floor": F_ELIG_ENVELOPE_FLOOR,
        "f_eligibility_dn_sigma": F_ELIG_DN_SIGMA,
        "eval_commit_threshold": EVAL_COMMIT_THRESHOLD,
        "env_kwargs": ENV_KWARGS,
    }
    with arm_cell(seed, config_slice=full_config, script_path=Path(__file__)) as cell:
        agent, env = _make_agent(seed, train_head=train_head, demotion=demotion)
        # Canonical high-threat / devalued z_harm, built ONCE per (arm, seed) and
        # shared by the P1 driver and the P2 eval so trained regimes == tested.
        g = torch.Generator().manual_seed(20000 + seed)
        z_harm_high = torch.randn(1, HARM_DIM, generator=g) * 1.0
        z_harm_low = torch.zeros(1, HARM_DIM)
        for ep in range(p0):
            _encoder_step(agent, env, steps, arm, ep + 1)
        p1m = _p1_train(agent, env, p1, steps, arm, train_head, z_harm_high, z_harm_low)
        cands, bank, z_world_rep = _collect_eval_candidates(agent, env, steps)
        if bank is None or z_world_rep is None or cands is None:
            beh = _empty_eval()
        else:
            beh = _demotion_committed_eval(
                agent, cands, bank, z_world_rep, z_harm_high, z_harm_low, seed
            )
            beh["eval_context_built"] = True
        row = {
            "arm": arm,
            "seed": seed,
            "ofc_train_state_bias_head": train_head,
            "use_f_eligibility_demotion": demotion,
            "head_weight_delta_norm": p1m["head_weight_delta_norm"],
            "n_grad_nonzero_updates": p1m["n_grad_nonzero_updates"],
            "n_spread_loss_terms": p1m["n_spread_loss_terms"],
            "n_antirange_loss_terms": p1m["n_antirange_loss_terms"],
            "p1_mean_abs_ofc_bias": p1m["p1_mean_abs_ofc_bias"],
            **beh,
        }
        cell.stamp(row)
    print(
        f"verdict: {'PASS' if beh.get('eval_context_built') else 'FAIL'}  "
        f"head_delta={p1m['head_weight_delta_norm']:.6f}  "
        f"bias_range_hi={beh['bias_range_high_threat']:.6f}  "
        f"deval_shift={beh['devaluation_selection_shift']:.4f}  "
        f"between_tv={beh['between_context_selection_tv']:.4f}  "
        f"excluded={beh['f_eligibility_excluded_count']}  "
        f"demotion_active={beh['f_eligibility_demotion_active']}",
        flush=True,
    )
    return row


# --------------------------------------------------------------------------- #
# aggregation + interpretation
# --------------------------------------------------------------------------- #
ARM_FROZEN = "ARM_0_frozen_demotion_on"
ARM_CEILING = "ARM_1_trained_demotion_off"
ARM_TEST = "ARM_2_trained_demotion_on"


def run(seeds: Optional[List[int]] = None, dry_run: bool = False) -> Dict:
    if seeds is None:
        seeds = [0, 1, 2]
    arms = [
        (ARM_FROZEN, False, True),
        (ARM_CEILING, True, False),
        (ARM_TEST, True, True),
    ]
    by_arm: Dict[str, List[Dict]] = {a[0]: [] for a in arms}
    for seed in seeds:
        for arm_label, train_head, demotion in arms:
            by_arm[arm_label].append(
                run_arm_seed(arm_label, train_head, demotion, seed, dry_run)
            )

    frozen = by_arm[ARM_FROZEN]
    ceiling = by_arm[ARM_CEILING]
    test = by_arm[ARM_TEST]
    n = len(seeds)

    # READINESS (non-vacuity, on the ARM_2 test arm).
    ready_seeds = sum(
        1
        for r in test
        if r["head_weight_delta_norm"] > HEAD_DELTA_MIN
        and r["bias_range_high_threat"] > BIAS_RANGE_FLOOR
        and r.get("eval_context_built", False)
    )
    nondegen_seeds = sum(
        1
        for r in test
        if int(r.get("f_eligibility_excluded_count", -1)) > 0
        and bool(r.get("f_eligibility_demotion_active", False))
    )
    readiness_met = ready_seeds >= MIN_PASS_SEEDS and nondegen_seeds >= MIN_PASS_SEEDS

    c1_seeds = 0
    c2_seeds = 0
    for ce, te in zip(ceiling, test):
        if (
            te["devaluation_selection_shift"] > DEVAL_SHIFT_MARGIN
            and te["devaluation_selection_shift"]
            > ce["devaluation_selection_shift"] + DEVAL_SHIFT_MARGIN
        ):
            c1_seeds += 1
        if (
            te["discrimination_separation_ratio"] >= SEPARATION_RATIO_MIN
            and te["between_context_selection_tv"] >= BETWEEN_TV_FLOOR
            and te["between_context_selection_tv"]
            > ce["between_context_selection_tv"] + BETWEEN_TV_FLOOR
        ):
            c2_seeds += 1
    c3_seeds = sum(
        1
        for r in frozen
        if r["devaluation_selection_shift"] < FROZEN_SILENCE_EPS
        and r["between_context_selection_tv"] < FROZEN_SILENCE_EPS
    )

    c1 = c1_seeds >= MIN_PASS_SEEDS
    c2 = c2_seeds >= MIN_PASS_SEEDS
    c3 = c3_seeds >= MIN_PASS_SEEDS
    overall_pass = readiness_met and c1 and c2 and c3

    return {
        "by_arm": by_arm,
        "n_seeds": n,
        "ready_seeds": ready_seeds,
        "nondegen_seeds": nondegen_seeds,
        "readiness_met": readiness_met,
        # Per-leg k-of-n readiness statistics, declared in interpretation.preconditions[]
        # so the indexer's authoritative recompute reproduces each entry's `met`
        # from its own (measured, threshold) pair. Routing is UNCHANGED: the label
        # and evidence_direction still come from `readiness_met`, which is the
        # per-seed CONJUNCTION count, not from these entries.
        "n_bias_range_seeds": sum(
            1 for r in test if r["bias_range_high_threat"] > BIAS_RANGE_FLOOR),
        "n_head_delta_seeds": sum(
            1 for r in test if r["head_weight_delta_norm"] > HEAD_DELTA_MIN),
        "kth_test_bias_range": _kth_largest(
            [r["bias_range_high_threat"] for r in test], MIN_PASS_SEEDS),
        "kth_test_head_delta": _kth_largest(
            [r["head_weight_delta_norm"] for r in test], MIN_PASS_SEEDS),
        "max_test_head_delta": max((r["head_weight_delta_norm"] for r in test), default=0.0),
        "max_test_bias_range": max((r["bias_range_high_threat"] for r in test), default=0.0),
        "max_test_bias_range_devalued": max((r["bias_range_devalued"] for r in test), default=0.0),
        "max_test_excluded_count": max((int(r.get("f_eligibility_excluded_count", -1)) for r in test), default=-1),
        "max_test_between_tv": max((r["between_context_selection_tv"] for r in test), default=0.0),
        "max_test_deval_shift": max((r["devaluation_selection_shift"] for r in test), default=0.0),
        "max_ceiling_deval_shift": max((r["devaluation_selection_shift"] for r in ceiling), default=0.0),
        "max_ceiling_between_tv": max((r["between_context_selection_tv"] for r in ceiling), default=0.0),
        "acceptance": {
            "C1_devaluation_behavioural_shift": c1,
            "C2_discrimination_behavioural_separation": c2,
            "C3_silence_control": c3,
            "n_c1_seeds": c1_seeds,
            "n_c2_seeds": c2_seeds,
            "n_c3_seeds": c3_seeds,
            "pass": overall_pass,
        },
    }


def _interpretation(result: Dict) -> Dict:
    acc = result["acceptance"]
    ready = bool(result["readiness_met"])
    if not ready:
        label = "substrate_not_ready_requeue"
    elif acc["pass"]:
        label = "sd033b_demotion_conversion_supported"
    else:
        label = "conversion_ceiling_persists_despite_demotion"

    preconditions = [
        {
            "name": "ofc_trained_head_bias_cross_candidate_range_supra_dv_floor",
            "description": (
                "ARM_2 trained-head OFC bias cross-candidate RANGE (max-min over the "
                "real candidate bank) at the high-threat positive-control state clears "
                "BIAS_RANGE_FLOOR (0.05 == DEVAL_SHIFT_MARGIN, the same statistic the "
                "load-bearing DVs route on). RANGE not magnitude. Below floor -> "
                "substrate_not_ready_requeue, NEVER a weakens."
            ),
            # k-OF-n, DECLARED AS SUCH. The shipped readiness leg is the per-seed
            # `bias_range_high_threat > BIAS_RANGE_FLOOR`, counted and required on
            # >= MIN_PASS_SEEDS seeds; `measured` is therefore the MIN_PASS_SEEDS-th
            # LARGEST per-seed range, for which `measured > threshold` is exactly
            # "at least MIN_PASS_SEEDS seeds cleared the floor". It previously
            # reported max-over-seeds, which the indexer's AUTHORITATIVE recompute
            # reads as met on a SINGLE clearing seed. Strict floor (`>`), declared
            # rather than left to the inclusive default.
            "measured": float(result["kth_test_bias_range"]),
            "threshold": BIAS_RANGE_FLOOR,
            "comparator": ">",
            "direction": "lower",
            "seeds_required": MIN_PASS_SEEDS,
            "seeds_clearing_floor": int(result["n_bias_range_seeds"]),
            "max_over_seeds_diagnostic": float(result["max_test_bias_range"]),
            "control": "ARM_2 trained-arm bias over a real candidate bank at the high-threat state",
            "met": result["n_bias_range_seeds"] >= MIN_PASS_SEEDS,
        },
        {
            "name": "ofc_head_weight_delta_supra_floor",
            "description": (
                "ARM_2 state_bias_head weight-delta-from-init L2 norm clears "
                "HEAD_DELTA_MIN on >= MIN_PASS_SEEDS seeds -- the head genuinely "
                "trained under the paired threat-conditioned driver."
            ),
            # k-OF-n, DECLARED AS SUCH (see the bias-range entry above): `measured`
            # is the MIN_PASS_SEEDS-th LARGEST per-seed head delta, for which
            # `measured > threshold` is exactly "at least MIN_PASS_SEEDS seeds
            # cleared the floor". It previously reported max-over-seeds, which the
            # indexer's authoritative recompute reads as met on a SINGLE clearing
            # seed -- strictly looser than the shipped gate. Strict floor (`>`),
            # declared rather than left to the inclusive default.
            "measured": float(result["kth_test_head_delta"]),
            "threshold": HEAD_DELTA_MIN,
            "comparator": ">",
            "direction": "lower",
            "seeds_required": MIN_PASS_SEEDS,
            "seeds_clearing_floor": int(result["n_head_delta_seeds"]),
            "max_over_seeds_diagnostic": float(result["max_test_head_delta"]),
            "control": "trainable-arm head trained on frozen-encoder state_code in P1",
            "met": result["n_head_delta_seeds"] >= MIN_PASS_SEEDS,
        },
        {
            "name": "mech448_f_eligibility_excluded_count_supra_zero",
            "description": (
                "MECH-448 NON-DEGENERACY: on ARM_2 the F->eligibility envelope actually "
                "EXCLUDED at least one candidate (f_eligibility_excluded_count > 0) AND "
                "the demotion path was active -- F was genuinely demoted on a divergent "
                "F pool, not an all-admit no-op -- on >= MIN_PASS_SEEDS seeds. "
                "excluded_count == 0 (non-divergent F) -> substrate_not_ready_requeue, "
                "NEVER a weakens. `measured` is the COUNT of seeds satisfying that "
                "per-seed conjunction; `threshold` is MIN_PASS_SEEDS."
            ),
            # WAS UNRECOMPUTABLE, and in the MISSED_UNMET direction: it reported
            # `max_test_excluded_count` against a threshold of 0.0, and under the
            # indexer's default inclusive floor `measured >= 0.0` holds for every
            # non-negative count -- the bound could not discriminate AT ALL, so a
            # genuine premise failure recomputed as MET. `met` was meanwhile gated
            # on `nondegen_seeds >= MIN_PASS_SEEDS`: a k-of-n count over a PER-SEED
            # CONJUNCTION (excluded_count > 0 AND demotion_active) whose second leg
            # was absent from the entry entirely.
            #
            # Declared here as that COUNT itself, which is exact and reproducible --
            # `nondegen_seeds >= MIN_PASS_SEEDS` IS the shipped predicate, verbatim.
            # A per-leg split (the usual conjunction fix) was rejected: unlike
            # `all()`, a k-of-n count does NOT distribute over a conjunction -- two
            # legs each cleared by k seeds can be cleared by DIFFERENT seeds while
            # the conjunction is cleared by fewer -- so splitting would have been
            # strictly LOOSER than the shipped gate. The raw max excluded count is
            # kept as a non-bound diagnostic key; extra keys the recompute ignores.
            "measured": float(result["nondegen_seeds"]),
            "threshold": float(MIN_PASS_SEEDS),
            "comparator": ">=",
            "direction": "lower",
            "max_excluded_count_over_seeds_diagnostic": float(result["max_test_excluded_count"]),
            "control": "ARM_2 demotion-on E3.select diagnostics on the trained-agent eval bank",
            "met": result["nondegen_seeds"] >= MIN_PASS_SEEDS,
        },
    ]
    criteria_non_degenerate = {
        "C1": bool(acc["C1_devaluation_behavioural_shift"]) and ready,
        "C2": bool(acc["C2_discrimination_behavioural_separation"]) and ready,
        "C3": bool(acc["C3_silence_control"]),
    }
    criteria = [
        {"name": "C1_devaluation_behavioural_shift", "load_bearing": True, "passed": bool(acc["C1_devaluation_behavioural_shift"])},
        {"name": "C2_discrimination_behavioural_separation", "load_bearing": True, "passed": bool(acc["C2_discrimination_behavioural_separation"])},
        {"name": "C3_silence_control", "load_bearing": True, "passed": bool(acc["C3_silence_control"])},
    ]
    return {
        "label": label,
        "preconditions": preconditions,
        "criteria_non_degenerate": criteria_non_degenerate,
        "criteria": criteria,
        "note": (
            "PRIMARY DVs measured on the COMMITTED selection through the MECH-448 "
            "demotion-enabled E3 selector (SelectionResult.selected_index), forced to "
            "the deterministic argmin-within-eligible path. Conversion is attributable "
            "to the demotion lever via the 3-arm dissociation: ARM_2 (trained head + "
            "demotion) must convert where ARM_1 (trained head, demotion off = "
            "F-dominance ceiling) and ARM_0 (frozen head + demotion = silence) do not. "
            "Below the bias-range floor OR with excluded_count == 0 -> "
            "substrate_not_ready_requeue (non_contributory), NEVER a false SD-033b/"
            "MECH-263 weakens. Readiness-met + DVs-fail is an honest result "
            "(conversion_ceiling_persists_despite_demotion -> /failure-autopsy / "
            "MECH-449 follow-on before stamping)."
        ),
    }


def _evidence_direction(result: Dict) -> str:
    if not result["readiness_met"]:
        return "non_contributory"          # substrate_not_ready_requeue, NOT a weakens
    return "supports" if result["acceptance"]["pass"] else "weakens"


def write_manifest(result: Dict, dry_run: bool, elapsed: float) -> Tuple[Path, str]:
    acc = result["acceptance"]
    interpretation = _interpretation(result)
    direction = _evidence_direction(result)
    outcome = "PASS" if (acc["pass"] and result["readiness_met"]) else "FAIL"
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
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "executing_hostname": socket.gethostname(),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "result": outcome,
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {
            "SD-033b": direction,
            "MECH-263": direction,
        },
        "non_degenerate": bool(result["readiness_met"]),
        "degeneracy_reason": (
            None if result["readiness_met"]
            else "substrate_not_ready_requeue: ARM_2 trained-head OFC bias range below "
                 "floor, head untrained, or MECH-448 envelope admitted all candidates "
                 "(excluded_count == 0) -> the behavioural test never ran"
        ),
        "dry_run": dry_run,
        "elapsed_sec": elapsed,
        "acceptance": acc,
        "interpretation": interpretation,
        "readiness_met": result["readiness_met"],
        "ready_seeds": result["ready_seeds"],
        "nondegen_seeds": result["nondegen_seeds"],
        "n_seeds": result["n_seeds"],
        "max_test_head_delta": result["max_test_head_delta"],
        "max_test_bias_range": result["max_test_bias_range"],
        "max_test_bias_range_devalued": result["max_test_bias_range_devalued"],
        "max_test_excluded_count": result["max_test_excluded_count"],
        "max_test_between_context_tv": result["max_test_between_tv"],
        "max_test_deval_shift": result["max_test_deval_shift"],
        "max_ceiling_deval_shift": result["max_ceiling_deval_shift"],
        "max_ceiling_between_tv": result["max_ceiling_between_tv"],
        "arm_results": [r for arm in result["by_arm"].values() for r in arm],
        "substrate_under_test": (
            "SD-033b train_state_bias_head + ofc_bias_scale=0.5 + SD-056 "
            "e2_action_contrastive P0 + paired threat-conditioned driver, ROUTED "
            "through the MECH-448 (ARC-107) rank-preserving F->eligibility "
            "demotion-enabled E3 selector (use_f_eligibility_demotion / "
            "f_eligibility_envelope_floor=0.30 / f_eligibility_dn_sigma=0.0)"
        ),
        "unblocks": (
            "commitment_closure:GAP-8 (SD-033b candidate -> provisional behavioural "
            "evidence); contributes to behavioral_diversity_isolation:GAP-I"
        ),
        "supersedes": "V3-EXQ-485h",
        "predecessor": (
            "V3-EXQ-485h (RAN FAIL / non_contributory 2026-06-19; MECH-439 F-dominance "
            "conversion-ceiling signature -- trained OFC head, real bias range ~0.50, "
            "ZERO conversion on the isolated softmax / F-dominated selector)"
        ),
        "predecessors": [
            "V3-EXQ-485b (devaluation, representation-level)",
            "V3-EXQ-485c (task-role discrimination, representation-level)",
            "V3-EXQ-485d (trained-OFC-head substrate readiness)",
            "V3-EXQ-485e (trained-OFC-head behavioural; non_contributory range-starved)",
            "V3-EXQ-485f (trained-OFC-head behavioural; non_contributory readiness-gate miscalibration)",
            "V3-EXQ-485g (trained-OFC-head behavioural; substrate_ceiling, first non-vacuous test, zero conversion)",
            "V3-EXQ-485h (disambiguating redesign; FAIL/non_contributory, F-dominance conversion ceiling)",
        ],
        "notes": (
            "Evidence-grade trained-OFC-head BEHAVIOURAL validation of MECH-263 "
            "signatures a (devaluation sensitivity) + b (task-role discrimination) on "
            "the MECH-448 demotion-enabled E3 selector. WHY NOW: the 2026-06-21 "
            "governance cycle promoted MECH-448 to provisional (V3-EXQ-689d PASS), "
            "operationally lifting the shared MECH-439 F-dominance conversion ceiling "
            "that 485h's behavioural DVs were stuck under. 485i re-targets the PRIMARY "
            "C1 devaluation_selection_shift + C2 between-context TV (names unchanged) "
            "from 485h's isolated OFC softmax to the COMMITTED selection through the "
            "real E3.select() with the OFC bias as score_bias -- where F dominates and "
            "where MECH-448 demotes it to eligibility, letting the OFC channel arbitrate "
            "the committed action with F removed from the final argmin. 3-arm "
            "dissociation attributes conversion to the CONJUNCTION trained-head AND "
            "demotion: ARM_2 (test) must convert where ARM_1 (trained, demotion off = "
            "F-dominance ceiling) and ARM_0 (frozen head + demotion = silence) do not. "
            "Same-statistic non-vacuity gate (OFC bias range > 0.05) PLUS a MECH-448 "
            "non-degeneracy gate (f_eligibility_excluded_count > 0 = F genuinely "
            "demoted on a divergent pool) self-route substrate_not_ready_requeue "
            "(non_contributory) below floor, NEVER a false SD-033b/MECH-263 weakens. "
            "PROMOTES NOTHING by itself; a PASS contributes provisional behavioural "
            "evidence and helps close behavioral_diversity_isolation:GAP-I."
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    t0 = time.time()
    _preflight()
    seeds = args.seeds if args.seeds is not None else ([0] if args.dry_run else [0, 1, 2])
    result = run(seeds=seeds, dry_run=args.dry_run)
    elapsed = time.time() - t0
    out_path, outcome = write_manifest(result, args.dry_run, elapsed)
    acc = result["acceptance"]
    interp = _interpretation(result)
    print("\n=== V3-EXQ-485i SUMMARY ===", flush=True)
    print(f"  readiness_met={result['readiness_met']} (ready_seeds={result['ready_seeds']}/{result['n_seeds']}, nondegen_seeds={result['nondegen_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  max_test_bias_range={result['max_test_bias_range']:.6f} (floor {BIAS_RANGE_FLOOR})", flush=True)
    print(f"  max_test_excluded_count={result['max_test_excluded_count']}", flush=True)
    print(f"  ARM_2 deval_shift={result['max_test_deval_shift']:.4f} vs ARM_1 ceiling={result['max_ceiling_deval_shift']:.4f}", flush=True)
    print(f"  ARM_2 between_tv={result['max_test_between_tv']:.4f} vs ARM_1 ceiling={result['max_ceiling_between_tv']:.4f}", flush=True)
    print(f"  C1_devaluation_behavioural_shift={acc['C1_devaluation_behavioural_shift']} ({acc['n_c1_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  C2_discrimination_behavioural_separation={acc['C2_discrimination_behavioural_separation']} ({acc['n_c2_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  C3_silence_control={acc['C3_silence_control']} ({acc['n_c3_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  interpretation: {interp['label']}", flush=True)
    print(f"  evidence_direction: {_evidence_direction(result)}", flush=True)
    print(f"  outcome: {outcome}", flush=True)
    print(f"Result written to: {out_path}", flush=True)
    return out_path, outcome


if __name__ == "__main__":
    out_path, outcome = main()
    _raw = str(outcome).upper()
    emit_outcome(
        outcome=_raw if _raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
        dry_run="--dry-run" in sys.argv,
    )
