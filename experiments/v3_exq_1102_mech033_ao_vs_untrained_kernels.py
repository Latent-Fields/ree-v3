#!/opt/local/bin/python3
"""
V3-EXQ-1102 -- MECH-033 GOV-FANOUT-1 portfolio, LEG 4 (axis: instrumentation).

Claim:   MECH-033  "E2 forward-prediction kernels seed hippocampal rollouts."
Source:  failure_autopsy_gflag0452-D2-cluster_2026-09-24, targets[5] (V3-EXQ-308),
         fanout_recommendation.suggested_probes[3] (hypothesis "H1").
Design:  REE_assembly/evidence/planning/mech033_fanout_portfolio_design_blocked_20260925.md
         (ratified 2026-09-25 by orchestrate-20260924-1707 under rec-20260924-fb429c72,
         comparator option A-i).

WHAT THIS LEG IS FOR
--------------------
This is the only leg that runs the REAL pipeline: the full REEAgent with its
HippocampalModule, CEM planning in action-object space, residue-field trajectory
scoring and E3 selection. The autopsy's "integration: isolated (inline toy
architecture, not the REEAgent pipeline)" finding against V3-EXQ-308 is answered
HERE; the sibling legs V3-EXQ-1103 (depth) and V3-EXQ-1101 (E2 scramble) run the
inline 308-lineage stack on purpose, because their manipulations are cleanly
controllable there.

It is V3-EXQ-055's design -- one trained agent, several eval conditions at matched
CEM budget -- with its comparator replaced and its single seed replicated.

WHY SELF_CHAIN IS NOT THE COMPARATOR (the measurement that forced this change)
------------------------------------------------------------------------------
The autopsy's sketch asked for AO kernels "vs seeded by random kernels at matched
CEM budget (the V3-EXQ-055 AO_CHAIN vs SELF_CHAIN design, with SELF_CHAIN first
checked for random-level degeneracy)". That check was run on 2026-09-25 against
055's recorded manifest (v3_exq_055_..._20260320T191345Z.json) and SELF_CHAIN
FAILED it:

    harm_per_step   self 0.0684271 vs random 0.0754013  -> 90.8% of random
    contact_rate    self 0.0527426 vs random 0.0605405  -> 87.1% of random
    cal_gap_approach self 0.0213    -> BELOW 055's own 0.03 C4 calibration floor
    (single seed, seed=0, unreplicated)

SELF_CHAIN rolls out in z_self space, which carries no harm-relevant
action-consequence structure, so it is planning-shaped noise: the same CLASS of
comparator as V3-EXQ-308's uniform-random NO_CHAIN. Using it as this leg's null
would have re-imported the D2 ablation confound the portfolio exists to remove.

So the null's comparator is UNTRAINED_E2_CHAIN (ratified option A-i): the entire
pipeline is held intact and ONLY E2's trained forward-prediction weights are
replaced by a fresh untrained init. Untrained E2 still produces
ACTION-CONDITIONED outputs, so CEM can still rank candidates -- the arm is a
competent planner working from worthless kernels, which is exactly the
isolation MECH-033's own claims.yaml what_would_answer asks for ("Ablate the
E2-kernel-to-hippocampal handoff ... while leaving rollouts and E3 scoring
otherwise intact").

SELF_CHAIN is still RUN and RECORDED, as an informational arm with its
degeneracy stated, so this leg also re-measures the 055 reading at 3 seeds.

ARMS (one trained agent per seed; all four eval conditions on that agent)
------------------------------------------------------------------------
    AO_CHAIN            -- the real pipeline: agent.generate_trajectories +
                           agent.select_action. Hippocampal CEM over
                           action-object kernels from the TRAINED E2.      (H1)
    UNTRAINED_E2_CHAIN  -- identical pipeline and identical CEM budget, with
                           e2.world_transition + e2.world_action_encoder
                           replaced by a fresh untrained init. THE NULL'S ARM.
    SELF_CHAIN          -- 055's CEM in z_self space at matched budget.
                           INFORMATIONAL: degenerate per the measurement above.
    RANDOM_REF          -- uniform random actions. INFORMATIONAL ONLY:
                           competence floor + data quality. NEVER the null's
                           comparator.

Why those two E2 submodules: e2.world_forward(z, a) is
`z + world_transition(cat[z, world_action_encoder(a)])`, and
rollout_with_world() advances z_world through it at every rollout step -- so the
action_object sequence handed to the hippocampal map, and the world_state
sequence the residue field scores, both depend on it. Replacing exactly those two
degrades E2's forward prediction and nothing else.

DECLARED NULL
-------------
    harm_rate(AO_CHAIN) is NOT below harm_rate(UNTRAINED_E2_CHAIN) by >= 0.01.
H1 predicts this null is rejected. H2/H3 predict it holds.

GFLAG-0491 TRAINER DECLARATION (and an honest disclosure)
--------------------------------------------------------
At REEConfig defaults the agent does NO waking gradient learning: every trained
parameter comes from a DRIVER-built optimizer. This driver builds three, exactly
as V3-EXQ-055 did, and they are named here so the trained set is legible:
    main_optimizer    -> everything except the wf and terrain param sets;
                         loss = E3 harm_eval BCE + E1 prediction loss.
    wf_optimizer      -> e2.world_transition + e2.world_action_encoder;
                         loss = MSE(world_forward(z_t, a_t), z_{t+1}).
                         THIS is MECH-033's kernel, and it is genuinely trained
                         (055 recorded world_forward_r2 0.9889).
    terrain_optimizer -> hippocampal.terrain_prior +
                         hippocampal.action_object_decoder.

DISCLOSED, not silently inherited: `e2.action_object_head` receives NO gradient
in this recipe (it sits in main_optimizer's param list but neither the harm BCE
nor the E1 loss reaches it), so the action-object ENCODING is at random init
throughout -- consistent with GFLAG-0491's finding that >= 6 act-path modules sit
at random init in every recipe. This is a property of the 055 recipe that this
leg deliberately does NOT change, because changing it would break comparability
with the run this leg exists to redo correctly. It bounds the reading: this leg
tests the contribution of E2's TRAINED FORWARD PREDICTION within a recipe whose
action-object encoder is untrained. It does NOT confound the contrast -- both
arms share that untrained encoder; only the forward-prediction weights differ.

GFLAG-0485 is handled by the C4 canary plus the recorded
e2_action_discrimination on both arms: if trained E2 discriminates actions no
better than untrained E2, this leg cannot discriminate anything and says so.

NEGATIVE-INSTRUMENT DISCIPLINE (CLAUDE.md "Negative instruments")
-----------------------------------------------------------------
  C3 TRAINING GATE    -- trained world_forward_r2 >= 0.20.
  C4 CANARY           -- untrained world_forward_r2 <= 0.05, so a null cannot be
                         reached by an ablation that did not bite.
  C5 COMPETENCE FLOOR -- AO_CHAIN must beat RANDOM_REF by >= 15%. If the real
                         pipeline is itself at random level there is no
                         contribution to ablate.
  C6 DATA QUALITY     -- RANDOM_REF must accumulate >= 5 harm events.

PRE-REGISTERED CRITERIA
-----------------------
C1: mean over seeds of (harm_rate_UNTRAINED_E2 - harm_rate_AO_CHAIN) >= 0.01  [rejects the null]
C2: harm_rate_AO_CHAIN < harm_rate_UNTRAINED_E2 on ALL seeds                  [consistency]
C3: world_forward_r2_trained >= 0.20 on ALL seeds                             [E2 trained]
C4: world_forward_r2_untrained <= 0.05 on ALL seeds                           [ablation canary]
C5: harm_rate_AO_CHAIN <= 0.85 * harm_rate_RANDOM_REF on ALL seeds            [competence floor]
C6: n_harm_events_RANDOM_REF >= 5 on ALL seeds                                [env presents harm]

PASS = C1..C6 all hold -> supports H1: E2's trained forward prediction is
load-bearing for harm avoidance in the real hippocampal planning pipeline. This
would be the first non-confounded support MECH-033 has had in five attempts.
Evidence-direction logic (evaluated in this order):
  C3 or C4 or C6 fails                 -> inconclusive (instrument inadequate)
  C5 fails                             -> inconclusive (pipeline degenerate;
                                          nothing to ablate; NOT support for H2/H3)
  C3..C6 hold and C1 or C2 fails        -> weakens (E2's trained kernels are not
                                          carrying harm avoidance in the real pipeline)
  all hold                             -> supports

PROTOCOL
--------
3 seeds (055 used 1, which the autopsy's GOV-REUSE-1 note flagged) x 600 warmup
episodes x 200 steps/episode, then 4 eval arms x 50 episodes. Env and REEConfig
are V3-EXQ-055's verbatim. Warmup budget 600 episodes matches the control
MECH-033's own what_would_answer specifies. This is the expensive leg --
estimated ~4-7 h; route to a cloud worker, not the laptop.
"""

import sys
import copy
import time
import random
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_1102_mech033_ao_vs_untrained_kernels"
QUEUE_ID = "V3-EXQ-1102"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-033"]
SOURCE_AUTOPSY = (
    "REE_assembly/evidence/planning/failure_autopsy_gflag0452-D2-cluster_2026-09-24.json"
    "#targets[5].fanout_recommendation.suggested_probes[3]"
)
DESIGN_DOC = (
    "REE_assembly/evidence/planning/"
    "mech033_fanout_portfolio_design_blocked_20260925.md"
)
PREDECESSOR_RUN = "v3_exq_055_mech033_kernel_chaining_20260320T191345Z_v3"

# --------------------------------------------------------------------------
# Pre-registered thresholds
# --------------------------------------------------------------------------
THRESH_C1_MIN_DEGRADATION = 0.01   # C1: mean(harm_UNTRAINED - harm_AO) >= 1 pp
THRESH_C3_E2_R2           = 0.20   # C3: trained world_forward_r2
THRESH_C4_ABLATED_R2_MAX  = 0.05   # C4: untrained world_forward_r2 (CANARY)
THRESH_C5_COMPETENCE      = 0.85   # C5: harm_AO <= 0.85 * harm_RANDOM_REF
THRESH_C6_MIN_CONTACTS    = 5      # C6: RANDOM_REF harm events

# --------------------------------------------------------------------------
# Protocol constants (V3-EXQ-055's, verbatim, except SEEDS)
# --------------------------------------------------------------------------
SEEDS             = [0, 7, 13]
WARMUP_EPISODES   = 600
EVAL_EPISODES     = 50
STEPS_PER_EPISODE = 200
LR                = 1e-3
SELF_DIM          = 32
WORLD_DIM         = 32
ALPHA_WORLD       = 0.9
ALPHA_SELF        = 0.3
HARM_SCALE        = 0.02
PROXIMITY_SCALE   = 0.05
R2_SAMPLE         = 200
DISCRIM_STEPS     = 200

# Matched CEM budget -- identical for AO_CHAIN, UNTRAINED_E2_CHAIN and SELF_CHAIN.
CANDIDATE_HORIZON = 5
N_CEM_CANDIDATES  = 8
N_CEM_ITERATIONS  = 3
ELITE_FRACTION    = 0.3

APPROACH_TTYPES = {"hazard_approach"}
CONTACT_TTYPES  = {"agent_caused_hazard", "env_caused_hazard"}

ARM_AO         = "AO_CHAIN"
ARM_UNTRAINED  = "UNTRAINED_E2_CHAIN"
ARM_SELF       = "SELF_CHAIN"
ARM_RANDOM_REF = "RANDOM_REF"
ARMS = [ARM_AO, ARM_UNTRAINED, ARM_SELF, ARM_RANDOM_REF]

# The two E2 submodules that constitute world_forward. See the module docstring.
WF_SUBMODULES = ("world_transition", "world_action_encoder")

SCOPE_STATEMENTS = [
    "SELF_CHAIN is INFORMATIONAL, never the null's comparator: measured at 90.8% "
    "of random harm and 87.1% of random contact_rate in V3-EXQ-055, with "
    "cal_gap 0.0213 below 055's own 0.03 floor. The null's comparator is "
    "UNTRAINED_E2_CHAIN (ratified option A-i).",
    "e2.action_object_head receives no gradient in this recipe (inherited from "
    "V3-EXQ-055 deliberately, for comparability), so the action-object ENCODING "
    "is at random init in BOTH arms. It bounds the reading -- this leg tests E2's "
    "trained FORWARD PREDICTION -- but it does not confound the contrast, since "
    "only the forward-prediction weights differ between the arms.",
    "CEM budget IS matched across all three planning arms (8 candidates, 3 "
    "iterations, horizon 5), because here the manipulation is kernel QUALITY. "
    "Contrast leg V3-EXQ-1103, where the manipulation is rollout DEPTH and "
    "compute therefore differs by construction.",
    "The ablation is applied at EVAL only, to deep-copied state, and the trained "
    "weights are restored and re-measured afterwards (recorded as "
    "world_forward_r2_trained_after_restore).",
    "3 seeds rather than V3-EXQ-055's 1, per the autopsy's GOV-REUSE-1 note that "
    "055 is unreplicated.",
    "V3-EXQ-308 is deliberately NOT re-lettered: a re-letter would inherit the "
    "shared uniform-random comparator confound.",
]

_T0 = time.perf_counter()


def _mean_safe(lst: list, default: float = 0.0) -> float:
    return float(sum(lst) / len(lst)) if lst else default


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _make_env(seed: int) -> CausalGridWorldV2:
    """V3-EXQ-055's env configuration, verbatim."""
    return CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=HARM_SCALE,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=PROXIMITY_SCALE,
        proximity_benefit_scale=PROXIMITY_SCALE * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )


# --------------------------------------------------------------------------
# SELF_CHAIN (informational) -- V3-EXQ-055's _cem_in_self_space, matched budget
# --------------------------------------------------------------------------

def _cem_in_self_space(agent, z_self, z_world) -> torch.Tensor:
    """CEM in z_self space at the SAME budget as AO_CHAIN. Carried forward from
    V3-EXQ-055 unchanged so this leg re-measures 055's own reading at 3 seeds.
    INFORMATIONAL ONLY -- see the module docstring for why it is not the null's
    comparator."""
    n_elite = max(1, int(N_CEM_CANDIDATES * ELITE_FRACTION))
    device = z_self.device
    action_dim = agent.e2.config.action_dim

    a_mean = torch.zeros(1, CANDIDATE_HORIZON, action_dim, device=device)
    a_std = torch.ones(1, CANDIDATE_HORIZON, action_dim, device=device)

    best_action = None
    best_score = float("inf")

    for _ in range(N_CEM_ITERATIONS):
        scores: List[float] = []
        actions_list: List[torch.Tensor] = []
        for _ in range(N_CEM_CANDIDATES):
            noise = torch.randn(1, CANDIDATE_HORIZON, action_dim, device=device)
            action_seq = a_mean + a_std * noise
            actions_list.append(action_seq)
            traj = agent.e2.rollout_with_world(
                z_self, z_world, action_seq, compute_action_objects=False
            )
            ws = traj.get_world_state_sequence()
            if ws is not None and not torch.isnan(ws).any():
                score = float(agent.residue_field.evaluate_trajectory(ws).sum().item())
            else:
                score = float("inf")
            scores.append(score)
            if score < best_score:
                best_score = score
                best_action = action_seq[:, 0, :].detach()
        idxs = sorted(range(len(scores)), key=lambda i: scores[i])[:n_elite]
        elite = torch.cat([actions_list[i] for i in idxs], dim=0)
        a_mean = elite.mean(dim=0, keepdim=True)
        a_std = elite.std(dim=0, keepdim=True) + 1e-6

    if best_action is None:
        best_action = torch.randn(1, action_dim, device=device)
    return best_action


# --------------------------------------------------------------------------
# THE MANIPULATION: replace E2's trained forward-prediction weights
# --------------------------------------------------------------------------

def _wf_state(agent) -> Dict[str, Dict]:
    """Deep-copied state_dicts of the two submodules that make up world_forward."""
    return {
        name: copy.deepcopy(getattr(agent.e2, name).state_dict())
        for name in WF_SUBMODULES
    }


def _load_wf_state(agent, state: Dict[str, Dict]) -> None:
    for name, sd in state.items():
        getattr(agent.e2, name).load_state_dict(sd)


def _fresh_wf_state(agent, seed: int) -> Dict[str, Dict]:
    """
    An UNTRAINED initialisation for the two world_forward submodules, produced by
    re-constructing modules of the same shapes under a dedicated generator.

    Built from the trained shapes so it cannot silently diverge from the live
    architecture, and deterministic in `seed` so the arm is reproducible.
    """
    g = torch.Generator().manual_seed(seed + 31337)
    out: Dict[str, Dict] = {}
    for name in WF_SUBMODULES:
        sd = copy.deepcopy(getattr(agent.e2, name).state_dict())
        fresh: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            t = torch.empty_like(v)
            if v.dim() > 1:
                # Kaiming-uniform-equivalent bound from fan_in, generator-driven.
                fan_in = v.shape[1]
                bound = (6.0 / max(fan_in, 1)) ** 0.5
                t.uniform_(-bound, bound, generator=g)
            else:
                t.uniform_(-0.1, 0.1, generator=g)
            fresh[k] = t
        out[name] = fresh
    return out


def _selfcheck_ablation(agent, seed: int) -> Dict:
    """
    Measure the blind spot (CLAUDE.md "The test half").

    (1) The ablation must actually CHANGE world_forward's output. An ablation
        helper that returned the trained weights would make the whole leg
        vacuous: both arms would be one arm and the null would hold by
        construction.
    (2) Restoring must return world_forward to bit-identical output, so the
        AO_CHAIN arm cannot be contaminated by an earlier ablated arm.
    Raises rather than warns: a broken manipulation must not produce a
    publishable manifest.
    """
    device = agent.device
    action_dim = agent.e2.config.action_dim
    z = torch.randn(1, WORLD_DIM, device=device, generator=None)
    a = _action_to_onehot(0, action_dim, device)

    trained = _wf_state(agent)
    with torch.no_grad():
        out_trained = agent.e2.world_forward(z, a).clone()

    _load_wf_state(agent, _fresh_wf_state(agent, seed))
    with torch.no_grad():
        out_ablated = agent.e2.world_forward(z, a).clone()

    _load_wf_state(agent, trained)
    with torch.no_grad():
        out_restored = agent.e2.world_forward(z, a).clone()

    delta = float((out_ablated - out_trained).abs().max().item())
    assert delta > 1e-6, (
        "ablation did not change world_forward's output -- AO_CHAIN and "
        "UNTRAINED_E2_CHAIN would be the same arm and the null would hold by "
        "construction"
    )
    assert torch.equal(out_restored, out_trained), (
        "restore did not return world_forward to bit-identical output -- a later "
        "arm would run on contaminated weights"
    )
    print(
        "[selfcheck] ablation OK: max|delta| on world_forward output = %.5f; "
        "restore is bit-identical" % delta,
        flush=True,
    )
    return {
        "ablation_changes_world_forward": True,
        "max_abs_output_delta": delta,
        "restore_bit_identical": True,
    }


# --------------------------------------------------------------------------
# Instruments
# --------------------------------------------------------------------------

def _wf_r2(agent, wf_buf: List[Tuple]) -> float:
    """world_forward R^2 on held transitions (V3-EXQ-055's estimator)."""
    if len(wf_buf) < 32:
        return 0.0
    with torch.no_grad():
        k = min(R2_SAMPLE, len(wf_buf))
        idxs = torch.randperm(len(wf_buf))[:k].tolist()
        zw_t = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
        a_t = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
        zw_t1 = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
        pred = agent.e2.world_forward(zw_t, a_t)
        ss_res = (zw_t1 - pred).pow(2).sum().item()
        ss_tot = (zw_t1 - zw_t1.mean(0, keepdim=True)).pow(2).sum().item()
        return float(max(-1.0, 1.0 - ss_res / (ss_tot + 1e-8)))


def _e2_action_discrimination(agent, wf_buf: List[Tuple]) -> float:
    """
    GFLAG-0485 CANARY. Mean pairwise L2 distance between world_forward's
    predictions across the action set, normalised by mean prediction norm.

    Near 0 means E2 is action-INVARIANT: every candidate predicts the same
    future, so no kernel-quality contrast can discriminate. Recorded for BOTH
    arms, so "trained E2 discriminates no better than untrained E2" is visible
    rather than inferred.
    """
    if not wf_buf:
        return 0.0
    action_dim = agent.e2.config.action_dim
    ratios: List[float] = []
    with torch.no_grad():
        k = min(DISCRIM_STEPS, len(wf_buf))
        idxs = torch.randperm(len(wf_buf))[:k].tolist()
        for i in idxs:
            z = wf_buf[i][0].to(agent.device)
            preds = [
                agent.e2.world_forward(z, _action_to_onehot(a, action_dim, agent.device))
                for a in range(action_dim)
            ]
            dists = [
                float((preds[x] - preds[y]).norm().item())
                for x in range(action_dim) for y in range(x + 1, action_dim)
            ]
            mean_norm = sum(float(p.norm().item()) for p in preds) / action_dim
            if mean_norm > 1e-8 and dists:
                ratios.append((sum(dists) / len(dists)) / mean_norm)
    return float(sum(ratios) / len(ratios)) if ratios else 0.0


# --------------------------------------------------------------------------
# Training -- V3-EXQ-055's recipe, three named driver-built optimizers
# --------------------------------------------------------------------------

def _train_agent(seed: int, dry_run: bool) -> Tuple:
    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        alpha_self=ALPHA_SELF,
        reafference_action_dim=env.action_dim,
    )
    agent = REEAgent(config)

    wf_params = (
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters())
    )
    terrain_params = (
        list(agent.hippocampal.terrain_prior.parameters())
        + list(agent.hippocampal.action_object_decoder.parameters())
    )
    wf_ids = set(id(p) for p in wf_params)
    terrain_ids = set(id(p) for p in terrain_params)
    main_params = [
        p for p in agent.parameters()
        if id(p) not in wf_ids and id(p) not in terrain_ids
    ]
    optimizer = optim.Adam(main_params, lr=LR)
    wf_optimizer = optim.Adam(wf_params, lr=1e-3)
    terrain_optimizer = optim.Adam(terrain_params, lr=5e-4)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_HARM_BUF = 1000
    MAX_WF_BUF = 2000

    warmup = WARMUP_EPISODES if not dry_run else 3
    steps = STEPS_PER_EPISODE if not dry_run else 25

    print("[V3-EXQ-1102] seed=%d Phase 1: training %d eps (full pipeline)"
          % (seed, warmup), flush=True)

    agent.train()
    e3_tick_total = 0

    for ep in range(warmup):
        _flat, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        z_self_prev: Optional[torch.Tensor] = None

        for step in range(steps):
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, WORLD_DIM, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            if ticks.get("e3_tick", False) and candidates:
                e3_tick_total += 1
                result = agent.e3.select(candidates, temperature=1.0)
                action = result.selected_action.detach()
                agent._last_action = action
                selected_ao = result.selected_trajectory.get_action_object_sequence()
                if selected_ao is not None:
                    ao_mean_pred = agent.hippocampal._get_terrain_action_object_mean(
                        theta_z, e1_prior=e1_prior.detach()
                    )
                    terrain_loss = F.mse_loss(ao_mean_pred, selected_ao.detach())
                    terrain_optimizer.zero_grad()
                    terrain_loss.backward()
                    nn.utils.clip_grad_norm_(terrain_params, 1.0)
                    terrain_optimizer.step()
            else:
                action = agent._last_action
                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim,
                        agent.device,
                    )

            _flat, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if ttype in APPROACH_TTYPES | CONTACT_TTYPES:
                harm_buf_pos.append(theta_z.squeeze(0))
                harm_buf_pos = harm_buf_pos[-MAX_HARM_BUF:]
            else:
                harm_buf_neg.append(theta_z.squeeze(0))
                harm_buf_neg = harm_buf_neg[-MAX_HARM_BUF:]

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev, action_prev, z_world_curr))
                wf_buf = wf_buf[-MAX_WF_BUF:]
                with torch.no_grad():
                    z_pred = agent.e2.world_forward(z_world_prev, action_prev)
                    agent.e3.update_running_variance(z_world_curr - z_pred)

            z_world_prev = z_world_curr
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()

            if len(harm_buf_pos) >= 8 and len(harm_buf_neg) >= 8 and step % 8 == 0:
                k = min(16, len(harm_buf_pos), len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k].tolist()
                pos_z = torch.stack([harm_buf_pos[i] for i in pos_idx]).to(agent.device)
                neg_z = torch.stack([harm_buf_neg[i] for i in neg_idx]).to(agent.device)
                z_batch = torch.cat([pos_z, neg_z], dim=0)
                labels = torch.cat(
                    [torch.ones(k, 1), torch.zeros(k, 1)], dim=0
                ).to(agent.device)
                harm_loss = F.binary_cross_entropy(agent.e3.harm_eval(z_batch), labels)
                e1_loss = agent.compute_prediction_loss()
                total_loss = harm_loss + e1_loss
                if total_loss.requires_grad:
                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(main_params, 1.0)
                    optimizer.step()

            if len(wf_buf) >= 16 and step % 4 == 0:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_t = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_t = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw_t1 = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_t, a_t), zw_t1)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    nn.utils.clip_grad_norm_(wf_params, 1.0)
                    wf_optimizer.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup - 1:
            print("  seed=%d ep %d/%d e3_ticks=%d"
                  % (seed, ep + 1, warmup, e3_tick_total), flush=True)

    return agent, env, wf_buf, e3_tick_total


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _run_arm(agent, env, seed: int, arm: str, dry_run: bool) -> Dict:
    """Eval one arm. The ABLATION is applied by the caller (weights are swapped
    around this call), so this function is arm-agnostic apart from the
    action-selection branch."""
    torch.manual_seed(seed + 999)
    random.seed(seed + 999)

    eval_eps = EVAL_EPISODES if not dry_run else 2
    steps = STEPS_PER_EPISODE if not dry_run else 25

    total_harm = 0.0
    total_steps = 0
    contact_count = 0
    approach_count = 0
    harm_events = 0
    harm_scores_approach: List[float] = []
    harm_scores_none: List[float] = []

    agent.eval()
    with torch.no_grad():
        for _ep in range(eval_eps):
            _flat, obs_dict = env.reset()
            agent.reset()
            for _step in range(steps):
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks["e1_tick"]
                    else torch.zeros(1, WORLD_DIM, device=agent.device)
                )
                theta_z = agent.theta_buffer.summary()
                harm_score = float(agent.e3.harm_eval(theta_z).mean().item())

                if arm in (ARM_AO, ARM_UNTRAINED):
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                    action = agent.select_action(candidates, ticks)
                elif arm == ARM_SELF:
                    action = _cem_in_self_space(
                        agent, latent.z_self.detach(), theta_z
                    )
                else:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim,
                        agent.device,
                    )

                _flat, harm_signal, done, info, obs_dict = env.step(action)
                ttype = info.get("transition_type", "none")

                h = abs(float(harm_signal))
                total_harm += h
                total_steps += 1
                if h > 0.0:
                    harm_events += 1
                if ttype in CONTACT_TTYPES:
                    contact_count += 1
                if ttype in APPROACH_TTYPES:
                    approach_count += 1
                    harm_scores_approach.append(harm_score)
                elif ttype not in CONTACT_TTYPES:
                    harm_scores_none.append(harm_score)

                if done:
                    break

    harm_rate = total_harm / max(total_steps, 1)
    res = {
        "arm": arm,
        "seed": seed,
        "harm_rate": harm_rate,
        "contact_rate": contact_count / max(total_steps, 1),
        "contact_count": contact_count,
        "approach_count": approach_count,
        "n_harm_events": harm_events,
        "total_steps": total_steps,
        "cal_gap_approach": (
            _mean_safe(harm_scores_approach) - _mean_safe(harm_scores_none)
        ),
    }
    print(
        "  [%-18s] seed=%d harm_rate=%.6f contact_rate=%.6f n_harm_events=%d "
        "cal_gap=%.4f" % (arm, seed, harm_rate, res["contact_rate"],
                          harm_events, res["cal_gap_approach"]),
        flush=True,
    )
    return res


# --------------------------------------------------------------------------
# Criteria
# --------------------------------------------------------------------------

def _evaluate_criteria(results: Dict[str, List[Dict]], per_seed: Dict) -> Tuple:
    ao = results[ARM_AO]
    un = results[ARM_UNTRAINED]
    rnd = results[ARM_RANDOM_REF]
    n_s = len(SEEDS)

    deltas = [un[i]["harm_rate"] - ao[i]["harm_rate"] for i in range(n_s)]
    mean_delta = sum(deltas) / max(len(deltas), 1)

    c1 = mean_delta >= THRESH_C1_MIN_DEGRADATION
    c2 = all(ao[i]["harm_rate"] < un[i]["harm_rate"] for i in range(n_s))
    c3 = all(per_seed[s]["world_forward_r2_trained"] >= THRESH_C3_E2_R2 for s in SEEDS)
    c4 = all(
        per_seed[s]["world_forward_r2_untrained"] <= THRESH_C4_ABLATED_R2_MAX
        for s in SEEDS
    )
    c5 = all(
        ao[i]["harm_rate"] <= THRESH_C5_COMPETENCE * rnd[i]["harm_rate"]
        for i in range(n_s)
    )
    c6 = all(rnd[i]["n_harm_events"] >= THRESH_C6_MIN_CONTACTS for i in range(n_s))

    criteria = {
        "C1_mean_ablation_degradation_ge_0.01": c1,
        "C2_ao_below_untrained_all_seeds": c2,
        "C3_trained_world_forward_r2_ge_0.20": c3,
        "C4_ablation_canary_untrained_r2_le_0.05": c4,
        "C5_ao_competence_floor_vs_random_ref": c5,
        "C6_random_ref_harm_events_ge_5": c6,
    }
    status = "PASS" if all(criteria.values()) else "FAIL"

    self_ratio = _mean_safe([
        results[ARM_SELF][i]["harm_rate"] / rnd[i]["harm_rate"]
        for i in range(n_s) if rnd[i]["harm_rate"] > 0
    ])

    if not (c3 and c4 and c6):
        direction = "inconclusive"
        note = (
            "Instrument inadequate (C3 trained_r2=%s, C4 ablation canary=%s, "
            "C6 harm_events=%s). CANNOT DETERMINE whether E2's trained kernels "
            "carry harm avoidance in the real pipeline. A C4 failure specifically "
            "means the untrained-E2 ablation did not destroy world_forward's "
            "predictive power, so the two arms were never really different."
            % (c3, c4, c6)
        )
    elif not c5:
        direction = "inconclusive"
        note = (
            "COMPETENCE FLOOR FAILED: AO_CHAIN -- the real hippocampal pipeline -- "
            "is not meaningfully better than the informational RANDOM_REF arm, so "
            "there is no contribution for the ablation to remove. Explicitly NOT "
            "support for H2/H3; this is the degenerate-comparator failure mode the "
            "portfolio exists to avoid. SELF_CHAIN/RANDOM_REF harm ratio %.3f."
            % self_ratio
        )
    elif not (c1 and c2):
        direction = "weakens"
        note = (
            "Declared null HOLDS with every instrument gate passing: replacing "
            "E2's trained forward-prediction weights with an untrained init "
            "(canary confirms r2 destroyed) leaves harm_rate within the 0.01 bar "
            "of the intact pipeline (mean degradation %.5f). E2's trained kernels "
            "are NOT carrying harm avoidance in the real hippocampal planning "
            "pipeline -- H1 weakened on the instrumentation axis. SELF_CHAIN/"
            "RANDOM_REF harm ratio %.3f (055 measured 0.908). Scope: the "
            "action-object encoder is untrained in BOTH arms (see "
            "scope_statements)." % (mean_delta, self_ratio)
        )
    else:
        direction = "supports"
        note = (
            "Declared null REJECTED: in the REAL pipeline (full REEAgent, "
            "HippocampalModule, matched CEM budget), replacing E2's trained "
            "forward-prediction weights with an untrained init makes harm "
            "materially worse (mean degradation %.5f, consistent on all seeds), "
            "with E2 trained (r2 gate), the ablation confirmed destructive "
            "(canary) and AO_CHAIN itself competent vs RANDOM_REF. This is the "
            "first non-confounded support MECH-033 has had: the comparator "
            "retains the HarmHead and the planner and differs ONLY in E2's "
            "trained content. SELF_CHAIN/RANDOM_REF harm ratio %.3f (055 measured "
            "0.908). Scope: the action-object encoder is untrained in BOTH arms."
            % (mean_delta, self_ratio)
        )
    return criteria, status, direction, note, mean_delta, deltas, self_ratio


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run(dry_run: bool = False) -> Dict:
    print("[V3-EXQ-1102] MECH-033 leg 4 (instrumentation axis): full REEAgent, "
          "AO kernels vs UNTRAINED E2 at matched CEM budget", flush=True)
    print("[V3-EXQ-1102] Declared null: harm_rate(AO_CHAIN) NOT below "
          "harm_rate(UNTRAINED_E2_CHAIN) by >= %.2f"
          % THRESH_C1_MIN_DEGRADATION, flush=True)
    print("[V3-EXQ-1102] SELF_CHAIN and RANDOM_REF are INFORMATIONAL ONLY; "
          "SELF_CHAIN was measured at 90.8%% of random in V3-EXQ-055", flush=True)

    results: Dict[str, List[Dict]] = {a: [] for a in ARMS}
    per_seed: Dict = {}
    selfchecks: Dict = {}
    # Kept so write_flat_manifest can record the z_goal stream stats: a DEAD
    # z_goal stream is otherwise invisible in the manifest (V3-EXQ-626 / 830).
    agents: List = []

    for seed in SEEDS:
        print("\n[V3-EXQ-1102] === seed %d ===" % seed, flush=True)
        agent, env, wf_buf, e3_ticks = _train_agent(seed, dry_run)
        agents.append(agent)

        selfchecks[str(seed)] = _selfcheck_ablation(agent, seed)

        trained_state = _wf_state(agent)
        r2_trained = _wf_r2(agent, wf_buf)
        discrim_trained = _e2_action_discrimination(agent, wf_buf)

        # AO_CHAIN, SELF_CHAIN, RANDOM_REF run on the TRAINED weights.
        results[ARM_AO].append(_run_arm(agent, env, seed, ARM_AO, dry_run))
        results[ARM_SELF].append(_run_arm(agent, env, seed, ARM_SELF, dry_run))
        results[ARM_RANDOM_REF].append(
            _run_arm(agent, env, seed, ARM_RANDOM_REF, dry_run)
        )

        # UNTRAINED_E2_CHAIN: swap in the untrained weights, run, restore.
        _load_wf_state(agent, _fresh_wf_state(agent, seed))
        r2_untrained = _wf_r2(agent, wf_buf)
        discrim_untrained = _e2_action_discrimination(agent, wf_buf)
        results[ARM_UNTRAINED].append(
            _run_arm(agent, env, seed, ARM_UNTRAINED, dry_run)
        )
        _load_wf_state(agent, trained_state)
        r2_restored = _wf_r2(agent, wf_buf)

        per_seed[seed] = {
            "world_forward_r2_trained": r2_trained,
            "world_forward_r2_untrained": r2_untrained,
            "world_forward_r2_trained_after_restore": r2_restored,
            "e2_action_discrimination_trained": discrim_trained,
            "e2_action_discrimination_untrained": discrim_untrained,
            "e3_tick_total": float(e3_ticks),
            "wf_buf_size": float(len(wf_buf)),
        }
        print(
            "[instr] seed=%d wf_r2 trained=%.4f untrained=%.4f restored=%.4f | "
            "discrim trained=%.4f untrained=%.4f"
            % (seed, r2_trained, r2_untrained, r2_restored, discrim_trained,
               discrim_untrained),
            flush=True,
        )
        assert abs(r2_restored - r2_trained) < 1e-6, (
            "world_forward r2 did not return to its trained value after restore "
            "(%.6f -> %.6f): a later arm ran on contaminated weights"
            % (r2_trained, r2_restored)
        )

    criteria, status, direction, note, mean_delta, deltas, self_ratio = (
        _evaluate_criteria(results, per_seed)
    )

    metrics: Dict = {
        "mean_ablation_degradation_untrained_minus_ao": mean_delta,
        "per_seed_ablation_degradation": deltas,
        "self_chain_over_random_ref_harm_ratio": self_ratio,
        "criteria_met": float(sum(1 for v in criteria.values() if v)),
        "criteria_total": float(len(criteria)),
    }
    for arm in ARMS:
        for i, s in enumerate(SEEDS):
            for key in ("harm_rate", "contact_rate", "cal_gap_approach"):
                metrics["%s_%s_seed%d" % (key, arm, s)] = results[arm][i][key]
            metrics["n_harm_events_%s_seed%d" % (arm, s)] = float(
                results[arm][i]["n_harm_events"]
            )
        metrics["harm_rate_%s_mean" % arm] = sum(
            r["harm_rate"] for r in results[arm]
        ) / len(SEEDS)
    for s in SEEDS:
        for k, v in per_seed[s].items():
            metrics["%s_seed%d" % (k, s)] = v
    for arm in (ARM_AO, ARM_UNTRAINED, ARM_SELF):
        for i, s in enumerate(SEEDS):
            denom = results[ARM_RANDOM_REF][i]["harm_rate"]
            metrics["harm_ratio_%s_over_random_ref_seed%d" % (arm, s)] = (
                results[arm][i]["harm_rate"] / denom if denom > 0 else float("nan")
            )
    for k, v in criteria.items():
        metrics["crit_%s" % k] = 1.0 if v else 0.0

    print("\n[V3-EXQ-1102] ===== VERDICT =====", flush=True)
    for k, v in criteria.items():
        print("  %-44s %s" % (k, "PASS" if v else "FAIL"), flush=True)
    print("  status=%s evidence_direction=%s" % (status, direction), flush=True)
    print("  %s" % note, flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = "%s_%s_v3" % (EXPERIMENT_TYPE, ts)

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "outcome": status,
        "status": status,
        "evidence_direction": direction,
        "verdict": note,
        "metrics": metrics,
        "criteria": criteria,
        "declared_null": (
            "harm_rate(AO_CHAIN) is NOT below harm_rate(UNTRAINED_E2_CHAIN) by "
            ">= %.2f" % THRESH_C1_MIN_DEGRADATION
        ),
        "predecessor_run_id": PREDECESSOR_RUN,
        "portfolio": {
            "name": "MECH-033 GOV-FANOUT-1 discrimination portfolio",
            "leg": "4 of 4 as sketched; 3rd of the 3 ratified legs",
            "axis": "instrumentation",
            "hypotheses_discriminated": ["H1-chaining", "H2-one-step", "H3-no-E2"],
            "siblings": {"V3-EXQ-1103": "algorithm axis (H1 vs H2)",
                         "V3-EXQ-1101": "integration axis (H3, inline stack)"},
            "comparator_option": (
                "A-i (ratified 2026-09-25): untrained/re-initialised E2 supplying "
                "kernels. NOT SELF_CHAIN, which was measured at random level."
            ),
            "source_autopsy": SOURCE_AUTOPSY,
            "design_doc": DESIGN_DOC,
            "does_not_supersede": (
                "V3-EXQ-308 is deliberately NOT re-lettered: a re-letter would "
                "inherit the shared uniform-random comparator confound."
            ),
        },
        "arm_results": {a: results[a] for a in ARMS},
        "informational_arms": {
            ARM_SELF: (
                "V3-EXQ-055's CEM-in-z_self arm at matched budget. Measured "
                "DEGENERATE in 055 (harm 90.8% of random, contact_rate 87.1% of "
                "random, cal_gap 0.0213 below 055's own 0.03 floor, single seed). "
                "Re-measured here at 3 seeds for the record; NEVER the null's "
                "comparator."
            ),
            ARM_RANDOM_REF: (
                "Competence floor (C5) + data quality (C6) reference ONLY. Never "
                "the comparator for the declared null -- using a uniform-random "
                "arm that way is the D2 ablation confound this portfolio removes."
            ),
        },
        "per_seed_instruments": {str(k): v for k, v in per_seed.items()},
        "ablation_selfcheck_per_seed": selfchecks,
        "trainer_declaration": {
            "gflag": "GFLAG-0491",
            "waking_gradient_learning_at_defaults": False,
            "optimizers_built_by_this_driver": {
                "main_optimizer": (
                    "all agent params except the wf and terrain sets; "
                    "loss = E3 harm_eval BCE + E1 prediction loss"
                ),
                "wf_optimizer": (
                    "e2.world_transition + e2.world_action_encoder; "
                    "loss = MSE(world_forward(z_t, a_t), z_{t+1}) -- MECH-033's kernel"
                ),
                "terrain_optimizer": (
                    "hippocampal.terrain_prior + hippocampal.action_object_decoder"
                ),
            },
            "act_path_modules_at_random_init": [
                "e2.action_object_head -- receives no gradient in this recipe "
                "(inherited from V3-EXQ-055 for comparability); shared by BOTH "
                "arms, so it bounds the reading without confounding the contrast"
            ],
        },
        "scope_statements": SCOPE_STATEMENTS,
        "ethics_preflight": {
            "involves_negative_valence": True,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
            "note": (
                "SENT-0. Grid-world hazard avoidance with a scalar harm signal at "
                "V3-EXQ-055's own configuration (hazard_harm=0.02); the agent can "
                "always move away from hazards. No new manipulation of valence "
                "relative to the predecessor run."
            ),
        },
    }

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config={
            "seeds": SEEDS,
            "arms": ARMS,
            "warmup_episodes": WARMUP_EPISODES if not dry_run else 3,
            "eval_episodes": EVAL_EPISODES if not dry_run else 2,
            "steps_per_episode": STEPS_PER_EPISODE if not dry_run else 25,
            "lr": LR,
            "self_dim": SELF_DIM,
            "world_dim": WORLD_DIM,
            "cem_budget": {
                "candidate_horizon": CANDIDATE_HORIZON,
                "n_candidates": N_CEM_CANDIDATES,
                "n_iterations": N_CEM_ITERATIONS,
                "elite_fraction": ELITE_FRACTION,
                "matched_across": [ARM_AO, ARM_UNTRAINED, ARM_SELF],
            },
            "thresholds": {
                "C1_min_degradation": THRESH_C1_MIN_DEGRADATION,
                "C3_trained_e2_r2": THRESH_C3_E2_R2,
                "C4_untrained_r2_max": THRESH_C4_ABLATED_R2_MAX,
                "C5_competence_ratio": THRESH_C5_COMPETENCE,
                "C6_min_contacts": THRESH_C6_MIN_CONTACTS,
            },
            "env": {
                "class": "CausalGridWorldV2", "size": 12, "num_hazards": 4,
                "num_resources": 5, "hazard_harm": HARM_SCALE,
                "env_drift_interval": 5, "env_drift_prob": 0.1,
                "proximity_scale": PROXIMITY_SCALE,
                "note": "V3-EXQ-055's configuration, verbatim",
            },
            "ablated_submodules": list(WF_SUBMODULES),
        },
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=_T0,
        agent=agents,
    )
    print("wrote: %s" % out_path, flush=True)
    return {"status": status, "evidence_direction": direction,
            "metrics": metrics, "manifest_path": str(out_path)}


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--dry-run", action="store_true")
    _args = _ap.parse_args()
    _res = run(dry_run=_args.dry_run)
    emit_outcome(
        outcome=_res["status"] if _res["status"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=_res["manifest_path"],
        dry_run=_args.dry_run,
    )
