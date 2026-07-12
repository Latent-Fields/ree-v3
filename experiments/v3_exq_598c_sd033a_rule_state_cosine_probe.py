#!/opt/local/bin/python3
"""V3-EXQ-598c: SD-033a metric-redesign falsifier -- direct rule_state cosine
probes on held-out rule-context pairs (replaces 598b reef-visit-band metric).

Supersedes V3-EXQ-598b. Routing per failure_autopsy_V3-EXQ-598_2026-05-29.md
(Learning #2 + Section 8) and arc062_rule_creator_scoping_2026-06-02.md Section
5.4 ("SD-033a successor (598c) needs a redesigned metric ... direct rule_state
cosine probes on held-out rule-context pairs, not the reef-visit-band integration
metric. Independent of the substrate work.").

WHY A NEW METRIC. The 598/598b C3 criterion (P2 reef_visit_fraction in (0.20,
0.80)) is a DOWNSTREAM-INTEGRATION metric: it requires the policy to behaviourally
inhabit reef-side states. Under the MECH-309 monomodal-policy collapse the agent
never visits the reef half, so C3 pins near 0 regardless of whether SD-033a's
rule_state is functioning -- it cannot test SD-033a in a regime where the claim
can express itself (598 autopsy Section 2). The 598 autopsy confirmed the SD-033a
substrate OPERATES as specified (598b C1 frozen_silent + C2 trainable_nonzero
both PASS) -- only the behavioural C3 fails, and that failure is attributable to
the blocked upstream ARC-065 / GAP-B behavioural-diversity cluster, NOT to SD-033a.

WHAT 598c TESTS INSTEAD. SD-033a functional signatures (i) stimulus-abstracted
rule representation and (ii) distractor-resistant persistence are REPRESENTATIONAL
properties of the rule_state vector. 598c probes them DIRECTLY:

  - Drive the agent through the SD-054 bipartite env (reef-half vs forage-half =
    two distinct rule-contexts) with RANDOM actions. Random exploration visits
    both halves regardless of policy collapse -- THIS is what decouples 598c from
    the blocked behavioural-diversity work. rule_state still advances via the
    real agent.select_action -> lateral_pfc.update path (update() reads only the
    observation stream z_delta / z_world, not the action, so a random step drives
    rule_state exactly as production does).
  - Bucket the read-out rule_state vectors by recent-context (sliding-window
    dominant half) into reef-context vs forage-context samples.
  - Measure cosine discriminability: mean within-context cosine (consistency)
    minus mean cross-context cosine (the discriminability margin). A functioning
    rule_state holds a context-discriminative, persistent representation -> high
    within, low cross -> positive margin. This is the Mansouri 2020 transfer /
    representational-similarity-analysis signature named in the 598 autopsy.

NON-VACUITY READINESS GATE (same-statistic, per the 642/643 lessons). A degenerate
read could come from an UNDERTRAINED ENCODER (z_world not context-discriminative ->
nothing for rule_state to carry) rather than from SD-033a. So 598c measures the
SAME cosine-discriminability statistic on z_world itself (the upstream signal) as a
readiness precondition. If z_world is NOT context-discriminable, the run self-routes
to substrate_not_ready_requeue (re-run with more encoder warm-up) -- it does NOT
emit an SD-033a verdict. Only once z_world clears the readiness floor does a
below-floor rule_state margin become a genuine SD-033a finding (the persistence EMA
smears context -- signature (i) not preserved through (ii)).

This is a clean SD-033a-specific probe: gated_policy and the ARC-062 discriminator
source are DISABLED (use_gated_policy=False, lateral_pfc_use_discriminator_source=
False) so the test isolates the pure SD-033a rule_state mechanism (delta_proj +
world_proj gate-modulated EMA) and is maximally decoupled from the blocked ARC-062
cluster.

Pre-registered acceptance (single claim SD-033a; lower-is-better cosine convention
does NOT apply -- cosine similarity is higher-is-more-similar):
  READINESS (precondition, NOT an SD-033a criterion):
    z_world cosine discriminability margin >= ZWORLD_MARGIN_FLOOR on >= MIN_PASS_SEEDS
    seeds AND every scored seed reached MIN_SAMPLES_PER_CONTEXT in both contexts.
    If unmet -> outcome FAIL, evidence_direction non_contributory, label
    substrate_not_ready_requeue. The SD-033a criteria below are NOT adjudicated.
  C1_rule_state_non_degenerate : mean ||rule_state|| > NORM_FLOOR AND cross-sample
    rule_state std > VAR_FLOOR on >= MIN_PASS_SEEDS seeds (rule_state is populated
    and varies -- not a frozen constant).
  C2_within_context_consistency : mean within-context rule_state cosine >=
    WITHIN_FLOOR on >= MIN_PASS_SEEDS seeds (the same context produces a
    consistent rule_state -- signature (ii) persistence at the representational
    level).
  C3_cross_context_discriminable (LOAD-BEARING) : rule_state cosine discriminability
    margin (within - cross) >= RULE_MARGIN_FLOOR on >= MIN_PASS_SEEDS seeds (the
    rule_state distinguishes reef-context from forage-context -- signature (i)
    stimulus-abstracted rule representation, preserved through the persistence EMA).
  PASS = READINESS met AND C1 AND C2 AND C3.

Evidence mapping (SD-033a, single claim):
  supports             : READINESS met AND C1 AND C2 AND C3 (signatures i+ii
                         delivered at the representational level).
  weakens              : READINESS met AND C1 but NOT C3 (rule_state populated and
                         z_world discriminable, yet rule_state is NOT context-
                         discriminable -> the persistence EMA smears context;
                         signature (i) not preserved through (ii)). A clean, genuine
                         negative decoupled from policy behaviour.
  non_contributory     : READINESS unmet (substrate_not_ready_requeue) OR C1 fail
                         with readiness met (rule_state_inert -- substrate not
                         holding state; re-check wiring/config, not an SD-033a
                         verdict).

claim_ids: SD-033a only. MECH-262 is NOT tagged: 598c probes the rule_state
representation directly, not the cross-tick rule-selective persistence under forced
internal_replay that MECH-262's behavioural signature requires.

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
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_598c_sd033a_rule_state_cosine_probe"
EXPERIMENT_PURPOSE = "evidence"
QUEUE_ID = "V3-EXQ-598c"
SUPERSEDES = "V3-EXQ-598b"
CLAIM_IDS = ["SD-033a"]

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
STEPS_PER_EPISODE = 200

LR_E1 = 1e-3
LR_E2_WF = 1e-3
LR_E3_HARM = 1e-3
BATCH_SIZE = 32
WF_BUF_MAX = 256
EMA_DECAY = 0.9

# Pooled-resample probe. Phase A collects a pool of real observations bucketed by
# the bipartite half the agent occupied (instantaneous label -> high coverage).
# Phase B runs N_TRIALS_PER_CONTEXT probe trials per context: reset the agent, then
# feed DWELL consecutive observations sampled from ONE context's pool so the
# rule_state EMA settles on a context-pure stimulus before it is read. This is the
# standard "present sustained category-A vs category-B stimuli, read the
# representation" probe -- it decouples rule_state settling from navigation (which a
# collapsed policy / random walk cannot reliably deliver) while driving rule_state
# via the real production sense -> select_action -> lateral_pfc.update path.
N_TRIALS_PER_CONTEXT = 40
DWELL = 12
COLLECT_RESETS = 12          # env resets (distinct layouts) for teleport sampling.
PER_HALF_PER_RESET = 8       # obs read per half per layout (balanced coverage).
# Half classification dead-band (in cells from the auto-detected split midline);
# positions inside the band are "band" (skipped), matching the agent spawn band.
HALF_BAND = 1
MIN_OBS_PER_HALF = 15         # pool must hold enough obs to resample DWELL-runs.
MIN_SAMPLES_PER_CONTEXT = 20  # probe trials scored per context.

# Pre-registered thresholds. Discriminability is measured as MEAN-CENTERED cosine
# (RSA convention): each vector has the grand mean across all probe samples (both
# contexts) subtracted before cosine, so the metric reflects the context-
# discriminative DEVIATION, not the large common-mode response that dominates raw
# rule_state vectors. Under centering, a genuinely context-separating representation
# gives within-context cosine ~ +1 and cross-context cosine ~ -1 (margin ~ up to 2);
# a non-discriminative / noisy one gives within ~ 0, cross ~ 0, margin ~ 0.
NORM_FLOOR = 1e-3            # C1: rule_state must be populated (RAW norm).
# C1 catches only an EXACTLY-frozen rule_state (std ~ 0 -> centered cosine would be
# pure float noise). Structured-but-low-magnitude variation is legitimate: signature
# (i) is about whether rule_state CARRIES context-discriminative information (the
# DIRECTION of deviation), not its magnitude. The within-context consistency
# criterion (C2) is what rules out "variation is just noise" -- it requires the
# centered deviations to be self-consistent within a context.
VAR_FLOOR = 1e-6            # C1: rule_state must not be bit-frozen (RAW std).
ZWORLD_MARGIN_FLOOR = 0.15  # readiness: z_world must be context-discriminable.
WITHIN_FLOOR = 0.20         # C2: within-context centered-cosine consistency.
RULE_MARGIN_FLOOR = 0.15    # C3 (load-bearing): rule_state discriminability margin.
MIN_PASS_SEEDS = 2


def _action_to_onehot(idx: int, n: int, device=None) -> torch.Tensor:
    v = torch.zeros(1, n, device=device if device is not None else "cpu")
    v[0, idx] = 1.0
    return v


def _make_agent(seed: int) -> Tuple[REEAgent, CausalGridWorldV2]:
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
        # SD-033a under test (pure rule_state mechanism). gated_policy + the
        # ARC-062 discriminator source are OFF so the probe isolates SD-033a and
        # is decoupled from the blocked ARC-062 cluster.
        use_gated_policy=False,
        use_dacc=False,
        dacc_weight=0.0,
        use_lateral_pfc_analog=True,
        lateral_pfc_use_discriminator_source=False,
        lateral_pfc_train_rule_bias_head=False,
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    agent = REEAgent(config)
    return agent, env


def _preflight() -> None:
    agent, _ = _make_agent(0)
    assert agent.lateral_pfc is not None, "lateral_pfc not constructed"
    assert agent.lateral_pfc.config.use_lateral_pfc_analog is True
    assert agent.lateral_pfc.config.use_discriminator_source is False
    assert getattr(agent, "gated_policy", None) is None, "gated_policy should be OFF"
    rs = agent.lateral_pfc.rule_state
    assert rs.shape[-1] == agent.lateral_pfc.config.rule_dim
    del agent
    print(
        "Preflight PASS: lateral_pfc ON, discriminator_source OFF, gated_policy OFF, "
        f"rule_dim={rs.shape[-1]}",
        flush=True,
    )


def _encoder_step(agent: REEAgent, env, steps: int, total_eps: int, ep: int) -> float:
    """P0 encoder warm-up (mirrors V3-EXQ-598b _encoder_step). Trains E1 / E2
    world-forward / harm-eval so z_world becomes context-discriminative BEFORE the
    probe -- without this the readiness precondition (z_world discriminability)
    would fail and the run would correctly self-route to substrate_not_ready."""
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
    for _step_i in range(steps):
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
            action = _action_to_onehot(random.randint(0, env.action_dim - 1), env.action_dim, device)
            agent._last_action = action
        _, harm, done, _, obs_dict = env.step(action)
        ep_reward += float(harm)
        if z_wp is not None and act_p is not None:
            wf_buf.append((z_wp.cpu(), act_p.cpu(), z_w.cpu()))
            if len(wf_buf) > WF_BUF_MAX:
                wf_buf = wf_buf[-WF_BUF_MAX:]
        he_buf.append((z_w.cpu(), torch.tensor([abs(float(harm)) if float(harm) < 0 else 0.0])))
        if len(wf_buf) >= BATCH_SIZE:
            idx = torch.randperm(len(wf_buf))[:BATCH_SIZE].tolist()
            zw = torch.cat([wf_buf[i][0] for i in idx]).to(device)
            a = torch.cat([wf_buf[i][1] for i in idx]).to(device)
            zw1 = torch.cat([wf_buf[i][2] for i in idx]).to(device)
            pred = agent.e2.world_forward(zw, a)
            loss = F.mse_loss(pred, zw1)
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
    if (ep + 1) % 10 == 0:
        print(
            f"  [train] probe seed phase=enc ep {ep+1}/{total_eps}  reward={ep_reward:.3f}",
            flush=True,
        )
    return ep_reward


def _build_half_classifier(env: CausalGridWorldV2):
    """Auto-detect the bipartite split axis + reef side from the actual reef-cell
    placement, so the classifier is robust to the (x, y) row/col convention.
    Returns a function pos(x, y) -> 'reef' | 'forage' | 'band'."""
    reef_cells = list(getattr(env, "_reef_cells", set()))
    size = int(getattr(env, "size", 12))
    center = (size - 1) / 2.0
    if not reef_cells:
        # No reef substrate -> degenerate; everything is 'band' (probe will then
        # fail coverage and self-route to substrate_not_ready).
        return lambda x, y: "band"
    cx = float(np.mean([c[0] for c in reef_cells]))
    cy = float(np.mean([c[1] for c in reef_cells]))
    if abs(cy - center) >= abs(cx - center):
        axis = 1  # split by y
        reef_disp = cy - center
    else:
        axis = 0  # split by x
        reef_disp = cx - center
    reef_sign = 1.0 if reef_disp >= 0 else -1.0

    def classify(x: int, y: int) -> str:
        coord = (x, y)[axis]
        disp = coord - center
        if abs(disp) <= HALF_BAND:
            return "band"
        s = 1.0 if disp >= 0 else -1.0
        return "reef" if s == reef_sign else "forage"

    return classify


def _snapshot_obs(obs_dict: Dict) -> Dict:
    """Detached CPU copy of the observation fields the agent's sense() consumes."""
    def _cp(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().clone()
        return v
    return {
        "body_state": _cp(obs_dict["body_state"]),
        "world_state": _cp(obs_dict["world_state"]),
        "harm_obs": _cp(obs_dict.get("harm_obs")),
        "harm_obs_a": _cp(obs_dict.get("harm_obs_a")),
        "harm_history": _cp(obs_dict.get("harm_history")),
        "benefit_exposure": float(obs_dict.get("benefit_exposure", 0.0)),
    }


def _collect_obs_pool(
    env: CausalGridWorldV2, n_resets: int, per_half_per_reset: int
) -> Tuple[List[Dict], List[Dict]]:
    """Phase A. Teleport-sample a balanced observation pool. For each env reset (a
    fixed hazard/resource/reef layout) read the observation at randomly-chosen
    reef-half and forage-half cells via env._get_observation_dict() (the same call
    env.reset / env.step use). This GUARANTEES balanced coverage of both bipartite
    halves regardless of policy collapse or survival in the dangerous
    food-attracted-hazard forage half -- the decoupling from the blocked
    behavioural-diversity work. Within-layout position contrast (same world, reef
    vs forage cell) is the clean rule-context contrast; across resets supplies
    held-out layout variety."""
    reef_obs: List[Dict] = []
    forage_obs: List[Dict] = []
    size = int(getattr(env, "size", 12))
    for _ in range(n_resets):
        env.reset()
        classify = _build_half_classifier(env)
        reef_cells = [(x, y) for x in range(size) for y in range(size) if classify(x, y) == "reef"]
        forage_cells = [(x, y) for x in range(size) for y in range(size) if classify(x, y) == "forage"]
        random.shuffle(reef_cells)
        random.shuffle(forage_cells)
        for (cx, cy) in reef_cells[:per_half_per_reset]:
            env.agent_x, env.agent_y = cx, cy
            reef_obs.append(_snapshot_obs(env._get_observation_dict()))
        for (cx, cy) in forage_cells[:per_half_per_reset]:
            env.agent_x, env.agent_y = cx, cy
            forage_obs.append(_snapshot_obs(env._get_observation_dict()))
    return reef_obs, forage_obs


def _probe_context(
    agent: REEAgent, pool: List[Dict], n_trials: int, dwell: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Phase B. For each trial: reset the agent, feed DWELL observations sampled
    from this context's pool through the real sense -> select_action ->
    lateral_pfc.update path so the rule_state EMA settles on a context-pure
    stimulus, then read rule_state + the settled z_world. Returns (rule_vecs,
    zworld_vecs)."""
    device = agent.device
    rule_vecs: List[np.ndarray] = []
    zw_vecs: List[np.ndarray] = []
    agent.eval()
    with torch.no_grad():
        for _t in range(n_trials):
            agent.reset()  # zero rule_state + clear buffers between trials
            latent = None
            idxs = [random.randrange(len(pool)) for _ in range(dwell)]
            for j in idxs:
                ob = pool[j]
                latent = agent.sense(
                    ob["body_state"],
                    ob["world_state"],
                    obs_harm=ob["harm_obs"],
                    obs_harm_a=ob["harm_obs_a"],
                    obs_harm_history=ob["harm_history"],
                )
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks.get("e1_tick", False)
                    else torch.zeros(1, WORLD_DIM, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                agent.update_z_goal(
                    benefit_exposure=max(0.0, ob["benefit_exposure"]),
                    drive_level=REEAgent.compute_drive_level(ob["body_state"]),
                )
                # Advances rule_state (update reads z_world / z_delta, not the
                # action). The returned action is discarded -- no env stepping.
                _ = agent.select_action(candidates, ticks, temperature=1.0)
            rs = agent.lateral_pfc.rule_state.detach().flatten().cpu().numpy().astype(np.float64)
            zw = latent.z_world.detach().flatten().cpu().numpy().astype(np.float64)
            rule_vecs.append(rs)
            zw_vecs.append(zw)
    return rule_vecs, zw_vecs


def _mean_within_cosine(vecs: List[np.ndarray]) -> float:
    """Mean off-diagonal cosine over a set of vectors (closed form on unit vecs)."""
    n = len(vecs)
    if n < 2:
        return 0.0
    u = np.stack(vecs, axis=0)
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    u = u / norms
    s = u.sum(axis=0)
    total = float(s @ s) - float(n)  # sum_{i!=j} u_i . u_j
    return total / (n * (n - 1))


def _mean_cross_cosine(a: List[np.ndarray], b: List[np.ndarray]) -> float:
    if len(a) < 1 or len(b) < 1:
        return 0.0
    ua = np.stack(a, axis=0)
    ub = np.stack(b, axis=0)
    na = np.linalg.norm(ua, axis=1, keepdims=True)
    nb = np.linalg.norm(ub, axis=1, keepdims=True)
    na = np.where(na < 1e-12, 1.0, na)
    nb = np.where(nb < 1e-12, 1.0, nb)
    ua = ua / na
    ub = ub / nb
    sa = ua.sum(axis=0)
    sb = ub.sum(axis=0)
    return float(sa @ sb) / (len(a) * len(b))


def _discriminability(a: List[np.ndarray], b: List[np.ndarray]) -> Dict[str, float]:
    """Mean-centered cosine discriminability (RSA). Subtract the grand mean across
    BOTH contexts' samples before computing cosine, so the metric reflects the
    context-discriminative deviation rather than the large shared common-mode
    component that dominates raw rule_state / z_world vectors."""
    if len(a) < 2 or len(b) < 2:
        return {"within_a": 0.0, "within_b": 0.0, "within": 0.0, "cross": 0.0, "margin": 0.0}
    grand_mean = np.stack(a + b, axis=0).mean(axis=0)
    ac = [v - grand_mean for v in a]
    bc = [v - grand_mean for v in b]
    within_a = _mean_within_cosine(ac)
    within_b = _mean_within_cosine(bc)
    cross = _mean_cross_cosine(ac, bc)
    within = 0.5 * (within_a + within_b)
    return {
        "within_a": within_a,
        "within_b": within_b,
        "within": within,
        "cross": cross,
        "margin": within - cross,
    }


def _rule_state_degeneracy(reef: List[np.ndarray], forage: List[np.ndarray]) -> Dict[str, float]:
    allv = reef + forage
    if len(allv) < 2:
        return {"mean_norm": 0.0, "sample_std": 0.0}
    u = np.stack(allv, axis=0)
    mean_norm = float(np.mean(np.linalg.norm(u, axis=1)))
    # Cross-sample variability: mean per-dim std across samples.
    sample_std = float(np.mean(np.std(u, axis=0)))
    return {"mean_norm": mean_norm, "sample_std": sample_std}


def run_seed(seed: int, dry_run: bool) -> Dict:
    p0 = 3 if dry_run else P0_EPISODES
    collect_resets = 2 if dry_run else COLLECT_RESETS
    per_half = 3 if dry_run else PER_HALF_PER_RESET
    steps = 60 if dry_run else STEPS_PER_EPISODE
    n_trials = 4 if dry_run else N_TRIALS_PER_CONTEXT
    dwell = 5 if dry_run else DWELL
    total = p0 + collect_resets
    print(f"\nSeed {seed} Condition probe", flush=True)
    agent, env = _make_agent(seed)
    for ep in range(p0):
        _encoder_step(agent, env, steps, total, ep)

    # Phase A: teleport-sample a balanced observation pool bucketed by bipartite half.
    reef_obs, forage_obs = _collect_obs_pool(env, collect_resets, per_half)
    for ci in range(collect_resets):
        print(
            f"  [train] probe seed phase=collect ep {p0+ci+1}/{total}  "
            f"reef_obs={len(reef_obs)} forage_obs={len(forage_obs)}",
            flush=True,
        )

    min_obs = 2 if dry_run else MIN_OBS_PER_HALF
    pool_ok = len(reef_obs) >= min_obs and len(forage_obs) >= min_obs

    # Phase B: probe rule_state under sustained reef-context vs forage-context.
    if pool_ok:
        reef_rule, reef_zw = _probe_context(agent, reef_obs, n_trials, dwell)
        forage_rule, forage_zw = _probe_context(agent, forage_obs, n_trials, dwell)
    else:
        reef_rule, reef_zw, forage_rule, forage_zw = [], [], [], []

    n_reef = len(reef_rule)
    n_forage = len(forage_rule)
    min_samples = 2 if dry_run else MIN_SAMPLES_PER_CONTEXT
    coverage_ok = pool_ok and n_reef >= min_samples and n_forage >= min_samples

    rule_disc = _discriminability(reef_rule, forage_rule)
    zw_disc = _discriminability(reef_zw, forage_zw)
    degen = _rule_state_degeneracy(reef_rule, forage_rule)

    zworld_ready = coverage_ok and (zw_disc["margin"] >= ZWORLD_MARGIN_FLOOR)
    c1_non_degenerate = coverage_ok and (degen["mean_norm"] > NORM_FLOOR) and (degen["sample_std"] > VAR_FLOOR)
    c2_within = coverage_ok and (rule_disc["within_a"] >= WITHIN_FLOOR) and (rule_disc["within_b"] >= WITHIN_FLOOR)
    c3_discriminable = coverage_ok and (rule_disc["margin"] >= RULE_MARGIN_FLOOR)

    seed_pass = bool(zworld_ready and c1_non_degenerate and c2_within and c3_discriminable)
    print(
        f"verdict: {'PASS' if seed_pass else 'FAIL'}  "
        f"n_reef={n_reef} n_forage={n_forage}  "
        f"zw_margin={zw_disc['margin']:.4f} rule_margin={rule_disc['margin']:.4f}  "
        f"rule_within={rule_disc['within']:.4f} rule_cross={rule_disc['cross']:.4f}  "
        f"rule_norm={degen['mean_norm']:.4f} rule_std={degen['sample_std']:.5f}",
        flush=True,
    )
    return {
        "seed": seed,
        "n_reef_obs": len(reef_obs),
        "n_forage_obs": len(forage_obs),
        "n_reef_samples": n_reef,
        "n_forage_samples": n_forage,
        "coverage_ok": coverage_ok,
        "zworld_margin": zw_disc["margin"],
        "zworld_within": zw_disc["within"],
        "zworld_cross": zw_disc["cross"],
        "rule_margin": rule_disc["margin"],
        "rule_within": rule_disc["within"],
        "rule_within_reef": rule_disc["within_a"],
        "rule_within_forage": rule_disc["within_b"],
        "rule_cross": rule_disc["cross"],
        "rule_state_mean_norm": degen["mean_norm"],
        "rule_state_sample_std": degen["sample_std"],
        "zworld_ready": zworld_ready,
        "c1_non_degenerate": c1_non_degenerate,
        "c2_within": c2_within,
        "c3_discriminable": c3_discriminable,
        "seed_pass": seed_pass,
    }


def run(seeds: Optional[List[int]] = None, dry_run: bool = False) -> Dict:
    if seeds is None:
        seeds = [0, 1, 2]
    rows = [run_seed(s, dry_run) for s in seeds]
    n_seeds = len(rows)

    n_ready = sum(1 for r in rows if r["zworld_ready"])
    n_c1 = sum(1 for r in rows if r["c1_non_degenerate"])
    n_c2 = sum(1 for r in rows if r["c2_within"])
    n_c3 = sum(1 for r in rows if r["c3_discriminable"])

    min_pass = min(MIN_PASS_SEEDS, n_seeds)
    readiness_met = n_ready >= min_pass
    c1 = n_c1 >= min_pass
    c2 = n_c2 >= min_pass
    c3 = n_c3 >= min_pass

    return {
        "rows": rows,
        "acceptance": {
            "READINESS_zworld_discriminable": readiness_met,
            "C1_rule_state_non_degenerate": c1,
            "C2_within_context_consistency": c2,
            "C3_cross_context_discriminable": c3,
            "n_ready_seeds": n_ready,
            "n_c1_seeds": n_c1,
            "n_c2_seeds": n_c2,
            "n_c3_seeds": n_c3,
            "n_seeds": n_seeds,
            "min_pass_seeds": min_pass,
            "pass": bool(readiness_met and c1 and c2 and c3),
        },
    }


def _interpret(acc: Dict) -> Tuple[str, str, str, str]:
    """Returns (outcome, evidence_direction, label, note)."""
    readiness = bool(acc["READINESS_zworld_discriminable"])
    c1 = bool(acc["C1_rule_state_non_degenerate"])
    c2 = bool(acc["C2_within_context_consistency"])
    c3 = bool(acc["C3_cross_context_discriminable"])
    if not readiness:
        return (
            "FAIL",
            "non_contributory",
            "substrate_not_ready_requeue",
            "z_world is not context-discriminable (encoder under-trained or coverage "
            "insufficient); re-run with more P0 encoder warm-up / longer probe. NOT an "
            "SD-033a verdict.",
        )
    if not c1:
        return (
            "FAIL",
            "non_contributory",
            "rule_state_inert",
            "z_world discriminable but rule_state is degenerate (near-zero norm or "
            "constant); SD-033a is not holding state -- re-check wiring/config, not an "
            "SD-033a representational verdict.",
        )
    if c1 and c2 and c3:
        return (
            "PASS",
            "supports",
            "rule_state_context_discriminable",
            "rule_state is populated, within-context consistent, and cross-context "
            "discriminable -- SD-033a signatures (i) stimulus-abstracted representation "
            "and (ii) persistence delivered at the representational level, decoupled "
            "from policy behaviour.",
        )
    # readiness + C1 met, but C2 and/or the load-bearing C3 failed.
    return (
        "FAIL",
        "weakens",
        "rule_state_not_context_discriminable",
        "rule_state populated and z_world discriminable, yet rule_state does NOT "
        "distinguish reef-context from forage-context (load-bearing C3 margin below "
        "floor): the persistence EMA smears context -- signature (i) not preserved "
        "through (ii). A clean negative decoupled from policy behaviour.",
    )


def write_manifest(result: Dict, dry_run: bool, elapsed: float) -> Tuple[Path, str]:
    acc = result["acceptance"]
    outcome, direction, label, note = _interpret(acc)
    rows = result["rows"]

    def _avg(key: str) -> float:
        vals = [r[key] for r in rows if r.get("coverage_ok")]
        return float(np.mean(vals)) if vals else 0.0

    zw_margin_mean = _avg("zworld_margin")
    rule_margin_mean = _avg("rule_margin")
    rule_norm_mean = _avg("rule_state_mean_norm")
    coverage_all = all(r.get("coverage_ok") for r in rows)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    interpretation = {
        "label": label,
        "note": note,
        # Same-statistic readiness precondition (642/643 lesson): the load-bearing
        # criterion (rule_state cosine margin) is only meaningful once the SAME
        # statistic on the upstream signal (z_world cosine margin) clears the floor.
        "preconditions": [
            {
                "name": "zworld_context_discriminable",
                "description": "cosine discriminability margin (within - cross) of z_world "
                "across reef/forage contexts, measured on the SAME statistic the "
                "load-bearing rule_state criterion routes on; the positive control is "
                "the trained encoder output that rule_state is built from.",
                "control": "P0-trained z_world encoder reading real reef/forage observations",
                "measured": round(zw_margin_mean, 6),
                "threshold": ZWORLD_MARGIN_FLOOR,
                "met": bool(acc["READINESS_zworld_discriminable"]),
            },
            {
                "name": "context_coverage_both_halves",
                "description": "every scored seed reached MIN_SAMPLES_PER_CONTEXT in BOTH "
                "reef and forage contexts (else the cosine statistics are starved).",
                "measured": int(sum(1 for r in rows if r.get("coverage_ok"))),
                "threshold": acc["min_pass_seeds"],
                "met": bool(coverage_all and acc["READINESS_zworld_discriminable"]),
            },
        ],
        "criteria_non_degenerate": {
            "C1_rule_state_non_degenerate": bool(acc["C1_rule_state_non_degenerate"]),
            "C2_within_context_consistency": bool(acc["C2_within_context_consistency"]),
            "C3_cross_context_discriminable": bool(acc["C3_cross_context_discriminable"]),
        },
    }

    criteria = [
        {"name": "READINESS_zworld_discriminable", "load_bearing": False,
         "passed": bool(acc["READINESS_zworld_discriminable"])},
        {"name": "C1_rule_state_non_degenerate", "load_bearing": False,
         "passed": bool(acc["C1_rule_state_non_degenerate"])},
        {"name": "C2_within_context_consistency", "load_bearing": False,
         "passed": bool(acc["C2_within_context_consistency"])},
        {"name": "C3_cross_context_discriminable", "load_bearing": True,
         "passed": bool(acc["C3_cross_context_discriminable"])},
    ]

    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "executing_hostname": socket.gethostname(),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": direction,
        "outcome": outcome,
        "interpretation": interpretation,
        "interpretation_label": label,
        "criteria": criteria,
        "dry_run": dry_run,
        "elapsed_sec": elapsed,
        "acceptance": acc,
        "summary": {
            "zworld_margin_mean": zw_margin_mean,
            "rule_margin_mean": rule_margin_mean,
            "rule_state_mean_norm": rule_norm_mean,
            "coverage_all_seeds": coverage_all,
            "thresholds": {
                "ZWORLD_MARGIN_FLOOR": ZWORLD_MARGIN_FLOOR,
                "RULE_MARGIN_FLOOR": RULE_MARGIN_FLOOR,
                "WITHIN_FLOOR": WITHIN_FLOOR,
                "NORM_FLOOR": NORM_FLOOR,
                "VAR_FLOOR": VAR_FLOOR,
                "DWELL": DWELL,
                "N_TRIALS_PER_CONTEXT": N_TRIALS_PER_CONTEXT,
                "MIN_OBS_PER_HALF": MIN_OBS_PER_HALF,
                "MIN_SAMPLES_PER_CONTEXT": MIN_SAMPLES_PER_CONTEXT,
                "discriminability_metric": "mean_centered_cosine_rsa",
            },
        },
        "seed_rows": rows,
        "autopsy_routing": "failure_autopsy_V3-EXQ-598_2026-05-29.md Learning #2 + Section 8; "
        "arc062_rule_creator_scoping_2026-06-02.md Section 5.4",
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
    result = run(seeds=args.seeds, dry_run=args.dry_run)
    elapsed = time.time() - t0
    out_path, outcome = write_manifest(result, args.dry_run, elapsed)
    acc = result["acceptance"]
    _, direction, label, _ = _interpret(acc)
    print("\n=== V3-EXQ-598c SUMMARY ===", flush=True)
    print(f"  READINESS_zworld_discriminable={acc['READINESS_zworld_discriminable']}", flush=True)
    print(f"  C1_rule_state_non_degenerate={acc['C1_rule_state_non_degenerate']}", flush=True)
    print(f"  C2_within_context_consistency={acc['C2_within_context_consistency']}", flush=True)
    print(f"  C3_cross_context_discriminable={acc['C3_cross_context_discriminable']} (LOAD-BEARING)", flush=True)
    print(f"  outcome={outcome}  evidence_direction={direction}  label={label}", flush=True)
    print(f"Result written to: {out_path}", flush=True)
    return out_path, outcome


if __name__ == "__main__":
    out_path, outcome = main()
    _raw = str(outcome).upper()
    emit_outcome(
        outcome=_raw if _raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
    )
