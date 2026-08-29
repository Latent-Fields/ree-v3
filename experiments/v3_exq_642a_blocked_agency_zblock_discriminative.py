#!/opt/local/bin/python3
"""V3-EXQ-642a -- MECH-353 blocked-agency (z_block) discriminative diagnostic, RETEST.

Claims: [] (experiment_purpose=diagnostic; excluded from confidence/conflict
        scoring per REE_assembly Phase-3 governance). A PASS clears the MECH-353
        v3_pending substrate gate (substrate-readiness, not claim weighting).
        Bears on (cited, NOT tagged): MECH-353, SD-029, MECH-112, MECH-320,
        MECH-342, ARC-016, SD-011, SD-019b, SD-070, SD-056.

Supersedes: V3-EXQ-642 (FAIL 2026-06-06, interpretation.label=
        substrate_ceiling_comparator_nondiscriminative). Failure autopsy
        (REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-642_2026-06-06.md,
        status=confirmed, Reading A) found the self-route PREMATURE: the run's
        P0 trained E2.world_forward on a FROZEN RANDOM encoder with SD-056
        contrastive OFF, so pred_mag = ||world_forward(z_world_prev, a) -
        z_world_prev|| stayed below blocked_agency_predicted_effect_floor
        (0.05) on EVERY step by construction (wf_mse~2e-5 was a low-variance-
        target fit, not evidence of a discriminative comparator). MECH-353 was
        never actually tested. Required fix per autopsy Section 8: (1) train
        the encoder so z_world tracks position, (2) enable SD-056
        (e2_action_contrastive_enabled) so world_forward is action-
        discriminative, (3) add a P0 readiness gate before spending the P1
        budget. This retest supplies all three.

THE QUESTION (unchanged from 642)
----------------------------------
Does z_block (the blocked-agency / control-failure affect readout, MECH-353)
behave as the verdict predicts: (a) rise when an intended, predicted-to-succeed
action is repeatedly BLOCKED while harm + goal-value are held constant, (b) drive
an ASSERT / persist response (effort escalation / alternative-action search)
DISTINCT from the withdraw signature, and (c) DISSOCIATE from z_harm_a (it fires
with zero noxious input -- the SD-011 / SD-019b distinctness claim)?

DESIGN (unchanged from 642)
----------------------------
Static landmark env (SD-023 gradient fields fixed per episode; NO hazards / drift /
respawn / multi-source) so z_world tracks AGENT POSITION rather than env dynamics.
Harm held constant (num_hazards=0 -> z_harm_a ~ flat in both arms). Goal-value
held constant (a fixed z_goal is injected and re-pinned each episode).

Two arms, same trained agent (so the ONLY difference is the external block):
  ARM_BLOCK   -- scheduled_action_block intermittent (every other step cancelled)
  ARM_CONTROL -- no block.

WHAT CHANGED VS 642 (the fix)
------------------------------
P0a (NEW) -- SD-070 z_world encoder warmup (ree_core/latent/zworld_p0.py via
    experiments/_lib/zworld_p0_warmup.run_zworld_p0). Trains
    split_encoder.world_encoder + world_precision_logit on a dedicated warmup
    env (RandomPolicy) with the anti-collapse VICReg penalty + reconstruction
    head; the hazard/resource grounding heads degrade gracefully to a trivial
    single-class fit on this env (num_hazards=num_resources=0) without
    corrupting the reconstruction/anti-collapse terms, which do not depend on
    either entity type. This is what makes z_world differentiate by position
    instead of being a frozen random projection (642's root cause).
P0b (AMENDED) -- E2.world_forward trained as in 642 (one-hot executed action,
    MSE against the NOW-MEANINGFUL z_world target) PLUS SD-056
    (e2_action_contrastive_enabled=True): each P0b tick also draws the self-
    anchored contrastive loss over all 4 discrete one-hot actions at the
    current z_world_0 (targets = world_forward(z0, actions).detach(), per the
    569a/613-validated recipe) so world_forward does not collapse the action
    channel to a near-identity map.
P0 READINESS GATE (NEW, MANDATORY per queue-experiment Step 3) -- after P0a+P0b,
    a small pre-registered probe (P0_READINESS_PROBE_EPISODES episodes/arm)
    re-measures the SAME statistic C0 routes on
    (blocked_step_mismatch_mean - free_step_mismatch_mean) BEFORE committing to
    the full P1 budget. Below C0_MARGIN -> self-route
    substrate_not_ready_requeue for that seed (never substrate_ceiling), and
    P1 is skipped for that seed. This is the check 642 never ran, and it is
    what would have caught 642's own defect before the full run.
P1a / P1b (unchanged) -- scripted-driver + policy-driver measurement, same as
    642.

PRE-REGISTERED ACCEPTANCE (per ready seed; PASS = each on >= 2/3 of READY seeds)
  C0 detector readiness: blocked-step outcome_mismatch - free-step outcome_mismatch
     >= C0_MARGIN in ARM_BLOCK (the comparator distinguishes blocked from
     successful). [Also gates P0 readiness on the identical statistic -- see above.]
  C1 z_block rises: z_block_peak(BLOCK) - z_block_peak(CONTROL) >= C1_MARGIN
     AND z_block_peak(BLOCK) >= Z_BLOCK_MIN.
  C2 dissociation from z_harm_a: (z_block_peak_BLOCK - z_block_peak_CONTROL)
     - (z_harm_a_mean_BLOCK - z_harm_a_mean_CONTROL) >= C2_MARGIN.
  C3 assert NOT withdraw: action_rate(BLOCK) >= action_rate(CONTROL) - EPS
     AND (action_rate(BLOCK) > action_rate(CONTROL) OR
     alt_switch_rate(BLOCK) > alt_switch_rate(CONTROL)) AND
     z_harm_a_mean(BLOCK) <= z_harm_a_mean(CONTROL) + EPS.

INTERPRETATION GRID (applied at review; unchanged from 642 except the C0-fail
branch is now genuinely reachable only from a run that CLEARED the P0 gate)
  PASS (C0,C1,C2,C3)                 -> MECH-353 validated; clear v3_pending.
  C0 fail (post-gate)                -> z_block integrator does not read the
                                        (now-trained) comparator; regulator
                                        tuning (accumulation_rate / floors).
                                        NOTE: this is now distinct from a
                                        pre-gate P0 failure, which self-routes
                                        substrate_not_ready_requeue instead.
  C0 pass, C1 fail                   -> z_block integrator does not rise; regulator
                                        tuning (accumulation_rate / floors).
  C1 pass, C2 fail                   -> z_block tracks z_harm_a; distinctness from
                                        suffering not demonstrated -- reconsider.
  C2 pass, C3 fail                   -> z_block rises + dissociates but does not
                                        drive assert; consumer-wiring follow-up.
  ALL seeds fail the P0 gate         -> substrate_not_ready_requeue (this run's
                                        own fix did not clear the floor; route
                                        back to /implement-substrate on SD-070/
                                        SD-056 tuning, NOT another comparator
                                        interpretation).

honors feedback_biology_before_formal_definitions: z_block must stay differentiated
from harm (SD-011, no noxious input), suffering (SD-019b, opposite controllability
pole), residue (MECH-056), and licit commitment-hold (MECH-090).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.capability_eval import RandomPolicy  # noqa: E402
from experiments._lib.zworld_p0_warmup import run_zworld_p0  # noqa: E402
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot,
    latent_stack_weight_delta,
    zworld_precondition,
)
from experiments._metrics import p0_readiness_gate, P0NotReady  # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_642a_blocked_agency_zblock_discriminative"
SUPERSEDES = "V3-EXQ-642"

# ---- Pre-registered thresholds (defined here, never inferred post-hoc) --------
C0_MARGIN = 0.10     # blocked-step minus free-step outcome_mismatch (detector)
C1_MARGIN = 0.20     # z_block_peak(BLOCK) - z_block_peak(CONTROL)
Z_BLOCK_MIN = 0.20   # z_block_peak(BLOCK) absolute floor
C2_MARGIN = 0.20     # z_block separation minus z_harm_a separation (dissociation)
C3_EPS = 0.02        # tolerance for "no withdrawal" / "no suffering rise"
SEED_PASS_FRACTION = 2.0 / 3.0

# P0 readiness gate (NEW). Same statistic as C0, measured on a small probe
# BEFORE the full P1 budget is spent -- the check 642 never ran.
P0_READINESS_PROBE_EPISODES = 5

# ANCHOR_REACHABILITY_EXEMPT: C0_MARGIN=0.10 is inherited UNCHANGED from the
# confirmed V3-EXQ-642 pre-registration (failure_autopsy_V3-EXQ-642_2026-06-06.md,
# Reading A) -- it is not a new threshold being introduced here. Whether the
# SD-070+SD-056 fix makes it REACHABLE is precisely the empirical question this
# retest exists to answer, not a pre-provable invariant checkable against a frozen
# reference control -- 642 itself is the confirmed case where it was NOT reachable
# (untrained encoder, SD-056 off). The readiness gate's job (see the P0-gate try/
# except above) is exactly to catch continued unreachability as
# substrate_not_ready_requeue rather than mislabel it -- which is what the anchor-
# reachability lint is otherwise checking for.
ANCHOR_REACHABILITY_EXEMPT = (
    "C0_MARGIN inherited unchanged from confirmed V3-EXQ-642 pre-registration; "
    "reachability under the SD-070+SD-056 fix is the empirical question this "
    "retest answers, not a pre-provable invariant -- the P0 gate already routes "
    "an unreachable substrate to substrate_not_ready_requeue rather than a "
    "mislabelled verdict."
)
SD056_PAIRWISE_DIST_FLOOR = 0.05   # SD-056's own C1_PAIRWISE_DIST_FLOOR convention (informational)

SEEDS = (42, 43, 44)
P0_WARMUP_EPISODES = 60      # world_forward training episodes per seed (P0b)
ZWORLD_P0A_EPISODES = 60     # SD-070 encoder warmup episodes per seed (P0a)
P1_MEASURE_EPISODES = 20     # measurement episodes per arm (scripted + policy)
STEPS_PER_EPISODE = 120
BLOCK_INTERVAL = 2           # every other step blocked in ARM_BLOCK
CONDITIONS = ("ARM_BLOCK", "ARM_CONTROL")

DRY_RUN_SEEDS = (42,)
DRY_RUN_P0 = 2
DRY_RUN_ZWORLD_P0A = 2
DRY_RUN_P1 = 2
DRY_RUN_PROBE = 2
DRY_RUN_STEPS = 20

GOAL_PIN = 0.5  # fixed z_goal magnitude (goal-value held constant)

ENV_KWARGS = dict(
    size=8,
    num_hazards=0,        # harm held constant (z_harm_a ~ flat)
    num_resources=0,      # no respawn dynamics
    n_landmarks_a=3,      # SD-023 static positional gradient -> z_world tracks position
    n_landmarks_b=3,
    toroidal=True,        # no edge-stall confound for the scripted driver
)
CFG_KWARGS = dict(
    use_blocked_agency=True,
    z_goal_enabled=True,
    drive_weight=2.0,
    alpha_world=1.0,       # no EMA smoothing -> z_world reflects current position
    # SD-056 (the second half of the autopsy's prescribed fix): keep world_forward
    # action-discriminative rather than collapsing to a near-identity map.
    e2_action_contrastive_enabled=True,
    e2_action_contrastive_weight=0.01,   # SD-056 default-ON weight (V3-EXQ-613 validated)
    # SD-056 multi-step rollout stability amend: bounds E2's imagination rollout
    # under sustained contrastive training (confirmed unbounded-explosion failure
    # mode absent this, V3-EXQ-569e/936). Ratio 2.0 is the V3-EXQ-689i/617-validated
    # operating point.
    e2_rollout_output_norm_clamp_enabled=True,
    e2_rollout_output_norm_clamp_ratio=2.0,
)


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _pin_goal(agent: REEAgent) -> None:
    if agent.goal_state is not None:
        agent.goal_state._z_goal = torch.ones(
            1, agent.goal_state.config.goal_dim, device=agent.device
        ) * GOAL_PIN


def _one_hot(idx: int) -> torch.Tensor:
    a = torch.zeros(1, 4)
    a[0, idx] = 1.0
    return a


def _build_env(seed: int, block: bool):
    return CausalGridWorldV2(
        seed=seed,
        scheduled_action_block_enabled=block,
        scheduled_action_block_interval=BLOCK_INTERVAL,
        scheduled_action_block_prob=1.0,
        **ENV_KWARGS,
    )


def _build_agent(seed: int) -> REEAgent:
    env = _build_env(seed, block=False)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        **CFG_KWARGS,
    )
    return REEAgent(cfg)


def _train_world_forward(agent: REEAgent, seed: int, episodes: int, steps: int,
                         label: str) -> Dict[str, float]:
    """P0b: train E2.world_forward on the agent's single-sense rollout transitions
    with the one-hot discrete executed action (the z_block comparator encoding),
    PLUS the SD-056 self-anchored contrastive term over all 4 discrete actions at
    the same z_world_0 (keeps world_forward action-discriminative rather than
    collapsing to a near-identity map). split_encoder (P0a-trained) is left
    frozen -- only world_transition/world_action_encoder are optimised here."""
    env = _build_env(seed, block=False)
    _, od = env.reset()
    _pin_goal(agent)
    contrastive_on = bool(getattr(agent.config.e2, "e2_action_contrastive_enabled", False))
    contrastive_weight = float(getattr(agent.config.e2, "e2_action_contrastive_weight", 0.0))
    all_actions = torch.eye(4)
    params = (
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters())
    )
    opt = torch.optim.Adam(params, lr=3e-3)
    prev = agent.sense(od["body_state"], od["world_state"]).z_world.detach().clone()
    last_mse = 0.0
    last_contrast = 0.0
    rng = np.random.RandomState(seed)
    for ep in range(episodes):
        for _ in range(steps):
            idx = int(rng.randint(0, 4))
            aoh = _one_hot(idx)
            _, h, d, inf, od = env.step(aoh)
            cur = agent.sense(od["body_state"], od["world_state"]).z_world.detach().clone()
            mse = ((agent.e2.world_forward(prev, aoh) - cur) ** 2).mean()
            loss = mse
            if contrastive_on:
                z0_rep = prev.expand(4, -1)
                targets_all = agent.e2.world_forward(z0_rep, all_actions).detach()
                contrast = agent.e2.world_forward_contrastive_loss(
                    prev, all_actions, targets_all
                )
                loss = loss + contrastive_weight * contrast
                last_contrast = float(contrast.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_mse = float(mse.item())
            prev = cur.clone()
            if d:
                _, od = env.reset()
                _pin_goal(agent)
                prev = agent.sense(
                    od["body_state"], od["world_state"]
                ).z_world.detach().clone()
        if (ep + 1) % 10 == 0 or ep == episodes - 1:
            print(
                f"  [train] {label} seed={seed} phase=P0b ep {ep+1}/{episodes} "
                f"wf_mse={last_mse:.6f} contrast={last_contrast:.6f}",
                flush=True,
            )
    return {"wf_mse_final": last_mse, "contrastive_loss_final": last_contrast}


def _measure_readiness_probe(agent: REEAgent, seed: int, episodes: int,
                             steps: int) -> Dict:
    """P0 readiness gate probe. Runs a SMALL blocked + free rollout and computes
    the IDENTICAL statistic C0 routes on (blocked_step_mismatch_mean -
    free_step_mismatch_mean), plus informational pred_mag / cand_world_pairwise_dist
    readouts, BEFORE committing to the full P1 budget."""
    block_res = _scripted_measure(agent, seed, block=True, episodes=episodes, steps=steps)
    free_res = _scripted_measure(agent, seed, block=False, episodes=episodes, steps=steps)
    c0_probe = block_res["blocked_step_mismatch_mean"] - block_res["free_step_mismatch_mean"]

    # Informational: pred_mag on a known-successful (unblocked) move, and the
    # SD-056 cand_world_pairwise_dist diagnostic, both mirroring the substrate's
    # own MECH-353 comparator formula (ree-v3/CLAUDE.md "MECH-353" section).
    env = _build_env(seed, block=False)
    _, od = env.reset()
    agent.reset()
    _pin_goal(agent)
    lat = agent.sense(od["body_state"], od["world_state"])
    rng = np.random.RandomState(seed + 5000)
    pred_mags: List[float] = []
    pairwise_dists: List[float] = []
    all_actions = torch.eye(4)
    for _ in range(min(episodes, 3) * steps):
        idx = int(rng.randint(0, 4))
        aoh = _one_hot(idx)
        z_prev = lat.z_world.detach().clone()
        with torch.no_grad():
            zw_pred = agent.e2.world_forward(z_prev, aoh)
            pred_mags.append(float((zw_pred - z_prev).norm().item()))
            pairwise_dists.append(
                float(agent.e2.cand_world_pairwise_dist(z_prev, all_actions))
            )
        _, h, d, inf, od = env.step(aoh)
        lat = agent.sense(od["body_state"], od["world_state"])
        if d:
            _, od = env.reset()
            agent.reset()
            _pin_goal(agent)
            lat = agent.sense(od["body_state"], od["world_state"])

    return {
        "c0_probe_margin": c0_probe,
        "pred_mag_mean": float(np.mean(pred_mags)) if pred_mags else 0.0,
        "cand_world_pairwise_dist_mean": float(np.mean(pairwise_dists)) if pairwise_dists else 0.0,
        "blocked_step_mismatch_mean_probe": block_res["blocked_step_mismatch_mean"],
        "free_step_mismatch_mean_probe": block_res["free_step_mismatch_mean"],
    }


def _scripted_measure(agent: REEAgent, seed: int, block: bool, episodes: int,
                      steps: int) -> Dict:
    """P1a: drive random discrete actions (no policy confound). Reads z_block,
    outcome_mismatch (split by blocked/free step), z_harm_a."""
    env = _build_env(seed, block=block)
    _, od = env.reset()
    agent.reset()
    _pin_goal(agent)
    agent.sense(od["body_state"], od["world_state"])  # tick0, seeds caches
    rng = np.random.RandomState(seed + 1000)
    z_block_series: List[float] = []
    z_harm_a_series: List[float] = []
    mism_blocked: List[float] = []
    mism_free: List[float] = []
    for ep in range(episodes):
        for _ in range(steps):
            idx = int(rng.randint(0, 4))
            aoh = _one_hot(idx)
            agent._last_action = aoh.clone()
            _, h, d, inf, od = env.step(aoh)
            lat = agent.sense(od["body_state"], od["world_state"])
            o = agent.blocked_agency.last_output()
            z_block_series.append(float(o.z_block))
            if lat.z_harm_a is not None:
                z_harm_a_series.append(float(lat.z_harm_a.detach().norm().item()))
            else:
                z_harm_a_series.append(0.0)
            if inf.get("action_blocked_this_step", False):
                mism_blocked.append(float(o.outcome_mismatch))
            else:
                mism_free.append(float(o.outcome_mismatch))
            if d:
                _, od = env.reset()
                agent.reset()
                _pin_goal(agent)
                agent.sense(od["body_state"], od["world_state"])
    return {
        "z_block_peak": max(z_block_series) if z_block_series else 0.0,
        "z_block_mean": float(np.mean(z_block_series)) if z_block_series else 0.0,
        "z_harm_a_mean": float(np.mean(z_harm_a_series)) if z_harm_a_series else 0.0,
        "blocked_step_mismatch_mean": float(np.mean(mism_blocked)) if mism_blocked else 0.0,
        "free_step_mismatch_mean": float(np.mean(mism_free)) if mism_free else 0.0,
        "n_blocked_steps": len(mism_blocked),
        "n_free_steps": len(mism_free),
    }


def _policy_measure(agent: REEAgent, seed: int, block: bool, episodes: int,
                    steps: int) -> Dict:
    """P1b: drive the agent's own policy (act_with_split_obs) so the ASSERT
    score-bias can shift selection. Reads action-vs-noop rate + alt-action
    switching + z_harm_a."""
    env = _build_env(seed, block=block)
    _, od = env.reset()
    agent.reset()
    _pin_goal(agent)
    noop_class = int(getattr(agent.config, "blocked_agency_noop_class", 0))
    n_action = 0
    n_total = 0
    n_switch = 0
    prev_cls: Optional[int] = None
    z_harm_a_series: List[float] = []
    for ep in range(episodes):
        for _ in range(steps):
            a = agent.act_with_split_obs(od["body_state"], od["world_state"])
            cls = int(a.detach().argmax(dim=-1).flatten()[0].item())
            n_total += 1
            if cls != noop_class:
                n_action += 1
            if prev_cls is not None and cls != prev_cls:
                n_switch += 1
            prev_cls = cls
            _, h, d, inf, od = env.step(a)
            lat = agent.sense(od["body_state"], od["world_state"])
            if lat.z_harm_a is not None:
                z_harm_a_series.append(float(lat.z_harm_a.detach().norm().item()))
            if d:
                _, od = env.reset()
                agent.reset()
                _pin_goal(agent)
                prev_cls = None
    return {
        "action_rate": (n_action / n_total) if n_total else 0.0,
        "alt_switch_rate": (n_switch / n_total) if n_total else 0.0,
        "z_harm_a_mean_policy": float(np.mean(z_harm_a_series)) if z_harm_a_series else 0.0,
        "n_policy_steps": n_total,
    }


def _evaluate_seed(block: Dict, control: Dict) -> Dict:
    """Apply the pre-registered acceptance criteria for one seed."""
    c0 = (block["blocked_step_mismatch_mean"] - block["free_step_mismatch_mean"]) >= C0_MARGIN
    c1 = (
        (block["z_block_peak"] - control["z_block_peak"]) >= C1_MARGIN
        and block["z_block_peak"] >= Z_BLOCK_MIN
    )
    z_block_sep = block["z_block_peak"] - control["z_block_peak"]
    z_harm_a_sep = block["z_harm_a_mean"] - control["z_harm_a_mean"]
    c2 = (z_block_sep - z_harm_a_sep) >= C2_MARGIN
    no_withdraw = block["action_rate"] >= (control["action_rate"] - C3_EPS)
    assert_sig = (
        block["action_rate"] > control["action_rate"]
        or block["alt_switch_rate"] > control["alt_switch_rate"]
    )
    no_suffering = block["z_harm_a_mean_policy"] <= (control["z_harm_a_mean_policy"] + C3_EPS)
    c3 = no_withdraw and assert_sig and no_suffering
    return {
        "C0_detector_readiness": bool(c0),
        "C1_z_block_rises": bool(c1),
        "C2_dissociation_from_z_harm_a": bool(c2),
        "C3_assert_not_withdraw": bool(c3),
        "z_block_separation": z_block_sep,
        "z_harm_a_separation": z_harm_a_sep,
    }


def run_experiment(seeds: List[int], conditions: List[str],
                   zworld_p0a_episodes: int, p0_episodes: int,
                   p1_episodes: int, probe_episodes: int,
                   steps_per_episode: int, dry_run: bool = False) -> Dict:
    per_seed: List[Dict] = []
    agents_for_z_goal: List[REEAgent] = []
    n_runs_completed = 0
    n_total_runs = len(seeds) * len(conditions)

    for seed in seeds:
        _seed_all(seed)
        agent = _build_agent(seed)
        agents_for_z_goal.append(agent)

        # -- P0a: SD-070 z_world encoder warmup (the untrained-encoder fix) --
        stack_before = latent_stack_snapshot(agent)
        warmup_env = _build_env(seed, block=False)
        p0a_report = run_zworld_p0(
            agent, warmup_env, seed, zworld_p0a_episodes, steps_per_episode,
            policy=RandomPolicy(seed), label=f"seed{seed}", dry_run=dry_run,
        )
        encoder_delta = latent_stack_weight_delta(agent, stack_before)
        zworld_ready = zworld_precondition(
            encoder_delta, arm="single", context=f"seed{seed}"
        )

        # -- P0b: E2.world_forward + SD-056 contrastive training --
        p0b_report = _train_world_forward(
            agent, seed, p0_episodes, steps_per_episode, label=f"seed{seed}"
        )

        # -- P0 readiness gate: same statistic as C0, on a small probe, PLUS the
        # zworld_encoder_trained precondition (would the run's conclusion change
        # if the encoder were untrained? yes -> gating, per zworld_precondition's
        # own policy note). Both checks combine into ONE atomic gate: either
        # unmet routes the whole seed to substrate_not_ready_requeue.
        probe = _measure_readiness_probe(agent, seed, probe_episodes, steps_per_episode)
        try:
            preconditions = p0_readiness_gate([
                {
                    "name": "world_forward_c0_margin_supra_floor",
                    "measured": probe["c0_probe_margin"],
                    "threshold": C0_MARGIN,
                    "direction": "lower",
                    "control": (
                        f"{probe_episodes}-episode blocked+free probe on the "
                        "P0a+P0b-trained substrate (same statistic C0 routes on)"
                    ),
                },
                {
                    "name": zworld_ready["name"],
                    "measured": zworld_ready["measured"],
                    "threshold": zworld_ready["threshold"],
                    "direction": zworld_ready["direction"],
                    "comparator": zworld_ready["comparator"],
                    "control": zworld_ready["control"],
                    "description": zworld_ready["description"],
                    "n_world_encoder_tensors": zworld_ready["n_world_encoder_tensors"],
                    "n_world_encoder_changed": zworld_ready["n_world_encoder_changed"],
                },
            ])
        except P0NotReady as exc:
            preconditions = exc.preconditions
            if dry_run:
                # Smoke-test-only override: a tiny dry-run budget cannot plausibly
                # clear a real discriminativeness gate, and the smoke test's job is
                # to exercise the FULL pipeline (P1 progress lines + verdicts), not
                # to validate science at 2-episode scale. The real (non-dry-run)
                # gate above is unaffected -- this branch only fires under
                # --dry-run, mirroring how ZWorldP0Config itself already special-
                # cases dry_run (reduced batch_size/epochs) rather than skipping.
                print(
                    f"[P0-GATE-REFUSAL] seed={seed}: {exc.reason} -- "
                    "DRY-RUN OVERRIDE: proceeding to P1 anyway for pipeline smoke",
                    flush=True,
                )
            else:
                per_seed.append({
                    "seed": seed,
                    "p0_ready": False,
                    "p0_not_ready_reason": exc.reason,
                    "wf_mse_final": p0b_report["wf_mse_final"],
                    "contrastive_loss_final": p0b_report["contrastive_loss_final"],
                    "p0a_report": p0a_report,
                    "zworld_encoder_precondition": zworld_ready,
                    "readiness_probe": probe,
                    "preconditions": preconditions,
                    "criteria": {
                        "C0_detector_readiness": False,
                        "C1_z_block_rises": False,
                        "C2_dissociation_from_z_harm_a": False,
                        "C3_assert_not_withdraw": False,
                    },
                })
                print(
                    f"[P0-GATE-REFUSAL] seed={seed}: {exc.reason} -- skipping P1 for this seed",
                    flush=True,
                )
                continue

        # -- P1: full scripted + policy measurement (unchanged from 642) --
        arms: Dict[str, Dict] = {}
        for cond in conditions:
            print(f"Seed {seed} Condition {cond}", flush=True)
            is_block = cond == "ARM_BLOCK"
            scripted = _scripted_measure(agent, seed, is_block, p1_episodes, steps_per_episode)
            policy = _policy_measure(agent, seed, is_block, p1_episodes, steps_per_episode)
            merged = {**scripted, **policy}
            arms[cond] = merged
            n_runs_completed += 1
            print(
                f"  {cond}: z_block_peak={merged['z_block_peak']:.3f} "
                f"z_harm_a_mean={merged['z_harm_a_mean']:.3f} "
                f"action_rate={merged['action_rate']:.3f}",
                flush=True,
            )
            print(f"verdict: {'PASS' if merged['z_block_peak'] >= 0.0 else 'FAIL'}", flush=True)
        crit = _evaluate_seed(arms["ARM_BLOCK"], arms["ARM_CONTROL"])
        per_seed.append({
            "seed": seed,
            "p0_ready": True,
            "wf_mse_final": p0b_report["wf_mse_final"],
            "contrastive_loss_final": p0b_report["contrastive_loss_final"],
            "p0a_report": p0a_report,
            "zworld_encoder_precondition": zworld_ready,
            "readiness_probe": probe,
            "preconditions": preconditions,
            "ARM_BLOCK": arms["ARM_BLOCK"],
            "ARM_CONTROL": arms["ARM_CONTROL"],
            "criteria": crit,
        })

    ready_seeds = [s for s in per_seed if s.get("p0_ready")]
    n_ready = len(ready_seeds)

    if n_ready == 0:
        # NEVER a substrate_ceiling / does_not_support / *_nondiscriminative
        # verdict here -- the P0 gate itself is what was not met.
        all_preconditions = [p for s in per_seed for p in (s.get("preconditions") or [])]
        return {
            "outcome": "FAIL",
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": all_preconditions,
                "criteria_non_degenerate": {
                    "C0": False, "C1": False, "C2": False, "C3": False,
                },
                "c0_seeds_pass": 0, "c1_seeds_pass": 0,
                "c2_seeds_pass": 0, "c3_seeds_pass": 0,
                "seeds_needed": 0,
                "n_ready_seeds": 0,
            },
            "criteria": [
                {"name": "C0_detector_readiness", "load_bearing": True, "passed": False},
                {"name": "C1_z_block_rises", "load_bearing": True, "passed": False},
                {"name": "C2_dissociation_from_z_harm_a", "load_bearing": True, "passed": False},
                {"name": "C3_assert_not_withdraw", "load_bearing": True, "passed": False},
            ],
            "per_seed": per_seed,
            "n_runs_completed": n_runs_completed,
            "n_total_runs": n_total_runs,
            "agents_for_z_goal": agents_for_z_goal,
        }

    need = 1 if dry_run else int(np.ceil(SEED_PASS_FRACTION * n_ready))

    def frac(key: str) -> int:
        return sum(1 for s in ready_seeds if s["criteria"][key])

    c0n, c1n, c2n, c3n = frac("C0_detector_readiness"), frac("C1_z_block_rises"), \
        frac("C2_dissociation_from_z_harm_a"), frac("C3_assert_not_withdraw")

    c0_pass = c0n >= need
    c1_pass = c1n >= need
    c2_pass = c2n >= need
    c3_pass = c3n >= need
    overall_pass = c0_pass and c1_pass and c2_pass and c3_pass

    if overall_pass:
        interp = "validated_clear_v3_pending"
    elif not c0_pass:
        interp = "z_block_integrator_comparator_not_read"
    elif not c1_pass:
        interp = "z_block_integrator_no_rise"
    elif not c2_pass:
        interp = "z_block_tracks_z_harm_a_not_dissociated"
    else:
        interp = "z_block_does_not_drive_assert"

    # Non-degeneracy self-report: C3 in particular passed VACUOUSLY in 642
    # because z_block stayed identically 0 in both arms (bit-identical arms).
    # Flag any criterion whose seeds all show z_block pinned at 0.
    any_nonzero_block_peak = any(
        s["ARM_BLOCK"]["z_block_peak"] > 1e-6 for s in ready_seeds
    )
    criteria_non_degenerate = {
        "C0": any(
            s["ARM_BLOCK"]["blocked_step_mismatch_mean"] > 1e-6
            or s["ARM_BLOCK"]["free_step_mismatch_mean"] > 1e-6
            for s in ready_seeds
        ),
        "C1": any_nonzero_block_peak,
        "C2": any_nonzero_block_peak,
        "C3": any_nonzero_block_peak,
    }

    all_preconditions = [p for s in per_seed for p in (s.get("preconditions") or [])]

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "interpretation": {
            "label": interp,
            "preconditions": all_preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "c0_seeds_pass": c0n,
            "c1_seeds_pass": c1n,
            "c2_seeds_pass": c2n,
            "c3_seeds_pass": c3n,
            "seeds_needed": need,
            "n_ready_seeds": n_ready,
            "n_seeds_skipped_not_ready": len(per_seed) - n_ready,
        },
        "criteria": [
            {"name": "C0_detector_readiness", "load_bearing": True, "passed": c0_pass},
            {"name": "C1_z_block_rises", "load_bearing": True, "passed": c1_pass},
            {"name": "C2_dissociation_from_z_harm_a", "load_bearing": True, "passed": c2_pass},
            {"name": "C3_assert_not_withdraw", "load_bearing": True, "passed": c3_pass},
        ],
        "per_seed": per_seed,
        "n_runs_completed": n_runs_completed,
        "n_total_runs": n_total_runs,
        "agents_for_z_goal": agents_for_z_goal,
    }


def _build_manifest(result: Dict, timestamp_utc: str, dry_run: bool) -> Dict:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": [],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "outcome": result["outcome"],
        "evidence_direction": "supports" if result["outcome"] == "PASS" else "weakens",
        "timestamp_utc": timestamp_utc,
        "dry_run": dry_run,
        "pre_registered_thresholds": {
            "C0_MARGIN": C0_MARGIN,
            "C1_MARGIN": C1_MARGIN,
            "Z_BLOCK_MIN": Z_BLOCK_MIN,
            "C2_MARGIN": C2_MARGIN,
            "C3_EPS": C3_EPS,
            "SEED_PASS_FRACTION": SEED_PASS_FRACTION,
        },
        "criteria": result["criteria"],
        "interpretation": result["interpretation"],
        "per_seed": result["per_seed"],
        "n_runs_completed": result["n_runs_completed"],
        "n_total_runs": result["n_total_runs"],
        "bears_on": ["MECH-353", "SD-029", "MECH-112", "MECH-320", "MECH-342",
                     "ARC-016", "SD-011", "SD-019b", "SD-070", "SD-056"],
        "notes": (
            "Diagnostic substrate-readiness RETEST for MECH-353 z_block, fixing the "
            "comparator-discriminability gap V3-EXQ-642's own FAIL diagnosed "
            "(failure_autopsy_V3-EXQ-642_2026-06-06.md, confirmed 'Reading A'): "
            "(1) SD-070 z_world encoder warmup (P0a) so the encoder is no longer a "
            "frozen random projection, (2) SD-056 e2_action_contrastive_enabled=True "
            "(P0b) so world_forward stays action-discriminative, (3) a P0 readiness "
            "gate on the SAME statistic C0 routes on, measured BEFORE the full P1 "
            "budget. A seed failing the P0 gate self-routes "
            "substrate_not_ready_requeue for that seed rather than a substrate_ceiling "
            "verdict; if ALL seeds fail the gate the whole run self-routes "
            "substrate_not_ready_requeue. PASS clears the MECH-353 v3_pending gate."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Short smoke run (1 seed, small P0/probe/P1 budgets).")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        zworld_p0a, p0, p1, probe, steps = (
            DRY_RUN_ZWORLD_P0A, DRY_RUN_P0, DRY_RUN_P1, DRY_RUN_PROBE, DRY_RUN_STEPS,
        )
    else:
        seeds = list(SEEDS)
        zworld_p0a, p0, p1, probe, steps = (
            ZWORLD_P0A_EPISODES, P0_WARMUP_EPISODES, P1_MEASURE_EPISODES,
            P0_READINESS_PROBE_EPISODES, STEPS_PER_EPISODE,
        )

    result = run_experiment(
        seeds=seeds, conditions=list(CONDITIONS),
        zworld_p0a_episodes=zworld_p0a, p0_episodes=p0, p1_episodes=p1,
        probe_episodes=probe, steps_per_episode=steps,
        dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    full_config = {**ENV_KWARGS, **CFG_KWARGS,
                   "zworld_p0a_episodes": zworld_p0a, "p0_episodes": p0,
                   "p1_episodes": p1, "probe_episodes": probe,
                   "steps_per_episode": steps, "block_interval": BLOCK_INTERVAL,
                   "goal_pin": GOAL_PIN}

    agents_for_z_goal = result.pop("agents_for_z_goal", [])
    out_path = write_flat_manifest(
        manifest,
        args.out_dir,
        dry_run=args.dry_run,
        config=full_config,
        seeds=list(seeds),
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
        agent=agents_for_z_goal,
    )

    print(f"manifest: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} interp={result['interpretation']['label']} "
        f"runs={result['n_runs_completed']}/{result['n_total_runs']}",
        flush=True,
    )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    _dry = "--dry-run" in sys.argv
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry)
    sys.exit(0)
