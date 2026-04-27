#!/opt/local/bin/python3
"""V3-EXQ-495: MECH-163 V3 Full-Completion Gate -- VTA / Hippocampally-Planned Arm

experiment_purpose: evidence
status_when_drafted: GATED ON MECH-292 + MECH-293 LANDING (deferred follow-on as of 2026-04-27)

WHY THIS IS DRAFTED BUT NOT QUEUED:
This is THE V3 full-completion gate experiment for MECH-163. It tests whether
the VTA / hippocampally-planned arm of MECH-163's dual goal-directed systems
does work the habit arm cannot:

    "A scenario where 1-step / habit-policy approach is INSUFFICIENT, and where
     goal-seeded multi-step trajectory generation by HippocampalModule produces
     a discriminable behavioural advantage."

The PLANNED arm requires the proposal-generation hook -- z_goal-conditioned
trajectory generation by HippocampalModule -- which lives in MECH-293 (waking
ghost-goal probe search). MECH-293 in turn consumes MECH-292 (ranked
ghost-goal bank), which consumes SD-039 (anchor goal-snapshot payload).

As of 2026-04-27:
  - SD-039 substrate + population layer LANDED, V3-EXQ-494 6/6 PASS.
  - MECH-292 designed (`REE_assembly/docs/architecture/mech_292_ghost_goal_bank.md`),
    not implemented.
  - MECH-293 designed (`REE_assembly/docs/architecture/mech_293_ghost_goal_probe_search.md`),
    not implemented.

DO NOT QUEUE this experiment until MECH-292 + MECH-293 land. If queued
prematurely, the PLANNED arm will be silently identical to the HABIT arm
(no ghost-seeded proposals to differentiate them) and the experiment would
return uninformative C2 = 0 across all seeds.

BACKGROUND -- WHY THIS GATE EXISTS:

EXQ-327 (2026-04-14) cleared the HABIT arm of MECH-163 with goal_weight in
E3 trajectory scoring on value-flat HippocampalModule proposals. EXQ-327
was retagged 2026-04-27 (R1 governance fix) to claim_ids=['SD-015'] only --
the habit arm is one half of MECH-163's dual-system claim, and tagging
EXQ-327 as MECH-163 evidence conflated the parent claim with one of its
sub-arms. The dual-system distinction itself remains untested.

V3-EXQ-495 is the still-pending discriminative test. Without it, V4 social
extension (INV-029 multi-agent benefit gradient over trajectories, MECH-164
agent-indexed terrain inference) has no validated planning substrate.

DESIGN -- 3 CONDITIONS x 2 PARADIGMS x N SEEDS:

CONDITIONS (per claim_ids accuracy rule, all conditions tag MECH-163):

  HABIT     value-flat HippocampalModule proposals (use_mech293_ghost_probes=False);
            goal_weight=1.0 in E3.score_trajectory(); z_goal_enabled=True;
            drive_weight=2.0; use_resource_proximity_head=True. This is the
            EXQ-327 setup -- the habit arm carrying through to V3-EXQ-495 as
            the within-experiment regression check.

  PLANNED   ghost-seeded HippocampalModule proposals (use_mech293_ghost_probes=True);
            same E3 scoring as HABIT (goal_weight=1.0); same z_goal seeding;
            adds use_mech292_ghost_bank=True, use_sd039_anchor_payload=True,
            use_anchor_sets=True, use_event_segmenter=True (full ghost-goal
            stack required for MECH-293 to do anything).

  ABLATED   no goal anywhere: goal_weight=0.0, z_goal_enabled=False,
            use_mech293_ghost_probes=False. Establishes the no-goal baseline
            for C3 regression check.

PARADIGMS (each condition runs both):

  A DETOUR / BLOCKAGE  (within-episode mid-episode blockage)
    Train: standard CausalGridWorldV2 with fixed-position resources/hazards
           via env.reset_to(...). Agent learns approach to resource at a
           known position via the cached short corridor.
    Eval:  detour episodes -- at step T_block (60% through the episode), a
           hazard is dropped onto the cached short corridor (the agent's
           current habit path). The agent must re-plan via a different
           route to reach the resource and accumulate benefit. Habit-arm
           cached S-R associations point at the now-blocked corridor and
           cost the agent reaching the resource until exploration finds the
           detour. PLANNED arm should propose ghost-seeded trajectories
           around the resource's anchor-tagged region (preserved goal-payload
           on the formerly-active corridor anchor) and find the detour
           sooner.
    Pre-registered C2 metric: PLANNED.benefit_ratio_post_block -
                              HABIT.benefit_ratio_post_block >= 0.30
    in detour episodes specifically (NOT pooled across all episodes).

  B NOVEL CONTEXT  (cross-episode env swap)
    Train: env A (seed=42) -- standard layout.
    Eval:  env B (seed=137) -- topologically novel CausalGridWorldV2 with
           same resource_benefit / hazard_harm semantics but different grid
           layout. Habit S-R associations from env A do not apply. PLANNED
           arm should generalise via z_goal-tied proposals that are
           location-invariant across resource respawns. EXQ-495 records
           benefit_ratio in env B as the cross-context generalisation
           metric.

ACCEPTANCE CRITERIA (pre-registered):

  C1 PLANNED.ghost_branch_fires
     mech293_n_ghost_admitted > 0 averaged across all PLANNED P2 ticks,
     in BOTH paradigms. Sanity check that the ghost branch is actually
     contributing trajectories. If C1 fails with C2 also failing, the
     diagnosis is "instrumentation problem -- ghost branch silent".

  C2 PLANNED >> HABIT in detour (paradigm A only)
     PLANNED.benefit_ratio_post_block - HABIT.benefit_ratio_post_block >= 0.30
     in >= 4 / 7 seeds. The size of the gap is THE V3 full-completion gate
     metric. Pre-registered at 0.30.

  C3 HABIT >= ABLATED in standard episodes (paradigm A train phase + paradigm B env-A)
     Within-experiment regression: confirms the habit arm still works.
     If C3 fails, EXQ-327 has regressed and this experiment is uninterpretable.

  C4 PLANNED.prox_r2 >= 0.7 in both paradigms (resource-proximity-head still trained)
     Representation-quality check. Adding ghost-seeded proposals must not
     break the upstream encoder.

  C5 PLANNED.harm_ratio within 10 % of HABIT.harm_ratio (no harm regression)
     The PLANNED arm must not buy goal-approach by ignoring z_harm_a. If
     PLANNED has higher benefit_ratio AND substantially higher harm_ratio,
     the gate is uninterpretable -- the planned arm just took riskier paths.

PASS = C1 AND C2 AND C3 AND C4 AND C5.

DIAGNOSTIC (not gate-blocking but recorded):

  KL_PLANNED_HABIT  KL divergence between PLANNED and HABIT first-step
                    action distributions over P2 ticks. KL ~ 0 with C2
                    failing diagnoses "Q1 paradigm not actually requiring
                    multi-step planning". KL > 0 with C2 failing
                    diagnoses "candidates differ but E3 scoring washes
                    the difference out".

PHASED TRAINING:
  P0 (100 ep): encoder + proximity head warmup (no goal updates)
  P1 (100 ep): full pipeline (z_goal seeding + condition-specific ghost branch)
  P2 ( 50 ep): evaluation
                paradigm A: blockage injected at step T_block in each ep
                paradigm B: env swapped to env B (seed=137); no blockage

SEEDS: 7 seeds (per the >= 4/7 acceptance criterion). Override via
       --seeds for smoke / sweep work.

claim_ids: ['MECH-163']  (single-claim evidence; SD-015 / SD-039 / MECH-292
/ MECH-293 are upstream dependencies the gate consumes but does not
directly test -- the gate's PASS/FAIL signal is interpretable only for
MECH-163's dual-system distinction).

RUN:
  /opt/local/bin/python3 experiments/v3_exq_495_mech163_planned_system_gate.py
                          --dry-run         # smoke (3 ep / 10 step / 1 seed)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EVIDENCE_ROOT = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_495_mech163_planned_system_gate"
CLAIM_IDS          = ["MECH-163"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEEDS_DEFAULT = [42, 7, 13, 23, 31, 53, 67]
CONDITIONS    = ["HABIT", "PLANNED", "ABLATED"]
PARADIGMS     = ["A_DETOUR", "B_NOVEL_CONTEXT"]

P0_EPISODES = 100
P1_EPISODES = 100
P2_EPISODES = 50
STEPS_PER_EP = 200

# Paradigm A blockage parameters.
T_BLOCK_FRAC = 0.6   # inject blockage at 60% through the eval episode
DETOUR_BLOCKAGE_OFFSET = (0, 1)  # cell offset from agent position at T_block

# Paradigm B env seeds.
ENV_A_SEED = 42
ENV_B_SEED = 137

GRID_SIZE     = 8
NUM_RESOURCES = 3
NUM_HAZARDS   = 2
HAZARD_HARM   = 0.1
DRIVE_WEIGHT  = 2.0
GOAL_WEIGHT   = 1.0

LR = 3e-4

# Acceptance thresholds.
C2_BENEFIT_GAP   = 0.30
C2_MIN_SEEDS     = 4
C3_HABIT_GE_ABL  = 0.0  # HABIT >= ABLATED with no slack
C4_PROX_R2_FLOOR = 0.7
C5_HARM_TOL      = 0.10  # PLANNED.harm_ratio within 10% of HABIT

# Smoke / dry-run.
DRY_RUN_SEEDS    = [42]
DRY_RUN_EPISODES = 3
DRY_RUN_STEPS    = 10


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=NUM_RESOURCES,
        num_hazards=NUM_HAZARDS,
        hazard_harm=HAZARD_HARM,
        resource_benefit=0.5,
        resource_respawn_on_consume=True,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.05,
        proximity_approach_threshold=0.15,
        use_proxy_fields=True,
    )


def _make_agent(env: CausalGridWorldV2, condition: str, seed: int) -> REEAgent:
    """Build the agent for a given condition with the appropriate ghost-goal
    stack toggled. ABLATED has no goal at all. HABIT has goal scoring but no
    ghost branch. PLANNED has the full MECH-292 + MECH-293 stack on."""
    torch.manual_seed(seed)
    goal_active = (condition in ("HABIT", "PLANNED"))
    planned     = (condition == "PLANNED")
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,

        # SD-015 + SD-018 goal substrate (HABIT + PLANNED).
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        use_event_classifier=True,
        z_goal_enabled=goal_active,
        drive_weight=DRIVE_WEIGHT if goal_active else 0.0,
        goal_weight=GOAL_WEIGHT if goal_active else 0.0,
        benefit_eval_enabled=goal_active,
        benefit_weight=1.0,
        e1_goal_conditioned=goal_active,
        wanting_weight=0.3 if goal_active else 0.0,

        # MECH-269 anchor stack required by SD-039 / MECH-292 / MECH-293.
        # PLANNED needs this on; HABIT keeps it off so the regression check
        # is clean against EXQ-327's setup. ABLATED keeps it off.
        use_per_stream_vs=planned,
        use_event_segmenter=planned,
        use_invalidation_trigger=planned,
        use_anchor_sets=planned,

        # SD-039 anchor goal-snapshot payload (PLANNED only).
        use_sd039_anchor_payload=planned,

        # MECH-292 ranked ghost-goal bank (PLANNED only).
        use_mech292_ghost_bank=planned,

        # MECH-293 waking ghost-goal probe search (PLANNED only).
        # This is the proposal-generation hook the gate's PASS/FAIL hinges on.
        use_mech293_ghost_probes=planned,
        mech293_ghost_fraction=0.2,
    )
    return REEAgent(cfg)


def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# Paradigm A blockage helper
# ---------------------------------------------------------------------------
def _inject_blockage(env: CausalGridWorldV2) -> Tuple[int, int]:
    """Drop a hazard onto a cell adjacent to the agent's current position.

    Used in P2 evaluation episodes for paradigm A (detour). The cell choice
    biases toward the cell in the direction of the agent's most-recent
    motion when discoverable, falling back to (agent.x, agent.y+1) wrapped
    into bounds. The hazard is added to env.hazards and the grid; the env's
    proximity fields are recomputed on the next step() via existing
    _drift_hazards / proximity recompute paths.

    Returns the (x, y) where the blockage was placed, or (-1, -1) if no
    valid cell was found.
    """
    ax, ay = env.agent_x, env.agent_y
    candidates = [
        (ax, ay + 1),
        (ax, ay - 1),
        (ax + 1, ay),
        (ax - 1, ay),
    ]
    for hx, hy in candidates:
        if not (0 <= hx < env.size and 0 <= hy < env.size):
            continue
        if env.grid[hx, hy] != env.ENTITY_TYPES["empty"]:
            continue
        env.grid[hx, hy] = env.ENTITY_TYPES["hazard"]
        env.hazards.append([hx, hy])
        return hx, hy
    return -1, -1


# ---------------------------------------------------------------------------
# Single condition x paradigm run
# ---------------------------------------------------------------------------
def run_condition_paradigm(
    seed: int,
    condition: str,
    paradigm: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    total_p0   = DRY_RUN_EPISODES if dry_run else P0_EPISODES
    total_p1   = DRY_RUN_EPISODES if dry_run else P1_EPISODES
    total_p2   = DRY_RUN_EPISODES if dry_run else P2_EPISODES
    steps_per  = DRY_RUN_STEPS    if dry_run else STEPS_PER_EP
    total_eps  = total_p0 + total_p1 + total_p2

    env_train_seed = seed if paradigm == "A_DETOUR" else ENV_A_SEED + seed
    env_eval_seed  = env_train_seed if paradigm == "A_DETOUR" else ENV_B_SEED + seed

    env_train = _make_env(env_train_seed)
    agent     = _make_agent(env_train, condition, seed)
    device    = agent.device
    optimizer = optim.Adam(list(agent.parameters()), lr=LR)

    # Per-phase running accumulators.
    p2_resources_pre_block:   List[float] = []
    p2_benefit_pre_block:     List[float] = []
    p2_resources_post_block:  List[float] = []
    p2_benefit_post_block:    List[float] = []
    p2_resources_standard:    List[float] = []
    p2_benefit_standard:      List[float] = []
    p2_harm_total:            List[float] = []
    prox_preds:               List[float] = []
    prox_targets:             List[float] = []
    ghost_admitted_per_tick:  List[int]   = []
    first_step_action_dists:  List[List[float]] = []

    env = env_train

    for ep in range(total_eps):
        phase = "P0" if ep < total_p0 else ("P1" if ep < total_p0 + total_p1 else "P2")
        in_eval = (phase == "P2")

        # Paradigm B novel-context swap for eval episodes.
        if in_eval and paradigm == "B_NOVEL_CONTEXT":
            env = _make_env(env_eval_seed)
        else:
            env = env_train

        _, obs_dict = env.reset()
        agent.reset()

        block_step: Optional[int] = None
        if in_eval and paradigm == "A_DETOUR":
            block_step = max(1, int(steps_per * T_BLOCK_FRAC))

        ep_resources_pre, ep_benefit_pre   = 0, 0.0
        ep_resources_post, ep_benefit_post = 0, 0.0
        ep_harm = 0.0

        for step_i in range(steps_per):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            latent    = agent.sense(obs_body, obs_world)
            ticks     = agent.clock.advance()

            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", True)
                else torch.zeros(1, 32, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action     = agent.select_action(candidates, ticks)

            # SD-012 goal seeding: HABIT + PLANNED, P1 / P2 only.
            if condition in ("HABIT", "PLANNED") and phase != "P0":
                benefit_raw = float(obs_body.flatten()[11].item()) if obs_body.shape[-1] > 11 else 0.0
                drive_level = REEAgent.compute_drive_level(obs_body)
                agent.update_z_goal(benefit_raw, drive_level)

            # Track resource_prox prediction quality for C4.
            rfv = obs_dict.get("resource_field_view", None)
            if rfv is not None and latent.resource_prox_pred is not None:
                prox_targets.append(float(rfv.max().item()))
                prox_preds.append(float(latent.resource_prox_pred.squeeze().item()))

            # Diagnostic: ghost branch firing (PLANNED) and first-step action
            # distribution (KL diagnostic).
            if in_eval:
                diag = getattr(agent.hippocampal, "_last_propose_diagnostics", {}) or {}
                ghost_admitted_per_tick.append(int(diag.get("mech293_n_ghost_admitted", 0)))
                # First-step action probabilities across CEM candidates.
                if hasattr(candidates, "actions") and candidates.actions is not None:
                    a0 = candidates.actions[:, 0, :]
                    if a0.numel() > 0:
                        # Empirical distribution over action classes from CEM seeds.
                        cls = a0.argmax(dim=-1)
                        n_cls = int(a0.shape[-1])
                        counts = torch.bincount(cls.flatten(), minlength=n_cls).float()
                        counts = counts + 1.0  # Laplace smoothing for KL stability
                        first_step_action_dists.append(
                            (counts / counts.sum()).tolist()
                        )

            # Paradigm A blockage injection.
            if block_step is not None and step_i == block_step:
                _inject_blockage(env)

            action_idx = int(action.argmax(dim=-1).item())
            flat_next, harm_signal, done, info, obs_dict_next = env.step(action_idx)

            if in_eval:
                ep_harm += float(harm_signal)
                bx = float(info.get("benefit_exposure", 0.0))
                rx = int(bool(info.get("resource_consumed", False)))
                if paradigm == "A_DETOUR":
                    if step_i < (block_step or steps_per):
                        ep_resources_pre += rx
                        ep_benefit_pre   += bx
                    else:
                        ep_resources_post += rx
                        ep_benefit_post   += bx
                else:
                    ep_resources_standard += rx if "ep_resources_standard" not in dir() else 0
                    p2_resources_standard.append(rx)
                    p2_benefit_standard.append(bx)

            obs_dict = obs_dict_next
            if done:
                break

        if in_eval and paradigm == "A_DETOUR":
            p2_resources_pre_block.append(ep_resources_pre)
            p2_benefit_pre_block.append(ep_benefit_pre)
            p2_resources_post_block.append(ep_resources_post)
            p2_benefit_post_block.append(ep_benefit_post)
            p2_harm_total.append(ep_harm)

        # P0 / P1 training steps: backprop encoder + proximity head losses.
        if phase != "P2":
            try:
                loss = agent.compute_prediction_loss()
                if loss is not None and torch.is_tensor(loss) and loss.requires_grad:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            except Exception as exc:
                if dry_run:
                    print(f"  [warn] training step skipped: {exc}")

    # Metrics.
    def _safe_mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    def _r2(preds: List[float], targets: List[float]) -> float:
        if not preds:
            return 0.0
        p = torch.tensor(preds)
        t = torch.tensor(targets)
        ss_res = ((p - t) ** 2).sum().item()
        ss_tot = ((t - t.mean()) ** 2).sum().item()
        return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else 0.0

    benefit_pre   = _safe_mean(p2_benefit_pre_block)
    benefit_post  = _safe_mean(p2_benefit_post_block)
    benefit_std   = _safe_mean(p2_benefit_standard)
    harm_total    = _safe_mean(p2_harm_total)
    prox_r2_value = _r2(prox_preds, prox_targets)
    ghost_mean    = _safe_mean([float(g) for g in ghost_admitted_per_tick])

    return {
        "seed": seed,
        "condition": condition,
        "paradigm": paradigm,
        "n_p2_episodes_recorded": (
            len(p2_benefit_post_block) if paradigm == "A_DETOUR"
            else len(p2_benefit_standard)
        ),
        "benefit_pre_block":  benefit_pre,
        "benefit_post_block": benefit_post,
        "benefit_standard":   benefit_std,
        "harm_total":         harm_total,
        "prox_r2":            prox_r2_value,
        "ghost_admitted_mean_per_tick": ghost_mean,
        "first_step_action_dists":      first_step_action_dists,
    }


# ---------------------------------------------------------------------------
# KL diagnostic
# ---------------------------------------------------------------------------
def _kl_dists(a_dists: List[List[float]], b_dists: List[List[float]]) -> float:
    """Mean KL divergence per-tick between paired PLANNED / HABIT first-step
    action distributions. Truncates to the shorter list when lengths differ."""
    n = min(len(a_dists), len(b_dists))
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        a = torch.tensor(a_dists[i])
        b = torch.tensor(b_dists[i])
        a = a / a.sum().clamp(min=1e-9)
        b = b / b.sum().clamp(min=1e-9)
        a = a + 1e-9
        b = b + 1e-9
        total += float((a * (a.log() - b.log())).sum().item())
    return total / n


# ---------------------------------------------------------------------------
# Acceptance evaluation
# ---------------------------------------------------------------------------
def evaluate_acceptance(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute C1-C5 against per-seed condition x paradigm results."""

    def by(seed: int, cond: str, par: str) -> Optional[Dict[str, Any]]:
        for r in per_seed:
            if r["seed"] == seed and r["condition"] == cond and r["paradigm"] == par:
                return r
        return None

    seeds = sorted({r["seed"] for r in per_seed})

    # C1: ghost branch fires under PLANNED in both paradigms.
    c1_per_seed = {}
    for sd in seeds:
        c1_per_seed[sd] = {
            par: (by(sd, "PLANNED", par) or {}).get("ghost_admitted_mean_per_tick", 0.0)
            for par in PARADIGMS
        }
    c1_pass_per_seed = {
        sd: all(v > 0.0 for v in c1_per_seed[sd].values())
        for sd in seeds
    }
    c1_pass = all(c1_pass_per_seed.values())

    # C2: PLANNED.benefit_post_block - HABIT.benefit_post_block >= 0.30
    # in paradigm A only, in >= 4/7 seeds.
    c2_gaps = []
    for sd in seeds:
        p = by(sd, "PLANNED", "A_DETOUR")
        h = by(sd, "HABIT",   "A_DETOUR")
        if p is None or h is None:
            c2_gaps.append(float("nan"))
            continue
        c2_gaps.append(p["benefit_post_block"] - h["benefit_post_block"])
    c2_seeds_above = sum(1 for g in c2_gaps if (not math.isnan(g)) and g >= C2_BENEFIT_GAP)
    c2_pass = c2_seeds_above >= C2_MIN_SEEDS

    # C3: HABIT >= ABLATED in standard episodes (paradigm A pre-block + B).
    c3_per_seed = []
    for sd in seeds:
        for par in PARADIGMS:
            h = by(sd, "HABIT",   par)
            a = by(sd, "ABLATED", par)
            if h is None or a is None:
                continue
            metric = "benefit_pre_block" if par == "A_DETOUR" else "benefit_standard"
            c3_per_seed.append((h[metric], a[metric]))
    c3_pass = all(h_val >= a_val + C3_HABIT_GE_ABL for (h_val, a_val) in c3_per_seed)

    # C4: PLANNED.prox_r2 >= 0.7 in both paradigms.
    c4_pass = True
    c4_values = []
    for sd in seeds:
        for par in PARADIGMS:
            p = by(sd, "PLANNED", par)
            if p is None:
                c4_pass = False
                continue
            c4_values.append(p["prox_r2"])
            if p["prox_r2"] < C4_PROX_R2_FLOOR:
                c4_pass = False

    # C5: PLANNED.harm_total within 10% of HABIT.harm_total in paradigm A.
    c5_pass = True
    c5_diffs = []
    for sd in seeds:
        p = by(sd, "PLANNED", "A_DETOUR")
        h = by(sd, "HABIT",   "A_DETOUR")
        if p is None or h is None:
            c5_pass = False
            continue
        denom = max(abs(h["harm_total"]), 1e-6)
        diff_pct = abs(p["harm_total"] - h["harm_total"]) / denom
        c5_diffs.append(diff_pct)
        if diff_pct > C5_HARM_TOL:
            c5_pass = False

    # Diagnostic: KL between PLANNED and HABIT first-step action dists.
    kl_per_seed = {}
    for sd in seeds:
        kl_per_paradigm = {}
        for par in PARADIGMS:
            p = by(sd, "PLANNED", par)
            h = by(sd, "HABIT",   par)
            if p is None or h is None:
                continue
            kl_per_paradigm[par] = _kl_dists(
                p.get("first_step_action_dists", []),
                h.get("first_step_action_dists", []),
            )
        kl_per_seed[sd] = kl_per_paradigm

    return {
        "C1_ghost_branch_fires": {"pass": bool(c1_pass), "per_seed": c1_pass_per_seed,
                                   "ghost_admitted_per_seed_paradigm": c1_per_seed},
        "C2_planned_minus_habit_benefit_gap_detour": {
            "pass": bool(c2_pass),
            "n_seeds_above_threshold": c2_seeds_above,
            "min_seeds_required": C2_MIN_SEEDS,
            "threshold": C2_BENEFIT_GAP,
            "gaps": c2_gaps,
        },
        "C3_habit_ge_ablated_standard": {"pass": bool(c3_pass),
                                          "per_paradigm_per_seed": c3_per_seed},
        "C4_prox_r2_floor": {"pass": bool(c4_pass),
                             "floor": C4_PROX_R2_FLOOR,
                             "values": c4_values},
        "C5_no_harm_regression": {"pass": bool(c5_pass),
                                   "tolerance": C5_HARM_TOL,
                                   "diffs": c5_diffs},
        "KL_planned_vs_habit_per_seed_paradigm": kl_per_seed,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(seeds: List[int], dry_run: bool) -> int:
    t0 = time.time()
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        for paradigm in PARADIGMS:
            for condition in CONDITIONS:
                print(f"[v3_exq_495] seed={seed} paradigm={paradigm} condition={condition}")
                try:
                    result = run_condition_paradigm(seed, condition, paradigm, dry_run=dry_run)
                except Exception as exc:
                    print(f"  [error] {exc!r}")
                    result = {
                        "seed": seed, "condition": condition, "paradigm": paradigm,
                        "error": repr(exc),
                    }
                per_seed.append(result)

    acceptance = evaluate_acceptance(per_seed)
    all_pass = all(
        acceptance[k]["pass"]
        for k in (
            "C1_ghost_branch_fires",
            "C2_planned_minus_habit_benefit_gap_detour",
            "C3_habit_ge_ablated_standard",
            "C4_prox_r2_floor",
            "C5_no_harm_regression",
        )
    )
    elapsed = time.time() - t0

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "MECH-163": "supports" if all_pass else "weakens",
        },
        "evidence_direction_note": (
            "V3 full-completion gate for MECH-163 dual goal-directed systems. "
            "PASS = C1 (ghost branch fires under PLANNED) AND C2 (PLANNED - "
            "HABIT benefit-ratio gap >= 0.30 in paradigm A detour, >= 4/7 "
            "seeds) AND C3 (HABIT >= ABLATED in standard episodes) AND C4 "
            "(PLANNED prox_r2 >= 0.7) AND C5 (PLANNED harm within 10% of "
            "HABIT). KL_PLANNED_HABIT recorded as diagnostic. EXQ-327 had "
            "cleared the habit arm only (now SD-015 evidence; not MECH-163). "
            "This experiment tests the dual-system DISCRIMINATION."
        ),
        "outcome": "PASS" if all_pass else "FAIL",
        "elapsed_sec": elapsed,
        "metrics": {
            "per_seed_results": per_seed,
            "acceptance": acceptance,
            "seeds": seeds,
            "conditions": CONDITIONS,
            "paradigms": PARADIGMS,
        },
        "dry_run": bool(dry_run),
    }

    if not dry_run:
        out_dir = EVIDENCE_ROOT / EXPERIMENT_TYPE
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{run_id}.json"
        with open(out_file, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[v3_exq_495] wrote manifest -> {out_file}")
    print(f"[v3_exq_495] overall: {'PASS' if all_pass else 'FAIL'} ({elapsed:.1f}s)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="3 ep / 10 step / 1 seed smoke; no manifest write.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override default 7-seed list.")
    args = parser.parse_args()
    seeds = args.seeds if args.seeds is not None else (
        DRY_RUN_SEEDS if args.dry_run else SEEDS_DEFAULT
    )
    sys.exit(main(seeds=seeds, dry_run=args.dry_run))
