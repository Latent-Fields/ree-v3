#!/opt/local/bin/python3
"""
V3-EXQ-228b -- ARC-032: Theta-Rate Pathway Behavioral Test, re-run on the
scaffolded_sd054_onboarding curriculum (substrate-corrected continuation of
the V3-EXQ-228/228a lineage).

Claims: ARC-032
EXPERIMENT_PURPOSE = "evidence"
Supersedes: V3-EXQ-228a

=== WHY THIS RE-RUN (228/228a -> 228b) ===

V3-EXQ-228 (2026-04-04) measured E1 prediction error, which turned out to have
zero dependency on theta_buffer -- a metric-ablation mismatch, not a real test.
V3-EXQ-228a (2026-04-05) fixed that by switching to behavioral outcome metrics
(resource_rate, harm_rate) measured via the full agent pipeline, with a
THETA_ACTIVE vs THETA_ZEROED (theta_buffer.summary() patched to always return
zeros) between-agents design. 228a's own non-degeneracy precondition --
z_goal_norm >= 0.05 after training -- FAILED on the old causal_grid_world
substrate + simplified mixed-greedy/random warmup (goal_norm_active ~0.031
averaged across seeds 42/7/13; V3-EXQ-247, the other named gate experiment,
also failed the same precondition on 2026-04-07 with goal_norm=0.0). Outcome
non_contributory/substrate_limitation both times: the theta pathway was never
actually put to the test, because there was no goal-context signal in the
buffer to test in the first place.

Since 2026-06-10 (V3-EXQ-603n PASS), `experiments/scaffolded_sd054_onboarding.py`
(Stage0 forced-feed -> Stage0b consolidation -> P0 goal-frozen encoder warmup ->
Stage-H isolated hazard avoidance -> P1 combined anneal -> P2 frozen-policy
measurement) reliably clears z_goal_norm >> 0.05 in unrelated experiments
(V3-EXQ-632 seed-42 z_goal_norm 3.01 at contact; V3-EXQ-866a zgoal_norm_mean_FULL
0.12-0.94 across seeds/runs) using the same FULL agent config this script
reuses verbatim (see build_agent_config below). Neither of those experiments
carries ARC-032 in claim_ids -- this is the first ARC-032-tagged run on that
substrate. Per ARC-032's claims.yaml what_would_answer (2026-08-08): "a genuine
theta-bypass ablation condition ... must be run WITH the precondition met."

=== DESIGN CHANGE FROM 228a: SHARED TRAINING, SPLIT AT MEASUREMENT ===

228a's warmup deliberately never called generate_trajectories(), so the theta
patch (applied before warmup) was provably inert during training and only
became operative at eval -- the single-variable property came from "both
agents trained identically because the patch does nothing yet."

scaffolded_sd054_onboarding's training loop (_train_episode, used by every
phase: Stage0/Stage0b/P0/Stage-H/P1) DOES call generate_trajectories() +
select_action() on every step -- that assumption does not carry over. Patching
theta from the start of training on THIS curriculum would make THETA_ZEROED
learn its entire policy under a zeroed theta channel throughout Stage0..P1,
confounding "does theta matter for a trained policy's trajectory scoring"
(what ARC-032 asks) with "can the agent learn at all without theta" (a
different, stronger question this script does not attempt to answer).

So: ONE agent is trained per seed through the full curriculum (Stage0 ->
Stage0b -> P0 -> Stage-H -> P1), using scaffolded_sd054_onboarding's own FULL
agent config verbatim (866a's build_agent_config(..., "FULL"), the empirically
validated config that clears the z_goal precondition). Immediately after P1
completes -- before any measurement -- the trained agent is deep-copied into
two independent clones (copy.deepcopy(agent), the same clone-and-fork pattern
used in V3-EXQ-838/817a for a post-P0 branch point). One clone's
theta_buffer.summary() is monkey-patched to always return zeros (THETA_ZEROED);
the other is left untouched (THETA_ACTIVE). Both clones therefore start
measurement with byte-identical weights, byte-identical goal_state (z_goal is
not reset by agent.reset() -- see ree_core/agent.py Agent.reset(), "Does NOT
reset residue" -- so the deep copy captures the trained z_goal too), and
byte-identical theta_buffer content (irrelevant either way: agent.reset()
clears theta_buffer at every episode boundary within measurement, and
THETA_ZEROED's patch makes buffer content moot regardless). The ONLY
difference between the two measurement runs is whether theta_buffer.summary()
returns its real content or zeros. This is a cleaner single-variable split
than either 228a's or 866a's own designs (no separate-training-run noise
between arms at all), and it is more sample-efficient (one training pass per
seed instead of two).

RNG is reset (reset_all_rng) once at the top of each seed's training, and
again immediately before EACH condition's P2 measurement (same seed both
times) -- so THETA_ACTIVE and THETA_ZEROED see the same environment resets
and the same stochastic action-sampling draws during measurement, isolating
the theta patch as the only source of divergence in the measurement rollout
itself, not just in the agent weights.

Per the arm-fingerprint contract's own escape hatch ("If arms share mutable
state across cells ... pass extra_ineligible_reasons"): both cells here share
the SAME trained agent object via deepcopy, so both are stamped
reuse_eligible=False with that reason. This is a one-off ARC-032 comparison,
not a lineage expected to need baseline reuse.

=== MEASUREMENT PROTOCOL (866a's P2 frozen-policy protocol, verbatim) ===

Frozen-policy (agent.eval(), no gradients) rollout on a P2-phase env, action
selection at temperature=0.5, update_z_goal() called every step so the goal
stream stays live during measurement. scaffold_p2_hazard_food_attraction_guard
is set to 0.3 (P2_HFA_GUARD), NOT the class default -1.0/"no guard" -- 866a's
own empirical check found the class default produces near-zero
resource_visit_rate for EVERY condition including RANDOM (the hard P2 env
suppresses contact for any policy, not just a bad one). resource_visit_rate
counts transition_type in {"benefit_approach", "resource"} (866a's more
complete categorization vs 228a's "resource"-only check).

Precondition: after P1 (before the deepcopy split), the single trained agent's
goal_norm (agent.compute_goal_maintenance_diagnostic()['goal_norm']) must
clear GOAL_NORM_THRESH (0.05) -- 228a's own threshold, and the exact quantity
228a's precondition failed on. Since both measurement clones share this
post-P1 state, one precondition check per seed covers both conditions.

=== PRE-REGISTERED CRITERIA (unchanged from 228a) ===

C1 (main test): resource_visit_rate_active - resource_visit_rate_zeroed >=
    LIFT_THRESH (0.05) in >= 2/3 seeds. (theta packaging improves resource
    collection -- ARC-032's main behavioral prediction.)
C2 (informational): harm_rate_active <= harm_rate_zeroed * HARM_RATIO_MAX
    (1.5) in >= 2/3 seeds. (theta does not cause harmful trajectory bias.)

PASS = precondition met AND C1 AND C2, majority >= 2/3 seeds each.

Evidence interpretation:
  precondition FAIL           -> non_contributory (substrate_limitation) --
    same failure mode as 228/228a, would mean even this curriculum cannot
    seed a testable goal-context signal.
  precondition met, C1 FAIL   -> does_not_support (theta has no measurable
    behavioral effect on a trained policy's trajectory scoring)
  precondition met, C1 PASS, C2 FAIL -> weakens (theta pathway introduces
    harmful trajectory bias)
  precondition met, C1 PASS, C2 PASS -> supports (ARC-032: theta pathway
    provides functionally meaningful E3 trajectory-scoring context)

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler, same as every other importer).
"""

import argparse
import copy
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.arm_fingerprint import reset_all_rng, compute_arm_fingerprint
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator

from experiments.scaffolded_sd054_onboarding import (
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    _benefit_and_drive,
    _sense_with_optional_harm,
)

# --------------------------------------------------------------------- #
# Experiment metadata
# --------------------------------------------------------------------- #
EXPERIMENT_TYPE = "v3_exq_228b_arc032_theta_bypass_onboarded"
QUEUE_ID = "V3-EXQ-228b"
SUPERSEDES = "V3-EXQ-228a"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["ARC-032"]
EXPERIMENT_PURPOSE = "evidence"

# Seeds: NOT 44 -- confirmed recurring per-seed instability (early episode
# death ~step 40) on reef-config envs (EXQ-539-540, V3-EXQ-538a autopsies).
# 45 substituted per that finding.
SEEDS = [42, 43, 45]
CONDITIONS = ["THETA_ACTIVE", "THETA_ZEROED"]

# --- Curriculum budgets (866a's validated Stage0/0b/P0/Stage-H/P1 budgets,
#     reused verbatim -- this is the exact recipe proven to clear z_goal_norm
#     >= 0.05 on this substrate) ---
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 30
STEPS_PER_EP = 200
P0_NUM_HAZARDS = 1
TOTAL_TRAIN_EPS = (
    STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET + P1_BUDGET
)  # 220 -- shared training, once per seed
TOTAL_EPS_PER_CONDITION = TOTAL_TRAIN_EPS + P2_BUDGET  # 250 -- progress denominator M

# Dry-run (smoke) budgets.
DRY_STAGE0, DRY_STAGE0B, DRY_P0, DRY_HAZARD, DRY_P1, DRY_P2, DRY_STEPS = 2, 2, 5, 5, 5, 8, 30
DRY_TOTAL_TRAIN_EPS = DRY_STAGE0 + DRY_STAGE0B + DRY_P0 + DRY_HAZARD + DRY_P1
DRY_TOTAL_EPS_PER_CONDITION = DRY_TOTAL_TRAIN_EPS + DRY_P2

# --- Stage-H regime (866a's validated anchor, same as 603q) ----------------
HAZARD_STAGE_NUM_HAZARDS = 6
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.10
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

# --- Seeding calibration + cue-recall bridge (mirror 866a / 603q) ----------
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

# --- SD-058 / MECH-357 protective-scaffold anneal ---------------------------
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2

# --- Harm-pathway training (603k) + stabilization (603q amend) --------------
HARM_PATHWAY_LR = 1e-3
HARM_PATHWAY_ENCODER_LR = 3e-4
HARM_PATHWAY_WARMUP_STEPS = 250

# --- P2 measurement guard -- see module docstring "MEASUREMENT PROTOCOL". --
P2_HFA_GUARD = 0.3

# --- Encoder / latent dims (mirror 866a) ------------------------------------
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0
ALPHA_WORLD = 0.9
SELF_DIM = 32

# --- Pre-registered thresholds (228a's own thresholds, unchanged) ----------
GOAL_NORM_THRESH = 0.05   # Precondition: z_goal must be genuinely seeded
LIFT_THRESH = 0.05        # C1: resource_visit_rate_active - _zeroed
HARM_RATIO_MAX = 1.5      # C2: harm_rate_active <= harm_rate_zeroed * this
MIN_SEEDS_PASS = 2        # of 3 -- ">= 2/3 seeds"


# --------------------------------------------------------------------- #
# Scaffold + agent config builders
# --------------------------------------------------------------------- #

def build_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            DRY_STAGE0, DRY_STAGE0B, DRY_P0, DRY_HAZARD, DRY_P1, DRY_P2, DRY_STEPS)
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, P2_BUDGET, STEPS_PER_EP)

    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_strict_goal_isolation=False,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=p2,
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=0.75,
        scaffold_contact_gated_goal_updates=True,
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=False,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        scaffold_feed_harm_stream=True,
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
        scaffold_harm_pathway_encoder_lr=HARM_PATHWAY_ENCODER_LR,
        scaffold_harm_pathway_warmup_steps=HARM_PATHWAY_WARMUP_STEPS,
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
    )
    if steps < 75:
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def build_agent_config(env) -> REEConfig:
    """REEConfig for the shared trained agent -- 866a's FULL-arm config
    verbatim (the empirically validated config that clears the z_goal
    precondition on this curriculum). ARC-032 needs the goal/wanting channel
    live; there is no AVOIDANCE_ONLY arm here (this experiment's manipulation
    is entirely post-training, in the theta patch -- not in REEConfig)."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_e2_harm_s_forward=True,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        use_escape_affordance_bridge=False,
        use_escape_relief_credit=False,
        use_escape_safety_credit=False,
        use_contextual_safety_terrain=True,
        use_conditioned_safety_store=True,
        use_suffering_derivative_comparator=True,
        e2_action_contrastive_enabled=True,
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        e1_goal_conditioned=True,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
    )
    cfg.latent.use_resource_encoder = True
    return cfg


# --------------------------------------------------------------------- #
# Theta ablation patch (228a's patch, applied post-training instead of
# pre-training -- see module docstring "DESIGN CHANGE FROM 228a")
# --------------------------------------------------------------------- #

def _patch_theta_zeroed(agent: REEAgent) -> None:
    """Patch theta_buffer.summary() to always return zeros, severing the
    E1 -> theta_buffer -> E3 pathway for trajectory proposals."""
    device = agent.device
    world_dim = agent.config.latent.world_dim

    def _zeroed_summary():
        return torch.zeros(1, world_dim, device=device)

    agent.theta_buffer.summary = _zeroed_summary


# --------------------------------------------------------------------- #
# P2 frozen-policy measurement (866a's protocol, verbatim)
# --------------------------------------------------------------------- #

def _measure_p2(
    agent: REEAgent,
    env,
    device: torch.device,
    cfg: ScaffoldedSD054OnboardingConfig,
    n_episodes: int,
    steps_per_ep: int,
) -> Dict[str, Any]:
    agent.eval()
    action_dim = env.action_dim

    resource_visits = 0
    harm_events = 0
    total_steps = 0
    zgoal_norms: List[float] = []

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _step in range(steps_per_ep):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            with torch.no_grad():
                latent = _sense_with_optional_harm(
                    agent, obs_body, obs_world, obs_dict, device,
                    cfg.scaffold_feed_harm_stream,
                )
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks.get("e1_tick", True)
                    else torch.zeros(1, agent.config.latent.world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks, temperature=0.5)
            action_idx = int(action.argmax(dim=-1).item())

            benefit, drive = _benefit_and_drive(obs_dict["body_state"].to(device))
            agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)
            if agent.goal_state is not None:
                zgoal_norms.append(float(agent.goal_state.z_goal.norm().item()))

            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            ttype = info.get("transition_type", "none")

            if ttype in ("benefit_approach", "resource"):
                resource_visits += 1
            if ttype in ("agent_caused_hazard", "hazard_approach"):
                harm_events += 1
            total_steps += 1
            if done:
                break

        if (ep + 1) % 10 == 0 or ep == n_episodes - 1:
            print(f"  [p2] ep {ep + 1}/{n_episodes}", flush=True)

    resource_visit_rate = resource_visits / max(1, total_steps)
    harm_rate = harm_events / max(1, total_steps)
    zgoal_norm_mean = sum(zgoal_norms) / len(zgoal_norms) if zgoal_norms else 0.0

    return {
        "resource_visit_rate": resource_visit_rate,
        "harm_rate": harm_rate,
        "zgoal_norm_mean": zgoal_norm_mean,
        "n_resource_events": resource_visits,
        "n_harm_events": harm_events,
        "total_steps": total_steps,
    }


# --------------------------------------------------------------------- #
# Per-seed run: shared training, deep-copy split, dual measurement
# --------------------------------------------------------------------- #

def _cell_config_slice(condition: str, seed: int, dry_run: bool) -> Dict:
    return {
        "condition": condition,
        "seed": seed,
        "dry_run": dry_run,
        "shared_training": True,
        "stage0_budget": STAGE0_BUDGET,
        "stage0b_budget": STAGE0B_BUDGET,
        "p0_budget": P0_BUDGET,
        "hazard_budget": HAZARD_STAGE_BUDGET,
        "p1_budget": P1_BUDGET,
        "p2_budget": P2_BUDGET,
        "steps_per_ep": STEPS_PER_EP,
        "goal_norm_thresh": GOAL_NORM_THRESH,
        "lift_thresh": LIFT_THRESH,
        "harm_ratio_max": HARM_RATIO_MAX,
        "p2_hfa_guard": P2_HFA_GUARD,
    }


def _run_seed(seed: int, zg: ZGoalStreamAccumulator, dry_run: bool) -> Dict[str, Any]:
    device = torch.device("cpu")
    sched_cfg = build_scaffold_cfg(dry_run)

    total_train = DRY_TOTAL_TRAIN_EPS if dry_run else TOTAL_TRAIN_EPS
    total_per_cond = DRY_TOTAL_EPS_PER_CONDITION if dry_run else TOTAL_EPS_PER_CONDITION
    p2_eps = DRY_P2 if dry_run else P2_BUDGET
    steps_per = DRY_STEPS if dry_run else STEPS_PER_EP

    # --- Shared training: one agent, once per seed. RNG reset here covers
    # the whole training trajectory (Stage0 -> Stage0b -> P0 -> Stage-H -> P1).
    reset_all_rng(seed)
    scheduler = ScaffoldedSD054OnboardingScheduler(sched_cfg)
    probe_env = _build_env(sched_cfg, phase="p2", seed=None)
    torch.manual_seed(seed)
    agent_cfg = build_agent_config(probe_env)
    agent = REEAgent(agent_cfg).to(device)

    ep_so_far = 0

    def _progress(phase_name: str, n_this_phase: int):
        nonlocal ep_so_far
        ep_so_far += n_this_phase
        print(
            f"  [train] seed={seed} shared ep {ep_so_far}/{total_train} phase={phase_name}",
            flush=True,
        )

    s0 = scheduler.run_stage0_nursery(agent, device)
    _progress("stage0", s0.n_episodes if not s0.aborted else 0)

    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    _progress("stage0b", s0b.n_episodes if not s0b.aborted else 0)

    p0 = scheduler.run_p0(agent, device)
    _progress("p0", p0.n_episodes)

    hz = scheduler.run_hazard_avoidance(agent, device)
    _progress("stage_h", hz.n_episodes)

    p1 = scheduler.run_p1(agent, device)
    _progress("p1", p1.n_episodes)

    diag = agent.compute_goal_maintenance_diagnostic()
    goal_norm_post_p1 = float(diag["goal_norm"])
    print(f"  [train done] seed={seed} goal_norm_post_p1={goal_norm_post_p1:.4f}", flush=True)

    zg.observe(agent)

    # --- Flush cached graph-attached tensors before deepcopy. P1 trains with
    # grad enabled, so several agent-internal caches hold non-leaf tensors
    # that torch.Tensor.__deepcopy__ refuses (confirmed: RuntimeError "Only
    # Tensors created explicitly by the user ... support the deepcopy
    # protocol"). One no-grad tick overwrites them -- same fix as
    # V3-EXQ-838/817a's post-P0 deepcopy split. ---
    agent.eval()
    with torch.no_grad():
        _, _flush_obs = probe_env.reset()
        agent.reset()
        _flush_body = _flush_obs["body_state"].to(device)
        _flush_world = _flush_obs["world_state"].to(device)
        _flush_latent = _sense_with_optional_harm(
            agent, _flush_body, _flush_world, _flush_obs, device,
            sched_cfg.scaffold_feed_harm_stream,
        )
        _flush_ticks = agent.clock.advance()
        _flush_prior = (
            agent._e1_tick(_flush_latent)
            if _flush_ticks.get("e1_tick", True)
            else torch.zeros(1, agent.config.latent.world_dim, device=device)
        )
        _flush_cand = agent.generate_trajectories(_flush_latent, _flush_prior, _flush_ticks)
        _ = agent.select_action(_flush_cand, _flush_ticks, temperature=0.5)

    # hippocampal._rng defaults to the `random` MODULE, which copy.deepcopy
    # cannot pickle ("cannot pickle 'module' object") -- same fix as
    # V3-EXQ-838. Installs a picklable per-seed random.Random instance.
    agent.hippocampal.seed_replay_rng(seed)

    # --- Deep-copy split. Both clones start byte-identical (weights +
    # goal_state, per module docstring). ---
    agent_active = agent
    agent_zeroed = copy.deepcopy(agent)
    _patch_theta_zeroed(agent_zeroed)

    train_summary = {
        "stage0_z_goal_norm_peak": s0.z_goal_norm_peak,
        "stage0_aborted": s0.aborted,
        "stage0b_retention_ratio": s0b.retention_ratio,
        "stage0b_aborted": s0b.aborted,
        "p0_mean_episode_length": p0.mean_episode_length,
        "hazard_median_last_window": hz.median_last_window_episode_length,
        "hazard_survival_gate_passed": hz.survival_gate_passed,
        "p1_median_last_window": p1.median_last_window_episode_length,
        "p1_survival_gate_passed": p1.survival_gate_passed,
        "goal_norm_post_p1": goal_norm_post_p1,
    }

    rows: Dict[str, Dict[str, Any]] = {}
    for condition, agent_cell in [
        ("THETA_ACTIVE", agent_active), ("THETA_ZEROED", agent_zeroed),
    ]:
        print(f"Seed {seed} Condition {condition}", flush=True)

        # RNG reset again, same seed, immediately before EACH condition's
        # measurement -- so both conditions see the same env resets and
        # stochastic action-sampling draws, isolating the theta patch as
        # the only source of divergence in the rollout itself.
        reset_all_rng(seed)
        p2_env = _build_env(sched_cfg, phase="p2", seed=None)
        metrics = _measure_p2(agent_cell, p2_env, device, sched_cfg, p2_eps, steps_per)

        print(
            f"  [train] seed={seed} {condition} ep {total_per_cond}/{total_per_cond} phase=p2",
            flush=True,
        )
        print(
            f"  [eval] {condition} seed={seed}"
            f" resource_visit_rate={metrics['resource_visit_rate']:.4f}"
            f" harm_rate={metrics['harm_rate']:.4f}"
            f" zgoal_norm_mean={metrics['zgoal_norm_mean']:.4f}",
            flush=True,
        )
        print("verdict: PASS", flush=True)  # cell ran to completion; scientific verdict is aggregate-level

        row: Dict[str, Any] = {"seed": seed, "condition": condition}
        row.update(train_summary)
        row.update(metrics)
        row["goal_norm_active"] = goal_norm_post_p1  # shared precondition value, both rows
        row["arm_fingerprint"] = compute_arm_fingerprint(
            config_slice=_cell_config_slice(condition, seed, dry_run),
            seed=seed,
            script_path=Path(__file__),
            rng_fully_reset=True,
            # Both cells share the same trained agent object via deepcopy --
            # not independently retrainable in isolation from config+seed
            # alone without re-running this exact shared-training procedure.
            extra_ineligible_reasons=["shared_trained_agent_deepcopy_split_across_theta_conditions"],
        )
        rows[condition] = row

    return rows


# --------------------------------------------------------------------- #
# Aggregate + acceptance criteria
# --------------------------------------------------------------------- #

def _frac_seeds(flags: List[bool]) -> float:
    return sum(1 for f in flags if f) / max(1, len(flags))


def run(dry_run: bool = False) -> Tuple[Dict[str, Any], ZGoalStreamAccumulator]:
    print(
        f"\n[{QUEUE_ID}] ARC-032 Theta-Rate Pathway Behavioral Test"
        f" (scaffolded_sd054_onboarding re-run, supersedes {SUPERSEDES})"
        f" dry_run={dry_run}",
        flush=True,
    )

    zg = ZGoalStreamAccumulator()
    arm_results: List[Dict] = []
    per_seed: Dict[int, Dict[str, Dict]] = {}

    for seed in SEEDS:
        per_seed[seed] = _run_seed(seed, zg, dry_run=dry_run)
        arm_results.append(per_seed[seed]["THETA_ACTIVE"])
        arm_results.append(per_seed[seed]["THETA_ZEROED"])

    gn_flags, c1_flags, c2_flags = [], [], []
    lifts, harm_ratios = [], []
    for seed in SEEDS:
        active = per_seed[seed]["THETA_ACTIVE"]
        zeroed = per_seed[seed]["THETA_ZEROED"]

        lift = active["resource_visit_rate"] - zeroed["resource_visit_rate"]
        harm_ratio = active["harm_rate"] / max(1e-9, zeroed["harm_rate"])

        lifts.append(lift)
        harm_ratios.append(harm_ratio)
        gn_flags.append(active["goal_norm_active"] >= GOAL_NORM_THRESH)
        c1_flags.append(lift >= LIFT_THRESH)
        c2_flags.append(harm_ratio <= HARM_RATIO_MAX)

    threshold = MIN_SEEDS_PASS / len(SEEDS)
    gn_frac, c1_frac, c2_frac = _frac_seeds(gn_flags), _frac_seeds(c1_flags), _frac_seeds(c2_flags)
    precond_pass = gn_frac >= threshold
    c1_pass = c1_frac >= threshold
    c2_pass = c2_frac >= threshold

    non_degenerate = True
    degeneracy_reason = None

    if not precond_pass:
        status = "FAIL"
        evidence_direction = "non_contributory"
        decision = "substrate_limitation"
        non_degenerate = False
        degeneracy_reason = (
            f"Precondition (goal_norm >= {GOAL_NORM_THRESH}) FAILED on "
            f"{gn_frac:.2f} of seeds even on the scaffolded_sd054_onboarding "
            "curriculum with 866a's own validated FULL agent config -- same "
            "failure mode as V3-EXQ-228/228a/247, one level deeper: this "
            "curriculum cannot seed a testable goal-context signal for this "
            "agent config either."
        )
        interpretation = (
            "ARC-032 NON-CONTRIBUTORY: the theta-bypass ablation still could "
            "not be tested, because the non-degeneracy precondition "
            "(z_goal_norm >= 0.05 after training) failed even on the "
            "curriculum and agent config that reliably clears it in "
            "V3-EXQ-632/866a. The theta pathway itself was never exercised "
            "under a live goal-context signal."
        )
    elif not c1_pass:
        status = "FAIL"
        evidence_direction = "does_not_support"
        decision = "inconclusive"
        interpretation = (
            "ARC-032 DOES NOT SUPPORT: with the precondition met (z_goal_norm "
            f">= {GOAL_NORM_THRESH} on {gn_frac:.2f} of seeds), THETA_ACTIVE "
            "and THETA_ZEROED show no measurable resource-collection lift "
            f"(C1 passed on only {c1_frac:.2f} of seeds, need >= {threshold:.2f}). "
            "The theta channel does not appear to be a necessary pathway for "
            "goal-context to reach E3 trajectory scoring on this substrate -- "
            "either the theta-averaged signal is redundant with instantaneous "
            "z_world already available to E3 through other channels, or E3 "
            "scoring is dominated by non-theta factors."
        )
    elif not c2_pass:
        status = "FAIL"
        evidence_direction = "weakens"
        decision = "retire_ree_claim"
        interpretation = (
            "ARC-032 WEAKENED: C1 passed (theta packaging does improve "
            f"resource collection, {c1_frac:.2f} of seeds) but C2 (harm "
            f"parity) failed on {c2_frac:.2f} of seeds -- the theta pathway "
            "introduces a harmful trajectory-scoring bias alongside its "
            "goal-context benefit."
        )
    else:
        status = "PASS"
        evidence_direction = "supports"
        decision = "retain_ree"
        interpretation = (
            "ARC-032 SUPPORTED: with the precondition met "
            f"(goal_norm >= {GOAL_NORM_THRESH} on {gn_frac:.2f} of seeds), "
            f"THETA_ACTIVE shows a resource-collection lift over THETA_ZEROED "
            f"(C1, {c1_frac:.2f} of seeds) without a harmful trajectory-"
            f"scoring bias (C2, {c2_frac:.2f} of seeds). The theta-rate "
            "packaging of E1 output is a functionally meaningful pathway for "
            "goal-context to reach E3's trajectory scoring, on the "
            "scaffolded_sd054_onboarding curriculum."
        )

    metrics: Dict[str, float] = {
        "precond_goal_frac_seeds": gn_frac,
        "c1_lift_frac_seeds": c1_frac,
        "c2_harm_frac_seeds": c2_frac,
        "precond_goal_pass": 1.0 if precond_pass else 0.0,
        "c1_lift_pass": 1.0 if c1_pass else 0.0,
        "c2_harm_pass": 1.0 if c2_pass else 0.0,
        "lift_mean": sum(lifts) / len(lifts),
        "harm_ratio_mean": sum(harm_ratios) / len(harm_ratios),
        "goal_norm_active_mean": sum(
            per_seed[s]["THETA_ACTIVE"]["goal_norm_active"] for s in SEEDS
        ) / len(SEEDS),
    }
    for cond in CONDITIONS:
        rvr = [per_seed[s][cond]["resource_visit_rate"] for s in SEEDS]
        hr = [per_seed[s][cond]["harm_rate"] for s in SEEDS]
        zg_vals = [per_seed[s][cond]["zgoal_norm_mean"] for s in SEEDS]
        metrics[f"resource_visit_rate_mean_{cond}"] = sum(rvr) / len(rvr)
        metrics[f"harm_rate_mean_{cond}"] = sum(hr) / len(hr)
        metrics[f"zgoal_norm_mean_{cond}"] = sum(zg_vals) / len(zg_vals)

    summary_markdown = (
        f"# {QUEUE_ID} -- ARC-032 Theta-Rate Pathway Behavioral Test "
        f"(scaffolded_sd054_onboarding)\n\n"
        f"**Status:** {status}  **Evidence direction:** {evidence_direction}  "
        f"**Decision:** {decision}\n"
        f"**Supersedes:** {SUPERSEDES}\n"
        f"**Claims:** ARC-032\n\n"
        f"## Gates\n\n"
        f"| Gate | Frac seeds | Pass |\n|---|---|---|\n"
        f"| Precondition (goal_norm >= {GOAL_NORM_THRESH}) | {gn_frac:.2f} | {precond_pass} |\n"
        f"| C1 lift (>= {LIFT_THRESH}) | {c1_frac:.2f} | {c1_pass} |\n"
        f"| C2 harm parity (<= {HARM_RATIO_MAX}x) [info] | {c2_frac:.2f} | {c2_pass} |\n\n"
        f"## Interpretation\n\n{interpretation}\n"
    )

    result: Dict[str, Any] = {
        "status": status,
        "outcome": status,
        "decision": decision,
        "metrics": metrics,
        "arm_results": arm_results,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "per_seed_results": per_seed,
        "fatal_error_count": 0,
    }
    if not non_degenerate:
        result["non_degenerate"] = False
        result["degeneracy_reason"] = degeneracy_reason

    return result, zg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result, zg_accumulator = run(dry_run=args.dry_run)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = ARCHITECTURE_EPOCH

    full_config = {
        "seeds": SEEDS,
        "conditions": CONDITIONS,
        "stage0_budget": STAGE0_BUDGET,
        "stage0b_budget": STAGE0B_BUDGET,
        "p0_budget": P0_BUDGET,
        "hazard_budget": HAZARD_STAGE_BUDGET,
        "p1_budget": P1_BUDGET,
        "p2_budget": P2_BUDGET,
        "steps_per_ep": STEPS_PER_EP,
        "p0_num_hazards": P0_NUM_HAZARDS,
        "hazard_stage_regime": [HAZARD_STAGE_NUM_HAZARDS, HAZARD_STAGE_NUM_RESOURCES,
                                 HAZARD_STAGE_HFA, HAZARD_STAGE_PROXIMITY_HARM],
        "seeding": [SEED_GAIN, SEED_BENEFIT_THRESHOLD, SEED_DRIVE_FLOOR, N_RESOURCE_TYPES],
        "drive_weight": DRIVE_WEIGHT,
        "p2_hfa_guard": P2_HFA_GUARD,
        "goal_norm_thresh": GOAL_NORM_THRESH,
        "lift_thresh": LIFT_THRESH,
        "harm_ratio_max": HARM_RATIO_MAX,
        "design": "shared_training_deepcopy_split_at_p2",
    }

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=__file__,
        started_at=t0,
        z_goal_stream_stats=zg_accumulator.stats(),
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)

    emit_outcome(
        outcome=result["status"] if result["status"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
