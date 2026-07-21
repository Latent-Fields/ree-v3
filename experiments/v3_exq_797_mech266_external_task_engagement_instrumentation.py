"""
V3-EXQ-797 (DIAGNOSTIC): why is the SD-032a external_task mode never occupied?

Instrumentation spike. Runs the SAME scaffolded_sd054_onboarding curriculum build as
V3-EXQ-467d at the identical 603n config, then -- instead of sweeping the MECH-266
exit-rail ratio -- logs, per genuine SalienceCoordinator tick, the four quantities that
sit on the causal path between "the agent is foraging" and "external_task is occupied".
It changes NO substrate: every quantity is read back from state the agent already writes.

SLEEP DRIVER: N/A (waking goal-pipeline onboarding scheduler; no sleep loop).

WHY THIS SPIKE EXISTS
---------------------
Four runs measured fraction_in_external_task = 0.0 at EVERY seed -- 464b, 464c, 467d,
464d -- the last two WITH use_external_task_drive=True, the substrate built specifically
to fix it. 467d self-routed substrate_not_ready_requeue / external_task_mode_not_occupied
and hedged that the "scaffolded_sd054_onboarding nav-competence / Stage-H dependency may
legitimately keep the agent from foraging long enough". Session-side scoping (2026-07-21)
falsified BOTH standing hypotheses without spending a run:

  H1 (nav-competence / Stage-H ceiling) -- FALSIFIED by 467d's own manifest. Its contact
     guard passed 3/3 seeds (contact_non_vacuity_met=true, guard_fraction=1.0), with
     p2_contact_rate 0.330 / 0.167 / ... and z_goal_norm_at_contact_peak 0.505 / 0.438
     against a 0.4 gate. The agent WAS foraging and DID hold a goal representation.

  H2 (coordinator logit arithmetic pins external_task sub-argmax) -- FALSIFIED by direct
     execution of SalienceCoordinator under 467d's exact registration (affinity 3.0,
     salience 2.0, agent.py:1924-1931). external_task wins the argmax and fires the
     MECH-266 trigger once external_task_drive >= ~0.567. The coordinator is fine.

What is left is the INJECTION PATH (agent.py:5744-5771):

    engagement = goal_active ? clip(commit_w*[beta_gate.is_elevated]
                                    + prox_w*goal_proximity(z_world), 0, 1) : 0

and goal_proximity = 1/(1 + MSE_SUM(z_world, z_goal)) (goal.py:832) is summed over all
world_dim dimensions, so at the measured norms it lands ~0.16-0.71 and only reaches the
flip point in a near-attainment geometry. Engagement is therefore effectively a BINARY on
the beta latch -- and beta elevates on result.committed (agent.py:7273, :7467), i.e. on
sustained multi-step action commitment, the basal-ganglia layer still under construction.

That is a hypothesis about which term is zero, not a measurement. Nobody has ever logged
these quantities: they appear in NO manifest. This spike measures them.

E3 CADENCE (load-bearing measurement decision, not a detail)
-----------------------------------------------------------
REEAgent.select_action() returns early at agent.py:5458-5480 on a non-E3 tick, BEFORE the
salience block at :5744. So the SalienceCoordinator ticks only on E3 ticks -- ~1 env step
in 10 (heartbeat.e3_steps_per_tick defaults to 10, clock.py:52). Reading _input_signals
once per ENV step would therefore re-record one coordinator tick as ~10 independent
observations (the documented latch/pseudo-replication hazard). Every row here is gated on
ticks["e3_tick"], and n_latched_ticks is emitted so the true denominator is auditable
rather than inferred from len(rows). NOTE this also means 467d's dwell was counted in env
steps while the mode can only change on an E3 tick -- recorded as a secondary observation,
not adjudicated here.

WHAT IT MEASURES (per genuine coordinator tick, per seed)
---------------------------------------------------------
  engagement            the injected external_task_drive (read back from the coordinator)
  commit_fired          did the commit term fire at INJECTION time -- reconstructed as
                        engagement - prox_w*proximity, NOT a post-hoc is_elevated read.
                        The engagement block (agent.py:5744) runs BEFORE this same call's
                        elevate() sites (:7273, :7467), so the property read after
                        select_action reports the POST-elevation latch and is out of phase
                        with the arithmetic that produced engagement. Caught by the
                        authoring smoke, which showed beta_elevated=True on 3/3 seeds while
                        engagement tracked proximity exactly (commit term contributing 0).
                        Both pre- and post-call reads are still recorded, as corroboration.
  goal_proximity        recomputed from the same z_world the agent just encoded
  goal_active           agent.goal_state.is_active()
  dacc_pe               the internal_planning-side logit driver + salience aggregate term
  counterfactual        would external_task WIN the argmax at engagement = 1.0, holding
                        every other signal at its measured value? (evaluated on a scratch
                        SalienceCoordinator built from the live config -- reuses the real
                        arithmetic rather than duplicating it)

DISCRIMINATION GRID (pre-registered; routes, never a claim verdict)
------------------------------------------------------------------
  D1  commit_fired_frac < BETA_FLOOR
        -> commitment_layer_starved. The latch is the only term that can carry engagement
           over the flip point, and it never fires. Routes UPSTREAM to the BG /
           action-commitment cluster; MECH-266 is not testable until that lands.
  D2  commit_fired_frac >= BETA_FLOOR AND engagement_ge_flip_frac < ENGAGEMENT_FRAC_FLOOR
        -> injection_path_underpowered. The latch fires but engagement still never clears
           the flip point (proximity leg dead + latch too brief). A calibration/build
           question at the injection site.
  D3  engagement_ge_flip_frac >= ENGAGEMENT_FRAC_FLOOR AND external_task still unoccupied
        -> coordinator_swamped. Engagement DOES clear the flip point yet the mode is not
           entered, i.e. dacc_pe rides above the break-even. This is the ONLY branch that
           makes the coordinator itself the build target.
  D4  fraction_in_external_task > OCCUPANCY_FLOOR
        -> occupancy_reproduced_nonzero. The 4-run zero does not reproduce; MECH-266 is
           re-runnable as-is and this spike's premise is void.

A diagnostic's label is a HYPOTHESIS, not a verdict (feedback_diagnostic_self_route_is_
hypothesis). All four branches route; none is scored as claim evidence.

READINESS PRECONDITION (positive control)
-----------------------------------------
The engagement measurement is only interpretable on a foraging-competent agent -- on an
incompetent one "engagement never high" is uninformative. So the 467d contact guard is
re-used verbatim as the readiness gate (per-seed contact_rate > CONTACT_GATE AND
z_goal_norm_at_contact_peak > P2_ZGOAL_GATE), plus a sample-size floor on genuine E3
ticks. Below either floor -> substrate_not_ready_requeue, NEVER a substrate verdict.

DV SYMMETRY
-----------
Single-arm OBSERVATIONAL logging: there is no manipulation, so there is no manipulation
that could be invariant under a symmetry of the DV. The one derived quantity that IS a
contrast -- the engagement=1.0 counterfactual -- varies the exact signal whose affinity
weight is non-zero on external_task (a per-mode logit contribution, not a broadcast
constant across modes), so it is not annihilated by the argmax it is read through.

claim_ids: MECH-266, SD-032a (lineage anchors; experiment_purpose=diagnostic, so excluded
from governance confidence/conflict scoring -- this yields routing, not verdicts).
predecessor: V3-EXQ-467d (diagnostic successor, NOT a supersede).
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.cingulate.salience_coordinator import SalienceCoordinator  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _sd049_kwargs,
    _sense_with_optional_harm,
    stage_plan,
)
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_797_mech266_external_task_engagement_instrumentation"
QUEUE_ID = "V3-EXQ-797"
CLAIM_IDS: List[str] = ["MECH-266", "SD-032a"]
EXPERIMENT_PURPOSE = "diagnostic"
PREDECESSOR = "V3-EXQ-467d (diagnostic successor, NOT a supersede)"

# Single-cell observational run: no arm grid, so no per-cell arm_fingerprint applies.
ARM_FINGERPRINT_EXEMPT = (
    "single-arm observational instrumentation; no OFF/treatment grid and no arm_results"
)

# All three readiness anchors are reachable by construction AND demonstrated reachable on
# this exact substrate. (1) forager_contact_rate: CONTACT_GATE=0.0, so the predicate IS
# the definition of "made any contact at all" -- no narrower hand-written proxy sits
# between it and the state it anchors to. (2) z_goal_norm_at_contact and (3) the E3-tick
# floor: V3-EXQ-467d shipped predicate (2) verbatim on this identical curriculum + config
# and recorded contact_non_vacuity_met=true at guard_fraction=1.0 (3/3 seeds, measured
# 0.505 / 0.438 / ... against the same 0.4 gate), which is the frozen recorded control an
# assert_anchor_reachable call would replay; and (3) is bounded below by construction --
# 12 episodes x up to 200 steps at e3_steps_per_tick=10 admits up to ~240 genuine ticks
# against a floor of 100. None of the three is a narrower-than-the-state predicate, which
# is the defect class the check exists to catch.
ANCHOR_REACHABILITY_EXEMPT = (
    "predicates are the degeneracy definitions themselves (contact_rate > 0) or are "
    "demonstrated reachable on this identical substrate by V3-EXQ-467d's 3/3 contact "
    "guard; E3-tick floor is bounded reachable by construction"
)

SEEDS = [42, 43, 44]
CONDITION_LABEL = "CURRICULUM_BUILT_ENGAGEMENT_INSTRUMENTATION"
STICKY_MODE = "external_task"

# Single ratio: this spike does not sweep MECH-266's dose-response, it measures the
# injection path. r=1.00 is the symmetric-baseline rail from the 467d sweep.
EVAL_RATIO = 1.00

# --- pre-registered discrimination thresholds (constants, not run-derived) ---
# Engagement needed for external_task to win the argmax, measured by direct execution of
# the coordinator under this exact registration at a representative dacc_pe. The
# per-tick counterfactual below is the config-independent form; this constant is the
# reporting threshold for engagement_ge_flip_frac.
ENGAGEMENT_FLIP_POINT = 0.5667
BETA_FLOOR = 0.05                # D1/D2 boundary: fraction of E3 ticks with beta elevated
ENGAGEMENT_FRAC_FLOOR = 0.05     # D2/D3 boundary: fraction of E3 ticks at/over the flip
OCCUPANCY_FLOOR = 0.10           # D4: 467d's own occupancy floor, re-used verbatim
MIN_E3_TICKS = 100               # sample-size floor per seed on GENUINE coordinator ticks

# --- 467d build constants, copied verbatim so the substrate is the same one that failed ---
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15
MODE_EVAL_EPISODES = 12
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
P1_REEF_SPAWN_HOLD_FRACTION = 0.4

HAZARD_STAGE_NUM_HAZARDS = 4
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.1
HAZARD_STAGE_SPAWN_IN_REEF = True
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2
HARM_PATHWAY_LR = 1e-3
STAGE0B_RETENTION_GATE = 0.75

P2_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0

# 467d's external_task_drive registration, unchanged -- this spike measures the substrate
# as it failed, it does not re-tune it.
ET_AFFINITY_WEIGHT = 3.0
ET_SALIENCE_WEIGHT = 2.0
ET_COMMIT_WEIGHT = 1.0
ET_PROXIMITY_WEIGHT = 1.0


def _make_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = 2, 2, 5, 5, 5, 2, 30
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, P2_BUDGET, TRAIN_STEPS,
        )
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=p2,
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_p1_anneal_hold_fraction=P1_HOLD_FRACTION,
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=STAGE0B_RETENTION_GATE,
        scaffold_contact_gated_goal_updates=True,
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=HAZARD_STAGE_SPAWN_IN_REEF,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        scaffold_feed_harm_stream=True,
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
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
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        e2_action_contrastive_enabled=True,
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        use_dacc=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        use_closure_operator=False,
        use_external_task_drive=True,
        external_task_drive_affinity_weight=ET_AFFINITY_WEIGHT,
        external_task_drive_salience_weight=ET_SALIENCE_WEIGHT,
        external_task_drive_commit_weight=ET_COMMIT_WEIGHT,
        external_task_drive_proximity_weight=ET_PROXIMITY_WEIGHT,
    )
    cfg.latent.use_resource_encoder = True
    cfg.heartbeat.beta_gate_bistable = True
    return cfg


def _build_dual_cue_env(scaffold_cfg: ScaffoldedSD054OnboardingConfig) -> CausalGridWorldV2:
    """P2-config foraging env WITH the GAP-3 dual_cue primitive (mirrors 467d)."""
    p2_hfa = (
        scaffold_cfg.scaffold_p2_hazard_food_attraction_guard
        if scaffold_cfg.scaffold_p2_hazard_food_attraction_guard >= 0.0
        else scaffold_cfg.scaffold_p2_hazard_food_attraction
    )
    return CausalGridWorldV2(
        size=scaffold_cfg.scaffold_env_size,
        num_hazards=scaffold_cfg.scaffold_p2_num_hazards,
        num_resources=scaffold_cfg.scaffold_p2_num_resources,
        hazard_food_attraction=p2_hfa,
        proximity_harm_scale=scaffold_cfg.scaffold_p2_proximity_harm_scale,
        limb_damage_enabled=True,
        reef_enabled=True,
        reef_bipartite_layout=True,
        reef_bipartite_axis=scaffold_cfg.scaffold_reef_bipartite_axis,
        reef_bipartite_agent_band_radius=scaffold_cfg.scaffold_reef_bipartite_agent_band_radius,
        reef_bipartite_agent_spawn_in_reef_half=False,
        dual_cue_enabled=True,
        dual_cue_min_active_ticks=10,
        dual_cue_replace_on_early_consume=False,
        dual_cue_type_tags=(1, 2),
        **_sd049_kwargs(scaffold_cfg),
    )


def _counterfactual_external_task_wins(
    live_coord: SalienceCoordinator, signals: Dict[str, float], engagement: float
) -> bool:
    """Would external_task win the mode argmax at this engagement, holding every other
    measured signal fixed?

    Evaluated on a SCRATCH coordinator built from a deepcopy of the live config, so it
    reuses the real logit/softmax arithmetic instead of duplicating it here (duplication
    is how a probe drifts out of sync with the substrate it is probing). argmax depends
    only on the logits, not on the scratch instance's discrete-mode history, so a fresh
    instance per call is correct.
    """
    scratch = SalienceCoordinator(
        config=copy.deepcopy(live_coord.config), mode_names=list(live_coord.mode_names)
    )
    extra = dict(signals)
    extra["external_task_drive"] = float(engagement)
    out = scratch.tick(
        dacc_bundle={
            "pe": signals.get("dacc_pe", 0.0),
            "foraging_value": signals.get("dacc_foraging", 0.0),
            "choice_difficulty": signals.get("dacc_difficulty", 0.0),
        },
        drive_level=signals.get("drive_level", 0.0),
        is_offline=bool(signals.get("is_offline", 0.0) >= 0.5),
        extra_signals=extra,
    )
    om = out["operating_mode"]
    return max(om.items(), key=lambda kv: kv[1])[0] == STICKY_MODE


def _eval_instrumented(
    agent: REEAgent,
    env: CausalGridWorldV2,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    device: torch.device,
    n_eps: int,
    steps_per_ep: int,
) -> Dict[str, Any]:
    """Frozen-policy eval mirroring 467d's _eval_mode_dwell, instrumented at the
    injection path. Rows are recorded ONLY on genuine E3/coordinator ticks."""
    agent.eval()
    world_dim = agent.config.latent.world_dim
    coord = agent.salience
    coord.set_hysteresis_ratio(EVAL_RATIO)
    feed_harm = scaffold_cfg.scaffold_feed_harm_stream

    rows: List[Dict[str, Any]] = []
    n_latched_ticks = 0
    total_steps = 0
    external_task_steps = 0

    with torch.no_grad():
        for _ep in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()

            for _ in range(steps_per_ep):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                latent = _sense_with_optional_harm(
                    agent, obs_body, obs_world, obs_dict, device, feed_harm
                )

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                # Beta state as it stands ENTERING select_action. The engagement block
                # (agent.py:5744) runs BEFORE this call's own elevate() sites (:7273,
                # :7467), so a post-call read reports the POST-elevation latch, not the
                # value the injection arithmetic actually saw. Both are recorded; the
                # routed quantity is the inferred commit term below, which is derived
                # from the injected engagement itself and cannot drift out of phase.
                beta_pre = bool(agent.beta_gate.is_elevated)
                action = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())

                total_steps += 1
                if coord.current_mode == STICKY_MODE:
                    external_task_steps += 1

                # The coordinator only ticks inside select_action's post-E3 block
                # (agent.py:5458 returns early otherwise), so _input_signals LATCHES on a
                # non-E3 step. Record a row ONLY on a genuine tick; count the rest.
                if not ticks.get("e3_tick"):
                    n_latched_ticks += 1
                else:
                    sig = dict(coord._input_signals)
                    last = agent._salience_last_tick or {}
                    engagement = float(sig.get("external_task_drive", 0.0))
                    goal_active = bool(
                        agent.goal_state is not None and agent.goal_state.is_active()
                    )
                    prox = 0.0
                    if agent.goal_state is not None:
                        prox = float(
                            agent.goal_state.goal_proximity(latent.z_world)
                            .reshape(-1)[0].item()
                        )
                    om = last.get("operating_mode") or {}
                    # engagement = clip(commit_w*[beta] + prox_w*prox, 0, 1) at injection
                    # time (agent.py:5754-5769). Subtracting the proximity leg measured on
                    # the SAME latent recovers whether the commit term actually fired --
                    # the quantity D1 routes on. Half the commit weight is the decision
                    # boundary: the term is either ~0 or ~commit_w, never in between.
                    commit_residual = engagement - ET_PROXIMITY_WEIGHT * prox
                    commit_fired = bool(commit_residual > 0.5 * ET_COMMIT_WEIGHT)
                    rows.append({
                        "engagement": engagement,
                        # Routed: reconstructed from the injected value itself.
                        "commit_fired": commit_fired,
                        "commit_residual": commit_residual,
                        # Corroborating only -- see the beta_pre comment above.
                        "beta_elevated_pre_call": beta_pre,
                        "beta_elevated_post_call": bool(agent.beta_gate.is_elevated),
                        "goal_active": goal_active,
                        "goal_proximity": prox,
                        "dacc_pe": float(sig.get("dacc_pe", 0.0)),
                        "dacc_foraging": float(sig.get("dacc_foraging", 0.0)),
                        "dacc_difficulty": float(sig.get("dacc_difficulty", 0.0)),
                        "drive_level": float(sig.get("drive_level", 0.0)),
                        "aic_salience": float(sig.get("aic_salience", 0.0)),
                        "salience_aggregate": float(last.get("salience_aggregate", 0.0)),
                        "enter_threshold": float(last.get("enter_threshold", 0.0)),
                        "exit_threshold": float(last.get("exit_threshold", 0.0)),
                        "current_mode_prob": float(last.get("current_mode_prob", 0.0)),
                        "current_mode": str(last.get("current_mode", "")),
                        "p_external_task": float(om.get(STICKY_MODE, 0.0)),
                        "mode_switch_trigger": bool(last.get("mode_switch_trigger", False)),
                        # Decisive H3c separator: at MAXIMUM injectable engagement, would
                        # external_task win the argmax against this tick's dACC/AIC load?
                        "cf_wins_at_engagement_1": _counterfactual_external_task_wins(
                            coord, sig, 1.0
                        ),
                    })

                _, _harm, done, _info, obs_dict = env.step(action_idx)
                if done:
                    break

    n = len(rows)

    def _mean(key: str) -> float:
        return float(statistics.fmean([r[key] for r in rows])) if n else 0.0

    def _max(key: str) -> float:
        return float(max(r[key] for r in rows)) if n else 0.0

    def _frac_true(key: str) -> float:
        return float(sum(1 for r in rows if r[key])) / n if n else 0.0

    engagement_vals = [r["engagement"] for r in rows]
    return {
        "ratio": EVAL_RATIO,
        "n_e3_ticks": n,
        "n_latched_ticks": n_latched_ticks,
        "total_steps": total_steps,
        "n_episodes": n_eps,
        "fraction_in_external_task": round(
            external_task_steps / total_steps if total_steps else 0.0, 4
        ),
        "commit_fired_frac": round(_frac_true("commit_fired"), 4),
        "beta_elevated_pre_call_frac": round(_frac_true("beta_elevated_pre_call"), 4),
        "beta_elevated_post_call_frac": round(_frac_true("beta_elevated_post_call"), 4),
        "goal_active_frac": round(_frac_true("goal_active"), 4),
        "engagement_mean": round(_mean("engagement"), 6),
        "engagement_max": round(_max("engagement"), 6),
        "engagement_ge_flip_frac": round(
            float(sum(1 for v in engagement_vals if v >= ENGAGEMENT_FLIP_POINT)) / n
            if n else 0.0, 4
        ),
        "goal_proximity_mean": round(_mean("goal_proximity"), 6),
        "goal_proximity_max": round(_max("goal_proximity"), 6),
        "dacc_pe_mean": round(_mean("dacc_pe"), 6),
        "dacc_pe_max": round(_max("dacc_pe"), 6),
        "p_external_task_mean": round(_mean("p_external_task"), 6),
        "cf_wins_at_engagement_1_frac": round(_frac_true("cf_wins_at_engagement_1"), 4),
        "mode_switch_trigger_frac": round(_frac_true("mode_switch_trigger"), 4),
        # Full per-tick rows are retained (Experimental Recording Standard: a false
        # record is free, a false omission forces a multi-hour re-run).
        "per_tick_rows": rows,
    }


def _aborted_seed_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    return {
        "seed": seed,
        "aborted_at": stage,
        "abort_reason": reason,
        "guard_pass": False,
        "stage0_z_goal_norm_peak": 0.0,
        "p2_contact_rate": 0.0,
        "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "instrumentation": {
            "ratio": EVAL_RATIO, "n_e3_ticks": 0, "n_latched_ticks": 0,
            "total_steps": 0, "n_episodes": 0, "fraction_in_external_task": 0.0,
            "commit_fired_frac": 0.0, "beta_elevated_pre_call_frac": 0.0,
            "beta_elevated_post_call_frac": 0.0, "goal_active_frac": 0.0,
            "engagement_mean": 0.0, "engagement_max": 0.0,
            "engagement_ge_flip_frac": 0.0, "goal_proximity_mean": 0.0,
            "goal_proximity_max": 0.0, "dacc_pe_mean": 0.0, "dacc_pe_max": 0.0,
            "p_external_task_mean": 0.0, "cf_wins_at_engagement_1_frac": 0.0,
            "mode_switch_trigger_frac": 0.0, "per_tick_rows": [],
        },
        "sample_floor_met": False,
    }


def _run_seed(seed: int, dry_run: bool, total_eps: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    device = torch.device("cpu")
    steps_per_ep = scaffold_cfg.scaffold_steps_per_episode
    eval_eps = 2 if dry_run else MODE_EVAL_EPISODES

    probe_env = _build_dual_cue_env(scaffold_cfg)
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition {CONDITION_LABEL}", flush=True)
    done = 0

    s0 = scheduler.run_stage0_nursery(agent, device)
    done += s0.n_episodes
    print(f"  [train] stage0_nursery seed={seed} ep {done}/{total_eps}"
          f" z_goal_peak={s0.z_goal_norm_peak:.4f} formed={s0.z_goal_formed}", flush=True)
    if s0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0 reason={s0.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "stage0", s0.abort_reason)

    s0b = scheduler.run_stage0b_consolidation(
        agent, device, stage0_baseline_norm=s0.z_goal_norm_peak
    )
    done += s0b.n_episodes
    print(f"  [train] stage0b_consolidate seed={seed} ep {done}/{total_eps}"
          f" retention={s0b.retention_ratio:.3f}"
          f" gate={'pass' if s0b.retention_gate_passed else 'FAIL'}", flush=True)
    if s0b.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0b reason={s0b.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "stage0b", s0b.abort_reason)

    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(f"  [train] p0_guided seed={seed} ep {done}/{total_eps}"
          f" mean_len={p0.mean_episode_length:.1f} rv={p0.final_running_variance:.5f}", flush=True)
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=p0 reason={p0.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "p0", p0.abort_reason)

    hz = scheduler.run_hazard_avoidance(agent, device)
    done += hz.n_episodes
    print(f"  [train] hazard_avoidance seed={seed} ep {done}/{total_eps}"
          f" median_last={hz.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}", flush=True)
    if hz.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=hazard reason={hz.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "hazard", hz.abort_reason)

    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
          f" median_last={p1.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}", flush=True)

    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    print(f"  [train] p2_guard seed={seed} ep {done}/{total_eps}"
          f" contact_rate={p2.contact_rate:.4f} contact_events={p2.num_contact_events}"
          f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}", flush=True)

    # Readiness positive control: 467d's contact guard, verbatim.
    guard_pass = bool(
        p2.contact_rate > CONTACT_GATE
        and p2.z_goal_norm_at_contact_peak > P2_ZGOAL_GATE
    )

    dual_env = _build_dual_cue_env(scaffold_cfg)
    dual_env.reset()

    print(f"Seed {seed} Condition r={EVAL_RATIO}", flush=True)
    instr = _eval_instrumented(
        agent, dual_env, scaffold_cfg, device, eval_eps, steps_per_ep
    )
    done += eval_eps

    min_ticks = 5 if dry_run else MIN_E3_TICKS
    sample_floor_met = bool(instr["n_e3_ticks"] >= min_ticks)

    print(f"  [train] instrumented_eval seed={seed} ep {done}/{total_eps}"
          f" n_e3_ticks={instr['n_e3_ticks']} n_latched={instr['n_latched_ticks']}"
          f" commit_frac={instr['commit_fired_frac']:.4f}"
          f" eng_mean={instr['engagement_mean']:.4f}"
          f" eng_ge_flip={instr['engagement_ge_flip_frac']:.4f}"
          f" prox_max={instr['goal_proximity_max']:.4f}"
          f" pe_mean={instr['dacc_pe_mean']:.4f}"
          f" cf_win={instr['cf_wins_at_engagement_1_frac']:.4f}"
          f" frac_ext={instr['fraction_in_external_task']:.4f}", flush=True)
    print(f"verdict: {'PASS' if (guard_pass and sample_floor_met) else 'FAIL'} seed={seed}"
          f" guard_pass={guard_pass} sample_floor_met={sample_floor_met}"
          f" (contact_rate={p2.contact_rate:.4f}"
          f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f})", flush=True)

    return {
        "seed": seed,
        "aborted_at": None,
        "abort_reason": "",
        "guard_pass": guard_pass,
        "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "hazard_stage_survival_pass": bool(hz.survival_gate_passed),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
        "p2_num_contact_events": int(p2.num_contact_events),
        "instrumentation": instr,
        "sample_floor_met": sample_floor_met,
    }


def _worst_cell(rows: List[Dict[str, Any]], key: str, seed_key: str = "seed"):
    """Extremum plus the offending cell id, so a `met` written as an all(...) quantifier
    recomputes exactly from `measured` (the mean-vs-quantifier trap)."""
    if not rows:
        return 0.0, None
    worst = min(rows, key=lambda r: r[key])
    return float(worst[key]), worst.get(seed_key)


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    eval_eps = 2 if dry_run else MODE_EVAL_EPISODES
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 2 + eval_eps
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + P2_BUDGET + eval_eps
        )

    per_seed = [_run_seed(s, dry_run, total_eps) for s in SEEDS]

    guard_rows = [r for r in per_seed if r["guard_pass"] and r["sample_floor_met"]]
    n_guard = len(guard_rows)
    guard_fraction = n_guard / len(per_seed) if per_seed else 0.0
    readiness_met = bool(guard_fraction >= MIN_FRACTION)

    # --- readiness preconditions (positive control + sample floor) ---
    contact_worst, contact_seed = _worst_cell(per_seed, "p2_contact_rate")
    zgoal_worst, zgoal_seed = _worst_cell(per_seed, "p2_z_goal_norm_at_contact_peak")
    ticks_rows = [{"seed": r["seed"], "n": r["instrumentation"]["n_e3_ticks"]}
                  for r in per_seed]
    ticks_worst, ticks_seed = _worst_cell(ticks_rows, "n")
    min_ticks = 5 if dry_run else MIN_E3_TICKS

    preconditions = [
        {
            "name": "forager_contact_rate_supra_floor",
            "kind": "readiness",
            "description": "per-seed P2 contact_rate above floor -- the positive control "
                           "that makes an engagement measurement interpretable",
            "control": "467d contact guard verbatim; P2 foraging stage on the trained agent",
            "measured": round(contact_worst, 6),
            "offending_cell": contact_seed,
            "threshold": CONTACT_GATE,
            "direction": "lower",
            "comparator": ">",
            "met": bool(all(r["p2_contact_rate"] > CONTACT_GATE for r in per_seed)),
        },
        {
            "name": "forager_z_goal_norm_at_contact_supra_floor",
            "kind": "readiness",
            "description": "per-seed z_goal norm at contact peak above floor -- the agent "
                           "holds a goal representation, so goal_proximity is meaningful",
            "control": "467d contact guard verbatim; P2 contact events on the trained agent",
            "measured": round(zgoal_worst, 6),
            "offending_cell": zgoal_seed,
            "threshold": P2_ZGOAL_GATE,
            "direction": "lower",
            "comparator": ">",
            "met": bool(all(
                r["p2_z_goal_norm_at_contact_peak"] > P2_ZGOAL_GATE for r in per_seed
            )),
        },
        {
            "name": "genuine_e3_tick_sample_floor",
            "kind": "readiness",
            "description": "per-seed count of GENUINE coordinator ticks (e3_tick-gated; "
                           "latched non-E3 steps excluded) -- the true denominator of "
                           "every engagement fraction routed on below",
            "control": "instrumented frozen-policy eval on the trained agent",
            "measured": float(ticks_worst),
            "offending_cell": ticks_seed,
            "threshold": float(min_ticks),
            "direction": "lower",
            "comparator": ">=",
            "met": bool(all(
                r["instrumentation"]["n_e3_ticks"] >= min_ticks for r in per_seed
            )),
        },
    ]

    # --- aggregate the discrimination statistics over guard-passing seeds only ---
    def _agg(key: str) -> float:
        vals = [r["instrumentation"][key] for r in guard_rows]
        return float(statistics.fmean(vals)) if vals else 0.0

    commit_frac = _agg("commit_fired_frac")
    eng_ge_flip = _agg("engagement_ge_flip_frac")
    frac_ext = _agg("fraction_in_external_task")
    cf_win = _agg("cf_wins_at_engagement_1_frac")
    prox_max = max(
        [r["instrumentation"]["goal_proximity_max"] for r in guard_rows], default=0.0
    )
    pe_mean = _agg("dacc_pe_mean")

    # --- pre-registered discrimination grid ---
    if not readiness_met:
        label = "substrate_not_ready_requeue"
        route = "substrate_not_ready_requeue"
        route_reason = (
            "contact guard and/or genuine-E3-tick sample floor not met on >= 2/3 seeds"
        )
    elif frac_ext > OCCUPANCY_FLOOR:
        label = "occupancy_reproduced_nonzero"
        route = "premise_void_mech266_rerunnable"
        route_reason = (
            "external_task IS occupied above the 467d floor; the 4-run zero does not "
            "reproduce and MECH-266's dose-response is re-runnable as-is"
        )
    elif commit_frac < BETA_FLOOR:
        label = "commitment_layer_starved"
        route = "route_upstream_bg_commitment"
        route_reason = (
            "the injection-time commit term fired on < BETA_FLOOR of genuine coordinator "
            "ticks (reconstructed from engagement, not a post-hoc latch read); the "
            "latch is the only engagement term that can clear the flip point, so "
            "external_task occupancy is gated on the BG / action-commitment layer"
        )
    elif eng_ge_flip < ENGAGEMENT_FRAC_FLOOR:
        label = "injection_path_underpowered"
        route = "build_injection_path"
        route_reason = (
            "the beta latch fires but engagement still clears the flip point on < "
            "ENGAGEMENT_FRAC_FLOOR of ticks -- the injection site, not the coordinator, "
            "is the build target"
        )
    else:
        label = "coordinator_swamped"
        route = "build_salience_coordinator"
        route_reason = (
            "engagement clears the flip point yet external_task is still unoccupied -- "
            "the dACC/AIC load rides above the break-even and the coordinator arithmetic "
            "is the build target"
        )

    # A criterion is degenerate if it was read off an empty or below-floor sample.
    criteria_non_degenerate = {
        "D_commit_fired_frac": bool(readiness_met and n_guard > 0),
        "D_engagement_ge_flip_frac": bool(readiness_met and n_guard > 0),
        "D_fraction_in_external_task": bool(readiness_met and n_guard > 0),
        "D_cf_wins_at_engagement_1_frac": bool(readiness_met and n_guard > 0),
    }

    outcome = "PASS" if readiness_met else "FAIL"

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        # A diagnostic yields routing, not claim verdicts.
        "evidence_direction": "non_contributory",
        "evidence_direction_per_claim": {
            "MECH-266": "non_contributory",
            "SD-032a": "non_contributory",
        },
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "predecessor": PREDECESSOR,
        "substrate": (
            "scaffolded_sd054_onboarding (full curriculum; 603n config) + "
            "SalienceCoordinator (SD-032a) + mode-governance-engagement external_task "
            "drive (use_external_task_drive=True) + GAP-3 dual_cue env primitive. "
            "use_closure_operator OFF. Instrumentation only -- NO substrate change."
        ),
        "condition": CONDITION_LABEL,
        "method_note": (
            "467d's build and eval, with the 5-ratio MECH-266 sweep replaced by a single "
            "r=1.00 arm instrumented at the external_task injection path. Per-tick rows "
            "are E3-gated (agent.py:5458 returns before the salience block at :5744, so "
            "coordinator inputs latch ~9 env steps in 10); n_latched_ticks is emitted so "
            "the true denominator is auditable."
        ),
        "readiness_note": (
            "Readiness = 467d's contact guard (the positive control: an engagement "
            "measurement is only interpretable on a foraging-competent agent) plus a "
            "sample floor on GENUINE coordinator ticks. Below either -> "
            "substrate_not_ready_requeue, never a substrate verdict."
        ),
        "pre_registered_thresholds": {
            "engagement_flip_point": ENGAGEMENT_FLIP_POINT,
            "beta_floor": BETA_FLOOR,
            "engagement_frac_floor": ENGAGEMENT_FRAC_FLOOR,
            "occupancy_floor": OCCUPANCY_FLOOR,
            "min_e3_ticks": MIN_E3_TICKS,
            "contact_gate": CONTACT_GATE,
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "min_fraction": MIN_FRACTION,
            "eval_ratio": EVAL_RATIO,
        },
        "scaffold_curriculum": "scaffolded_sd054_onboarding",
        "stage_plan": stage_plan(),
        "acceptance": {
            "readiness_met": readiness_met,
            "guard_fraction": round(guard_fraction, 4),
            "n_guard_passing_seeds": n_guard,
            "route_reason": route_reason,
        },
        "discrimination": {
            "commit_fired_frac": round(commit_frac, 4),
            "beta_elevated_pre_call_frac": round(_agg("beta_elevated_pre_call_frac"), 4),
            "beta_elevated_post_call_frac": round(_agg("beta_elevated_post_call_frac"), 4),
            "engagement_ge_flip_frac": round(eng_ge_flip, 4),
            "fraction_in_external_task": round(frac_ext, 4),
            "cf_wins_at_engagement_1_frac": round(cf_win, 4),
            "goal_proximity_max": round(prox_max, 6),
            "dacc_pe_mean": round(pe_mean, 6),
        },
        "criteria": [
            {"name": "readiness_met", "load_bearing": True, "passed": readiness_met},
        ],
        "interpretation": {
            "label": label,
            "readiness_route": route,
            "route_reason": route_reason,
            "hypothesis_not_verdict": (
                "A diagnostic self-route is a hypothesis. H1 (nav-competence) and H2 "
                "(coordinator argmax frozen) were falsified BEFORE this run -- H1 by "
                "467d's own 3/3 contact guard, H2 by direct execution of the coordinator. "
                "This spike discriminates within the remaining injection-path space."
            ),
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "secondary_observation": (
                "467d counted mean_dwell in ENV steps while coord.current_mode can only "
                "change on an E3 tick (~1 step in 10), so its dwell units were not "
                "decision units. Recorded, not adjudicated here."
            ),
        },
        "per_seed": per_seed,
    }

    stamp_recording_core(
        manifest,
        config={
            "scaffold": {k: v for k, v in vars(scaffold_cfg).items()
                         if isinstance(v, (int, float, str, bool, type(None)))},
            "world_dim": WORLD_DIM,
            "eval_ratio": EVAL_RATIO,
            "mode_eval_episodes": eval_eps,
            "steps_per_episode": scaffold_cfg.scaffold_steps_per_episode,
            "external_task_drive": {
                "affinity_weight": ET_AFFINITY_WEIGHT,
                "salience_weight": ET_SALIENCE_WEIGHT,
                "commit_weight": ET_COMMIT_WEIGHT,
                "proximity_weight": ET_PROXIMITY_WEIGHT,
            },
            "dry_run": dry_run,
        },
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
    )
    return manifest


def main(dry_run: bool = False) -> Dict[str, Any]:
    manifest = run_experiment(dry_run=dry_run)
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=dry_run,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )

    print(json.dumps(manifest["acceptance"], indent=2))
    print(json.dumps(manifest["discrimination"], indent=2))
    print(f"label: {manifest['interpretation']['label']}")
    print(f"route: {manifest['interpretation']['readiness_route']}")
    print(f"manifest: {out_path}")
    return {"outcome": manifest["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true", help="short smoke run")
    args = parser.parse_args()
    _res = main(dry_run=args.dry_run)
    _outcome_raw = str(_res["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_res["manifest_path"],
        dry_run=args.dry_run,
    )
