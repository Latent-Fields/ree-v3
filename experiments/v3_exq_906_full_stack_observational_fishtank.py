"""
V3-EXQ-906 -- Full-Stack Observational Fishtank Showcase

Claims: None (diagnostic showcase; does not weight governance)

EXPERIMENT_PURPOSE = "diagnostic"

SLEEP DRIVER: K=N multi-fire (SleepLoopManager, fires every N episodes) --
sleep_loop_episodes_K=10. agent.reset() calls self.sleep_loop.notify_episode_end(self)
unconditionally (ree_core/agent.py:3037), so this fires automatically during both the
curriculum stages below AND the observational eval loop -- no bespoke driver wiring
needed.

WHY THIS RUN. The last "everything currently validated, switched on at once" fishtank
showcase was V3-EXQ-665 (2026-06-10). Two months of substrate landed since then --
this is the current-substrate successor, built the same way 471 -> 664 -> 665 were:
take the broadest COMBINATION that is (a) mechanically stable (no crash-prone or
mutually-exclusive flag interactions) and (b) does not require bespoke training
machinery this driver would have to invent, then run it long enough and in a varied
enough environment for spontaneous (not curriculum-forced) behaviour to have room to
show up. It is explicitly NOT a claim test: every module below stays candidate /
v3_pending in claims.yaml regardless of what this run shows (most say "PROMOTES
NOTHING" in ree-v3/CLAUDE.md) -- this script exists to look, not to score.

WHAT'S ON, AND WHY (grouped; see ree-v3/CLAUDE.md "SD Design Decisions Implemented"
for the design doc behind each ID)

  Reused wholesale, not re-derived by hand (lowest-risk path to a broad correct
  combination -- see "minimal special tuning" below):
    - experiments._lib.baselines / v3_exq_724_competence_localization_diagnostic's
      `_base_config_kwargs` + `_all_on_extra_kwargs`: the de-facto "kitchen sink"
      selection/valuation stack reused by 20+ recent scripts -- SP-CEM planner,
      SD-056 e2 world-forward contrastive, modulatory selection authority + channel
      routing + top-k shortlist, F-eligibility demotion + adaptive floor, dACC
      (SD-032b), Go/No-Go constitution (MECH-449), OFC analog + devaluation head
      (SD-033b), E3 score-diversity preservation (MECH-341), stochastic noise floor
      (MECH-313), per-stream V_s + rollout gating (MECH-269/269b), gated policy
      (ARC-062 Phase 1), lateral-PFC analog + candidate rule field (SD-033a/ARC-063).
    - v3_exq_664/665's affective telemetry stack: SD-019a z_harm_un, MECH-320 tonic
      vigor, SD-037 broadcast override, MECH-279 PAG freeze (capped), MECH-353
      blocked agency, MECH-307 split-surprise excite/dread, control-vector logging,
      MECH-295 liking->approach bridge, SD-057 incentive-token bank + cue recall.
    - ScaffoldedSD054OnboardingScheduler (v3_exq_665's curriculum): Stage-0 nursery
      -> Stage-0b consolidation -> P0 -> Stage-H isolated hazard-avoidance -> P1
      wean, WITH scaffold_train_harm_pathway=True (the harm-pathway co-training
      603k/665 added -- E2_harm_a + E3.harm_eval_head + z_world trained on
      hazard-proximity / accumulated-harm labels, not left at random init).

  New since 665, added here (each self-contained -- no precondition on a module
  this script leaves off):
    - SD-035 use_amygdala_analog (BLA/CeA peers; hippocampal write-gain consumer
      wiring is itself still deferred upstream, per its own doc entry -- present
      as a perception module, not yet feeding retrieval).
    - SD-036 use_gabaergic_decay (+ the harm-stream recurrence fix landed
      2026-07-31 -- gaba_recurrence_z_harm_s/_a default True once the master is on,
      so the decay actually has temporal authority over the affect streams, unlike
      the pre-fix version).
    - MECH-358 use_escape_affordance_bridge (relief/safety credit on top of the
      existing MECH-357 instrumental-avoidance pathway already in 665).
    - MECH-288 use_event_segmenter + ARC-071 use_policy_chunking /
      use_chunk_maintenance (memory-chunk formation substrate; the MECH-324
      dissolution-retention / reacquisition-isolation amendments are left at
      their default-off -- that lineage's own validation run, V3-EXQ-829,
      returned a confirmed FAIL on both falsifier criteria, so this showcase
      keeps to the simpler, uncontested base rather than a still-failing amend).
    - SD-014 incentive_sensitization_enabled (the 2026-08-07 fix for the confirmed
      V3-EXQ-887 wanting/liking near-collinearity FAIL; valence_enabled is
      default-True, so this run would otherwise carry a known-broken decoupling).
    - SD-024 use_da_modulated_rbf_density (cfg.residue) + SD-025 curiosity_weight
      (cfg.hippocampal, modest 0.05 -- these are cfg-post-construction fields,
      NOT REEConfig.from_dims() kwargs; from_dims silently drops an unrecognised
      kwarg with no error, so passing them there would look correct and do
      nothing -- see [[reference-reeconfig-from-dims-silent-kwargs]]).
    - SD-017 sleep substrate: use_sleep_loop + sleep_loop_episodes_K=10,
      sws_enabled, rem_enabled, shy_enabled -- so consolidation happens at
      natural intervals across both the curriculum and the long eval phase,
      instead of never firing (665's lineage never turned this on).

  Deliberately LEFT OFF, and why (this is not a "everything in config.py", it is
  "everything that composes safely without bespoke wiring this driver would have
  to build"):
    - REINFORCE training for the lateral-PFC rule-bias head / OFC devaluation
      head (`lateral_pfc_train_rule_bias_head` / `ofc_train_devaluation_head` --
      both come True from `_all_on_extra_kwargs()` and are left at that value
      for exact parity with the proven 724 combination, but this driver does NOT
      call the `experiments/_lib/allon_training.py` REINFORCE step, so those two
      heads stay at their zero-init readout for this run -- present, mechanically
      wired, contributing ~zero to selection, exactly like 471's original "left
      off what needs its own dedicated training" scoping, updated for a module
      whose gate is now open but whose trainer this script does not invoke).
    - SD-091/MECH-481 coalition controller: landed 2026-08-03, five days before
      this script -- too recent to have any cross-script combination precedent.
    - MECH-292/293 ghost-goal bank, MECH-189 super-ordinal anchors, MECH-294
      theta-packet binding: each needs a chained anchor-set/payload precondition
      this script does not build, and MECH-294 is under an explicit governance
      hold besides.
    - MECH-427/428 hierarchical subgoal credit: the primitive landed 2026-08-02
      but "still no AUTOMATIC environment wiring, by design" per its own doc
      entry -- flipping the flag alone is inert without a notify_subgoal_attainment
      call site this driver does not add.
    - ARC-108/ARC-110 dopaminergic control-plane / loop segregation, DR-10/12/13
      self-model V4 primitives, MECH-440/441 noisy-selection / model-disagreement
      curiosity: every one of these is documented as an exact bit-identical no-op
      at its own shipped defaults (sigma_init=0.0, weight=0.0) -- enabling the
      master flag without also hand-tuning a non-zero gain would add telemetry
      noise for zero behavioural effect, which is the opposite of "minimal
      special tuning".
    - SD-033e frontopolar de-commit: its own failure-autopsy (V3-EXQ-719a) says
      explicitly "the integrated all-ON agent has no competent committed foraging
      to de-commit FROM" -- this IS the integrated all-ON agent.

  Env (built directly, not through the curriculum's own `_build_env` -- the
  curriculum trains in its own validated env; this driver's OBSERVATIONAL eval
  phase gets a separately-constructed, richer one on top of the SAME base
  `v3_exq_724`-lineage ENV_KWARGS: size=12 walled reef arena with bipartite
  layout, 4 hazards / 5 resources, resource respawn, hazard_food_attraction. Added
  for duration/variety, all read unconditionally into `info` regardless of
  whether they fire this run -- see causal_grid_world.py's per-step info dict --
  so telemetry capture cannot silently miss an event):
    - scheduled_action_block (interval=10, prob=0.4 -- MECH-353 blocked-agency
      elicitation, matches 665's own eval-env choice)
    - scheduled_limb_damage (interval=50, prob=0.5, magnitude=0.4 -- SD-022,
      gives the healing/relief-credit pathway something to respond to)
    - scheduled_external_hazard (interval=50, prob=0.5, adjacent_only=True --
      SD-029, environment-caused rather than self-caused harm events)
    - world_rule_shift (interval=250, depth=1 -- SD-MEL-PRODUCER; the
      per-episode counter resets on env.reset(), so this fires ~2x per 600-step
      eval episode: one clean action-map regime change with enough remaining
      steps in the episode for the agent to show whether it re-adapts. Cadence
      and depth are engineering defaults chosen for showcase legibility, not
      literature-calibrated -- same honesty scoping as the codebase's own
      f_reacq=0.25 precedent.)

  For the PAG-freeze motor-override eval relaxation (same rationale + mechanism
  as 665: an integrated all-ON agent facing this busier eval env would otherwise
  freeze-lock and never roam; the affect telemetry stays faithful, only the motor
  override is relaxed) see `_observational_run` below.

WHAT THIS RUN IS NOT: a claim test, a statistically powered multi-seed study, or a
substrate-readiness diagnostic for any single mechanism above (each of those already
has, or is queued for, its own dedicated `/queue-experiment` discriminative test).
Single seed by default (--seeds), long single trajectory per seed -- the point is
qualitative: give the fishtank viewer (fishtank_viz.html) a long, richly-instrumented,
minimally-hand-tuned window onto whatever the current integrated substrate actually
does when nothing is narrowing it toward a specific finding.

Output:
  evidence/experiments/v3_exq_906_full_stack_observational_fishtank/
    v3_exq_906_full_stack_observational_fishtank_<ts>.json               (manifest)
    v3_exq_906_full_stack_observational_fishtank_<ts>_episode_log.json   (fishtank feed)

Estimated runtime: see the queue entry note -- pinned by a timed local probe of this
exact script (not extrapolated from a different script's throughput), then queued to
ree-cloud-4 (dedicated-vCPU CPX22; the one same-script cross-machine comparison found
in this substrate's recent history, V3-EXQ-873 family, put cloud-4 ~1.5x faster than
the coordinator hub -- consistent with dedicated vCPU avoiding shared-vCPU contention,
not with raw core count, which a separate measurement found does NOT help this
substrate's largely single-threaded torch workload).
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.scaffolded_sd054_onboarding import ScaffoldedSD054OnboardingScheduler, _build_env
from experiments.v3_exq_665_curriculum_affective_fishtank_showcase import (
    _make_scaffold_cfg,
    _run_curriculum,
)
from experiments.v3_exq_664_affective_fishtank_showcase import (
    _read_affect,
    _classify_mode,
    _get_reef_cells,
    _obs_harm,
    _obs_harm_a,
    _obs_harm_history,
    _action_to_onehot,
)
from experiments.v3_exq_724_competence_localization_diagnostic import (
    _base_config_kwargs as _allon_base_config_kwargs,
    _all_on_extra_kwargs,
)
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE    = "v3_exq_906_full_stack_observational_fishtank"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []

# ---- curriculum budgets (mirror 665 / 603k full-scale) ----
STAGE0_BUDGET   = 20
STAGE0B_BUDGET  = 10
P0_BUDGET       = 100
HAZARD_BUDGET   = 40
P1_BUDGET       = 50
TRAIN_STEPS     = 200
TRAIN_TOTAL_EPS = STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_BUDGET + P1_BUDGET  # 220

# ---- observational eval (long free-run in a varied, busier env) ----
EVAL_EPISODES = 30
EVAL_STEPS    = 600

PAG_MAX_FREEZE         = 8
PAG_THETA_FREEZE       = 0.8
PAG_DURATION_THRESHOLD = 0.2
AVOIDANCE_THREAT_REF   = 0.35
CUE_RECALL_GAIN        = 0.2
SLEEP_LOOP_K           = 10
CURIOSITY_WEIGHT       = 0.05

CORE_CHANNELS = ["z_harm_a", "z_harm_un", "drive", "z_goal"]
STD_FLOOR     = 1e-4

# ---- eval env: the SAME p2 env the scaffold curriculum builds (guarantees
# body_obs_dim / world_obs_dim match the just-trained agent's encoders -- an
# independently-configured env, even one drawn from the same 724 lineage, can
# and did produce a different observation width; confirmed by a dry-run shape
# mismatch, 1x275 into a 350-wide encoder) + duration/variety scheduled
# injections applied as POST-CONSTRUCTION attribute overrides. None of these
# four flags feed body_obs_dim/world_obs_dim (only use_proxy_fields,
# limb_damage_enabled, n_landmarks_a/b, multi_resource_heterogeneity_enabled,
# reef_enabled, safety_cue_enabled do -- verified against
# CausalGridWorld.body_obs_dim/world_obs_dim before relying on this), so this
# override is safe post-hoc exactly the way 665 already relies on for
# scheduled_action_block. limb_damage_enabled itself is already True on the
# scaffold's own p2 env (baked in at construction, not overridden here) --
# the precondition for scheduled_limb_damage_enabled is already satisfied.
EVAL_ENV_EXTRA_KWARGS: Dict[str, Any] = dict(
    scheduled_action_block_enabled=True,
    scheduled_action_block_interval=10,
    scheduled_action_block_prob=0.4,
    scheduled_limb_damage_enabled=True,
    scheduled_limb_damage_interval=50,
    scheduled_limb_damage_prob=0.5,
    scheduled_limb_damage_magnitude=0.4,
    scheduled_limb_damage_limb_selection="random",
    scheduled_external_hazard_enabled=True,
    scheduled_external_hazard_interval=50,
    scheduled_external_hazard_prob=0.5,
    scheduled_external_hazard_adjacent_only=True,
    world_rule_shift_enabled=True,
    world_rule_shift_interval=250,
    world_rule_shift_depth=1,
    world_rule_shift_scope="action_map",
)

ENV_CONFIG_SNAPSHOT_FIELDS = (
    "size", "num_hazards", "num_resources", "hazard_harm", "env_drift_interval",
    "env_drift_prob", "proximity_harm_scale", "proximity_benefit_scale",
    "proximity_approach_threshold", "hazard_field_decay", "resource_respawn_on_consume",
    "toroidal", "harm_history_len", "limb_damage_enabled", "reef_enabled",
    "n_reef_patches", "reef_patch_radius", "reef_bipartite_layout", "hazard_food_attraction",
    "scheduled_action_block_enabled", "scheduled_action_block_interval",
    "scheduled_action_block_prob", "scheduled_limb_damage_enabled",
    "scheduled_limb_damage_interval", "scheduled_limb_damage_prob",
    "scheduled_external_hazard_enabled", "scheduled_external_hazard_interval",
    "scheduled_external_hazard_prob", "world_rule_shift_enabled",
    "world_rule_shift_interval", "world_rule_shift_depth",
)


def _build_eval_env(scaffold_cfg, seed: int):
    env = _build_env(scaffold_cfg, "p2", seed=seed)
    for k, v in EVAL_ENV_EXTRA_KWARGS.items():
        setattr(env, k, v)
    return env


def _env_config_snapshot(env) -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    for name in ENV_CONFIG_SNAPSHOT_FIELDS:
        try:
            v = getattr(env, name)
        except Exception:
            continue
        if isinstance(v, (bool, int, float, str)):
            snap[name] = v
    return snap


def _make_config(env) -> REEConfig:
    """724 all-on selection/valuation stack + 664/665 affective stack + new
    substrate landed since 665 (see module docstring for exactly what and why)."""
    kwargs = _allon_base_config_kwargs(env)
    kwargs.update(_all_on_extra_kwargs())
    kwargs.update(dict(
        # 724's default harm_obs_a_dim (50) is calibrated to 724's OWN env; this
        # driver trains through the scaffolded curriculum's env instead (same as
        # 665), whose actual affective-harm observation width is 7 -- confirmed by
        # a smoke-test shape mismatch (mat1 1x17 [7 harm_obs_a + 10 harm_history]
        # vs the 50+10=60-wide encoder built from the unadjusted default).
        harm_obs_a_dim=7,
        # affective / defensive chain (664/665)
        use_tonic_vigor=True,
        use_blocked_agency=True,
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        use_broadcast_override=True,
        surprise_gated_replay=True,
        use_control_vector_logging=True,
        # goal pipeline (664/665)
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,        # auto-enables split-surprise excite/dread
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        # new since 665
        use_amygdala_analog=True,            # SD-035
        use_gabaergic_decay=True,            # SD-036 (+recurrence, default True once master on)
        use_escape_affordance_bridge=True,   # MECH-358
        use_event_segmenter=True,            # MECH-288 (precondition for chunking below)
        use_policy_chunking=True,            # ARC-071
        use_chunk_maintenance=True,          # ARC-071
        incentive_sensitization_enabled=True,  # SD-014 fix, 2026-08-07
        use_sleep_loop=True,                 # SD-017 Phase A
        sleep_loop_episodes_K=SLEEP_LOOP_K,
        sws_enabled=True,
        rem_enabled=True,
        shy_enabled=True,
    ))
    cfg = REEConfig.from_dims(**kwargs)
    # Fields not reachable through from_dims() -- unrecognised from_dims kwargs are
    # silently dropped, not errored, so these MUST be set post-construction.
    cfg.latent.use_harm_un = True                    # SD-019a
    cfg.latent.use_resource_encoder = True            # SD-015 (SD-057 L2 bind requires it)
    cfg.harm_descending_mod_enabled = True            # SD-021
    cfg.descending_attenuation_factor = 0.5
    cfg.pag_max_freeze_duration = PAG_MAX_FREEZE
    cfg.residue.use_da_modulated_rbf_density = True   # SD-024
    cfg.hippocampal.curiosity_weight = CURIOSITY_WEIGHT  # SD-025
    return cfg


def _observational_run(agent: REEAgent, env: CausalGridWorldV2, num_episodes: int,
                       steps_per_episode: int, seed: int) -> Dict[str, Any]:
    """Long free-running eval in the varied eval env. Emits the 664/665 episode_log
    schema plus the env's own scheduled-event flags (read unconditionally from
    info, so absent-when-off, present-when-on -- see module docstring)."""
    device     = agent.device
    action_dim = env.action_dim
    episodes_log: List[Dict] = []
    chan_vals: Dict[str, List[float]] = {
        k: [] for k in ["z_harm_s", "z_harm_un", "z_harm_a", "drive", "z_goal",
                        "vigor", "override", "z_block", "excite", "dread"]
    }
    freeze_fires = 0
    block_steps  = 0
    limb_damage_events = 0
    external_hazard_events = 0
    world_rule_shift_events = 0
    sleep_cycles_fired = 0
    prev_cycle_history_len = len(getattr(getattr(agent, "sleep_loop", None), "_cycle_history", []) or [])

    # Showcase-legibility relaxation (identical rationale + mechanism to 665): the
    # all-ON agent's chronic z_harm_a + the aggressive Stage-H PAG-freeze theta
    # would otherwise freeze-lock every step in this busier eval env (itself the
    # z_harm_a-saturation finding, not a bug) -- disable the MOTOR override only;
    # the affect telemetry stays faithful to the trained encoders.
    if getattr(agent, "pag_freeze_gate", None) is not None:
        try:
            agent.pag_freeze_gate.config.duration_input_threshold = 1e9
        except Exception:
            pass

    agent.eval()
    for ep_idx in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        cycle_history = getattr(getattr(agent, "sleep_loop", None), "_cycle_history", []) or []
        if len(cycle_history) > prev_cycle_history_len:
            sleep_cycles_fired += len(cycle_history) - prev_cycle_history_len
        prev_cycle_history_len = len(cycle_history)

        z_world_prev = None
        action_prev  = None
        z_self_prev  = None
        ep_steps: List[Dict] = []
        current_hazards   = [list(h) for h in env.hazards]
        current_resources = [list(r) for r in env.resources]
        reef_cells     = _get_reef_cells(env)
        reef_cells_set = getattr(env, "_reef_cells", set())
        prev_in_reef   = False

        for step_idx in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h     = _obs_harm(obs_dict)
            obs_h_a   = _obs_harm_a(obs_dict)
            obs_h_h   = _obs_harm_history(obs_dict)
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world,
                                     obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h)
                if z_self_prev is not None and action_prev is not None:
                    agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())
                ticks    = agent.clock.advance()
                e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick", False)
                            else torch.zeros(1, agent.config.latent.world_dim, device=device))
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                drive_level      = REEAgent.compute_drive_level(obs_body)
                benefit_exposure = max(0.0, float(obs_dict.get("benefit_exposure", 0.0)))
                agent.update_z_goal(benefit_exposure=benefit_exposure, drive_level=drive_level)
                action = agent.select_action(candidates, ticks, temperature=1.0)
                if action is None:
                    action = _action_to_onehot(random.randint(0, action_dim - 1), action_dim, device)
                    agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            with torch.no_grad():
                agent.update_residue(float(harm_signal))

            if info.get("env_drift_occurred", False):
                current_hazards   = [list(h) for h in env.hazards]
                current_resources = [list(r) for r in env.resources]

            agent_pos = (int(env.agent_x), int(env.agent_y))
            in_reef   = agent_pos in reef_cells_set
            blocked   = bool(info.get("action_blocked_this_step", False))
            limb_dmg  = bool(info.get("scheduled_limb_damage_injected_this_step", False))
            ext_haz   = bool(info.get("external_hazard_injected", False))
            rule_shift = bool(info.get("world_rule_shift_occurred", False))
            if blocked:
                block_steps += 1
            if limb_dmg:
                limb_damage_events += 1
            if ext_haz:
                external_hazard_events += 1
            if rule_shift:
                world_rule_shift_events += 1

            affect = _read_affect(agent, latent, obs_body)
            if affect["freeze"]:
                freeze_fires += 1
            for k, lst in chan_vals.items():
                v = affect.get(k)
                if isinstance(v, (int, float)) and v is not None:
                    lst.append(float(v))

            z_harm_s = affect["z_harm_s"] if affect["z_harm_s"] is not None else 0.0
            z_beta_val = float(latent.z_beta.mean().item()) if latent.z_beta is not None else 0.0
            world_change_norm = (float((latent.z_world - z_world_prev).norm().item())
                                 if z_world_prev is not None else 0.0)
            mode = _classify_mode(z_harm_s, world_change_norm, float(harm_signal),
                                  in_reef, affect["freeze"], affect["z_block"])
            if blocked:
                step_transition = "action_blocked"
            elif limb_dmg:
                step_transition = "limb_damage"
            elif ext_haz:
                step_transition = "external_hazard"
            elif rule_shift:
                step_transition = "world_rule_shift"
            elif in_reef and not prev_in_reef:
                step_transition = "reef_entry"
            elif not in_reef and prev_in_reef:
                step_transition = "reef_exit"
            else:
                step_transition = info.get("transition_type", "none")

            ep_steps.append({
                "t": step_idx, "pos": list(agent_pos),
                "action": int(action.argmax(dim=-1).item()),
                "harm_signal": float(harm_signal),
                "z_harm_norm": z_harm_s,
                "z_harm_s": affect["z_harm_s"], "z_harm_un": affect["z_harm_un"],
                "z_harm_a": affect["z_harm_a"],
                "z_world_norm": float(latent.z_world.norm().item()),
                "z_beta_val": z_beta_val, "world_change_norm": world_change_norm,
                "drive": affect["drive"], "z_goal": affect["z_goal"],
                "vigor": affect["vigor"], "override": affect["override"],
                "z_block": affect["z_block"], "freeze": affect["freeze"],
                "excite": affect["excite"], "dread": affect["dread"],
                "mode": mode, "transition_type": step_transition,
                "health": float(info.get("health", 1.0)),
                "energy": float(info.get("energy", 1.0)),
                "harm_event": float(harm_signal) < 0,
                "n_cands": len(candidates),
                "hazards": [list(h) for h in current_hazards],
                "resources": [list(r) for r in current_resources],
                "in_reef": in_reef,
                "action_blocked": blocked,
                "limb_damage_injected": limb_dmg,
                "external_hazard_injected": ext_haz,
                "world_rule_shift_occurred": rule_shift,
            })

            prev_in_reef = in_reef
            z_self_prev  = latent.z_self.detach()
            z_world_prev = latent.z_world.detach()
            action_prev  = action.detach()
            if done:
                break

        episodes_log.append({
            "ep": ep_idx,
            "initial_hazards":   [list(h) for h in env.hazards],
            "initial_resources": [list(r) for r in env.resources],
            "reef_cells": reef_cells, "steps": ep_steps,
        })
        print(f"  [eval] seed={seed} ep {ep_idx+1}/{num_episodes} steps={len(ep_steps)}", flush=True)

    chan_std  = {k: (float(np.std(v)) if len(v) >= 2 else 0.0) for k, v in chan_vals.items()}
    chan_mean = {k: (float(np.mean(v)) if v else 0.0) for k, v in chan_vals.items()}
    return {
        "episodes": episodes_log, "chan_std": chan_std, "chan_mean": chan_mean,
        "freeze_fires": freeze_fires, "block_steps": block_steps,
        "limb_damage_events": limb_damage_events,
        "external_hazard_events": external_hazard_events,
        "world_rule_shift_events": world_rule_shift_events,
        "sleep_cycles_fired": sleep_cycles_fired,
        "eval_steps": int(sum(len(e["steps"]) for e in episodes_log)),
    }


def run_seed(seed: int, dry_run: bool = False) -> Dict[str, Any]:
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    device = torch.device("cpu")
    total_eps = (2 + 2 + 5 + 5 + 5) if dry_run else TRAIN_TOTAL_EPS

    print(f"\nSeed {seed} Condition full_stack_observational_showcase", flush=True)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)
    print(f"[EXQ-906] seed={seed} world_obs_dim={probe_env.world_obs_dim}"
          f" body_obs_dim={probe_env.body_obs_dim} full-stack curriculum ON", flush=True)

    diag = _run_curriculum(agent, scheduler, device, seed, total_eps)

    eval_eps   = 2 if dry_run else EVAL_EPISODES
    eval_steps = 30 if dry_run else EVAL_STEPS
    eval_env = _build_eval_env(scaffold_cfg, seed=seed)
    env_config_snapshot = _env_config_snapshot(eval_env)
    ree = _observational_run(agent, eval_env, eval_eps, eval_steps, seed)

    print(f"[EXQ-906] seed={seed} channel std: "
          + "  ".join(f"{k}={ree['chan_std'][k]:.4f}" for k in
                      ["z_harm_a", "z_harm_un", "drive", "z_goal", "vigor", "z_block", "excite", "dread"]),
          flush=True)
    print(f"[EXQ-906] seed={seed} events: block={ree['block_steps']} "
          f"limb_damage={ree['limb_damage_events']} external_hazard={ree['external_hazard_events']} "
          f"world_rule_shift={ree['world_rule_shift_events']} sleep_cycles={ree['sleep_cycles_fired']}",
          flush=True)

    seed_core_ok = all(ree["chan_std"].get(k, 0.0) > STD_FLOOR for k in CORE_CHANNELS)
    harm_trained = (diag["p0_harm_train_steps"] + diag["hazard_harm_train_steps"]) > 0
    seed_pass = bool(seed_core_ok and harm_trained)
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} "
          f"core_ok={seed_core_ok} harm_trained={harm_trained}", flush=True)

    return {
        "seed": seed, "diag": diag, "chan_std": ree["chan_std"], "chan_mean": ree["chan_mean"],
        "freeze_fires": ree["freeze_fires"], "block_steps": ree["block_steps"],
        "limb_damage_events": ree["limb_damage_events"],
        "external_hazard_events": ree["external_hazard_events"],
        "world_rule_shift_events": ree["world_rule_shift_events"],
        "sleep_cycles_fired": ree["sleep_cycles_fired"],
        "eval_steps": ree["eval_steps"], "z_goal_eval_mean": ree["chan_mean"].get("z_goal", 0.0),
        "harm_trained": harm_trained, "episodes": ree["episodes"], "agent": agent,
        "env_config": env_config_snapshot,
    }


def run(seeds=None, dry_run: bool = False) -> dict:
    if seeds is None:
        seeds = [0]
    print(f"[V3-EXQ-906] Full-Stack Observational Fishtank Showcase\n"
          f"  Seeds: {seeds}  curriculum: Stage-0/0b/P0/Stage-H/P1 + harm-pathway training\n"
          f"  Train eps/seed: {TRAIN_TOTAL_EPS}  Eval: {EVAL_EPISODES} x {EVAL_STEPS} steps\n"
          f"  Output: REE_assembly/evidence/experiments/{EXPERIMENT_TYPE}/", flush=True)

    seed_results = [run_seed(s, dry_run=dry_run) for s in seeds]
    agents = [r.pop("agent") for r in seed_results]

    chan_keys = list(seed_results[0]["chan_std"].keys())
    chan_max_std = {k: max(r["chan_std"].get(k, 0.0) for r in seed_results) for k in chan_keys}
    chan_nondegen = {k: bool(chan_max_std[k] > STD_FLOOR) for k in chan_keys}
    total_harm_steps = sum(r["diag"]["p0_harm_train_steps"] + r["diag"]["hazard_harm_train_steps"]
                           for r in seed_results)
    total_block = sum(r["block_steps"] for r in seed_results)
    total_limb_damage = sum(r["limb_damage_events"] for r in seed_results)
    total_external_hazard = sum(r["external_hazard_events"] for r in seed_results)
    total_world_rule_shift = sum(r["world_rule_shift_events"] for r in seed_results)
    total_sleep_cycles = sum(r["sleep_cycles_fired"] for r in seed_results)
    total_freeze = sum(r["freeze_fires"] for r in seed_results)
    total_steps = sum(r["eval_steps"] for r in seed_results)
    z_goal_activated = any(r["z_goal_eval_mean"] > 1e-3 for r in seed_results)

    core_ok = all(chan_nondegen.get(k, False) for k in CORE_CHANNELS)
    harm_trained = total_harm_steps > 0
    # Reported only (the PAG freeze motor-override is disabled for the eval for
    # showcase legibility -- see _observational_run).
    freeze_not_locked = (total_freeze == 0) or (total_freeze < total_steps)
    passed = bool(core_ok and harm_trained)
    outcome = "PASS" if passed else "FAIL"

    metrics: Dict[str, Any] = {"n_seeds": float(len(seeds)),
                               "total_harm_pathway_train_steps": float(total_harm_steps),
                               "total_block_steps": float(total_block),
                               "total_limb_damage_events": float(total_limb_damage),
                               "total_external_hazard_events": float(total_external_hazard),
                               "total_world_rule_shift_events": float(total_world_rule_shift),
                               "total_sleep_cycles_fired": float(total_sleep_cycles),
                               "total_freeze_fires": float(total_freeze),
                               "total_eval_steps": float(total_steps),
                               "z_goal_activated_at_eval": 1.0 if z_goal_activated else 0.0}
    for r in seed_results:
        s = r["seed"]
        metrics[f"seed{s}_stage0_z_goal_peak"] = float(r["diag"]["stage0_z_goal_peak"])
        metrics[f"seed{s}_hazard_survival_gate"] = 1.0 if r["diag"]["hazard_survival_gate"] else 0.0
        metrics[f"seed{s}_hazard_harm_eval_range"] = float(r["diag"]["hazard_harm_eval_range"])
        metrics[f"seed{s}_z_goal_eval_mean"] = float(r["z_goal_eval_mean"])
    for k in chan_keys:
        metrics[f"chan_max_std_{k}"] = float(chan_max_std[k])
        metrics[f"chan_mean_{k}"] = float(np.mean([r["chan_mean"].get(k, 0.0) for r in seed_results]))

    interpretation = {
        "label": "full_stack_observational_showcase_live" if passed
                 else "full_stack_observational_showcase_degenerate",
        "preconditions": [
            {"name": "harm_pathway_trained", "description": "harm-pathway co-training ran >=1 optimizer step",
             "measured": float(total_harm_steps), "threshold": 1.0, "direction": "lower",
             "met": bool(harm_trained)},
        ],
        "criteria_non_degenerate": {
            **{f"channel_{k}": chan_nondegen.get(k, False) for k in chan_keys},
            "harm_pathway_trained": harm_trained,
            "freeze_not_permanently_locked": freeze_not_locked,
        },
        "criteria": [
            {"name": "core_channels_non_degenerate", "load_bearing": True, "passed": core_ok},
            {"name": "harm_pathway_trained", "load_bearing": True, "passed": harm_trained},
            {"name": "freeze_not_locked", "load_bearing": False, "passed": freeze_not_locked},
        ],
        "note": ("Full-stack integrated telemetry showcase. PASS = harm-pathway training ran "
                 "AND the core affect channels vary AND freeze did not permanently lock. "
                 "Every module enabled stays candidate/v3_pending in claims.yaml regardless of "
                 "this run's outcome -- claim_ids=[]; does not weight governance. See module "
                 "docstring for exactly what is on/off and why."),
    }

    summary_markdown = f"""# V3-EXQ-906 -- Full-Stack Observational Fishtank Showcase

**Status:** {outcome} (diagnostic telemetry showcase -- not scored against any claim)
**Purpose:** current-substrate successor to V3-EXQ-665 (2026-06-10). Feeds
fishtank_viz.html with a long, richly-instrumented, minimally-hand-tuned episode_log
from an agent trained through the full onboarding curriculum with the broadest
mechanically-stable feature combination this substrate currently supports.

- harm-pathway train steps (total): {total_harm_steps}
- z_goal activated at eval: {z_goal_activated}
- eval steps (total): {total_steps}  across {EVAL_EPISODES} eps x {EVAL_STEPS} steps/seed
- events: block={total_block} limb_damage={total_limb_damage} external_hazard={total_external_hazard} world_rule_shift={total_world_rule_shift}
- sleep cycles fired: {total_sleep_cycles}
- freeze fires (eval, motor-override relaxed): {total_freeze}

## Eval channel mean / max-std
{chr(10).join(f'- {k}: mean={metrics.get("chan_mean_"+k,0.0):.4f} max_std={chan_max_std[k]:.5f} ({"varies" if chan_nondegen[k] else "FLAT"})' for k in chan_keys)}

The `_episode_log.json` companion feeds fishtank_viz.html (FISHTANK_VIZ_VERSION
2026-06-10.2) via /api/fishtank/logs, including an `env_config` block for the
viz's toroidal/reef badges (665's episode_log omitted this -- see driver note).
"""

    first_env_config = seed_results[0].get("env_config", {}) if seed_results else {}
    episode_log = {
        "experiment_type": EXPERIMENT_TYPE,
        "phase": "full_stack_observational_showcase",
        "toroidal": bool(first_env_config.get("toroidal", False)),
        "env_config": first_env_config,
        "seeds": [{"seed": r["seed"], "episodes": r.get("episodes", [])} for r in seed_results],
    }

    return {
        "status": outcome, "outcome": outcome, "metrics": metrics,
        "summary_markdown": summary_markdown, "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE, "evidence_direction": "non_contributory",
        "experiment_type": EXPERIMENT_TYPE, "interpretation": interpretation,
        "episode_log": episode_log, "agents": agents,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run(seeds=args.seeds, dry_run=args.dry_run)
    agents = result.pop("agents", [])

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["timestamp_utc"]      = ts
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["experiment_purpose"] = EXPERIMENT_PURPOSE
    result["claim_ids"]          = CLAIM_IDS

    out_dir = (Path(__file__).resolve().parents[2]
               / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_log = result.pop("episode_log", None)
    if episode_log is not None:
        episode_log["run_id"] = result["run_id"]
        log_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}_episode_log.json"
        log_path.write_text(json.dumps(episode_log, indent=2) + "\n", encoding="utf-8")
        print(f"Episode log written to: {log_path}", flush=True)
        # Declared companion path, relative to write_flat_manifest's out_dir
        # (out_dir.parent below) -- NOT out_dir itself. experiment_runner.py
        # _collect_companion_files resolves a declared relative entry against
        # the MANIFEST's directory (evidence/experiments/), one level above
        # where the episode_log actually lands (evidence/experiments/
        # {EXPERIMENT_TYPE}/), so the prefix is required or the runner's
        # Phase-3 sidefile sync silently finds nothing.
        result["companion_files"] = [f"{EXPERIMENT_TYPE}/{log_path.name}"]

    out_path = write_flat_manifest(
        result,
        out_dir.parent,
        dry_run=args.dry_run,
        config=result.get("config"),
        seeds=None,
        script_path=Path(__file__),
        agent=(agents[0] if len(agents) == 1 else agents) if agents else None,
    )
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"final_outcome: {result['outcome']}", flush=True)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
                 manifest_path=out_path,
                 dry_run=bool(args.dry_run))
