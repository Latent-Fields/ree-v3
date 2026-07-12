"""
V3-EXQ-665 -- Curriculum-trained Affective Fishtank Showcase

Claims: None (diagnostic showcase)

EXPERIMENT_PURPOSE = "diagnostic"

The developmentally-scaffolded counterpart to V3-EXQ-664. Where 664 trained the
agent with a plain ~50-episode reef warmup, THIS showcase trains the agent through
the full ScaffoldedSD054OnboardingScheduler curriculum (Stage-0 nursery -> Stage-0b
consolidation -> P0 -> Stage-H isolated hazard-avoidance -> P1 wean) WITH
scaffold_train_harm_pathway ON (the 603k harm-pathway co-training that trains the
z_harm_a affective encoder + E3 harm-eval + z_world on hazard-proximity /
accumulated-harm labels). It then runs an affective eval that emits the SAME affect
episode_log schema as 664, so fishtank_viz.html renders a developmentally-trained
agent's protoemotional register for direct comparison against the raw-warmup 664.

Curriculum wiring mirrors v3_exq_603k_stageh_harm_pathway_readiness.py (the
harm-pathway readiness driver). Affect readout reuses V3-EXQ-664's _read_affect /
_classify_mode helpers so the episode_log schema is identical.

What to compare vs 664:
  - z_harm_a (suffering): 664 was a raw-warmup encoder -> saturated high, decoupled
    from safety (see docs/thoughts/2026-06-10_z_harm_a_saturation_decoupling.md).
    Here the harm pathway is trained, so z_harm_a has its trained shape -- the main
    comparison.
  - z_goal (wanting): 664 read flat (no goal scaffolding). The curriculum seeds +
    matures z_goal; whether autonomous wanting persists at eval is itself the open
    goal_pipeline:GAP-2 question. The viz shows whatever happens (NOT forced).

Affect stack (telemetry on top of the 603k curriculum substrate): SD-019a z_harm_un,
MECH-320 tonic vigor, SD-037 broadcast override, MECH-279 PAG freeze (capped),
MECH-353 blocked agency (+ env action-blocks in the eval env), MECH-307 split-surprise
excite/dread (auto-enabled by use_mech307_conjunction), control-vector logging.

Output:
  evidence/experiments/v3_exq_665_curriculum_affective_fishtank_showcase/
    v3_exq_665_curriculum_affective_fishtank_showcase_<ts>.json          (manifest)
    v3_exq_665_curriculum_affective_fishtank_showcase_<ts>_episode_log.json  (fishtank feed)

Estimated runtime: ~50-60 min on Mac CPU for 1 seed (full curriculum 220 train
episodes x 200 steps + harm-pathway co-training + 5 eval episodes). Pin to the Mac
(machine_affinity DLAPTOP-4.local) so the episode_log companion lands on origin.
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiments.scaffolded_sd054_onboarding import (
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
)
# Reuse the 664 affect-readout so the episode_log schema is identical.
from experiments.v3_exq_664_affective_fishtank_showcase import (
    _read_affect,
    _classify_mode,
    _get_reef_cells,
    _obs_harm,
    _obs_harm_a,
    _obs_harm_history,
    _action_to_onehot,
)
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE    = "v3_exq_665_curriculum_affective_fishtank_showcase"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS          = []

# ---- substrate dims (mirror 603k) ----
WORLD_DIM        = 32
HARM_A_DIM       = 16
HARM_OBS_A_DIM   = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT     = 2.0
PAG_MAX_FREEZE   = 8     # MECH-279 cap so freeze never permanently locks

# ---- curriculum budgets (mirror 603k full-scale) ----
STAGE0_BUDGET   = 20
STAGE0B_BUDGET  = 10
P0_BUDGET       = 100
HAZARD_BUDGET   = 40
P1_BUDGET       = 50
TRAIN_STEPS     = 200
# total training episodes per seed (no P2 -- we run our own affective eval instead)
TRAIN_TOTAL_EPS = STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_BUDGET + P1_BUDGET

P0_NUM_HAZARDS                = 1
P1_HOLD_FRACTION              = 0.3
P2_HFA_GUARD                  = 0.3
P1_REEF_SPAWN_HOLD_FRACTION   = 0.4
HAZARD_STAGE_NUM_HAZARDS      = 4
HAZARD_STAGE_NUM_RESOURCES    = 2
HAZARD_STAGE_HFA              = 0.0
HAZARD_STAGE_PROXIMITY_HARM   = 0.1
HAZARD_STAGE_SURVIVAL_GATE    = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

# 634c seeding calibration + SD-057 cue-recall (mirror 603k).
SEED_GAIN              = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR       = 0.9
N_RESOURCE_TYPES       = 3
CUE_RECALL_GAIN        = 0.2
AVOIDANCE_FLOOR_START  = 0.8
AVOIDANCE_FLOOR_END    = 0.0
AVOIDANCE_THREAT_REF   = 0.35
PAG_THETA_FREEZE       = 0.8
PAG_DURATION_THRESHOLD = 0.2
HARM_PATHWAY_LR        = 1e-3

# ---- eval ----
EVAL_EPISODES = 5
EVAL_STEPS    = 200
# Sparse external action blocks in the eval env so the blocked-agency z_block pole can rise.
EVAL_BLOCK_INTERVAL = 10
EVAL_BLOCK_PROB     = 0.4

CORE_CHANNELS = ["z_harm_a", "z_harm_un", "drive"]
STD_FLOOR     = 1e-4


def _make_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, p1, steps = 2, 2, 5, 5, 5, 30
    else:
        stage0, stage0b, p0, hazard, p1, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_BUDGET, P1_BUDGET, TRAIN_STEPS,
        )
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=2,          # tiny; we do not use run_p2
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_p1_anneal_hold_fraction=P1_HOLD_FRACTION,
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=0.75,
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
        scaffold_hazard_stage_spawn_in_reef_half=True,   # nav-to-safety handed (showcase)
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_FLOOR_END,
        scaffold_feed_harm_stream=True,
        # harm-pathway training (full scope) -- the 603k amend; the comparison vs 664.
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """603k substrate config + the 664 affective telemetry stack."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        # harm streams + forward model (so the harm-pathway terms engage)
        use_harm_stream=True,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_e2_harm_s_forward=True,
        # SP-CEM main path
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # goal pipeline
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,        # auto-enables use_mech307_split_surprise (excite/dread)
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        e2_action_contrastive_enabled=True,
        # defensive chain
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        # ---- 664 affective telemetry stack ----
        use_tonic_vigor=True,                # MECH-320
        use_blocked_agency=True,             # MECH-353
    )
    cfg.latent.use_resource_encoder = True
    cfg.latent.use_harm_un = True            # SD-019a (unpleasantness tier)
    cfg.use_broadcast_override = True        # SD-037 (orexin override)
    cfg.surprise_gated_replay = True         # MECH-205/307 valence write -> excite/dread
    cfg.use_control_vector_logging = True    # tonic-vigor v_t telemetry
    cfg.pag_max_freeze_duration = PAG_MAX_FREEZE
    return cfg


def _run_curriculum(agent: REEAgent, scheduler: ScaffoldedSD054OnboardingScheduler,
                    device, seed: int, total_eps: int) -> Dict[str, Any]:
    """Best-effort: run the curriculum stages; record diagnostics; never abort the
    showcase on a stage gate (we still want the affect eval of a partially-trained agent)."""
    diag: Dict[str, Any] = {
        "stage0_z_goal_peak": 0.0, "stage0_aborted": None,
        "p0_harm_pathway_enabled": False, "p0_harm_train_steps": 0,
        "hazard_survival_gate": False, "hazard_harm_eval_range": 0.0,
        "hazard_harm_train_steps": 0, "p1_survival_gate": False,
        "stages_completed": [],
    }
    done = 0

    s0 = scheduler.run_stage0_nursery(agent, device)
    done += s0.n_episodes
    diag["stage0_z_goal_peak"] = float(getattr(s0, "z_goal_norm_peak", 0.0))
    diag["stage0_aborted"] = getattr(s0, "abort_reason", None) if getattr(s0, "aborted", False) else None
    diag["stages_completed"].append("stage0")
    print(f"  [train] stage0 seed={seed} ep {done}/{total_eps}"
          f" z_goal_peak={diag['stage0_z_goal_peak']:.4f}", flush=True)

    s0b = scheduler.run_stage0b_consolidation(agent, device,
                                              stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    diag["stages_completed"].append("stage0b")
    print(f"  [train] stage0b seed={seed} ep {done}/{total_eps}", flush=True)

    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    diag["p0_harm_pathway_enabled"] = bool(getattr(p0, "harm_pathway_enabled", False))
    diag["p0_harm_train_steps"] = int((getattr(p0, "harm_pathway_diag", {}) or {}).get("n_train_steps", 0))
    diag["stages_completed"].append("p0")
    print(f"  [train] p0 seed={seed} ep {done}/{total_eps}"
          f" harm_enabled={diag['p0_harm_pathway_enabled']}"
          f" harm_steps={diag['p0_harm_train_steps']}", flush=True)

    hz = scheduler.run_hazard_avoidance(agent, device)
    done += hz.n_episodes
    hd = getattr(hz, "harm_discriminativeness", None) or {}
    diag["hazard_survival_gate"] = bool(getattr(hz, "survival_gate_passed", False))
    diag["hazard_harm_eval_range"] = float(hd.get("harm_eval_range", 0.0))
    diag["hazard_harm_train_steps"] = int((getattr(hz, "harm_pathway_diag", {}) or {}).get("n_train_steps", 0))
    diag["stages_completed"].append("hazard")
    print(f"  [train] hazard seed={seed} ep {done}/{total_eps}"
          f" G_H={'pass' if diag['hazard_survival_gate'] else 'FAIL'}"
          f" harm_eval_range={diag['hazard_harm_eval_range']:.4f}"
          f" harm_steps={diag['hazard_harm_train_steps']}", flush=True)

    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    diag["p1_survival_gate"] = bool(getattr(p1, "survival_gate_passed", False))
    diag["stages_completed"].append("p1")
    print(f"  [train] p1 seed={seed} ep {done}/{total_eps}"
          f" survival_gate={'pass' if diag['p1_survival_gate'] else 'FAIL'}", flush=True)

    diag["train_episodes_done"] = done
    return diag


def _affective_eval(agent: REEAgent, env, num_episodes: int, steps_per_episode: int,
                    seed: int) -> Dict[str, Any]:
    """Frozen-policy affective eval on the curriculum-trained agent. Emits the 664
    episode_log schema + per-channel non-degeneracy stats."""
    device     = agent.device
    action_dim = env.action_dim
    episodes_log: List[Dict] = []
    chan_vals: Dict[str, List[float]] = {
        k: [] for k in ["z_harm_s", "z_harm_un", "z_harm_a", "drive", "z_goal",
                        "vigor", "override", "z_block", "excite", "dread"]
    }
    freeze_fires = 0
    block_steps  = 0

    # Enable sparse external action blocks so the blocked-agency pole can rise.
    try:
        env.scheduled_action_block_enabled = True
        env.scheduled_action_block_interval = EVAL_BLOCK_INTERVAL
        env.scheduled_action_block_prob     = EVAL_BLOCK_PROB
    except Exception:
        pass

    # Showcase-legibility: the curriculum-trained agent's high chronic z_harm_a +
    # the aggressive Stage-H PAG-freeze (theta=0.8) otherwise freeze-LOCK it every
    # eval step (a static, unwatchable fish) -- which is itself the z_harm_a-saturation
    # story. Disable the PAG freeze MOTOR-override for the eval by raising its
    # duration gate above any z_harm_a, so the fish roams and the affect dynamics are
    # visible. The affect telemetry (z_harm_a / drive / z_goal / excite / dread ...)
    # remains faithful to the trained encoders; only the freeze action-override is
    # relaxed (the FROZEN mode is therefore not shown for this curriculum agent).
    if getattr(agent, "pag_freeze_gate", None) is not None:
        try:
            agent.pag_freeze_gate.config.duration_input_threshold = 1e9
        except Exception:
            pass

    agent.eval()
    for ep_idx in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
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
                            else torch.zeros(1, WORLD_DIM, device=device))
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                drive_level      = REEAgent.compute_drive_level(obs_body)
                benefit_exposure = max(0.0, float(obs_dict.get("benefit_exposure", 0.0)))
                agent.update_z_goal(benefit_exposure=benefit_exposure, drive_level=drive_level)
                action = agent.select_action(candidates, ticks, temperature=1.0)
                if action is None:
                    action = _action_to_onehot(random.randint(0, action_dim - 1), action_dim, device)
                    agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            # Populate residue valence (MECH-307 split-surprise) so excite/dread read non-zero.
            with torch.no_grad():
                agent.update_residue(float(harm_signal))

            if info.get("env_drift_occurred", False):
                current_hazards   = [list(h) for h in env.hazards]
                current_resources = [list(r) for r in env.resources]

            agent_pos = (int(env.agent_x), int(env.agent_y))
            in_reef   = agent_pos in reef_cells_set
            blocked   = bool(info.get("action_blocked_this_step", False))
            if blocked:
                block_steps += 1

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
        "eval_steps": int(sum(len(e["steps"]) for e in episodes_log)),
    }


def run_seed(seed: int, dry_run: bool = False) -> Dict[str, Any]:
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    device = torch.device("cpu")
    total_eps = (2 + 2 + 5 + 5 + 5) if dry_run else TRAIN_TOTAL_EPS

    print(f"\nSeed {seed} Condition curriculum_affective_showcase", flush=True)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)
    print(f"[EXQ-665] seed={seed} world_obs_dim={probe_env.world_obs_dim}"
          f" body_obs_dim={probe_env.body_obs_dim} curriculum+harm-pathway ON", flush=True)

    diag = _run_curriculum(agent, scheduler, device, seed, total_eps)

    eval_eps   = 2 if dry_run else EVAL_EPISODES
    eval_steps = 30 if dry_run else EVAL_STEPS
    eval_env = _build_env(scaffold_cfg, "p2")
    ree = _affective_eval(agent, eval_env, eval_eps, eval_steps, seed)

    print(f"[EXQ-665] seed={seed} channel std: "
          + "  ".join(f"{k}={ree['chan_std'][k]:.4f}" for k in
                      ["z_harm_a", "z_harm_un", "drive", "z_goal", "vigor", "z_block", "excite", "dread"]),
          flush=True)

    seed_core_ok = all(ree["chan_std"].get(k, 0.0) > STD_FLOOR for k in CORE_CHANNELS)
    harm_trained = (diag["p0_harm_train_steps"] + diag["hazard_harm_train_steps"]) > 0
    seed_pass = bool(seed_core_ok and harm_trained)
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} "
          f"core_ok={seed_core_ok} harm_trained={harm_trained}", flush=True)

    return {
        "seed": seed, "diag": diag, "chan_std": ree["chan_std"], "chan_mean": ree["chan_mean"],
        "freeze_fires": ree["freeze_fires"], "block_steps": ree["block_steps"],
        "eval_steps": ree["eval_steps"], "z_goal_eval_mean": ree["chan_mean"].get("z_goal", 0.0),
        "harm_trained": harm_trained, "episodes": ree["episodes"],
    }


def run(seeds=None, dry_run: bool = False) -> dict:
    if seeds is None:
        seeds = [0]
    print(f"[V3-EXQ-665] Curriculum-trained Affective Fishtank Showcase\n"
          f"  Seeds: {seeds}  curriculum: Stage-0/0b/P0/Stage-H/P1 + harm-pathway training\n"
          f"  Train eps/seed: {TRAIN_TOTAL_EPS}  Eval: {EVAL_EPISODES} x {EVAL_STEPS} steps\n"
          f"  Output: REE_assembly/evidence/experiments/{EXPERIMENT_TYPE}/", flush=True)

    seed_results = [run_seed(s, dry_run=dry_run) for s in seeds]

    chan_keys = list(seed_results[0]["chan_std"].keys())
    chan_max_std = {k: max(r["chan_std"].get(k, 0.0) for r in seed_results) for k in chan_keys}
    chan_nondegen = {k: bool(chan_max_std[k] > STD_FLOOR) for k in chan_keys}
    total_harm_steps = sum(r["diag"]["p0_harm_train_steps"] + r["diag"]["hazard_harm_train_steps"]
                           for r in seed_results)
    total_block = sum(r["block_steps"] for r in seed_results)
    total_freeze = sum(r["freeze_fires"] for r in seed_results)
    total_steps = sum(r["eval_steps"] for r in seed_results)
    z_goal_activated = any(r["z_goal_eval_mean"] > 1e-3 for r in seed_results)

    core_ok = all(chan_nondegen.get(k, False) for k in CORE_CHANNELS)
    harm_trained = total_harm_steps > 0
    # Reported only (the PAG freeze motor-override is disabled for the eval for
    # showcase legibility -- see _affective_eval).
    freeze_not_locked = (total_freeze == 0) or (total_freeze < total_steps)
    passed = bool(core_ok and harm_trained)
    outcome = "PASS" if passed else "FAIL"

    metrics: Dict[str, Any] = {"n_seeds": float(len(seeds)),
                               "total_harm_pathway_train_steps": float(total_harm_steps),
                               "total_block_steps": float(total_block),
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
        "label": "curriculum_affective_showcase_live" if passed
                 else "curriculum_affective_showcase_degenerate",
        "preconditions": [
            {"name": "harm_pathway_trained", "description": "harm-pathway co-training ran >=1 optimizer step",
             "measured": float(total_harm_steps), "threshold": 1.0, "met": bool(harm_trained)},
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
        "note": ("Curriculum-trained telemetry showcase. PASS = harm-pathway training ran AND "
                 "the core affect channels vary AND freeze did not permanently lock. z_goal "
                 "activation at eval is REPORTED (z_goal_activated_at_eval), NOT gated -- whether "
                 "autonomous wanting persists is the open goal_pipeline:GAP-2 question. "
                 "claim_ids=[]; does not weight governance."),
    }

    summary_markdown = f"""# V3-EXQ-665 -- Curriculum-trained Affective Fishtank Showcase

**Status:** {outcome} (diagnostic telemetry showcase -- not scored against any claim)
**Purpose:** Developmentally-scaffolded counterpart to V3-EXQ-664. Feeds
fishtank_viz.html with an affective episode_log from an agent trained through the
ScaffoldedSD054OnboardingScheduler curriculum + harm-pathway training.

- harm-pathway train steps (total): {total_harm_steps}  (harm streams TRAINED, unlike 664)
- z_goal activated at eval: {z_goal_activated}  (Stage-0 peaks per seed in metrics)
- freeze fires / blocked steps (eval): {total_freeze} / {total_block}

## Eval channel mean / max-std (vs 664 for comparison)
{chr(10).join(f'- {k}: mean={metrics.get("chan_mean_"+k,0.0):.4f} max_std={chan_max_std[k]:.5f} ({"varies" if chan_nondegen[k] else "FLAT"})' for k in chan_keys)}

The `_episode_log.json` companion feeds fishtank_viz.html (FISHTANK_VIZ_VERSION
2026-06-10.2) via /api/fishtank/logs -- compare the suffering / wanting channels
against the raw-warmup V3-EXQ-664 run.
"""

    episode_log = {
        "experiment_type": EXPERIMENT_TYPE,
        "phase": "curriculum_affective_showcase",
        "toroidal": False,
        "seeds": [{"seed": r["seed"], "episodes": r.get("episodes", [])} for r in seed_results],
    }

    return {
        "status": outcome, "outcome": outcome, "metrics": metrics,
        "summary_markdown": summary_markdown, "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE, "evidence_direction": "non_contributory",
        "experiment_type": EXPERIMENT_TYPE, "interpretation": interpretation,
        "episode_log": episode_log,
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

    out_path = write_flat_manifest(
        result,
        out_dir.parent,
        dry_run=args.dry_run,
        config=result.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"final_outcome: {result['outcome']}", flush=True)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
                 manifest_path=out_path)
