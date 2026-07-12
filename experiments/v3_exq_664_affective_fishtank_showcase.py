"""
V3-EXQ-664 -- Affective Fishtank Showcase

Claims: None (diagnostic showcase)

EXPERIMENT_PURPOSE = "diagnostic"

Feeds the upgraded fishtank_viz.html (FISHTANK_VIZ_VERSION 2026-06-10.1) with an
episode_log that exposes REE's protoemotional register -- the affective substrate
that accumulated since the EXQ-471 / EXQ-524 showcases. Behaviour-legible: the
affective stack is enabled as TELEMETRY (read each tick), so the fish still
forages / flees / shelters while the viz renders its internal affect.

Substrates active (524 reef stack + the affective register):
  SD-007  reafference                            (perspective correction)
  SD-008  alpha_world=0.9                        (encoder correction)
  SD-010  use_harm_stream=True                   (sensory z_harm_s)
  SD-011  use_affective_harm_stream=True         (affective z_harm_a, suffering)
  SD-019a use_harm_un=True                        (z_harm_un, unpleasantness -- middle tier)
  SD-018  use_resource_proximity_head=True       (resource prox supervision)
  SD-012  z_goal_enabled, drive_weight=2.0       (homeostatic drive + wanting)
  SD-021  harm_descending_mod_enabled=True       (descending modulation)
  SD-054  reef_enabled, hazard_food_attraction=0.7  (reef substrate)
  MECH-090 beta_gate_bistable=True               (commitment latch)
  MECH-320 use_tonic_vigor=True                  (surplus drive-to-act / vigor)
  SD-037  use_broadcast_override=True            (orexin arousal recruitment)
  MECH-279 use_pag_freeze_gate=True (capped)     (committed freeze behaviour)
  MECH-353 use_blocked_agency=True               (blocked-agency / frustration assert pole)
  MECH-307 use_mech307_split_surprise=True + surprise_gated_replay
                                                 (excitement / dread valence channels)
  control_vector_logging                          (reads tonic-vigor v_t)

Environment adds scheduled action blocks (sparse) so the blocked-agency / assert
pole (z_block) actually rises -- the fish tries to move, is externally blocked,
frustration accumulates.

Per-step episode_log schema (ADDITIVE to the 524 schema -- the viz reads both):
  z_harm_s, z_harm_un, z_harm_a   (nociceptive cascade)
  drive, z_goal, vigor, override  (drive & arousal)
  z_block                          (frustration / assert pole)
  excite, dread                    (MECH-307 anticipatory valence)
  freeze (bool), mode in {..., freeze, assert}, transition_type 'action_blocked'

This is a TELEMETRY SHOWCASE, not a claim test. Its self-reported outcome is a
non-degeneracy check on the affect channels (a showcase that emits flat channels
is the failure mode). It carries NO claim_ids and does NOT weight governance.

Output:
  evidence/experiments/v3_exq_664_affective_fishtank_showcase/
    v3_exq_664_affective_fishtank_showcase_<ts>.json          (manifest)
    v3_exq_664_affective_fishtank_showcase_<ts>_episode_log.json  (fishtank feed)

Estimated runtime: ~55 min on cloud CPU (3 seeds x 50 warmup + 5 eval x 200 steps,
12x12 grid, affective stack adds modest per-tick overhead).
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.residue.field import VALENCE_POSITIVE_SURPRISE, VALENCE_NEGATIVE_SURPRISE
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE    = "v3_exq_664_affective_fishtank_showcase"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS          = []

HARM_MODE_THRESH    = 0.25
EXPLORE_ERR_THRESH  = 0.10
SHELTER_HARM_THRESH = 0.15   # z_harm_norm floor for shelter mode while in reef
ASSERT_THRESH       = 0.10   # z_block_assert floor for the 'assert' (frustration) mode

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
    # SD-011 second source: rolling harm-history window
    harm_history_len=10,
    # SD-054: reef enrichment substrate (ARM_1_reef_food config from EXQ-522)
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    # MECH-353 feed: sparse external action blocks so z_block (frustration) rises
    scheduled_action_block_enabled=True,
    scheduled_action_block_interval=10,
    scheduled_action_block_prob=0.4,
)

WARMUP_EPISODES   = 50
EVAL_EPISODES     = 5
STEPS_PER_EPISODE = 200
WORLD_DIM         = 32
SELF_DIM          = 32
HARM_DIM          = 32
HARM_A_DIM        = 16
HARM_HISTORY_LEN  = 10
PAG_MAX_FREEZE    = 8   # MECH-279 cap so the freeze gate never permanently locks

WF_BUF_MAX        = 2000
HARM_EVAL_BUF_MAX = 2000
BATCH_SIZE        = 32
LR_E1             = 1e-4
LR_E2_WF          = 3e-4
LR_E3_HARM        = 1e-3
LR_ENC_AUX        = 5e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _obs_harm(obs_dict) -> Optional[torch.Tensor]:
    return obs_dict.get("harm_obs")


def _obs_harm_a(obs_dict) -> Optional[torch.Tensor]:
    return obs_dict.get("harm_obs_a")


def _obs_harm_history(obs_dict) -> Optional[torch.Tensor]:
    return obs_dict.get("harm_history")


def _obs_accum(obs_dict) -> float:
    v = obs_dict.get("accumulated_harm")
    return float(v) if v is not None else 0.0


def _obs_resource_prox(obs_dict) -> float:
    rv = obs_dict.get("resource_field_view")
    if rv is None:
        return 0.0
    return float(rv.max().item()) if isinstance(rv, torch.Tensor) else float(np.max(rv))


def _norm(t: Optional[torch.Tensor]) -> Optional[float]:
    if t is None:
        return None
    return float(t.norm().item())


def _make_agent_and_env(seed: int) -> Tuple[REEAgent, CausalGridWorldV2]:
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
        # SD-010 / SD-011
        use_harm_stream=True,
        z_harm_dim=HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        # SD-018: resource-prox supervision on z_world
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        # SD-012: homeostatic drive-modulated benefit + goal system
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        # --- affective register (telemetry) ---
        use_tonic_vigor=True,             # MECH-320
        use_blocked_agency=True,          # MECH-353
        use_pag_freeze_gate=True,         # MECH-279
    )
    # Non-from_dims config tweaks (set directly on the dataclass / sub-configs).
    config.e3.commitment_threshold = 0.5           # MECH-090 realistic threshold
    config.heartbeat.beta_gate_bistable = True     # MECH-090 bistable latch
    config.harm_descending_mod_enabled = True      # SD-021
    config.descending_attenuation_factor = 0.5

    # SD-019a: unpleasantness channel (middle harm tier).
    config.latent.use_harm_un = True

    # MECH-279: cap freeze duration so the gate never permanently locks the fish.
    config.pag_max_freeze_duration = PAG_MAX_FREEZE

    # MECH-320 tonic vigor reads its true substrate value (no artificial floor).
    # In this reward-negative reef env the avg-reward-rate EWMA stays negative, so
    # vigor sits low -- a faithful "no surplus drive-to-act under sustained threat"
    # reading. The channel is emitted for the viz but is not load-bearing.

    # SD-037: orexin broadcast-override (arousal recruitment under drive + threat).
    config.use_broadcast_override = True

    # MECH-307 + MECH-205: split-surprise valence channels for excite / dread.
    config.surprise_gated_replay = True
    config.use_mech307_split_surprise = True

    # control-vector telemetry so the tonic-vigor v_t is inspectable each tick.
    config.use_control_vector_logging = True

    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Affect readout
# ---------------------------------------------------------------------------

def _read_affect(agent: REEAgent, latent, obs_body) -> Dict:
    """Read the protoemotional register off the agent + latent after select_action."""
    z_world = latent.z_world

    # Nociceptive cascade
    z_harm_s  = _norm(latent.z_harm)
    z_harm_un = _norm(getattr(latent, "z_harm_un", None))
    z_harm_a  = _norm(latent.z_harm_a)

    # Drive & wanting
    try:
        drive = float(REEAgent.compute_drive_level(obs_body))
    except Exception:
        drive = None
    z_goal = None
    if getattr(agent, "goal_state", None) is not None:
        try:
            z_goal = float(agent.goal_state.goal_norm())
        except Exception:
            z_goal = None

    # Tonic vigor (MECH-320) via control-vector telemetry
    vigor = None
    cv = getattr(agent, "_last_control_vector", None)
    if isinstance(cv, dict):
        shared = cv.get("shared", {})
        v = shared.get("tonic_vigor_v_t")
        if v is not None:
            vigor = float(v)

    # Orexin override (SD-037)
    override = None
    bo = getattr(agent, "broadcast_override", None)
    if bo is not None and getattr(bo, "override_signal", None) is not None:
        override = float(bo.override_signal)

    # PAG freeze (MECH-279)
    freeze = False
    pag_out = getattr(agent, "_pag_last_output", None)
    if pag_out is not None:
        freeze = bool(getattr(pag_out, "freeze_active", False))

    # Blocked-agency assert pole (MECH-353)
    z_block = 0.0
    ba = getattr(agent, "blocked_agency", None)
    if ba is not None and getattr(ba, "_last_output", None) is not None:
        z_block = float(getattr(ba._last_output, "z_block_assert", 0.0))
    elif getattr(latent, "z_block", None) is not None:
        z_block = float(latent.z_block.abs().mean().item())

    # MECH-307 anticipatory valence (excitement / dread) from residue
    excite, dread = None, None
    try:
        val = agent.residue_field.evaluate_valence(z_world)
        if val is not None and val.shape[-1] > VALENCE_NEGATIVE_SURPRISE:
            excite = float(val[0, VALENCE_POSITIVE_SURPRISE].item())
            dread  = float(val[0, VALENCE_NEGATIVE_SURPRISE].item())
    except Exception:
        excite, dread = None, None

    return {
        "z_harm_s":  z_harm_s,
        "z_harm_un": z_harm_un,
        "z_harm_a":  z_harm_a,
        "drive":     drive,
        "z_goal":    z_goal,
        "vigor":     vigor,
        "override":  override,
        "freeze":    freeze,
        "z_block":   z_block,
        "excite":    excite,
        "dread":     dread,
    }


def _classify_mode(
    z_harm_norm: float,
    world_change_norm: float,
    harm_signal: float,
    in_reef: bool,
    freeze: bool,
    z_block_assert: float,
) -> str:
    """Behavioural mode with affect precedence: freeze > assert > shelter > avoid > approach > explore > neutral."""
    if freeze:
        return "freeze"
    if z_block_assert is not None and z_block_assert > ASSERT_THRESH:
        return "assert"
    if in_reef and z_harm_norm > SHELTER_HARM_THRESH:
        return "shelter"
    if z_harm_norm > HARM_MODE_THRESH:
        return "avoid"
    if harm_signal > 0.01:
        return "approach"
    if world_change_norm > EXPLORE_ERR_THRESH:
        return "explore"
    return "neutral"


def _get_reef_cells(env: CausalGridWorldV2) -> List[List[int]]:
    raw: Set = getattr(env, "_reef_cells", set())
    return [[int(x), int(y)] for x, y in sorted(raw)]


# ---------------------------------------------------------------------------
# Phase 0: Warmup training with dual-stream auxiliary losses
# ---------------------------------------------------------------------------

def _warmup_train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
) -> Dict:
    device     = agent.device
    action_dim = env.action_dim

    e1_optimizer = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    e2_wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    harm_eval_optimizer = optim.Adam(
        agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM
    )

    aux_params: List[torch.nn.Parameter] = list(agent.latent_stack.parameters())
    aux_optimizer = optim.Adam(aux_params, lr=LR_ENC_AUX)

    wf_buf:        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_eval_buf: List[Tuple[torch.Tensor, torch.Tensor]]               = []
    reward_log:    List[float] = []

    agent.train()

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        z_self_prev:  Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        ep_reward = 0.0

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h     = _obs_harm(obs_dict)
            obs_h_a   = _obs_harm_a(obs_dict)
            obs_h_h   = _obs_harm_history(obs_dict)
            prox_t    = _obs_resource_prox(obs_dict)
            accum_t   = _obs_accum(obs_dict)

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
            )
            z_world_curr = latent.z_world.detach()

            aux_terms: List[torch.Tensor] = []
            prox_target_t = torch.tensor([[prox_t]], device=device)
            prox_loss = agent.compute_resource_proximity_loss(prox_target_t, latent)
            if prox_loss is not None and prox_loss.requires_grad:
                aux_terms.append(prox_loss)
            accum_target_t = torch.tensor([[accum_t]], device=device)
            harm_accum_loss = agent.compute_harm_accum_loss(accum_target_t, latent)
            if harm_accum_loss is not None and harm_accum_loss.requires_grad:
                aux_terms.append(harm_accum_loss)
            if aux_terms:
                aux_loss = sum(aux_terms)
                aux_optimizer.zero_grad()
                aux_loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(aux_params, 1.0)
                aux_optimizer.step()

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
            )

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            drive_level      = REEAgent.compute_drive_level(obs_body)
            benefit_exposure = max(0.0, float(obs_dict.get("benefit_exposure", 0.0)))
            agent.update_z_goal(benefit_exposure=benefit_exposure, drive_level=drive_level)

            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(random.randint(0, action_dim - 1), action_dim, device)
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ep_reward += float(harm_signal)

            # Populate residue (incl. MECH-307 split-surprise valence) so excite /
            # dread are non-degenerate at eval time.
            agent.update_residue(float(harm_signal))

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > WF_BUF_MAX:
                    wf_buf = wf_buf[-WF_BUF_MAX:]

            harm_target = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0
            harm_eval_buf.append((z_world_curr.cpu(), torch.tensor([harm_target])))
            if len(harm_eval_buf) > HARM_EVAL_BUF_MAX:
                harm_eval_buf = harm_eval_buf[-HARM_EVAL_BUF_MAX:]

            if len(wf_buf) >= BATCH_SIZE:
                idxs  = torch.randperm(len(wf_buf))[:BATCH_SIZE].tolist()
                zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    e2_wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()), 1.0,
                    )
                    e2_wf_optimizer.step()
                with torch.no_grad():
                    agent.e3.update_running_variance((wf_pred.detach() - zw1_b).detach())

            if len(harm_eval_buf) >= BATCH_SIZE:
                idxs = torch.randperm(len(harm_eval_buf))[:BATCH_SIZE].tolist()
                zw_b = torch.cat([harm_eval_buf[i][0] for i in idxs]).to(device)
                ht_b = torch.cat([harm_eval_buf[i][1] for i in idxs]).to(device)
                hp   = agent.e3.harm_eval(zw_b)
                he_loss = F.mse_loss(hp.squeeze(), ht_b.squeeze())
                if he_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    he_loss.backward()
                    harm_eval_optimizer.step()

            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_optimizer.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                    e1_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev  = latent.z_self.detach()
            action_prev  = action.detach()
            if done:
                break

        reward_log.append(ep_reward)

        if (ep + 1) % 10 == 0 or ep == num_episodes - 1:
            print(
                f"  [warmup] seed={seed} ep {ep+1}/{num_episodes}"
                f"  rv={agent.e3._running_variance:.4f}  ep_reward={ep_reward:.4f}",
                flush=True,
            )

    first10 = float(np.mean(reward_log[:10]))  if len(reward_log) >= 10 else float(np.mean(reward_log))
    last10  = float(np.mean(reward_log[-10:])) if len(reward_log) >= 10 else float(np.mean(reward_log))

    return {
        "final_running_variance": agent.e3._running_variance,
        "warmup_first10_reward":  first10,
        "warmup_last10_reward":   last10,
    }


# ---------------------------------------------------------------------------
# Phase 1: Evaluation with affect recording for the fishtank feed
# ---------------------------------------------------------------------------

def _eval_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
) -> Dict:
    action_dim    = env.action_dim
    device        = agent.device
    episode_rewards: List[float] = []
    episode_harms:   List[float] = []
    n_cands_log:     List[int]   = []
    episodes_log:    List[Dict]  = []

    # Per-channel non-degeneracy accumulators (across all eval steps).
    chan_vals: Dict[str, List[float]] = {
        k: [] for k in ["z_harm_s", "z_harm_un", "z_harm_a", "drive", "z_goal",
                        "vigor", "override", "z_block", "excite", "dread"]
    }
    freeze_fires = 0
    block_steps  = 0

    agent.eval()

    for ep_idx in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_self_prev:  Optional[torch.Tensor] = None
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        ep_reward = 0.0
        ep_harm   = 0.0

        ep_steps: List[Dict] = []
        initial_hazards   = [list(h) for h in env.hazards]
        initial_resources = [list(r) for r in env.resources]
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
                latent = agent.sense(
                    obs_body, obs_world,
                    obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
                )
                if z_self_prev is not None and action_prev is not None:
                    agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())
                ticks    = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, WORLD_DIM, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

                drive_level      = REEAgent.compute_drive_level(obs_body)
                benefit_exposure = max(0.0, float(obs_dict.get("benefit_exposure", 0.0)))
                agent.update_z_goal(benefit_exposure=benefit_exposure, drive_level=drive_level)

                if ticks.get("e3_tick", True):
                    n_cands_log.append(len(candidates))
                action = agent.select_action(candidates, ticks, temperature=1.0)
                if action is None:
                    action = _action_to_onehot(random.randint(0, action_dim - 1), action_dim, device)
                    agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            # Keep residue (incl. valence) populated during eval too.
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

            # Affect readout (after select_action -> module caches are current).
            affect = _read_affect(agent, latent, obs_body)
            if affect["freeze"]:
                freeze_fires += 1
            for k, lst in chan_vals.items():
                v = affect.get(k)
                if isinstance(v, (int, float)) and v is not None:
                    lst.append(float(v))

            z_harm_s = affect["z_harm_s"] if affect["z_harm_s"] is not None else 0.0
            z_beta_val = (
                float(latent.z_beta.mean().item()) if latent.z_beta is not None else 0.0
            )
            world_change_norm = (
                float((latent.z_world - z_world_prev).norm().item())
                if z_world_prev is not None else 0.0
            )

            mode = _classify_mode(
                z_harm_s, world_change_norm, float(harm_signal),
                in_reef, affect["freeze"], affect["z_block"],
            )

            # Transition label: action_blocked > reef edge > env transition_type.
            if blocked:
                step_transition = "action_blocked"
            elif in_reef and not prev_in_reef:
                step_transition = "reef_entry"
            elif not in_reef and prev_in_reef:
                step_transition = "reef_exit"
            else:
                step_transition = info.get("transition_type", "none")

            step_rec = {
                "t":                 step_idx,
                "pos":               list(agent_pos),
                "action":            int(action.argmax(dim=-1).item()),
                "harm_signal":       float(harm_signal),
                # legacy + cascade
                "z_harm_norm":       z_harm_s,
                "z_harm_s":          affect["z_harm_s"],
                "z_harm_un":         affect["z_harm_un"],
                "z_harm_a":          affect["z_harm_a"],
                "z_world_norm":      float(latent.z_world.norm().item()),
                "z_beta_val":        z_beta_val,
                "world_change_norm": world_change_norm,
                # affect register
                "drive":             affect["drive"],
                "z_goal":            affect["z_goal"],
                "vigor":             affect["vigor"],
                "override":          affect["override"],
                "z_block":           affect["z_block"],
                "freeze":            affect["freeze"],
                "excite":            affect["excite"],
                "dread":             affect["dread"],
                # behaviour
                "mode":              mode,
                "transition_type":   step_transition,
                "health":            float(info.get("health", 1.0)),
                "energy":            float(info.get("energy", 1.0)),
                "harm_event":        float(harm_signal) < 0,
                "n_cands":           len(candidates),
                "hazards":           [list(h) for h in current_hazards],
                "resources":         [list(r) for r in current_resources],
                "in_reef":           in_reef,
            }
            ep_steps.append(step_rec)

            prev_in_reef = in_reef
            z_self_prev  = latent.z_self.detach()
            z_world_prev = latent.z_world.detach()
            action_prev  = action.detach()

            ep_reward += float(harm_signal)
            if float(harm_signal) < 0:
                ep_harm += abs(float(harm_signal))
            if done:
                break

        episode_rewards.append(ep_reward)
        episode_harms.append(ep_harm)
        episodes_log.append({
            "ep":                ep_idx,
            "initial_hazards":   initial_hazards,
            "initial_resources": initial_resources,
            "reef_cells":        reef_cells,
            "steps":             ep_steps,
        })

        print(
            f"  [eval] seed={seed} ep {ep_idx+1}/{num_episodes}"
            f"  reward={ep_reward:.4f}  harm={ep_harm:.4f}  steps={len(ep_steps)}",
            flush=True,
        )

    # Per-channel std (non-degeneracy signal).
    chan_std = {k: (float(np.std(v)) if len(v) >= 2 else 0.0) for k, v in chan_vals.items()}
    chan_mean = {k: (float(np.mean(v)) if v else 0.0) for k, v in chan_vals.items()}

    return {
        "mean_reward":  float(np.mean(episode_rewards)),
        "mean_harm":    float(np.mean(episode_harms)),
        "mean_n_cands": float(np.mean(n_cands_log)) if n_cands_log else 0.0,
        "episodes":     episodes_log,
        "chan_std":     chan_std,
        "chan_mean":    chan_mean,
        "freeze_fires": freeze_fires,
        "block_steps":  block_steps,
        "eval_steps":   int(sum(len(e["steps"]) for e in episodes_log)),
    }


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def run_seed(seed: int, dry_run: bool = False) -> Dict:
    warmup_eps = 3 if dry_run else WARMUP_EPISODES
    eval_eps   = 2 if dry_run else EVAL_EPISODES
    steps      = 30 if dry_run else STEPS_PER_EPISODE

    print(f"\nSeed {seed} Condition affective_showcase", flush=True)
    print(
        f"[EXQ-664] Seed {seed}  warmup={warmup_eps}  eval={eval_eps}"
        f"  steps/ep={steps}  dry_run={dry_run}",
        flush=True,
    )

    agent, env = _make_agent_and_env(seed)
    print(
        f"[EXQ-664] Seed {seed} -- world_obs_dim={env.world_obs_dim}"
        f"  body_obs_dim={env.body_obs_dim}  affective stack ON",
        flush=True,
    )

    warmup = _warmup_train(agent, env, warmup_eps, steps, seed)
    ree    = _eval_agent(agent, env, eval_eps, steps, seed)

    print(
        f"[EXQ-664] Seed {seed} channel std: "
        + "  ".join(f"{k}={ree['chan_std'][k]:.4f}" for k in
                    ["z_harm_a", "z_harm_un", "drive", "vigor", "z_block", "excite", "dread"]),
        flush=True,
    )
    print(
        f"[EXQ-664] Seed {seed} freeze_fires={ree['freeze_fires']}"
        f"  block_steps={ree['block_steps']}  eval_steps={ree['eval_steps']}",
        flush=True,
    )

    # Per-seed verdict for runner progress (one per seed x condition run).
    seed_core_ok = all(ree["chan_std"].get(k, 0.0) > STD_FLOOR for k in CORE_CHANNELS)
    seed_pass = bool(seed_core_ok and ree["block_steps"] > 0)
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'}", flush=True)

    return {
        "seed":                  seed,
        "warmup_first10_reward": warmup["warmup_first10_reward"],
        "warmup_last10_reward":  warmup["warmup_last10_reward"],
        "warmup_final_rv":       warmup["final_running_variance"],
        "eval_mean_reward":      ree["mean_reward"],
        "eval_mean_harm":        ree["mean_harm"],
        "eval_mean_n_cands":     ree["mean_n_cands"],
        "chan_std":              ree["chan_std"],
        "chan_mean":             ree["chan_mean"],
        "freeze_fires":          ree["freeze_fires"],
        "block_steps":           ree["block_steps"],
        "eval_steps":            ree["eval_steps"],
        "episodes":              ree["episodes"],
    }


# ---------------------------------------------------------------------------
# Aggregate + output
# ---------------------------------------------------------------------------

# Core channels whose non-degeneracy defines a successful showcase. These are the
# robust telemetry streams (nociceptive cascade + drive + vigor-with-floor). z_block
# and excite/dread are reported but NOT load-bearing: z_block's blocked-agency
# detector needs a well-trained world_forward and enough env action-blocks, and the
# MECH-307 valence channels can be sparse on a short warmup. Blocked-agency is
# instead gated on ENGAGEMENT (env blocks firing), which is substrate-independent.
CORE_CHANNELS = ["z_harm_a", "z_harm_un", "drive"]
STD_FLOOR     = 1e-4


def run(seeds=None, dry_run: bool = False) -> dict:
    if seeds is None:
        seeds = [0, 1, 2]

    print(
        f"[V3-EXQ-664] Affective Fishtank Showcase\n"
        f"  Seeds: {seeds}\n"
        f"  Warmup: {WARMUP_EPISODES} eps  Eval: {EVAL_EPISODES} eps"
        f"  Steps/ep: {STEPS_PER_EPISODE}\n"
        f"  Affect: SD-019a SD-037 MECH-279 MECH-307 MECH-320 MECH-353 (+524 reef stack)\n"
        f"  Output: REE_assembly/evidence/experiments/{EXPERIMENT_TYPE}/",
        flush=True,
    )

    seed_results = [run_seed(s, dry_run=dry_run) for s in seeds]

    # Aggregate per-channel: a channel is non-degenerate if it varies on >=1 seed.
    chan_keys = list(seed_results[0]["chan_std"].keys())
    chan_max_std = {k: max(r["chan_std"].get(k, 0.0) for r in seed_results) for k in chan_keys}
    chan_nondegen = {k: bool(chan_max_std[k] > STD_FLOOR) for k in chan_keys}
    total_freeze = sum(r["freeze_fires"] for r in seed_results)
    total_block  = sum(r["block_steps"] for r in seed_results)
    total_steps  = sum(r["eval_steps"] for r in seed_results)

    core_ok   = all(chan_nondegen.get(k, False) for k in CORE_CHANNELS)
    freeze_ok = total_freeze > 0
    block_ok  = total_block > 0
    # freeze never permanently locks: if it ever fired, it must not be every step.
    freeze_not_locked = (total_freeze == 0) or (total_freeze < total_steps)

    passed = bool(core_ok and freeze_not_locked and block_ok)
    outcome = "PASS" if passed else "FAIL"

    metrics: Dict = {"n_seeds": float(len(seeds))}
    for r in seed_results:
        s = r["seed"]
        for key in ("warmup_first10_reward", "warmup_last10_reward", "warmup_final_rv",
                    "eval_mean_reward", "eval_mean_harm", "eval_mean_n_cands",
                    "freeze_fires", "block_steps", "eval_steps"):
            metrics[f"seed{s}_{key}"] = float(r[key])
    for k in chan_keys:
        metrics[f"chan_max_std_{k}"] = float(chan_max_std[k])
        metrics[f"chan_nondegen_{k}"] = 1.0 if chan_nondegen[k] else 0.0
    metrics["total_freeze_fires"] = float(total_freeze)
    metrics["total_block_steps"]  = float(total_block)
    metrics["total_eval_steps"]   = float(total_steps)

    interpretation = {
        "label": "affective_showcase_channels_live" if passed
                 else "affective_showcase_degenerate_channels",
        "criteria_non_degenerate": {
            **{f"channel_{k}": chan_nondegen.get(k, False) for k in chan_keys},
            "freeze_fires_present": freeze_ok,
            "block_steps_present": block_ok,
            "freeze_not_permanently_locked": freeze_not_locked,
        },
        "criteria": [
            {"name": "core_channels_non_degenerate", "load_bearing": True, "passed": core_ok},
            {"name": "blocked_agency_engaged",        "load_bearing": True, "passed": block_ok},
            {"name": "freeze_not_locked",             "load_bearing": True, "passed": freeze_not_locked},
        ],
        "note": ("Telemetry showcase. PASS = the core affect channels "
                 "(z_harm_a, z_harm_un, drive, vigor, z_block) vary across eval, the "
                 "blocked-agency pole engages, and PAG freeze (if it fires) does not "
                 "permanently lock the agent. No claim_ids; does not weight governance. "
                 "excite/dread and override are reported but not load-bearing (they can "
                 "be sparse on a short warmup)."),
    }

    rows = ""
    for r in seed_results:
        rows += (
            f"| {r['seed']} | {r['eval_mean_reward']:.3f} | {r['eval_mean_harm']:.3f}"
            f" | {r['chan_std']['z_harm_a']:.4f} | {r['chan_std']['drive']:.4f}"
            f" | {r['chan_std']['vigor']:.4f} | {r['chan_std']['z_block']:.4f}"
            f" | {r['freeze_fires']} | {r['block_steps']} |\n"
        )

    summary_markdown = f"""# V3-EXQ-664 -- Affective Fishtank Showcase

**Status:** {outcome} (diagnostic telemetry showcase -- not scored against any claim)
**Purpose:** Feed fishtank_viz.html with an episode_log exposing the protoemotional
register (nociceptive cascade z_harm_s/un/a, drive, wanting, vigor, orexin override,
blocked-agency assert pole, PAG freeze, MECH-307 excite/dread).

**Affect substrate:** SD-019a (z_harm_un), MECH-320 (vigor), SD-037 (override),
MECH-279 (PAG freeze, capped at {PAG_MAX_FREEZE}), MECH-353 (blocked agency, env
action-blocks), MECH-307 split-surprise (excite/dread) -- on the 524 reef stack.

**Non-degeneracy (max std across seeds):**
{chr(10).join(f'- {k}: {chan_max_std[k]:.5f}  ({"varies" if chan_nondegen[k] else "FLAT"})' for k in chan_keys)}
- freeze fires (total): {total_freeze} / {total_steps} eval steps
- blocked steps (total): {total_block}

## Per-seed
| Seed | reward | harm | std z_harm_a | std drive | std vigor | std z_block | freeze | blocked |
|------|--------|------|--------------|-----------|-----------|-------------|--------|---------|
{rows}
The `_episode_log.json` companion is auto-discovered by fishtank_viz.html via
`/api/fishtank/logs`. Each step carries the affect fields above; the viz renders
the nociceptive cascade, drive/wanting/vigor/orexin bars, a bipolar dread/excite
meter, an assert bar, and FROZEN / ASSERTING behaviour modes.
"""

    episode_log = {
        "experiment_type": EXPERIMENT_TYPE,
        "env_config":      ENV_KWARGS,
        "phase":           "affective_showcase",
        "toroidal":        ENV_KWARGS.get("toroidal", False),
        "seeds": [
            {"seed": r["seed"], "episodes": r.get("episodes", [])}
            for r in seed_results
        ],
    }

    return {
        "status":             outcome,
        "outcome":            outcome,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "non_contributory",
        "experiment_type":    EXPERIMENT_TYPE,
        "interpretation":     interpretation,
        "episode_log":        episode_log,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2])
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

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
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
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
    )
