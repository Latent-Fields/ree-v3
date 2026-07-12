#!/opt/local/bin/python3
"""
V3-EXQ-556 -- Module-init swap diagnostic (localises which submodule's
seed-7 init enables the V3-EXQ-555 C0/C2 escape from monostrategy).

Claims: [] (monostrategy-investigation diagnostic; no substrate claim under test)

Purpose (evidence_direction_note: diagnostic)
---------------------------------------------
V3-EXQ-555 (completed 2026-05-12T05:21Z) factored seed=(env_seed, agent_seed)
into a 2x2 cell sweep and found:
    C0 (env=7,  agent=7)  entropy = 0.679
    C1 (env=7,  agent=42) entropy = 0.000
    C2 (env=42, agent=7)  entropy = 0.690
    C3 (env=42, agent=42) entropy = 0.000

Interpretation: AGENT-side seed=7 init is what enables diverse policies;
env_seed is irrelevant. The substrate has capacity for diverse policies --
most seeds land in a monostrategy basin; seed=7 escapes it.

V3-EXQ-556 now drills one level deeper: which SPECIFIC submodule's
weight-initialization is responsible for the escape? Take the
agent_seed=42 baseline (monostrategy, entropy=0). One module at a time,
override its init RNG to seed=7. All other modules stay at seed=42.
Identify which single-module swap is sufficient to lift entropy back to
~0.68.

Eight arms (env_seed=42 fixed across all arms; agent_seed by module)
--------------------------------------------------------------------
    ARM_BASELINE_42:    all modules seed=42  (control; expect entropy ~0)
    ARM_BASELINE_7:     all modules seed=7   (control; expect entropy ~0.68)
    ARM_SWAP_LATENT:    latent_stack seed=7, rest seed=42
    ARM_SWAP_E1:        e1 seed=7, rest seed=42
    ARM_SWAP_E2:        e2 seed=7, rest seed=42
    ARM_SWAP_RESIDUE:   residue_field seed=7, rest seed=42
    ARM_SWAP_E3:        e3 seed=7, rest seed=42
    ARM_SWAP_HIPPO:     hippocampal seed=7, rest seed=42

Single seed-pair per cell -- the seed pair IS the variable; no per-cell
replication. Estimated ~12-15 min per arm at P0=40, P1=60, 200 steps/ep
(matches V3-EXQ-555 wallclock); ~100-120 min total.

Implementation: per-module __init__ seeding via class-level monkey-patch
---------------------------------------------------------------------
REEAgent constructs its submodules sequentially in __init__ (latent_stack,
e1, e2, residue_field, e3, hippocampal). torch.nn.Linear / GRU / LSTM
weight inits consume torch's global RNG state. To control per-module
seeds, we patch each module's class __init__ to reset torch.manual_seed
(and numpy + random for completeness) at the start of __init__, reading
from a module-name -> seed mapping configured by the experiment harness.

After REEAgent construction we restore the original __init__ methods so
the patch does not leak into subsequent cells or other experiments. The
patch IS class-level so it must be applied + cleaned up around the
single REEAgent(...) call site.

Verification (smoke-test invariants)
------------------------------------
After ARM_BASELINE_42, ARM_BASELINE_7, and ARM_SWAP_E1 are each built,
inspect each agent's first-parameter checksum per module:
  - latent_stack first param, e1 first param, e2 first param,
    residue_field first param, e3.harm_eval_head first param, hippocampal
    first param.
Expected:
  ARM_BASELINE_42.<mod>     == ARM_SWAP_E1.<mod>      for mod in {latent, e2, residue, e3, hippo}
  ARM_BASELINE_7.e1         == ARM_SWAP_E1.e1
  ARM_BASELINE_42.e1        != ARM_SWAP_E1.e1
i.e. SWAP_E1 inherits E1 weights from a seed=7 init while keeping the
other modules at seed=42 init.

Pre-registered interpretation grid (6 rows; embedded in
evidence_direction_note)
--------------------------------------------------------------------------
  R1 single_module_sufficient:  ONE swap arm shows entropy >= 0.30
    -> that module's seed-7 init is THE basin-determining init. Routes to
       deep diagnostic of that module's init distribution (which weight
       layer? what is structurally different at seed=7?).
  R2 multi_module_conjunctive:  NO single-module swap >= 0.30 AND
                                 ARM_BASELINE_7 still ~ 0.68
    -> escape requires multi-module seed-7 conjunction. Routes to a
       pair-swap follow-up (V3-EXQ-558 candidate: enumerate 2-module swap
       combinations).
  R3 latent_stack_only:  Only ARM_SWAP_LATENT lifts -> z-encoder init drives diversity.
  R4 e3_only:            Only ARM_SWAP_E3 lifts -> value-head / selector init drives.
  R5 hippocampal_only:   Only ARM_SWAP_HIPPO lifts -> action-object decoder init drives.
  R6 replication_failure: ARM_BASELINE_7 fails to reproduce entropy ~ 0.68
    -> something about the patching disturbs the baseline; debug the
       patching mechanism before drawing conclusions from the swap arms.

R3/R4/R5 are specialisations of R1 (single-module sufficient with a
specific module). The generic R1 is recorded whenever exactly one
single-module swap lifts and it is NOT latent / e3 / hippocampal (i.e.
e1, e2, residue, or unexpected).

experiment_purpose=diagnostic (decomposes a known anomaly; no falsifiable
substrate claim under test).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from experiment_protocol import emit_outcome
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# Submodule classes that get per-module seeded inits.
from ree_core.latent.stack import LatentStack
from ree_core.predictors.e1_deep import E1DeepPredictor
from ree_core.predictors.e2_fast import E2FastPredictor
from ree_core.residue.field import ResidueField
from ree_core.predictors.e3_selector import E3TrajectorySelector
from ree_core.hippocampal.module import HippocampalModule
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_556_module_init_swap"
QUEUE_ID = "V3-EXQ-556"
CLAIM_IDS: List[str] = []  # monostrategy-investigation diagnostic
EXPERIMENT_PURPOSE = "diagnostic"

# Eight arms: (arm_label, module_seed_overrides dict).
# Default base agent_seed is 42 across all modules; override entries set
# the named submodule to seed=7. Env_seed is fixed at 42 in all arms.
BASE_SEED = 42
SWAP_SEED = 7
ENV_SEED = 42

# Module-name -> (class_object, attribute_on_agent) for the six core
# submodules. Attribute is the name on agent.<attr> we read for the
# verification checksum.
MODULE_REGISTRY: List[Tuple[str, type, str]] = [
    ("latent_stack", LatentStack,           "latent_stack"),
    ("e1",           E1DeepPredictor,       "e1"),
    ("e2",           E2FastPredictor,       "e2"),
    ("residue",      ResidueField,          "residue_field"),
    ("e3",           E3TrajectorySelector,  "e3"),
    ("hippocampal",  HippocampalModule,     "hippocampal"),
]

ARMS: List[Tuple[str, Dict[str, int]]] = [
    ("ARM_BASELINE_42", {}),
    ("ARM_BASELINE_7",  {name: SWAP_SEED for name, _, _ in MODULE_REGISTRY}),
    ("ARM_SWAP_LATENT", {"latent_stack": SWAP_SEED}),
    ("ARM_SWAP_E1",     {"e1":           SWAP_SEED}),
    ("ARM_SWAP_E2",     {"e2":           SWAP_SEED}),
    ("ARM_SWAP_RESIDUE",{"residue":      SWAP_SEED}),
    ("ARM_SWAP_E3",     {"e3":           SWAP_SEED}),
    ("ARM_SWAP_HIPPO",  {"hippocampal":  SWAP_SEED}),
]

# Phased schedule -- matches V3-EXQ-555 / V3-EXQ-552 ARM_NORMAL canonical pattern.
P0_TRAIN_EPISODES = 40
P1_EVAL_EPISODES = 60
STEPS_PER_EPISODE = 200

# Latent dims (match V3-EXQ-555).
WORLD_DIM = 32
SELF_DIM = 32
HARM_DIM = 32
HARM_A_DIM = 16
HARM_HISTORY_LEN = 10

WF_BUF_MAX = 2000
HARM_EVAL_BUF_MAX = 2000
BATCH_SIZE = 32

LR_E1 = 1e-4
LR_E2_WF = 3e-4
LR_E3_HARM = 1e-3
LR_ENC_AUX = 5e-4

# Env config: VERBATIM copy of V3-EXQ-555 ENV_KWARGS.
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


# ---------------------------------------------------------------------------
# Per-module init seeding (class-level monkey-patch)
# ---------------------------------------------------------------------------

def _seed_all(seed: int) -> None:
    """Seed torch + numpy + random global RNGs to `seed`.

    CUDA RNG seeded too if available; ree-v3 runs on CPU on the canonical
    macbook and ree-cloud machines, but the guard is cheap.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _patch_module_class_init(cls: type, name: str,
                             overrides: Dict[str, int],
                             default_seed: int) -> Callable:
    """Wrap cls.__init__ so that the first thing it does is _seed_all() to
    the configured seed for this module.

    Returns the original __init__ so the caller can restore it after the
    REEAgent construction call site completes.
    """
    original_init = cls.__init__

    def patched(self, *a, **kw):
        seed = overrides.get(name, default_seed)
        _seed_all(seed)
        return original_init(self, *a, **kw)

    cls.__init__ = patched
    return original_init


def _build_agent_with_module_seeds(env: CausalGridWorldV2,
                                   overrides: Dict[str, int],
                                   default_seed: int) -> REEAgent:
    """Construct REEAgent with per-module init seeds.

    Args:
      env: pre-constructed env (so we can read env.action_dim / dims).
      overrides: {module_name: seed}; unmentioned modules use default_seed.
      default_seed: base seed for any module without an override.

    Returns:
      REEAgent with each registered submodule initialised under its
      configured seed.
    """
    # Patch each registered submodule class so its __init__ reseeds first.
    originals: List[Tuple[type, Callable]] = []
    try:
        for name, cls, _attr in MODULE_REGISTRY:
            orig = _patch_module_class_init(cls, name, overrides, default_seed)
            originals.append((cls, orig))

        # Re-seed at default_seed BEFORE REEAgent() so any pre-submodule
        # work in REEAgent.__init__ (super().__init__() call, config
        # capture, etc.) is reproducible at default_seed.
        _seed_all(default_seed)

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
        )
        config.e3.commitment_threshold = 0.5
        config.heartbeat.beta_gate_bistable = True
        config.harm_descending_mod_enabled = True
        config.descending_attenuation_factor = 0.5

        agent = REEAgent(config)
        return agent
    finally:
        # Restore original __init__ methods so the patch never leaks.
        for cls, orig in originals:
            cls.__init__ = orig


def _make_env(env_seed: int) -> CausalGridWorldV2:
    """Construct env with env_seed. Env uses np.random.default_rng(env_seed),
    independent of torch / numpy global RNGs.
    """
    return CausalGridWorldV2(seed=env_seed, **ENV_KWARGS)


# ---------------------------------------------------------------------------
# Per-module checksum (smoke verification + per-cell fingerprint)
# ---------------------------------------------------------------------------

def _first_param_checksum(module: torch.nn.Module) -> Tuple[float, List[float]]:
    """Return (sum, first-8-flat-elems) of the first INFORMATIVE leaf
    parameter of `module`.

    "Informative" means: at least 8 elements AND non-zero L1 magnitude
    (constant-init or zero-init biases like LatentStack's
    split_encoder.self_precision_logit are skipped because they are
    seed-independent and produce a misleading fingerprint).

    The fingerprint must be RNG-derived to verify the patch worked.
    """
    for p in module.parameters():
        flat = p.detach().flatten()
        if flat.numel() < 8:
            continue
        if float(flat.abs().sum().item()) < 1e-12:
            # Constant/zero init -- seed-independent, useless for verification.
            continue
        n = min(8, flat.numel())
        head = [float(flat[i].item()) for i in range(n)]
        return (float(flat.sum().item()), head)
    return (0.0, [])


def _per_module_signature(agent: REEAgent) -> Dict[str, Dict]:
    """Read first-param checksum for every registered submodule on agent."""
    sigs: Dict[str, Dict] = {}
    for name, _cls, attr in MODULE_REGISTRY:
        mod = getattr(agent, attr, None)
        if mod is None:
            sigs[name] = {"sum": None, "head": []}
            continue
        s, head = _first_param_checksum(mod)
        sigs[name] = {"sum": s, "head": head}
    return sigs


# ---------------------------------------------------------------------------
# Helpers (mirror V3-EXQ-555)
# ---------------------------------------------------------------------------

def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _obs_harm(obs_dict):
    return obs_dict.get("harm_obs")


def _obs_harm_a(obs_dict):
    return obs_dict.get("harm_obs_a")


def _obs_harm_history(obs_dict):
    return obs_dict.get("harm_history")


def _obs_accum(obs_dict) -> float:
    v = obs_dict.get("accumulated_harm")
    return float(v) if v is not None else 0.0


def _obs_resource_prox(obs_dict) -> float:
    rv = obs_dict.get("resource_field_view")
    if rv is None:
        return 0.0
    if isinstance(rv, torch.Tensor):
        return float(rv.max().item())
    return float(np.max(rv))


def _shannon_entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return ent


# ---------------------------------------------------------------------------
# Training + eval loop (matches V3-EXQ-555 verbatim)
# ---------------------------------------------------------------------------

def _run_one_phase(
    agent: REEAgent,
    env: CausalGridWorldV2,
    phase_label: str,
    num_episodes: int,
    steps_per_episode: int,
    train: bool,
    optimizers_and_params: Optional[Dict],
    rng_module,
    action_count_window: Optional[Dict[int, int]] = None,
) -> Dict:
    device = agent.device
    action_dim = env.action_dim

    if train:
        assert optimizers_and_params is not None, \
            "train=True requires optimizers_and_params"
        e1_optimizer = optimizers_and_params["e1_optimizer"]
        e2_wf_optimizer = optimizers_and_params["e2_wf_optimizer"]
        harm_eval_optimizer = optimizers_and_params["harm_eval_optimizer"]
        aux_optimizer = optimizers_and_params["aux_optimizer"]
        aux_params = optimizers_and_params["aux_params"]
        wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = \
            optimizers_and_params["wf_buf"]
        harm_eval_buf: List[Tuple[torch.Tensor, torch.Tensor]] = \
            optimizers_and_params["harm_eval_buf"]
        agent.train()
    else:
        agent.eval()

    n_total_actions = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        for _step in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h = _obs_harm(obs_dict)
            obs_h_a = _obs_harm_a(obs_dict)
            obs_h_h = _obs_harm_history(obs_dict)
            prox_t = _obs_resource_prox(obs_dict)
            accum_t = _obs_accum(obs_dict)

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a,
                obs_harm_history=obs_h_h,
            )
            z_world_curr = latent.z_world.detach()

            if train:
                aux_terms: List[torch.Tensor] = []
                prox_target_t = torch.tensor([[prox_t]], device=device)
                prox_loss = agent.compute_resource_proximity_loss(
                    prox_target_t, latent,
                )
                if prox_loss is not None and prox_loss.requires_grad:
                    aux_terms.append(prox_loss)
                accum_target_t = torch.tensor([[accum_t]], device=device)
                harm_accum_loss = agent.compute_harm_accum_loss(
                    accum_target_t, latent,
                )
                if harm_accum_loss is not None and harm_accum_loss.requires_grad:
                    aux_terms.append(harm_accum_loss)
                if aux_terms:
                    aux_loss = sum(aux_terms)
                    aux_optimizer.zero_grad()
                    aux_loss.backward(retain_graph=False)
                    torch.nn.utils.clip_grad_norm_(aux_params, 1.0)
                    aux_optimizer.step()

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(
                    z_self_prev, action_prev, latent.z_self.detach(),
                )

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a,
                obs_harm_history=obs_h_h,
            )

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            drive_level = REEAgent.compute_drive_level(obs_body)
            benefit_exposure = max(
                0.0, float(obs_dict.get("benefit_exposure", 0.0)),
            )
            agent.update_z_goal(
                benefit_exposure=benefit_exposure,
                drive_level=drive_level,
            )

            proposed_action = agent.select_action(candidates, ticks, temperature=1.0)
            if proposed_action is None:
                proposed_action = _action_to_onehot(
                    rng_module.randint(0, action_dim - 1), action_dim, device,
                )
                agent._last_action = proposed_action

            action = proposed_action
            n_total_actions += 1

            if action_count_window is not None:
                a_idx = int(action[0].argmax().item())
                action_count_window[a_idx] = (
                    action_count_window.get(a_idx, 0) + 1
                )

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            if train:
                if z_world_prev is not None and action_prev is not None:
                    wf_buf.append(
                        (z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()),
                    )
                    if len(wf_buf) > WF_BUF_MAX:
                        del wf_buf[:len(wf_buf) - WF_BUF_MAX]

                harm_target = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0
                harm_eval_buf.append(
                    (z_world_curr.cpu(), torch.tensor([harm_target])),
                )
                if len(harm_eval_buf) > HARM_EVAL_BUF_MAX:
                    del harm_eval_buf[:len(harm_eval_buf) - HARM_EVAL_BUF_MAX]

                if len(wf_buf) >= BATCH_SIZE:
                    idxs = torch.randperm(len(wf_buf))[:BATCH_SIZE].tolist()
                    zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                    a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                    zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                    wf_pred = agent.e2.world_forward(zw_b, a_b)
                    wf_loss = F.mse_loss(wf_pred, zw1_b)
                    if wf_loss.requires_grad:
                        e2_wf_optimizer.zero_grad()
                        wf_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(agent.e2.world_transition.parameters())
                            + list(agent.e2.world_action_encoder.parameters()),
                            1.0,
                        )
                        e2_wf_optimizer.step()
                    with torch.no_grad():
                        agent.e3.update_running_variance(
                            (wf_pred.detach() - zw1_b).detach(),
                        )

                if len(harm_eval_buf) >= BATCH_SIZE:
                    idxs = torch.randperm(len(harm_eval_buf))[:BATCH_SIZE].tolist()
                    zw_b = torch.cat([harm_eval_buf[i][0] for i in idxs]).to(device)
                    t_b = torch.stack(
                        [harm_eval_buf[i][1] for i in idxs],
                    ).to(device).view(-1, 1)
                    he_pred = agent.e3.harm_eval_head(zw_b)
                    he_loss = F.mse_loss(he_pred, t_b)
                    if he_loss.requires_grad:
                        harm_eval_optimizer.zero_grad()
                        he_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            agent.e3.harm_eval_head.parameters(), 1.0,
                        )
                        harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()

            if done:
                break

    return {
        "phase_label": phase_label,
        "n_total_actions": n_total_actions,
    }


def _run_arm(arm_label: str, module_overrides: Dict[str, int]) -> Dict:
    """Run one arm and return P1 action-class entropy + fingerprints."""
    env = _make_env(ENV_SEED)
    agent = _build_agent_with_module_seeds(
        env=env, overrides=module_overrides, default_seed=BASE_SEED,
    )
    device = agent.device

    per_module_sig = _per_module_signature(agent)

    e1_optimizer = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    e2_wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    harm_eval_optimizer = optim.Adam(
        agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM,
    )
    aux_params = list(agent.latent_stack.parameters())
    aux_optimizer = optim.Adam(aux_params, lr=LR_ENC_AUX)

    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_eval_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []

    optimizers_and_params = {
        "e1_optimizer": e1_optimizer,
        "e2_wf_optimizer": e2_wf_optimizer,
        "harm_eval_optimizer": harm_eval_optimizer,
        "aux_optimizer": aux_optimizer,
        "aux_params": aux_params,
        "wf_buf": wf_buf,
        "harm_eval_buf": harm_eval_buf,
    }

    rng_module = random.Random(ENV_SEED)

    p0_diag = _run_one_phase(
        agent=agent, env=env, phase_label="P0",
        num_episodes=P0_TRAIN_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        train=True, optimizers_and_params=optimizers_and_params,
        rng_module=rng_module, action_count_window=None,
    )
    print(
        f"  [train] arm={arm_label} P0 {P0_TRAIN_EPISODES}/{P0_TRAIN_EPISODES} "
        f"n_total={p0_diag['n_total_actions']}",
        flush=True,
    )

    action_count_window: Dict[int, int] = {}
    p1_diag = _run_one_phase(
        agent=agent, env=env, phase_label="P1",
        num_episodes=P1_EVAL_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        train=False, optimizers_and_params=None,
        rng_module=rng_module, action_count_window=action_count_window,
    )
    p1_entropy = _shannon_entropy(action_count_window)
    print(
        f"  [eval]  arm={arm_label} P1 {P1_EVAL_EPISODES}/{P1_EVAL_EPISODES} "
        f"entropy={p1_entropy:.4f} n_actions={sum(action_count_window.values())}",
        flush=True,
    )

    return {
        "arm": arm_label,
        "module_overrides": module_overrides,
        "p1_action_class_counts": action_count_window,
        "p1_action_class_entropy": p1_entropy,
        "p1_n_actions": sum(action_count_window.values()),
        "p0_n_total_actions": p0_diag["n_total_actions"],
        "p1_n_total_actions": p1_diag["n_total_actions"],
        "per_module_signature": per_module_sig,
    }


# ---------------------------------------------------------------------------
# Interpretation grid
# ---------------------------------------------------------------------------

ENTROPY_HIGH = 0.30   # "real diversity, not noise"
ENTROPY_LOW = 0.10    # "near zero"
BASELINE_7_LOW = 0.58
BASELINE_7_HIGH = 0.78


def _classify_interpretation(by_arm: Dict[str, float]) -> Tuple[str, str]:
    """Map per-arm entropy values to a 6-row interpretation grid.

    Returns (row_label, row_description).
    """
    b42 = by_arm.get("ARM_BASELINE_42", 0.0)
    b7  = by_arm.get("ARM_BASELINE_7",  0.0)

    # R6: ARM_BASELINE_7 must replicate the V3-EXQ-555 ~0.68 finding.
    b7_replicates = (BASELINE_7_LOW <= b7 <= BASELINE_7_HIGH)
    if not b7_replicates:
        return (
            "R6_replication_failure",
            f"ARM_BASELINE_7 entropy {b7:.4f} is NOT within "
            f"[{BASELINE_7_LOW}, {BASELINE_7_HIGH}] of V3-EXQ-555's C0 "
            f"value (0.679). The patching mechanism may disturb the "
            f"baseline -- debug the class-level __init__ patch before "
            f"drawing conclusions from the swap arms."
        )

    # Identify the swap arms (excluding the two baselines).
    swap_results = {
        a: by_arm[a] for a in by_arm
        if a not in ("ARM_BASELINE_42", "ARM_BASELINE_7")
    }
    high_swaps = [a for a, e in swap_results.items() if e >= ENTROPY_HIGH]
    low_swaps  = [a for a, e in swap_results.items() if e <= ENTROPY_LOW]

    # R2: NO single-module swap lifts -> conjunctive.
    if len(high_swaps) == 0:
        return (
            "R2_multi_module_conjunctive",
            f"NO single-module swap arm shows entropy >= {ENTROPY_HIGH} "
            f"AND ARM_BASELINE_7 still reproduces (entropy {b7:.4f}). "
            f"Escape requires multi-module seed-7 conjunction. "
            f"Routes to a pair-swap follow-up enumerating 2-module "
            f"combinations (V3-EXQ-558 candidate). "
            f"swap entropies: {swap_results}"
        )

    # R1 / specialisations: at least one single-module swap lifts.
    if len(high_swaps) == 1:
        arm = high_swaps[0]
        if arm == "ARM_SWAP_LATENT":
            return (
                "R3_latent_stack_only",
                f"Only ARM_SWAP_LATENT lifts (entropy "
                f"{swap_results[arm]:.4f}); z-encoder init is the "
                f"basin-determining init. Routes to deep diagnostic of "
                f"LatentStack init distribution. "
                f"all swap entropies: {swap_results}"
            )
        if arm == "ARM_SWAP_E3":
            return (
                "R4_e3_only",
                f"Only ARM_SWAP_E3 lifts (entropy "
                f"{swap_results[arm]:.4f}); E3 value-head / selector "
                f"init is the basin-determining init. Routes to deep "
                f"diagnostic of E3TrajectorySelector init distribution. "
                f"all swap entropies: {swap_results}"
            )
        if arm == "ARM_SWAP_HIPPO":
            return (
                "R5_hippocampal_only",
                f"Only ARM_SWAP_HIPPO lifts (entropy "
                f"{swap_results[arm]:.4f}); HippocampalModule "
                f"action-object decoder init is the basin-determining "
                f"init. Routes to deep diagnostic of HippocampalModule "
                f"init distribution. "
                f"all swap entropies: {swap_results}"
            )
        return (
            "R1_single_module_sufficient",
            f"Only {arm} lifts (entropy {swap_results[arm]:.4f}); that "
            f"module's seed-7 init is the basin-determining init. "
            f"Routes to deep diagnostic of that module's init "
            f"distribution. all swap entropies: {swap_results}"
        )

    # Multiple single-module swaps lift -> either-sufficient sub-result.
    return (
        "R1_multiple_modules_sufficient",
        f"Multiple single-module swaps lift entropy >= {ENTROPY_HIGH}: "
        f"{high_swaps}. Either-sufficient outcome: more than one module "
        f"individually escapes the basin. Routes to a tighter follow-up "
        f"to identify what these modules share (depth? init scheme?). "
        f"all swap entropies: {swap_results}"
    )


# ---------------------------------------------------------------------------
# Plan / smoke / main
# ---------------------------------------------------------------------------

def _print_plan() -> None:
    print(f"{EXPERIMENT_TYPE} -- module-init swap diagnostic", flush=True)
    print(f"Arms ({len(ARMS)}): {[a[0] for a in ARMS]}", flush=True)
    for arm_label, overrides in ARMS:
        print(f"  {arm_label}: env_seed={ENV_SEED} module_seeds="
              f"{ {n: overrides.get(n, BASE_SEED) for n, _, _ in MODULE_REGISTRY} }",
              flush=True)
    print(
        f"P0 train: {P0_TRAIN_EPISODES} ep x {STEPS_PER_EPISODE} steps; "
        f"P1 eval: {P1_EVAL_EPISODES} ep x {STEPS_PER_EPISODE} steps",
        flush=True,
    )
    print(
        f"Metric: P1 action_class_entropy per arm (single run per arm; "
        f"the seed-pair IS the variable).",
        flush=True,
    )
    print(
        "Interpretation rows: R1 single_module_sufficient / R2 "
        "multi_module_conjunctive / R3 latent_stack_only / R4 e3_only / "
        "R5 hippocampal_only / R6 replication_failure.",
        flush=True,
    )
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)


def _run_smoke() -> None:
    """Verify the per-module init patch works as designed.

    Build ARM_BASELINE_42, ARM_BASELINE_7, and ARM_SWAP_E1 (each just the
    agent, no training). Compare each module's first-parameter checksum
    across the three:
      - For ARM_SWAP_E1: e1 checksum must equal ARM_BASELINE_7's e1
        checksum (E1 init drew from seed=7); every other module's
        checksum must equal ARM_BASELINE_42's (they drew from seed=42).
      - ARM_BASELINE_42 != ARM_BASELINE_7 on every module (sanity).
    Then boot a 1-ep x 20-step P0+P1 loop on ARM_BASELINE_42 to confirm
    the agent runs end-to-end.
    """
    print(
        "SMOKE MODE: ARM_BASELINE_42 + ARM_BASELINE_7 + ARM_SWAP_E1 "
        "init-signature comparison; 1 ep x 20 steps boot test; no manifest write",
        flush=True,
    )

    env42_a = _make_env(ENV_SEED)
    a42 = _build_agent_with_module_seeds(
        env42_a, overrides={}, default_seed=BASE_SEED,
    )
    sig42 = _per_module_signature(a42)

    env42_b = _make_env(ENV_SEED)
    a7 = _build_agent_with_module_seeds(
        env42_b,
        overrides={name: SWAP_SEED for name, _, _ in MODULE_REGISTRY},
        default_seed=BASE_SEED,
    )
    sig7 = _per_module_signature(a7)

    env42_c = _make_env(ENV_SEED)
    a_swap_e1 = _build_agent_with_module_seeds(
        env42_c, overrides={"e1": SWAP_SEED}, default_seed=BASE_SEED,
    )
    sig_swap_e1 = _per_module_signature(a_swap_e1)

    print("--- Per-module first-param checksum (sum) ---", flush=True)
    print(f"{'module':<14} {'BASELINE_42':>14} {'BASELINE_7':>14} {'SWAP_E1':>14}",
          flush=True)
    rows: List[Tuple[str, float, float, float]] = []
    for name, _cls, _attr in MODULE_REGISTRY:
        s42 = sig42[name]["sum"]
        s7  = sig7[name]["sum"]
        ssw = sig_swap_e1[name]["sum"]
        rows.append((name, s42, s7, ssw))
        print(f"{name:<14} {s42:>14.6f} {s7:>14.6f} {ssw:>14.6f}", flush=True)

    # Invariants:
    #   (1) BASELINE_42 != BASELINE_7 for every module
    #   (2) SWAP_E1.e1   == BASELINE_7.e1
    #   (3) SWAP_E1.<mod> == BASELINE_42.<mod>  for mod in MODULE_REGISTRY \ {e1}
    eq_tol = 1e-9

    def _eq(x: Optional[float], y: Optional[float]) -> bool:
        if x is None or y is None:
            return False
        return abs(x - y) < eq_tol

    all_baselines_differ = True
    for name, s42, s7, _ssw in rows:
        if _eq(s42, s7):
            print(f"  INVARIANT FAIL: BASELINE_42[{name}]={s42} == "
                  f"BASELINE_7[{name}]={s7} (seeds collide somehow)",
                  flush=True)
            all_baselines_differ = False

    swap_e1_e1_matches_b7 = False
    swap_e1_others_match_b42 = True
    for name, s42, s7, ssw in rows:
        if name == "e1":
            swap_e1_e1_matches_b7 = _eq(ssw, s7)
            if not swap_e1_e1_matches_b7:
                print(f"  INVARIANT FAIL: SWAP_E1.e1={ssw} != "
                      f"BASELINE_7.e1={s7}", flush=True)
        else:
            if not _eq(ssw, s42):
                print(f"  INVARIANT FAIL: SWAP_E1.{name}={ssw} != "
                      f"BASELINE_42.{name}={s42}", flush=True)
                swap_e1_others_match_b42 = False

    print("--- Patch verification ---", flush=True)
    print(f"  all_baselines_differ (BASELINE_42 vs BASELINE_7 on every mod): "
          f"{all_baselines_differ}", flush=True)
    print(f"  SWAP_E1.e1 matches BASELINE_7.e1:                              "
          f"{swap_e1_e1_matches_b7}", flush=True)
    print(f"  SWAP_E1.<other> matches BASELINE_42.<other> on all 5 others:   "
          f"{swap_e1_others_match_b42}", flush=True)

    all_pass = (
        all_baselines_differ
        and swap_e1_e1_matches_b7
        and swap_e1_others_match_b42
    )
    if not all_pass:
        print("verdict: FAIL -- patching invariants not satisfied", flush=True)
        raise RuntimeError(
            "Smoke patch-invariant check failed. See printed checksums above."
        )

    # Boot the run loop on ARM_BASELINE_42 (1 ep x 20 steps in P0 + P1) to
    # confirm end-to-end stability.
    e1_optimizer = optim.Adam(a42.e1.parameters(), lr=LR_E1)
    e2_wf_optimizer = optim.Adam(
        list(a42.e2.world_transition.parameters())
        + list(a42.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    harm_eval_optimizer = optim.Adam(
        a42.e3.harm_eval_head.parameters(), lr=LR_E3_HARM,
    )
    aux_params = list(a42.latent_stack.parameters())
    aux_optimizer = optim.Adam(aux_params, lr=LR_ENC_AUX)
    wf_buf: List = []
    harm_eval_buf: List = []
    optimizers_and_params = {
        "e1_optimizer": e1_optimizer,
        "e2_wf_optimizer": e2_wf_optimizer,
        "harm_eval_optimizer": harm_eval_optimizer,
        "aux_optimizer": aux_optimizer,
        "aux_params": aux_params,
        "wf_buf": wf_buf,
        "harm_eval_buf": harm_eval_buf,
    }
    rng_module = random.Random(ENV_SEED)
    _run_one_phase(
        agent=a42, env=env42_a, phase_label="P0",
        num_episodes=1, steps_per_episode=20,
        train=True, optimizers_and_params=optimizers_and_params,
        rng_module=rng_module, action_count_window=None,
    )
    action_count_window: Dict[int, int] = {}
    _run_one_phase(
        agent=a42, env=env42_a, phase_label="P1",
        num_episodes=1, steps_per_episode=20,
        train=False, optimizers_and_params=None,
        rng_module=rng_module, action_count_window=action_count_window,
    )
    print(f"  smoke BASELINE_42 P1 action_counts={dict(action_count_window)}",
          flush=True)
    print("verdict: PASS", flush=True)
    print("SMOKE OK", flush=True)


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_TYPE} module-init swap diagnostic",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan and exit; do not execute.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="3-arm init-signature smoke + 1 ep x 20 step boot test.",
    )
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

    if args.dry_run:
        _print_plan()
        print("DRY RUN OK", flush=True)
        return (None, None)

    if args.smoke:
        _run_smoke()
        return (None, None)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    for arm_label, overrides in ARMS:
        print(f"Arm {arm_label}: overrides={overrides}", flush=True)
        r = _run_arm(arm_label, overrides)
        print(f"verdict: PASS", flush=True)
        all_results.append(r)

    by_arm = {r["arm"]: r["p1_action_class_entropy"] for r in all_results}
    row_label, row_description = _classify_interpretation(by_arm)

    summary = {
        "per_arm_entropy": by_arm,
        "interpretation_row": row_label,
        "interpretation_description": row_description,
    }

    # Diagnostic, not substrate-claim test. Map "ran to completion +
    # produced an interpretable row" to PASS; R6_replication_failure is
    # the only outcome routed to FAIL.
    if row_label == "R6_replication_failure":
        outcome = "FAIL"
        evidence_direction = "inconclusive"
    else:
        outcome = "PASS"
        evidence_direction = "non_contributory"

    print(f"\nOutcome: {outcome}", flush=True)
    print(f"Interpretation row: {row_label}", flush=True)
    for arm_label, _overrides in ARMS:
        e = by_arm.get(arm_label, 0.0)
        print(f"  {arm_label:<18} entropy = {e:.4f}", flush=True)
    print(f"  {row_description}", flush=True)

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_note": (
            "Monostrategy-investigation diagnostic following V3-EXQ-555 "
            "(env vs agent factorization), which showed agent-side "
            "seed=7 init enables diverse policies while env_seed is "
            "irrelevant. V3-EXQ-556 localises WHICH submodule's seed-7 "
            "init is responsible. Eight arms: ARM_BASELINE_42 (all "
            "modules seed=42; control), ARM_BASELINE_7 (all modules "
            "seed=7; control), and six single-module swaps "
            "(latent_stack / e1 / e2 / residue / e3 / hippocampal) "
            "each at seed=7 with the others at seed=42. env_seed=42 "
            "fixed across all arms. Per-module seed override applied "
            "via class-level __init__ monkey-patch that calls "
            "torch.manual_seed before each module's original __init__, "
            "restored after REEAgent construction. "
            "Pre-registered interpretation rows: "
            "(R1) single_module_sufficient -- exactly one swap arm "
            "shows entropy >= 0.30; that module's seed-7 init is the "
            "basin-determining init. (R2) multi_module_conjunctive -- "
            "NO single-module swap lifts AND ARM_BASELINE_7 reproduces "
            "(~0.68); escape requires multi-module seed-7 conjunction; "
            "routes to a pair-swap follow-up (V3-EXQ-558 candidate). "
            "(R3) latent_stack_only -- only ARM_SWAP_LATENT lifts; "
            "z-encoder init drives diversity. (R4) e3_only -- only "
            "ARM_SWAP_E3 lifts; value-head / selector init drives. "
            "(R5) hippocampal_only -- only ARM_SWAP_HIPPO lifts; "
            "action-object decoder init drives. (R6) replication_"
            "failure -- ARM_BASELINE_7 NOT within [0.58, 0.78] of "
            "V3-EXQ-555 C0 (0.679); patch mechanism disturbs the "
            "baseline; debug patching before drawing conclusions. "
            "experiment_purpose=diagnostic; evidence_direction set to "
            "non_contributory so governance does not weight any claim "
            "from this run."
        ),
        "pass_criteria_summary": summary,
        "per_arm_results": all_results,
        "config": {
            "arms": [
                {"label": label, "module_overrides": overrides,
                 "env_seed": ENV_SEED, "base_seed": BASE_SEED}
                for label, overrides in ARMS
            ],
            "modules_registry": [n for n, _, _ in MODULE_REGISTRY],
            "p0_train_episodes": P0_TRAIN_EPISODES,
            "p1_eval_episodes": P1_EVAL_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "env_kwargs": ENV_KWARGS,
        },
    }

    out_file = write_flat_manifest(
        output,
        out_dir,
        dry_run=False,
        config=output.get("config"),
        seeds=None,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
    )
    print(f"Result written to: {out_file}", flush=True)

    return (outcome, str(out_file))


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    if _outcome is not None and _manifest_path is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
