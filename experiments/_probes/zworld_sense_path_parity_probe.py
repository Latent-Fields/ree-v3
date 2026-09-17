"""SD-ZWORLD-SENSE-PATH-PARITY discriminating probe -- a MEASUREMENT SPIKE, not an experiment.

WHAT THIS IS
------------
The confirmed `failure_autopsy_V3-EXQ-1030_2026-09-14` found that SD-070's P0a recipe
(`ZWorldP0Trainer._z_world_path`) trains `split_encoder.world_encoder` on RAW `world_state`,
while `REEAgent.sense()` reads that encoder THROUGH `agent.world_obs_encoder` -- a separately
initialised, within the SD-070 lineage UNTRAINED, `Linear(275,275) + ReLU`. V3-EXQ-1030
measured the sense()-path z_world ~0.10 absolute LOWER in held-out linear-probe decodability
of the 5-way oracle waypoint direction than the direct path.

The 2026-09-16 cross-model red-team (F3) marked that ATTRIBUTION UNMEASURED:

  Cause A -- train/inference DIVERGENCE. The encoder was optimised on a distribution it never
            sees at inference. Fix: train P0a THROUGH `world_obs_encoder`.
  Cause B -- INTRINSIC loss in the random ReLU projection. A random Linear+ReLU destroys
            linear separability whatever the encoder was trained on. Fix: bypass it.

The two fixes help under different causes, and `SD-ZWORLD-SENSE-PATH-PARITY` (substrate_queue,
`pending_implementation`) mandates running the discriminating probe BEFORE choosing.

WHY THE MANDATED ARM ALONE IS NOT ENOUGH (and what this script adds)
--------------------------------------------------------------------
The chip specifies one extra arm: `world_encoder(FRESH random Linear+ReLU(world_state))`.
That arm is necessary but NOT sufficient, for two reasons found while reading the code:

 1. Within the SD-070 lineage `agent.world_obs_encoder` is ITSELF untrained random. So a
    fresh random projection and the agent's own projection are the same KIND of object,
    differing only by RNG draw. Comparing them tests whether the particular weights matter --
    it does not separate divergence from intrinsic loss, because BOTH paths feed a
    raw-trained encoder an off-distribution input. The only arm that separates them is one
    where the encoder was TRAINED THROUGH the projection (`*_T` below): if decodability
    recovers to the direct level, the loss was divergence (A); if it stays low, the
    projection intrinsically destroyed the information (B).

 2. `agent.sense()` differs from `_direct_zworld` in FOUR ways, not one:
      (i) the `world_obs_encoder` pre-projection        <- the hypothesised cause
     (ii) top-down conditioning from the z_beta stack
    (iii) SD-007 reafference correction
     (iv) the alpha_world=0.9 temporal EMA (stack.py:1584)
    So 1030's "sense minus direct" gap is an upper bound on (i)'s contribution and attributes
    nothing on its own. `direct_wobs_R` below applies ONLY the pre-projection, with no
    top-down / reafference / EMA, which is the arm that isolates (i).

THREE MORE CONTROLS FOR THINGS THAT COULD FAKE A 0.10 GAP
----------------------------------------------------------
 * SPLIT-SEED NOISE. 1030 fitted each arm under a DIFFERENT episode-split seed
   (`seed`, `seed+1`, `seed+2`, `seed+3`; see its `_fit_and_eval_probe` calls), so its
   sense-vs-direct contrast varies the train/test EPISODE PARTITION as well as the feature
   set. Here EVERY arm is fitted on the SAME splits, and each arm is fitted under THREE
   split seeds so the between-split spread is reported next to the between-arm gap. If the
   gap does not exceed the split spread, that is the finding.
 * FEATURE SCALE. A ReLU projection changes feature scale, and an unnormalised linear probe
   at a fixed lr/step budget can converge differently for that reason alone. Every arm is
   therefore fitted TWICE: once unnormalised (1030-identical) and once z-scored using
   TRAIN-SPLIT statistics only. A gap that survives standardisation is not a scale artifact.
 * DIMENSIONALITY. Probe capacity scales with input width, so arms are only ever compared
   within a matched-width family: the 32-dim z_world family, the 275-dim world_obs family,
   and the 50-dim raw positive-control slice on its own.

RNG DISCIPLINE (CLAUDE.md measurement contract)
------------------------------------------------
`reset_all_rng(seed)` normally lives in `arm_cell.__enter__`. This script does NOT use
`arm_cell` -- it calls its own per-seed function directly -- so it calls `reset_all_rng(seed)`
EXPLICITLY at cell entry. Both P0a variants additionally run under `_rng_neutral()` (inherited
from `run_zworld_p0`), and both start from a byte-identical agent snapshot, so the R/T contrast
is not confounded by a global-stream offset.

WHAT IT DOES NOT DO
-------------------
No manifest. Nothing is written to `REE_assembly/evidence/experiments/`. No queue entry. No
substrate file under `ree_core/` is modified -- the train-through variant is a probe-local
SUBCLASS of `ZWorldP0Trainer` overriding two methods.

Usage:
    /opt/local/bin/python3 experiments/_probes/zworld_sense_path_parity_probe.py --smoke
    /opt/local/bin/python3 experiments/_probes/zworld_sense_path_parity_probe.py \
        --seeds 42 43 44 45 --arm field_off --out <path>.json
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.optim as optim  # noqa: E402

from experiments._lib.arm_fingerprint import reset_all_rng  # noqa: E402
from experiments._lib.capability_eval import RandomPolicy  # noqa: E402
from experiments._lib.zworld_p0_warmup import (  # noqa: E402
    _rng_neutral,
    resolve_p0a_config,
    resolve_target_fn,
    run_zworld_p0,
)
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot,
    latent_stack_weight_delta,
)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.latent.zworld_p0 import ZWorldP0Trainer  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

# ---------------------------------------------------------------------------------------
# Constants -- ALL copied verbatim from
# experiments/v3_exq_1030_mech428_inv086_waypoint_field_zworld_decodability.py so the probe
# measures the same regime the 0.10 gap was measured in.
# ---------------------------------------------------------------------------------------
GRID_SIZE = 12
N_WAYPOINTS = 3
STEPS_PER_EPISODE = 150
NUM_HAZARDS = 2
NUM_RESOURCES = 3
WAYPOINT_VISIT_REWARD = 0.2
WAYPOINT_FIELD_DECAY = 0.25
WAYPOINT_COMPLETION_REWARD = 0.8
SEQUENCE_COMMITMENT_TIMEOUT = 20
SELF_DIM = 32
WORLD_DIM = 32
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3
P0_EPISODES = 200
PROBE_COLLECT_EPISODES = 40
PROBE_STEPS = 150
PROBE_LR = 5e-3
N_DIRECTIONS = 5
N_QUADRANTS = 4
PROBE_ENV_SEED_OFFSET = 500_000_003
ACTION_RNG_SEED_OFFSET = 9973
LOCAL_VIEW_DIMS = 175
N_ENTITY_TYPES = 7
WAYPOINT_ENTITY_CHANNEL = 6
FIELD_DIMS = 25
MIN_TEST_SAMPLES = 20
MIN_CLASSES_OBSERVED = 2
DEVICE = torch.device("cpu")

# Probe-local. A distinct offset so the FRESH random projection cannot coincide with any
# module the agent built at construction time.
FRESH_PROJ_SEED_OFFSET = 777_000_001
# Number of episode-split seeds each arm is fitted under (the split-noise control). 10,
# not 3: the first pilot measured a between-split spread of 0.34 absolute accuracy WITHIN a
# single arm and seed, so three splits cannot bound the noise the 0.10 gap has to clear.
N_SPLIT_SEEDS = 10

# The three probe CONDITIONS every arm is fitted under. `unnormalised` is 1030-identical.
# `standardised` removes feature scale. `unnormalised_long` keeps 1030's raw features but
# gives the probe 10x the gradient budget -- it separates "this path carries less
# information" from "this path's features are 5x smaller, so a fixed lr/step budget leaves
# its probe UNDER-CONVERGED". The first pilot found the unnormalised sense/pre-projection
# probes reaching only ~0.51 TRAIN accuracy against ~0.60 for the direct path, which is the
# signature of the latter, so the distinction is load-bearing rather than decorative.
PROBE_CONDITIONS = (
    ("unnormalised", False, 1),
    ("standardised", True, 1),
    ("unnormalised_long", False, 10),
)

ARM_OFF = "field_off"
ARM_ON = "field_on"

# The matched-width comparison families. Never compare across families -- probe capacity
# scales with input width.
FAMILY_Z32 = "z_world_32d"
FAMILY_WOBS = "world_obs_275d"
FAMILY_RAW = "raw_slice_50d"


# ---------------------------------------------------------------------------------------
# Environment / agent -- verbatim from the 1030 driver
# ---------------------------------------------------------------------------------------
class _FieldMaskedEnv:
    """Verbatim from the 1030 driver: always builds the WIDE (field-enabled) env so both
    arms share world_obs_dim=275 and an identical agent-construction RNG sequence; under
    field_on=False the trailing FIELD_DIMS columns of world_state are zeroed."""

    def __init__(self, seed: int, field_on: bool) -> None:
        self._env = CausalGridWorldV2(
            seed=seed,
            size=GRID_SIZE,
            use_proxy_fields=True,
            subgoal_mode=True,
            num_waypoints=N_WAYPOINTS,
            waypoint_visit_reward=WAYPOINT_VISIT_REWARD,
            subgoal_arrival_position_check=True,
            num_hazards=NUM_HAZARDS,
            num_resources=NUM_RESOURCES,
            waypoint_completion_reward=WAYPOINT_COMPLETION_REWARD,
            sequence_commitment_timeout=SEQUENCE_COMMITMENT_TIMEOUT,
            waypoint_proximity_field_enabled=True,
            waypoint_field_decay=WAYPOINT_FIELD_DECAY,
        )
        self._field_on = bool(field_on)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _mask(self, obs: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if self._field_on or obs is None:
            return obs
        ws = obs["world_state"].clone()
        ws[-FIELD_DIMS:] = 0.0
        masked = dict(obs)
        masked["world_state"] = ws
        return masked

    def reset(self):
        flat, obs = self._env.reset()
        return flat, self._mask(obs)

    def step(self, action):
        flat, r, done, info, obs = self._env.step(action)
        return flat, r, done, info, self._mask(obs)


def _build_env(field_on: bool, seed: int) -> _FieldMaskedEnv:
    return _FieldMaskedEnv(seed=seed, field_on=field_on)


def _build_agent(env: _FieldMaskedEnv, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    np.random.seed(seed)
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
    return REEAgent(config).to(DEVICE)


def _oracle_direction(env: _FieldMaskedEnv) -> int:
    """Verbatim from the 1030 driver: greedy 5-way direction toward the pending waypoint,
    from env ground truth. A LABEL only -- the rollout policy is uniform random."""
    idx = int(getattr(env, "_next_waypoint_idx", 0))
    wps = getattr(env, "waypoints", []) or []
    if not wps or idx >= len(wps):
        return 4
    wx, wy = int(wps[idx][0]), int(wps[idx][1])
    ax, ay = int(env.agent_x), int(env.agent_y)
    dx, dy = wx - ax, wy - ay
    if abs(dx) >= abs(dy) and dx != 0:
        return 1 if dx > 0 else 0
    if dy != 0:
        return 3 if dy > 0 else 2
    return 4


def _quadrant_label(env: _FieldMaskedEnv) -> int:
    half = GRID_SIZE / 2.0
    ax, ay = int(env.agent_x), int(env.agent_y)
    return (1 if ax >= half else 0) + (2 if ay >= half else 0)


def _raw_probe_vector(obs_dict: Dict[str, Any], field_on: bool) -> torch.Tensor:
    """Verbatim from the 1030 driver: the narrow 25/50-dim agent-centred waypoint-channel
    plus-field slice, zero-padded to 50 under OFF. The RAW positive control."""
    ws = np.asarray(obs_dict["world_state"], dtype=np.float32).reshape(-1)
    wp_local = ws[:LOCAL_VIEW_DIMS][WAYPOINT_ENTITY_CHANNEL::N_ENTITY_TYPES]
    if field_on:
        fv = np.asarray(obs_dict["waypoint_proximity_field_view"],
                        dtype=np.float32).reshape(-1)
    else:
        fv = np.zeros(FIELD_DIMS, dtype=np.float32)
    return torch.as_tensor(np.concatenate([wp_local, fv]), dtype=torch.float32)


# ---------------------------------------------------------------------------------------
# The train-through-the-pre-projection P0a variant. PROBE-LOCAL SUBCLASS -- no substrate
# file is modified. Overriding exactly `_z_world_path` and `world_path_parameters` is
# sufficient: every supervision target in `ZWorldP0Trainer.train()` is derived from the
# buffered RAW `obs`, and both the training loop and `_holdout_report` route their encoder
# call through `_z_world_path`, so the projection enters the optimised path and nothing else
# moves.
# ---------------------------------------------------------------------------------------
class _ThroughPreProjectionTrainer(ZWorldP0Trainer):
    def __init__(self, latent_stack: Any, pre_projection: nn.Module, config: Any = None) -> None:
        super().__init__(latent_stack, config)
        self._pre = pre_projection

    def _z_world_path(self, world_obs: torch.Tensor) -> torch.Tensor:
        return super()._z_world_path(self._pre(world_obs))

    def world_path_parameters(self) -> List[torch.Tensor]:
        return super().world_path_parameters() + list(self._pre.parameters())


def run_zworld_p0_through_preprojection(agent: Any, warmup_env: Any, seed: int, episodes: int,
                                        steps_per_episode: int, policy: Any,
                                        label: str = "", dry_run: bool = False) -> Dict[str, Any]:
    """`run_zworld_p0`, but with `agent.world_obs_encoder` INSIDE the optimised path.

    Structure copied from `experiments/_lib/zworld_p0_warmup.run_zworld_p0` (same config
    resolution, same target fn, same `_rng_neutral()` isolation, same dedicated warmup env
    contract) so the only difference from variant R is the encoder path itself."""
    if episodes <= 0:
        return {"p0a_recipe": "sd070_through_preprojection", "p0a_ran": False,
                "p0a_reason": "episodes<=0"}
    cfg = resolve_p0a_config(seed, dry_run, 0.0, None)
    target = resolve_target_fn(None)
    out: Dict[str, Any] = {"p0a_recipe": "sd070_through_preprojection", "p0a_ran": True}
    with _rng_neutral():
        trainer = _ThroughPreProjectionTrainer(agent.latent_stack, agent.world_obs_encoder, cfg)
        for ep in range(int(episodes)):
            _flat0, obs_dict = warmup_env.reset()
            policy.reset(warmup_env)
            for _step in range(int(steps_per_episode)):
                trainer.observe(obs_dict["world_state"].float(), target(obs_dict))
                action = policy.act(warmup_env, obs_dict)
                with torch.no_grad():
                    _flat, _harm, done, _info, obs_dict = warmup_env.step(action)
                if done:
                    break
            cur = ep + 1
            if cur == 1 or cur % 50 == 0 or cur == int(episodes):
                print("  [train] %s seed=%d phase=P0a-THROUGH ep %d/%d"
                      % (label or "zworld_p0", int(seed), cur, int(episodes)), flush=True)
        out["p0a_n_buffered"] = int(trainer.n_buffered)
        try:
            stats = trainer.train()
        except ValueError as exc:
            out["p0a_ran"] = False
            out["p0a_reason"] = "trainer_refused_buffer: %s" % (exc,)
            return out
    out["p0a_mean_loss"] = stats.get("mean_loss")
    out["p0a_final_loss"] = stats.get("final_loss")
    out["p0a_n_steps"] = stats.get("n_steps")
    out["p0a_holdout"] = stats.get("holdout")
    ho = stats.get("holdout") or {}
    out["p0a_holdout_mean_lift"] = ho.get("mean_lift")
    return out


# ---------------------------------------------------------------------------------------
# Feature paths
# ---------------------------------------------------------------------------------------
def _direct_zworld_batch(agent: REEAgent, world_obs: torch.Tensor) -> torch.Tensor:
    """`split_encoder.world_encoder` + SD-106 skip + precision gate, batched -- exactly
    `ZWorldP0Trainer._z_world_path`, which is the distribution P0a trains on. `world_obs`
    is [N, world_obs_dim] and may already have a pre-projection applied by the caller."""
    se = agent.latent_stack.split_encoder
    with torch.no_grad():
        z = se.world_encoder(world_obs)
        skip = getattr(se, "world_encoder_skip", None)
        if skip is not None:
            z = z + skip(world_obs)
        z = z * torch.sigmoid(se.world_precision_logit).unsqueeze(0)
    return z.detach().cpu()


def _make_fresh_projection(dim: int, seed: int) -> nn.Module:
    """A FRESH `Linear(dim, dim) + ReLU` with the SAME construction as
    `REEAgent.world_obs_encoder` (agent.py:3117) and an independent seed."""
    g_state = torch.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU()).to(DEVICE)
    finally:
        torch.set_rng_state(g_state)
    for p in proj.parameters():
        p.requires_grad_(False)
    return proj


def _apply_module(mod: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return mod(x).detach().cpu()


def _within_episode_ema(feats: torch.Tensor, episode_ids: torch.Tensor,
                        alpha: float) -> torch.Tensor:
    """`z = alpha * z_instant + (1 - alpha) * z_prev`, reset at each episode boundary --
    the temporal smoothing `LatentStack` applies at stack.py:1584, applied to an otherwise
    instantaneous feature path so the EMA's own contribution can be read off."""
    out = torch.empty_like(feats)
    prev: Optional[torch.Tensor] = None
    last_ep = None
    for i in range(feats.shape[0]):
        ep = int(episode_ids[i])
        if ep != last_ep:
            prev = None
            last_ep = ep
        cur = feats[i] if prev is None else alpha * feats[i] + (1.0 - alpha) * prev
        out[i] = cur
        prev = cur
    return out


def _replay_sense(agent: REEAgent, episodes: List[Dict[str, Any]]) -> torch.Tensor:
    """Replay the STORED (body_state, world_state) sequences through `agent.sense()`, with
    `agent.reset()` at each episode boundary -- reproducing the 1030 driver's own collection
    loop exactly, but from stored observations so every encoder variant sees a bit-identical
    input sequence. `sense()` is stateful (EMA, reafference, z_beta top-down), which is why
    it must be replayed rather than recomputed from a stored instantaneous feature."""
    agent.eval()
    zs: List[torch.Tensor] = []
    for ep in episodes:
        agent.reset()
        for body, world in zip(ep["body"], ep["world"]):
            with torch.no_grad():
                latent = agent.sense(body, world)
            zs.append(latent.z_world.detach().cpu().reshape(-1))
    return torch.stack(zs)


# ---------------------------------------------------------------------------------------
# Probe fitting -- logic verbatim from the 1030 driver's `_episode_split` /
# `_fit_and_eval_probe`, with two additions: an optional train-split standardisation, and
# a split seed decoupled from the probe-init seed so the split can be held FIXED across arms.
# ---------------------------------------------------------------------------------------
def _episode_split(episode_ids: torch.Tensor, seed: int, train_frac: float = 0.8):
    uniq = sorted(int(e) for e in episode_ids.unique().tolist())
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(uniq), generator=g).tolist()
    n_train_eps = max(1, int(len(uniq) * train_frac))
    if n_train_eps >= len(uniq) and len(uniq) > 1:
        n_train_eps = len(uniq) - 1
    train_eps = {uniq[i] for i in perm[:n_train_eps]}
    train_mask = torch.tensor([int(e) in train_eps for e in episode_ids.tolist()])
    test_mask = ~train_mask
    return train_mask.nonzero(as_tuple=True)[0], test_mask.nonzero(as_tuple=True)[0]


def _fit_probe(features: torch.Tensor, labels: torch.Tensor, episode_ids: torch.Tensor,
               n_classes: int, n_steps: int, split_seed: int, init_seed: int,
               standardise: bool, lr: float = PROBE_LR) -> Dict[str, Any]:
    n = int(features.shape[0])
    n_obs = int(labels.unique().numel()) if n else 0
    if n < 2 or n_obs < MIN_CLASSES_OBSERVED:
        return {"accuracy": None, "reason": "degenerate_labels"}
    tr, te = _episode_split(episode_ids, split_seed)
    if tr.numel() == 0 or te.numel() == 0:
        return {"accuracy": None, "reason": "empty_split"}
    x_tr, y_tr = features[tr], labels[tr]
    x_te, y_te = features[te], labels[te]
    if standardise:
        # TRAIN-SPLIT statistics only -- a test-split statistic would leak.
        mu = x_tr.mean(dim=0, keepdim=True)
        sd = x_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
        x_tr = (x_tr - mu) / sd
        x_te = (x_te - mu) / sd
    majority = float(torch.bincount(y_te, minlength=n_classes).max().item()
                     / max(1, int(y_te.shape[0])))
    # Probe weight init seeded independently of ambient global state (1030's F6).
    g_state = torch.get_rng_state()
    try:
        torch.manual_seed(int(init_seed) * 7919 + 3)
        probe = nn.Linear(int(features.shape[1]), n_classes)
    finally:
        torch.set_rng_state(g_state)
    opt = optim.Adam(probe.parameters(), lr=lr)
    probe.train()
    for _ in range(n_steps):
        loss = F.cross_entropy(probe(x_tr), y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        acc = float((probe(x_te).argmax(dim=-1) == y_te).float().mean().item())
        tr_acc = float((probe(x_tr).argmax(dim=-1) == y_tr).float().mean().item())
    return {"accuracy": acc, "train_accuracy": tr_acc, "majority_class_rate": majority,
            "n_test": int(te.numel()), "n_train": int(tr.numel()),
            "test_size_adequate": bool(te.numel() >= MIN_TEST_SAMPLES)}


# ---------------------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------------------
def _collect_observations(env: _FieldMaskedEnv, rng: np.random.RandomState, n_episodes: int,
                          field_on: bool) -> Dict[str, Any]:
    """Random-policy rollout on a dedicated probe env, storing the RAW observations and
    labels ONLY. Agent-independent by construction -- actions come from `rng`, never from the
    agent -- so every encoder variant is later evaluated on a bit-identical trajectory set."""
    episodes: List[Dict[str, Any]] = []
    labels: List[int] = []
    quads: List[int] = []
    ep_ids: List[int] = []
    raws: List[torch.Tensor] = []
    worlds: List[torch.Tensor] = []
    for ep in range(n_episodes):
        _flat, obs = env.reset()
        ep_body: List[torch.Tensor] = []
        ep_world: List[torch.Tensor] = []
        for _t in range(STEPS_PER_EPISODE):
            labels.append(int(_oracle_direction(env)))
            quads.append(int(_quadrant_label(env)))
            ep_ids.append(int(ep))
            raws.append(_raw_probe_vector(obs, field_on))
            b = obs["body_state"].detach().float().clone()
            w = obs["world_state"].detach().float().clone()
            ep_body.append(b)
            ep_world.append(w)
            worlds.append(w.reshape(-1))
            action = int(rng.randint(0, int(env.action_dim)))
            _flat, _r, done, _info, obs = env.step(action)
            if done:
                break
        episodes.append({"body": ep_body, "world": ep_world})
    return {"episodes": episodes,
            "world_flat": torch.stack(worlds),
            "raw": torch.stack(raws),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "quadrants": torch.as_tensor(quads, dtype=torch.long),
            "episode_ids": torch.as_tensor(ep_ids, dtype=torch.long)}


# ---------------------------------------------------------------------------------------
# One seed-cell
# ---------------------------------------------------------------------------------------
def run_cell(seed: int, arm: str, smoke: bool, p0_episodes: Optional[int] = None,
             collect_episodes: Optional[int] = None) -> Dict[str, Any]:
    field_on = (arm == ARM_ON)
    p0_episodes = p0_episodes if p0_episodes is not None else (4 if smoke else P0_EPISODES)
    collect_episodes = (collect_episodes if collect_episodes is not None
                        else (4 if smoke else PROBE_COLLECT_EPISODES))
    probe_steps = 20 if smoke else PROBE_STEPS
    n_split_seeds = 2 if smoke else N_SPLIT_SEEDS

    # MEASUREMENT CONTRACT: reset_all_rng normally lives in arm_cell.__enter__; this probe
    # calls its cell function directly, so the reset is explicit here.
    reset_all_rng(seed)

    spec_env = _build_env(field_on, seed)
    agent = _build_agent(spec_env, seed)
    world_obs_dim = int(spec_env.world_obs_dim)
    init_state = copy.deepcopy(agent.state_dict())
    wobs_init = copy.deepcopy(agent.world_obs_encoder.state_dict())

    # --- variant R: SD-070 P0a on RAW world_state (the 1030 configuration, verbatim call)
    before_R = latent_stack_snapshot(agent)
    p0a_R = run_zworld_p0(agent, _build_env(field_on, seed), seed=seed, episodes=p0_episodes,
                          steps_per_episode=STEPS_PER_EPISODE, policy=RandomPolicy(seed),
                          label="zworld_parity_probe|R", dry_run=smoke)
    delta_R = latent_stack_weight_delta(agent, before_R)
    wobs_moved_R = _state_dict_delta(wobs_init, agent.world_obs_encoder.state_dict())
    state_R = copy.deepcopy(agent.state_dict())

    # --- variant T: identical start, P0a THROUGH agent.world_obs_encoder
    agent.load_state_dict(init_state)
    before_T = latent_stack_snapshot(agent)
    p0a_T = run_zworld_p0_through_preprojection(
        agent, _build_env(field_on, seed), seed=seed, episodes=p0_episodes,
        steps_per_episode=STEPS_PER_EPISODE, policy=RandomPolicy(seed),
        label="zworld_parity_probe|T", dry_run=smoke)
    delta_T = latent_stack_weight_delta(agent, before_T)
    wobs_moved_T = _state_dict_delta(wobs_init, agent.world_obs_encoder.state_dict())
    state_T = copy.deepcopy(agent.state_dict())

    # --- probe data (agent-independent)
    probe_env = _build_env(field_on, seed + PROBE_ENV_SEED_OFFSET)
    rng = np.random.RandomState(seed + ACTION_RNG_SEED_OFFSET)
    data = _collect_observations(probe_env, rng, collect_episodes, field_on)
    ws = data["world_flat"]
    ep_ids = data["episode_ids"]
    labels = data["labels"]

    fresh = _make_fresh_projection(world_obs_dim, seed + FRESH_PROJ_SEED_OFFSET)

    feats: Dict[str, Tuple[str, torch.Tensor]] = {}

    # ---- encoder trained on RAW (the current, shipped SD-070 recipe)
    agent.load_state_dict(state_R)
    wobs_R = agent.world_obs_encoder
    ws_wobs_R = _apply_module(wobs_R, ws)
    ws_fresh = _apply_module(fresh, ws)
    z_direct_R = _direct_zworld_batch(agent, ws)
    z_wobs_R = _direct_zworld_batch(agent, ws_wobs_R)
    z_fresh_R = _direct_zworld_batch(agent, ws_fresh)
    feats["R_direct_raw"] = (FAMILY_Z32, z_direct_R)
    feats["R_direct_wobs"] = (FAMILY_Z32, z_wobs_R)
    feats["R_direct_fresh_random"] = (FAMILY_Z32, z_fresh_R)
    feats["R_direct_raw_ema"] = (FAMILY_Z32, _within_episode_ema(z_direct_R, ep_ids, ALPHA_WORLD))
    feats["R_direct_wobs_ema"] = (FAMILY_Z32, _within_episode_ema(z_wobs_R, ep_ids, ALPHA_WORLD))
    feats["R_sense"] = (FAMILY_Z32, _replay_sense(agent, data["episodes"]))
    feats["R_wobs_output"] = (FAMILY_WOBS, ws_wobs_R)

    # ---- encoder (and pre-projection) trained THROUGH the pre-projection
    agent.load_state_dict(state_T)
    wobs_T = agent.world_obs_encoder
    ws_wobs_T = _apply_module(wobs_T, ws)
    z_wobs_T = _direct_zworld_batch(agent, ws_wobs_T)
    feats["T_direct_wobs"] = (FAMILY_Z32, z_wobs_T)
    feats["T_direct_raw"] = (FAMILY_Z32, _direct_zworld_batch(agent, ws))
    feats["T_direct_wobs_ema"] = (FAMILY_Z32, _within_episode_ema(z_wobs_T, ep_ids, ALPHA_WORLD))
    feats["T_sense"] = (FAMILY_Z32, _replay_sense(agent, data["episodes"]))
    feats["T_wobs_output"] = (FAMILY_WOBS, ws_wobs_T)

    # ---- encoder-free controls: what a random ReLU projection does to linear separability
    #      on its own. Matched width (275) on both sides, so no capacity confound.
    feats["ctrl_world_state"] = (FAMILY_WOBS, ws)
    feats["ctrl_world_state_fresh_random"] = (FAMILY_WOBS, ws_fresh)
    feats["ctrl_raw_slice"] = (FAMILY_RAW, data["raw"])

    split_seeds = [seed + k for k in range(n_split_seeds)]
    results: Dict[str, Any] = {}
    for name, (family, mat) in feats.items():
        per_norm: Dict[str, Any] = {}
        for norm_key, standardise, step_mult in PROBE_CONDITIONS:
            accs = []
            fits = {}
            for ss in split_seeds:
                r = _fit_probe(mat, labels, ep_ids, N_DIRECTIONS, probe_steps * step_mult,
                               split_seed=ss, init_seed=seed, standardise=standardise)
                fits[str(ss)] = r
                if r.get("accuracy") is not None:
                    accs.append(r["accuracy"])
            tr = [v["train_accuracy"] for v in fits.values()
                  if v.get("train_accuracy") is not None]
            per_norm[norm_key] = {
                "mean_accuracy": float(statistics.fmean(accs)) if accs else None,
                "mean_train_accuracy": float(statistics.fmean(tr)) if tr else None,
                "per_split": fits,
                "split_spread": (max(accs) - min(accs)) if len(accs) > 1 else None,
            }
        results[name] = {"family": family, "dim": int(mat.shape[1]), **per_norm}

    # Auxiliary positive control (quadrant), on the two headline arms only.
    quad = {}
    for name in ("R_direct_raw", "R_sense"):
        quad[name] = _fit_probe(feats[name][1], data["quadrants"], ep_ids, N_QUADRANTS,
                                probe_steps, split_seed=seed, init_seed=seed,
                                standardise=False)

    label_counts = {int(c): int((labels == c).sum().item()) for c in range(N_DIRECTIONS)}
    return {
        "seed": seed,
        "arm": arm,
        "n_samples": int(labels.shape[0]),
        "n_episodes": collect_episodes,
        "world_obs_dim": world_obs_dim,
        "split_seeds": split_seeds,
        "label_counts": label_counts,
        "probes": results,
        "quadrant_control": quad,
        "p0a_R": {k: v for k, v in p0a_R.items() if k != "p0a_config"},
        "p0a_T": p0a_T,
        "weight_delta_R": delta_R,
        "weight_delta_T": delta_T,
        "world_obs_encoder_moved_R": wobs_moved_R,
        "world_obs_encoder_moved_T": wobs_moved_T,
        "feature_scale": {name: {"mean_abs": float(mat.abs().mean().item()),
                                 "std": float(mat.std().item()),
                                 "frac_zero": float((mat == 0).float().mean().item())}
                          for name, (_fam, mat) in feats.items()},
    }


def _state_dict_delta(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> float:
    """Total L2 norm of the change across a state_dict -- the readiness signal for
    'did this module actually receive gradient'."""
    tot = 0.0
    for k, v in after.items():
        b = before.get(k)
        if b is not None and b.shape == v.shape:
            tot += float((v.detach() - b.detach()).norm().item())
    return tot


# ---------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="SD-ZWORLD-SENSE-PATH-PARITY discriminating probe")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    ap.add_argument("--arm", choices=[ARM_OFF, ARM_ON], default=ARM_OFF)
    ap.add_argument("--smoke", action="store_true", help="tiny budgets; exercises every path")
    ap.add_argument("--p0-episodes", type=int, default=None)
    ap.add_argument("--collect-episodes", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    per_seed = []
    for s in args.seeds:
        print("=== seed %d arm %s ===" % (s, args.arm), flush=True)
        per_seed.append(run_cell(s, args.arm, args.smoke,
                                 p0_episodes=args.p0_episodes,
                                 collect_episodes=args.collect_episodes))
        r = per_seed[-1]["probes"]
        print("  R_direct_raw=%s R_sense=%s R_direct_wobs=%s T_direct_wobs=%s T_sense=%s"
              % tuple("%.4f" % r[k]["unnormalised"]["mean_accuracy"]
                      if r[k]["unnormalised"]["mean_accuracy"] is not None else "n/a"
                      for k in ("R_direct_raw", "R_sense", "R_direct_wobs",
                                "T_direct_wobs", "T_sense")), flush=True)

    out = {"probe": "zworld_sense_path_parity_probe",
           "spike_for": "SD-ZWORLD-SENSE-PATH-PARITY",
           "not_an_experiment": True,
           "started_utc": started,
           "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "arm": args.arm, "seeds": args.seeds, "smoke": bool(args.smoke),
           "p0_episodes": args.p0_episodes if args.p0_episodes is not None
                          else (4 if args.smoke else P0_EPISODES),
           "collect_episodes": args.collect_episodes if args.collect_episodes is not None
                               else (4 if args.smoke else PROBE_COLLECT_EPISODES),
           "per_seed": per_seed,
           "summary": _summarise(per_seed)}
    txt = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(txt, encoding="utf-8")
        print("wrote %s" % args.out, flush=True)
    else:
        print(txt)
    return 0


def _summarise(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-arm means and the four contrasts the attribution turns on. Every contrast is
    within one matched-width family."""
    if not per_seed:
        return {}
    names = list(per_seed[0]["probes"].keys())
    summary: Dict[str, Any] = {"arm_means": {}, "contrasts": {}}
    for norm_key, _std, _mult in PROBE_CONDITIONS:
        means = {}
        for nm in names:
            vals = [c["probes"][nm][norm_key]["mean_accuracy"] for c in per_seed
                    if c["probes"][nm][norm_key]["mean_accuracy"] is not None]
            means[nm] = float(statistics.fmean(vals)) if vals else None
        summary["arm_means"][norm_key] = means
        spreads = [c["probes"][nm][norm_key]["split_spread"] for c in per_seed for nm in names
                   if c["probes"][nm][norm_key]["split_spread"] is not None]
        contrasts = {
            # The 1030 headline, reproduced on ONE fixed split set.
            "1030_gap_direct_minus_sense": _sub(means, "R_direct_raw", "R_sense"),
            # (i) the pre-projection ALONE, no top-down / reafference / EMA.
            "preprojection_only": _sub(means, "R_direct_raw", "R_direct_wobs"),
            # The mandated arm: a FRESH random projection in the same position.
            "fresh_random_only": _sub(means, "R_direct_raw", "R_direct_fresh_random"),
            # The EMA alone.
            "ema_only": _sub(means, "R_direct_raw", "R_direct_raw_ema"),
            # Everything sense() does BEYOND the pre-projection.
            "sense_beyond_preprojection": _sub(means, "R_direct_wobs", "R_sense"),
            # CAUSE DISCRIMINATOR: does training through the projection recover it?
            "train_through_recovery": _sub(means, "T_direct_wobs", "R_direct_wobs"),
            "residual_after_train_through": _sub(means, "R_direct_raw", "T_direct_wobs"),
            "sense_recovery_train_through": _sub(means, "T_sense", "R_sense"),
            # Encoder-free: what a random ReLU does to linear separability by itself.
            "random_relu_intrinsic_loss": _sub(means, "ctrl_world_state",
                                               "ctrl_world_state_fresh_random"),
            "untrained_wobs_intrinsic_loss": _sub(means, "ctrl_world_state", "R_wobs_output"),
            "trained_wobs_intrinsic_loss": _sub(means, "ctrl_world_state", "T_wobs_output"),
        }
        summary["contrasts"][norm_key] = contrasts
        summary.setdefault("paired", {})[norm_key] = {
            key: _paired(per_seed, norm_key, a, b)
            for key, (a, b) in _PAIRS.items()}
        summary.setdefault("max_split_spread", {})[norm_key] = max(spreads) if spreads else None
    return summary


# The contrasts worth a PAIRED statistic: the difference is taken within one (seed, split)
# cell, so seed-level and split-level variance -- which the pilot showed dominates -- cancels
# instead of being carried into the comparison.
_PAIRS = {
    "1030_gap_direct_minus_sense": ("R_direct_raw", "R_sense"),
    "preprojection_only": ("R_direct_raw", "R_direct_wobs"),
    "fresh_random_only": ("R_direct_raw", "R_direct_fresh_random"),
    "ema_only": ("R_direct_raw", "R_direct_raw_ema"),
    "train_through_recovery": ("T_direct_wobs", "R_direct_wobs"),
    "residual_after_train_through": ("R_direct_raw", "T_direct_wobs"),
    "sense_recovery_train_through": ("T_sense", "R_sense"),
    "random_relu_intrinsic_loss": ("ctrl_world_state", "ctrl_world_state_fresh_random"),
    "untrained_wobs_intrinsic_loss": ("ctrl_world_state", "R_wobs_output"),
}


def _paired(per_seed: List[Dict[str, Any]], norm_key: str, a: str,
            b: str) -> Dict[str, Any]:
    """Per-(seed, split) differences a-b. Returns the mean, the SD, the n, and the
    fraction of cells in which a > b -- the sign-consistency the 1030 write-up reported
    as '8 of 10 cells'."""
    diffs: List[float] = []
    for cell in per_seed:
        pa = cell["probes"].get(a, {}).get(norm_key, {}).get("per_split", {})
        pb = cell["probes"].get(b, {}).get(norm_key, {}).get("per_split", {})
        for k, va in pa.items():
            vb = pb.get(k)
            if va.get("accuracy") is None or vb is None or vb.get("accuracy") is None:
                continue
            diffs.append(float(va["accuracy"] - vb["accuracy"]))
    if not diffs:
        return {"n": 0}
    return {"n": len(diffs),
            "mean": float(statistics.fmean(diffs)),
            "sd": float(statistics.pstdev(diffs)) if len(diffs) > 1 else 0.0,
            "frac_a_greater": float(sum(1 for d in diffs if d > 0) / len(diffs)),
            "min": float(min(diffs)), "max": float(max(diffs))}


def _sub(means: Dict[str, Optional[float]], a: str, b: str) -> Optional[float]:
    if means.get(a) is None or means.get(b) is None:
        return None
    return float(means[a] - means[b])


if __name__ == "__main__":
    raise SystemExit(main())
