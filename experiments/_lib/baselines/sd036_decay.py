"""Canonical OFF/baseline arm for the SD-036 GABAergic-decay validation lineage.

Arm-reuse Phase 0 (instrument-only). Design plan:
REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md (sections 2, 7b).

WHAT THIS MODULE IS
-------------------
Single source of truth for the OFF / baseline arm of the SD-036 cross-stream
decay-regulator validation lineage (V3-EXQ-854 -> 854a -> ...). Being under
`experiments/_lib/**` it is matched by the arm-fingerprint substrate glob, so
any edit here correctly flips the substrate hash and REFUSES a stale reuse.

THE OFF ARM IS `use_gabaergic_decay=False`, NOT `gaba_tone=0.0`
---------------------------------------------------------------
This is the single most important fact in this module and it changed on
2026-07-31 (ree-v3 `35e8969`). Before the harm-stream recurrence landed, the
regulator degenerated to a one-step constant rescale and `gaba_tone=0.0` was
bit-equal to the master switch being off. It no longer is: at tone 0.0 the
DECAY is suspended but the RECURRENCE (the `prev_state` blend in
`LatentStack.encode()`) is still active, so the stream is still stateful. So:

  * legacy / no-regulator control      -> `use_gabaergic_decay=False`   <- THIS MODULE
  * decay suspended, recurrence live   -> `use_gabaergic_decay=True, gaba_tone=0.0`
  * isolate recurrence from decay      -> `gaba_recurrence_z_harm_s/_a=False`

Using tone=0.0 as "the control" would silently compare two stateful substrates
and attribute the recurrence's effect to the decay regulator.

WHY THE SLICE IS NARROW
-----------------------
`off_path_config_slice()` declares ONLY what the OFF computation reads: the
471-lineage env kwargs, the training schedule, the substrate-operating config
every arm shares, and the OFF arm's own flags. It deliberately does NOT include
the ON-arm `gaba_tone` sweep values, the per-stream taus (inert when the master
switch is off), or the acceptance thresholds -- none of those changes the OFF
computation, so folding them in would refuse reuse for no reason (plan 7b
constraint 2).

MINT CONTRACT
-------------
Consumers must call `compute_arm_fingerprint(..., include_driver_script_in_hash=False)`
on this arm, matching the emit side, or a sibling with a different driver can
never match the mint (plan sections 9.4 / 9.7).

ASCII-only output (repo rule).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Path shim -- the _lib idiom. Both `experiments/` (for _harness, pack_writer)
# and `ree-v3/` (for ree_core) must be importable regardless of the caller's cwd.
_THIS = Path(__file__).resolve()
_EXPERIMENTS_DIR = _THIS.parents[2]
_REE_V3_ROOT = _THIS.parents[3]
for _p in (str(_EXPERIMENTS_DIR), str(_REE_V3_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

from _harness import StepHarness  # noqa: E402


# ---------------------------------------------------------------------------
# Environment -- the 471-lineage grid, verbatim from
# tests/contracts/test_sd_036_gabaergic_decay.py (_ENV_KWARGS), which is itself
# matched to V3-EXQ-471 / V3-EXQ-475. Keeping it identical is what makes this
# experiment a read of the SAME phenomenon the origin exemplar exhibited.
# ---------------------------------------------------------------------------
ENV_KWARGS: Dict[str, Any] = dict(
    size=10,
    num_hazards=3,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    use_proxy_fields=True,
    toroidal=False,
    harm_history_len=10,
    limb_damage_enabled=True,
    damage_increment=0.15,
    failure_prob_scale=0.3,
    heal_rate=0.002,
    n_landmarks_b=2,
)

SEEDS: List[int] = [42, 43, 44]

# Training schedule. Both arms train identically; only the regulator differs.
P0_WARMUP_EPISODES = 40
P1_MAIN_EPISODES = 40
TOTAL_TRAIN_EPISODES = P0_WARMUP_EPISODES + P1_MAIN_EPISODES
STEPS_PER_EPISODE = 200

# Eval: one frozen rollout per condition. Long enough that a ~20-step (z_harm)
# and ~50-step (z_harm_a) half-life both have room to express.
EVAL_STEPS = 300

# Latent dims -- shared by every arm.
SELF_DIM = 32
WORLD_DIM = 32
HARM_DIM = 32
Z_HARM_A_DIM = 16
HARM_HISTORY_LEN = 10

# SD-008: alpha_world default 0.3 suppresses events; >= 0.9 is required for any
# experiment whose read depends on z_world fidelity.
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3


def make_env(seed: int) -> CausalGridWorldV2:
    """The lineage environment. Identical for every arm and every tone."""
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def build_config(
    env: CausalGridWorldV2,
    obs_dict: Dict[str, Any],
    *,
    use_gabaergic_decay: bool,
    gaba_tone: float = 1.0,
) -> REEConfig:
    """Shared config builder for BOTH arms.

    The only difference between the OFF and ON arms is `use_gabaergic_decay`
    (and, for the ON arm, the starting `gaba_tone`). Everything else is held
    identical, which is what makes the contrast attributable to the regulator.

    `use_pag_freeze_gate` is left at its default False. That is deliberate and
    load-bearing: MECH-279 consumes `gaba_tone` as a DIRECT SCALAR
    (`exit_threshold = theta_freeze * gaba_tone`, agent.py) and never through
    the decay path, so an active freeze gate would give the tone sweep a second,
    non-decay route to the behavioural readout and confound SD-036 with
    MECH-279.
    """
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        harm_dim=HARM_DIM,
        alpha_world=ALPHA_WORLD,
        alpha_self=ALPHA_SELF,
        reafference_action_dim=env.action_dim,
        use_harm_stream=True,
        z_harm_dim=HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=Z_HARM_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        # Derived, not hardcoded: the affective-harm channel width is an
        # environment property, and a literal here would rot silently into a
        # shape error the moment the env grows a channel.
        harm_obs_a_dim=int(obs_dict["harm_obs_a"].shape[-1]),
        # --- the manipulation ---
        use_gabaergic_decay=use_gabaergic_decay,
        gaba_tone=gaba_tone,
        # --- held OFF to de-confound from MECH-279 (see docstring) ---
        use_pag_freeze_gate=False,
    )


def make_agent(
    env: CausalGridWorldV2,
    obs_dict: Dict[str, Any],
    *,
    use_gabaergic_decay: bool,
    gaba_tone: float = 1.0,
) -> REEAgent:
    return REEAgent(
        build_config(
            env,
            obs_dict,
            use_gabaergic_decay=use_gabaergic_decay,
            gaba_tone=gaba_tone,
        )
    )


def off_path_config_slice() -> Dict[str, Any]:
    """The fingerprint slice for the OFF arm.

    Declares exactly what the OFF computation reads. Deliberately EXCLUDES the
    per-stream taus and the tone sweep (inert when the master switch is off) and
    all acceptance thresholds (not read by the computation at all).
    """
    return {
        "lineage": "sd036_decay",
        "env_kwargs": dict(ENV_KWARGS),
        "schedule": {
            "p0_warmup_episodes": P0_WARMUP_EPISODES,
            "p1_main_episodes": P1_MAIN_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "eval_steps": EVAL_STEPS,
        },
        "latent_dims": {
            "self_dim": SELF_DIM,
            "world_dim": WORLD_DIM,
            "harm_dim": HARM_DIM,
            "z_harm_a_dim": Z_HARM_A_DIM,
            "harm_history_len": HARM_HISTORY_LEN,
            "alpha_world": ALPHA_WORLD,
            "alpha_self": ALPHA_SELF,
        },
        "substrate_flags": {
            "use_harm_stream": True,
            "use_affective_harm_stream": True,
            "use_pag_freeze_gate": False,
        },
        "off_arm_flags": {
            # The OFF arm is the master switch, NOT gaba_tone=0.0. See the
            # module docstring for why that distinction is load-bearing.
            "use_gabaergic_decay": False,
        },
    }


# ---------------------------------------------------------------------------
# Phased training (P0 encoder warmup -> P1 frozen-encoder downstream).
#
# The mandate binds here because P1 trains E2's world-forward on z_world, which
# is an encoder output. Training it jointly with the encoder makes it chase a
# moving target (EXQ-166b/c/d identity collapse, EXQ-085l position noise). So P0
# trains the encoder with NO downstream loss, then P1 freezes the encoder and
# trains the downstream head on `.detach()`ed latents only.
# ---------------------------------------------------------------------------
LR_E1 = 1e-3
LR_E2_WF = 1e-3
LR_ENC_AUX = 1e-4
BATCH_SIZE = 32
WF_BUF_MAX = 2000


def train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    *,
    label: str,
    seed: int,
) -> Dict[str, float]:
    """Train one cell. Identical for both arms -- only the regulator differs.

    Returns a small dict of training diagnostics (not a DV).
    """
    import torch.nn.functional as F
    import torch.optim as optim

    device = agent.device
    stack_params = list(agent.latent_stack.parameters())

    e1_optimizer = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    aux_optimizer = optim.Adam(stack_params, lr=LR_ENC_AUX)
    e2_wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )

    harness = StepHarness(agent, env, train_mode=True, seed=seed)
    wf_buf: List[Any] = []
    wf_losses: List[float] = []

    agent.train()
    for ep in range(TOTAL_TRAIN_EPISODES):
        # PHASE BOUNDARY: episodes [0, P0) are encoder warmup; from P0 on the
        # encoder is frozen and only the downstream head learns.
        in_p0 = ep < P0_WARMUP_EPISODES
        if ep == P0_WARMUP_EPISODES:
            for p in stack_params:
                p.requires_grad_(False)

        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        z_world_prev = None
        action_prev = None

        for _ in range(STEPS_PER_EPISODE):
            result = harness.step(obs_dict)
            latent = result.latent
            z_world_curr = latent.z_world.detach()

            if in_p0:
                # P0: encoder only. No downstream head is trained at all.
                rv = result.next_obs_dict.get("resource_field_view")
                if isinstance(rv, torch.Tensor):
                    prox_t = float(rv.max().item())
                else:
                    prox_t = float(np.max(rv)) if rv is not None else 0.0
                prox_target = torch.tensor([[prox_t]], device=device)
                prox_loss = agent.compute_resource_proximity_loss(prox_target, latent)
                if prox_loss is not None and prox_loss.requires_grad:
                    aux_optimizer.zero_grad()
                    prox_loss.backward()
                    torch.nn.utils.clip_grad_norm_(stack_params, 1.0)
                    aux_optimizer.step()

                if len(agent._world_experience_buffer) >= 2:
                    e1_loss = agent.compute_prediction_loss()
                    if e1_loss.requires_grad:
                        e1_optimizer.zero_grad()
                        e1_loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                        e1_optimizer.step()
            else:
                # P1: encoder frozen. E1 still learns (its gradients cannot reach
                # the frozen encoder), and the downstream world-forward trains on
                # DETACHED latents -- the fixed target the phased protocol exists
                # to provide.
                if len(agent._world_experience_buffer) >= 2:
                    e1_loss = agent.compute_prediction_loss()
                    if e1_loss.requires_grad:
                        e1_optimizer.zero_grad()
                        e1_loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                        e1_optimizer.step()

                if z_world_prev is not None and action_prev is not None:
                    wf_buf.append(
                        (z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu())
                    )
                    if len(wf_buf) > WF_BUF_MAX:
                        wf_buf = wf_buf[-WF_BUF_MAX:]

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
                        e2_wf_optimizer.step()
                        wf_losses.append(float(wf_loss.item()))

            z_world_prev = z_world_curr
            action_prev = result.action.detach()
            obs_dict = result.next_obs_dict
            if result.done:
                break

        if (ep + 1) % 10 == 0 or (ep + 1) == TOTAL_TRAIN_EPISODES:
            print(
                f"  [train] {label} ep {ep + 1}/{TOTAL_TRAIN_EPISODES}",
                flush=True,
            )

    # Restore grad flags so the cell leaves no global side effect behind.
    for p in stack_params:
        p.requires_grad_(True)

    return {
        "train_episodes": float(TOTAL_TRAIN_EPISODES),
        "wf_loss_final": float(np.mean(wf_losses[-50:])) if wf_losses else float("nan"),
        "wf_updates": float(len(wf_losses)),
    }


# ---------------------------------------------------------------------------
# Stream recording -- the measurement instrument.
# ---------------------------------------------------------------------------
def record_stream_trajectories(
    agent: REEAgent,
    env: CausalGridWorldV2,
    *,
    steps: int,
) -> Dict[str, np.ndarray]:
    """Run a FROZEN rollout and record the per-step norm of each decay stream.

    No learning happens here (`train_mode=False`, and no optimizer is
    constructed), so the same trained agent can be re-evaluated at several
    `gaba_tone` values without the earlier evaluations changing its weights.
    """
    out: Dict[str, List[float]] = {"z_harm": [], "z_harm_a": [], "z_beta": []}
    harness = StepHarness(agent, env, train_mode=False)
    _, obs_dict = env.reset()
    agent.reset()
    harness.reset()
    agent.eval()

    with torch.no_grad():
        for _ in range(steps):
            result = harness.step(obs_dict)
            latent = result.latent
            for key in out:
                z = getattr(latent, key, None)
                out[key].append(float(z.norm()) if z is not None else float("nan"))
            obs_dict = result.next_obs_dict
            if result.done:
                _, obs_dict = env.reset()
                harness.reset()

    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Scale-free DVs. These are the shapes the pre-fix substrate made VACUOUS, and
# they are used unchanged here precisely so that a regression to the pre-fix
# wiring shows up as a collapse toward zero spread rather than as a plausible
# result.
# ---------------------------------------------------------------------------
def sustain_ratio(a: np.ndarray) -> float:
    """mean/peak -- the pre-registered falsifier's DV. Exactly scale-free."""
    if a.size == 0:
        return 0.0
    peak = float(np.nanmax(a))
    return float(np.nanmean(a)) / peak if peak > 1e-12 else 0.0


def peak_normalised(a: np.ndarray) -> np.ndarray:
    peak = float(np.nanmax(np.abs(a)))
    return a / peak if peak > 1e-12 else a


def shape_deviation(a: np.ndarray, b: np.ndarray) -> float:
    """Max deviation between two peak-normalised trajectories (scale-free)."""
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    return float(np.nanmax(np.abs(peak_normalised(a[:n]) - peak_normalised(b[:n]))))


def encoder_floor_norms(agent: REEAgent) -> Dict[str, float]:
    """The ambient encoder floor -- the caveat the SD-036 design doc raises.

    `harm_encoder(zeros)` and `affective_harm_encoder(zeros, zeros)` have
    non-zero norm, so the V3-EXQ-471 exemplar's "z_harm_norm pinned at ~0.7
    despite zero harm input" is substantially a floor response to the ambient
    hazard field rather than pure temporal persistence. Decay fights that floor
    to an EQUILIBRIUM; it does not return the stream to zero. Recording the
    floor is what lets a reader tell those two readings apart, and it is why
    observable #1 as literally worded is NOT this experiment's load-bearing
    criterion.

    Returns an empty dict rather than raising if an encoder is absent -- this is
    context, not a gate.
    """
    out: Dict[str, float] = {}
    stack = getattr(agent, "latent_stack", None)
    if stack is None:
        return out
    with torch.no_grad():
        # Dims live on the ENCODER objects, not on the stack config.
        he = getattr(stack, "harm_encoder", None)
        if he is not None:
            try:
                dim = int(getattr(he, "harm_obs_dim", 0))
                if dim > 0:
                    # HarmEncoder.forward -> Tensor
                    out["harm_encoder_zero_norm"] = float(
                        he(torch.zeros(1, dim)).norm()
                    )
            except Exception:
                pass
        ahe = getattr(stack, "affective_harm_encoder", None)
        if ahe is not None:
            try:
                a_dim = int(getattr(ahe, "harm_obs_a_dim", 0))
                h_len = int(getattr(ahe, "harm_history_len", 0))
                if a_dim > 0:
                    hist = torch.zeros(1, h_len) if h_len > 0 else None
                    # AffectiveHarmEncoder.forward -> (z_harm_a, harm_accum_pred)
                    z_a, _ = ahe(torch.zeros(1, a_dim), hist)
                    out["affective_harm_encoder_zero_norm"] = float(z_a.norm())
            except Exception:
                pass
    return out
