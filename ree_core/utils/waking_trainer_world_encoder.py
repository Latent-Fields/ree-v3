"""WorldEncoderMember: SD-070 P0a on the LIVE sense path, as a WakingTrainer member (W6a).

Why this exists
---------------
SD-070's P0a recipe (``ree_core/latent/zworld_p0.py`` ``ZWorldP0Trainer``) trains
``split_encoder.world_encoder`` + ``world_precision_logit`` on RAW ``world_state``
(``_z_world_path``). ``REEAgent.sense`` does not read that path: it reads
``world_obs_encoder`` (a separately initialised ``Linear + ReLU``) -> ``latent_stack.encode``
(split encoder, top-down conditioning, SD-007 reafference, the SD-008 alpha_world EMA). So
the recipe optimises a distribution the agent never senses, and leaves ``world_obs_encoder``
a frozen random projection (``SD-ZWORLD-SENSE-PATH-PARITY``; failure autopsy V3-EXQ-1030;
gradient-reach census ``940c690c9dd`` row 5). Only DRIVERS ever invoked the recipe, once,
offline, before the run (``experiments/_lib/zworld_p0_warmup.run_zworld_p0``).

Coupled-loop-repair campaign plan (REE_assembly
``evidence/planning/coupled_loop_repair_campaign_plan.md`` sec 3 W-trainer, user decision
Q4c) moves the P0a objective onto the sense path as a waking-trainer member: this module.

The trained path (the ZSelfP0 ``_native_chain`` pattern, ree-v3 ``863d23d``)
---------------------------------------------------------------------------
The member records the RAW sensory inputs of each waking tick (``on_sense``, from the
existing ``WakingTrainer.on_sense`` hook in ``REEAgent.sense``) and, at update time, re-runs
the agent's OWN modules over a recorded warm-up window, WITH gradient::

    prev = latent_stack.init_state(B)                 # fresh, detached: no graph survives
    for each recorded tick j of the window:           # exactly what sense() does after reset
        enc  = cat(body_obs_encoder(body_j), world_obs_encoder(world_j))
        prev = latent_stack.encode(enc, prev, prev_action=a_{j-1}, harm_obs=..., ...)
    z = prev.z_world                                  # the SENSED z_world of the last tick

The same module objects ``sense`` calls -- not a copy and not the direct
``world_encoder(world_state)`` shortcut -- so an optimizer step here changes the very next
``sense`` output (contract W6a-04), and it bumps the ``_version`` of the read-path tensors
the W3 ``E2WorldMember`` re-encode cache is keyed on, so W3's next replay re-encodes
through the updated encoder (contract W6a-06). The window is ``auto_reencode_window``
(alpha_world 0.3 -> 26 warm-up ticks, 0.9 -> 4; fewer at an episode start): the
dropped prefix weighs <= 1e-4 in the EMA, so the chain's z is the live latent to that
tolerance (contract W6a-05).

Objective: the SD-070 P0a recipe, reused, not re-derived
--------------------------------------------------------
On the batch of sensed z (``ZWorldP0Config`` defaults, the measured operating point):
four static scene-structure grounding heads (``scene_structure_targets`` of the tick's own
raw ``world_obs``: hazard / resource presence and bucketed Chebyshev distance) under
class-balanced CE (``balanced_class_weights`` over the member's CURRENT buffer), the VICReg
variance + covariance penalty (``variance_covariance_penalty``), the world_obs
reconstruction head, and -- only when the stack has ``resource_proximity_head`` and
world_obs carries the SD-018 field slice -- the resource-proximity MSE (target
``max(world_obs[225:250])``, the same value ``resource_prox_target`` reads from
``resource_field_view``). The grounding / reconstruction heads are the TRAINER's (as in
``ZWorldP0Trainer``): built here under a forked, seeded RNG so constructing them draws
nothing from the global stream, and held outside ``REEAgent.state_dict()``. Not carried:
the SD-018-amend directional-field leg and the SD-106 preservation leg (both default 0.0
in ``ZWorldP0Config``); ``world_encoder_skip`` IS trained when the stack has it.
Gradient-norm clip ``grad_clip`` (SD-070 ``max_grad_norm`` 1.0) applies to the encoder
path only, as in the recipe (``clip_parameters``).

What changes relative to the offline recipe, stated
---------------------------------------------------
* ONLINE: one mini-batch per update from a sliding buffer (``waking_trainer_buffer_max``,
  oldest ticks age out), not 12 epochs over a fixed rollout; there is no held-out split.
* SENSED z: the loss reads the EMA-smoothed, top-down-conditioned z_world. At alpha_world
  0.3 the last frame contributes 30% of it, so the static single-frame targets are partly
  confounded by the previous ticks -- the SD-008 floor (alpha >= 0.9) makes them 90%.
* The group is the WORLD read path: ``world_obs_encoder`` + ``split_encoder.world_encoder``
  + ``world_precision_logit`` (+ ``world_encoder_skip`` / ``resource_proximity_head`` when
  present) + the trainer heads. Gradient from the sensed z also reaches the depth stack
  (``beta/theta/delta_encoder``, ``*_to_*``, ``world_topdown``; census 940c690c9dd #13,
  DEFERRED) and, through top-down, the z_self path (``body_obs_encoder``,
  ``self_encoder``; owned by ``sd_zself_training_path``). Those tensors are NOT stepped
  here: the guard reports them as G5 leaks (report-only), contract W6a-03 pins the
  leak set to exactly those families, and the trainer restores their ``.grad`` after each
  update (``restore_outside_grads``), so nothing accumulates there and a driver's own
  gradients on them are left as they were.

Buffer staleness for the OTHER members (plan P8)
------------------------------------------------
With this member ON the encoder moves, so every member that replays a STORED latent goes
stale: ``HarmEvalMember`` and ``CodecMember`` (stored z_world), ``E1Member`` (the agent's
own experience buffers) and the W3 ``E2WorldMember`` in ``replay_latent="stored"`` mode.
The W3 member in its default ``"reencode"`` mode re-encodes through the current read path
and does not go stale. Which policy each stored-latent member needs is probe N2's
question; nothing here changes them.

Default OFF, bit-identical: constructed only when ``waking_trainer_enabled`` and
``waking_trainer_world_encoder_enabled`` are both True, and imported only then.
Contract: ``tests/contracts/test_w6a_world_encoder_member.py``. Evidence domain of the
member's own checks: D1 (gradient reaches the sense path and the weights move). Whether the
trained encoder helps any consumer is N2 / A1, not claimed here.

MECH-094: not applicable -- trains on recorded waking observations only.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.latent.zworld_p0 import (
    RESOURCE_FIELD_SLICE,
    ZWorldP0Config,
    balanced_class_weights,
    scene_structure_targets,
    variance_covariance_penalty,
)
from ree_core.utils.waking_trainer import (
    WakingTrainerMember,
    _raw_tensor,
    _resolve_named,
    auto_reencode_window,
)

_TARGET_KEYS = ("hazard_present", "resource_present", "hazard_distance", "resource_distance")


class WorldEncoderMember(WakingTrainerMember):
    """SD-070 P0a objective on the live sense path (W6a). See the module docstring."""

    name = "world_encoder"
    # The sensed z also reaches the depth stack and the z_self path (not stepped here):
    # the trainer restores those tensors' .grad after each update (module docstring).
    restore_outside_grads = True

    def __init__(self, agent: Any, lr: float = 1e-3, batch_size: int = 64,
                 buffer_max: int = 2000, window: int = 0, grad_clip: float = 1.0,
                 updates_per_step: int = 1, seed: int = 0,
                 p0_config: Optional[ZWorldP0Config] = None) -> None:
        self._agent = agent
        self.lr = float(lr)
        self.batch_size = max(2, int(batch_size))
        self.grad_clip = float(grad_clip) if grad_clip and float(grad_clip) > 0 else None
        self.updates_per_step = max(1, int(updates_per_step))
        self.cfg = p0_config or ZWorldP0Config()
        ww = int(window)
        self.window = (auto_reencode_window(agent.config.latent.alpha_world)
                       if ww <= 0 else ww)
        ls = agent.latent_stack
        se = ls.split_encoder
        self._se = se
        self.world_dim = int(se.world_dim)
        self.world_obs_dim = int(agent.config.latent.world_obs_dim)

        # The encoder read path (agent-level names, so the guard reports real names).
        enc_mods: List[nn.Module] = [agent.world_obs_encoder, se.world_encoder]
        skip = getattr(se, "world_encoder_skip", None)
        if skip is not None:
            enc_mods.append(skip)
        self._prox_head = getattr(se, "resource_proximity_head", None)
        self.use_prox = (self._prox_head is not None and self.cfg.proximity_weight > 0.0
                         and self.world_obs_dim >= RESOURCE_FIELD_SLICE.stop)
        if self.use_prox:
            enc_mods.append(self._prox_head)
        named = _resolve_named(agent, enc_mods)
        ids = {id(p) for _, p in named}
        named += [(n, p) for n, p in agent.named_parameters()
                  if p is se.world_precision_logit and id(p) not in ids]
        self._encoder_named = named

        # Trainer-owned heads (as in ZWorldP0Trainer), built under a forked, seeded RNG so
        # constructing them draws nothing from the global torch stream.
        n_classes = {"hazard_present": 2, "resource_present": 2,
                     "hazard_distance": int(self.cfg.n_distance_buckets),
                     "resource_distance": int(self.cfg.n_distance_buckets)}
        self._n_classes = n_classes
        dev = agent.device
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 70)
            self.heads = nn.ModuleDict({k: nn.Linear(self.world_dim, c)
                                        for k, c in n_classes.items()}).to(dev)
            self.recon_head = (nn.Linear(self.world_dim, self.world_obs_dim).to(dev)
                               if self.cfg.reconstruction_weight > 0.0 else None)
        head_named = [("world_encoder_member.heads." + n, p)
                      for n, p in self.heads.named_parameters()]
        if self.recon_head is not None:
            head_named += [("world_encoder_member.recon_head." + n, p)
                           for n, p in self.recon_head.named_parameters()]
        self._head_named = head_named

        # Buffer: records share the per-tick raw tensors (no per-record copy).
        self._buf: Deque[Dict[str, Any]] = deque(maxlen=int(buffer_max))
        self._hist: Deque[Tuple[Any, ...]] = deque(maxlen=self.window + 1)
        self._cur_raw: Optional[Tuple[Any, ...]] = None
        self._last_a: Optional[torch.Tensor] = None
        self.n_observed = 0
        self.n_chain_encodes = 0
        self.last_terms: Dict[str, float] = {}

    # -- parameters ------------------------------------------------------------------------
    def named_parameters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        return list(self._encoder_named) + list(self._head_named)

    def encoder_named_parameters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        return list(self._encoder_named)

    def clip_parameters(self) -> List[torch.nn.Parameter]:
        """SD-070 clips the encoder path only (``max_grad_norm`` over world params)."""
        return [p for _, p in self._encoder_named]

    # -- recording ---------------------------------------------------------------------------
    def on_sense(self, obs_body: Any, obs_world: Any, obs_harm: Any = None,
                 obs_harm_a: Any = None, obs_harm_history: Any = None) -> None:
        self._cur_raw = (_raw_tensor(obs_body), _raw_tensor(obs_world), _raw_tensor(obs_harm),
                         _raw_tensor(obs_harm_a), _raw_tensor(obs_harm_history))

    def on_env_reset(self) -> None:
        self._hist.clear()
        self._cur_raw = None
        self._last_a = None

    def observe(self, agent: Any, harm_signal: float) -> None:
        raw = self._cur_raw
        self._cur_raw = None
        if raw is None or raw[1] is None:
            return
        self.n_observed += 1
        # prev_a = the action executed before THIS sensed tick (what sense() fed as the
        # SD-007 prev_action); None at an episode start.
        self._hist.append(raw + (self._last_a,))
        win = tuple(self._hist)
        tgt = scene_structure_targets(raw[1], n_distance_buckets=self.cfg.n_distance_buckets)
        self._buf.append({
            "obs": tuple(w[:5] for w in win),
            "prev_a": (None,) + tuple(w[5] for w in win[1:]),
            "targets": {k: tgt[k].reshape(1) for k in _TARGET_KEYS},
            "step": self.n_observed,
        })
        act = agent._last_action
        self._last_a = None if act is None else act.detach().reshape(1, -1).clone()

    # -- the native chain ----------------------------------------------------------------------
    def sensed_z_world(self, recs: List[Dict[str, Any]]) -> torch.Tensor:
        """Sensed z_world of each record's LAST tick, re-run through the agent's own read
        path from ``init_state`` (gradient flows when the caller enables it). Rows are
        returned in ``recs`` order."""
        agent = self._agent
        dev = agent.device
        ls = agent.latent_stack
        out: List[Optional[torch.Tensor]] = [None] * len(recs)
        groups: Dict[int, List[int]] = {}
        for i, r in enumerate(recs):
            groups.setdefault(len(r["obs"]), []).append(i)
        vol = (agent.e3.volatility_estimate
               if agent.config.latent.volatility_signal_dim > 0 else None)
        for L, idx in groups.items():
            prev = ls.init_state(batch_size=len(idx), device=dev)
            for j in range(L):
                def col(k: int) -> Optional[torch.Tensor]:
                    xs = [recs[i]["obs"][j][k] for i in idx]
                    if any(x is None for x in xs):
                        return None
                    return torch.cat(xs, dim=0).to(dev)
                ob, ow, harm = col(0), col(1), col(2)
                if agent.lpb_router is not None and harm is not None:
                    harm = agent.lpb_router.mask_external_harm_obs(harm)
                pas = [recs[i]["prev_a"][j] for i in idx]
                pa = (None if any(p is None for p in pas)
                      else torch.cat(pas, dim=0).to(dev).float())
                enc = torch.cat([agent.body_obs_encoder(ob), agent.world_obs_encoder(ow)],
                                dim=-1)
                prev = ls.encode(enc, prev, prev_action=pa, harm_obs=harm,
                                 harm_obs_a=col(3), harm_history=col(4),
                                 volatility_signal=vol)
                self.n_chain_encodes += 1
            for n, i in enumerate(idx):
                out[i] = prev.z_world[n:n + 1]
        return torch.cat(out, dim=0)  # type: ignore[arg-type]

    # -- replay ------------------------------------------------------------------------------
    def ready(self) -> bool:
        return len(self._buf) >= self.batch_size

    def objective(self, z: torch.Tensor, recs: List[Dict[str, Any]]) -> torch.Tensor:
        """The SD-070 P0a loss on a batch of sensed z (``ZWorldP0Trainer.train``'s terms)."""
        cfg = self.cfg
        dev = z.device
        world = torch.cat([r["obs"][-1][1] for r in recs], dim=0).to(dev)
        loss = z.sum() * 0.0
        terms: Dict[str, float] = {}
        for k, head in self.heads.items():
            w_ = cfg.presence_weight if k.endswith("_present") else cfg.distance_weight
            if w_ <= 0.0:
                continue
            y = torch.cat([r["targets"][k] for r in recs]).to(dev)
            all_y = torch.cat([r["targets"][k] for r in self._buf]).to(dev)
            ce = F.cross_entropy(head(z), y,
                                 weight=balanced_class_weights(all_y, self._n_classes[k]))
            loss = loss + w_ * ce
            terms["ce_" + k] = float(ce.detach())
        if self.use_prox:
            prox = world[:, RESOURCE_FIELD_SLICE].max(dim=1).values
            mse = F.mse_loss(self._prox_head(z).reshape(-1), prox)
            loss = loss + cfg.proximity_weight * mse
            terms["proximity"] = float(mse.detach())
        if self.recon_head is not None:
            rec = F.mse_loss(self.recon_head(z), world)
            loss = loss + cfg.reconstruction_weight * rec
            terms["reconstruction"] = float(rec.detach())
        var_t, cov_t = variance_covariance_penalty(z, cfg.variance_gamma)
        loss = loss + cfg.variance_weight * var_t + cfg.covariance_weight * cov_t
        terms["variance"] = float(var_t.detach())
        terms["covariance"] = float(cov_t.detach())
        self.last_terms = terms
        return loss

    def loss(self, agent: Any) -> Optional[torch.Tensor]:
        if not self.ready():
            return None
        ii = torch.randint(0, len(self._buf), (self.batch_size,)).tolist()
        recs = [self._buf[i] for i in ii]
        z = self.sensed_z_world(recs)
        return self.objective(z, recs)
