"""WakingTrainer: a ree_core-owned native waking trainer (skeleton + harm_eval member).

Why this exists
---------------
The 2026-09-25 gradient-reach census (REE_assembly
``evidence/planning/gradient_reach_census_20260925.md``, commit 940c690c9dd) measured that
at ``REEConfig`` defaults the agent does NO waking gradient learning: 40 train-mode
``StepHarness`` ticks construct 0 optimizers, and every trained parameter in REE is trained
by a DRIVER-built optimizer, each recipe a different subset. ``e3.harm_eval_head`` -- read
on every E3 tick through ``compute_harm_cost_fallback`` under a comment calling it
"TRAINED" -- was untrained in 4 of 5 recipes (census section 4 row 10; GFLAG-0491). The
design record (``evidence/planning/native_waking_trainer_design_20260925.md``, commit
0c0f5b76ec) specifies an agent-owned trainer with per-module optimizer groups, stepped from
``update_residue()`` every K ticks, armed with a gradient-reach guard. The user chose the
HYBRID (design section 6, D1): this object is the spine; the existing phased trainers
(SD-070 ``ZWorldP0Trainer``, ``ZSelfP0Trainer``) become scheduled members later; per-tick
live-latent losses are reserved for the REINFORCE heads only.

What this landing contains (and deliberately does not)
------------------------------------------------------
* The trainer spine: member registration, one optimizer per member group, the K-tick
  cadence, RNG isolation, the armed guard.
* ``HarmEvalMember`` (design section 1a row 5 / section 4a M3): always registered when
  the trainer is ON.
* T1 (coupled-loop-repair campaign plan, REE_assembly
  ``evidence/planning/coupled_loop_repair_campaign_plan.md`` section 3 W-trainer):
  ``E1Member`` (map rows 1-2, the native ``compute_prediction_loss``) behind
  ``waking_trainer_e1_enabled`` and ``E2SelfMember`` (map row 3, the native
  ``compute_e2_loss`` over the member's OWN transitions, design section 2 (i) point 3)
  behind ``waking_trainer_e2_self_enabled``. Both default OFF; with them OFF an ON
  trainer is the C1 trainer exactly (one member).
* ``CodecMember`` (``waking_trainer_codec.py``; W1 codec part (1), integration branch)
  behind ``waking_trainer_codec_enabled`` (default OFF).
* NOT registered here (the coupled campaign's branch work): E2-world, SD-070 P0, ZSelfP0,
  the terrain prior. ``WakingTrainer.register`` is the seam they plug into.

Retained-graph hazard (design section 2 (iii)): ``agent._last_action`` is NOT detached by
default and carries the previous tick's selection graph (SD-007 reafference); an
undetached action in a replay buffer makes a replay backward walk E3's graph and fail with
"modified by an inplace operation" (``REEAgent.record_transition`` docstring). Every sample
a member records is therefore ``.detach().clone()``-ed at record time, and E1 replays the
agent's own already-detached experience buffers. A second, ON-only exposure is stated
rather than hidden: an update steps parameters IN PLACE inside ``update_residue``, so a
DRIVER that holds a live tick graph through a trained module (e.g. a REINFORCE log-prob
through E3 scoring) and calls ``backward()`` on it AFTER ``update_residue`` would hit the
same in-place error. No ree_core path does that at defaults (the census found 0 native
optimizers); drivers that train their own heads should keep the trainer OFF or backprop
before ``update_residue``.

Default OFF, bit-identical
--------------------------
The agent builds this object ONLY when ``config.waking_trainer_enabled`` is True. When OFF
nothing is constructed, nothing is imported, no buffer exists and no RNG is drawn; the agent
is byte-identical to the pre-change code (pinned by
``tests/contracts/test_waking_trainer.py``).

RNG when ON
-----------
Every member update runs inside ``torch.random.fork_rng`` with the trainer's OWN private
torch RNG state swapped in and saved back afterwards (design section 2, "RNG": measured
neutral). Python ``random`` and ``numpy`` states are also snapshotted and restored around
the update. So the act path draws exactly the numbers it would have drawn with the trainer
absent; only the trained weights differ. Recording a replay sample draws nothing.

The guard (design section 3; user/orchestrator decision Q4d)
------------------------------------------------------------
Each member group gets a ``GradReachGuard`` (``ree_core/utils/grad_reach_guard.py``) armed
for the group's first ``waking_trainer_guard_min_steps`` optimizer steps. When the window
closes the verdict is computed once: FAIL raises ``WakingTrainerReachError`` (a dead group
must stop the run, not annotate a manifest); CANNOT_DETERMINE after a full window (an empty
or fully-allowlisted group) also raises, as a distinct message, because an ON trainer whose
group cannot be checked is a structural defect. The guard then disarms, so its
steady-state cost is zero. The frozen-by-design allowlist is the guard's explicit
``FROZEN_BY_DESIGN`` table; none of its entries is in the harm_eval group.

Evidence domain of a PASS: D1 (gradient reaches the group's parameters and they move). It
says nothing about whether the trained head changes any consumer or behaviour.

Retained graph: none by construction. Every member trains from DETACHED replay samples, so
a member's backward cannot reach the live tick graph (design section 2 (i)).
"""

from __future__ import annotations

import random
from collections import OrderedDict, deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ree_core.utils.grad_reach_guard import (
    CANNOT_DETERMINE,
    FAIL,
    FROZEN_BY_DESIGN,
    PASS,
    GradReachGuard,
    GradReachResult,
)


class WakingTrainerReachError(RuntimeError):
    """Raised when a member group's gradient-reach guard FAILs (or cannot determine)."""


class WakingTrainerMember:
    """Interface for one trainer group.

    A member owns: a name, the named parameters its optimizer holds, a replay buffer it
    fills in ``observe``, and a replay loss. It must train from DETACHED recorded samples
    only (no live tick graph), and may draw randomness from the GLOBAL torch RNG only --
    the trainer swaps its private state in around ``loss``.

    To register one of the existing phased trainers (SD-070 P0, ZSelfP0) later, wrap it in
    a subclass whose ``observe`` records the raw observations the trainer's buffer needs
    and whose ``loss`` returns one batch of that trainer's objective.
    """

    name: str = "member"
    lr: float = 1e-3

    def named_parameters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        raise NotImplementedError

    def observe(self, agent: Any, harm_signal: float) -> None:
        raise NotImplementedError

    def ready(self) -> bool:
        raise NotImplementedError

    def loss(self, agent: Any) -> Optional[torch.Tensor]:
        """One replay-batch loss, or None when the buffer is not ready."""
        raise NotImplementedError

    def on_env_reset(self) -> None:
        """Episode / env boundary (``REEAgent.notify_env_reset``). Default: nothing."""
        return None


def _resolve_named(agent: Any, modules: Sequence[torch.nn.Module]) -> List[Tuple[str, torch.nn.Parameter]]:
    """Agent-level (name, param) pairs for the trainable params of ``modules``."""
    ids = {id(p) for m in modules for p in m.parameters() if p.requires_grad}
    return [(n, p) for n, p in agent.named_parameters() if id(p) in ids]


class HarmEvalMember(WakingTrainerMember):
    """``e3.harm_eval_head`` regression on experienced harm (design section 1a row 5).

    Target: ``max(-harm_signal, 0)`` -- the experienced harm magnitude, 0 on a benign or
    benefit step. ``harm_signal`` is the env reward passed to ``update_residue``
    (negative = harm, positive = benefit; ``experiments/_harness.py`` StepResult). The
    design's module->loss map writes "abs(harm_signal)"; the precedent it cites
    (``experiments/_lib/goal_pipeline_tier1.py`` harm_eval buffer append) clamps to the
    harm side, and ``abs`` would train BENEFIT steps as harm, so the clamp is used.
    Input: the tick's ``z_world`` (``agent._current_latent.z_world``, the latent sensed
    before the action whose consequence ``harm_signal`` is), DETACHED -- the same
    (z_world_t, harm_t) pairing as that precedent. Loss: MSE of the head's sigmoid
    output against the target, over a uniformly-sampled replay batch.
    """

    name = "harm_eval"

    def __init__(self, agent: Any, lr: float, batch_size: int, buffer_max: int) -> None:
        self._head = agent.e3.harm_eval_head
        self._harm_eval = agent.e3.harm_eval
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self._buf: Deque[Tuple[torch.Tensor, float]] = deque(maxlen=int(buffer_max))
        # Resolve names against the agent so the guard reports agent-level names.
        ids = {id(p) for p in self._head.parameters()}
        self._named = [(n, p) for n, p in agent.named_parameters() if id(p) in ids]

    def named_parameters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        return list(self._named)

    def observe(self, agent: Any, harm_signal: float) -> None:
        lat = agent._current_latent
        if lat is None or getattr(lat, "z_world", None) is None:
            return
        z = lat.z_world.detach().reshape(1, -1).clone()
        target = float(-harm_signal) if harm_signal < 0 else 0.0
        self._buf.append((z, target))

    def ready(self) -> bool:
        return len(self._buf) >= self.batch_size

    def loss(self, agent: Any) -> Optional[torch.Tensor]:
        if not self.ready():
            return None
        idx = torch.randperm(len(self._buf))[: self.batch_size].tolist()
        zw = torch.cat([self._buf[i][0] for i in idx], dim=0)
        tgt = torch.tensor([[self._buf[i][1]] for i in idx], dtype=zw.dtype, device=zw.device)
        pred = self._harm_eval(zw)
        return F.mse_loss(pred, tgt)


class E1Member(WakingTrainerMember):
    """E1 multi-step latent prediction (design section 1a rows 1-2; T1).

    Loss: the agent's NATIVE ``compute_prediction_loss`` -- unchanged -- over the agent's
    own ``_self/_world_experience_buffer`` (appended DETACHED inside the E1 tick, so the
    replay backward cannot reach the live tick graph). ``observe`` records nothing: the
    agent fills those buffers itself. ``compute_prediction_loss`` saves and restores
    ``e1._hidden_state`` around its rollout, so the live E1 recurrence is untouched; its one
    ``torch.randint`` runs on the trainer's private RNG state (``WakingTrainer._update``).

    Group: every trainable ``agent.e1`` tensor (the design's A_e1 recipe, guard PASS on
    seeds 42/43). ``e1.context_memory.write_gate`` stays IN the group on purpose: it is in
    ``FROZEN_BY_DESIGN`` (non-gradient write under ``torch.no_grad``), so the guard's G3
    check verifies on every ON run that it still receives no gradient -- a stale allowlist
    entry would FAIL rather than hide.
    """

    name = "e1"

    def __init__(self, agent: Any, lr: float) -> None:
        self._agent = agent
        self.lr = float(lr)
        self._named = _resolve_named(agent, [agent.e1])

    def named_parameters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        return list(self._named)

    def observe(self, agent: Any, harm_signal: float) -> None:
        return None  # the agent's own E1 tick fills the (detached) experience buffers

    def ready(self) -> bool:
        return len(self._agent._world_experience_buffer) >= 2

    def loss(self, agent: Any) -> Optional[torch.Tensor]:
        if not self.ready():
            return None
        return agent.compute_prediction_loss()


class E2SelfMember(WakingTrainerMember):
    """E2 motor-sensory forward model on the member's OWN transitions (map row 3; T1).

    Records (z_self_t, a_t, z_self_{t+1}) itself from ``agent._current_latent`` and
    ``agent._last_action`` at each waking ``update_residue`` call (design section 2 (i)
    point 3), so a driver that never calls ``record_transition`` cannot silently starve
    the group. At the tick-t call ``_current_latent`` is the latent sensed at t and
    ``_last_action`` is a_t (StepHarness order: sense -> act -> env.step ->
    update_residue); the pair is completed by the NEXT call's z_self. All three tensors
    are ``.detach().clone()``-ed at record time (the retained-graph hazard: ``_last_action``
    is undetached by default). A pending pair is dropped at an env boundary
    (``on_env_reset``, fired by ``REEAgent.notify_env_reset``, which ``reset()`` calls) and
    whenever the agent's step counter went backwards, so no cross-episode transition is
    recorded.

    Loss: the agent's NATIVE ``compute_e2_loss`` -- unchanged -- run with the member's
    buffer swapped in for ``agent._e2_transition_buffer`` for the duration of the call and
    restored after (the harness-filled agent buffer is never read or modified).
    Group: ``e2.self_transition`` + ``e2.self_action_encoder``, the only tensors
    ``predict_next_self`` reads.
    """

    name = "e2_self"

    def __init__(self, agent: Any, lr: float, batch_size: int, buffer_max: int) -> None:
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self._buf: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(
            maxlen=int(buffer_max))
        self._pending: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None
        self._named = _resolve_named(agent, [agent.e2.self_transition,
                                             agent.e2.self_action_encoder])

    def named_parameters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        return list(self._named)

    def on_env_reset(self) -> None:
        self._pending = None

    def observe(self, agent: Any, harm_signal: float) -> None:
        lat = agent._current_latent
        z = getattr(lat, "z_self", None) if lat is not None else None
        step = int(getattr(agent, "_step_count", 0))
        if z is None:
            self._pending = None
            return
        z = z.detach().reshape(1, -1).clone()
        pend = self._pending
        if pend is not None and step >= pend[2]:
            self._buf.append((pend[0], pend[1], z))
        act = agent._last_action
        self._pending = (None if act is None
                         else (z, act.detach().reshape(1, -1).clone(), step))

    def ready(self) -> bool:
        return len(self._buf) >= max(2, self.batch_size)

    def loss(self, agent: Any) -> Optional[torch.Tensor]:
        if not self.ready():
            return None
        saved = agent._e2_transition_buffer
        agent._e2_transition_buffer = list(self._buf)
        try:
            return agent.compute_e2_loss(batch_size=self.batch_size)
        finally:
            agent._e2_transition_buffer = saved


class WakingTrainer:
    """Agent-owned waking trainer: per-member optimizers, K-tick cadence, armed guard.

    Plain Python object (NOT an ``nn.Module``), so building it adds nothing to the agent's
    ``parameters()`` / ``state_dict()``.
    """

    def __init__(self, agent: Any, config: Any, members: Optional[Sequence[WakingTrainerMember]] = None) -> None:
        self._agent = agent
        self.every_k = max(1, int(getattr(config, "waking_trainer_every_k", 1)))
        self.guard_min_steps = int(getattr(config, "waking_trainer_guard_min_steps", 8))
        seed = int(getattr(config, "waking_trainer_seed", 0))
        # Private torch RNG state, created without touching the global generator.
        self._rng_state: torch.Tensor = torch.Generator().manual_seed(seed).get_state()
        self.members: "OrderedDict[str, WakingTrainerMember]" = OrderedDict()
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.steps: Dict[str, int] = {}
        self._guards: Dict[str, Optional[GradReachGuard]] = {}
        self.guard_results: Dict[str, GradReachResult] = {}
        self.ticks: int = 0
        if members is None:
            batch_size = int(getattr(config, "waking_trainer_batch_size", 16))
            buffer_max = int(getattr(config, "waking_trainer_buffer_max", 2000))
            members = [HarmEvalMember(
                agent,
                lr=float(getattr(config, "waking_trainer_harm_eval_lr", 1e-3)),
                batch_size=batch_size,
                buffer_max=buffer_max,
            )]
            # T1 members: each behind its own default-OFF knob (read defensively).
            if bool(getattr(config, "waking_trainer_e1_enabled", False)):
                members.append(E1Member(
                    agent, lr=float(getattr(config, "waking_trainer_e1_lr", 1e-3))))
            if bool(getattr(config, "waking_trainer_e2_self_enabled", False)):
                members.append(E2SelfMember(
                    agent,
                    lr=float(getattr(config, "waking_trainer_e2_self_lr", 1e-3)),
                    batch_size=batch_size,
                    buffer_max=buffer_max,
                ))
            # W1 codec member (coupled-loop-repair campaign plan sec 3 W1 part (1)):
            # behind its own default-OFF knob, read defensively; OFF -> never imported.
            if bool(getattr(config, "waking_trainer_codec_enabled", False)):
                from ree_core.utils.waking_trainer_codec import CodecMember
                members.append(CodecMember(
                    agent,
                    lr=float(getattr(config, "waking_trainer_codec_lr", 1e-3)),
                    batch_size=int(getattr(config, "waking_trainer_batch_size", 16)),
                    buffer_max=int(getattr(config, "waking_trainer_buffer_max", 2000)),
                    code_l2=float(getattr(config, "waking_trainer_codec_code_l2", 1e-3)),
                ))
        for m in members:
            self.register(m)

    # -- registration ------------------------------------------------------------------
    def register(self, member: WakingTrainerMember) -> None:
        if member.name in self.members:
            raise ValueError("duplicate waking-trainer member: %s" % member.name)
        named = member.named_parameters()
        if not named:
            raise ValueError("waking-trainer member %s holds no parameters" % member.name)
        self.members[member.name] = member
        self.optimizers[member.name] = torch.optim.Adam([p for _, p in named], lr=member.lr)
        self.steps[member.name] = 0
        self._guards[member.name] = GradReachGuard(
            named_parameters=self._agent.named_parameters(),
            allowlist=FROZEN_BY_DESIGN,
            min_steps=self.guard_min_steps,
        ) if self.guard_min_steps > 0 else None

    def group_names(self, member: str) -> List[str]:
        return [n for n, _ in self.members[member].named_parameters()]

    def on_env_reset(self) -> None:
        """Env / episode boundary (from ``REEAgent.notify_env_reset``). Draws nothing."""
        for m in self.members.values():
            m.on_env_reset()

    # -- the per-step entry (called from REEAgent.update_residue) ------------------------
    def on_waking_step(self, harm_signal: float) -> Dict[str, Any]:
        """Record this tick's replay samples; every K ticks, step each ready member."""
        out: Dict[str, Any] = {}
        for m in self.members.values():
            m.observe(self._agent, float(harm_signal))
        self.ticks += 1
        if self.ticks % self.every_k != 0:
            return out
        for name, m in self.members.items():
            if not m.ready():
                continue
            loss_val = self._update(name, m)
            if loss_val is not None:
                out["waking_trainer_%s_loss" % name] = loss_val
        return out

    def _update(self, name: str, member: WakingTrainerMember) -> Optional[float]:
        opt = self.optimizers[name]
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(self._rng_state)
                try:
                    with torch.enable_grad():
                        loss = member.loss(self._agent)
                        if loss is None or not loss.requires_grad:
                            return None
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        guard = self._guards.get(name)
                        if guard is not None:
                            guard.observe_optimizer(opt)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        loss_val = float(loss.detach().item())
                finally:
                    self._rng_state = torch.get_rng_state()
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
        self.steps[name] += 1
        self._maybe_close_guard(name)
        return loss_val

    def _maybe_close_guard(self, name: str) -> None:
        guard = self._guards.get(name)
        if guard is None or self.steps[name] < self.guard_min_steps:
            return
        res = guard.result()
        self.guard_results[name] = res
        self._guards[name] = None  # disarm: the check is structural
        if res.status == FAIL:
            raise WakingTrainerReachError(
                "waking-trainer group '%s' FAILED the gradient-reach guard:\n%s"
                % (name, res.summary()))
        if res.status == CANNOT_DETERMINE:
            raise WakingTrainerReachError(
                "waking-trainer group '%s': gradient-reach guard CANNOT_DETERMINE after a "
                "full %d-step window (empty or fully-allowlisted group):\n%s"
                % (name, self.guard_min_steps, res.summary()))
        assert res.status == PASS

    def report(self) -> Dict[str, Any]:
        """ASCII-safe summary for manifests: steps and guard verdict per group."""
        return {
            name: {
                "steps": self.steps[name],
                "guard": (self.guard_results[name].status if name in self.guard_results
                          else ("armed" if self._guards.get(name) is not None else "off")),
                "n_tensors": len(self.members[name].named_parameters()),
            }
            for name in self.members
        }
