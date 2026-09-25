"""CodecMember: the W1 joint action codec as a WakingTrainer member group (W1 part (1)).

Why this exists
---------------
The CEM proposer samples in action-object space O, decodes with
``hippocampal.action_object_decoder`` and rolls the decoded vector out through E2, which
re-encodes each step with ``e2.action_object_head``; the elite refit then moves toward
those encoder codes. The two maps were never tied: no loss in ree_core reaches either
(``action_decoder_training_trace_20260924.md``; SD-080 for the encoder half), so the
untrained composition pins the pool to one bias class. Training the decoder ALONE on the
frozen encoder's image (trace candidate 1) was probed and FAILED
(``action_decoder_training_causal_probe_20260925.md``, a369f411ff8): an honest inverse
did no better than a label-shuffled one, because the O-space interface around it was
also broken (raw decoder logits as the rollout action; iteration-0 samples ~12x outside
the encoder image). Those two interface defects are parts (2) and (3) of W1, built in
``HippocampalModule`` behind ``use_codec_bounded_decode`` and
``use_codec_iter0_image_match``. This member is part (1): the JOINT codec (trace
candidate 2).

Objective
---------
For a replay batch of recorded ``z_world`` states and EVERY action class c::

    o      = e2.action_object(z_world, onehot(c))        # encoder, no cue bias
    logits = hippocampal.action_object_decoder(o)        # decoder
    loss   = CE(logits, c) + code_l2 * mean(||o||^2)

Gradient reaches BOTH maps, so the encoder is shaped into a class-separable image and
the decoder into its inverse on that image. The class labels are ENUMERATED by the
member, not read from the agent's executed actions: the agent's own replay can lack
whole classes (ADDENDUM 2; e.g. 1,427 of 1,492 steps in one class), and a decoder never
shown a class cannot emit it -- the round trip is a property of the codec, not of the
policy. Only the agent's own sensed ``z_world`` is recorded; nothing privileged. The
code-norm penalty bounds the image scale: a pure CE objective can always lower its loss
by inflating ``||o||`` (larger logits), which would move the encoder image away from
whatever scale part (3) and any O reader was calibrated on.

The cue bias (SD-016 ``action_bias``) is additive in O after the encoder and is not in
this objective; with the bias ON the rollout's codes are the image shifted by the bias.
The encoder does NOT feed E2's world transition (``rollout_with_world`` steps
``world_forward(z_world, action)`` on the raw action), so training it changes no world
prediction; it changes O and the E3 cue-bias term that reads O.

Contract: ``tests/contracts/test_w1_codec.py``. Evidence domain of the member gate: D1
(the codec round-trips and is well-conditioned in the CEM loop); it is not evidence that
proposals become choice-relevant (gate (e), W3/W4/W5).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ree_core.utils.waking_trainer import WakingTrainerMember


class CodecMember(WakingTrainerMember):
    """Joint encoder/decoder round-trip member (W1 codec part (1))."""

    name = "codec"

    def __init__(self, agent: Any, lr: float, batch_size: int, buffer_max: int,
                 code_l2: float = 1e-3) -> None:
        self._encoder = agent.e2.action_object_head
        self._action_object = agent.e2.action_object
        self._decoder = agent.hippocampal.action_object_decoder
        self.action_dim = int(agent.e2.config.action_dim)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.code_l2 = float(code_l2)
        self._buf: Deque[torch.Tensor] = deque(maxlen=int(buffer_max))
        ids = {id(p) for m in (self._encoder, self._decoder)
               for p in m.parameters() if p.requires_grad}
        self._named = [(n, p) for n, p in agent.named_parameters() if id(p) in ids]

    def named_parameters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        return list(self._named)

    def observe(self, agent: Any, harm_signal: float) -> None:
        lat = agent._current_latent
        z = getattr(lat, "z_world", None) if lat is not None else None
        if z is None:
            return
        self._buf.append(z.detach().reshape(1, -1).clone())

    def add_states(self, z_worlds: Any) -> None:
        """Append recorded z_world states directly (drivers / contracts). Detached."""
        for z in z_worlds:
            self._buf.append(z.detach().reshape(1, -1).clone())

    def ready(self) -> bool:
        return len(self._buf) >= self.batch_size

    def batch_loss(self, zw: torch.Tensor) -> torch.Tensor:
        """The member objective on an explicit ``[B, world_dim]`` batch."""
        A = self.action_dim
        B = zw.shape[0]
        z_rep = zw.repeat_interleave(A, dim=0)                       # [B*A, D]
        eye = torch.eye(A, dtype=zw.dtype, device=zw.device)
        acts = eye.repeat(B, 1)                                      # [B*A, A]
        labels = torch.arange(A, device=zw.device).repeat(B)         # [B*A]
        codes = self._action_object(z_rep, acts)
        logits = self._decoder(codes)
        loss = F.cross_entropy(logits, labels)
        if self.code_l2 > 0.0:
            loss = loss + self.code_l2 * codes.pow(2).sum(dim=-1).mean()
        return loss

    def loss(self, agent: Any) -> Optional[torch.Tensor]:
        if not self.ready():
            return None
        idx = torch.randperm(len(self._buf))[: self.batch_size].tolist()
        zw = torch.cat([self._buf[i] for i in idx], dim=0)
        return self.batch_loss(zw)
