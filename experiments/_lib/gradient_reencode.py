"""Re-encode buffered observations WITH gradient, at training time.

THE DEFECT THIS ANSWERS (failure_autopsy_dv-headroom-diagnostics-cluster_2026-09-07,
REE_assembly cb4a71fbd9, V3-EXQ-972a target; user-ratified 2026-09-08 by governance
cycle governance-20260908-0703). The ContextMemory write-content portfolio drivers
(v3_exq_970/_971/_972/_970a) train `write_addr_tagger` on a rolling buffer of PAST
states so its contrastive loss can compare classes across many samples -- but that
buffer stores each state `.detach().clone()`'d (970a:876), so the loss it trains can
never move `agent.latent_stack` (the encoder). V3-EXQ-972a measured this directly:
0 of 49 `latent_stack` parameters received gradient across the whole run, and the
resulting latents were bit-identical, 8/8 seeds, to an UNTRAINED_ENCODER control.
The optimizer's `standard_params` (970a:984-986) *includes* the encoder, so the run
LOOKED like it trained one; the graph was severed upstream of the loss, not absent
from the optimizer.

THIS IS NOT A BUG IN THE DETACH ITSELF. Every experience buffer `ree_core/agent.py`
keeps (`_self_experience_buffer`, `_world_experience_buffer`, `_e2_transition_buffer`)
stores its latents `.detach().clone()`'d BY DESIGN (agent.py:5835-5836) -- do not
"fix" that call site. It is the same single-step-truncated-BPTT convention used
EVERYWHERE in this codebase, including the live, un-buffered path: `agent.sense()`
itself only carries gradient through one step of recurrence, because the
`prev_state` it is handed (`agent._current_latent`) was itself produced by a PRIOR
`_e1_tick()` call that already detached-and-cloned it. `compute_event_contrastive_
loss`'s docstring states the corollary explicitly: to reach the encoder at all, a
loss must be computed on the LatentState returned directly by `sense()`, not on
anything read back out of `agent._current_latent` or a buffer.

For a loss that needs a BATCH of PAST states -- write_addr_tagger's contrastive loss
over a rolling safe/dangerous buffer being the motivating case -- "compute it on the
fresh LatentState at capture time" is not available: training happens on a delayed
cadence, well after the original per-step computational graphs were freed by an
earlier `.backward()` call. The only way to give such a loss a genuine gradient path
to the encoder is to RE-RUN the encoder's forward path on the RAW OBSERVATION at
training time, using the SAME (already-detached) recurrent input the original step
used -- reproducing exactly the single-step-truncated gradient the live pass would
have given, just delayed. That is what this module does.

USAGE (matches the 970/971/972/970a driver family's `state_buf` pattern; wiring the
drivers to it is a separate `/queue-experiment`-mediated follow-up -- CLAUDE.md
"Experiment Scripts": experiment-script LOGIC changes must go through that skill,
not a housekeeping-bundle edit):

    from experiments._lib.gradient_reencode import ObservationCapture, reencode_batch_z

    capture_buf: List[ObservationCapture] = []
    ...
    # BEFORE agent.sense() advances the recurrent state for this step:
    capture_buf.append(ObservationCapture.capture(agent, obs_body, obs_world, obs_harm))
    latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
    ...
    # later, at the training cadence, in place of reading detached buffered latents:
    batch = reencode_batch_z(agent, [capture_buf[i] for i in idx])   # requires_grad,
                                                                       # connected to
                                                                       # agent.latent_stack
    h1_loss = h1_contrastive_loss_k(tagger, [batch_safe, batch_dang])

Nothing in `ree_core` or any existing driver imports this module, so it changes no
default codepath and no existing run's behaviour.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch


@dataclass
class ObservationCapture:
    """One buffered (raw observation, recurrent-input) pair, detached at capture
    time. Cheap: no encoder forward pass has happened yet for `obs_body`/
    `obs_world`/`obs_harm`, so detaching them severs nothing; `prev_state` is a
    detach of an ALREADY-LIVE latent (`agent._current_latent`), exactly like every
    other experience buffer in this codebase already does.
    """

    obs_body: torch.Tensor
    obs_world: torch.Tensor
    obs_harm: Optional[torch.Tensor]
    prev_state: Optional[object]  # Optional["ree_core.latent.stack.LatentState"]

    @staticmethod
    def capture(agent, obs_body: torch.Tensor, obs_world: torch.Tensor,
                obs_harm: Optional[torch.Tensor] = None) -> "ObservationCapture":
        """Snapshot the raw observation plus the recurrent input `agent.sense()`
        is about to consume. Call this BEFORE `agent.sense()` advances the
        recurrent state for the current step (so `agent._current_latent` still
        holds the PREVIOUS tick's state) -- exactly the ordering the drivers'
        existing `state_buf.append(...)` call already uses relative to
        `agent.sense()`.
        """
        prev = getattr(agent, "_current_latent", None)
        prev_snapshot = prev.detach() if prev is not None else None
        return ObservationCapture(
            obs_body=obs_body.detach().clone(),
            obs_world=obs_world.detach().clone(),
            obs_harm=(obs_harm.detach().clone() if obs_harm is not None else None),
            prev_state=prev_snapshot,
        )


def reencode_z_with_grad(agent, capture: ObservationCapture) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-run ONLY the encoder forward path (`agent.body_obs_encoder`,
    `agent.world_obs_encoder`, `agent.latent_stack.encode`) on one captured
    observation. Returns `(z_self, z_world)`, both connected to
    `agent.latent_stack`'s (and the two per-modality encoders') parameters --
    `.backward()` on a loss built from these reaches the encoder.

    Deliberately narrower than calling `agent.sense()` again: it does not touch
    `agent._current_latent`, `agent._last_action`, `agent._per_axis_drive`, the
    LPB router, or any other live-rollout bookkeeping `sense()` performs as a
    side effect. Re-encoding a BUFFERED observation for a delayed training step
    must not perturb the agent's actual online trajectory.
    """
    obs_body = capture.obs_body
    obs_world = capture.obs_world
    if obs_body.dim() == 1:
        obs_body = obs_body.unsqueeze(0)
        obs_world = obs_world.unsqueeze(0)
    obs_body = obs_body.to(agent.device).float()
    obs_world = obs_world.to(agent.device).float()

    enc_body = agent.body_obs_encoder(obs_body)
    enc_world = agent.world_obs_encoder(obs_world)
    enc_combined = torch.cat([enc_body, enc_world], dim=-1)

    harm_obs = capture.obs_harm
    if harm_obs is not None:
        harm_obs = harm_obs.to(agent.device).float()

    new_latent = agent.latent_stack.encode(
        enc_combined,
        capture.prev_state,
        prev_action=None,
        harm_obs=harm_obs,
        harm_obs_a=None,
        harm_history=None,
        volatility_signal=None,
        self_e1_anchor=None,
    )
    return new_latent.z_self, new_latent.z_world


def reencode_batch_z(agent, captures: Sequence[ObservationCapture]) -> torch.Tensor:
    """Re-encode a list of captured observations and return a stacked
    `[N, self_dim + world_dim]` batch, gradient-connected to the encoder -- the
    drop-in replacement for reading `torch.cat([z_self, z_world])` straight out
    of a detached `state_buf` entry (970a:876's pattern)."""
    rows: List[torch.Tensor] = []
    for cap in captures:
        z_self, z_world = reencode_z_with_grad(agent, cap)
        rows.append(torch.cat([z_self, z_world], dim=-1))
    return torch.cat(rows, dim=0)


def latent_stack_param_snapshot(agent) -> List[torch.Tensor]:
    """Detached CPU clones of every `agent.latent_stack` parameter -- the
    identity-control snapshot a caller takes BEFORE a training step, to compare
    against `latent_stack_moved(agent, snapshot)` AFTER it. Mirrors V3-EXQ-972a's
    own hash-verified identity control (LINEAGE-vs-UNTRAINED_ENCODER), generalised
    into a reusable pre/post floor check instead of a one-off script computation.
    """
    return [p.detach().clone().cpu() for p in agent.latent_stack.parameters()]


def latent_stack_moved(agent, before: List[torch.Tensor], floor: float = 1e-8) -> bool:
    """True if at least one `agent.latent_stack` parameter's max-abs delta since
    `before` (a `latent_stack_param_snapshot` taken earlier) exceeds `floor`."""
    after = agent.latent_stack.parameters()
    for p_before, p_after in zip(before, after):
        delta = (p_after.detach().cpu() - p_before).abs().max().item()
        if delta > floor:
            return True
    return False
