"""SD-ZSELF-P0 (sd_zself_training_path): the P0 objective that actually reaches z_self.

WHY THIS EXISTS
---------------
DR-13 (SELF-1) shipped a dedicated gated self-recurrence for z_self
(`ree_core/latent/self_recurrence.py`) on the premise -- stated in its own design doc --
that the cell "trains via the EXISTING E1/E2 z_self prediction losses; v1 adds no new
loss". THAT PREMISE IS FALSE, and it was measured false twice:

  V3-EXQ-1078 (3/3 seeds, 20 x 100-step P0 warmup with
  `compute_prediction_loss() + compute_e2_loss()` over `agent.parameters()`, Adam 1e-3):
      gru_param_max_delta        = 0.0
      latent_stack_tensors_changed = 0 / 53
  `REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-1078_2026-09-24.md`

  The causal-reach trace (probe 1, 40 grad-enabled ticks;
  `REE_assembly/evidence/planning/zself_causal_reach_trace_20260924.md`) re-derived WHY:
  every z_self the E1/E2 losses see is a DETACHED copy. The three loci, at ree-v3
  `00210b5`, are `agent.py:5855` (`_current_latent = new_latent.detach()`),
  `agent.py:6294` (`_self_experience_buffer.append(z_self.detach().clone())`) and
  `agent.py:11397` (`record_transition`, all three arguments detached). The synthetic
  upper-bound row `sum(live z_self)` reaches 11 tensors including the GRU, so the path
  EXISTS -- it is the stored copies that are severed. Nothing in production trains it
  either: the all-ON recipe's optimizer groups (e2, lpfc bias, ofc deval) and SD-070's
  `world_path_parameters()` contain ZERO z_self-path members.

The same root was found for ContextMemory by V3-EXQ-972a; SD-CM-LIVETAP (ree-v3
`30f40ab`, default OFF) restored a route there for a SINGLE-TICK write-addressing
objective. That fix does not generalise to this one: the z_self buffer is consumed
later and in bulk, so backpropagating through accumulated live-tick tensors would
re-arm exactly the "modified by an inplace operation" hazard the `agent.py:5855`
comment records. This module therefore takes the SD-070 route instead -- a PHASED
trainer that runs its OWN forward passes over RECORDED observations, with a fresh
detached init state per chunk, so no graph survives an optimizer step and the
retained-graph hazard is avoided BY CONSTRUCTION rather than by care.

THE OBJECTIVE, AND WHY THIS ONE
-------------------------------
A BODY FORWARD MODEL: `head([z_self_t, a_t]) -> body_obs_{t+1}`, MSE.

Four candidate forms were prototyped end to end and measured on held-out native
experience (trace Section 5, config A, self_dim = world_dim = 32, seeds 42 and 43,
300 updates; ridge probe, 5-fold grouped by episode):

    candidate                       R2 next body   R2 action(t-2)   eff. rank   E1 swap
    base (E1/E2 P0 only)            0.456 / 0.233  0.100 / 0.051    1.21 / 1.13  0.7%
    (i)  fwd  body forward model    0.660 / 0.638  0.170 / 0.244    2.61 / 1.91  6.6%
    (ii) contr temporal InfoNCE     0.668 / 0.553  0.177 / 0.091    1.47 / 1.79  113%
    (iii) anchor -> E1-predicted    0.067 / 0.033  0.003 / 0.001    1.02 / 1.03  0.2%
    (iv) livetap E2-self live ends  0.336 / 0.208  0.012 / 0.013    1.73 / 1.33  0.1%

Reading that table:

  * (iii) and (iv) COLLAPSE the self-state on 2/2 seeds -- effective rank to ~1.0, or
    the z norm shrunk 5x with the loss driven to 3e-7 (the trivial solution). Both are
    the same shape: a learned predictor whose target IS the trainable latent, with no
    stop-gradient / EMA target. **Letting the existing E1/E2 losses train through a
    live tap is the collapse route, not the fix.** A collapsed z_self makes INV-069's
    coherence trivially maximal and MECH-113's D_eff trivially minimal -- both would
    read as SUCCESS and be vacuous.
  * (ii) is competitive on information but its objective IS successive-z_self
    similarity, which is what INV-069's V_s consecutive-difference coherence proxy
    reads: any later coherence PASS on it is circular. It also swamps E1 (the prior
    becomes z_self-dominated, 103-151% change under a null intervention).
  * (i) has the best information with the least distortion, the highest effective rank,
    genuine RECURRENT content -- action(t-2) R2 0.17-0.24 where predicting from the raw
    instantaneous body observation gives -0.03, so that content is not in the current
    frame -- and a moderate, interpretable ~10x rise in E1 coupling. It optimises no
    INV-069 DV directly.

Decision of record: the Orchestrator `orchestrate-20260924-breakthrough` chose (i)
under the standing decide-the-obvious delegation (`rec-20260924-fb429c72`), trace
Section 8 option O3.

WHAT THIS BUILD CANNOT REACH, STATED RATHER THAN OMITTED
-------------------------------------------------------
BEHAVIOURAL CONSEQUENCE (D3) IS NOT REACHABLE BY THIS BUILD, and no acceptance
criterion here claims it. Measured in the same trace: swap / zero / matched-norm noise
/ permute on z_self changed the committed action at **0 of 68 E3 ticks**, and a
persistent closed-loop intervention changed **0 actions in 12/12 whole episodes** --
while the z_world canary in the SAME harness moved 4-30% of E3-tick actions and
diverged by tick 0-6. The harness is sound; the negative is real. The cause is
structural and downstream of this module: E3 scores `world_states` only
(`e3_selector.py:1305-1321`), the per-candidate E2 self-rollout is computed and
DISCARDED (`e2_fast.py:838`), E1's z_self response reaches selection only through
`hippocampal.terrain_prior` which no loss in `ree_core` trains, and DR-10 is
default-off with no z_self-derived producer.

So an INV-069 / MECH-113 retest run on top of this build measures SELF-STATE QUALITY
and E1 USE ONLY -- not behavioural self-reach. The missing edge (a z_self-reading
per-candidate self-viability valuation consumer) is routed separately to /governance
as its own substrate_queue row; GFLAG-0481. It is deliberately NOT built here.

ACCEPTANCE, AND WHY THE OBVIOUS TEST IS NOT ENOUGH
--------------------------------------------------
"Nonzero GRU delta AND a changed self_encoder param" is NECESSARY BUT NOT SUFFICIENT:
every one of the four candidates above passes it, INCLUDING the two that collapse. So
`train()` returns a `holdout_before` / `holdout_after` pair computed on HELD-OUT
EPISODES THE OBJECTIVE NEVER SAW, and the contract (`tests/contracts/
test_zself_p0_training_path.py`) gates on all four of the trace's Section 9 criteria:

  1. nonzero attributable change in the GRU AND `self_encoder` (flag ON), bit-identity OFF
  2. NON-COLLAPSE: held-out effective rank and z norm not below the untrained baseline
  3. held-out INFORMATION GAIN on episodes not trained on: next-body AND a history
     target (action(t-2)), the history target above the raw-instantaneous-observation
     ceiling
  4. at least one native consumer (today: only E1) responds to a z_self intervention
     above the untrained baseline

MECH-094: not applicable. Trains on recorded waking observations; writes nothing to
memory in any non-waking state.

See `REE_assembly/docs/architecture/dr13_self_recurrence_temporal_depth.md`
("Phased training" -- corrected), `REE_assembly/evidence/planning/
zself_causal_reach_trace_20260924.md`, and `ree_core/latent/zworld_p0.py` (SD-070, the
sibling pattern this mirrors).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ZSelfP0Config",
    "ZSelfP0Trainer",
    "effective_rank",
    "ridge_cv_r2",
    "self_path_parameter_names",
]

# The z_self recognition path, as MODULE-PATH SEGMENTS. `self_predictor` is deliberately
# ABSENT: it is E1/E2's head on z_self, not part of the encoder path this objective
# trains, and including it would let the readout report movement that the existing
# (detached) losses caused.
#
# SEGMENTS, NOT SUBSTRINGS, AND THAT IS LOAD-BEARING. The obvious spelling -- a
# ".self_recurrence." substring test -- silently MISSES the DR-13 GRU whenever the names
# are read straight off the LatentStack (`self_recurrence.cell.weight_ih`, no leading
# dot) rather than off the agent (`latent_stack.self_recurrence.cell.weight_ih`). That
# drops the GRU out of the optimizer while every other module still trains, which is
# this module's OWN defect reproduced one level down -- caught here only because
# `param_delta` reports the GRU separately and read 0.0. Matching whole segments is
# insensitive to which root the names were enumerated from.
_SELF_PATH_SEGMENTS: Tuple[str, ...] = (
    "self_encoder",
    "self_recurrence",
    "self_topdown",
)
_SELF_PATH_LEAVES: Tuple[str, ...] = ("self_precision_logit",)

# Parameter-delta report groups. Keys are the names the acceptance criteria use.
_DELTA_GROUPS: Tuple[str, ...] = (
    "self_encoder",
    "self_recurrence",
    "self_topdown",
    "self_precision_logit",
    "body_obs_encoder",
)


def _name_has_segment(name: str, segment: str) -> bool:
    """True when `segment` is a whole dot-separated component of a parameter name."""
    return segment in name.split(".")


def self_path_parameter_names(latent_stack: Any) -> List[str]:
    """Names of the z_self-path parameters inside `latent_stack`.

    Exported because the acceptance criteria, the DR-13 doc and any later readiness
    check must all agree on WHICH tensors count as 'the z_self path'. A second,
    hand-written copy of this predicate is how a readout starts reporting a different
    parameter set from the one the optimizer holds.
    """
    out = []
    for n, _ in latent_stack.named_parameters():
        parts = n.split(".")
        if any(seg in parts for seg in _SELF_PATH_SEGMENTS) or parts[-1] in _SELF_PATH_LEAVES:
            out.append(n)
    return out


def effective_rank(z: torch.Tensor) -> float:
    """Participation-ratio effective rank of a [n, d] batch of latents.

    (sum s_i^2)^2 / sum s_i^4 over the singular values of the CENTRED matrix -- the
    same statistic the causal-reach trace reports as `z_eff_rank`, so a number computed
    here is directly comparable with the table in this module's docstring. ~1.0 means
    collapse onto a single effective direction.
    """
    if z.dim() != 2 or z.shape[0] < 2:
        return 0.0
    zc = (z - z.mean(dim=0, keepdim=True)).double()
    s = torch.linalg.svdvals(zc)
    s2 = (s ** 2).sum()
    s4 = (s ** 4).sum()
    if float(s4) <= 1e-24:
        return 0.0
    return float(s2 * s2 / s4)


def ridge_cv_r2(
    x: torch.Tensor,
    y: torch.Tensor,
    groups: Sequence[int],
    n_folds: int = 5,
    lam: float = 1e-2,
) -> Optional[float]:
    """Grouped k-fold ridge R^2, averaged over output dimensions.

    Folds are formed over the DISTINCT group ids (here: episode indices), never over
    rows, so consecutive ticks of one episode can never straddle the train/test split.
    Returns None -- NOT 0.0 -- when the readout could not be computed at all (too few
    groups, or every target dimension constant): a probe that could not run must not be
    indistinguishable from a probe that ran and found nothing. Callers that gate on an
    R^2 must treat None as 'cannot determine' rather than as a failing score.
    """
    if x.dim() != 2 or y.dim() != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 4:
        return None
    ug = sorted(set(int(g) for g in groups))
    if len(ug) < 2:
        return None
    n_folds = max(2, min(int(n_folds), len(ug)))
    folds = [ug[i::n_folds] for i in range(n_folds)]
    g = torch.tensor([int(v) for v in groups])
    xb = torch.cat([x.double(), torch.ones(x.shape[0], 1, dtype=torch.float64)], dim=1)
    yd = y.double()
    ss_res = torch.zeros(yd.shape[1], dtype=torch.float64)
    ss_tot = torch.zeros(yd.shape[1], dtype=torch.float64)
    eye = torch.eye(xb.shape[1], dtype=torch.float64)
    used = 0
    for f in folds:
        if not f:
            continue
        te = torch.isin(g, torch.tensor(f))
        tr = ~te
        if int(te.sum()) == 0 or int(tr.sum()) < max(5, xb.shape[1] // 4):
            continue
        xt, yt = xb[tr], yd[tr]
        try:
            w = torch.linalg.solve(xt.t() @ xt + lam * eye, xt.t() @ yt)
        except RuntimeError:
            continue
        pred = xb[te] @ w
        ss_res += ((yd[te] - pred) ** 2).sum(0)
        ss_tot += ((yd[te] - yt.mean(0)) ** 2).sum(0)
        used += 1
    if used == 0:
        return None
    keep = ss_tot > 1e-8
    if not bool(keep.any()):
        return None
    return float((1.0 - ss_res[keep] / ss_tot[keep]).mean())


@dataclass
class ZSelfP0Config:
    """Hyperparameters for the z_self P0 body-forward-model recipe.

    Defaults are the operating point the causal-reach trace measured candidate (i) at
    (300 updates, Adam 1e-3, B=16 chunks of L=16). Like `ZWorldP0Config` these are NOT
    no-op defaults: the no-op guarantee is STRUCTURAL (nothing constructs a trainer
    unless an experiment asks, and the `run_zself_p0` wrapper returns without touching
    a parameter at `episodes <= 0`), not a flag that can be left in the wrong state.
    """

    # Objective.
    head_hidden: int = 64
    # Optimisation.
    updates: int = 300
    batch_size: int = 16          # chunks per update
    chunk_length: int = 16        # L; each chunk uses L+1 observations and L actions
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    seed: int = 0
    # Readout. Episodes are held out WHOLE -- the information-gain criterion is only
    # meaningful on experience the objective never saw.
    holdout_fraction: float = 0.2
    ridge_lambda: float = 1e-2
    ridge_folds: int = 5
    # Which body_obs dimensions the next/current-body R^2 is reported over. None =
    # the leading min(5, body_obs_dim) dims, which on CausalGridWorldV2 are the body
    # core (x, y, health, energy, footprint) the trace reports. An env with a different
    # body layout should pass its own indices rather than silently probing the wrong
    # columns.
    body_core_indices: Optional[Sequence[int]] = None
    # Lag for the history probe. The point of this target is that it is NOT in the
    # instantaneous observation, so it can only be carried by the recurrence.
    history_lag: int = 2


class ZSelfP0Trainer:
    """Runs the z_self P0 body-forward-model recipe against a live LatentStack.

    Trains exactly the z_self recognition path -- `body_obs_encoder`,
    `split_encoder.self_encoder`, `split_encoder.self_topdown`,
    `split_encoder.self_precision_logit` and, when DR-13 is on,
    `self_recurrence` (the GRUCell) -- plus this objective's own prediction head,
    which is discarded afterwards.

    The forward pass is the NATIVE encode path, not a shortcut through
    `self_encoder` alone: `body_obs_encoder`/`world_obs_encoder` ->
    `LatentStack.encode` including the two-pass top-down conditioning, the precision
    gate and the DR-13 recurrence. Anything less would train a different function from
    the one `agent.sense()` runs at act time, which is the failure this whole module
    exists to repair.

    The E1 anchor is deliberately NOT supplied during training (`self_e1_anchor=None`
    -> pure recurrence). Supplying it would make the objective's target depend on a
    predictor that is itself being fitted to z_self -- candidate (iii) in the module
    docstring, measured to collapse the self-state on 2/2 seeds.

    Usage:
        trainer = ZSelfP0Trainer(agent, ZSelfP0Config(seed=seed))
        for each rollout step:
            trainer.observe(body_obs, world_obs, action)   # action taken FROM this obs
        trainer.end_episode()                              # after each episode
        stats = trainer.train()
    """

    def __init__(self, agent: Any, config: Optional[ZSelfP0Config] = None) -> None:
        self.config = config or ZSelfP0Config()
        self.agent = agent
        ls = getattr(agent, "latent_stack", None)
        be = getattr(agent, "body_obs_encoder", None)
        we = getattr(agent, "world_obs_encoder", None)
        if ls is None or be is None or we is None:
            raise ValueError(
                "ZSelfP0Trainer requires an agent exposing latent_stack, "
                "body_obs_encoder and world_obs_encoder. Got %r."
                % (type(agent).__name__,)
            )
        se = getattr(ls, "split_encoder", None)
        if se is None or not hasattr(se, "self_encoder"):
            raise ValueError(
                "ZSelfP0Trainer requires a LatentStack with a SplitEncoder exposing "
                "self_encoder (SD-005). Got %r." % (type(ls).__name__,)
            )
        self.latent_stack = ls
        self.split_encoder = se
        self.body_obs_encoder = be
        self.world_obs_encoder = we
        self.self_dim = int(se.self_dim)
        self.action_dim = self._resolve_action_dim(agent.config)
        # Closed episodes: (body [T+1, bd], world [T+1, wd], action [T, ad]).
        self._episodes: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._cur_body: List[torch.Tensor] = []
        self._cur_world: List[torch.Tensor] = []
        self._cur_action: List[Optional[torch.Tensor]] = []

    @staticmethod
    def _resolve_action_dim(config: Any) -> int:
        """The one-hot action width, read from the config rather than assumed.

        `REEConfig` has no top-level `action_dim`: `from_dims` wires the SAME argument
        into `config.e2`, `config.e1` and `config.hippocampal`, so any of the three is
        authoritative and they cannot disagree. REFUSE rather than fall back to 0 -- a
        zero width silently disables the one-hot width check in `observe()`, and the
        objective would then be trained on a scalar action index whose magnitude is
        meaningless, which is a wrong answer dressed as a working one.
        """
        for sub in ("e2", "e1", "hippocampal"):
            node = getattr(config, sub, None)
            width = int(getattr(node, "action_dim", 0) or 0) if node is not None else 0
            if width > 0:
                return width
        raise ValueError(
            "ZSelfP0Trainer: could not resolve action_dim from the agent config "
            "(looked at config.e2/e1/hippocampal.action_dim). The objective conditions "
            "on a one-hot action and cannot proceed without its width."
        )

    # -- buffer ------------------------------------------------------------------------
    def observe(
        self,
        body_obs: torch.Tensor,
        world_obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> None:
        """Append one rollout step: the observation at t and the action taken FROM it.

        `action` may be None on the terminal frame of an episode (there is no action
        from it); that frame is still kept, as the final prediction TARGET. A scalar
        action index is accepted and one-hot encoded (the `_lib.capability_eval.Policy`
        surface returns an int), as is an already-one-hot vector. Everything is stored
        detached, cloned and flattened -- the buffer must not retain a graph across the
        rollout, and must not alias a tensor the environment reuses between steps.
        """
        self._cur_body.append(torch.as_tensor(body_obs).detach().float().reshape(-1).clone())
        self._cur_world.append(torch.as_tensor(world_obs).detach().float().reshape(-1).clone())
        if action is None:
            self._cur_action.append(None)
        else:
            self._cur_action.append(self._as_one_hot(action))

    def _as_one_hot(self, action: Any) -> torch.Tensor:
        """Action index or one-hot vector -> a one-hot float vector [action_dim]."""
        a = torch.as_tensor(action).detach()
        if a.numel() == 1 and self.action_dim > 1:
            idx = int(a.reshape(()).item())
            if not 0 <= idx < self.action_dim:
                raise ValueError(
                    "ZSelfP0Trainer.observe: action index %d is outside "
                    "[0, %d)." % (idx, self.action_dim)
                )
            oh = torch.zeros(self.action_dim, dtype=torch.float32)
            oh[idx] = 1.0
            return oh
        a = a.float().reshape(-1).clone()
        if self.action_dim and a.numel() != self.action_dim:
            raise ValueError(
                "ZSelfP0Trainer.observe: action has %d elements, expected %d "
                "(an index, or a one-hot over the action space)."
                % (a.numel(), self.action_dim)
            )
        return a

    def end_episode(self) -> None:
        """Close the current episode and move it into the buffer.

        An episode is truncated at its FIRST unrecorded action: a gap would make
        `prev_action` at that step a lie, and SD-007 reafference correction reads it.
        """
        n = len(self._cur_body)
        if n >= 2:
            k = 0
            while k < n - 1 and self._cur_action[k] is not None:
                k += 1
            # keep observations 0..k and actions 0..k-1
            if k >= 1:
                body = torch.stack(self._cur_body[: k + 1])
                world = torch.stack(self._cur_world[: k + 1])
                act = torch.stack([a for a in self._cur_action[:k]])  # type: ignore[misc]
                self._episodes.append((body, world, act))
        self._cur_body, self._cur_world, self._cur_action = [], [], []

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    @property
    def n_buffered(self) -> int:
        return int(sum(int(e[2].shape[0]) for e in self._episodes))

    # -- the trained path --------------------------------------------------------------
    def self_path_parameters(self) -> List[torch.Tensor]:
        """Exactly the z_self recognition path.

        `body_obs_encoder` is included because on this substrate it is z_self-exclusive:
        `SplitEncoder.forward` routes `body_obs` to `self_encoder` and NOTHING else
        (`stack.py`), so training it cannot leak into the world channel. It is also the
        first layer the objective's gradient meets, and leaving it frozen would make the
        recipe's own input encoding a fixed random projection -- the SD-070 lesson.
        """
        names = set(self_path_parameter_names(self.latent_stack))
        # A guard against this module's own most likely defect: DR-13 is ON, so the GRU
        # is in the forward path and MUST be in the optimizer -- if the name predicate
        # ever stops matching it, everything else still trains and the run looks healthy
        # while the one module the build exists for stays frozen. Refuse loudly instead.
        if self.uses_self_recurrence() and not any(
            _name_has_segment(n, "self_recurrence") for n in names
        ):
            raise ValueError(
                "ZSelfP0Trainer: use_self_recurrence is ON but no self_recurrence "
                "parameter matched the z_self-path predicate, so the DR-13 GRU would "
                "not be trained. LatentStack parameter names: %s"
                % (sorted(n for n, _ in self.latent_stack.named_parameters())[:12],)
            )
        params = [p for n, p in self.latent_stack.named_parameters() if n in names]
        params += list(self.body_obs_encoder.parameters())
        return params

    def uses_self_recurrence(self) -> bool:
        """True when the DR-13 GRU is actually in the forward path being trained."""
        return (
            getattr(self.latent_stack, "self_recurrence", None) is not None
            and bool(getattr(self.latent_stack.config, "use_self_recurrence", False))
        )

    def _native_chain(
        self,
        body: torch.Tensor,
        world: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Stateful z_self over a chunk, through the native encode path.

        body/world: [B, L+1, obs_dim]; action: [B, L, action_dim].
        Returns z_self [B, L+1, self_dim] with a chunk-local graph -- the init state is
        fresh and detached, so no graph survives the optimizer step.
        """
        ls = self.latent_stack
        prev = ls.init_state(body.shape[0], body.device)
        out = []
        for t in range(body.shape[1]):
            enc = torch.cat(
                [self.body_obs_encoder(body[:, t]), self.world_obs_encoder(world[:, t])],
                dim=-1,
            )
            prev_action = action[:, t - 1] if t > 0 else None
            st = ls.encode(enc, prev, prev_action=prev_action)
            out.append(st.z_self)
            prev = st
        return torch.stack(out, dim=1)

    # -- readouts ----------------------------------------------------------------------
    def _core_indices(self, body_dim: int) -> List[int]:
        idx = self.config.body_core_indices
        if idx is None:
            return list(range(min(5, body_dim)))
        return [int(i) for i in idx if 0 <= int(i) < body_dim]

    def _holdout_report(
        self,
        episodes: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Dict[str, Any]:
        """The acceptance readout, on episodes the objective never trained on.

        Every field is either a float or None. None means CANNOT DETERMINE -- too few
        episodes, or a degenerate target -- and is deliberately distinct from a low
        score: a caller gating on `R2_next_body` must not read a broken probe as a
        failed one.
        """
        lag = max(1, int(self.config.history_lag))
        zs: List[torch.Tensor] = []
        b_now: List[torch.Tensor] = []
        b_next: List[torch.Tensor] = []
        groups: List[int] = []
        h_z: List[torch.Tensor] = []
        h_b: List[torch.Tensor] = []
        h_a: List[torch.Tensor] = []
        h_groups: List[int] = []
        was_training = self.latent_stack.training
        self.latent_stack.eval()
        self.body_obs_encoder.eval()
        self.world_obs_encoder.eval()
        try:
            with torch.no_grad():
                for ei, (body, world, act) in enumerate(episodes):
                    z = self._native_chain(
                        body.unsqueeze(0), world.unsqueeze(0), act.unsqueeze(0)
                    )[0]                                   # [T+1, self_dim]
                    t_max = int(act.shape[0])              # transitions 0..t_max-1
                    for t in range(t_max):
                        zs.append(z[t])
                        b_now.append(body[t])
                        b_next.append(body[t + 1])
                        groups.append(ei)
                        if t >= lag:
                            h_z.append(z[t])
                            h_b.append(body[t])
                            h_a.append(act[t - lag])
                            h_groups.append(ei)
        finally:
            if was_training:
                self.latent_stack.train()
                self.body_obs_encoder.train()
                self.world_obs_encoder.train()

        out: Dict[str, Any] = {
            "n_rows": len(zs),
            "n_episodes": len(episodes),
            "n_history_rows": len(h_z),
            "history_lag": lag,
            "z_eff_rank": None,
            "z_norm_mean": None,
            "R2_next_body": None,
            "R2_cur_body": None,
            "R2_action_history": None,
            "ceiling_R2_next_body_from_raw_body": None,
            "ceiling_R2_action_history_from_raw_body": None,
        }
        if not zs:
            return out
        z_mat = torch.stack(zs)
        bn_mat = torch.stack(b_now)
        bx_mat = torch.stack(b_next)
        core = self._core_indices(bn_mat.shape[1])
        out["z_eff_rank"] = effective_rank(z_mat)
        out["z_norm_mean"] = float(z_mat.norm(dim=1).mean())
        if core:
            cf = self.config
            out["R2_next_body"] = ridge_cv_r2(
                z_mat, bx_mat[:, core], groups, cf.ridge_folds, cf.ridge_lambda)
            out["R2_cur_body"] = ridge_cv_r2(
                z_mat, bn_mat[:, core], groups, cf.ridge_folds, cf.ridge_lambda)
            out["ceiling_R2_next_body_from_raw_body"] = ridge_cv_r2(
                bn_mat, bx_mat[:, core], groups, cf.ridge_folds, cf.ridge_lambda)
        if h_z:
            cf = self.config
            hz = torch.stack(h_z)
            hb = torch.stack(h_b)
            ha = torch.stack(h_a)
            out["R2_action_history"] = ridge_cv_r2(
                hz, ha, h_groups, cf.ridge_folds, cf.ridge_lambda)
            # The ceiling that makes the history target MEAN something: how much of
            # action(t-lag) the raw instantaneous body observation already carries. The
            # trace measured this at -0.03, i.e. essentially nothing, so a positive
            # score through z_self is recurrent content and not a re-read of the frame.
            out["ceiling_R2_action_history_from_raw_body"] = ridge_cv_r2(
                hb, ha, h_groups, cf.ridge_folds, cf.ridge_lambda)
        return out

    # -- training ----------------------------------------------------------------------
    def _snapshot(self) -> Dict[str, torch.Tensor]:
        snap = {
            "latent_stack." + n: p.detach().clone()
            for n, p in self.latent_stack.named_parameters()
        }
        snap.update({
            "body_obs_encoder." + n: p.detach().clone()
            for n, p in self.body_obs_encoder.named_parameters()
        })
        return snap

    def _deltas(self, snap: Dict[str, torch.Tensor]) -> Dict[str, Optional[float]]:
        live: Dict[str, torch.Tensor] = {
            "latent_stack." + n: p for n, p in self.latent_stack.named_parameters()
        }
        live.update({
            "body_obs_encoder." + n: p
            for n, p in self.body_obs_encoder.named_parameters()
        })
        out: Dict[str, Optional[float]] = {}
        for group in _DELTA_GROUPS:
            vals = [
                float((p.detach() - snap[n]).abs().max())
                for n, p in live.items()
                if n in snap and (_name_has_segment(n, group) or n.split(".")[-1] == group)
            ]
            # None = the module is not present at all (e.g. the GRU with DR-13 off),
            # which must not read the same as 'present and did not move'.
            out[group] = max(vals) if vals else None
        return out

    def _split_episodes(
        self, rng: random.Random
    ) -> Tuple[List[int], List[int]]:
        n = len(self._episodes)
        idx = list(range(n))
        rng.shuffle(idx)
        n_hold = int(math.floor(n * float(self.config.holdout_fraction)))
        n_hold = max(0, min(n_hold, n - 1))
        # The readout needs >= 2 groups to form a single ridge fold at all.
        if n_hold == 1 and n >= 4:
            n_hold = 2
        hold = sorted(idx[:n_hold])
        train = sorted(idx[n_hold:])
        return train, hold

    def _sample_chunks(
        self, pool: Sequence[int], rng: random.Random
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        L = int(self.config.chunk_length)
        usable = [i for i in pool if int(self._episodes[i][2].shape[0]) >= L]
        if not usable:
            return None
        bs, ws, as_ = [], [], []
        for _ in range(int(self.config.batch_size)):
            body, world, act = self._episodes[usable[rng.randrange(len(usable))]]
            s = rng.randrange(0, int(act.shape[0]) - L + 1)
            bs.append(body[s: s + L + 1])
            ws.append(world[s: s + L + 1])
            as_.append(act[s: s + L])
        return torch.stack(bs), torch.stack(ws), torch.stack(as_)

    def train(self) -> Dict[str, Any]:
        """Run the recipe over the buffered episodes. Returns diagnostic statistics.

        `holdout_before` / `holdout_after` are the acceptance readout, computed on
        WHOLE EPISODES that were never sampled for training. `holdout_before` is the
        untrained baseline the trace's criteria 2 and 3 are stated against, so a caller
        never has to reconstruct one; it is measured on this same agent before the first
        optimizer step, which is a tighter control than a separately-seeded build.
        """
        cf = self.config
        L = int(cf.chunk_length)
        n_ep = len(self._episodes)
        if n_ep < 2:
            raise ValueError(
                "ZSelfP0Trainer.train() needs at least 2 buffered episodes, got %d. "
                "Roll out further (and call end_episode()) before training." % (n_ep,)
            )
        if not any(int(e[2].shape[0]) >= L for e in self._episodes):
            raise ValueError(
                "ZSelfP0Trainer.train() needs at least one episode with >= "
                "chunk_length=%d transitions; longest is %d."
                % (L, max(int(e[2].shape[0]) for e in self._episodes))
            )

        rng = random.Random(int(cf.seed))
        train_idx, hold_idx = self._split_episodes(rng)
        if not any(int(self._episodes[i][2].shape[0]) >= L for i in train_idx):
            raise ValueError(
                "ZSelfP0Trainer.train(): no TRAINING episode reaches chunk_length=%d "
                "after the holdout split (%d train / %d holdout episodes)."
                % (L, len(train_idx), len(hold_idx))
            )
        hold_eps = [self._episodes[i] for i in hold_idx]

        stats: Dict[str, Any] = {
            "recipe": "zself_p0_body_forward_model",
            "n_episodes": n_ep,
            "n_train_episodes": len(train_idx),
            "n_holdout_episodes": len(hold_eps),
            "n_buffered_transitions": self.n_buffered,
            "used_self_recurrence": self.uses_self_recurrence(),
        }
        stats["holdout_before"] = self._holdout_report(hold_eps)

        body_dim = int(self._episodes[0][0].shape[1])
        head = nn.Sequential(
            nn.Linear(self.self_dim + self.action_dim, int(cf.head_hidden)),
            nn.ReLU(),
            nn.Linear(int(cf.head_hidden), body_dim),
        )
        params = self.self_path_parameters() + list(head.parameters())
        opt = torch.optim.Adam(params, lr=float(cf.learning_rate))

        snap = self._snapshot()
        was_training = self.latent_stack.training
        self.latent_stack.train()
        self.body_obs_encoder.train()
        self.world_obs_encoder.train()
        losses: List[float] = []
        try:
            for _u in range(int(cf.updates)):
                chunk = self._sample_chunks(train_idx, rng)
                if chunk is None:
                    break
                body, world, act = chunk
                z = self._native_chain(body, world, act)          # [B, L+1, sd]
                pred = head(torch.cat([z[:, :-1], act], dim=-1))  # [B, L, body_dim]
                loss = F.mse_loss(pred, body[:, 1:])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cf.max_grad_norm and cf.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, float(cf.max_grad_norm))
                opt.step()
                losses.append(float(loss.detach()))
        finally:
            # Grad hygiene. The objective's backward pass necessarily touches SHARED
            # parameters (the top-down path runs through beta/theta/delta and the world
            # encoder), which this optimizer deliberately does not hold and never
            # steps -- but their .grad buffers are left populated. A later phase whose
            # first action is step() rather than zero_grad() would then apply this
            # objective's gradient to modules it was never meant to train. Clearing is
            # safe here because the recipe is a discrete phase with nothing in flight.
            for mod in (self.latent_stack, self.body_obs_encoder,
                        self.world_obs_encoder, head):
                for p in mod.parameters():
                    p.grad = None
            if not was_training:
                self.latent_stack.eval()
                self.body_obs_encoder.eval()
                self.world_obs_encoder.eval()

        stats["n_updates"] = len(losses)
        stats["mean_loss"] = float(sum(losses) / len(losses)) if losses else None
        stats["first10_loss"] = (
            float(sum(losses[:10]) / len(losses[:10])) if losses else None)
        stats["final_loss"] = (
            float(sum(losses[-10:]) / len(losses[-10:])) if losses else None)
        stats["param_delta"] = self._deltas(snap)
        stats["holdout_after"] = self._holdout_report(hold_eps)
        return stats
