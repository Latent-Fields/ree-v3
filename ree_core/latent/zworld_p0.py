"""SD-070: P0 z_world encoder-training recipe that differentiates rather than collapses.

WHY THIS EXISTS
---------------
The P0 the substrate previously prescribed -- SD-009 event-contrastive CE plus SD-018
resource-proximity MSE, applied online at batch=1 -- does not train a usable z_world. It
COLLAPSES it. Measured 2026-07-18 on this substrate (world_dim=128, seed 42, 40 eval
episodes, CausalGridWorldV2 at the x724 ENV_KWARGS rung):

    untrained                     participation_ratio = 9.21   contrast_ratio = 0.1222
    SD-009 + SD-018 (lr 1e-4)     participation_ratio = 1.06   contrast_ratio = 0.0726

A participation ratio of ~1 means z_world has collapsed onto a single effective dimension:
no discriminative geometry at all. Any downstream comparator built on it is vacuous, which
is the MECH-353 / V3-EXQ-642 lesson restated -- and it is why `substrate_queue.json:971`
(SD-031 / E2WorldForward) and `predictors/e2_world.py:42-54` both name a TRAINED encoder as
the precondition for the forward model.

THREE MEASURED FAULTS, AND WHAT EACH FORCES
-------------------------------------------
1. THE SD-009 TARGET IS UNLEARNABLE FROM THE CHANNEL IT IS WIRED TO.
   `agent.compute_event_contrastive_loss` classifies `transition_type` from z_world alone.
   But transition_type is a property of the TRANSITION (t-1 -> t), while z_world is a
   STATIC single-frame encoding. Probing the label with an identical MLP-128 from each
   channel (n=721, held-out macro-recall minus chance):

       world_obs  (what z_world sees)   -0.014  (3-class)   -0.060  (6-class)
       body_obs   (what z_self sees)    +0.121               +0.144
       world delta (w_t - w_prev)       +0.167               +0.329
       body delta  (b_t - b_prev)       +0.240               +0.427

   The world channel is AT OR BELOW CHANCE. The information lives in the deltas, and mostly
   in the BODY delta -- which SD-005's split encoder deliberately routes to z_self, so
   z_world structurally cannot see it. This is a wiring fault, not a labelling one: a
   repaired 6-class map that gives 'hazard_approach' and 'resource' their own classes
   (instead of folding both into class 0, as `agent._EVENT_LABEL_MAP` does) still sits at
   chance from world_obs. Class rebalancing cannot recover information that is not present.
   Recorded for governance in
   `REE_assembly/evidence/planning/sd009_event_contrastive_channel_mismatch_2026-07-18.md`;
   adjudication of SD-009's own status is deferred to a /governance cycle.

   FORCES: the P0 target set must consist of STATIC, single-frame properties of world_obs.

2. NOTHING PENALISES COLLAPSE. Predicting a near-constant class (the shipped map is ~95%
   class-0 saturated here) and regressing one scalar are both served perfectly by a 1-D
   representation. Variance in every other direction only adds noise to those two heads, so
   gradient descent removes it. Note the collapse is NOT the multiplicative precision gate:
   across the full P0 the `world_precision_logit` sigmoid moves only 0.4966-0.5074 and the
   final layer stays full-rank (singular-value PR ~55). The encoder's FUNCTION collapses
   while its weights look healthy.

   FORCES: an explicit anti-collapse term on the batch statistics of z_world.

3. THE LOOP IS ONLINE AT BATCH=1, which makes any variance or covariance statistic
   undefined -- there is no batch to compute it over.

   FORCES: mini-batching over a rollout buffer.

THE RECIPE
----------
    static scene-structure grounding targets   (fault 1)
  + class-balanced CE                          (residual imbalance in those targets)
  + VICReg variance/covariance penalty         (fault 2)
  + optional world_obs reconstruction          (fault 2, structural: one scalar can be
                                                served by 1-D, 275 outputs cannot)
  + mini-batched training over a buffer        (fault 3)

The grounding targets are derived from world_obs alone -- hazard-in-view, resource-in-view,
and bucketed Chebyshev distance to the nearest of each -- with no env introspection, so the
recipe has no dependency on environment internals and cannot leak privileged state. They
are decodable: probed from raw world_obs at balanced accuracy 0.961 / 0.965 / 0.943 / 0.948
against chance 0.5 / 0.5 / 0.333 / 0.333. They are also the distinctions SD-009 was reaching
for (harm-relevance in the world channel), expressed as targets the observation determines.

MEASURED RESULT (world_dim=128, this recipe at the config defaults, 3 seeds):

    seed   untrained PR -> trained PR     CR              hazard-presence (chance 0.50)
    42     9.21 -> 5.27                   0.122 -> 0.210  0.890
    43     6.63 -> 4.74                   0.134 -> 0.223  0.775
    44     8.56 -> 4.08                   0.156 -> 0.218  0.742

Against the V3-EXQ-783 anti-collapse gate (mean trained PR >= 0.5x mean dim-matched
untrained PR, AND >= 2.0 absolute): 4.70 / 8.13 = 0.578 and 4.70 absolute. Contrast ratio
rises on 3/3 seeds, clearing the untrained 0.13-0.15 band.

VICReg (Bardes, Ponce & LeCun 2022) supplies the variance-hinge + off-diagonal-covariance
form. It is borrowed as an ENGINEERING technique for a measured failure mode (representation
collapse under a low-information objective); it carries no architectural authority here. The
variance hinge alone does NOT raise the participation ratio -- perfectly correlated
dimensions at unit std still give PR=1 -- so the covariance term is the actual PR lever and
the two are not interchangeable. That is why the covariance weight defaults high (50) rather
than to VICReg's published ratio, which was measured here to be far too weak (w_cov=0.04
gave PR 1.80; w_cov=50 gave PR 4.02 in the same sweep).

BIT-IDENTICAL OFF BY CONSTRUCTION. This module adds no field to LatentStackConfig, no head
to SplitEncoder, and no method to REEAgent. It operates on an existing LatentStack from the
outside, and the auxiliary heads belong to the trainer rather than to the substrate. Nothing
runs unless an experiment explicitly constructs a ZWorldP0Trainer, so no existing experiment
can change behaviour. There is no flag to leave in the wrong state.

PHASED TRAINING. This is P0 only. P1 (E2WorldForward) must train on stop-gradient z_world
targets with the encoder optimiser NOT stepped; P2 is measurement. Joint training collapses
downstream heads (EXQ-166b/c/d).

MECH-094: not applicable -- this trains an encoder on live observations and writes nothing
to memory during any non-waking state, so no hypothesis_tag obligation arises.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LOCAL_VIEW_CELLS",
    "LOCAL_VIEW_ENTITY_STRIDE",
    "HAZARD_ENTITY_INDEX",
    "RESOURCE_ENTITY_INDEX",
    "ZWorldP0Config",
    "ZWorldP0Trainer",
    "balanced_class_weights",
    "chebyshev_offsets",
    "entity_presence_mask",
    "scene_structure_targets",
    "variance_covariance_penalty",
]

# --------------------------------------------------------------------------------------
# Local-view layout. Mirrors SplitEncoder.HAZARD_INDICES: world_obs[0:175] is a 5x5x7
# one-hot local view, cell-major, so entity e of cell c sits at index c*7 + e.
# CausalGridWorldV2.ENTITY_TYPES: empty 0, wall 1, resource 2, hazard 3.
# --------------------------------------------------------------------------------------
LOCAL_VIEW_GRID = 5
LOCAL_VIEW_CELLS = LOCAL_VIEW_GRID * LOCAL_VIEW_GRID
LOCAL_VIEW_ENTITY_STRIDE = 7
HAZARD_ENTITY_INDEX = 3
RESOURCE_ENTITY_INDEX = 2


def chebyshev_offsets() -> torch.Tensor:
    """Chebyshev (chessboard) distance of each local-view cell from the agent's centre cell.

    Returns [LOCAL_VIEW_CELLS] float tensor. The centre of a 5x5 view is cell (2, 2), so
    distances run 0 (the agent's own cell) through 2 (the view edge).
    """
    c = LOCAL_VIEW_GRID // 2
    return torch.tensor(
        [float(max(abs(i // LOCAL_VIEW_GRID - c), abs(i % LOCAL_VIEW_GRID - c)))
         for i in range(LOCAL_VIEW_CELLS)],
        dtype=torch.float32,
    )


def entity_presence_mask(world_obs: torch.Tensor, entity_index: int) -> torch.Tensor:
    """Per-cell occupancy mask for one entity type.

    Args:
        world_obs: [batch, world_obs_dim] -- the same tensor SplitEncoder consumes.
        entity_index: entity slot within the 7-wide one-hot (hazard 3, resource 2).

    Returns:
        [batch, LOCAL_VIEW_CELLS] float mask, 1.0 where that entity occupies the cell.
    """
    if world_obs.dim() == 1:
        world_obs = world_obs.unsqueeze(0)
    idx = list(range(entity_index,
                     LOCAL_VIEW_CELLS * LOCAL_VIEW_ENTITY_STRIDE,
                     LOCAL_VIEW_ENTITY_STRIDE))
    return world_obs[:, idx]


def scene_structure_targets(
    world_obs: torch.Tensor,
    n_distance_buckets: int = 4,
) -> Dict[str, torch.Tensor]:
    """Derive the SD-070 grounding targets from world_obs ALONE.

    No environment introspection: everything is read out of the observation the encoder
    itself receives, so the targets carry no privileged state the agent could not in
    principle compute. This is what makes them a legitimate self-supervised P0 signal
    rather than an oracle.

    Targets (all STATIC single-frame properties -- see fault 1 in the module docstring):
        hazard_present    [batch] long in {0,1}
        resource_present  [batch] long in {0,1}
        hazard_distance   [batch] long in [0, n_distance_buckets-1]
        resource_distance [batch] long in [0, n_distance_buckets-1]

    Distance is the Chebyshev distance to the NEAREST cell holding that entity, bucketed
    and clamped; when the entity is absent from the view the target saturates at the last
    bucket, which is the correct "nothing near" reading rather than a missing label.
    """
    if world_obs.dim() == 1:
        world_obs = world_obs.unsqueeze(0)
    offs = chebyshev_offsets().to(world_obs.device)
    absent = float(n_distance_buckets)  # saturates to the last bucket after clamping
    out: Dict[str, torch.Tensor] = {}

    for name, entity in (("hazard", HAZARD_ENTITY_INDEX),
                         ("resource", RESOURCE_ENTITY_INDEX)):
        mask = entity_presence_mask(world_obs, entity) > 0        # [batch, cells] bool
        present = mask.any(dim=1)                                  # [batch] bool
        dist = torch.where(mask, offs.unsqueeze(0).expand_as(mask),
                           torch.full_like(mask, absent, dtype=torch.float32))
        nearest = dist.min(dim=1).values
        nearest = torch.where(present, nearest,
                              torch.full_like(nearest, absent))
        out["%s_present" % name] = present.long()
        out["%s_distance" % name] = nearest.clamp(0, n_distance_buckets - 1).long()
    return out


def balanced_class_weights(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Inverse-frequency CE weights over the classes actually PRESENT in `labels`.

    Absent classes get weight 0 rather than an infinite weight, so a class the rollout
    never visited cannot dominate (or NaN) the loss.

    This is the standard inverse-frequency ("balanced") weighting w_i = N / (k * c_i) over
    the k present classes. Its FREQUENCY-WEIGHTED mean is exactly 1.0 -- i.e. the expected
    weight of a randomly drawn sample is 1 -- which is the property that keeps the weighted
    CE on the same scale as an unweighted one, so the other loss weights do not have to be
    retuned when the label balance shifts. Note the PLAIN (unweighted) mean over present
    classes is NOT 1 and grows as the balance skews; that is expected, not a defect.
    """
    counts = torch.bincount(labels, minlength=n_classes).float()
    present = counts > 0
    w = torch.zeros(n_classes, dtype=torch.float32, device=labels.device)
    if not bool(present.any()):
        return w
    w[present] = counts[present].sum() / (counts[present] * float(present.sum()))
    return w


def variance_covariance_penalty(
    z: torch.Tensor,
    variance_gamma: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """VICReg-style anti-collapse penalty on a batch of latents.

    Args:
        z: [batch, dim]. Needs batch >= 2 for the statistics to be defined.
        variance_gamma: target floor on each dimension's standard deviation.

    Returns:
        (variance_term, covariance_term)

        variance_term    mean over dims of relu(gamma - std_j). Puts a FLOOR under every
                         dimension's spread, so squashing an axis costs something.
        covariance_term  sum of squared off-diagonal covariances, normalised by dim.
                         This is the participation-ratio lever: the variance term alone
                         cannot raise PR, because perfectly correlated dimensions at unit
                         std still occupy one effective dimension.

    Both terms are returned separately (rather than pre-summed) so callers can weight and
    report them independently -- they do different jobs and their useful scales differ by
    more than an order of magnitude.
    """
    if z.dim() != 2:
        raise ValueError("variance_covariance_penalty expects [batch, dim], got %r"
                         % (tuple(z.shape),))
    n, d = int(z.shape[0]), int(z.shape[1])
    if n < 2:
        zero = z.sum() * 0.0
        return zero, zero
    zc = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0, unbiased=False) + eps)
    var_term = F.relu(float(variance_gamma) - std).mean()
    cov = (zc.T @ zc) / float(n - 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    cov_term = (off ** 2).sum() / float(d)
    return var_term, cov_term


@dataclass
class ZWorldP0Config:
    """Hyperparameters for the SD-070 P0 recipe.

    Defaults are the measured operating point from the 2026-07-18 sweep (see the module
    docstring's result table). They are NOT no-op defaults, because this whole object only
    exists once an experiment has chosen to run the recipe -- the no-op guarantee is
    structural (nothing constructs a trainer unless asked), not flag-based.
    """
    # Anti-collapse. w_cov is the PR lever and is deliberately far above VICReg's published
    # ratio: measured w_cov=0.04 -> PR 1.80, w_cov=50 -> PR 4.02 on the same sweep.
    variance_weight: float = 25.0
    covariance_weight: float = 50.0
    variance_gamma: float = 1.0
    # Grounding heads.
    presence_weight: float = 1.0
    distance_weight: float = 1.0
    n_distance_buckets: int = 4
    # SD-018 resource proximity, retained: it is the one leg of the old P0 that DOES learn
    # (held-out R2 0.794 from raw obs, 0.20-0.38 through the trained encoder).
    proximity_weight: float = 0.5
    # Structural anti-collapse: reconstructing world_obs cannot be served by a 1-D code.
    # 0.0 disables the head entirely.
    reconstruction_weight: float = 10.0
    # Optimisation.
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 12
    max_grad_norm: float = 1.0
    seed: int = 0
    # Fraction of the buffer held out for the readiness/discriminativeness readout.
    holdout_fraction: float = 0.2


class ZWorldP0Trainer:
    """Runs the SD-070 P0 recipe against an existing LatentStack.

    Trains exactly the z_world path -- `split_encoder.world_encoder` and
    `world_precision_logit` -- which is precisely the parameter set the V3-EXQ-783
    weight-delta readiness check watches. Top-down conditioning and the alpha_world
    temporal smoothing are applied at sense() time and carry no P0 gradient, so they are
    deliberately outside this path.

    Usage:
        trainer = ZWorldP0Trainer(agent.latent_stack, ZWorldP0Config(seed=seed))
        for each rollout step:
            trainer.observe(world_obs, resource_proximity_target=prox)
        stats = trainer.train()
    """

    def __init__(self, latent_stack: Any, config: Optional[ZWorldP0Config] = None) -> None:
        self.config = config or ZWorldP0Config()
        self.latent_stack = latent_stack
        se = getattr(latent_stack, "split_encoder", None)
        if se is None or not hasattr(se, "world_encoder"):
            raise ValueError(
                "ZWorldP0Trainer requires a LatentStack with a SplitEncoder exposing "
                "world_encoder (SD-005). Got %r." % (type(latent_stack).__name__,)
            )
        self.split_encoder = se
        self.world_dim = int(se.world_dim)
        self._obs: List[torch.Tensor] = []
        self._prox: List[float] = []
        self._heads: Dict[str, nn.Module] = {}
        self._recon_head: Optional[nn.Module] = None

    # -- buffer ------------------------------------------------------------------------
    def observe(
        self,
        world_obs: torch.Tensor,
        resource_proximity_target: Optional[float] = None,
    ) -> None:
        """Append one rollout observation to the P0 buffer.

        Stored detached and cloned: the buffer must not retain graph references across the
        rollout, and must not alias a tensor the environment reuses between steps.
        """
        w = world_obs.detach().float().reshape(-1).clone()
        self._obs.append(w)
        self._prox.append(
            float("nan") if resource_proximity_target is None
            else float(resource_proximity_target)
        )

    @property
    def n_buffered(self) -> int:
        return len(self._obs)

    # -- the trained path --------------------------------------------------------------
    def _z_world_path(self, world_obs: torch.Tensor) -> torch.Tensor:
        """Batched encoder + precision gate. Gradient reaches world_encoder and
        world_precision_logit -- exactly the world-path tensors the 783 readiness check
        counts as 'changed'."""
        se = self.split_encoder
        z = se.world_encoder(world_obs)
        return z * torch.sigmoid(se.world_precision_logit).unsqueeze(0)

    def world_path_parameters(self) -> List[torch.Tensor]:
        se = self.split_encoder
        return list(se.world_encoder.parameters()) + [se.world_precision_logit]

    # -- training ----------------------------------------------------------------------
    def train(self) -> Dict[str, Any]:
        """Run the recipe over the buffered rollout. Returns diagnostic statistics.

        The returned `holdout` block is the discriminativeness readout: balanced (macro-
        recall) accuracy per grounding head on data never trained on, against its own
        chance baseline. It exists because the anti-collapse gate alone can be satisfied
        VACUOUSLY -- a regulariser can hold the participation ratio up while the encoder
        learns nothing. A caller that reports PR without reporting these accuracies cannot
        tell a differentiated representation from a merely un-collapsed one.
        """
        cfg = self.config
        n = self.n_buffered
        if n < max(cfg.batch_size, 8):
            raise ValueError(
                "ZWorldP0Trainer.train() needs at least max(batch_size, 8)=%d buffered "
                "observations, got %d. Roll out further before training."
                % (max(cfg.batch_size, 8), n)
            )

        device = self.split_encoder.world_precision_logit.device
        obs = torch.stack(self._obs).to(device)
        prox = torch.tensor(self._prox, dtype=torch.float32, device=device)
        prox_ok = torch.isfinite(prox)

        targets = scene_structure_targets(obs, n_distance_buckets=cfg.n_distance_buckets)
        n_classes = {
            "hazard_present": 2,
            "resource_present": 2,
            "hazard_distance": cfg.n_distance_buckets,
            "resource_distance": cfg.n_distance_buckets,
        }

        gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        perm = torch.randperm(n, generator=gen).to(device)
        n_train = max(int(round((1.0 - cfg.holdout_fraction) * n)), cfg.batch_size)
        n_train = min(n_train, n - 1) if n > cfg.batch_size else n
        tr_idx, te_idx = perm[:n_train], perm[n_train:]

        # Heads are the TRAINER's, not the substrate's -- see the bit-identical-OFF note.
        self._heads = {
            k: nn.Linear(self.world_dim, c).to(device) for k, c in n_classes.items()
        }
        head_params: List[torch.Tensor] = [
            p for h in self._heads.values() for p in h.parameters()
        ]
        prox_head = getattr(self.split_encoder, "resource_proximity_head", None)
        use_prox = (
            prox_head is not None
            and cfg.proximity_weight > 0.0
            and int(prox_ok[tr_idx].sum()) >= 2
        )
        if use_prox:
            head_params += list(prox_head.parameters())
        if cfg.reconstruction_weight > 0.0:
            self._recon_head = nn.Linear(self.world_dim, int(obs.shape[1])).to(device)
            head_params += list(self._recon_head.parameters())
        else:
            self._recon_head = None

        class_w = {
            k: balanced_class_weights(targets[k][tr_idx], c) for k, c in n_classes.items()
        }
        world_params = self.world_path_parameters()
        opt = torch.optim.Adam(world_params + head_params, lr=cfg.learning_rate)

        n_steps = max((n_train // cfg.batch_size) * int(cfg.epochs), 1)
        bgen = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + 1)
        losses: List[float] = []
        last: Dict[str, float] = {}

        for _step in range(n_steps):
            sel = tr_idx[torch.randint(0, n_train, (cfg.batch_size,), generator=bgen).to(device)]
            z = self._z_world_path(obs[sel])
            loss = z.sum() * 0.0
            for k, head in self._heads.items():
                w_ = cfg.presence_weight if k.endswith("_present") else cfg.distance_weight
                if w_ <= 0.0:
                    continue
                loss = loss + w_ * F.cross_entropy(
                    head(z), targets[k][sel], weight=class_w[k]
                )
            if use_prox:
                m = prox_ok[sel]
                if bool(m.any()):
                    loss = loss + cfg.proximity_weight * F.mse_loss(
                        prox_head(z[m]).reshape(-1), prox[sel][m]
                    )
            if self._recon_head is not None:
                loss = loss + cfg.reconstruction_weight * F.mse_loss(
                    self._recon_head(z), obs[sel]
                )
            var_t, cov_t = variance_covariance_penalty(z, cfg.variance_gamma)
            loss = loss + cfg.variance_weight * var_t + cfg.covariance_weight * cov_t
            last = {"variance_term": float(var_t.detach()),
                    "covariance_term": float(cov_t.detach())}

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_params, max_norm=cfg.max_grad_norm)
            opt.step()
            losses.append(float(loss.detach()))

        stats: Dict[str, Any] = {
            "n_buffered": n,
            "n_train": int(n_train),
            "n_holdout": int(te_idx.numel()),
            "n_steps": n_steps,
            "final_loss": losses[-1] if losses else None,
            "mean_loss": float(sum(losses) / len(losses)) if losses else None,
            "used_proximity_head": bool(use_prox),
            "used_reconstruction_head": self._recon_head is not None,
            "label_balance": {
                k: (torch.bincount(targets[k], minlength=c).float() / float(n)).tolist()
                for k, c in n_classes.items()
            },
        }
        stats.update(last)
        stats["holdout"] = self._holdout_report(obs, targets, n_classes, te_idx)
        return stats

    def _holdout_report(
        self,
        obs: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        n_classes: Dict[str, int],
        te_idx: torch.Tensor,
    ) -> Dict[str, Any]:
        """Per-head balanced accuracy on held-out data, with its own chance baseline.

        Macro-recall over classes with at least 3 held-out examples; chance is 1/k for the
        k classes actually scored, so the baseline moves with the balance rather than
        being assumed at 1/n_classes.
        """
        if int(te_idx.numel()) < 8:
            return {"insufficient": True, "n_holdout": int(te_idx.numel())}
        out: Dict[str, Any] = {"insufficient": False}
        with torch.no_grad():
            z = self._z_world_path(obs[te_idx])
            for k, head in self._heads.items():
                pred = head(z).argmax(dim=-1)
                y = targets[k][te_idx]
                recalls = []
                for c in range(n_classes[k]):
                    m = y == c
                    if int(m.sum()) >= 3:
                        recalls.append(float((pred[m] == c).float().mean()))
                if recalls:
                    bal = float(sum(recalls) / len(recalls))
                    out[k] = {
                        "balanced_accuracy": bal,
                        "chance": 1.0 / float(len(recalls)),
                        "lift": bal - 1.0 / float(len(recalls)),
                        "n_classes_scored": len(recalls),
                    }
                else:
                    out[k] = {"balanced_accuracy": None, "note": "no class had >=3 held-out"}
        lifts = [v["lift"] for v in out.values()
                 if isinstance(v, dict) and v.get("lift") is not None]
        out["mean_lift"] = float(sum(lifts) / len(lifts)) if lifts else None
        return out
