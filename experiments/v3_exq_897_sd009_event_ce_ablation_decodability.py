"""V3-EXQ-897 -- SD-009 event-contrastive CE ablation: does the auxiliary loss make
event-type genuinely DECODABLE from z_world, held on top of the SD-070 anti-collapse
recipe?

CLAIM UNDER TEST. SD-009 / MECH-100: "z_world encoder requires event-type
cross-entropy auxiliary loss during training." genuine_exp_count=0 for SD-009 in
`REE_assembly/evidence/experiments/claim_evidence.v1.json` despite literature
confidence 0.724 -- the only prior exp evidence (EXQ-020/023 PASS) is tagged on the
old manifest with `claim_ids_tested` (not `claim_ids`) and no `substrate_hash`
(pre-2026-07-12 standard), so it never counted for SD-009 in the current indexer and
is, separately, not directly reusable here (GOV-REUSE-1 below).

OPEN GOVERNANCE QUESTION THIS RESOLVES. `REE_assembly/evidence/planning/
sd009_event_contrastive_channel_mismatch_2026-07-18.md` (AWAITING GOVERNANCE
ADJUDICATION) reports an UNREGISTERED, single-seed probe: `transition_type` is a
property of the TRANSITION (t-1 -> t) while z_world is a STATIC single-frame
encoding, and a probe trained directly on raw `world_obs` (no encoder in the path)
sits AT OR BELOW CHANCE for the event-type label (lift -0.014 3-class / -0.060
6-class), while the same label probes strongly from the BODY delta that SD-005
routes to z_self, not z_world. Trained through the (then-current, pre-SD-070) P0,
the SD-009 head itself reached held-out balanced accuracy 0.272-0.311 vs chance
0.333 -- never above chance. But that probe (a) used the PRE-SD-070 P0, which
COLLAPSES z_world (participation_ratio ~1.06, `zworld_p0.py` module docstring) --
a confound that would contaminate ANY naive SD-009 replication on the current
substrate, since a collapsed representation cannot be decodable for ANYTHING; and
(b) had no matched WITH-vs-WITHOUT-loss ablation, no positive control, and was
never registered as an experiment. This run repairs both: it holds the SD-070
anti-collapse machinery constant across arms (so neither arm collapses for reasons
unrelated to the event-CE term) and ablates ONLY the event-CE loss as an additive
term on top of it, with a genuine OFF arm and a validated positive-control probe.
SD-070 itself is NOT re-litigated here -- both arms use it identically; only the
extra event-type head is switched.

WHY THE PRIMARY DV IS DECODABILITY, NOT SELECTIVITY MARGIN. EXQ-020/023's statistic
(`selectivity_margin`, a cosine separation between class-mean z_world vectors) can be
non-zero from *correlates* of the event label (e.g. hazard-in-view, which the world
channel DOES encode -- see the positive control below) without the label ITSELF being
decodable. `selectivity_margin` and `event_classification_acc` are recorded here for
continuity/reconciliation with EXQ-020, but the PRE-REGISTERED PASS/FAIL routes on
held-out DECODABILITY MARGIN (macro-recall minus chance) from a genuinely independent,
frozen-encoder probe -- the sharper statistic the 2026-07-18 doc's open question turns
on.

DESIGN -- 2 arms x 3 seeds (44 excluded per the documented reef-config seed-44
instability):

  ARM_EVENT_CE_ON   SD-070 P0 recipe (unmodified, default hyperparameters) PLUS an
                     additional event-type CE head, trained IDENTICALLY to the
                     recipe's own grounding heads (class-balanced CE, same buffered
                     minibatch loop, same optimizer step) -- one more entry in the
                     trainer's per-head dict. Implemented as
                     ZWorldP0TrainerEventCE(ZWorldP0Trainer), a thin experiment-local
                     subclass that overrides train() with the SAME buffered-minibatch
                     structure (not a separate online per-step path), reusing
                     zworld_p0's public helpers (scene_structure_targets,
                     balanced_class_weights, variance_covariance_penalty). Does NOT
                     modify ree_core/latent/zworld_p0.py.
  ARM_EVENT_CE_OFF  Identical SD-070 recipe, plain (unmodified) ZWorldP0Trainer --
                     no event-type head, no event-type loss term. Everything else
                     (env exposure, rollout policy, buffer, other heads, optimiser,
                     hyperparameters) is bit-identical to the ON arm.

Neither arm sets `use_event_classifier=True` / uses
`agent.compute_event_contrastive_loss` (the substrate's own SD-009 wiring): that path
is designed for ONLINE per-step training at batch=1 -- exactly SD-070's documented
"fault 3" (no batch to compute variance/covariance statistics over). Reusing it here
would silently reintroduce the very collapse mechanism this run is designed to
control for. Instead the ON arm's event-type head is trained the SD-070 way: buffered
minibatches, class-balanced CE, alongside the anti-collapse penalty.

DVs, measured IDENTICALLY in both arms from a FROZEN encoder (phased: P0 encoder +/-
event-CE -> P1 freeze + independent probe train/holdout):

  PRIMARY (pre-registered, load-bearing): event-type held-out decodability margin --
    a FRESH nn.Linear(world_dim, 3) probe (never touched during P0), trained with
    class-balanced CE on an 80/20 train/holdout split of a NEW rollout collected AFTER
    the encoder is frozen, on .detach()ed z_world features. Reports held-out macro-
    recall minus its own class-count-adjusted chance (mirrors ZWorldP0Trainer's own
    _holdout_report convention: chance = 1/k over classes with >=3 held-out examples).
  Continuity (recorded, non-gating): `selectivity_margin` (cosine separation between
    mean z_world of event vs open states, mirroring V3-EXQ-177/EXQ-020's definition)
    and `event_classification_acc` (ON arm's own P0-internal holdout accuracy on the
    event head, i.e. what EXQ-020 reported -- None for the OFF arm, which trains no
    such head).
  POSITIVE CONTROL (readiness precondition, not a PASS criterion): hazard-in-view
    decodability, using `scene_structure_targets(world_obs)["hazard_present"]` (the
    SAME channel, SAME probe methodology, SAME train/holdout split as the primary DV)
    -- known decodable from raw world_obs at ~0.96 balanced accuracy per the
    2026-07-18 characterisation. If this probe cannot recover strong decodability for
    a target KNOWN to be present, that is a probe/instrument defect
    (`substrate_not_ready_requeue`), never a verdict on SD-009.

DV-SYMMETRY DECLARATION (mandatory per /queue-experiment Step 3 "DV-SYMMETRY
INVARIANCE"). The DV is held-out probe balanced-recall minus chance, a function of the
TRAINED WEIGHTS of the frozen z_world encoder (theta), estimated by an independently
re-initialised, freshly-trained linear head on theta's output. It is NOT an
argmax/rank-based statistic over interchangeable candidates, NOT a set-aggregate over
permutable units, and involves no broadcast-scalar or monotone-rescaling manipulation
anywhere in the pipeline -- the ON/OFF manipulation is an additive gradient term
applied to theta itself during P0, which is exactly the quantity the DV is sensitive
to. No symmetry in the checklist (broadcast-constant-invariant argmax, monotone-
rescaling-invariant rank statistic, permutation-invariant set-aggregate) applies: a
differently-trained theta produces a differently-expressive frozen representation, and
the probe is re-fit from scratch on each arm's actual features, so the DV can and does
move with the manipulation. Applies identically to both arms (symmetric pipeline,
asymmetric-by-construction P0 loss only).

PRE-REGISTERED PASS/FAIL (constants below, decided BEFORE running):
  supports  >= majority (2 of 3) seeds show: ON arm's event decodability margin
            clears EVENT_DECODABILITY_FLOOR AND exceeds the SAME-SEED OFF arm's
            margin by >= EVENT_CE_DELTA_MIN (paired, per-seed comparison).
  weakens   readiness OK, supports-count below majority, AND the ON arm's mean
            decodability margin itself sits at/below EVENT_DECODABILITY_FLOOR --
            i.e. event-type is genuinely NOT decodable from z_world even WITH the
            auxiliary loss and WITHOUT the pre-SD-070 collapse confound. This
            formalises (with a real ablation + validated positive control) the
            2026-07-18 informal finding, and is as informative a result as
            `supports`.
  unknown   readiness OK but neither of the above holds cleanly (e.g. ON is
            decodable but the ON-vs-OFF delta is not consistently positive -- SD-070's
            own grounding heads may already induce some event-correlated structure
            with or without the extra CE term, in which case the auxiliary loss's
            MARGINAL contribution is genuinely ambiguous from this design).
  substrate_not_ready_requeue  the positive-control precondition (hazard
            decodability) or the "encoder actually trained" precondition is unmet.
            NEVER read as an SD-009 verdict in that case.
outcome: PASS iff label=="supports"; FAIL otherwise (weakens / unknown /
  substrate_not_ready_requeue are all completed runs with a definite, informative,
  pre-registered non-support outcome -- not a run failure).

Single claim tagged (SD-009 only) -- no evidence_direction_per_claim needed. This run
does NOT tag MECH-100 (that mechanism claim rests on more than the CE-wiring question
tested here) and does NOT adjudicate SD-070's own status (unmodified, held constant).

GOV-REUSE-1: decisive readout = "held-out event-type decodability margin under the
SD-070 recipe, WITH vs WITHOUT an additional event-CE loss term." `reanalysis_query.py
query --readout event_classification_acc --claim SD-009` and `--readout
selectivity_margin --claim SD-009` (run from REE_assembly/) both returned 0 manifests
across the full 832-manifest corpus -- no run tags SD-009 with either readout name.
Direct inspection of the one candidate carrying `claim_ids_tested` including SD-009
(claim_probe_mech_100/runs/20260320T165149Z_v3_exq_020_event_contrastive_v3/
manifest.json, EXQ-020) confirms it predates the 2026-07-12 recording standard and
carries no `substrate_hash` at all -- UNVERIFIABLE / not substrate-compatible per
GOV-REUSE-1, and in any case trained under the pre-SD-070 (collapse-confounded) P0, so
even a compatible hash would not answer THIS question (SD-070-held-constant ablation).
Not recoverable -> run.

SUBSTRATE READINESS (2.5/2.5a). `ree-v3/CLAUDE.md`: SD-009 IMPLEMENTED (line 2241,
EXQ-020 PASS), SD-005 IMPLEMENTED (z_self/z_world split), SD-008 IMPLEMENTED
(alpha_world=0.9 factory preset), SD-070 IMPLEMENTED 2026-07-18. Empirically confirmed
(2.5a probe, this session): `ree_core/agent.py:9458-9516`
compute_event_contrastive_loss exists and is gated on
`config.latent.use_event_classifier` (not used by this design -- see above);
`ree_core/latent/stack.py`: `LatentState.event_logits` (~770), `SplitEncoder`
builds `self.event_classifier = nn.Linear(world_dim, 3)` when
`use_event_classifier=True` (~887) and computes it FROM z_world (~964); `LatentStack`
threads `use_event_classifier` (~1065); `REEConfig.from_dims` accepts and threads
`use_event_classifier` (config.py ~5577/~6688). `ree_core/latent/zworld_p0.py`
(ZWorldP0Trainer, ZWorldP0Config, scene_structure_targets, balanced_class_weights,
variance_covariance_penalty) read in full. `ree_core/environment/causal_grid_world.py`
confirms `info["transition_type"]` (line 3137) is the env's actual info-dict key.

SD-009's own status is NOT adjudicated by this run's design -- it is the run whose
outcome IS that adjudication once it lands and is reviewed; queuing it is not
pre-judging it.

PHASED TRAINING: P0 encoder (+/- event-CE) warmup, no downstream-probe loss. P1
freezes the encoder (no further optimiser steps on world_encoder /
world_precision_logit anywhere after P0) and trains fresh, independent probe heads on
.detach()ed z_world from a NEW rollout. P2 is the held-out evaluation split of that
same P1 rollout.

SLEEP DRIVER: not applicable -- no sleep loop is used.
"""

import argparse
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_protocol import emit_outcome

from experiments.pack_writer import write_flat_manifest

from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.capability_eval import LocalViewGreedyPolicy, RandomPolicy

from ree_core.agent import REEAgent
from ree_core.latent.zworld_p0 import (
    ZWorldP0Config,
    ZWorldP0Trainer,
    balanced_class_weights,
    scene_structure_targets,
    variance_covariance_penalty,
)
from ree_core.utils.config import REEConfig

import experiments.v3_exq_724_competence_localization_diagnostic as x724

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_897_sd009_event_ce_ablation_decodability"
QUEUE_ID = "V3-EXQ-897"
CLAIM_IDS: List[str] = ["SD-009"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

EVIDENCE_DIR = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"

# ---------------------------------------------------------------------------
# Env / agent config (pattern from experiments/_lib/baselines/exq783_zworld_granularity.py
# -- NOT that module's fingerprint machinery, a different lineage/claim)
# ---------------------------------------------------------------------------
WORLD_DIM = 128  # the ON/richer dim -- SD-009 is about representational content, not
                 # the granularity axis V3-EXQ-783 tests.


def env_kwargs() -> Dict[str, Any]:
    return dict(x724.ENV_KWARGS)


def make_env(seed: int):
    from ree_core.environment.causal_grid_world import CausalGridWorldV2
    return CausalGridWorldV2(seed=seed, **env_kwargs())


def agent_config_kwargs(env) -> Dict[str, Any]:
    """x724's base config, world_dim overridden to 128, use_event_classifier left
    False (this design uses a script-local event head instead of the substrate's,
    for the online-vs-batched reason stated in the module docstring)."""
    kwargs = x724._base_config_kwargs(env)
    kwargs["world_dim"] = WORLD_DIM
    kwargs["use_event_classifier"] = False
    return kwargs


def make_agent(env) -> REEAgent:
    cfg = REEConfig.from_dims(**agent_config_kwargs(env))
    return REEAgent(cfg)


ARMS: List[Dict[str, Any]] = [
    {"arm_id": "ARM_EVENT_CE_ON", "event_ce": True, "is_off_arm": False},
    {"arm_id": "ARM_EVENT_CE_OFF", "event_ce": False, "is_off_arm": True},
]

SEEDS: List[int] = [42, 43, 45]  # 44 excluded -- documented reef-config instability

# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
P0_ENCODER_EPISODES = 60
PROBE_EPISODES = 40
TOTAL_EPISODES_PER_CELL = P0_ENCODER_EPISODES + PROBE_EPISODES
STEPS_PER_EPISODE = 150
EXPLORE_EPSILON = 0.35

PROBE_HOLDOUT_FRACTION = 0.2
PROBE_TRAIN_EPOCHS = 30
PROBE_BATCH_SIZE = 64
PROBE_LR = 1e-2

# Weight of the added event-CE term in the ON arm, on the same scale as the SD-070
# recipe's own presence_weight (1.0) -- not tuned against this run's own outcome.
EVENT_CE_WEIGHT = 1.0
N_EVENT_CLASSES = 3

# SD-009 / MECH-100 event label convention (mirrors ree_core/agent.py _EVENT_LABEL_MAP
# exactly): obs_t is labeled with the transition_type that PRODUCED obs_t.
_EVENT_LABEL_MAP = {"none": 0, "env_caused_hazard": 1, "agent_caused_hazard": 2}

# ---------------------------------------------------------------------------
# Pre-registered thresholds (constants; never derived from this run's own statistics)
# ---------------------------------------------------------------------------
EVENT_DECODABILITY_FLOOR = 0.05      # macro-recall minus chance, ON arm must clear
EVENT_CE_DELTA_MIN = 0.03            # ON must beat paired-seed OFF by at least this
HAZARD_POSITIVE_CONTROL_FLOOR = 0.30  # positive-control probe sanity floor (informal
                                       # 2026-07-18 finding: ~0.96 acc vs chance 0.5,
                                       # i.e. ~0.46 margin -- floor set well below that
                                       # to tolerate probe/training variance)
MIN_CHANGED_WORLD_TENSORS = 1        # readiness: encoder must have actually trained
MAJORITY_SEEDS = 2                   # of 3 -- majority rule for "supports"


# ---------------------------------------------------------------------------
# Exploratory rollout policy (mirrors V3-EXQ-783's ExploratoryBalancedPolicy)
# ---------------------------------------------------------------------------
class ExploratoryBalancedPolicy:
    """LocalViewGreedy with epsilon-random exploration, so both hazard and resource
    events are visited (greedy alone under-visits hazards; pure random under-visits
    resources)."""

    name = "exploratory_balanced"

    def __init__(self, seed: int, epsilon: float = EXPLORE_EPSILON) -> None:
        self._greedy = LocalViewGreedyPolicy()
        self._random = RandomPolicy(seed=seed)
        self._rng = np.random.RandomState(int(seed) + 9973)
        self._epsilon = float(epsilon)

    def reset(self, env: Any) -> None:
        self._greedy.reset(env)
        self._random.reset(env)

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        if self._rng.rand() < self._epsilon:
            return self._random.act(env, obs_dict)
        return self._greedy.act(env, obs_dict)


def _resource_prox_target(obs_dict: Dict[str, Any]) -> Optional[float]:
    rfv = obs_dict.get("resource_field_view")
    if rfv is None:
        return None
    try:
        return float(torch.as_tensor(rfv).max().item())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SD-009 ablation trainer: adds an event-type CE head to the SD-070 recipe.
#
# This subclass overrides train() with the SAME buffered-minibatch structure as
# ZWorldP0Trainer.train() (same optimiser, same batching, same anti-collapse
# penalty), reusing its public helpers (scene_structure_targets,
# balanced_class_weights, variance_covariance_penalty) and its own buffer/path
# internals (self._obs / self._prox / self._z_world_path / world_path_parameters).
# It is the SAME training loop with one extra per-head entry -- NOT a separate
# online per-step path -- and it does not modify ree_core/latent/zworld_p0.py.
# ---------------------------------------------------------------------------
class ZWorldP0TrainerEventCE(ZWorldP0Trainer):
    """SD-009 ablation: event-type CE head trained alongside the SD-070 grounding
    heads, class-balanced, same buffered minibatch loop."""

    def __init__(self, latent_stack: Any, config: Optional[ZWorldP0Config] = None) -> None:
        super().__init__(latent_stack, config)
        self._event_labels: List[int] = []

    def observe_event(
        self,
        world_obs: torch.Tensor,
        event_label: int,
        resource_proximity_target: Optional[float] = None,
    ) -> None:
        self.observe(world_obs, resource_proximity_target)
        self._event_labels.append(int(event_label))

    def train(self) -> Dict[str, Any]:  # noqa: C901 -- mirrors ZWorldP0Trainer.train()
        cfg = self.config
        n = self.n_buffered
        if n < max(cfg.batch_size, 8):
            raise ValueError(
                "ZWorldP0TrainerEventCE.train() needs at least max(batch_size, 8)=%d "
                "buffered observations, got %d." % (max(cfg.batch_size, 8), n)
            )
        if len(self._event_labels) != n:
            raise ValueError(
                "event labels (%d) must be buffered 1:1 with observations (%d) via "
                "observe_event()." % (len(self._event_labels), n)
            )

        device = self.split_encoder.world_precision_logit.device
        obs = torch.stack(self._obs).to(device)
        prox = torch.tensor(self._prox, dtype=torch.float32, device=device)
        prox_ok = torch.isfinite(prox)
        event_y = torch.tensor(self._event_labels, dtype=torch.long, device=device)

        targets = scene_structure_targets(obs, n_distance_buckets=cfg.n_distance_buckets)
        targets["event_type"] = event_y
        n_classes = {
            "hazard_present": 2,
            "resource_present": 2,
            "hazard_distance": cfg.n_distance_buckets,
            "resource_distance": cfg.n_distance_buckets,
            "event_type": N_EVENT_CLASSES,
        }

        gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        perm = torch.randperm(n, generator=gen).to(device)
        n_train = max(int(round((1.0 - cfg.holdout_fraction) * n)), cfg.batch_size)
        n_train = min(n_train, n - 1) if n > cfg.batch_size else n
        tr_idx, te_idx = perm[:n_train], perm[n_train:]

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
                if k == "event_type":
                    w_ = EVENT_CE_WEIGHT
                else:
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


# ---------------------------------------------------------------------------
# P0 -- encoder warmup
# ---------------------------------------------------------------------------
def _world_path_param_names(agent: REEAgent) -> List[str]:
    names = []
    for name, _p in agent.latent_stack.named_parameters():
        if ("world_encoder" in name or "world_topdown" in name
                or "world_precision_logit" in name or "world_predictor" in name):
            names.append(name)
    return names


def _snapshot_world_path(agent: REEAgent) -> Dict[str, torch.Tensor]:
    keep = set(_world_path_param_names(agent))
    return {n: p.detach().clone() for n, p in agent.latent_stack.named_parameters() if n in keep}


def _count_changed(agent: REEAgent, snapshot: Dict[str, torch.Tensor]) -> Tuple[int, int, float]:
    changed = 0
    max_delta = 0.0
    for name, p in agent.latent_stack.named_parameters():
        if name not in snapshot:
            continue
        d = float((p.detach() - snapshot[name]).abs().max().item())
        max_delta = max(max_delta, d)
        if d > 0.0:
            changed += 1
    return changed, len(snapshot), max_delta


def _train_p0(
    agent: REEAgent, env, seed: int, arm_id: str, event_ce: bool, episodes: int,
    steps_per_episode: int, dry_run: bool = False,
) -> Tuple[Dict[str, Any], Any]:
    policy = ExploratoryBalancedPolicy(seed=seed)
    # A dry run buffers only a few dozen observations, far too few for the recipe's
    # batch statistics -- the trainer refuses such a buffer by design. Scale the
    # batch down EXPLICITLY for the smoke path so it still exercises the real
    # training code (mirrors V3-EXQ-783's own dry-run override), never the real-run
    # config.
    p0_cfg = ZWorldP0Config(seed=int(seed), batch_size=8, epochs=2) if dry_run \
        else ZWorldP0Config(seed=int(seed))
    trainer = ZWorldP0TrainerEventCE(agent.latent_stack, p0_cfg) if event_ce \
        else ZWorldP0Trainer(agent.latent_stack, p0_cfg)

    label_counts: Dict[str, int] = {}

    for ep in range(episodes):
        _flat0, obs_dict = env.reset()
        policy.reset(env)
        prev_ttype = "none"  # obs_0 has no producing transition (reset)

        for _step in range(steps_per_episode):
            world_obs = obs_dict["world_state"].float()
            label_counts[prev_ttype] = label_counts.get(prev_ttype, 0) + 1
            if event_ce:
                trainer.observe_event(
                    world_obs, _EVENT_LABEL_MAP.get(prev_ttype, 0),
                    _resource_prox_target(obs_dict),
                )
            else:
                trainer.observe(world_obs, _resource_prox_target(obs_dict))

            action = policy.act(env, obs_dict)
            with torch.no_grad():
                _flat, _harm, done, info, obs_dict = env.step(action)
            if not isinstance(info, dict):
                info = {}
            prev_ttype = str(info.get("transition_type", "none"))
            if done:
                break

        if ep == 0 or (ep + 1) % 20 == 0:
            print(
                "  [train] %s seed=%d ep %d/%d (P0 SD-070%s)"
                % (arm_id, seed, ep + 1, TOTAL_EPISODES_PER_CELL,
                   " + event-CE" if event_ce else ""),
                flush=True,
            )

    stats = trainer.train()
    out: Dict[str, Any] = {
        "p0_n_buffered": int(trainer.n_buffered),
        "p0_label_counts": dict(label_counts),
        "p0_mean_loss": stats.get("mean_loss"),
        "p0_final_loss": stats.get("final_loss"),
        "p0_n_steps": stats.get("n_steps"),
        "p0_variance_term": stats.get("variance_term"),
        "p0_covariance_term": stats.get("covariance_term"),
        "p0_grounding_label_balance": stats.get("label_balance"),
        "p0_holdout": stats.get("holdout"),
    }
    if event_ce:
        ho = stats.get("holdout") or {}
        ev = ho.get("event_type") or {}
        out["event_classification_acc"] = ev.get("balanced_accuracy")
    else:
        out["event_classification_acc"] = None
    return out, trainer


# ---------------------------------------------------------------------------
# P1/P2 -- freeze encoder, train + evaluate independent probes
# ---------------------------------------------------------------------------
def _train_and_eval_probe(
    z_train: torch.Tensor, y_train: torch.Tensor,
    z_holdout: torch.Tensor, y_holdout: torch.Tensor,
    n_classes: int, seed: int,
) -> Dict[str, Any]:
    """A FRESH, independent linear probe (never touched during P0), class-balanced
    CE, trained on .detach()ed frozen z_world. Held-out macro-recall minus its own
    class-count-adjusted chance, mirroring ZWorldP0Trainer._holdout_report."""
    torch.manual_seed(int(seed) + 31337)
    dim = int(z_train.shape[1])
    head = nn.Linear(dim, n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=PROBE_LR)
    class_w = balanced_class_weights(y_train, n_classes)
    n = int(z_train.shape[0])
    bs = min(PROBE_BATCH_SIZE, max(n, 1))
    gen = torch.Generator().manual_seed(int(seed) + 271828)

    for _epoch in range(PROBE_TRAIN_EPOCHS):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if idx.numel() == 0:
                continue
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(z_train[idx]), y_train[idx], weight=class_w)
            loss.backward()
            opt.step()

    with torch.no_grad():
        pred = head(z_holdout).argmax(dim=-1)
        recalls = []
        for c in range(n_classes):
            m = y_holdout == c
            if int(m.sum()) >= 3:
                recalls.append(float((pred[m] == c).float().mean()))
        if not recalls:
            return {"balanced_accuracy": None, "chance": None, "margin": None,
                    "n_classes_scored": 0, "n_holdout": int(y_holdout.shape[0]),
                    "insufficient": True}
        bal = float(sum(recalls) / len(recalls))
        chance = 1.0 / float(len(recalls))
        return {
            "balanced_accuracy": bal, "chance": chance, "margin": bal - chance,
            "n_classes_scored": len(recalls), "n_holdout": int(y_holdout.shape[0]),
            "insufficient": False,
        }



def _selectivity_margin(z_all: torch.Tensor, event_y: torch.Tensor) -> Optional[float]:
    """Cosine separation between mean z_world of event (label != 0) vs open (label
    == 0) states -- mirrors V3-EXQ-177/EXQ-020's selectivity_margin definition, kept
    for continuity/reconciliation with the prior evidence (not gating)."""
    event_mask = event_y != 0
    open_mask = event_y == 0
    if int(event_mask.sum()) < 10 or int(open_mask.sum()) < 10:
        return None
    m_e = z_all[event_mask].mean(dim=0)
    m_o = z_all[open_mask].mean(dim=0)
    denom = float(torch.linalg.norm(m_e) * torch.linalg.norm(m_o))
    if denom < 1e-12:
        return None
    cos_sim = float(torch.dot(m_e, m_o) / denom)
    return float(1.0 - cos_sim)


def _collect_probe_data_full(
    agent: REEAgent, trainer: ZWorldP0Trainer, env, seed: int, episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """As _collect_probe_data, but also buffers world_obs so hazard-in-view (the
    positive control) can be derived with scene_structure_targets over the exact
    same states."""
    policy = ExploratoryBalancedPolicy(seed=seed + 5000)
    z_rows: List[torch.Tensor] = []
    obs_rows: List[torch.Tensor] = []
    event_labels: List[int] = []
    label_counts: Dict[str, int] = {}

    for ep in range(episodes):
        _flat0, obs_dict = env.reset()
        policy.reset(env)
        prev_ttype = "none"

        for _step in range(steps_per_episode):
            world_obs = obs_dict["world_state"].float()
            with torch.no_grad():
                z = trainer._z_world_path(
                    world_obs.unsqueeze(0) if world_obs.dim() == 1 else world_obs
                )
            z_rows.append(z.detach().squeeze(0))
            obs_rows.append(world_obs.detach().reshape(-1).clone())
            event_labels.append(_EVENT_LABEL_MAP.get(prev_ttype, 0))
            label_counts[prev_ttype] = label_counts.get(prev_ttype, 0) + 1

            action = policy.act(env, obs_dict)
            with torch.no_grad():
                _flat, _harm, done, info, obs_dict = env.step(action)
            if not isinstance(info, dict):
                info = {}
            prev_ttype = str(info.get("transition_type", "none"))
            if done:
                break

        if ep == 0 or (ep + 1) % 20 == 0:
            print(
                "  [train] probe seed=%d ep %d/%d (P1 probe rollout)"
                % (seed, P0_ENCODER_EPISODES + ep + 1, TOTAL_EPISODES_PER_CELL),
                flush=True,
            )

    z_all = torch.stack(z_rows) if z_rows else torch.zeros((0, trainer.world_dim))
    obs_all = torch.stack(obs_rows) if obs_rows else torch.zeros((0, 1))
    event_y = torch.tensor(event_labels, dtype=torch.long)
    hazard_y = (
        scene_structure_targets(obs_all)["hazard_present"]
        if obs_all.shape[0] > 0 else torch.zeros((0,), dtype=torch.long)
    )
    return {
        "z": z_all, "event_y": event_y, "hazard_y": hazard_y,
        "label_counts": label_counts, "n": int(z_all.shape[0]),
    }


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------
def _config_slice_for(arm: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "lineage": "exq896_sd009_event_ce_ablation",
        "env_kwargs": env_kwargs(),
        "world_dim": WORLD_DIM,
        "event_ce": arm["event_ce"],
        "use_event_classifier": False,
        "p0_recipe": "sd070_plus_event_ce" if arm["event_ce"] else "sd070",
        "alpha_world": 0.9,
        "schedule": {
            "p0_encoder_episodes": P0_ENCODER_EPISODES,
            "probe_episodes": PROBE_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
        },
        "probe_config": {
            "n_event_classes": N_EVENT_CLASSES,
            "probe_holdout_fraction": PROBE_HOLDOUT_FRACTION,
            "probe_train_epochs": PROBE_TRAIN_EPOCHS,
            "probe_batch_size": PROBE_BATCH_SIZE,
            "probe_lr": PROBE_LR,
        },
    }


def _run_cell(arm: Dict[str, Any], seed: int, dry_run: bool) -> Dict[str, Any]:
    arm_id = arm["arm_id"]
    event_ce = bool(arm["event_ce"])

    p0_eps = 2 if dry_run else P0_ENCODER_EPISODES
    probe_eps = 2 if dry_run else PROBE_EPISODES
    steps = 25 if dry_run else STEPS_PER_EPISODE

    print("Seed %d Condition %s" % (seed, arm_id), flush=True)

    row: Dict[str, Any] = {"arm_id": arm_id, "seed": seed, "event_ce": event_ce}

    with arm_cell(
        seed, config_slice=_config_slice_for(arm), script_path=Path(__file__),
        config_slice_declared=True, include_driver_script_in_hash=False,
    ) as cell:
        env = make_env(seed)
        agent = make_agent(env)

        pre = _snapshot_world_path(agent)
        p0_out, trainer = _train_p0(agent, env, seed, arm_id, event_ce, p0_eps, steps,
                                     dry_run=dry_run)
        changed, total, max_delta = _count_changed(agent, pre)
        row.update(p0_out)
        row["world_path_tensors_changed"] = int(changed)
        row["world_path_tensors_total"] = int(total)
        row["world_path_max_abs_delta"] = float(max_delta)

        # P1/P2: encoder frozen from here on -- no optimiser touches world_encoder /
        # world_precision_logit again in this cell.
        probe_data = _collect_probe_data_full(agent, trainer, env, seed, probe_eps, steps)
        row["probe_label_counts"] = probe_data["label_counts"]
        row["probe_n_states"] = probe_data["n"]

        n = probe_data["n"]
        if n >= 8:
            gen = torch.Generator().manual_seed(int(seed) + 777)
            perm = torch.randperm(n, generator=gen)
            n_train = max(int(round((1.0 - PROBE_HOLDOUT_FRACTION) * n)), 1)
            n_train = min(n_train, n - 1) if n > 1 else n
            tr, te = perm[:n_train], perm[n_train:]

            z_all, event_y = probe_data["z"], probe_data["event_y"]
            hazard_y = probe_data["hazard_y"]

            event_probe = _train_and_eval_probe(
                z_all[tr], event_y[tr], z_all[te], event_y[te], N_EVENT_CLASSES, seed
            )
            hazard_probe = _train_and_eval_probe(
                z_all[tr], hazard_y[tr], z_all[te], hazard_y[te], 2, seed
            )
            selectivity = _selectivity_margin(z_all, event_y)
        else:
            event_probe = {"balanced_accuracy": None, "chance": None, "margin": None,
                            "insufficient": True, "n_holdout": 0}
            hazard_probe = dict(event_probe)
            selectivity = None

        row["event_probe"] = event_probe
        row["hazard_probe"] = hazard_probe
        row["selectivity_margin"] = selectivity

        cell.stamp(row)

    ok = row.get("event_probe", {}).get("margin") is not None
    print("verdict: %s" % ("PASS" if ok else "FAIL"), flush=True)
    return row


# ---------------------------------------------------------------------------
# Aggregation + pre-registered self-route
# ---------------------------------------------------------------------------
def _label_balance(counts: Dict[str, int]) -> Dict[str, float]:
    tot = float(sum(counts.values())) or 1.0
    return {k: float(v) / tot for k, v in sorted(counts.items())}


def _self_route(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pre-registered interpretation grid. A HYPOTHESIS routing rule, applied
    mechanically to the measured margins -- either 'supports' or 'weakens' is a
    valid, informative outcome (see module docstring)."""
    preconditions: List[Dict[str, Any]] = []

    # Readiness 1 (POSITIVE CONTROL): hazard-in-view decodability, SAME probe
    # methodology as the primary DV, min across all cells (both arms -- the world
    # channel's static scene structure is present regardless of the event-CE term).
    hazard_margins = [
        float(r["hazard_probe"]["margin"]) for r in rows
        if r.get("hazard_probe", {}).get("margin") is not None
    ]
    hazard_min = float(min(hazard_margins)) if hazard_margins else 0.0
    preconditions.append({
        "name": "positive_control_hazard_in_view_decodable",
        "description": (
            "hazard_in_view (scene_structure_targets['hazard_present'], a STATIC "
            "single-frame property of world_obs known decodable at ~0.96 balanced "
            "accuracy per the 2026-07-18 characterisation) must be recoverable by "
            "the SAME probe methodology as the primary event-type DV. If this fails, "
            "the probe pipeline itself is broken -- substrate_not_ready_requeue, "
            "never an SD-009 verdict."
        ),
        "control": "hazard_in_view target, min held-out margin across all cells (both arms)",
        "measured": hazard_min,
        "threshold": HAZARD_POSITIVE_CONTROL_FLOOR,
        "met": bool(hazard_min >= HAZARD_POSITIVE_CONTROL_FLOOR),
        "load_bearing": True,
    })

    # Readiness 2: every cell's encoder must have actually moved (P0 optimiser ran).
    changed_counts = [int(r.get("world_path_tensors_changed", 0)) for r in rows]
    min_changed = int(min(changed_counts)) if changed_counts else 0
    preconditions.append({
        "name": "encoder_actually_trained",
        "description": (
            "Every cell must show >=1 changed world-path tensor across P0 -- the "
            "manipulation did not take otherwise, and no SD-009 conclusion is "
            "licensed."
        ),
        "control": "world-path latent_stack tensors, min across all cells",
        "measured": min_changed,
        "threshold": MIN_CHANGED_WORLD_TENSORS,
        "met": bool(min_changed >= MIN_CHANGED_WORLD_TENSORS),
        "load_bearing": True,
    })

    readiness_ok = all(bool(p["met"]) for p in preconditions)

    # Paired per-seed comparison.
    on_by_seed = {r["seed"]: r for r in rows if r["arm_id"] == "ARM_EVENT_CE_ON"}
    off_by_seed = {r["seed"]: r for r in rows if r["arm_id"] == "ARM_EVENT_CE_OFF"}
    paired_seeds = sorted(set(on_by_seed) & set(off_by_seed))

    per_seed: List[Dict[str, Any]] = []
    on_margins: List[float] = []
    for s in paired_seeds:
        on_m = on_by_seed[s].get("event_probe", {}).get("margin")
        off_m = off_by_seed[s].get("event_probe", {}).get("margin")
        if on_m is None or off_m is None:
            per_seed.append({"seed": s, "on_margin": on_m, "off_margin": off_m,
                              "delta": None, "supports": False})
            continue
        on_margins.append(float(on_m))
        delta = float(on_m) - float(off_m)
        supports = bool(on_m >= EVENT_DECODABILITY_FLOOR and delta >= EVENT_CE_DELTA_MIN)
        per_seed.append({"seed": s, "on_margin": float(on_m), "off_margin": float(off_m),
                          "delta": delta, "supports": supports})

    supports_count = sum(1 for p in per_seed if p["supports"])
    mean_on_margin = float(np.mean(on_margins)) if on_margins else None

    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        direction = "unknown"
    elif supports_count >= MAJORITY_SEEDS:
        label = "sd009_event_ce_supports_decodability"
        direction = "supports"
    elif mean_on_margin is not None and mean_on_margin <= EVENT_DECODABILITY_FLOOR:
        label = "sd009_event_ce_at_or_below_chance"
        direction = "weakens"
    else:
        label = "sd009_event_ce_effect_inconclusive"
        direction = "unknown"

    criteria = [
        {"name": "event_ce_on_clears_floor_and_beats_off_majority_seeds",
         "load_bearing": True,
         "passed": bool(readiness_ok and supports_count >= MAJORITY_SEEDS)},
    ]
    criteria_non_degenerate = {
        "primary": bool(readiness_ok and len(per_seed) >= 2
                         and any(p["delta"] is not None for p in per_seed)),
    }

    return {
        "label": label,
        "direction": direction,
        "preconditions": preconditions,
        "readiness_ok": readiness_ok,
        "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
        "combination_rule": (
            "supports iff readiness_ok AND >=%d of %d seeds individually satisfy "
            "(on_margin >= %.3f AND on_margin - off_margin >= %.3f); weakens iff "
            "readiness_ok AND mean(on_margin) <= %.3f; else unknown (readiness_ok "
            "but neither clean pattern holds); substrate_not_ready_requeue iff NOT "
            "readiness_ok. Majority + paired-delta rule, not a plain criteria AND."
            % (MAJORITY_SEEDS, len(paired_seeds), EVENT_DECODABILITY_FLOOR,
               EVENT_CE_DELTA_MIN, EVENT_DECODABILITY_FLOOR)
        ),
        "per_seed": per_seed,
        "supports_count": supports_count,
        "n_seeds": len(paired_seeds),
        "mean_on_margin": mean_on_margin,
        "thresholds": {
            "event_decodability_floor": EVENT_DECODABILITY_FLOOR,
            "event_ce_delta_min": EVENT_CE_DELTA_MIN,
            "hazard_positive_control_floor": HAZARD_POSITIVE_CONTROL_FLOOR,
            "min_changed_world_tensors": MIN_CHANGED_WORLD_TENSORS,
            "majority_seeds": MAJORITY_SEEDS,
        },
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = SEEDS[:1] if dry_run else SEEDS
    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            rows.append(_run_cell(arm, seed, dry_run))

    interp = _self_route(rows)
    direction = interp["direction"]
    outcome = "PASS" if direction == "supports" else "FAIL"

    p0_train_counts: Dict[str, int] = {}
    probe_counts: Dict[str, int] = {}
    for r in rows:
        for k, v in (r.get("p0_label_counts") or {}).items():
            p0_train_counts[k] = p0_train_counts.get(k, 0) + v
        for k, v in (r.get("probe_label_counts") or {}).items():
            probe_counts[k] = probe_counts.get(k, 0) + v

    return {
        "outcome": outcome,
        "arm_results": rows,
        "interpretation": interp,
        "label_balance": {
            "transition_type": {
                "p0_train_fracs": _label_balance(p0_train_counts),
                "probe_fracs": _label_balance(probe_counts),
                "p0_train_counts": p0_train_counts,
                "probe_counts": probe_counts,
            },
        },
        "seeds_used": seeds,
    }


def _build_manifest(result: Dict[str, Any], timestamp_utc: str,
                    dry_run: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    run_id = "%s_%s_v3" % (EXPERIMENT_TYPE, timestamp_utc)
    interp = result["interpretation"]

    full_config = {
        "env_kwargs": env_kwargs(),
        "arms": ARMS,
        "seeds": result["seeds_used"],
        "world_dim": WORLD_DIM,
        "p0_encoder_episodes": P0_ENCODER_EPISODES,
        "probe_episodes": PROBE_EPISODES,
        "total_episodes_per_cell": TOTAL_EPISODES_PER_CELL,
        "steps_per_episode": STEPS_PER_EPISODE,
        "explore_epsilon": EXPLORE_EPSILON,
        "p0_recipe": "sd070",
        "p0_config": asdict(ZWorldP0Config()),
        "event_ce_weight": EVENT_CE_WEIGHT,
        "n_event_classes": N_EVENT_CLASSES,
        "probe_holdout_fraction": PROBE_HOLDOUT_FRACTION,
        "probe_train_epochs": PROBE_TRAIN_EPOCHS,
        "probe_batch_size": PROBE_BATCH_SIZE,
        "probe_lr": PROBE_LR,
        "thresholds": interp["thresholds"],
        "dry_run": bool(dry_run),
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": result["outcome"],
        "timestamp_utc": timestamp_utc,
        "dry_run": bool(dry_run),
        "evidence_direction": interp["direction"],
        "interpretation": interp,
        "arm_results": result["arm_results"],
        "label_balance": result["label_balance"],
        "custom_information": {
            "bears_on": (
                "SD-009 ONLY. Does NOT tag or adjudicate MECH-100 (broader mechanism "
                "claim) or SD-070 (held constant, unmodified, identically in both "
                "arms)."
            ),
            "open_question_source": (
                "REE_assembly/evidence/planning/"
                "sd009_event_contrastive_channel_mismatch_2026-07-18.md "
                "(AWAITING GOVERNANCE ADJUDICATION) -- this run is a registered "
                "ablation + validated positive control designed to resolve that "
                "doc's open reconciliation question (selectivity_margin vs "
                "decodability), holding SD-070's anti-collapse recipe constant so "
                "the pre-SD-070 collapse confound documented there cannot "
                "contaminate the result."
            ),
            "gov_reuse_1_check": (
                "reanalysis_query.py query --readout event_classification_acc "
                "--claim SD-009 and --readout selectivity_margin --claim SD-009 "
                "(run from REE_assembly/) -> 0 manifests matched across the full "
                "832-manifest corpus. The one candidate with claim_ids_tested "
                "including SD-009 (EXQ-020, claim_probe_mech_100/runs/"
                "20260320T165149Z_v3_exq_020_event_contrastive_v3/manifest.json) "
                "carries no substrate_hash (pre-2026-07-12 standard) and trained "
                "under the pre-SD-070 collapse-confounded P0 -- UNVERIFIABLE / not "
                "substrate-compatible, and not the same question even if it were. "
                "Not recoverable -> run."
            ),
        },
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
    }
    return manifest, full_config


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started_at = time.perf_counter()
    result = run_experiment(dry_run=args.dry_run)
    timestamp_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest, full_config = _build_manifest(result, timestamp_utc, args.dry_run)

    out_path = write_flat_manifest(
        manifest,
        EVIDENCE_DIR,
        dry_run=args.dry_run,
        config=full_config,
        seeds=result["seeds_used"],
        script_path=Path(__file__),
        started_at=started_at,
        json_default=str,
    )

    print("", flush=True)
    print("=" * 72, flush=True)
    print("%s -- outcome=%s" % (QUEUE_ID, manifest["outcome"]), flush=True)
    print("self-route label: %s" % manifest["interpretation"]["label"], flush=True)
    for a in ARMS:
        margins = [
            r["event_probe"]["margin"] for r in result["arm_results"]
            if r["arm_id"] == a["arm_id"] and r.get("event_probe", {}).get("margin") is not None
        ]
        if margins:
            print("  %-18s event decodability margin mean=%.4f"
                  % (a["arm_id"], float(np.mean(margins))), flush=True)
    print("manifest: %s" % out_path, flush=True)
    print("=" * 72, flush=True)

    return str(out_path), manifest["outcome"], args.dry_run


if __name__ == "__main__":
    _out_path, _outcome, _dry = main()
    _outcome_raw = str(_outcome).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=_dry,
    )
