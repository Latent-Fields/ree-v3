"""Shared 724-A0 all-ON training toolkit -- the SUBSTRATE half of the driver family.

WHY THIS MODULE EXISTS: A SUBSTRATE-HASH UNDER-INCLUSION DEFECT (found 2026-07-22).
`experiments/_lib/arm_fingerprint.py` hashes a cell's computation from `_SUBSTRATE_GLOBS`
-- `ree_core/**`, `experiments/_harness.py`, `experiments/_metrics.py` and
`experiments/_lib/**`. A file under `experiments/` that is NOT in `_lib` enters the hash
by exactly one route: as the calling script's own path, folded in by
`include_driver_script_in_hash` (default True).

Cross-driver arm REUSE requires `include_driver_script_in_hash=False` on both the mint and
the consumer -- otherwise the two distinct drivers can never hash-match. Two such callers
exist (v3_exq_742_mech457_actor_critic_onoff, v3_exq_742m_..._bias_head_baseline_mint) and
both executed training code that lived in DRIVER files: x734's `_train_all_on_agent` and the
x724 helpers it calls. Those files are in no glob AND excluded as own-script by that very
flag, so an edit to the training recipe was INVISIBLE to the substrate hash -- a banked arm
computed by the old code could cache-HIT a consumer running the new code, silently comparing
a treatment and a control trained by different code.

arm_fingerprint's own safety direction (see its docstring) is that OVER-inclusion causes
false MISSES only; UNDER-inclusion is the unsafe direction. This was under-inclusion.

THE FIX IS STRUCTURAL, NOT PER-CALLER. Passing an explicit `scope=` naming the driver would
also work but has to be remembered at every call site. Living inside `experiments/_lib/**`
means the existing glob covers this code and every future caller inherits the coverage.

THE CONTRACT THIS ESTABLISHES: code that a cell EXECUTES belongs in `experiments/_lib/`;
code that is specific to one driver (env construction, arm tables, manifest shape) stays in
that driver. Each caller keeps its own env construction and its own call-site guard -- only
the shared computation moved.

WHAT MOVED, AND FROM WHERE (contents are verbatim; only `x724.` prefixes were dropped, since
those symbols are now local):
  - v3_exq_724_competence_localization_diagnostic.py : the SD-056 e2 online-training
    primitives, the obs helpers, `_consumed_summaries`, and the P1 two-head REINFORCE losses,
    plus the constants they read.
  - v3_exq_734_env_difficulty_competence_recovery_sweep.py : `_train_all_on_agent` (the
    rung-parameterised A0 recipe) and `E2_TRAIN_IN_P1`.
Both drivers import these names back, so `x724._e2_contrastive_step`,
`x734._train_all_on_agent` and every other existing attribute path still resolve. The other
importers of x734's function (v3_exq_737, v3_exq_742, v3_exq_808, `_lib/mech457_fanout.py`,
`_lib/baselines/exq742_mech457_bias_head_baseline.py`) are unchanged.

DELIBERATELY NOT COLLAPSED: v3_exq_728b_trained_allon_capability_point.py keeps its own
independent `_train_all_on_agent` and its own vendored copy of the x724 toolkit. That was
re-decided DO-NOT-COLLAPSE on 2026-07-22 (see the `collapse_decision` field for
`sd_zworld_warmup_optimizer_group` in REE_assembly/evidence/planning/substrate_queue.json):
x728b vendors the toolkit as deliberate baseline insulation and has a live PASS
(v3_exq_728b_trained_allon_capability_point_20260721T113845Z_v3) whose banked arms an edit
would invalidate. It imports nothing from x724, so this move does not touch its hash.

ONE-TIME HASH BUST, IN THE SAFE DIRECTION. Moving these bytes INTO `experiments/_lib/**`
changes `compute_substrate_hash` for every cell using the default scope, so previously
banked arms miss once and are recomputed. That is the false-MISS direction (wasted compute,
never a wrong comparison) and is the intended cost of closing the false-HIT hole.

ASCII-only in printed output (repo rule).
"""

from __future__ import annotations

import math
import random
from collections import deque  # noqa: F401  (callers build buffers with this)
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

from experiments._lib.capability_eval import RandomPolicy
from experiments._lib.zworld_p0_warmup import run_zworld_p0
from experiments._lib.zharm_a_p0_warmup import ZHarmAP0Config, run_zharm_a_p0
from ree_core.latent.zworld_p0 import ZWorldP0Config  # SD-106: zworld_p0_config type
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2

# ---------------------------------------------------------------------------
# Training-recipe constants. MOVED HERE FROM x724/x734 ON PURPOSE: a learning rate is
# exactly the kind of edit that must bust the substrate hash, and a constant left behind in
# a driver file would not be hashed even though the moved functions read it.
# ---------------------------------------------------------------------------
LR_OFC_DEVAL = 2e-3

# SD-056 online e2 training (714 harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# P1 bias-head REINFORCE training (714).
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9

# A0 recipe: SD-056 e2 encoder FROZEN through P1 (moved from x734).
E2_TRAIN_IN_P1 = False


# ---------------------------------------------------------------------------
# SD-056 online e2 training (mirror V3-EXQ-719a)
# ---------------------------------------------------------------------------
def _sample_class_diverse_batch(
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    k: int,
    rng: random.Random,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None
    pool = list(buffer)
    rng.shuffle(pool)
    seen_classes: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for tup in pool:
        cls = int(tup[1].argmax().item())
        if cls not in seen_classes:
            seen_classes[cls] = tup
        if len(seen_classes) >= k:
            break
    if len(seen_classes) < MIN_CLASSES_FOR_TRAIN:
        return None
    samples = list(seen_classes.values())
    picked_ids = {id(s) for s in samples}
    for tup in pool:
        if len(samples) >= k:
            break
        if id(tup) in picked_ids:
            continue
        samples.append(tup)
        picked_ids.add(id(tup))
    return samples


def _e2_contrastive_step(
    agent: REEAgent,
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimiser: torch.optim.Optimizer,
    rng: random.Random,
) -> Optional[float]:
    batch = _sample_class_diverse_batch(buffer, CONTRASTIVE_BATCH_K, rng)
    if batch is None:
        return None
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)
    optimiser.zero_grad(set_to_none=True)
    loss = agent.e2.world_forward_contrastive_loss(
        z_world_0=z0_K,
        actions=actions_K,
        z_world_1_targets=z1_K,
        simulation_mode=False,
    )
    if not torch.is_tensor(loss):
        return None
    loss_val = float(loss.detach().item())
    if not math.isfinite(loss_val):
        return loss_val
    if not loss.requires_grad or loss_val == 0.0:
        return loss_val
    weighted = SD056_WEIGHT * loss
    weighted.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return loss_val


# ---------------------------------------------------------------------------
# obs helpers (mirror V3-EXQ-719a)
# ---------------------------------------------------------------------------
def _obs_harm(obs_dict):
    h = obs_dict.get("harm_obs")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_a(obs_dict):
    h = obs_dict.get("harm_obs_a")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_history(obs_dict):
    h = obs_dict.get("harm_history")
    return h.float().unsqueeze(0) if h is not None else None


def _consumed_summaries(agent: REEAgent, candidates) -> Optional[torch.Tensor]:
    summ = agent._candidate_world_summaries(candidates)
    if summ is not None:
        return summ.detach()
    rows: List[torch.Tensor] = []
    for c in candidates:
        if c.world_states is not None:
            rows.append(c.get_world_state_sequence()[0, 0, :].detach())
        elif agent._current_latent is not None:
            rows.append(agent._current_latent.z_world[0].detach())
        else:
            return None
    return torch.stack(rows, dim=0) if rows else None


# ---------------------------------------------------------------------------
# ABSORPTION / CONVERSION TELEMETRY (sd-allon-training-signal-absorption-telemetry)
#
# WHY THIS EXISTS. `_train_all_on_agent` returned tick and training-step counts only, so a
# run whose MANIPULATION is on the reward / training-signal channel could produce a
# flat-across-arms null that is indistinguishable between two very different worlds: the
# manipulation was ABSORBED (it never reached a parameter) versus it COULD have converted
# into different committed behaviour and did not. V3-EXQ-1039 is the confirmed case -- a
# per-episode return 250-500x larger than its control reproduced that control bit-identically
# on every recorded channel in all 3 seeds, and the readiness gate built to catch exactly
# that (C0d) passed at 0.667. Nothing in the returned dict could tell the two apart.
#
# NO-OP BY CONSTRUCTION, NOT BY FLAG. Everything below is read-only accounting over values
# the recipe already computes. No optimizer step, no parameter write, and -- the property
# that matters most -- NO RNG DRAW: the two `np.random.choice` calls in the REINFORCE helpers
# are untouched in count and in order, so every fixed-seed comparison that spans this commit
# stays valid (cf. the MECH-091 phase_reset incident, where a shifted fixed-seed RNG stream
# silently confounded every comparison across its boundary). The change is ADDITIVE KEYS in
# the returned dict plus one optional-keyword accumulator on the two loss helpers.
#
# ABSENT IS None, NEVER 0.0 OR False. A silently-always-zero field is the exact defect class
# this instrument exists to remove, so every statistic over an empty sample reports None with
# its `_n` at 0, and every head-derived reading carries a `_present` / `_available` flag.
# ---------------------------------------------------------------------------

# E3 selection diagnostics averaged over P1 ticks for the CONVERSION half. All are written
# by E3SelectorTrajectory.select() into `e3.last_score_diagnostics` on every tick, so this
# is a READ of machinery that already exists (the substrate entry says so explicitly).
_E3_CONVERSION_KEYS = (
    "e3_raw_score_range_mean",
    "e3_raw_score_std_mean",
    "score_bias_abs_mean",
    "score_bias_range_mean",
    "score_bias_to_raw_range_ratio",
    "modulatory_authority_active",
    "modulatory_authority_scale_factor",
    "modulatory_authority_ratio",
    "modulatory_authority_ratio_competitive",
    "modulatory_authority_range",
)


def _dist_stats(values: List[float], prefix: str) -> Dict[str, Any]:
    """n / mean / sd (population, ddof=0) / min / max, flattened under `prefix`.

    An EMPTY sample reports n=0 and None for every moment -- never 0.0, which would be
    indistinguishable from a real measured zero.
    """
    n = len(values)
    if n == 0:
        return {
            f"{prefix}_n": 0, f"{prefix}_mean": None, f"{prefix}_sd": None,
            f"{prefix}_min": None, f"{prefix}_max": None,
        }
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {
        f"{prefix}_n": int(n),
        f"{prefix}_mean": float(mean),
        f"{prefix}_sd": float(math.sqrt(var)),
        f"{prefix}_min": float(min(values)),
        f"{prefix}_max": float(max(values)),
    }


def _new_adv_stats() -> Dict[str, float]:
    """Accumulator for the `adv_stats` keyword of the two REINFORCE helpers.

    A PASSED-IN ACCUMULATOR RATHER THAN A RETURNED TUPLE, on purpose: both helpers are
    imported and called positionally by v3_exq_724 (and re-exported under their public
    aliases), so widening the RETURN would break every existing call site, while an
    optional trailing keyword defaulting to None leaves them all bit-identical.
    """
    return {
        "n_sampled": 0,        # terms drawn from the outcome buffer
        "n_adv_surviving": 0,  # ... that cleared abs(adv) >= ADV_MIN_THRESHOLD
        "n_bias_skipped": 0,   # ... then dropped by the OFC head's own grad/shape guard
        "adv_abs_sum": 0.0,
        "adv_abs_max": 0.0,
    }


def _head_norm_snapshot(head: Optional[Any]) -> Optional[Dict[str, float]]:
    """L2 norms of a bias head: the whole head, and its LAST Linear weight/bias.

    Returns None when the head is absent, so a reader can separate "no head" from "a head
    that did not move" -- a distinction the boolean `moved > 1e-9` readout used by
    V3-EXQ-1039's C0d gate could not make, and which is why that gate passed on a head
    whose output stayed at chance.
    """
    if head is None:
        return None
    try:
        modules = list(head)
    except TypeError:
        return None
    last_linear = None
    for mod in reversed(modules):
        if isinstance(mod, torch.nn.Linear):
            last_linear = mod
            break
    with torch.no_grad():
        sq = 0.0
        for p in head.parameters():
            sq += float((p.detach().float() ** 2).sum().item())
        snap = {"head_weight_norm": float(math.sqrt(sq))}
        if last_linear is not None:
            snap["last_linear_weight_norm"] = float(
                last_linear.weight.detach().norm().item()
            )
            snap["last_linear_bias_norm"] = (
                float(last_linear.bias.detach().norm().item())
                if last_linear.bias is not None else 0.0
            )
    return snap


def _head_norm_delta(
    before: Optional[Dict[str, float]],
    after: Optional[Dict[str, float]],
    label: str,
) -> Dict[str, Any]:
    """Signed post-minus-pre deltas as FLOATS (see `_head_norm_snapshot`'s docstring)."""
    out: Dict[str, Any] = {}
    if before is None or after is None:
        out[f"{label}_present"] = False
        for k in ("head_weight_norm", "last_linear_weight_norm", "last_linear_bias_norm"):
            out[f"{label}_{k}_pre"] = None
            out[f"{label}_{k}_post"] = None
            out[f"{label}_{k}_delta"] = None
        return out
    out[f"{label}_present"] = True
    for k, post in after.items():
        pre = before.get(k)
        out[f"{label}_{k}_pre"] = float(pre) if pre is not None else None
        out[f"{label}_{k}_post"] = float(post)
        out[f"{label}_{k}_delta"] = (
            float(post - pre) if pre is not None else None
        )
    return out


def _candidate_summary_spread(
    summaries: Optional[torch.Tensor],
) -> Optional[Tuple[float, float]]:
    """(pre, post) mean row-L2 norms of the candidate summaries, around cross-candidate
    centering. None when there is nothing to measure (< 2 candidates, or non-finite).

    COMPUTED HERE ON PURPOSE -- never read back from
    `lateral_pfc.get_state()["candidate_summary_degenerate"]`. That field is assigned only
    inside `if self.config.rule_readout_consumer and k >= 2:` (lateral_pfc_analog.py), and
    `rule_readout_consumer` defaults False (LateralPFCConfig, and REEConfig's
    `lateral_pfc_rule_readout_consumer`) and is NOT set by the all-ON recipe. On this path
    that field is therefore an __init__ constant False for every run, so a gate reading it
    is VACUOUS rather than negative -- the same silently-always-false defect class this
    instrument exists to remove. The formula below is the guard's own, applied to the same
    tensor the head consumes, so the two agree whenever the consumer IS enabled.
    """
    if summaries is None or summaries.dim() != 2 or summaries.shape[0] < 2:
        return None
    with torch.no_grad():
        centered = summaries - summaries.mean(dim=0, keepdim=True)
        pre = float(summaries.norm(dim=-1).mean().item())
        post = float(centered.norm(dim=-1).mean().item())
    if not (math.isfinite(pre) and math.isfinite(post)):
        return None
    return pre, post


def _distinct_first_action_classes(candidates) -> Optional[int]:
    """How many DISTINCT first-action classes the candidate set offered this tick.

    A selection layer cannot convert a bias into different committed behaviour when every
    candidate commits the same first action, however large the bias is. Counted with the
    same accessor pattern the P1 credit-assignment block already uses.
    """
    classes = set()
    for c in candidates:
        acts = getattr(c, "actions", None)
        if acts is None or acts.shape[1] < 1:
            continue
        classes.add(int(acts[:, 0, :].argmax(-1).reshape(-1)[0].item()))
    return len(classes) if classes else None


# ---------------------------------------------------------------------------
# P1 two-head REINFORCE (mirror V3-EXQ-719a; no-ops for minimal, heads absent)
# ---------------------------------------------------------------------------
def _lpfc_reinforce_loss(
    agent: REEAgent,
    outcome_buf: List[Tuple[torch.Tensor, int, float]],
    baseline: float,
    device,
    adv_stats: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """`adv_stats` (OPTIONAL, default None) is the absorption accumulator described on
    `_new_adv_stats`. It is written to and never read, so passing it cannot change the
    returned loss, the parameters, or the RNG stream: the `np.random.choice` draw below is
    identical in count and in order whether it is supplied or not. Every pre-existing
    (positional, 4-argument) call site keeps working unchanged.
    """
    if getattr(agent, "lateral_pfc", None) is None or len(outcome_buf) < 2:
        return torch.zeros(1, device=device)
    n = len(outcome_buf)
    idxs = np.random.choice(n, size=min(REINFORCE_BATCH_SIZE, n), replace=False)
    terms: List[torch.Tensor] = []
    for i in idxs:
        cand_features, sel_idx, ep_return = outcome_buf[int(i)]
        adv = ep_return - baseline
        if adv_stats is not None:
            adv_stats["n_sampled"] += 1
            adv_stats["adv_abs_sum"] += abs(adv)
            adv_stats["adv_abs_max"] = max(adv_stats["adv_abs_max"], abs(adv))
        if abs(adv) < ADV_MIN_THRESHOLD:
            continue
        if adv_stats is not None:
            adv_stats["n_adv_surviving"] += 1
        bias = agent.lateral_pfc.compute_bias(cand_features.to(device))
        log_p = torch.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
        terms.append(-adv * log_p[min(sel_idx, bias.shape[0] - 1)])
    if not terms:
        return torch.zeros(1, device=device)
    return torch.stack(terms).mean()


def _ofc_deval_reinforce_loss(
    agent: REEAgent,
    outcome_buf: List[Tuple[torch.Tensor, int, float]],
    baseline: float,
    device,
    adv_stats: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """`adv_stats`: see `_lpfc_reinforce_loss`. Same write-only contract, same RNG
    neutrality. This head has a SECOND skip route (its own grad/shape guard) counted
    separately as `n_bias_skipped`, so a zero gradient here can be attributed to the
    advantage threshold or to the head rather than left ambiguous.
    """
    ofc = getattr(agent, "ofc", None)
    if ofc is None or len(outcome_buf) < 2:
        return torch.zeros(1, device=device)
    n = len(outcome_buf)
    idxs = np.random.choice(n, size=min(REINFORCE_BATCH_SIZE, n), replace=False)
    terms: List[torch.Tensor] = []
    for i in idxs:
        cand_features, sel_idx, ep_return = outcome_buf[int(i)]
        adv = ep_return - baseline
        if adv_stats is not None:
            adv_stats["n_sampled"] += 1
            adv_stats["adv_abs_sum"] += abs(adv)
            adv_stats["adv_abs_max"] = max(adv_stats["adv_abs_max"], abs(adv))
        if abs(adv) < ADV_MIN_THRESHOLD:
            continue
        if adv_stats is not None:
            adv_stats["n_adv_surviving"] += 1
        bias = ofc.compute_devaluation_bias(cand_features.to(device))
        if not bias.requires_grad or bias.shape[0] < 2:
            if adv_stats is not None:
                adv_stats["n_bias_skipped"] += 1
            continue
        log_p = torch.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
        terms.append(-adv * log_p[min(sel_idx, bias.shape[0] - 1)])
    if not terms:
        return torch.zeros(1, device=device)
    return torch.stack(terms).mean()


# ---------------------------------------------------------------------------
# Train the all-ON agent on a PROVIDED env with the 724-A0 recipe:
# P0a SD-070 z_world encoder warmup (OPT-IN) THEN P0b e2 forward-model warmup THEN P1 two-head
# REINFORCE (lateral-PFC bias + OFC devaluation).
#
# THE V3-EXQ-780 DEFECT AND ITS REMEDY. The three optimizer groups below (e2, lPFC bias, OFC
# devaluation) cover NO latent_stack parameter, so on this path split_encoder.world_encoder is
# never stepped and z_world stays a frozen random projection -- measured 0 of 61 latent_stack
# tensors changed at p0_episodes=200 on two independent drivers (V3-EXQ-737a, V3-EXQ-728).
# `zworld_p0_episodes > 0` adds the SD-070 recipe as a P0a stage AHEAD of the e2 warmup, which
# is the remedy adjudicated by V3-EXQ-783. See experiments/_lib/zworld_p0_warmup.py.
#
# ORDERING IS NOT ARBITRARY: e2 regresses on z_world, so the encoder must be trained BEFORE the
# e2 warmup. Training it afterwards would leave e2 fitted to the random projection -- the same
# defect one phase later.
#
# DEFAULT zworld_p0_episodes=0 IS EXACTLY THE PRIOR BEHAVIOUR, bit-identical: no extra tensor,
# no extra optimizer group, and no RNG draw (the warmup is RNG-neutral by construction and runs
# on its own env instance). Every existing caller is unaffected until it opts in.
#
# NOTE 737 and 742 import this function; the guard lives at each driver's own call site, not
# here, so each can set its own strictness.
# SD-056 e2 encoder FROZEN through P1. Mirrors V3-EXQ-728._train_all_on_agent, but the env is
# passed in (difficulty-parameterised). No P2 phase -- competence is measured downstream by the
# capability yardstick's REEForwardPolicy eval.
# ---------------------------------------------------------------------------
def _train_all_on_agent(
    agent,
    train_env: CausalGridWorldV2,
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
    rung_id: str,
    total_denominator: int,
    zworld_p0_episodes: int = 0,
    zworld_p0_env: Optional[CausalGridWorldV2] = None,
    zworld_p0_dry_run: bool = False,
    zworld_p0_resource_field_weight: float = 0.0,
    zworld_p0_config: Optional[ZWorldP0Config] = None,
    zharm_a_p0_episodes: int = 0,
    zharm_a_p0_env: Optional[CausalGridWorldV2] = None,
    zharm_a_p0_dry_run: bool = False,
    zharm_a_p0_config: Optional[ZHarmAP0Config] = None,
) -> Dict[str, Any]:
    env = train_env

    # -- P0a: SD-070 z_world encoder warmup (opt-in; see the header note) -------------------
    zworld_p0_stats: Dict[str, Any] = {"p0a_recipe": "sd070", "p0a_ran": False}
    if zworld_p0_episodes > 0:
        if zworld_p0_env is None:
            raise ValueError(
                "zworld_p0_episodes=%d requires zworld_p0_env: the warmup rollout consumes "
                "env RNG, so reusing train_env would shift the layout sequence P0b/P1 then "
                "see. Build a dedicated env with the same seed and kwargs."
                % (zworld_p0_episodes,)
            )
        # zworld_p0_resource_field_weight: SD-018 AMEND directional-field leg weight. Default
        # 0.0 = leg OFF = bit-identical to every pre-2026-09-02 caller. Needs
        # use_resource_field_head=True on the agent config as well; run_zworld_p0's
        # p0a_used_resource_field_head reports whether the leg actually ran.
        zworld_p0_stats = run_zworld_p0(
            agent, zworld_p0_env, seed, zworld_p0_episodes, steps_per_episode,
            policy=RandomPolicy(seed), label=f"ree_allon rung={rung_id}",
            dry_run=zworld_p0_dry_run,
            # SD-106: `config=` lets a caller supply its own P0 objective weights (e.g.
            # preservation_weight). `run_zworld_p0` REFUSES a config together with a non-zero
            # resource_field_weight kwarg -- the config carries its own -- so pass the kwarg
            # only on the legacy path. Default None = legacy construction = bit-identical to
            # every pre-SD-106 caller.
            **({"config": zworld_p0_config} if zworld_p0_config is not None
               else {"resource_field_weight": float(zworld_p0_resource_field_weight)}),
        )
    # -- P0h: SD-011 affective harm encoder warmup (opt-in; see the header note) -----------
    # Same defect as P0a above, on the OTHER encoder: the three optimizer groups built below
    # cover none of the 4 AffectiveHarmEncoder tensors either, so z_harm_a is a frozen random
    # projection on this path too (measured 2026-09-18; see experiments/_lib/zharm_a_p0_warmup.py
    # and sd086_zharma_readout_precondition_staged_20260918.md sec 4b). DEFAULT 0 = bit-identical
    # prior behaviour: no optimizer, no tensor, no RNG draw.
    #
    # ORDERING: after P0a, before the P0b e2 warmup. z_harm_a feeds E3 commit gating on every
    # tick of P0b and P1, so training it later would leave every selection decision in those
    # phases taken against the random projection -- the same defect one phase later.
    zharm_a_p0_stats: Dict[str, Any] = {"p0h_recipe": "sd011_harm_accum", "p0h_ran": False}
    if zharm_a_p0_episodes > 0:
        if zharm_a_p0_env is None:
            raise ValueError(
                "zharm_a_p0_episodes=%d requires zharm_a_p0_env: the warmup rollout consumes "
                "env RNG, so reusing train_env would shift the layout sequence P0b/P1 then "
                "see. Build a dedicated env with the same seed, kwargs AND harm_history_len."
                % (zharm_a_p0_episodes,)
            )
        zharm_a_p0_stats = run_zharm_a_p0(
            agent, zharm_a_p0_env, seed, zharm_a_p0_episodes, steps_per_episode,
            policy=RandomPolicy(seed), label=f"ree_allon rung={rung_id}",
            dry_run=zharm_a_p0_dry_run,
            config=zharm_a_p0_config,
        )

    has_ofc = getattr(agent, "ofc", None) is not None
    has_lpfc = getattr(agent, "lateral_pfc", None) is not None

    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    bias_opt = (
        torch.optim.Adam(list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS)
        if has_lpfc else None
    )
    ofc_deval_opt = (
        torch.optim.Adam(list(agent.ofc.devaluation_bias_head_parameters()), lr=LR_OFC_DEVAL)
        if has_ofc else None
    )
    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    total_train_eps = p0_episodes + p1_episodes
    p1_start = p0_episodes
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_e2_train_steps = 0

    reinforce_baseline = 0.0
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

    # -- absorption/conversion telemetry state (read-only; see the section header above) --
    _lpfc = getattr(agent, "lateral_pfc", None)
    _lpfc_cfg = getattr(_lpfc, "config", None)
    lpfc_head = getattr(_lpfc, "rule_bias_head", None)
    ofc_head = getattr(getattr(agent, "ofc", None), "devaluation_bias_head", None)
    lpfc_head_pre = _head_norm_snapshot(lpfc_head)
    ofc_head_pre = _head_norm_snapshot(ofc_head)
    # capture_head_diagnostics gates hidden_dead_relu_frac in lateral_pfc_analog; when it is
    # OFF the attribute sits at its __init__ 0.0, which must be reported as UNAVAILABLE, not
    # as a measured zero.
    capture_head_diag = bool(getattr(_lpfc_cfg, "capture_head_diagnostics", False))
    summary_degeneracy_floor = float(
        getattr(_lpfc_cfg, "candidate_summary_degeneracy_floor", 1e-4)
    )
    p0_ep_returns: List[float] = []
    p1_ep_returns: List[float] = []
    p1_ep_advantages: List[float] = []
    adv_stats_lpfc = _new_adv_stats()
    adv_stats_ofc = _new_adv_stats()
    summary_norm_ratios: List[float] = []
    n_summary_degenerate = 0
    distinct_action_classes: List[float] = []
    dead_relu_fracs: List[float] = []
    e3_diag_sums: Dict[str, float] = {}
    e3_diag_counts: Dict[str, int] = {}
    e3_normalize_basis: Optional[str] = None
    n_p1_diag_ticks = 0

    for ep in range(total_train_eps):
        is_p1 = (ep >= p1_start)
        is_p0 = not is_p1
        phase_label = "P1" if is_p1 else "P0"

        _flat, obs_dict = env.reset()
        agent.reset()

        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        tick_in_ep = 0

        ep_reward = 0.0
        # Separate from ep_reward ON PURPOSE: ep_reward accumulates only under `is_p1`
        # because it is the REINFORCE return, and touching it would change the recipe.
        # ep_harm_sum accumulates in BOTH phases so a P0-only manipulation is also visible.
        ep_harm_sum = 0.0
        ep_buf: List[Tuple[torch.Tensor, int]] = []

        for _step in range(steps_per_episode):
            body = obs_dict["body_state"].float()
            world = obs_dict["world_state"].float()
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)

            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=_obs_harm(obs_dict),
                obs_harm_a=_obs_harm_a(obs_dict),
                obs_harm_history=_obs_harm_history(obs_dict),
            )

            if pending_capture is not None:
                z0_prev, a_prev = pending_capture
                z1_obs = latent.z_world.detach().reshape(-1).clone()
                if (
                    torch.isfinite(z0_prev).all()
                    and torch.isfinite(a_prev).all()
                    and torch.isfinite(z1_obs).all()
                ):
                    transition_buffer.append((z0_prev, a_prev, z1_obs))
                pending_capture = None

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            p1_snap_summaries: Optional[torch.Tensor] = None
            if is_p1 and has_lpfc and candidates and len(candidates) >= 2:
                cs = _consumed_summaries(agent, candidates)
                if cs is not None and torch.isfinite(cs).all():
                    p1_snap_summaries = cs.clone()
                    _spread = _candidate_summary_spread(cs)
                    if _spread is not None:
                        _pre_n, _post_n = _spread
                        summary_norm_ratios.append(
                            float(_post_n / _pre_n) if _pre_n > 0.0 else 0.0
                        )
                        if _post_n <= summary_degeneracy_floor * _pre_n:
                            n_summary_degenerate += 1

            action = agent.select_action(candidates, ticks)
            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                break

            committed_class = int(action[0].argmax().item())

            if is_p1 and p1_snap_summaries is not None:
                sel = 0
                for ci, c in enumerate(candidates):
                    if (
                        getattr(c, "actions", None) is not None
                        and c.actions.shape[1] >= 1
                        and int(c.actions[:, 0, :].argmax(-1).reshape(-1)[0].item())
                        == committed_class
                    ):
                        sel = min(ci, p1_snap_summaries.shape[0] - 1)
                        break
                ep_buf.append((p1_snap_summaries, sel))

            if is_p1:
                # CONVERSION half: what the selection layer had to work with this tick.
                # All of it is already written by E3.select(); this only records it.
                _diag = getattr(getattr(agent, "e3", None), "last_score_diagnostics", None)
                if isinstance(_diag, dict) and _diag:
                    n_p1_diag_ticks += 1
                    for _key in _E3_CONVERSION_KEYS:
                        _val = _diag.get(_key)
                        if isinstance(_val, bool):
                            _val = 1.0 if _val else 0.0
                        elif not isinstance(_val, (int, float)):
                            continue
                        _val = float(_val)
                        if not math.isfinite(_val):
                            continue
                        e3_diag_sums[_key] = e3_diag_sums.get(_key, 0.0) + _val
                        e3_diag_counts[_key] = e3_diag_counts.get(_key, 0) + 1
                    _basis = _diag.get("modulatory_authority_normalize_basis")
                    if isinstance(_basis, str):
                        e3_normalize_basis = _basis
                if candidates:
                    _ndc = _distinct_first_action_classes(candidates)
                    if _ndc is not None:
                        distinct_action_classes.append(float(_ndc))
                if capture_head_diag and _lpfc is not None:
                    _frac = getattr(_lpfc, "_last_hidden_dead_relu_frac", None)
                    if isinstance(_frac, (int, float)) and math.isfinite(float(_frac)):
                        dead_relu_fracs.append(float(_frac))

            if is_p1:
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            # SD-056 e2 training -- P0 always; P1 only if the recipe unfreezes (A0: frozen).
            train_e2_now = is_p0 or (is_p1 and E2_TRAIN_IN_P1)
            if train_e2_now and (tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0):
                if _e2_contrastive_step(agent, transition_buffer, e2_opt, sample_rng) is not None:
                    n_e2_train_steps += 1

            _flat, _harm_signal, done, info, obs_dict = env.step(action)
            harm_signal = float(_harm_signal)
            ep_harm_sum += harm_signal
            if is_p1:
                ep_reward += harm_signal

            with torch.no_grad():
                agent.update_residue(
                    harm_signal=harm_signal, world_delta=None,
                    hypothesis_tag=False, owned=True,
                )
            if agent.goal_state is not None:
                benefit_exposure = float(info.get("benefit_exposure", 0.0))
                energy = float(body[0, 3].item())
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure,
                    drive_level=max(0.0, 1.0 - energy),
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            tick_in_ep += 1
            if done:
                break

        if is_p1:
            p1_ep_returns.append(float(ep_harm_sum))
        else:
            p0_ep_returns.append(float(ep_harm_sum))

        # P1 end-of-episode: TWO-head REINFORCE (lateral-PFC bias + OFC devaluation).
        if is_p1 and (has_lpfc or has_ofc):
            reinforce_baseline = (
                EMA_DECAY * reinforce_baseline + (1.0 - EMA_DECAY) * ep_reward
            )
            # The advantage THIS episode's own outcomes carry into the buffer, against the
            # baseline the losses below are actually handed.
            p1_ep_advantages.append(float(ep_reward - reinforce_baseline))
            for cand_features, sel in ep_buf:
                outcome_buf.append((cand_features, sel, ep_reward))
            if len(outcome_buf) > OUTCOME_BUF_MAX:
                outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]
            if has_lpfc and bias_opt is not None:
                l_loss = _lpfc_reinforce_loss(
                    agent, outcome_buf, reinforce_baseline, agent.device,
                    adv_stats=adv_stats_lpfc,
                )
                if l_loss.requires_grad:
                    bias_opt.zero_grad()
                    l_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.lateral_pfc.bias_head_parameters(), 1.0)
                    bias_opt.step()
            if has_ofc and ofc_deval_opt is not None:
                ofc_loss = _ofc_deval_reinforce_loss(
                    agent, outcome_buf, reinforce_baseline, agent.device,
                    adv_stats=adv_stats_ofc,
                )
                if ofc_loss.requires_grad:
                    ofc_deval_opt.zero_grad()
                    ofc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.ofc.devaluation_bias_head_parameters(), 1.0)
                    ofc_deval_opt.step()

        cur = ep + 1
        if cur % 50 == 0 or cur == total_train_eps or phase_label == "P1":
            print(
                f"  [train] ree_allon rung={rung_id} seed={seed} phase={phase_label} "
                f"ep {cur}/{total_denominator}",
                flush=True,
            )

    # -- assemble the telemetry (ADDITIVE keys only; the four above are untouched) --------
    absorption: Dict[str, Any] = {}
    absorption.update(_dist_stats(p0_ep_returns, "p0_ep_return"))
    absorption.update(_dist_stats(p1_ep_returns, "p1_ep_return"))
    absorption.update(_dist_stats(p1_ep_advantages, "p1_ep_advantage"))
    absorption["adv_min_threshold"] = float(ADV_MIN_THRESHOLD)
    for _label, _st in (("lpfc", adv_stats_lpfc), ("ofc", adv_stats_ofc)):
        _n = int(_st["n_sampled"])
        absorption[f"{_label}_adv_n_sampled"] = _n
        absorption[f"{_label}_adv_n_surviving"] = int(_st["n_adv_surviving"])
        # THE KEY ABSORPTION NUMBER. ~0 here means the run received no gradient at all,
        # whatever its reward was; None means no REINFORCE term was ever drawn.
        absorption[f"{_label}_adv_surviving_frac"] = (
            float(_st["n_adv_surviving"]) / _n if _n > 0 else None
        )
        absorption[f"{_label}_adv_abs_mean"] = (
            float(_st["adv_abs_sum"]) / _n if _n > 0 else None
        )
        absorption[f"{_label}_adv_abs_max"] = float(_st["adv_abs_max"]) if _n > 0 else None
    absorption["ofc_adv_n_bias_skipped"] = int(adv_stats_ofc["n_bias_skipped"])
    absorption.update(
        _head_norm_delta(lpfc_head_pre, _head_norm_snapshot(lpfc_head), "lpfc_bias_head")
    )
    absorption.update(
        _head_norm_delta(ofc_head_pre, _head_norm_snapshot(ofc_head), "ofc_deval_head")
    )
    # UNAVAILABLE is not zero: with capture_head_diagnostics OFF the underlying attribute is
    # an __init__ constant, so nothing is recorded and every moment stays None.
    absorption["hidden_dead_relu_frac_available"] = bool(
        capture_head_diag and _lpfc is not None
    )
    absorption.update(_dist_stats(dead_relu_fracs, "hidden_dead_relu_frac"))

    conversion: Dict[str, Any] = {
        "n_p1_diag_ticks": int(n_p1_diag_ticks),
        "modulatory_authority_normalize_basis": e3_normalize_basis,
    }
    for _key in _E3_CONVERSION_KEYS:
        _cnt = int(e3_diag_counts.get(_key, 0))
        conversion[f"{_key}_n"] = _cnt
        conversion[f"{_key}_mean"] = (
            float(e3_diag_sums[_key]) / _cnt if _cnt > 0 else None
        )
    conversion.update(
        _dist_stats(distinct_action_classes, "distinct_first_action_classes")
    )
    conversion.update(
        _dist_stats(summary_norm_ratios, "candidate_summary_post_pre_norm_ratio")
    )
    _n_sr = len(summary_norm_ratios)
    conversion["candidate_summary_degenerate_frac"] = (
        float(n_summary_degenerate) / _n_sr if _n_sr > 0 else None
    )
    conversion["candidate_summary_degeneracy_floor"] = summary_degeneracy_floor
    # Flag for a downstream gate: this verdict was COMPUTED from the summaries the head
    # consumed, NOT read from lateral_pfc's own field (which is vacuous on this recipe --
    # see `_candidate_summary_spread`).
    conversion["candidate_summary_degenerate_computed"] = True

    return {
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_e2_train_steps": int(n_e2_train_steps),
        "zworld_p0": zworld_p0_stats,
        "zharm_a_p0": zharm_a_p0_stats,
        # sd-allon-training-signal-absorption-telemetry: ADDITIVE, never read back by this
        # recipe. A drive-axis null measured through this harness is interpretable only
        # against these two blocks.
        "absorption": absorption,
        "conversion": conversion,
    }

# Public aliases. The underscore names above are kept verbatim from the driver files so that
# every existing `x724._helper` / `x734._train_all_on_agent` attribute path keeps resolving
# through the re-export; these are the names new code should use.
train_all_on_agent = _train_all_on_agent
e2_contrastive_step = _e2_contrastive_step
lpfc_reinforce_loss = _lpfc_reinforce_loss
ofc_deval_reinforce_loss = _ofc_deval_reinforce_loss
consumed_summaries = _consumed_summaries
obs_harm = _obs_harm
obs_harm_a = _obs_harm_a
obs_harm_history = _obs_harm_history

__all__ = [
    "train_all_on_agent", "_train_all_on_agent",
    "e2_contrastive_step", "_e2_contrastive_step",
    "_sample_class_diverse_batch",
    "lpfc_reinforce_loss", "_lpfc_reinforce_loss",
    "ofc_deval_reinforce_loss", "_ofc_deval_reinforce_loss",
    "consumed_summaries", "_consumed_summaries",
    "obs_harm", "_obs_harm", "obs_harm_a", "_obs_harm_a",
    "obs_harm_history", "_obs_harm_history",
    "LR_OFC_DEVAL", "SD056_WEIGHT", "E2_CONTRASTIVE_LR", "E2_TRAIN_EVERY_K_TICKS",
    "CONTRASTIVE_BATCH_K", "TRANSITION_BUFFER_MAX", "MIN_BUFFER_BEFORE_TRAIN",
    "MIN_CLASSES_FOR_TRAIN", "MAX_GRAD_NORM", "LR_LPFC_BIAS", "REINFORCE_BATCH_SIZE",
    "OUTCOME_BUF_MAX", "POLICY_TEMPERATURE", "ADV_MIN_THRESHOLD", "EMA_DECAY",
    "E2_TRAIN_IN_P1",
    # sd-allon-training-signal-absorption-telemetry
    "_dist_stats", "_new_adv_stats", "_head_norm_snapshot", "_head_norm_delta",
    "_candidate_summary_spread", "_distinct_first_action_classes",
    "_E3_CONVERSION_KEYS",
]
