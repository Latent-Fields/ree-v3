"""SD-074: trained-enough agent substrate for read-only control-plane telemetry probes.

WHY THIS EXISTS
---------------
The 2x2 read-only telemetry-probe family (V3-EXQ-777 / 777a, MECH-063 sub-claim (i))
measures gain/bias regulators -- MECH-320 tonic_vigor score-bias and MECH-313
noise_floor temperature -- on the E3 pre-commit softmax while running an UNTRAINED
agent. A regulator that MODULATES a distribution is unobservable when that
distribution has no dynamic range.

V3-EXQ-777a quantified this. Its dependent variable D_action_mass (the pre-commit
softmax mass on non-noop candidates) was pinned at ceiling on 7 of 14 seeds and at
floor on 2 more -- 64% degenerate -- and

    corr(distance of D_action_mass_mean from saturation, norm_v_score) = 0.884

i.e. the score axis's measurable authority is very largely DETERMINED by how far the
action-value mass sits from its 0/1 bounds. That capped informative-seed yield at
4 of 14 (28.6%) while the load-bearing criterion needs ~51 informative seeds --
~177 raw seeds, ~31 h. Infeasible.

Critically this is NOT a sampling defect. V3-EXQ-777a's sample-driven stopping worked
perfectly (all 56 cells reached 250 fresh E3 selections, zero starved cells) and the
saturation rate barely moved from its starved predecessor. Saturation is a property
of an untrained agent's degenerate action-value distribution: either one candidate
dominates (D -> 1) or the scores are flat (D -> 0). More samples cannot repair it.

So the fix is upstream of measurement: bring the agent to a non-degenerate
action-value landscape BEFORE telemetry collection begins.

    Adjudicated by failure_autopsy_MECH-063-777a-779a-cluster_2026-07-18
    (REE_assembly/evidence/planning/), which fired the re-derive brake on the 777
    lineage and routed here. A further lettered iteration of that probe against an
    untrained agent is REFUSED until this substrate exists.

WHAT THIS IS AND IS NOT
-----------------------
This is TRAINING-REGIME SUBSTRATE ENRICHMENT FOR THE PROBE HARNESS -- the same class
as the V3-EXQ-603c precedent. It is NOT a ree_core mechanism change: nothing under
ree_core/ is touched, no config default moves, and no existing script imports this
module. Backward compatibility is therefore total by construction.

It composes two things that already exist rather than writing a fourth training loop:

  * experiments/_lib/goal_pipeline_tier1.warmup_train -- the canonical
    StepHarness-coupled warmup (Adam over e1, e2.world_transition +
    e2.world_action_encoder, e3.harm_eval_head, latent_stack).
  * experiments/_lib/baselines/maturation_curriculum -- the atomic checkpoint-cache
    discipline (os.replace, key re-verified on load, agent rebuilt on BOTH paths so
    RNG consumption is identical hit-vs-miss).

DESIGN DECISIONS (user-confirmed 2026-07-18)
--------------------------------------------
1. RECORD, DO NOT ABORT. A seed that stays saturated after warmup is reported
   (saturated=True) with its realised D_action_mass_mean; the CONSUMER decides
   whether to drop it. The autopsy explicitly asks for the realised per-seed
   saturation distribution to be recorded so informative yield is AUDITABLE rather
   than inferred -- aborting would discard exactly that data. `assert_any_informative`
   is provided for the one case worth failing loudly on: the warmup provably did
   nothing (zero seeds de-saturated).
2. TARGET ENV ONLY, BUDGET-SWEPT. Warmup runs on the same env the probe measures in,
   with the episode budget an explicit swept parameter. No 603c-style easy-env
   curriculum by default: it would add a second env as a confound before the
   de-saturation question is even answered.

THREE HAZARDS THIS MODULE DEFENDS AGAINST
-----------------------------------------
(H1) THE NON-state_dict ATTRIBUTE SURFACE IS NOT IN state_dict, AND IT IS NOT JUST
     E3. `nn.Module.state_dict()` carries registered Parameters and buffers only;
     every plain-Python attribute hanging off the agent's 121-module tree (465 of
     them at construction, 587 once warmed) is silently dropped. The load-bearing
     ones are agent-level EXPERIENCE BUFFERS, not E3 scalars --
     `_self_experience_buffer`, `_world_experience_buffer`, `_e2_transition_buffer`,
     which `agent.compute_prediction_loss()` (ree_core/agent.py:9970) samples a
     RANDOM WINDOW from on every warmup_train tick. A read that grows them puts
     eval-rollout observations into the E1 TRAINING POOL. They cap at 1000
     (agent.py:5319) and V3-EXQ-784's realised reads are 1081-1401 env steps, so one
     read fully DISPLACES the pool. e3._running_variance (e3_selector.py:291, feeds
     commit_variance at :2703) is a real member of this surface but a small one.

     This module therefore captures and restores the WHOLE surface mechanically --
     see capture_agent_surface() -- rather than a hand-listed set of names. The
     hand-listed version (`_E3_NONBUFFER_STATE`, removed 2026-08-15) is exactly how
     this defect arose: it named four E3 attributes, one of which
     (`_last_error_var`) does not exist anywhere in ree_core, while the channel that
     actually moved the dependent variable was not in E3 at all.

     Measured (probe_warmup_nonbuffer_audit_2026-08-15.md, governance flag
     GFLAG-0036): under the hand-listed restore a single 40-step read left 21
     attributes changed, 17 of them surviving agent.reset(); replicating V3-EXQ-784's
     ladder at its real scale, restoring the full surface changed the dependent
     variable in 6 of 6 post-first-read cells and FLIPPED the saturation regime in
     2 of 6. V3-EXQ-784's informative_yield is provisional as a result.

(H2) PER-SEED CHECKPOINT SHARING ACROSS ARMS. The 2x2's arms differ only in
     use_tonic_vigor / use_noise_floor, and both regulators are documented
     "no learned parameters, no nn.Module inheritance" (policy/tonic_vigor.py:225,
     policy/noise_floor.py:129), so they contribute ZERO state_dict keys. One warmed
     checkpoint per seed therefore loads cleanly into all four arms, which is what
     makes the 2x2 clean: arms differ ONLY in regulator scalars at the e3.select()
     call site. We ASSERT this (assert_state_dict_shareable) rather than assume it,
     so a future arm that flips a flag which DOES construct a module fails loudly.

(H3) THE CONSUMER'S generate_trajectories MONKEYPATCH IS INSTANCE-LEVEL and is not
     part of state_dict. A probe that captures candidates that way (777a:484-491)
     MUST re-apply the patch AFTER load_state_dict. If it is lost, every observe()
     returns None, the cell yields zero samples, and the run self-routes to
     "sample_starvation_requeue" -- i.e. a lost patch masquerades as a sampling bug.
     See reapply_candidate_capture().

MECH-094: N/A. Warmup is waking-only gradient training. It runs no simulation, no
replay, and writes nothing to memory, so the hypothesis_tag requirement does not
arise.

NOT REPAIRED HERE (consumer-side, deliberately out of scope): V3-EXQ-777a's c1_robust
bar is written `(mean_sin - pstdev_sin) > MARGIN` (script:697, pstdev at :541). A
POPULATION dispersion does not shrink with n, so that bar is unreachable at ANY sample
size and conflates seed-to-seed dispersion with measurement noise. This module cannot
fix a criterion it does not own -- the successor experiment does.

    The successor author no longer has to hand-roll that re-expression. As of
    ree-v3 de09887093, `experiments/_lib/robustness_bars.py` provides it:

      * robust_by_sem(vals, margin, k=1.0, min_n=3) -- the SEM-denominated
        replacement, `mean - k*SEM > margin`. Use this wherever the intent is
        "the effect exceeds its own MEASUREMENT NOISE". Returns
        `sample_size_improvable: True`; propagate that into the manifest.
      * exceeds_cross_seed_dispersion(vals, margin=0.0, min_n=3) -- the
        dispersion bar KEPT but explicitly NAMED, for when the claim really is
        "the typical seed shows the effect". Returns
        `sample_size_improvable: False`, so a failure can never be misread as
        "add seeds".
      * seeds_required_for_sem_bar(mean, pstdev_est, margin, k,
        informative_yield) -- the DESIGN-TIME cost check. Run it before queueing.

    Which one to reach for depends on MARGIN, and the choice is not free:
    MARGIN > 0 compounds a non-shrinking denominator with a positive threshold
    into unreachability -- that is the 777a defect, and there `robust_by_sem` is
    the repair. At MARGIN == 0 the dispersion form is merely a CONSERVATIVE bar,
    STRICTER than the SEM form, so a run that already PASSED it must NOT be
    re-denominated: doing so would loosen a criterion retroactively.

Note also the autopsy's finding that repairing the bar is NECESSARY BUT NOT
SUFFICIENT: the binding constraint was always the informative-seed yield, which is
what `seeds_required_for_sem_bar`'s `informative_yield` argument exists to price.

ASCII-only output (CLAUDE.md).
"""

from __future__ import annotations

import collections
import copy
import math
import os
import pickle
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import torch

# Path shim -- the _lib idiom (see baselines/maturation_curriculum.py:117-122).
# Required, not cosmetic: goal_pipeline_tier1 imports its harness FLAT
# (`from _harness import StepHarness`), so experiments/ must be on sys.path before
# it is imported at all. Without this, importing this module raises
# ModuleNotFoundError: _harness for any caller that only added ree-v3/ to the path.
_LIB_DIR = Path(__file__).resolve().parent          # experiments/_lib
_EXP_DIR = _LIB_DIR.parent                          # experiments
_REPO_ROOT = _EXP_DIR.parent                        # ree-v3
for _p in (str(_REPO_ROOT), str(_EXP_DIR), str(_LIB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _harness import StepHarness  # noqa: E402
from arm_fingerprint import compute_substrate_hash, machine_class, resolve_substrate_identity  # noqa: E402
from sample_driven_rollout import (  # noqa: E402
    RolloutBudget,
    TickContext,
    run_cell_until_samples,
)
from _lib.goal_pipeline_tier1 import warmup_train  # noqa: E402

__all__ = [
    "D_SAT_LOW",
    "D_SAT_HIGH",
    "PROBE_WARMUP_SCHEMA",
    "WarmupRecipe",
    "WarmupOutcome",
    "AgentSurface",
    "warm_agent",
    "measure_action_mass",
    "capture_agent_surface",
    "restore_agent_surface",
    "saturation_regime",
    "saturation_summary",
    "assert_state_dict_shareable",
    "assert_any_informative",
    "reapply_candidate_capture",
    "NoInformativeSeeds",
]

# v2 (2026-08-15): the warm-start blob carries the WHOLE non-state_dict attribute
# surface, not the four hand-listed E3 scalars of v1. The schema string is part of
# the cache key, so every v1 blob auto-MISSes rather than restoring a partial agent.
#
# v3 (2026-08-30, IGW-20260830-222 / SD-PROBE-WARMUP): fixes the V3-EXQ-963
# cache-key + restore-clobber defect (see _warmup_key and _restore_cached_surface
# below). Bumped so every v2 blob -- possibly minted under a DIFFERENT arm's
# regulator config than the one now requesting it -- auto-MISSes rather than being
# restored onto a mismatched live agent.
PROBE_WARMUP_SCHEMA = "probe_warmup.v3"

# Saturation bounds. Deliberately IDENTICAL to V3-EXQ-777a's D_SAT_LOW / D_SAT_HIGH
# (script:246-247) so this substrate's success criterion is expressed in the SAME
# units as the failure record that motivated it. Do not retune these to make a
# warmup look better -- that would silently move the goalposts the autopsy set.
D_SAT_LOW = 0.05
D_SAT_HIGH = 0.95

# CausalGridWorldV2 ACTIONS[4] = (0,0) = stay/no-op. Mirrors 777a:226.
NOOP_CLASS = 4

# nn.Module's OWN bookkeeping keys, derived from a live probe instance rather than
# hardcoded so a torch upgrade that adds one is excluded automatically. `training`
# is deliberately ADDED BACK to the captured surface: agent.eval() flips it on all
# 121 submodules, and the agent-level `agent.train()` that used to undo that would
# wrongly wake a submodule the caller had deliberately left in eval mode.
_MODULE_INTERNALS: FrozenSet[str] = frozenset(vars(torch.nn.Module()).keys()) - {"training"}

# Leaves that are STATE-FREE at attribute level and must be carried by REFERENCE.
# The live case is hippocampal._rng, which defaults to the stdlib `random` MODULE
# itself (ree_core/hippocampal/module.py:186) -- copying a module object is both
# impossible (TypeError: cannot pickle 'module' object) and wrong, since its state
# is process-global. See the GLOBAL RNG note on measure_action_mass.
_BY_REFERENCE_TYPES: Tuple[type, ...] = (
    types.ModuleType,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    type,
)

_ATOMIC_TYPES: Tuple[type, ...] = (bool, int, float, complex, str, bytes)


# ---------------------------------------------------------------------------
# Non-state_dict attribute surface: mechanical capture / restore (hazard H1).
# ---------------------------------------------------------------------------

@dataclass
class AgentSurface:
    """One snapshot of everything on the agent that state_dict() does not carry.

    `entries` is (module, module_path, {attr_name: copied_value}, frozenset(names))
    per module in the agent's tree. The name set is kept SEPARATELY from the value
    dict so restore can also DELETE attributes the read ADDED -- which is not a
    hypothetical: measure_action_mass installs the hazard-H3 generate_trajectories
    capture as an instance attribute, and without the delete each read would wrap
    the previous read's wrapper, growing call depth by one per read forever.

    `by_reference` names attributes that could not be copied and are therefore NOT
    protected. It is expected to be EMPTY under every config that uses this module;
    the contract pins that, so a new uncopyable attribute fails loudly and gets
    classified rather than silently leaking. Do not read a non-empty list as benign.
    """

    entries: List[Tuple[Any, str, Dict[str, Any], FrozenSet[str]]]
    by_reference: List[str]
    n_modules: int
    n_attrs: int

    def report(self) -> Dict[str, Any]:
        return {
            "n_modules": int(self.n_modules),
            "n_attrs": int(self.n_attrs),
            "n_by_reference": len(self.by_reference),
            "by_reference": list(self.by_reference),
        }


def _copy_state(value: Any, memo: Dict[int, Any], by_reference: List[str], path: str) -> Any:
    """Structural deep copy with a by-reference floor. Never raises.

    THIS IS DELIBERATELY NOT `copy.deepcopy`, and the reason is a real bug rather
    than taste. A plain deepcopy of a live REEAgent attribute fails on the four
    trajectory-shaped attributes (`_committed_candidates`,
    `_last_e3_selection_result`, e3._persistent_committed_trajectory,
    e3._last_selected_trajectory) because Trajectory.metadata can hold a module
    object. Worse, deepcopy fails *midway*: `copy._reconstruct` registers the
    half-built, EMPTY object in the shared memo BEFORE deep-copying its state, so a
    fallback that then consults that same memo hands back an empty Trajectory and the
    restore silently WIPES four behaviour-bearing attributes. Measured: 4 residual
    drifts that read as "restored" until the object was inspected.

    Two properties that are load-bearing and must survive any refactor:

      * `memo` is PRE-SEEDED with every live nn.Module in the tree, so an attribute
        holding a reference to a submodule keeps pointing at the LIVE module rather
        than at a detached clone whose parameters no optimiser owns.
      * `memo` is SHARED across the whole capture, so two attributes aliasing one
        object still alias one object after the restore.
    """
    key = id(value)
    if key in memo:
        return memo[key]
    if value is None or isinstance(value, _ATOMIC_TYPES) or isinstance(value, _BY_REFERENCE_TYPES):
        return value
    if torch.is_tensor(value):
        out = value.detach().clone()
        memo[key] = out
        return out
    if isinstance(value, torch.nn.Module):
        # Not in memo => a module hanging off an attribute but NOT registered in the
        # agent's tree, so state_dict does not carry it either. Cloning it would
        # duplicate parameters and break identity; flag it instead of guessing.
        by_reference.append("%s <unregistered nn.Module>" % path)
        return value
    if isinstance(value, list):
        out_list: List[Any] = []
        memo[key] = out_list
        out_list.extend(_copy_state(v, memo, by_reference, path) for v in value)
        return out_list
    if isinstance(value, dict):
        out_dict: Dict[Any, Any] = {}
        memo[key] = out_dict
        for k, v in value.items():
            out_dict[_copy_state(k, memo, by_reference, path)] = _copy_state(v, memo, by_reference, path)
        return out_dict
    if isinstance(value, collections.deque):
        out_deque: Any = collections.deque(maxlen=value.maxlen)
        memo[key] = out_deque
        out_deque.extend(_copy_state(v, memo, by_reference, path) for v in value)
        return out_deque
    if isinstance(value, tuple):
        out = tuple(_copy_state(v, memo, by_reference, path) for v in value)
        memo[key] = out
        return out
    if isinstance(value, (set, frozenset)):
        out = type(value)(_copy_state(v, memo, by_reference, path) for v in value)
        memo[key] = out
        return out
    d = getattr(value, "__dict__", None)
    if d is not None:
        try:
            out = value.__class__.__new__(value.__class__)
        except Exception:
            out = None
        if out is not None:
            memo[key] = out
            for k, v in d.items():
                object.__setattr__(out, k, _copy_state(v, memo, by_reference, path))
            return out
    try:
        # __slots__ objects and anything else without a __dict__. A FRESH memo, so a
        # failure here cannot poison the shared one (see the docstring).
        out = copy.deepcopy(value)
    except Exception:
        by_reference.append(path)
        return value
    memo[key] = out
    return out


def capture_agent_surface(agent: Any) -> AgentSurface:
    """Snapshot every plain-Python attribute across the agent's module tree.

    Cheap enough to run on every read: ~0.15 s for 587 attributes across 121 modules
    with 120-entry experience buffers, measured on ree-cloud-5. Pair with
    restore_agent_surface(); the snapshot is CONSUMED by the restore (its values are
    installed directly rather than re-copied) so do not restore the same snapshot twice.
    """
    memo: Dict[int, Any] = {}
    modules = list(agent.named_modules())
    for _path, module in modules:
        memo[id(module)] = module
    by_reference: List[str] = []
    entries: List[Tuple[Any, str, Dict[str, Any], FrozenSet[str]]] = []
    n_attrs = 0
    for mpath, module in modules:
        live = vars(module)
        names = [n for n in live if n not in _MODULE_INTERNALS]
        values: Dict[str, Any] = {}
        for name in names:
            path = ("%s.%s" % (mpath, name)) if mpath else name
            values[name] = _copy_state(live[name], memo, by_reference, path)
        n_attrs += len(names)
        entries.append((module, mpath, values, frozenset(names)))
    return AgentSurface(entries=entries, by_reference=by_reference,
                        n_modules=len(modules), n_attrs=n_attrs)


def restore_agent_surface(agent: Any, surface: AgentSurface) -> Dict[str, Any]:
    """Put every captured attribute back, and delete every attribute the read added.

    In-place for the mutable containers (list / dict / set / deque) when the live
    object is still the same type, so an external holder of that container -- a
    closure, another module -- sees the restored contents rather than keeping a
    reference to the drifted one. Plain `object.__setattr__` otherwise, which writes
    straight into the module `__dict__` the value came from and therefore cannot
    accidentally re-register a value as a parameter or submodule.

    `agent` is accepted for symmetry and for the tree-shape check only: the snapshot
    already holds direct module references.
    """
    n_deleted = 0
    n_inplace = 0
    n_setattr = 0
    for module, _mpath, values, names in surface.entries:
        live = vars(module)
        for extra in [n for n in list(live) if n not in _MODULE_INTERNALS and n not in names]:
            try:
                object.__delattr__(module, extra)
                n_deleted += 1
            except Exception:
                pass
        for name, value in values.items():
            current = live.get(name, None)
            if current is not None and current is not value and type(current) is type(value):
                if isinstance(current, list):
                    current[:] = value
                    n_inplace += 1
                    continue
                if isinstance(current, dict):
                    current.clear()
                    current.update(value)
                    n_inplace += 1
                    continue
                if isinstance(current, set):
                    current.clear()
                    current.update(value)
                    n_inplace += 1
                    continue
                if isinstance(current, collections.deque):
                    current.clear()
                    current.extend(value)
                    n_inplace += 1
                    continue
            object.__setattr__(module, name, value)
            n_setattr += 1
    report = surface.report()
    report.update({
        "n_restored_inplace": n_inplace,
        "n_restored_setattr": n_setattr,
        "n_deleted_added_attrs": n_deleted,
    })
    return report


def _picklable_surface(surface: AgentSurface) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Project a captured surface down to what torch.save can actually write.

    The warm-start blob goes through pickle, and the surface legitimately contains
    values pickle refuses (hippocampal._rng is the stdlib `random` module). Dropping
    those per-attribute keeps the cache honest -- a dropped attribute is NAMED in the
    blob and re-reported on load -- instead of letting one unpicklable leaf take the
    whole cache store down, which is how a cache silently stops existing.
    """
    out: Dict[str, Dict[str, Any]] = {}
    dropped: List[str] = []
    for _module, mpath, values, _names in surface.entries:
        keep: Dict[str, Any] = {}
        for name, value in values.items():
            try:
                pickle.dumps(value)
            except Exception:
                dropped.append(("%s.%s" % (mpath, name)) if mpath else name)
                continue
            keep[name] = value
        if keep:
            out[mpath] = keep
    return out, dropped


class NoInformativeSeeds(RuntimeError):
    """Raised by assert_any_informative when NO seed de-saturated.

    This is the one failure worth raising on: it means the warmup provably did not
    move the action-value landscape at all, so every downstream number is vacuous.
    Per-seed saturation on SOME seeds is a RECORDED OUTCOME, not an error.
    """


@dataclass(frozen=True)
class WarmupRecipe:
    """Declared warmup regime. Every field is part of the cache key.

    num_episodes is THE swept parameter: the question this substrate exists to answer
    is how much training de-saturation actually needs. Sweep it; do not tune it
    silently.

    PROBE BUDGET DENOMINATION (why probe_max_episodes defaults to DERIVED)
    ---------------------------------------------------------------------
    The de-saturation read is sample-driven: it wants `probe_selections` fresh E3
    selections and bounds itself with `probe_max_env_steps`. An EPISODE cap smaller
    than the STEP cap silently re-denominates that read, because every episode costs
    at least one env step -- so `probe_max_episodes < probe_max_env_steps` means a
    seed whose episodes are SHORT cannot spend its step budget at all.

    This module shipped with an independent `probe_max_episodes = 40` against
    `probe_max_env_steps = 4000`, i.e. a read that starved for any seed averaging
    under 100 steps/episode. That is not hypothetical: the V3-EXQ-779a seed-23 shape
    dies in ~7 steps/episode, which would have bought ~280 env steps against a
    120-selection floor -- a de-saturation verdict built on a starved sample, or a
    "collected ZERO selections" note that reads as a sampling bug rather than as the
    budget defect it is. Same defect class as MECH-063 autopsy followup #3
    (failure_autopsy_MECH-063-777a-779a-cluster_2026-07-18, targets[1]), for which
    the shared helper was hardened in ree-v3 6ac2a0d.

    So `probe_max_episodes = 0` means DERIVE (= probe_max_env_steps), which is the
    step-denominated form in which the episode cap can never bind first. The sentinel
    rather than a hardcoded 4000 is deliberate: a future author who retunes
    probe_max_env_steps alone gets a correct budget for free, where a duplicated
    literal would silently rot back into a tight cap.

    An explicit tight cap is still available, but it must be DECLARED: set
    probe_max_episodes AND allow_tight_episode_cap=True. Either way the realised
    binding constraint reaches the manifest via WarmupOutcome (probe_max_episodes,
    probe_max_env_steps, probe_episode_cap_can_bind), so a reader can see that a
    probe's read was episode-denominated without re-deriving it.

    Cache-key note: as_dict() emits the RESOLVED episode cap, not the sentinel, so
    the key describes the budget that actually ran.
    """

    num_episodes: int
    steps_per_episode: int = 300
    regime: str = "target_env"          # "target_env" (default) | "curriculum" (not built)
    probe_selections: int = 120         # fresh E3 selections for the de-saturation read
    probe_max_env_steps: int = 4000
    probe_max_episodes: int = 0         # 0 => DERIVE from probe_max_env_steps (see docstring)
    allow_tight_episode_cap: bool = False

    @property
    def resolved_probe_max_episodes(self) -> int:
        """The episode cap that will actually be passed to RolloutBudget."""
        explicit = int(self.probe_max_episodes)
        if explicit > 0:
            return explicit
        return int(self.probe_max_env_steps)

    @property
    def probe_episode_cap_can_bind(self) -> bool:
        """True when the EPISODE cap can bind before the STEP cap on the probe read.

        Mirrors RolloutBudget.episode_cap_can_bind exactly. False under the derived
        default, by construction.
        """
        return self.resolved_probe_max_episodes < int(self.probe_max_env_steps)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "num_episodes": int(self.num_episodes),
            "steps_per_episode": int(self.steps_per_episode),
            "regime": str(self.regime),
            "probe_selections": int(self.probe_selections),
            "probe_max_env_steps": int(self.probe_max_env_steps),
            "probe_max_episodes": int(self.resolved_probe_max_episodes),
            "probe_max_episodes_derived": bool(int(self.probe_max_episodes) <= 0),
            "probe_episode_cap_can_bind": bool(self.probe_episode_cap_can_bind),
            "probe_allow_tight_episode_cap": bool(self.allow_tight_episode_cap),
        }


@dataclass
class WarmupOutcome:
    """Realised result for ONE seed. Emit this verbatim into the manifest."""

    seed: int
    d_action_mass_mean: Optional[float]
    d_action_mass_std: Optional[float]
    regime: str                     # "ceiling" | "headroom" | "floor" | "unmeasured"
    saturated: bool
    n_probe_selections: int
    probe_stop_reason: str
    warmup_episodes: int
    cache_hit: bool
    recipe: Dict[str, Any]
    notes: List[str] = field(default_factory=list)
    # Realised probe BUDGET, not just which cap fired. `probe_stop_reason` says which
    # cap DID bind; these say which cap COULD have, and how much of the budget the
    # read actually spent. Without them a starved read is indistinguishable from a
    # cheap one that met its floors early -- that ambiguity is exactly what made the
    # V3-EXQ-779a seed-23 defect survive a manifest review.
    n_probe_env_steps: int = 0
    n_probe_episodes: int = 0
    probe_max_env_steps: int = 0
    probe_max_episodes: int = 0
    probe_episode_cap_can_bind: bool = False
    probe_floors_met: bool = True

    @property
    def informative(self) -> bool:
        return not self.saturated and self.d_action_mass_mean is not None

    def as_manifest_fields(self) -> Dict[str, Any]:
        return {
            "seed": int(self.seed),
            "d_action_mass_mean": self.d_action_mass_mean,
            "d_action_mass_std": self.d_action_mass_std,
            "saturation_regime": self.regime,
            "saturated": bool(self.saturated),
            "informative": bool(self.informative),
            "n_probe_selections": int(self.n_probe_selections),
            "probe_stop_reason": self.probe_stop_reason,
            "n_probe_env_steps": int(self.n_probe_env_steps),
            "n_probe_episodes": int(self.n_probe_episodes),
            "probe_max_env_steps": int(self.probe_max_env_steps),
            "probe_max_episodes": int(self.probe_max_episodes),
            "probe_episode_cap_can_bind": bool(self.probe_episode_cap_can_bind),
            "probe_floors_met": bool(self.probe_floors_met),
            "warmup_episodes": int(self.warmup_episodes),
            "warmup_cache_hit": bool(self.cache_hit),
            "warmup_recipe": dict(self.recipe),
            "notes": list(self.notes),
        }


def saturation_regime(d_mean: Optional[float]) -> str:
    """Classify a seed's D_action_mass_mean. Mirrors 777a:_saturation_regime."""
    if d_mean is None:
        return "unmeasured"
    if d_mean >= D_SAT_HIGH:
        return "ceiling"
    if d_mean <= D_SAT_LOW:
        return "floor"
    return "headroom"


# ---------------------------------------------------------------------------
# Candidate / action-mass measurement (read-only).
# ---------------------------------------------------------------------------

def _candidate_noop_mask(candidates: Any) -> Optional[torch.Tensor]:
    """Per-candidate boolean mask: True where the FIRST-step action class == NOOP.

    Byte-for-byte the same rule as 777a:_candidate_noop_mask, so the D_action_mass
    this module reports is the SAME quantity the failure record is denominated in.
    """
    classes: List[int] = []
    for c in candidates:
        acts = getattr(c, "actions", None)
        if acts is None:
            return None
        a = acts.reshape(-1, acts.shape[-1])[0]
        classes.append(int(a.argmax().item()))
    if not classes:
        return None
    return torch.tensor(classes) == NOOP_CLASS


def reapply_candidate_capture(agent: Any) -> Dict[str, Any]:
    """Install the read-only generate_trajectories capture and return its dict.

    Hazard H3. StepHarness calls agent.generate_trajectories() internally, so the
    candidate list is not otherwise visible to the caller. The patch is an INSTANCE
    attribute and is therefore NOT restored by load_state_dict -- call this AFTER any
    checkpoint load, and once per arm-agent.

    Returns the mutable dict whose "cands" key holds the most recent candidate list.
    """
    captured: Dict[str, Any] = {"cands": None}
    orig = agent.generate_trajectories

    def _capture(*a: Any, **k: Any) -> Any:
        cands = orig(*a, **k)
        captured["cands"] = cands
        return cands

    agent.generate_trajectories = _capture
    return captured


def measure_action_mass(
    agent: Any,
    env: Any,
    *,
    seed: int,
    n_selections: int,
    max_env_steps: int,
    max_episodes: int,
    steps_per_episode: int,
    captured: Optional[Dict[str, Any]] = None,
    label: str = "",
    allow_tight_episode_cap: bool = False,
) -> Dict[str, Any]:
    """Read-only de-saturation probe: collect D_action_mass over fresh E3 selections.

    NO gradient work and no optimiser -- this measures the landscape the warmup
    produced, it does not train. Delegates stopping to the shared family helper so
    the read is sample-driven (the F1 fix of the 777 autopsy) rather than
    step-budgeted: a short-lived agent in a hazard-terminating env would otherwise
    return a handful of samples and a misleading mean.

    NON-DESTRUCTIVE, IN BOTH DIRECTIONS THAT MATTER. torch.no_grad() + agent.eval()
    stop GRADIENT updates, but they do not stop either of the two ways this agent
    carries state forward while merely being stepped:

      (a) THE PLAIN-PYTHON ATTRIBUTE SURFACE -- 587 attributes across 121 modules,
          none of them in state_dict. The load-bearing members are the experience
          buffers `_self_experience_buffer` / `_world_experience_buffer` /
          `_e2_transition_buffer`, which `agent.compute_prediction_loss()` samples a
          random window from on every subsequent warmup_train tick, so a read that
          grows them feeds EVAL-ROLLOUT DATA INTO THE E1 TRAINING POOL. Smaller
          members drift too (e3._running_variance 0.001839 -> 0.001855 over a
          25-selection read; it feeds commit_variance at e3_selector.py:2703).
      (b) REGISTERED BUFFERS -- the three-factor plasticity / eligibility traces
          (e3_selector.py:373-462) are updated in-place under no_grad. Measured: two
          agents identical at load diverged by max |dw| = 2.5e-01 after two
          de-saturation reads of different lengths (71 vs 50 env steps).

    Both would perturb the very distribution being measured, and would make
    `measure=True` hand back a different agent than `measure=False`. So the full
    state_dict (b) AND the whole non-state_dict attribute surface (a) are snapshotted
    and restored -- the caller gets back bit-identically the agent it passed in.

    Until 2026-08-15 (a) was covered by a hand-listed set of four E3 attribute names,
    which left 21 attributes drifting per read, 17 of them surviving agent.reset();
    see hazard H1 in the module docstring for the measured consequence. The
    verification that this is now genuinely empty is a CONTRACT, not a comment:
    tests/contracts/test_probe_warmup_nondestructive.py fingerprints the whole
    surface either side of a real read and requires an EMPTY diff.

    WHAT IS STILL NOT RESTORED, STATED PLAINLY: process-global RNG. The read consumes
    torch (and stdlib `random`, via hippocampal._rng) draws, so a subsequent training
    leg starts at a different stream offset than it would have without the read. That
    is deliberately out of scope here -- it is the CALLER's stage boundary to pin, it
    is not agent state, and restoring it silently would be an unmeasured semantic
    change to every consumer. The consequence to know about: two reads of the same
    restored agent do NOT return the same D_action_mass. A ladder that wants a
    controlled comparison must re-pin RNG at each stage boundary itself.

    BUDGET DENOMINATION. `max_episodes` should be >= `max_env_steps` so the STEP cap
    is what binds; see WarmupRecipe's docstring for why a tight episode cap starves a
    short-episode seed. A tight cap here raises RolloutBudget's EpisodeCapWarning
    unless `allow_tight_episode_cap=True` declares the intent. Either way the realised
    budget comes back in the return dict and reaches the manifest, so a read that was
    episode-denominated is visible rather than inferred.
    """
    surface = capture_agent_surface(agent)
    _sd_snapshot = copy.deepcopy(agent.state_dict())

    if captured is None:
        captured = reapply_candidate_capture(agent)

    harness = StepHarness(agent, env, train_mode=False, seed=seed)
    d_vals: List[float] = []

    def _observe(ctx: TickContext) -> Optional[Mapping[str, int]]:
        probs = ctx.probs
        cands = captured.get("cands")
        if not (ctx.fresh and cands is not None and probs is not None):
            return None
        if len(cands) != int(probs.numel()):
            return None
        mask = _candidate_noop_mask(cands)
        if mask is None:
            return None
        pp = probs.detach().reshape(-1).float()
        d_vals.append(float(pp[~mask].sum().item()))
        return {"selections": 1}

    agent.eval()
    try:
        with torch.no_grad():
            outcome = run_cell_until_samples(
                env=env,
                agent=agent,
                harness=harness,
                budget=RolloutBudget(
                    sample_floors={"selections": int(n_selections)},
                    max_env_steps=int(max_env_steps),
                    steps_per_episode=int(steps_per_episode),
                    max_episodes=int(max_episodes),
                    allow_tight_episode_cap=bool(allow_tight_episode_cap),
                ),
                observe=_observe,
                progress_label=label,
            )
    finally:
        # Undo BOTH drifts this read caused, so the measurement leaves the warmed
        # agent bit-identical to how it arrived. state_dict FIRST (it copies into the
        # live parameter/buffer tensors), then the attribute surface -- which also
        # restores each module's `training` flag, so the eval() above needs no
        # explicit undo and cannot wake a submodule the caller left in eval mode.
        agent.load_state_dict(_sd_snapshot)
        restore_report = restore_agent_surface(agent, surface)

    # Realised budget, on BOTH return paths. The zero-selection path needs these most:
    # "collected ZERO selections" reads as a sampling bug, and only the budget fields
    # distinguish that from a read the episode cap cut short.
    budget_fields = {
        "n_env_steps": int(outcome.n_env_steps),
        "n_episodes": int(outcome.n_episodes),
        "max_env_steps": int(outcome.max_env_steps),
        "max_episodes": int(outcome.max_episodes),
        "episode_cap_can_bind": bool(outcome.episode_cap_can_bind),
        "floors_met": bool(outcome.floors_met),
        # Auditable proof that the read was non-destructive, next to the number it
        # qualifies. `by_reference` non-empty means some attribute could NOT be
        # protected and this read may have leaked into the next training leg.
        "restore_report": dict(restore_report),
    }

    if not d_vals:
        return {
            "d_action_mass_mean": None,
            "d_action_mass_std": None,
            "n_selections": 0,
            "stop_reason": outcome.stop_reason,
            **budget_fields,
        }

    mean = sum(d_vals) / len(d_vals)
    var = sum((v - mean) ** 2 for v in d_vals) / len(d_vals)
    return {
        "d_action_mass_mean": float(mean),
        "d_action_mass_std": float(math.sqrt(var)),
        "n_selections": len(d_vals),
        "stop_reason": outcome.stop_reason,
        **budget_fields,
    }


# ---------------------------------------------------------------------------
# Checkpoint cache (machine-local). Discipline mirrors maturation_curriculum.
# ---------------------------------------------------------------------------

def _cache_dir(cache_dir: Optional[Path]) -> Path:
    if cache_dir is not None:
        d = Path(cache_dir)
    else:
        env = os.environ.get("REE_PROBE_WARMUP_CACHE_DIR")
        d = Path(env) if env else (Path.home() / ".ree_probe_warmup_cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_disabled() -> bool:
    return os.environ.get("REE_PROBE_WARMUP_CACHE_DISABLE", "").strip() not in (
        "",
        "0",
        "false",
        "False",
    )


def _warmup_key(
    *,
    seed: int,
    recipe: WarmupRecipe,
    env_kwargs: Mapping[str, Any],
    arm_key: Optional[Mapping[str, Any]] = None,
) -> str:
    """Content-addressed cache key.

    DELIBERATELY OVER-INCLUSIVE, per maturation_curriculum's governing asymmetry:
    a false HIT corrupts a conclusion, a false MISS only wastes compute. The whole
    substrate is hashed (scope=None) rather than a declared narrow closure -- the
    narrow form needs the scope-conservatism machinery to stay honest, and this
    warmup is cheap enough that the extra busting is not worth that risk.

    `arm_key` (2026-08-30, IGW-20260830-222 / SD-PROBE-WARMUP fix): the CALLER's
    arm-conditional regulator/config flags (e.g. {"use_noise_floor": True,
    "use_phasic_burst": False}), folded into the hash so two arms that build
    DIFFERENT root-level agent attributes (e.g. one constructs agent.noise_floor,
    the other leaves it None) NEVER share a minted surface. V3-EXQ-963's defect
    was exactly this: WarmupRecipe.as_dict() carries only the warmup TRAINING
    schedule (num_episodes, steps_per_episode, ...), never the regulator flags the
    CALLER used to build `agent` before warm_agent() was invoked, so all four arms
    of a 2x2 hashed identically and only the first (T0P0, use_noise_floor=False)
    ever minted -- every other arm silently cache-HIT and restored T0P0's surface,
    including its `agent.noise_floor = None`, onto agents that had just constructed
    a real regulator. Pass every regulator-affecting flag the AGENT's construction
    (not the recipe) depends on. `None` (the default) is itself hashed as part of
    the payload below, so this bump also invalidates every pre-fix (schema v2 and
    earlier) cache entry regardless of whether a given caller opts in.
    """
    import hashlib
    import json

    # resolve_substrate_identity, NOT compute_substrate_hash: this key is written onto a
    # CACHED artifact, so a mid-run checkout move would key the warmup to a substrate it
    # was not built with -- the same false-HIT channel the 2026-07-20 executed-substrate
    # fix closed in arm_fingerprint. Identical value on a stable run, so no cache is
    # invalidated by this change.
    sub = resolve_substrate_identity(scope=None)
    arm_key_dict: Dict[str, Any] = dict(arm_key) if arm_key is not None else {}
    payload = {
        "schema": PROBE_WARMUP_SCHEMA,
        "substrate_hash": sub["substrate_hash"],
        "machine_class": machine_class(),
        "seed": int(seed),
        "recipe": recipe.as_dict(),
        "env_kwargs": {k: env_kwargs[k] for k in sorted(env_kwargs)},
        "arm_key": {k: arm_key_dict[k] for k in sorted(arm_key_dict)},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _cache_load(key: str, cache_dir: Optional[Path], logger: Callable[[str], None]):
    if _cache_disabled():
        return None
    path = _cache_dir(cache_dir) / ("%s.pt" % key)
    if not path.is_file():
        return None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # corrupt / partial / version skew -> MISS
        logger("probe_warmup cache: unreadable (%s) -> MISS: %s" % (path.name, exc))
        return None
    # Re-verify the stored key (guards a truncated write or filename collision).
    if not isinstance(blob, dict) or blob.get("key") != key:
        logger("probe_warmup cache: key mismatch on %s -> MISS" % path.name)
        return None
    return blob


def _cache_store(key: str, blob: Dict[str, Any], cache_dir: Optional[Path],
                 logger: Callable[[str], None]) -> None:
    if _cache_disabled():
        return
    d = _cache_dir(cache_dir)
    path = d / ("%s.pt" % key)
    tmp = d / ("%s.pt.tmp.%d" % (key, os.getpid()))
    try:
        torch.save(blob, tmp)
        os.replace(tmp, path)  # atomic within a filesystem; safe under parallel workers
    except Exception as exc:
        logger("probe_warmup cache: store failed for %s: %s" % (path.name, exc))
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _capture_cached_surface(agent: Any, logger: Callable[[str], None]) -> Dict[str, Any]:
    """Build the picklable non-state_dict payload for a warm-start cache blob.

    Same capture the read-restore uses, projected down to what pickle accepts, so a
    cache HIT restores the SAME surface a MISS ends up with rather than four E3
    scalars (hazard H1). Dropped paths are stored in the blob so the load side can
    say which state a HIT is missing instead of leaving it to be discovered.
    """
    surface = capture_agent_surface(agent)
    payload, dropped = _picklable_surface(surface)
    if surface.by_reference:
        logger("probe_warmup cache: %d attr(s) captured BY REFERENCE and not cached: %s"
               % (len(surface.by_reference), ", ".join(sorted(surface.by_reference)[:6])))
    if dropped:
        logger("probe_warmup cache: %d attr(s) are unpicklable and are NOT cached: %s"
               % (len(dropped), ", ".join(sorted(dropped)[:6])))
    return {"attrs": payload, "dropped": sorted(dropped),
            "by_reference": sorted(surface.by_reference)}


def _restore_cached_surface(agent: Any, cached: Mapping[str, Any],
                            logger: Callable[[str], None]) -> None:
    """Put a cached surface back onto a freshly-constructed agent (cache HIT path).

    CLOBBER GUARD (2026-08-30, IGW-20260830-222 / SD-PROBE-WARMUP fix): a cached
    attribute is applied ONLY when its TYPE matches whatever the freshly-
    constructed HIT agent already built for that name (type(None) counts as a
    type, so None-vs-None is a match). SYMMETRIC on purpose: a cached None is
    never written over a live non-None value (would silently delete a regulator
    the HIT agent's own config constructed), AND a cached non-None value is never
    written over a live None (would silently install a regulator the HIT agent's
    own config never asked for). This is what V3-EXQ-963 needed and did not have:
    the mint arm (use_noise_floor=False) captured `agent.noise_floor = None`;
    without this guard, restoring that surface onto a HIT agent built with
    use_noise_floor=True (a real NoiseFloor instance already constructed by
    __init__) silently overwrote the live regulator with None, skipping the
    `if self.noise_floor is not None:` gate at every subsequent select_action()
    tick for the rest of the run. Paired with the arm-conditional _warmup_key()
    fix so mismatched-config arms should no longer cache-HIT each other in the
    first place; this guard is what actually saves a mismatched HIT when they do
    (e.g. a caller that never passes arm_key, or a stale pre-fix cache entry read
    under a relaxed key) -- it is the primary defense for any existing driver that
    is not updated to pass arm_key, since REEAgent.__init__ has already built each
    arm's OWN regulator set, correctly, before warm_agent() is ever called; this
    guard's whole job is to stop the cache restore from overwriting that. A
    skipped attribute is counted and logged, never silent.
    """
    payload = cached.get("attrs") or {}
    if not payload:
        logger("probe_warmup cache: blob carries NO attribute surface -- a HIT agent "
               "will differ from a MISS agent in every non-state_dict attribute")
        return
    modules = dict(agent.named_modules())
    _MISSING = object()
    n_set = 0
    n_clobber_skipped = 0
    clobber_skipped_paths: List[str] = []
    missing_modules: List[str] = []
    for mpath, values in payload.items():
        module = modules.get(mpath)
        if module is None:
            missing_modules.append(mpath)
            continue
        live_vars = vars(module)
        for name, value in values.items():
            current = live_vars.get(name, _MISSING)
            # Case 1: this instance never set the attribute at all (no __init__
            # default, nothing lazily created yet) -- nothing live to protect,
            # restore verbatim (matches pre-fix behaviour for genuinely-new
            # attributes). Deliberately NOT `getattr(module, name, None)`: that
            # would fall through to a same-named class-level method/attribute
            # and misread "never set on this instance" as "set to None".
            if current is _MISSING:
                object.__setattr__(module, name, value)
                n_set += 1
                continue
            # Case 2: the attribute IS live on this instance (e.g.
            # `self.noise_floor: Optional[NoiseFloor] = None` in __init__, always
            # present, sometimes None) but its TYPE disagrees with the cached
            # value -- refuse. type(None) is type(None), so None-vs-None still
            # matches and falls through to case 3.
            if type(current) is not type(value):
                n_clobber_skipped += 1
                clobber_skipped_paths.append(("%s.%s" % (mpath, name)) if mpath else name)
                continue
            # Case 3: live and cached agree in kind -- restore the warmed value.
            object.__setattr__(module, name, value)
            n_set += 1
    if clobber_skipped_paths:
        logger(
            "probe_warmup cache: %d attr(s) NOT restored -- cached value's type "
            "disagreed with the HIT agent's own live value for that attribute "
            "(protecting the HIT agent's own arm-conditional construction from "
            "being clobbered by a differently-configured mint): %s"
            % (n_clobber_skipped, ", ".join(sorted(clobber_skipped_paths)[:6]))
        )
    if missing_modules:
        logger("probe_warmup cache: %d cached module path(s) absent on this substrate: %s"
               % (len(missing_modules), ", ".join(sorted(missing_modules)[:6])))
    for path in cached.get("dropped") or []:
        logger("probe_warmup cache: attr %s was not cacheable -- HIT keeps its "
               "fresh-construction value" % path)
    # The load-bearing scalar, kept as an explicit assertion rather than trusted to
    # the generic walk: if _running_variance did not round-trip, a cache-HIT agent
    # commits differently from a MISS agent (e3_selector.py:2703) and nothing
    # downstream would say so.
    e3 = getattr(agent, "e3", None)
    want = ((payload.get("e3") or {}).get("_running_variance"))
    if e3 is not None and want is not None and hasattr(e3, "_running_variance"):
        got = float(getattr(e3, "_running_variance"))
        if not math.isclose(got, float(want), rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(
                "probe_warmup: _running_variance round trip failed (want %r got %r). "
                "A cache HIT would diverge from a MISS in E3 commit behaviour."
                % (float(want), got)
            )


# ---------------------------------------------------------------------------
# Cross-arm checkpoint-sharing guard (hazard H2).
# ---------------------------------------------------------------------------

def assert_state_dict_shareable(agents: Sequence[Any], labels: Optional[Sequence[str]] = None) -> None:
    """Refuse to share one warmed checkpoint across arms whose parameter sets differ.

    The 2x2's regulators (TonicVigor, NoiseFloor) are zero-parameter, non-nn.Module,
    so all four arms have identical state_dict key sets and shapes -- which is what
    licenses one warmup per seed. This costs nothing and protects against a FUTURE
    arm that flips a flag which DOES construct a module: without it, load_state_dict
    would either raise deep in a run or (with strict=False) silently leave a module
    at random init.
    """
    if len(agents) < 2:
        return
    names = list(labels) if labels is not None else ["arm%d" % i for i in range(len(agents))]
    ref = agents[0].state_dict()
    ref_shapes = {k: tuple(v.shape) for k, v in ref.items() if hasattr(v, "shape")}
    for agent, name in zip(agents[1:], names[1:]):
        sd = agent.state_dict()
        if set(sd.keys()) != set(ref.keys()):
            only_ref = sorted(set(ref.keys()) - set(sd.keys()))[:5]
            only_arm = sorted(set(sd.keys()) - set(ref.keys()))[:5]
            raise RuntimeError(
                "probe_warmup: state_dict key sets differ between %s and %s -- a shared "
                "warmed checkpoint is NOT valid across these arms. only_in_%s=%r "
                "only_in_%s=%r" % (names[0], name, names[0], only_ref, name, only_arm)
            )
        shapes = {k: tuple(v.shape) for k, v in sd.items() if hasattr(v, "shape")}
        bad = [k for k in ref_shapes if shapes.get(k) != ref_shapes[k]]
        if bad:
            raise RuntimeError(
                "probe_warmup: state_dict shapes differ between %s and %s for %r -- a "
                "shared warmed checkpoint is NOT valid." % (names[0], name, bad[:5])
            )


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------

def warm_agent(
    agent: Any,
    env: Any,
    *,
    seed: int,
    recipe: WarmupRecipe,
    env_kwargs: Mapping[str, Any],
    arm_key: Optional[Mapping[str, Any]] = None,
    label: str = "",
    cache_dir: Optional[Path] = None,
    logger: Callable[[str], None] = print,
    measure: bool = True,
) -> WarmupOutcome:
    """Warm ONE agent to a (hopefully) non-degenerate action-value landscape.

    The caller MUST have already built `agent` and `env` -- this function never
    constructs them. That is deliberate and mirrors maturation_curriculum's
    discipline: because construction happens on BOTH the cache-hit and cache-miss
    paths in the caller's own RNG stream, downstream initialisation is bit-identical
    either way. Building the agent inside a cache-miss branch only would silently
    desynchronise the two paths.

    `arm_key` (2026-08-30, IGW-20260830-222 / SD-PROBE-WARMUP fix): pass every
    regulator-affecting flag `agent`'s construction used (e.g.
    {"use_noise_floor": agent_cfg.use_noise_floor, "use_phasic_burst":
    agent_cfg.use_phasic_burst}) so that two arms whose agents differ in which
    root-level attributes got constructed (some None, some real regulator
    instances) mint SEPARATE cache entries instead of sharing one. See
    `_warmup_key` and `_restore_cached_surface` for the full defect this closes
    (V3-EXQ-963) -- the `_restore_cached_surface` type-matching guard is the
    primary protection and applies even when `arm_key` is omitted (None, the
    default); passing it additionally avoids the wasted MISS/retrain a caller
    would otherwise pay every time the guard has to refuse a mismatched
    attribute.

    Per the user-confirmed gate policy this RECORDS rather than aborts: a seed that
    stays saturated comes back with saturated=True and its realised mean. Only
    assert_any_informative() raises, and only when NO seed de-saturated.

    Returns a WarmupOutcome; emit .as_manifest_fields() into the run manifest so
    informative yield is auditable rather than inferred.
    """
    key = _warmup_key(seed=seed, recipe=recipe, env_kwargs=env_kwargs, arm_key=arm_key)
    notes: List[str] = []
    blob = _cache_load(key, cache_dir, logger)
    cache_hit = blob is not None

    if cache_hit:
        agent.load_state_dict(blob["agent_state"])
        _restore_cached_surface(agent, blob.get("surface", {}), logger)
        notes.append("warmup restored from cache")
        logger("  [warmup] %s seed=%d HIT (%s eps, cached)"
               % (label, seed, recipe.num_episodes))
    else:
        if recipe.regime != "target_env":
            raise ValueError(
                "probe_warmup: regime %r is declared but not built; only 'target_env' "
                "is implemented (SD-074 default). A curriculum regime would add a "
                "second env as a confound and must be justified in its own SD."
                % (recipe.regime,)
            )
        logger("  [warmup] %s seed=%d MISS -- training %d episodes"
               % (label, seed, recipe.num_episodes))
        warmup_train(
            agent,
            env,
            num_episodes=int(recipe.num_episodes),
            steps_per_episode=int(recipe.steps_per_episode),
            label="%s warmup seed=%d" % (label, seed),
        )
        _cache_store(
            key,
            {
                "key": key,
                "schema": PROBE_WARMUP_SCHEMA,
                "seed": int(seed),
                "recipe": recipe.as_dict(),
                "agent_state": agent.state_dict(),
                "surface": _capture_cached_surface(agent, logger),
            },
            cache_dir,
            logger,
        )
        notes.append("warmup trained fresh")

    if not measure:
        return WarmupOutcome(
            seed=int(seed),
            d_action_mass_mean=None,
            d_action_mass_std=None,
            regime="unmeasured",
            saturated=False,
            n_probe_selections=0,
            probe_stop_reason="not_measured",
            warmup_episodes=int(recipe.num_episodes),
            cache_hit=cache_hit,
            recipe=recipe.as_dict(),
            notes=notes + ["de-saturation probe skipped (measure=False)"],
        )

    read = measure_action_mass(
        agent,
        env,
        seed=seed,
        n_selections=recipe.probe_selections,
        max_env_steps=recipe.probe_max_env_steps,
        max_episodes=recipe.resolved_probe_max_episodes,
        steps_per_episode=recipe.steps_per_episode,
        label="%s desat seed=%d" % (label, seed),
        allow_tight_episode_cap=recipe.allow_tight_episode_cap,
    )
    d_mean = read["d_action_mass_mean"]
    regime = saturation_regime(d_mean)
    if d_mean is None:
        notes.append("de-saturation probe collected ZERO selections (stop=%s)"
                     % read["stop_reason"])
    # Say it in the notes too, not only in a boolean field: a starved or
    # episode-denominated read is a caveat on the saturation VERDICT, and a reader
    # scanning notes should not have to cross-reference the budget fields to find it.
    if read["episode_cap_can_bind"]:
        notes.append(
            "probe read was EPISODE-denominated (max_episodes=%d < max_env_steps=%d): "
            "a short-episode seed could not spend its full step budget"
            % (read["max_episodes"], read["max_env_steps"])
        )
    if read["restore_report"]["n_by_reference"]:
        notes.append(
            "de-saturation read could NOT protect %d attr(s) (%s): this read may have "
            "leaked into any subsequent training leg -- see hazard H1"
            % (read["restore_report"]["n_by_reference"],
               ", ".join(read["restore_report"]["by_reference"][:4]))
        )
    if not read["floors_met"]:
        notes.append(
            "probe read STARVED: %d of %d selections in %d env steps / %d episodes "
            "(stop=%s) -- treat this seed's saturation verdict as low-confidence"
            % (read["n_selections"], int(recipe.probe_selections),
               read["n_env_steps"], read["n_episodes"], read["stop_reason"])
        )

    logger("  [warmup] %s seed=%d D_action_mass_mean=%s regime=%s (n=%d, stop=%s)"
           % (label, seed,
              "None" if d_mean is None else ("%.4f" % d_mean),
              regime, read["n_selections"], read["stop_reason"]))

    return WarmupOutcome(
        seed=int(seed),
        d_action_mass_mean=d_mean,
        d_action_mass_std=read["d_action_mass_std"],
        regime=regime,
        saturated=(regime != "headroom"),
        n_probe_selections=int(read["n_selections"]),
        probe_stop_reason=str(read["stop_reason"]),
        n_probe_env_steps=int(read["n_env_steps"]),
        n_probe_episodes=int(read["n_episodes"]),
        probe_max_env_steps=int(read["max_env_steps"]),
        probe_max_episodes=int(read["max_episodes"]),
        probe_episode_cap_can_bind=bool(read["episode_cap_can_bind"]),
        probe_floors_met=bool(read["floors_met"]),
        warmup_episodes=int(recipe.num_episodes),
        cache_hit=cache_hit,
        recipe=recipe.as_dict(),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Yield accounting -- the auditable record the autopsy asked for.
# ---------------------------------------------------------------------------

def saturation_summary(outcomes: Sequence[WarmupOutcome]) -> Dict[str, Any]:
    """Aggregate realised per-seed saturation into manifest fields.

    The autopsy's target condition is 'a MAJORITY of seeds with D_action_mass_mean
    strictly inside (0.05, 0.95)', so `target_met` is informative_yield > 0.5 --
    strictly a majority, not >=. The V3-EXQ-777a baseline (0.286) is carried in the
    output so any reader can see the movement without going back to the autopsy.
    """
    n = len(outcomes)
    regimes = {"ceiling": 0, "headroom": 0, "floor": 0, "unmeasured": 0}
    for o in outcomes:
        regimes[o.regime] = regimes.get(o.regime, 0) + 1
    informative = [o for o in outcomes if o.informative]
    yield_frac = (len(informative) / n) if n else 0.0
    # Budget audit. An informative_yield computed over STARVED reads is not a
    # de-saturation result, it is a sampling artefact -- so the aggregate must state
    # how many seeds' reads were budget-limited, next to the yield it qualifies.
    measured = [o for o in outcomes if o.regime != "unmeasured"]
    starved = sorted(o.seed for o in measured if not o.probe_floors_met)
    cap_bind = sorted(o.seed for o in measured if o.probe_episode_cap_can_bind)
    return {
        "n_seeds": n,
        "n_informative": len(informative),
        "informative_yield": float(yield_frac),
        "informative_seeds": sorted(o.seed for o in informative),
        "saturated_seeds": sorted(o.seed for o in outcomes if o.saturated),
        "regimes": regimes,
        "n_probe_starved": len(starved),
        "probe_starved_seeds": starved,
        "n_probe_episode_cap_can_bind": len(cap_bind),
        "probe_episode_cap_can_bind_seeds": cap_bind,
        "probe_budget_clean": bool(not starved and not cap_bind),
        "probe_budget_note": (
            "probe_budget_clean=False means at least one seed's de-saturation read was "
            "budget-limited (starved below its selection floor, or run under an episode "
            "cap that can bind before the step cap). Qualify informative_yield "
            "accordingly -- a starved read's saturation verdict is low-confidence."
        ),
        "per_seed": [o.as_manifest_fields() for o in outcomes],
        "d_sat_low": D_SAT_LOW,
        "d_sat_high": D_SAT_HIGH,
        "target_condition": "majority of seeds with D_action_mass_mean strictly inside (0.05, 0.95)",
        "target_met": bool(yield_frac > 0.5),
        "baseline_v3_exq_777a_informative_yield": 0.286,
        "baseline_note": (
            "V3-EXQ-777a (untrained agent): 4 of 14 informative (28.6%); 7 ceiling, "
            "2 floor. That run is the failure record this substrate must move."
        ),
    }


def assert_any_informative(outcomes: Sequence[WarmupOutcome]) -> None:
    """Raise only when the warmup provably did nothing (zero seeds de-saturated).

    Per-seed saturation is a recorded outcome, not an error -- see the module
    docstring's gate policy. This guards the one case where continuing would produce
    a manifest full of vacuous numbers.
    """
    if not outcomes:
        raise NoInformativeSeeds("probe_warmup: no warmup outcomes were produced at all")
    if not any(o.informative for o in outcomes):
        raise NoInformativeSeeds(
            "probe_warmup: ZERO of %d seeds de-saturated after warmup "
            "(regimes=%r). The warmup did not move the action-value landscape, so "
            "every downstream telemetry number would be vacuous. Sweep "
            "WarmupRecipe.num_episodes upward before interpreting anything."
            % (len(outcomes), [o.regime for o in outcomes])
        )
