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
(H1) e3._running_variance IS NOT IN state_dict. It is a plain Python float
     (ree_core/predictors/e3_selector.py:291), not a register_buffer, and it feeds
     commit_variance at e3_selector.py:2703 -- directly upstream of the very probs
     distribution D_action_mass measures. A naive state_dict-only cache would make
     cache-HIT and cache-MISS agents differ in commit behaviour, SILENTLY. We
     therefore carry the declared non-buffer E3 scalars explicitly in the blob and
     assert the round trip.

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

import copy
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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
from arm_fingerprint import compute_substrate_hash, machine_class  # noqa: E402
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
    "warm_agent",
    "measure_action_mass",
    "saturation_regime",
    "saturation_summary",
    "assert_state_dict_shareable",
    "assert_any_informative",
    "reapply_candidate_capture",
    "NoInformativeSeeds",
]

PROBE_WARMUP_SCHEMA = "probe_warmup.v1"

# Saturation bounds. Deliberately IDENTICAL to V3-EXQ-777a's D_SAT_LOW / D_SAT_HIGH
# (script:246-247) so this substrate's success criterion is expressed in the SAME
# units as the failure record that motivated it. Do not retune these to make a
# warmup look better -- that would silently move the goalposts the autopsy set.
D_SAT_LOW = 0.05
D_SAT_HIGH = 0.95

# CausalGridWorldV2 ACTIONS[4] = (0,0) = stay/no-op. Mirrors 777a:226.
NOOP_CLASS = 4

# E3 state that is NOT in state_dict (hazard H1). Plain Python attributes set in
# E3Selector.__init__; each is captured if present and restored on a cache hit.
# _running_variance is the load-bearing one (feeds commit_variance, e3_selector.py:2703);
# the others are accumulators whose omission would make a HIT agent subtly younger
# than a MISS agent.
_E3_NONBUFFER_STATE: Tuple[str, ...] = (
    "_running_variance",   # e3_selector.py:291  -- float, EMA of PE-MSE
    "_rv_history",         # e3_selector.py:312  -- deque(maxlen=100)
    "_novelty_ema",        # e3_selector.py:268  -- float
    "_last_error_var",     # SD-069 instantaneous-PE source, if present
)


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

      (a) plain-Python accumulators -- e3._running_variance drifted 0.001839 ->
          0.001855 over a 25-selection read, and it feeds commit_variance
          (e3_selector.py:2703);
      (b) REGISTERED BUFFERS -- the three-factor plasticity / eligibility traces
          (e3_selector.py:373-462) are updated in-place under no_grad. Measured: two
          agents identical at load diverged by max |dw| = 2.5e-01 after two
          de-saturation reads of different lengths (71 vs 50 env steps).

    Both would perturb the very distribution being measured, and would make
    `measure=True` hand back a different agent than `measure=False`. So the full
    state_dict AND the declared non-buffer scalars are snapshotted and restored --
    the caller gets back bit-identically the agent it passed in.

    BUDGET DENOMINATION. `max_episodes` should be >= `max_env_steps` so the STEP cap
    is what binds; see WarmupRecipe's docstring for why a tight episode cap starves a
    short-episode seed. A tight cap here raises RolloutBudget's EpisodeCapWarning
    unless `allow_tight_episode_cap=True` declares the intent. Either way the realised
    budget comes back in the return dict and reaches the manifest, so a read that was
    episode-denominated is visible rather than inferred.
    """
    if captured is None:
        captured = reapply_candidate_capture(agent)

    # Snapshot everything the rollout can drift (see docstring (a) and (b)).
    _e3_snapshot = _capture_e3_nonbuffer(agent)
    _sd_snapshot = copy.deepcopy(agent.state_dict())

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

    was_training = bool(getattr(agent, "training", False))
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
        if was_training:
            agent.train()
        # Undo BOTH drifts this read caused (buffers, then scalars), so the
        # measurement leaves the warmed agent bit-identical to how it arrived.
        agent.load_state_dict(_sd_snapshot)
        e3 = getattr(agent, "e3", None)
        if e3 is not None:
            for _name, _value in _e3_snapshot.items():
                if hasattr(e3, _name):
                    setattr(e3, _name, copy.deepcopy(_value))

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


def _warmup_key(*, seed: int, recipe: WarmupRecipe, env_kwargs: Mapping[str, Any]) -> str:
    """Content-addressed cache key.

    DELIBERATELY OVER-INCLUSIVE, per maturation_curriculum's governing asymmetry:
    a false HIT corrupts a conclusion, a false MISS only wastes compute. The whole
    substrate is hashed (scope=None) rather than a declared narrow closure -- the
    narrow form needs the scope-conservatism machinery to stay honest, and this
    warmup is cheap enough that the extra busting is not worth that risk.
    """
    import hashlib
    import json

    sub = compute_substrate_hash(scope=None)
    payload = {
        "schema": PROBE_WARMUP_SCHEMA,
        "substrate_hash": sub["substrate_hash"],
        "machine_class": machine_class(),
        "seed": int(seed),
        "recipe": recipe.as_dict(),
        "env_kwargs": {k: env_kwargs[k] for k in sorted(env_kwargs)},
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


def _capture_e3_nonbuffer(agent: Any) -> Dict[str, Any]:
    """Snapshot the E3 attributes that state_dict() does NOT carry (hazard H1)."""
    e3 = getattr(agent, "e3", None)
    if e3 is None:
        return {}
    out: Dict[str, Any] = {}
    for name in _E3_NONBUFFER_STATE:
        if hasattr(e3, name):
            out[name] = copy.deepcopy(getattr(e3, name))
    return out


def _restore_e3_nonbuffer(agent: Any, state: Mapping[str, Any],
                          logger: Callable[[str], None]) -> None:
    e3 = getattr(agent, "e3", None)
    if e3 is None:
        return
    for name, value in state.items():
        if hasattr(e3, name):
            setattr(e3, name, copy.deepcopy(value))
        else:
            logger("probe_warmup: cached E3 attr %s absent on this substrate" % name)
    # The load-bearing one. If this did not round-trip, a cache-HIT agent would
    # commit differently from a cache-MISS agent and nothing downstream would say so.
    if "_running_variance" in state and hasattr(e3, "_running_variance"):
        got = float(getattr(e3, "_running_variance"))
        want = float(state["_running_variance"])
        if not math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(
                "probe_warmup: _running_variance round trip failed (want %r got %r). "
                "A cache HIT would diverge from a MISS in E3 commit behaviour." % (want, got)
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

    Per the user-confirmed gate policy this RECORDS rather than aborts: a seed that
    stays saturated comes back with saturated=True and its realised mean. Only
    assert_any_informative() raises, and only when NO seed de-saturated.

    Returns a WarmupOutcome; emit .as_manifest_fields() into the run manifest so
    informative yield is auditable rather than inferred.
    """
    key = _warmup_key(seed=seed, recipe=recipe, env_kwargs=env_kwargs)
    notes: List[str] = []
    blob = _cache_load(key, cache_dir, logger)
    cache_hit = blob is not None

    if cache_hit:
        agent.load_state_dict(blob["agent_state"])
        _restore_e3_nonbuffer(agent, blob.get("e3_nonbuffer", {}), logger)
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
                "e3_nonbuffer": _capture_e3_nonbuffer(agent),
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
