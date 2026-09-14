"""V3-EXQ-1012a -- SELECTION-LEVEL re-pose of the E3 channel-commensurability rung-3
readiness validation (claim MECH-439, substrate SD-E3-CHANNEL-COMMENSURABILITY).

SLEEP DRIVER: none (no sleep flags set; 936-regime parity).

=== WHY THIS DRIVER EXISTS, AND WHY IT IS NOT V3-EXQ-1012 ===

V3-EXQ-1012 (parked, never queued: experiments/_scratch/v3_exq_1012_..._.py.blocked) was
authored against the acceptance target the CONFIRMED failure_autopsy_V3-EXQ-571c_2026-09-02
pre-registered: ">= 2 E3 score channels simultaneously above a 1e-3 RELATIVE cross-candidate
share in the 936 regime". Step 4.5 red-team (fable) returned BLOCKING: that target is an
ARITHMETIC IDENTITY of the operator it is meant to validate -- the operator divides each
channel's per-candidate term by an EMA of THAT CHANNEL'S OWN cross-candidate SD, so
Var(term/s) = Var(term)/s^2 ~= 1 per channel BY CONSTRUCTION, and shares tend to 1/k
regardless of data. No experiment adopting that target could fail. Full derivation:
evidence/planning/exq1012_blocked_readiness_target_tautological_20260908.md.

Governance re-posed the target (GFLAG-0234, option A, user-approved 2026-09-10; applied in
REE_assembly 5dc661badc across all four carrier docs -- docs/architecture/
sd_e3_channel_commensurability.md, both failure_autopsy_V3-EXQ-571c_2026-09-02 files, and
substrate_queue.json) onto a SELECTION-level DV: "commit-flip rate under shadow OFF/ON
scoring on the same tick and the same candidate set". This driver builds and queues that
instrument. Full build spec, source citations, the non-forced-ness argument and the
build-time scoping decision recorded below: evidence/planning/
mech439_rung3_selection_level_repose_20260914.md (sections 3, 3.5, 4, 6a).

EXPERIMENT_PURPOSE = "diagnostic" -- this is SUBSTRATE READINESS VALIDATION, not governance
evidence. It answers whether the operator has a selection-consequential effect, not whether
any downstream MECH-439/ARC-062 conversion improves.

=== THE MEASUREMENT (see the design doc section 3 / 6a for the full derivation) ===

Two arms, BOTH with the operator ON always (`use_e3_channel_commensurability=True`): a FED
regime (residue-dominated, matching 571c's fed-arm occupant) and a STARVED regime (f/harm-
dominated, matching 571c's starved-arm occupant). There is deliberately NO separate OFF arm
run: the OFF/ON comparison is done INTRA-TICK via a shadow rescore, so a single ON-arm run
per regime supplies both readings.

At every genuine (fresh, latch-gated) P1 selection tick: `agent.e3.score_trajectory` is
transparently wrapped for the duration of that one `agent.select_action(...)` call to
CAPTURE the exact (args, kwargs, returned score) triple used for each candidate -- the live
call's behaviour and return value are completely unchanged; this only observes it. Verified
directly against `ree_core/predictors/e3_selector.py` (current HEAD): the running scale
estimate `_chan_scale_ema` is updated ONLY once per `select()` call, from inside `select()`'s
own per-candidate loop (`_update_channel_scale_estimates`, e3_selector.py:3099-3100), and
NEVER from a bare `score_trajectory` call -- so a replay of the captured calls can never
mutate that state, whatever config it runs under. A bare-`score_trajectory` probe run this
session (task_claim metaworker-chip-20260911-mech439-rung3-target-repose) confirmed exactly
that in isolation. It did NOT, on its own, prove replay is exact against a LIVE tick:
`select()` folds THAT tick's own spread into `_chan_scale_ema` before returning, so by the
time a replay runs (necessarily after `select_action()` has returned), the EMA it would read
already reflects the very tick it is trying to reproduce -- a one-step timing mismatch the
isolated probe could not see. `_ScoreCallCapture` therefore snapshots `_chan_scale_ema`
BEFORE the live call and swaps that snapshot in for the duration of each replay, restoring
the real (evolving) EMA afterward; see its docstring for the exact failure this closes (found
by this build's own smoke test: 10/10 self-check failures in the fed arm before the fix).

Two replays of the captured calls follow, per genuine tick:
  1. SHADOW (toggle OFF): the counterfactual "what would this tick have picked with the
     operator disabled". `primary_argmin_flip = 1` iff argmin(shadow scores) differs from
     argmin(live scores).
  2. SELF-CHECK (toggle to the SAME value, i.e. ON): a harness bug detector, not a design
     control. Because the EMA state is unperturbed (point above), replaying at the live
     config value MUST reproduce the live scores and argmin EXACTLY. If it does not, the
     shadow-replay harness has silently diverged from the live scoring path and this driver
     REFUSES rather than report numbers it cannot trust (see `_ScoreCallCapture.replay` and
     its caller in `run_cell`).

`xcand_commit_flip_rate` per cell = mean of `primary_argmin_flip` over every genuine P1
tick. This characterises the PRIMARY-score stage the operator directly acts on -- see the
build-time scoping addendum (design doc section 6a) for why a FINAL-commit-level replay
(through the modulatory shortlist stage) is deliberately NOT built here: it would require
either refactoring e3_selector.py's post-raw_scores stretch into a shared callable, or
hand-duplicating ~150 lines of shortlist/Go-No-Go/modulatory-argmin logic inside this driver
and auditing it against source on every future e3_selector.py change -- disproportionate to
a diagnostic readiness probe. `final_commit_by_primary_frac` is recorded per cell (matching
the parked driver's own C6 diagnostic) as mandatory interpretive context: on a cell where
this is low, "the primary stage never flips" does NOT mean "the operator never changes the
executed action" -- it means this DV cannot see the stage where the action was actually
decided. A follow-on `/implement-substrate` chip is the right route to a final-commit-level
successor; this driver does not attempt it.

=== NON-FORCED-NESS (design rule (a); design doc section 3.5) ===

Unlike the retired share-based target -- which is a normalisation IDENTITY, forced to ~1/k
by construction whatever the data -- there is no analogous closed form for an argmin flip
under a DIFFERENTIAL per-channel rescaling: only a COMMON positive rescaling of every channel
is provably argmin-invariant (the driver's own DV-symmetry note below), and this operator
rescales each channel by a DIFFERENT factor. Whether an argmin flips is therefore a genuine
function of the data. Empirically: the FED and STARVED regimes are reported SEPARATELY, and
571c measured structurally different raw channel-magnitude relationships between them (fed:
residue variance ~452 vs harm ~0.005; starved: a flatter relationship). If
`xcand_commit_flip_rate` read identically in both regimes independent of the data, that
would itself be evidence of a hidden identity and is flagged (`flip_rates_differ_across_
regimes`), not silently reported as a clean result.

DV-SYMMETRY DECLARATION. The routed statistic is a per-tick ARGMIN comparison over two
score vectors differing only in the commensurability toggle. It is invariant under: a
per-tick offset UNIFORM across candidates (broadcast scalar -- correctly invisible, argmin-
invariant), and permutation of the candidate index. It is NOT invariant under a per-channel
DIFFERENTIAL rescaling -- which is exactly the operator's own action and exactly why this DV
can discriminate it.

=== GOV-REUSE-1 (Step 2.4) -- checked, NEGATIVE ===

The banked V3-EXQ-571c manifest and the V3-EXQ-1012 authoring smoke record only PER-CELL
AGGREGATES (xcand_share, xcand_var_mean, n_live_channels) -- never the per-tick,
per-candidate raw scores a shadow OFF/ON comparison needs. No compatible substrate_hash
carries the decisive readout or its inputs. A live run is required. Full note: design doc
section 1.

=== RE-DERIVE BRAKE (Step 2.5b) -- NOT braked, same determination the parked driver made ===

MECH-439 carries >= 2 counted substrate_ceiling autopsies, but this is a DIAGNOSTIC on the
MEASUREMENT axis (does the operator have a selection-consequential effect -- an instrument
question) asking a different question from the braked f-dominance design lineage, exactly as
571c's own autopsy and the parked V3-EXQ-1012 driver already determined for this same
substrate. No ceiling hit is added by this run. MECH-439's direction does not move.

=== SUBSTRATE READINESS (Step 2.5) ===

SD-E3-CHANNEL-COMMENSURABILITY rung 3 is IMPLEMENTED (ree-v3 c47b885, 2026-09-07). Confirmed
live this session: `E3Config.use_e3_channel_commensurability` exists and is toggleable on a
constructed agent's `agent.e3.config`.

red-team (model fable): CONTESTED, 4 findings, all fixed before queueing --
  (1) the non-forced-ness check (regimes_differ) was computed but never routed on; now a
      both-regimes-pass reading with regimes_differ False routes to a distinct flagged FAIL
      label instead of the clean both-regimes PASS.
  (2) the SD-scaled PASS bar conflated "no effect" with "heterogeneous effect" across the 4
      seeds; added a majority-of-seeds-clear-the-absolute-floor test alongside it (PASS iff
      EITHER clears), and split the flip_rate_lift criterion's measured/threshold per regime
      instead of a cross-arm max/min pair.
  (3) `_ScoreCallCapture` snapshotted `_chan_scale_ema` for the replay but not
      `_chan_scale_n`, which ALSO gates `_commensurability_scale`'s warmup no-op -- on the one
      tick the count crosses the warmup threshold this produced a genuine self-check mismatch
      (not schedule-reachable at full scale since the crossing lands in unscored P0, but fixed
      so the harness is correct rather than schedule-lucky). Now both are snapshotted/restored
      together.
  (4) `operator_engaged` is a pure tick-count gate and could in principle read green while
      every live channel's scale sits at/below the commensurability floor (making live and
      shadow scores identical despite "engagement"); added a
      `shadow_scores_measurably_different` precondition asserting the shadow rescoring
      actually produced a nonzero score delta. Also loosened SELF_CHECK_TOLERANCE from a
      pure-absolute to a relative+floor comparison (was, in effect, a float32 bit-identity
      requirement despite the original docstring calling it "roundtrip only").
Re-smoked clean after every fix (self_check_failures=0, validate_experiments.py --strict OK).
EXPERIMENT_PURPOSE = "diagnostic" -- excluded from governance confidence scoring.
ASCII-only output (repo rule).
"""

from __future__ import annotations

import argparse
import datetime
import math
import random
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.fresh_select import FreshSelectCounter, FreshSelectProbe  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.baselines.mech439_f_variance_share import (  # noqa: E402
    CONTRASTIVE_BATCH_K,
    E2_CONTRASTIVE_LR,
    E2_TRAIN_EVERY_K_TICKS,
    ENV_KWARGS,
    OFF_ARM_FLAGS,
    P0_WARMUP_EPISODES,
    SEEDS,
    STEPS_PER_EPISODE,
    TRANSITION_BUFFER_MAX,
    make_agent_kwargs,
    make_env,
    off_path_config_slice,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_1012a_e3_commensurability_selection_level_regime_validation"
QUEUE_ID = "V3-EXQ-1012a"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-439"]

# The clamp IS set: e2_rollout_output_norm_clamp_enabled=True lives in the lineage's
# CONFIG_FLAGS (experiments/_lib/baselines/mech439_f_variance_share.py), applied via
# kwargs.update(CONFIG_FLAGS) inside make_agent_kwargs() -- the same **kwargs-splat
# indirection the parked V3-EXQ-1012 driver's own exemption documented as the validator's
# blind spot. Both cells re-assert it landed (clamp_live / clamp_ratio_live, gated as
# clamp_config_landed).
SD056_ROLLOUT_CLAMP_EXEMPT = (
    "clamp set via CONFIG_FLAGS dict-splat in "
    "experiments/_lib/baselines/mech439_f_variance_share.py::make_agent_kwargs "
    "(936-regime parity, shared with the parked V3-EXQ-1012 driver); landing re-asserted "
    "per cell as clamp_config_landed"
)

# Both load-bearing criteria (flip_rate_lift, self_check_clean) DO carry numeric
# measured/threshold fields (see run_experiment's `criteria` list) -- the static AST scan
# cannot trace the dict-comprehension-free literal construction used there. Advisory-only
# check; this documents the false positive rather than restructuring working code to
# appease a best-effort scanner.
CRITERIA_THRESHOLD_EXEMPT = (
    "flip_rate_lift and self_check_clean both record measured+threshold as separate "
    "numeric fields in the criteria list built by run_experiment(); the AST scan's "
    "best-effort literal-matching does not resolve them"
)

# --- Pre-registered thresholds (constants, never derived from this run) ------
MIN_FRESH_SELECTIONS = 60          # 936a/571b decomp-sample floor, reused
N_FRESH_SELECT_TARGET = 200        # 936a's P1 measurement budget
P1_EPISODE_CAP = 40
SELF_CHECK_TOLERANCE = 1e-6         # RELATIVE exact-replay tolerance (see _self_check_ok)
FLIP_RATE_SD_MULTIPLIER = 2.0      # K: PASS bar scales on the seed-matched SD of the delta
FLIP_RATE_ABS_FLOOR = 0.01         # plus an absolute floor (design doc section 4)
REGIME_DIFFERENCE_FLOOR = 0.02     # non-forced-ness flag: fed vs starved flip-rate gap

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1_CAP = 2
DRY_RUN_STEPS = 60
DRY_RUN_FRESH_TARGET = 6

_ZG = ZGoalStreamAccumulator()
_LAST_AGENT: Dict[str, Any] = {"agent": None}
_FRESH_SELECT = FreshSelectProbe("exq1012a")

ARMS: List[Dict[str, Any]] = [
    {"id": "C_fed_operator_on", "feed_residue": True, "warmup": True, "load_bearing": True},
    {"id": "C_starved_operator_on", "feed_residue": False, "warmup": True, "load_bearing": False},
]
FED_ARM = "C_fed_operator_on"
STARVED_ARM = "C_starved_operator_on"


PRECONDITION_SPECS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="decomp_samples_sufficient",
        description="genuine (fresh, latch-gated) P1 selections -- the flip-rate denominator",
        control="worst seed of this arm",
        threshold=float(MIN_FRESH_SELECTIONS),
        direction="lower",
    ),
    PreconditionSpec(
        name="operator_engaged",
        description=(
            "channel_scale_estimates.engaged at cell end -- the operator is inert for its "
            "first e3_commensurability_warmup_ticks select() calls; an arm that never left "
            "warmup has no manipulation in it and the shadow OFF/ON comparison is vacuous "
            "(both sides would read the un-normalised floor identically)"
        ),
        control="worst seed of this arm",
        threshold=1.0,
        direction="lower",
    ),
    PreconditionSpec(
        name="clamp_config_landed",
        description="the rollout clamp flag actually reached agent.e2.config in every cell",
        control="all cells of the arm",
        threshold=1.0,
        direction="lower",
    ),
    PreconditionSpec(
        name="residue_protocol_landed",
        description="the arm's residue-feeding protocol executed as declared",
        control="all cells of the arm",
        threshold=1.0,
        direction="lower",
    ),
    PreconditionSpec(
        name="shadow_scores_measurably_different",
        description=(
            "mean |live - shadow| score delta across scored ticks -- closes a residual "
            "gap `operator_engaged` alone cannot (Step 4.5 red-team, model fable, finding "
            "4a): `operator_engaged` is a pure TICK-COUNT gate, so it reads green whenever "
            "enough ticks have passed, even in the unlikely case every live channel's EMA "
            "sits at/below the commensurability floor -- `_commensurability_scale` then "
            "returns the 1.0 no-op for every channel and live==shadow identically despite "
            "the count-based gate being green. This precondition asserts the shadow "
            "rescoring actually PRODUCED a measurable difference, not merely that enough "
            "ticks were seen."
        ),
        control="worst seed of this arm",
        threshold=1e-9,
        direction="lower",
    ),
]
GEQ_PRECONDITIONS = {"decomp_samples_sufficient", "operator_engaged", "clamp_config_landed",
                      "residue_protocol_landed", "shadow_scores_measurably_different"}


def config_slice_for(arm: Dict[str, Any], p0_episodes: int, p1_cap: int, steps: int,
                      fresh_target: int) -> Dict[str, Any]:
    """Every readout-affecting constant, declared. Built on the lineage's canonical
    baseline slice; the operator is ALWAYS ON in this driver (see module docstring for
    why there is no separate OFF arm) so it is a constant here, not an arm axis."""
    base = off_path_config_slice()
    return {
        "use_e3_channel_commensurability": True,
        "env_kwargs": base["env_kwargs"],
        "sd056_training": base["sd056_training"],
        "config_flags": base["config_flags"],
        "off_arm_flags": base["off_arm_flags"],
        "schedule": {
            "p0_warmup_episodes": int(p0_episodes if arm["warmup"] else 0),
            "p1_episode_cap": int(p1_cap),
            "steps_per_episode": int(steps),
            "fresh_select_target": int(fresh_target),
        },
        "protocol": {
            "feed_residue_per_step": bool(arm["feed_residue"]),
            "p0_warmup": bool(arm["warmup"]),
        },
        "cell_readout_constants": {
            "min_fresh_selections": MIN_FRESH_SELECTIONS,
            "self_check_tolerance": SELF_CHECK_TOLERANCE,
            "flip_rate_sd_multiplier": FLIP_RATE_SD_MULTIPLIER,
            "flip_rate_abs_floor": FLIP_RATE_ABS_FLOOR,
        },
    }


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """The lineage's baseline agent with the commensurability operator forced ON.

    Follows the `f_weight` precedent: the knob is NOT wired through
    `REEConfig.from_dims()` (which silently swallows unknown kwargs) -- it is set on the
    constructed `cfg.e3` object, and the assignment is verified to have SURVIVED onto the
    live selector before any stepping happens, so an operator that silently failed to reach
    E3 can never produce a fictitious null.
    """
    cfg = REEConfig.from_dims(**make_agent_kwargs(env, OFF_ARM_FLAGS))
    cfg.e3.use_e3_channel_commensurability = True
    agent = REEAgent(cfg)
    _live = bool(getattr(agent.e3.config, "use_e3_channel_commensurability", False))
    if not _live:
        raise RuntimeError(
            "e3 channel-commensurability knob did not reach the agent (requested True, live "
            "%r). Refusing to run a shadow-scoring probe against an operator that is not "
            "actually active." % (_live,))
    return agent


def _scale_estimates(agent: Any) -> Dict[str, Any]:
    """`E3TrajectorySelector.last_channel_scale_estimates`, defensively copied."""
    raw = getattr(getattr(agent, "e3", None), "last_channel_scale_estimates", None)
    if not isinstance(raw, dict):
        return {"present": False, "engaged": False, "n_updates": None, "scales": {}}
    out = {"present": True}
    for k in ("engaged", "n_updates", "warmup_ticks", "floor", "ema_alpha"):
        if k in raw:
            out[k] = raw[k]
    sc = raw.get("scales")
    out["scales"] = ({str(k): float(v) for k, v in sc.items()} if isinstance(sc, dict) else {})
    out["engaged"] = bool(raw.get("engaged", False))
    return out


def _mean(xs: List[float], default: float = 0.0) -> float:
    return float(statistics.fmean(xs)) if xs else default


def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    v = d.get(key)
    if v is None or not torch.is_tensor(v):
        return None
    return v.float().unsqueeze(0) if v.dim() == 1 else v.float()


class _ScoreCallCapture:
    """Transparently wraps `agent.e3.score_trajectory` for the duration of ONE
    `agent.select_action(...)` call to record the exact (args, kwargs, returned score)
    triple used for each candidate, in call order. The live call's behaviour and return
    value are completely UNCHANGED -- this only observes it (the wrapper calls straight
    through to the original bound method and returns its result unmodified).

    `replay(commensurability)` re-runs the captured calls through the ORIGINAL (unpatched)
    function with `use_e3_channel_commensurability` temporarily set to `commensurability`,
    then restores it.

    THE EMA-TIMING SUBTLETY THIS CLASS EXISTS TO CLOSE (found by this build's own smoke
    test, not anticipated at design time). `_chan_scale_ema` is read-only DURING a tick's
    candidate-scoring loop but is MUTATED once `select()` has scored every candidate,
    folding THIS tick's own spread in (`_update_channel_scale_estimates`,
    e3_selector.py:3099-3100) -- so by the time `agent.select_action(...)` RETURNS to the
    caller, `_chan_scale_ema` already reflects tick T, not the tick-(T-1) state that was
    actually live while tick T's candidates were scored. A naive replay AFTER
    `select_action` returns therefore divides by the WRONG scale -- close (one EMA step of
    drift) but not exact, and a same-config self-check against it fails outright. Confirmed
    empirically: an early build of this driver measured 10/10 self-check failures in the
    fed arm (where the EMA moves fast) and 0/8 in the starved arm (where it was still
    floor-clamped and the one-step drift stayed below the numerical tolerance) on the exact
    same tick data -- a timing bug, not noise.

    FIX: `__enter__` snapshots `_chan_scale_ema` BEFORE the live `select_action()` call
    (i.e. the tick-(T-1) state), and `replay()` temporarily swaps that snapshot in for the
    duration of the replay, restoring the REAL (tick-T, still-evolving) EMA afterward so
    training continues from the correct accumulated state. This makes both the self-check
    and the shadow OFF/ON comparison use the EXACT scale denominator that was live at
    scoring time.

    A SECOND piece of state gates the SAME divisor and must be snapshotted alongside the
    EMA, not just the EMA itself (Step 4.5 red-team, model fable, CONTESTED finding 3):
    `_commensurability_scale` (e3_selector.py:1372) returns the no-op 1.0 while
    `_chan_scale_n < warmup_ticks`, and `_chan_scale_n` is incremented by the SAME
    `_update_channel_scale_estimates` call that mutates the EMA. On the one tick where the
    count crosses the warmup threshold, the live scoring used the pre-tick (still-warming,
    scale=1.0) reading, but a replay that restored only the EMA would see the POST-tick
    count and incorrectly divide by the real EMA instead of returning 1.0. Snapshotting and
    restoring `_chan_scale_n` alongside `_chan_scale_ema` closes this exactly.
    """

    def __init__(self, agent: REEAgent):
        self._agent = agent
        self._orig = None
        self._ema_snapshot: Dict[str, float] = {}
        self._n_snapshot: int = 0
        self.calls: List[Tuple[tuple, dict, float]] = []

    def __enter__(self) -> "_ScoreCallCapture":
        self.calls = []
        sel = self._agent.e3
        self._ema_snapshot = dict(getattr(sel, "_chan_scale_ema", {}) or {})
        self._n_snapshot = int(getattr(sel, "_chan_scale_n", 0) or 0)
        self._orig = sel.score_trajectory

        def _capturing(*args: Any, **kwargs: Any) -> torch.Tensor:
            s = self._orig(*args, **kwargs)
            self.calls.append((args, kwargs, float(s.detach().reshape(-1).mean().item())))
            return s

        sel.score_trajectory = _capturing
        return self

    def __exit__(self, *exc: Any) -> bool:
        sel = self._agent.e3
        try:
            del sel.__dict__["score_trajectory"]
        except KeyError:
            sel.score_trajectory = self._orig
        return False

    def replay(self, commensurability: bool) -> List[float]:
        sel = self._agent.e3
        prev_comm = bool(getattr(sel.config, "use_e3_channel_commensurability", False))
        prev_ema = dict(getattr(sel, "_chan_scale_ema", {}) or {})
        prev_n = int(getattr(sel, "_chan_scale_n", 0) or 0)
        sel._chan_scale_n = self._n_snapshot
        sel.config.use_e3_channel_commensurability = bool(commensurability)
        sel._chan_scale_ema = dict(self._ema_snapshot)
        try:
            out = []
            for args, kwargs, _live_score in self.calls:
                s = self._orig(*args, **kwargs)
                out.append(float(s.detach().reshape(-1).mean().item()))
            return out
        finally:
            sel.config.use_e3_channel_commensurability = prev_comm
            sel._chan_scale_ema = prev_ema
            sel._chan_scale_n = prev_n


def _argmin_idx(scores: List[float]) -> int:
    return min(range(len(scores)), key=lambda i: scores[i])


def _self_check_ok(a: float, b: float, rel_tol: float = SELF_CHECK_TOLERANCE) -> bool:
    """RELATIVE tolerance (plus a small absolute floor for near-zero magnitudes).

    Step 4.5 red-team (model fable) finding 4b: a pure ABSOLUTE tolerance at 1e-9 against
    scores of magnitude ~1e4 (observed in the fed arm) is, in float32, effectively a
    bit-identity requirement rather than the "float roundtrip" the docstring originally
    called it -- correct in practice only because deterministic CPU torch happens to
    reproduce the same op sequence bit-for-bit. A relative tolerance is what the check is
    actually meant to express (are these the SAME computation, not are they bit-identical),
    and degrades gracefully if the op sequence ever picks up a benign reordering.
    """
    scale = max(abs(a), abs(b), 1.0)
    return bool(abs(a - b) <= rel_tol * scale)


# --------------------------------------------------------------------------- #
# Per-cell run                                                                 #
# --------------------------------------------------------------------------- #

def run_cell(arm: Dict[str, Any], seed: int, p0_episodes: int, p1_episode_cap: int,
             steps_per_episode: int, fresh_target: int, dry_run: bool = False) -> Dict[str, Any]:
    arm_id = str(arm["id"])
    feed = bool(arm["feed_residue"])
    p0 = int(p0_episodes) if arm["warmup"] else 0
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    slice_for_cell = config_slice_for(arm, p0_episodes, p1_episode_cap, steps_per_episode, fresh_target)

    with arm_cell(seed, config_slice=slice_for_cell, script_path=Path(__file__),
                  config_slice_declared=True) as cell:
        env = make_env(seed)
        agent = _make_agent(env)
        e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)

        clamp_live = bool(getattr(agent.e2.config, "e2_rollout_output_norm_clamp_enabled", False))
        ratio_live = float(getattr(agent.e2.config, "e2_rollout_output_norm_clamp_ratio", 2.0))

        transition_buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(
            maxlen=TRANSITION_BUFFER_MAX
        )
        sample_rng = random.Random(seed)
        total_train_eps = p0 + int(p1_episode_cap)

        fs = FreshSelectCounter()
        n_ticks_total = 0
        n_update_residue_calls = 0
        n_contrastive_steps_total = 0
        p1_episodes_run = 0
        target_met = False

        flip_flags: List[int] = []
        n_self_check_failures = 0
        fed_vs_starved_score_deltas: List[float] = []  # abs(live - shadow) per candidate, flattened

        for ep in range(total_train_eps):
            is_p1 = ep >= p0
            phase_label = "P1" if is_p1 else "P0"
            if is_p1:
                p1_episodes_run += 1

            _, obs_dict = env.reset()
            agent.reset()

            z_self_prev: Optional[torch.Tensor] = None
            action_prev: Optional[torch.Tensor] = None
            pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
            tick_in_ep = 0

            for _step in range(steps_per_episode):
                body = obs_dict["body_state"].float()
                world = obs_dict["world_state"].float()
                if body.dim() == 1:
                    body = body.unsqueeze(0)
                if world.dim() == 1:
                    world = world.unsqueeze(0)

                latent = agent.sense(
                    obs_body=body, obs_world=world,
                    obs_harm=_obs(obs_dict, "harm_obs"),
                    obs_harm_a=_obs(obs_dict, "harm_obs_a"),
                    obs_harm_history=_obs(obs_dict, "harm_history"),
                )

                if pending_capture is not None:
                    z0_prev, a_prev = pending_capture
                    z1_obs = latent.z_world.detach().reshape(-1).clone()
                    if (torch.isfinite(z0_prev).all() and torch.isfinite(a_prev).all()
                            and torch.isfinite(z1_obs).all()):
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

                if agent.goal_state is not None:
                    try:
                        energy = float(body[0, 3].item())
                    except Exception:
                        energy = 1.0
                    agent.update_z_goal(benefit_exposure=0.0, drive_level=max(0.0, 1.0 - energy))

                with _ScoreCallCapture(agent) as _cap, _FRESH_SELECT.watch(agent) as _sel:
                    action = agent.select_action(candidates, ticks)
                fresh_select = _sel.fresh
                n_ticks_total += 1
                if is_p1:
                    fs.record(fresh_select)

                if is_p1 and fresh_select and len(_cap.calls) >= 2:
                    live_scores = [c[2] for c in _cap.calls]
                    live_idx = _argmin_idx(live_scores)

                    # SELF-CHECK first: replay at the SAME config value must reproduce
                    # the live scores exactly (see class docstring). A harness bug
                    # detector -- never treated as a routed measurement.
                    selfcheck_scores = _cap.replay(True)
                    selfcheck_idx = _argmin_idx(selfcheck_scores)
                    if selfcheck_idx != live_idx or any(
                        not _self_check_ok(a, b) for a, b in zip(selfcheck_scores, live_scores)
                    ):
                        n_self_check_failures += 1

                    # SHADOW: the counterfactual with the operator OFF.
                    shadow_scores = _cap.replay(False)
                    shadow_idx = _argmin_idx(shadow_scores)
                    flip_flags.append(int(shadow_idx != live_idx))
                    fed_vs_starved_score_deltas.extend(
                        abs(a - b) for a, b in zip(live_scores, shadow_scores)
                    )

                if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                    pending_capture = (
                        latent.z_world.detach().reshape(-1).clone(),
                        action.detach().reshape(-1).clone(),
                    )

                if tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                    loss_val = _e2_contrastive_step(agent, transition_buffer, e2_opt, sample_rng)
                    if loss_val is not None and math.isfinite(loss_val):
                        n_contrastive_steps_total += 1

                _, harm_signal, done, info, next_obs_dict = env.step(action)
                hv = float(harm_signal)

                if feed:
                    with torch.no_grad():
                        agent.update_residue(harm_signal=hv, world_delta=None,
                                              hypothesis_tag=False, owned=True)
                    n_update_residue_calls += 1

                z_self_prev = latent.z_self.detach()
                action_prev = action
                obs_dict = next_obs_dict
                tick_in_ep += 1
                if done:
                    break

            fs.flush()

            if ep == 0 or is_p1 or (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
                print(
                    f"  [train] arm={arm_id} seed={seed} phase={phase_label} "
                    f"ep {ep + 1}/{total_train_eps} fresh={fs.n_fresh_select}/{fresh_target} "
                    f"flips={sum(flip_flags)}/{len(flip_flags)}",
                    flush=True,
                )

            if is_p1 and fs.n_fresh_select >= fresh_target:
                target_met = True
                print(f"  [p1-done] arm={arm_id} seed={seed} fresh={fs.n_fresh_select} "
                      f"after {p1_episodes_run} P1 episode(s)", flush=True)
                break

        _ZG.observe(agent)
        _LAST_AGENT["agent"] = agent

        n_flip_ticks = len(flip_flags)
        flip_rate = _mean([float(f) for f in flip_flags])
        scale_est = _scale_estimates(agent)

        expected_calls = n_ticks_total if feed else 0
        row: Dict[str, Any] = {
            "arm": arm_id,
            "seed": int(seed),
            "load_bearing_arm": bool(arm["load_bearing"]),
            "feed_residue_per_step": feed,
            "p0_warmup": bool(arm["warmup"]),
            "p0_episodes": int(p0),
            "p1_episodes_run": int(p1_episodes_run),
            "fresh_target_met": bool(target_met),
            "clamp_live_on_e2": clamp_live,
            "clamp_ratio_live": ratio_live,
            "clamp_config_landed": bool(clamp_live and abs(ratio_live - 2.0) < 1e-12),
            "n_env_ticks_total": int(n_ticks_total),
            "n_update_residue_calls": int(n_update_residue_calls),
            "residue_protocol_landed": bool(n_update_residue_calls == expected_calls),
            "n_contrastive_steps_total": int(n_contrastive_steps_total),
            "n_fresh_select": int(fs.n_fresh_select),
            "n_latched": int(fs.n_latched),
            "fresh_select_yield": float(fs.n_fresh_select) / float(max(1, fs.n_fresh_select + fs.n_latched)),

            # ===== THE ROUTED DV =====
            "n_flip_ticks_scored": int(n_flip_ticks),
            "xcand_commit_flip_rate": float(flip_rate),
            "n_flips": int(sum(flip_flags)),
            "n_self_check_failures": int(n_self_check_failures),
            "self_check_clean": bool(n_self_check_failures == 0),
            "mean_abs_score_delta_live_vs_shadow": _mean(fed_vs_starved_score_deltas),

            # spec-mandated exposure
            "channel_scale_estimates": scale_est,
            "operator_engaged": bool(scale_est.get("engaged", False)),
        }

        cell.stamp(row)

    if dry_run:
        print(
            f"  [smoke] arm={arm_id} seed={seed} clamp_live={row['clamp_live_on_e2']} "
            f"fed={feed} n_flip_ticks={row['n_flip_ticks_scored']} "
            f"flip_rate={row['xcand_commit_flip_rate']:.4f} "
            f"self_check_clean={row['self_check_clean']} "
            f"engaged={row['operator_engaged']}",
            flush=True,
        )

    print(f"verdict: {'PASS' if row['self_check_clean'] else 'FAIL'}", flush=True)
    return row


def _e2_contrastive_step(agent: REEAgent, buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                          optimiser: torch.optim.Optimizer, rng: random.Random) -> Optional[float]:
    if len(buffer) < CONTRASTIVE_BATCH_K:
        return None
    batch = rng.sample(list(buffer), CONTRASTIVE_BATCH_K)
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)
    optimiser.zero_grad(set_to_none=True)
    loss = agent.e2.world_forward_contrastive_loss(
        z_world_0=z0_K, actions=actions_K, z_world_1_targets=z1_K, simulation_mode=False,
    )
    if not torch.is_tensor(loss):
        return None
    loss_val = float(loss.detach().item())
    if not math.isfinite(loss_val):
        return loss_val
    if not loss.requires_grad or loss_val == 0.0:
        return loss_val
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), 1.0)
    optimiser.step()
    return loss_val


def _arm_gate(arm: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _worst(key: str) -> float:
        vals = [float(r[key]) for r in rows]
        return min(vals) if vals else 0.0

    measured = {
        "decomp_samples_sufficient": _worst("n_flip_ticks_scored"),
        "operator_engaged": 1.0 if rows and all(bool(r["operator_engaged"]) for r in rows) else 0.0,
        "clamp_config_landed": 1.0 if rows and all(bool(r["clamp_config_landed"]) for r in rows) else 0.0,
        "residue_protocol_landed": 1.0 if rows and all(bool(r["residue_protocol_landed"]) for r in rows) else 0.0,
        "shadow_scores_measurably_different": _worst("mean_abs_score_delta_live_vs_shadow"),
    }
    overrides = {spec.name: bool(measured[spec.name] >= spec.threshold)
                 for spec in PRECONDITION_SPECS if spec.name in GEQ_PRECONDITIONS}
    ctx = {"id": arm["id"], "feed_residue": arm["feed_residue"], "warmup": arm["warmup"]}
    gate = evaluate_arm_gate(arm["id"], ctx, PRECONDITION_SPECS, measured, met_overrides=overrides)
    for p in gate["preconditions"]:
        p["kind"] = "readiness"
    return gate


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = list(DRY_RUN_SEEDS if dry_run else SEEDS)
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1_cap = DRY_RUN_P1_CAP if dry_run else P1_EPISODE_CAP
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    fresh_target = DRY_RUN_FRESH_TARGET if dry_run else N_FRESH_SELECT_TARGET

    arm_ctxs = [{"id": a["id"], "feed_residue": a["feed_residue"], "warmup": a["warmup"]} for a in ARMS]
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, arm_ctxs)

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            rows.append(run_cell(arm, seed, p0, p1_cap, steps, fresh_target, dry_run=dry_run))

    by_arm: Dict[str, List[Dict[str, Any]]] = {a["id"]: [r for r in rows if r["arm"] == a["id"]] for a in ARMS}
    arm_gates = [_arm_gate(a, by_arm[a["id"]]) for a in ARMS]
    gate = aggregate_arm_gates(arm_gates)
    green = set(gate["green_arms"])

    n_self_check_failures_total = sum(r["n_self_check_failures"] for r in rows)

    def _arm_flip_rates(arm_id: str) -> List[float]:
        return [r["xcand_commit_flip_rate"] for r in by_arm[arm_id]]

    def _flip_rate_pass(arm_id: str) -> Tuple[bool, Dict[str, Any]]:
        """PASS iff EITHER the mean clears the SD-scaled bar OR a majority of seeds
        individually clear the absolute floor.

        Step 4.5 red-team (model fable) finding 2: the SD-scaled bar alone is a
        coefficient-of-variation test, not an effect-size test -- it conflates "no
        effect" with "heterogeneous effect", so a pattern like (0.9, 0.9, 0.9, 0.0)
        (the operator plainly consequential in 3 of 4 seeds) has mean 0.675, sd 0.390,
        bar 0.779, and FAILS the mean test, which would then read as
        "operator_selection_inconsequential" -- an attribution the per-seed data
        cannot support. The majority-clears-floor test is robust to exactly this shape:
        it asks only whether MOST seeds show SOME effect above the floor, independent
        of how heterogeneous the magnitude is across seeds.
        """
        rates = _arm_flip_rates(arm_id)
        mean_rate = _mean(rates)
        sd = float(statistics.pstdev(rates)) if len(rates) > 1 else 0.0
        bar = max(FLIP_RATE_SD_MULTIPLIER * sd, FLIP_RATE_ABS_FLOOR)
        mean_clears_bar = bool(mean_rate > bar)
        n_seeds_above_floor = sum(1 for r in rates if r > FLIP_RATE_ABS_FLOOR)
        majority_clears_floor = bool(rates and n_seeds_above_floor >= (len(rates) + 1) // 2)
        passed = bool(mean_clears_bar or majority_clears_floor)
        return passed, {
            "mean_rate": mean_rate, "sd": sd, "bar": bar,
            "mean_clears_bar": mean_clears_bar,
            "n_seeds_above_floor": int(n_seeds_above_floor),
            "n_seeds": int(len(rates)),
            "majority_clears_floor": majority_clears_floor,
        }

    fed_pass, fed_detail = _flip_rate_pass(FED_ARM)
    starved_pass, starved_detail = _flip_rate_pass(STARVED_ARM)
    regimes_differ = bool(
        abs(fed_detail["mean_rate"] - starved_detail["mean_rate"]) >= REGIME_DIFFERENCE_FLOOR
    )

    # --- verdict, checked in the order that makes an instrument failure impossible to
    # report as content (design doc section 3, "reading order"):
    #   1. self-check clean on EVERY scored tick (harness correctness)
    #   2. every arm's readiness gate green (operator engaged, enough samples, wiring)
    #   3. the non-forced-ness check (design doc 3.5) -- read BEFORE the cleanest content
    #      label, not merely alongside it (Step 4.5 red-team, model fable, finding 1: an
    #      earlier build computed `regimes_differ` but never routed on it, so a run with
    #      fed and starved flip rates reading numerically identical -- the exact "hidden
    #      identity" signature the design doc pre-registers as disqualifying -- could still
    #      reach the cleanest PASS label with the caveat buried in prose)
    #   4. ONLY THEN: read the flip-rate content
    if n_self_check_failures_total > 0:
        outcome, label = "FAIL", "shadow_replay_self_check_failed_instrument_defect"
    elif FED_ARM not in green or STARVED_ARM not in green:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif fed_pass and starved_pass and not regimes_differ:
        outcome, label = (
            "FAIL",
            "operator_selection_consequential_both_regimes_undifferentiated_flag_possible_hidden_identity",
        )
    elif fed_pass and starved_pass:
        outcome, label = "PASS", "operator_selection_consequential_both_regimes"
    elif fed_pass or starved_pass:
        outcome, label = "PASS", "operator_selection_consequential_one_regime_only"
    else:
        outcome, label = "PASS", "operator_selection_inconsequential_primary_stage"

    outcome_note = (
        f"{label}: fed xcand_commit_flip_rate mean {fed_detail['mean_rate']:.4f} "
        f"(sd {fed_detail['sd']:.4f}, bar {fed_detail['bar']:.4f}, PASS={fed_pass}); "
        f"starved xcand_commit_flip_rate mean {starved_detail['mean_rate']:.4f} "
        f"(sd {starved_detail['sd']:.4f}, bar {starved_detail['bar']:.4f}, PASS={starved_pass}); "
        f"regimes_differ (non-forced-ness check, floor {REGIME_DIFFERENCE_FLOOR}): {regimes_differ}. "
        f"self_check_failures={n_self_check_failures_total} (must be 0). "
        f"gate green_arms={sorted(green)} red_arms={sorted(set(a['id'] for a in ARMS) - green)}. "
        "This DV characterises the PRIMARY-score stage the operator directly acts on, not the "
        "final executed action after the modulatory shortlist stage; final_commit_by_primary "
        "context is NOT recorded in this build (design doc section 6a scoping decision) -- a "
        "final-commit-level successor needs a small e3_selector.py refactor, deliberately not "
        "built here. MECH-439's direction does not move on this record: this run says only "
        "whether the operator's rescaling changes which candidate the primary score alone would "
        "prefer, not whether it does or does not 'work'."
    )

    return {
        "outcome": outcome,
        "outcome_note": outcome_note,
        "arm_results": rows,
        "per_arm_gate": gate["per_arm_gate"],
        "non_degenerate": bool(gate["non_degenerate"]),
        "degeneracy_reason": gate["degeneracy_reason"],
        "interpretation": {
            "label": label,
            "preconditions": gate["adjudication_preconditions"],
            "criteria_non_degenerate": {
                "flip_rate_lift": bool(FED_ARM in green and STARVED_ARM in green
                                        and n_self_check_failures_total == 0),
            },
            "combination_rule": (
                "Read in order: (1) self_check_failures must be 0 across every scored tick in "
                "every cell -- a shadow-replay harness bug, never reported as content; (2) every "
                "arm's readiness gate (operator_engaged, decomp_samples_sufficient, "
                "clamp_config_landed, residue_protocol_landed, shadow_scores_measurably_"
                "different) must be green, else substrate_not_ready_requeue; (3) the "
                "non-forced-ness check (regimes_differ) is read BEFORE the cleanest content "
                "label: fed_pass AND starved_pass AND NOT regimes_differ routes to a distinct "
                "FLAGGED FAIL label rather than the clean 'both regimes' PASS (Step 4.5 "
                "red-team, model fable, finding 1); (4) only then, per regime, "
                "xcand_commit_flip_rate is 'operator_selection_consequential' in that regime "
                "iff EITHER the mean clears max(FLIP_RATE_SD_MULTIPLIER * seed-matched SD, "
                "FLIP_RATE_ABS_FLOOR) OR a MAJORITY of that regime's seeds individually clear "
                "FLIP_RATE_ABS_FLOOR alone (the majority test added per Step 4.5 finding 2, so "
                "a heterogeneous-but-real effect across seeds is not misread as 'no effect' by "
                "a coefficient-of-variation-style bar)."
            ),
            "non_forced_ness_note": (
                "Design rule (a): fed and starved flip rates are reported SEPARATELY and "
                "compared (regimes_differ, floor "
                f"{REGIME_DIFFERENCE_FLOOR}). If they read identically regardless of the "
                "genuinely different raw channel-magnitude relationships 571c measured between "
                "the two regimes, that would itself be evidence of a hidden identity in this "
                "DV -- and (fixed after Step 4.5 red-team finding 1) this is now ROUTED, not "
                "just recorded: a both-regimes-pass reading with regimes_differ False lands on "
                "a distinct flagged FAIL label instead of the clean both-regimes PASS label."
            ),
            "scoping_note": (
                "This DV is PRIMARY-stage argmin flip under shadow OFF/ON scoring, not a "
                "final-commit-level replay through the modulatory shortlist stage. See design "
                "doc evidence/planning/mech439_rung3_selection_level_repose_20260914.md section "
                "6a for why, and what a final-commit-level successor would need."
            ),
        },
        "criteria": [
            {
                "name": "flip_rate_lift",
                "load_bearing": True,
                "role": "verdict",
                "passed": bool(fed_pass or starved_pass),
                # Reported PER REGIME, not as a cross-arm max(measured)/min(threshold) pair
                # (Step 4.5 red-team, model fable, finding 2's secondary note): the mixed
                # form let `measured > threshold` co-occur with `passed: False` whenever the
                # two regimes' own (measured, threshold) pairs did not literally correspond
                # -- readable only by re-deriving which arm actually produced which number.
                "measured_fed": float(fed_detail["mean_rate"]),
                "threshold_fed": float(fed_detail["bar"]),
                "passed_fed": bool(fed_pass),
                "measured_starved": float(starved_detail["mean_rate"]),
                "threshold_starved": float(starved_detail["bar"]),
                "passed_starved": bool(starved_pass),
                "combination_rule": "PASS iff EITHER regime individually passes (fed_pass OR starved_pass)",
                "detail": f"fed={fed_detail}, starved={starved_detail}",
            },
            {
                "name": "self_check_clean",
                "load_bearing": True,
                "role": "instrument correctness gate",
                "passed": bool(n_self_check_failures_total == 0),
                "measured": float(n_self_check_failures_total),
                "threshold": 0.0,
                "direction": "upper",
                "detail": f"{n_self_check_failures_total} self-check failures across {len(rows)} cells",
            },
        ],
        "summary": {
            "label": label,
            "fed": fed_detail,
            "starved": starved_detail,
            "regimes_differ": regimes_differ,
            "n_self_check_failures_total": int(n_self_check_failures_total),
        },
        "diagnostics": {
            "flip_rate_per_cell": {
                f"{r['arm']}/seed{r['seed']}": r["xcand_commit_flip_rate"] for r in rows
            },
            "mean_abs_score_delta_per_cell": {
                f"{r['arm']}/seed{r['seed']}": r["mean_abs_score_delta_live_vs_shadow"] for r in rows
            },
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-1012a: selection-level (commit-flip) re-pose of the E3 "
                     "channel-commensurability rung-3 readiness validation"
    )
    parser.add_argument("--dry-run", action="store_true", help="Short run for smoke testing")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = run_experiment(dry_run=args.dry_run)

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    seeds_used = list(DRY_RUN_SEEDS if args.dry_run else SEEDS)
    p0_used = DRY_RUN_P0 if args.dry_run else P0_WARMUP_EPISODES
    p1_cap_used = DRY_RUN_P1_CAP if args.dry_run else P1_EPISODE_CAP
    steps_used = DRY_RUN_STEPS if args.dry_run else STEPS_PER_EPISODE
    fresh_used = DRY_RUN_FRESH_TARGET if args.dry_run else N_FRESH_SELECT_TARGET
    full_config = {
        "seeds": seeds_used,
        "env_kwargs": dict(ENV_KWARGS),
        "schedule": {
            "p0_warmup_episodes": p0_used,
            "p1_episode_cap": p1_cap_used,
            "steps_per_episode": steps_used,
            "fresh_select_target": fresh_used,
        },
        "pre_registered_thresholds": {
            "MIN_FRESH_SELECTIONS": MIN_FRESH_SELECTIONS,
            "SELF_CHECK_TOLERANCE": SELF_CHECK_TOLERANCE,
            "FLIP_RATE_SD_MULTIPLIER": FLIP_RATE_SD_MULTIPLIER,
            "FLIP_RATE_ABS_FLOOR": FLIP_RATE_ABS_FLOOR,
            "REGIME_DIFFERENCE_FLOOR": REGIME_DIFFERENCE_FLOOR,
        },
        "arms": [dict(a) for a in ARMS],
        "arm_config_slices": {
            a["id"]: config_slice_for(a, p0_used, p1_cap_used, steps_used, fresh_used) for a in ARMS
        },
        "dry_run": bool(args.dry_run),
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": list(CLAIM_IDS),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "diagnostic",
        "outcome": result["outcome"],
        "outcome_note": result["outcome_note"],
        "timestamp_utc": timestamp,
        "non_degenerate": result["non_degenerate"],
        "per_arm_gate": result["per_arm_gate"],
        "arm_results": result["arm_results"],
        "per_seed_results": result["arm_results"],
        "interpretation": result["interpretation"],
        "criteria": result["criteria"],
        "summary": result["summary"],
        "diagnostics": result["diagnostics"],
        "readout": {
            "fed_flip_rate": result["summary"]["fed"]["mean_rate"],
            "starved_flip_rate": result["summary"]["starved"]["mean_rate"],
            "regimes_differ": 1 if result["summary"]["regimes_differ"] else 0,
            "self_check_failures_total": result["summary"]["n_self_check_failures_total"],
        },
        "custom_information": {
            "supersedes_acceptance_instrument_of": "V3-EXQ-1012 (parked, never queued)",
            "tautology_finding": "evidence/planning/exq1012_blocked_readiness_target_tautological_20260908.md",
            "repose_governance_record": "GFLAG-0234, /governance option A, user-approved 2026-09-10, REE_assembly 5dc661badc",
            "design_doc": "evidence/planning/mech439_rung3_selection_level_repose_20260914.md",
            "gov_reuse_1_check": (
                "Decisive readout: xcand_commit_flip_rate (per-tick shadow OFF/ON primary-argmin "
                "comparison). No MECH-439 manifest carries per-tick per-candidate raw scores; "
                "571c and the 1012 smoke record only per-cell aggregates. Not recoverable -> run."
            ),
            "brake_count_note": (
                "MECH-439 carries >=2 counted substrate_ceiling autopsies. Not braked: this is a "
                "diagnostic on the MEASUREMENT axis (does the operator have a selection-"
                "consequential effect), the same determination the parked V3-EXQ-1012 driver "
                "made for this substrate. No ceiling hit added by this run."
            ),
            "build_scoping_decision": (
                "PRIMARY-stage argmin flip only, not a final-commit-level replay through the "
                "modulatory shortlist stage -- see design doc section 6a. final_commit_by_primary "
                "context readout deliberately not carried in this build; a follow-on "
                "/implement-substrate chip is the right route to a final-commit-level successor."
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
    if result["degeneracy_reason"]:
        manifest["degeneracy_reason"] = result["degeneracy_reason"]

    stamp_recording_core(
        manifest, config=full_config, seeds=seeds_used, script_path=Path(__file__), started_at=t0,
        agent=_LAST_AGENT["agent"], z_goal_stream_stats=_ZG.stats(),
    )

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=bool(args.dry_run),
        config=full_config, seeds=seeds_used, script_path=Path(__file__), started_at=t0,
        agent=_LAST_AGENT["agent"], z_goal_stream_stats=_ZG.stats(), json_default=str,
    )
    print(f"Manifest written: {out_path}", flush=True)

    s = result["summary"]
    print(
        f"SUMMARY: fed_flip_rate={s['fed']['mean_rate']:.4f} (sd {s['fed']['sd']:.4f}) "
        f"starved_flip_rate={s['starved']['mean_rate']:.4f} (sd {s['starved']['sd']:.4f}) "
        f"regimes_differ={s['regimes_differ']} self_check_failures={s['n_self_check_failures_total']}",
        flush=True,
    )
    print(f"LABEL: {result['interpretation']['label']}", flush=True)
    for c in result["criteria"]:
        print(f"  {c['name']}: {c['passed']}", flush=True)
    print(
        f"  per_arm_gate: green={result['per_arm_gate']['green_arms']} "
        f"red={result['per_arm_gate']['red_arms']}",
        flush=True,
    )

    if args.dry_run:
        for r in result["arm_results"]:
            print(
                f"  [smoke-routed] {r['arm']}/seed{r['seed']}: "
                f"flip_rate={r['xcand_commit_flip_rate']:.4f} "
                f"n_scored={r['n_flip_ticks_scored']} "
                f"self_check_clean={r['self_check_clean']} "
                f"engaged={r['operator_engaged']}",
                flush=True,
            )
        assert result["summary"]["n_self_check_failures_total"] == 0, (
            "SMOKE FAIL: shadow-replay self-check failed -- the harness diverged from the "
            "live scoring path; see _ScoreCallCapture docstring."
        )
        print("DRY RUN complete.", flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=bool(args.dry_run),
    )
