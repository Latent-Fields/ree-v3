"""V3-EXQ-1012c -- E3 channel-commensurability rung-3 validation at the ELIGIBILITY stage
(substrate f_dominance_conversion_ceiling rung 3 / SD-E3-CHANNEL-COMMENSURABILITY; claim
tag MECH-439, non_contributory on every branch -- this is NOT a MECH-439 test).

SLEEP DRIVER: none (no sleep flags set; 936-regime parity with V3-EXQ-571c / V3-EXQ-1012a).

red-team (fable, Step 4.5, run on the design spec before authoring): CONTESTED, 6 findings,
all dispositioned in the design doc -- F1 identity instrument gate replaced by a residual gate
plus the 1012a replay self-check plus an independent OFF-replay cross-check; F2 float32 cutoff
rounding handled by reproducing the live margin rule in torch float32 and gating on a mismatch
RATE; F3 verdict grid made a partition with stated precedence; F4 PASS prior stated, ORACLE and
ON share one content-tick set, shape + r_c(t) promoted to reported readouts; F5 starved OFF gap
corrected (4-25x, not 1e3-1e5), P1 floor kept at 2 for byte-identity with 571c/1012a; F6 I3
kept as a recorded consistency count only.

=== WHY THIS DRIVER EXISTS (GFLAG-0297) ===

V3-EXQ-1012a measured the PRIMARY-argmin flip rate (shadow operator-OFF vs live ON): 0.82 fed /
0.72 starved, PASS, adjudicated non_contributory -- no reference could grade the number. Its
routed successor V3-EXQ-1012b (a label-permutation placebo of the operator's own channel
scales) was REFUSED: with scales spanning ~4 orders of magnitude every derangement keeps the
dominant channel on top, so the placebo cannot discriminate by construction.

Design record (why ANY primary-stage null is non-validating, and the derivation behind this
driver): REE_assembly/evidence/planning/gflag0297_mech439_rung3_null_design.md. In short:
  * at the primary stage "commensurable" has no referent outside the operator's own
    definition (equal cross-candidate SD) -- a unit rescale is cancelled exactly, a divisor
    jitter is decided by its sigma knob, a bootstrap/tick-shuffle null asks only whether the
    running estimate is current, and per-channel argmin agreement is entailed;
  * in this regime the primary score reaches the EXECUTED action ONLY through membership of
    the margin-eligible set: e3_selector.py select() builds
        E = { i : raw_i <= min(raw) + modulatory_shortlist_margin * (max(raw) - min(raw)) }
    and then picks within E by _modulatory_accum, with the primary score absent. V3-EXQ-571c
    (same regime) recorded modulatory_shortlist_active_frac 1.0 in all 16 cells, shortlist
    size 5-16 of k=32, final_commit_by_primary_frac 0.005-0.146.
The eligibility margin is RANGE-denominated while the operator equalises SD, so whether the
operator's equalisation survives to where the action is decided is NOT an identity: it depends
on each channel's realised per-tick spread relative to its EMA (r_c(t) = sd_c(t)/s_hat_c) and
on each channel's per-candidate SHAPE. That is what this driver measures.

=== THE MEASUREMENT ===

Two arms, operator ALWAYS ON (live): C_fed_operator_on, C_starved_operator_on; seeds
42/43/45/46; P0 60 warmup episodes, P1 to 200 genuine latch-gated selections (cap 40 eps x 200
steps). Byte-identical regime to 1012a (lineage baseline experiments/_lib/baselines/
mech439_f_variance_share.py; support_preserving_min_first_action_classes stays at the lineage
value 2 -- GFLAG-0072's precondition binds CLASS-ENTROPY DVs, whose ceiling is ln(~3); this DV
is candidate-level eligibility membership, which that ceiling does not touch, and raising it
would break the regime match to 571c/1012a that this design's motivation rests on).

At every genuine P1 tick, the live select() call is transparently captured (1012a's
_ScoreCallCapture, extended): per candidate i, the live score and the RAW per-channel terms
t[c,i] from E3TrajectorySelector._last_commensurability_raw (written per score_trajectory call,
read immediately after each call -- never once per tick, red-team F1). The operator divisor
m[c] is reproduced from the PRE-tick EMA snapshot with _commensurability_scale's exact
semantics (warmup / floor -> 1.0). Channel sign: f, harm, residue add; benefit, goal subtract.

  S_ON[i]     = sum_c sign_c t[c,i] / m[c]
  S_OFF[i]    = sum_c sign_c t[c,i]
  S_ORACLE[i] = sum_c sign_c t[c,i] / o[c](t),  o[c](t) = tick-local cross-candidate SD
                                                 (1.0 at/below the floor)
  E(S) = margin rule above, same margin as the live config.
  Knockout of channel c under scoring X: S_X minus channel c's (scaled, signed) term.

DV: for each channel live by EMA, J_c^X = mean over c's CONTENT ticks (sd_c(t) >= 0.1*s_hat_c;
the SAME tick set for X in {ON, OFF, ORACLE}) of the Jaccard distance between E(S_X) and
E(S_X with c knocked out). R^X = min_c J_c^X / max_c J_c^X over channels with >= 30 content
ticks (0.0 when every J is 0 -- no channel moves eligibility at all).

INSTRUMENT GATES (routed FIRST; any failure -> instrument_defect, never content):
  I1 residual: |live_i - S_ON[i]| <= 1e-5 * max(1, max_c |t[c,i]/m[c]|) on every scored tick
     (dual-system arbitration, pe-confidence and self-viability penalties are all off in this
     regime, so a non-zero residual IS a capture bug -- the per-tick-read / post-update-divisor
     classes red-team F1 named).
  I1b replay self-check (1012a): replay at the live config reproduces the live scores.
  I1c OFF cross-check: a genuine operator-OFF replay equals the reconstructed S_OFF -- an
     independent check that the captured terms are the terms the selector used.
  I2 |E_live| reconstructed in torch float32 from agent.e3.last_raw_scores with the selector's
     own ops equals the live modulatory_shortlist_size on >= 99% of shortlist-active ticks
     (red-team F2: exact equality at float32 score magnitudes ~1e3-1e4 is ULP-fragile).
  Recorded only (F6): the live selected index lies in E_live.

CRITERION C1 (single verdict criterion): per regime, PASS iff R^ON >= 0.25 in >= 3 of 4 seeds.
Anchor (Monte Carlo, k=32, margin 0.25, standardised channels): R = 0.998 at equal realised
spread; 0.41 / 0.26 / 0.135 at one channel 3x / 5x / 10x -- 0.25 ~ "no live channel runs more
than ~5x the others at the eligibility stage". PRIOR, stated per red-team F4: under Gaussian or
sparse-HIGH channel shapes at r ~ 1, PASS is the expected reading; it FAILS (gate-green) for a
sparse-LOW channel shape even at perfect SD equalisation (R 0.21-0.23), for sparse-HIGH at
r >= 3, or for a channel running >= ~5x its EMA. The EMA-lag failure route is close to
unreachable in this protocol (residue does not reset per episode; content-conditioned bursts
need duty > 0.97) -- recorded, not relied on.

VERDICT GRID (a partition; precedence instrument > readiness > per-regime class):
  instrument_defect                                         -> FAIL
  substrate_not_ready_requeue (BOTH arm gates red)          -> FAIL
  per regime, exactly one class (a red arm gate scopes ONLY that regime out):
    not_ready  that arm's readiness gate is red
    moot       R^OFF >= 0.25 in >= 3/4 seeds (no eligibility monopoly to remove; EXPECTED
               DEAD from the landed 1012a scales: R^OFF ~0.000 fed, ~0.06-0.08 starved)
    commensurate   not moot, C1 passes
    ema_lag        not moot, C1 fails, R^ORACLE >= 0.25 in >= 3/4 seeds
    shape_range    not moot, C1 fails, R^ORACLE fails the same 3/4 rule
  both regimes commensurate -> PASS commensurate_at_eligibility_both_regimes; any other
  combination -> FAIL, label "fed_<class>__starved_<class>".
evidence_direction non_contributory (+ note, + per-claim) on every branch. A PASS means the
operator's equalisation survives to the stage where the action is decided -- the release
condition f_dominance_conversion_ceiling.depends_on_unresolved[2] needs; governance decides the
release. A PASS includes f's micro-spread (relative ~1e-4 of its mean) amplified to parity by
the operator's design -- read it that way.

DV-SYMMETRY DECLARATION (both arms): R is invariant under a per-tick offset uniform across
candidates (E is shift-invariant: the cutoff moves with min) and under candidate permutation;
it is NOT invariant under a per-channel differential rescaling (the operator's own action) nor
under a change of a channel's per-candidate shape -- which is what it discriminates.

GOV-REUSE-1 (Step 2.4): no manifest records per-tick per-candidate per-channel terms (571c and
1012a record per-cell aggregates only) -> not recoverable, run.
RE-DERIVE BRAKE (Step 2.5b): MECH-439 is braked for re-tests of its hypothesis; this is
substrate validation on the measurement axis (the work the brake routes TO), same determination
as 1012a. No ceiling hit added.
STEP 2.5c: runs under open corrupting entry contextmemory-write-path-addressing-degeneracy
(e1_deep.py ContextMemory.write, live on waking steps; fix default-off) as a KNOWN LIMITATION:
it shapes the candidate inputs upstream of the stage under test, and rung 3 must be validated in
the regime the conversion falsifiers run in, which carries the same defect (as 1012a/571c did).
A later default flip of contextmemory_write_usage_balancing changes the regime.

EXPERIMENT_PURPOSE = "diagnostic". ASCII-only output (repo rule).
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
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

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
EXPERIMENT_TYPE = "v3_exq_1012c_e3_commensurability_eligibility_stage_validation"
QUEUE_ID = "V3-EXQ-1012c"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-439"]

SD056_ROLLOUT_CLAMP_EXEMPT = (
    "clamp set via CONFIG_FLAGS dict-splat in "
    "experiments/_lib/baselines/mech439_f_variance_share.py::make_agent_kwargs "
    "(936-regime parity with 571c/1012a); landing re-asserted per cell as clamp_config_landed"
)

CRITERIA_THRESHOLD_EXEMPT = (
    "every load-bearing criterion records numeric measured + threshold (C1 per regime: n seeds at bar vs SEEDS_REQUIRED, plus per_seed_measured vs per_seed_threshold R_BAR; instrument_clean: failure count vs 0) -- built in a loop in run_experiment(), which the AST scan does not resolve"
)

# --- Pre-registered thresholds (constants, never derived from this run) ------
MIN_FRESH_SELECTIONS = 60          # 936a/571b decomp-sample floor, reused
N_FRESH_SELECT_TARGET = 200        # 936a's P1 measurement budget
P1_EPISODE_CAP = 40
R_BAR = 0.25                       # C1 bar: ~5x realised dominance (MC anchor in docstring)
SEEDS_REQUIRED = 3                 # of 4 seeds per regime
CONTENT_FRAC = 0.1                 # channel has content on a tick iff sd_c(t) >= 0.1 * s_hat_c
MIN_CONTENT_TICKS = 30             # a channel enters R only with >= this many content ticks
RESIDUAL_REL_TOL = 1e-5            # I1 / I1c relative tolerance
SELF_CHECK_TOLERANCE = 1e-6        # I1b relative tolerance (1012a)
I2_MAX_MISMATCH_RATE = 0.01        # I2: float32 eligible-set size mismatch rate

# Channel -> sign in score_trajectory (score = f + harm + residue - benefit - goal).
CHANNEL_SIGN: Dict[str, float] = {
    "f_weighted": 1.0,
    "harm_weighted": 1.0,
    "residue_weighted": 1.0,
    "benefit_weighted": -1.0,
    "goal_weighted": -1.0,
}
CHANNELS: Tuple[str, ...] = tuple(CHANNEL_SIGN.keys())
SCORINGS: Tuple[str, ...] = ("ON", "OFF", "ORACLE")

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 4
DRY_RUN_P1_CAP = 2
DRY_RUN_STEPS = 60
DRY_RUN_FRESH_TARGET = 16
DRY_RUN_MIN_CONTENT_TICKS = 3   # smoke only: lets the smoke exercise R/J on few ticks

_ZG = ZGoalStreamAccumulator()
_LAST_AGENT: Dict[str, Any] = {"agent": None}
_FRESH_SELECT = FreshSelectProbe("exq1012c")

ARMS: List[Dict[str, Any]] = [
    {"id": "C_fed_operator_on", "feed_residue": True, "warmup": True, "load_bearing": True},
    {"id": "C_starved_operator_on", "feed_residue": False, "warmup": True, "load_bearing": True},
]
FED_ARM = "C_fed_operator_on"
STARVED_ARM = "C_starved_operator_on"


PRECONDITION_SPECS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="decomp_samples_sufficient",
        description="genuine (fresh, latch-gated) P1 selections scored -- the R denominator",
        control="worst seed of this arm",
        threshold=float(MIN_FRESH_SELECTIONS),
        direction="lower",
    ),
    PreconditionSpec(
        name="operator_engaged",
        description=(
            "channel_scale_estimates.engaged at cell end -- an arm that never left the "
            "operator's warmup has no manipulation in it"
        ),
        control="all cells of the arm",
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
        name="n_content_channels",
        description=(
            "channels live by EMA with >= MIN_CONTENT_TICKS content ticks -- R needs >= 2 "
            "(restores 571c's n_live_channels >= 2 gate, 1012a autopsy change 7). Does not "
            "read R, J, E sizes, r_c(t) or channel shape."
        ),
        control="worst seed of this arm",
        threshold=2.0,
        direction="lower",
    ),
]
GEQ_PRECONDITIONS = {s.name for s in PRECONDITION_SPECS}


def config_slice_for(arm: Dict[str, Any], p0_episodes: int, p1_cap: int, steps: int,
                      fresh_target: int) -> Dict[str, Any]:
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
            "r_bar": R_BAR,
            "content_frac": CONTENT_FRAC,
            "min_content_ticks": MIN_CONTENT_TICKS,
            "residual_rel_tol": RESIDUAL_REL_TOL,
            "self_check_tolerance": SELF_CHECK_TOLERANCE,
        },
    }


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """Lineage baseline agent with the commensurability operator forced ON on cfg.e3 (never
    through from_dims, which swallows unknown kwargs), verified live before any stepping."""
    cfg = REEConfig.from_dims(**make_agent_kwargs(env, OFF_ARM_FLAGS))
    cfg.e3.use_e3_channel_commensurability = True
    agent = REEAgent(cfg)
    if not bool(getattr(agent.e3.config, "use_e3_channel_commensurability", False)):
        raise RuntimeError("e3 channel-commensurability knob did not reach the agent")
    for flag in ("use_dualsystem_arbitration", "use_pe_confidence_weighting"):
        if bool(getattr(agent.e3.config, flag, False)):
            raise RuntimeError(
                f"{flag} is ON: the channel-sum reconstruction assumes it is off "
                "(it adds a non-channel term between the channel sum and raw_scores)")
    if str(getattr(agent.e3.config, "modulatory_shortlist_mode", "margin")) != "margin":
        raise RuntimeError("eligibility reconstruction assumes the margin shortlist mode")
    for flag in ("use_f_eligibility_demotion", "use_go_nogo_constitution"):
        if bool(getattr(agent.e3.config, flag, False)):
            raise RuntimeError(f"{flag} is ON: eligibility would not be the margin rule")
    return agent


def _scale_estimates(agent: Any) -> Dict[str, Any]:
    raw = getattr(getattr(agent, "e3", None), "last_channel_scale_estimates", None)
    if not isinstance(raw, dict):
        return {"present": False, "engaged": False, "n_updates": None, "scales": {}}
    out: Dict[str, Any] = {"present": True}
    for k in ("engaged", "n_updates", "warmup_ticks", "floor", "ema_alpha"):
        if k in raw:
            out[k] = raw[k]
    sc = raw.get("scales")
    out["scales"] = ({str(k): float(v) for k, v in sc.items()} if isinstance(sc, dict) else {})
    out["engaged"] = bool(raw.get("engaged", False))
    return out


def _mean(xs: Sequence[float], default: float = 0.0) -> float:
    return float(statistics.fmean(xs)) if xs else default


def _quantiles(xs: Sequence[float]) -> Dict[str, Optional[float]]:
    if not xs:
        return {"n": 0, "p10": None, "p50": None, "p90": None, "max": None}
    s = sorted(xs)
    n = len(s)

    def q(p: float) -> float:
        return float(s[min(n - 1, max(0, int(round(p * (n - 1)))))])

    return {"n": n, "p10": q(0.10), "p50": q(0.50), "p90": q(0.90), "max": float(s[-1])}


def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    v = d.get(key)
    if v is None or not torch.is_tensor(v):
        return None
    return v.float().unsqueeze(0) if v.dim() == 1 else v.float()


def _rel_close(a: float, b: float, scale: float, tol: float) -> bool:
    return bool(abs(a - b) <= tol * max(1.0, abs(scale)))


class _ScoreCallCapture:
    """1012a's capture harness, extended to record each candidate's RAW per-channel terms.

    Wraps agent.e3.score_trajectory for ONE select_action() call; the live behaviour and return
    value are unchanged. After EACH wrapped call it copies E3TrajectorySelector.
    _last_commensurability_raw -- a single dict the selector OVERWRITES per candidate, so it
    must be read per call, never once per tick (red-team F1). __enter__ snapshots
    _chan_scale_ema AND _chan_scale_n before the live call: select() folds this tick's own
    spread into the EMA after scoring, so a replay or divisor read after select_action()
    returns would otherwise use the tick-T divisor rather than the tick-(T-1) one that scored
    the tick (1012a's documented 10/10 self-check failure). replay(flag) re-runs the captured
    calls at use_e3_channel_commensurability=flag with the snapshot swapped in, then restores.
    """

    def __init__(self, agent: REEAgent):
        self._agent = agent
        self._orig = None
        self._ema_snapshot: Dict[str, float] = {}
        self._n_snapshot: int = 0
        self.calls: List[Tuple[tuple, dict, float]] = []
        self.raw_terms: List[Dict[str, float]] = []

    def __enter__(self) -> "_ScoreCallCapture":
        self.calls = []
        self.raw_terms = []
        sel = self._agent.e3
        self._ema_snapshot = dict(getattr(sel, "_chan_scale_ema", {}) or {})
        self._n_snapshot = int(getattr(sel, "_chan_scale_n", 0) or 0)
        self._orig = sel.score_trajectory

        def _capturing(*args: Any, **kwargs: Any) -> torch.Tensor:
            s = self._orig(*args, **kwargs)
            self.calls.append((args, kwargs, float(s.detach().reshape(-1).mean().item())))
            raw = getattr(sel, "_last_commensurability_raw", None) or {}
            self.raw_terms.append({c: float(raw.get(c, 0.0)) for c in CHANNELS})
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

    def divisors(self) -> Dict[str, float]:
        """_commensurability_scale's exact semantics, evaluated on the PRE-tick snapshot."""
        cfg = self._agent.e3.config
        warm = int(cfg.e3_commensurability_warmup_ticks)
        floor = float(cfg.e3_commensurability_floor)
        out: Dict[str, float] = {}
        for c in CHANNELS:
            if self._n_snapshot < warm:
                out[c] = 1.0
                continue
            s = float(self._ema_snapshot.get(c, 0.0))
            out[c] = s if s > floor else 1.0
        return out

    def replay(self, commensurability: bool) -> List[float]:
        sel = self._agent.e3
        prev_comm = bool(getattr(sel.config, "use_e3_channel_commensurability", False))
        prev_ema = dict(getattr(sel, "_chan_scale_ema", {}) or {})
        prev_n = int(getattr(sel, "_chan_scale_n", 0) or 0)
        prev_raw = dict(getattr(sel, "_last_commensurability_raw", {}) or {})
        sel._chan_scale_n = self._n_snapshot
        sel.config.use_e3_channel_commensurability = bool(commensurability)
        sel._chan_scale_ema = dict(self._ema_snapshot)
        try:
            out = []
            for args, kwargs, _live in self.calls:
                s = self._orig(*args, **kwargs)
                out.append(float(s.detach().reshape(-1).mean().item()))
            return out
        finally:
            sel.config.use_e3_channel_commensurability = prev_comm
            sel._chan_scale_ema = prev_ema
            sel._chan_scale_n = prev_n
            sel._last_commensurability_raw = prev_raw


def _eligible(scores: Sequence[float], margin: float) -> frozenset:
    lo = min(scores)
    rg = max(scores) - lo
    cut = lo + margin * rg
    return frozenset(i for i, s in enumerate(scores) if s <= cut)


def _eligible_live_float32(raw: torch.Tensor, margin: float) -> Tuple[int, frozenset]:
    """The selector's own margin ops (e3_selector.py select(), MARGIN branch), float32."""
    raw = raw.detach().reshape(-1)
    raw_score_range = float((raw.max() - raw.min()).item())
    best_raw = float(raw.min().item())
    cutoff = best_raw + margin * raw_score_range
    idx = torch.nonzero(raw <= cutoff, as_tuple=False).flatten()
    return int(idx.numel()), frozenset(int(i) for i in idx.tolist())


def _jaccard_distance(a: frozenset, b: frozenset) -> float:
    u = len(a | b)
    return 0.0 if u == 0 else 1.0 - len(a & b) / float(u)


def _sd(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mu = sum(xs) / n
    return math.sqrt(sum((x - mu) * (x - mu) for x in xs) / n)


def _shape(xs: Sequence[float], sd: float) -> Tuple[float, float]:
    """(max - median)/sd and (median - min)/sd: sparse-HIGH vs sparse-LOW index."""
    s = sorted(xs)
    med = s[len(s) // 2]
    return (s[-1] - med) / sd, (med - s[0]) / sd


def _r_from_j(jbar: Dict[str, float]) -> Optional[float]:
    if len(jbar) < 2:
        return None
    mx = max(jbar.values())
    if mx <= 0.0:
        return 0.0
    return float(min(jbar.values()) / mx)


def run_cell(arm: Dict[str, Any], seed: int, p0_episodes: int, p1_episode_cap: int,
             steps_per_episode: int, fresh_target: int, dry_run: bool = False) -> Dict[str, Any]:
    min_content = DRY_RUN_MIN_CONTENT_TICKS if dry_run else MIN_CONTENT_TICKS
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
        margin = float(getattr(agent.e3.config, "modulatory_shortlist_margin", 0.25))
        floor = float(agent.e3.config.e3_commensurability_floor)

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

        # instrument counters
        n_scored = 0
        n_residual_fail = 0
        max_residual_rel = 0.0
        n_selfcheck_fail = 0
        n_offcheck_fail = 0
        n_shortlist_active = 0
        n_i2_mismatch = 0
        n_selected_outside_e = 0
        n_final_by_primary = 0
        n_primary_flip = 0

        # DV accumulators: per scoring, per channel, list of per-tick Jaccard distances
        j_ticks: Dict[str, Dict[str, List[float]]] = {x: {c: [] for c in CHANNELS} for x in SCORINGS}
        e_sizes: Dict[str, List[int]] = {x: [] for x in SCORINGS}
        e_live_sizes: List[int] = []
        k_series: List[int] = []
        r_series: Dict[str, List[float]] = {c: [] for c in CHANNELS}
        shape_hi: Dict[str, List[float]] = {c: [] for c in CHANNELS}
        shape_lo: Dict[str, List[float]] = {c: [] for c in CHANNELS}
        live_by_ema_ticks: Dict[str, int] = {c: 0 for c in CHANNELS}

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
                    n_scored += 1
                    live = [c[2] for c in _cap.calls]
                    terms = _cap.raw_terms
                    k = len(live)
                    k_series.append(k)
                    m = _cap.divisors()

                    # --- I1 residual: reconstructed S_ON must equal the live score -----
                    s_on: List[float] = []
                    s_off: List[float] = []
                    tick_resid_fail = False
                    for i in range(k):
                        parts = [CHANNEL_SIGN[c] * terms[i][c] / m[c] for c in CHANNELS]
                        s = sum(parts)
                        s_on.append(s)
                        s_off.append(sum(CHANNEL_SIGN[c] * terms[i][c] for c in CHANNELS))
                        scale = max([abs(p) for p in parts] + [abs(live[i])])
                        rel = abs(live[i] - s) / max(1.0, scale)
                        max_residual_rel = max(max_residual_rel, rel)
                        if rel > RESIDUAL_REL_TOL:
                            tick_resid_fail = True
                    if tick_resid_fail:
                        n_residual_fail += 1

                    # --- I1b replay self-check (1012a) ----------------------------------
                    sc_scores = _cap.replay(True)
                    if any(not _rel_close(a, b, max(abs(a), abs(b)), SELF_CHECK_TOLERANCE)
                           for a, b in zip(sc_scores, live)):
                        n_selfcheck_fail += 1

                    # --- I1c OFF cross-check: genuine OFF replay == reconstructed S_OFF ---
                    off_replay = _cap.replay(False)
                    if any(not _rel_close(a, b, max([abs(a), abs(b)] + [abs(terms[i][c]) for c in CHANNELS]),
                                          RESIDUAL_REL_TOL)
                           for i, (a, b) in enumerate(zip(off_replay, s_off))):
                        n_offcheck_fail += 1

                    # continuity with 1012a: primary argmin flip ON vs OFF
                    if min(range(k), key=lambda i: s_on[i]) != min(range(k), key=lambda i: s_off[i]):
                        n_primary_flip += 1

                    # --- I2: live eligible set, float32, selector's own ops -------------
                    diag = getattr(agent.e3, "last_score_diagnostics", None) or {}
                    sl_active = bool(diag.get("modulatory_shortlist_active", False))
                    sl_size = int(diag.get("modulatory_shortlist_size", 0) or 0)
                    if (not sl_active) or sl_size <= 1:
                        n_final_by_primary += 1
                    raw_t = getattr(agent.e3, "last_raw_scores", None)
                    if sl_active and torch.is_tensor(raw_t) and int(raw_t.numel()) == k:
                        n_shortlist_active += 1
                        n_live_e, e_live = _eligible_live_float32(raw_t, margin)
                        e_live_sizes.append(n_live_e)
                        if n_live_e != sl_size:
                            n_i2_mismatch += 1
                        sel_idx = getattr(agent.e3, "last_selected_idx", None)
                        if sel_idx is not None and int(sel_idx) not in e_live:
                            n_selected_outside_e += 1

                    # --- DV: eligibility-stage knockouts under ON / OFF / ORACLE --------
                    sd_t: Dict[str, float] = {}
                    for c in CHANNELS:
                        col = [terms[i][c] for i in range(k)]
                        sd_t[c] = _sd(col)
                        if m[c] != 1.0:
                            live_by_ema_ticks[c] += 1
                            r_series[c].append(sd_t[c] / m[c])
                            if sd_t[c] > floor:
                                hi, lo = _shape(col, sd_t[c])
                                shape_hi[c].append(hi)
                                shape_lo[c].append(lo)
                    oracle_div = {c: (sd_t[c] if sd_t[c] > floor else 1.0) for c in CHANNELS}
                    scaled = {
                        "ON": {c: [CHANNEL_SIGN[c] * terms[i][c] / m[c] for i in range(k)] for c in CHANNELS},
                        "OFF": {c: [CHANNEL_SIGN[c] * terms[i][c] for i in range(k)] for c in CHANNELS},
                        "ORACLE": {c: [CHANNEL_SIGN[c] * terms[i][c] / oracle_div[c] for i in range(k)]
                                   for c in CHANNELS},
                    }
                    for x in SCORINGS:
                        s_x = [sum(scaled[x][c][i] for c in CHANNELS) for i in range(k)]
                        e_x = _eligible(s_x, margin)
                        e_sizes[x].append(len(e_x))
                        for c in CHANNELS:
                            # content tick: live by EMA AND sd_c(t) >= CONTENT_FRAC * s_hat_c;
                            # the SAME tick set for ON, OFF and ORACLE (red-team F4)
                            if m[c] == 1.0 or sd_t[c] < CONTENT_FRAC * m[c]:
                                continue
                            s_ko = [s_x[i] - scaled[x][c][i] for i in range(k)]
                            j_ticks[x][c].append(_jaccard_distance(e_x, _eligible(s_ko, margin)))

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
                    f"scored={n_scored}",
                    flush=True,
                )

            if is_p1 and fs.n_fresh_select >= fresh_target:
                target_met = True
                print(f"  [p1-done] arm={arm_id} seed={seed} fresh={fs.n_fresh_select} "
                      f"after {p1_episodes_run} P1 episode(s)", flush=True)
                break

        _ZG.observe(agent)
        _LAST_AGENT["agent"] = agent

        scale_est = _scale_estimates(agent)
        content_channels = sorted(c for c in CHANNELS if len(j_ticks["ON"][c]) >= min_content)
        excluded = {
            c: (f"live by EMA on {live_by_ema_ticks[c]} scored ticks but only "
                f"{len(j_ticks['ON'][c])} content ticks (< {min_content})")
            for c in CHANNELS if c not in content_channels and live_by_ema_ticks[c] > 0
        }
        jbar = {x: {c: _mean(j_ticks[x][c]) for c in content_channels} for x in SCORINGS}
        r_vals = {x: _r_from_j(jbar[x]) for x in SCORINGS}

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
            "n_ticks_scored": int(n_scored),

            # ===== instrument =====
            "n_residual_failures": int(n_residual_fail),
            "max_residual_rel": float(max_residual_rel),
            "n_self_check_failures": int(n_selfcheck_fail),
            "n_off_crosscheck_failures": int(n_offcheck_fail),
            "n_shortlist_active_scored": int(n_shortlist_active),
            "n_i2_size_mismatch": int(n_i2_mismatch),
            "i2_mismatch_rate": float(n_i2_mismatch / n_shortlist_active) if n_shortlist_active else 0.0,
            "n_selected_outside_live_e": int(n_selected_outside_e),

            # ===== THE ROUTED DV =====
            "content_channels": content_channels,
            "n_content_channels": int(len(content_channels)),
            "excluded_channels": excluded,
            "jbar": jbar,
            "n_content_ticks": {c: len(j_ticks["ON"][c]) for c in CHANNELS},
            "R_ON": r_vals["ON"],
            "R_OFF": r_vals["OFF"],
            "R_ORACLE": r_vals["ORACLE"],

            # ===== readouts (red-team F4: shape + r_c(t) reported, not buried) =====
            "r_realised_quantiles": {c: _quantiles(r_series[c]) for c in CHANNELS if r_series[c]},
            "shape_high_quantiles": {c: _quantiles(shape_hi[c]) for c in CHANNELS if shape_hi[c]},
            "shape_low_quantiles": {c: _quantiles(shape_lo[c]) for c in CHANNELS if shape_lo[c]},
            "eligible_size_mean": {x: _mean([float(v) for v in e_sizes[x]]) for x in SCORINGS},
            "eligible_size_live_float32_mean": _mean([float(v) for v in e_live_sizes]),
            "k_quantiles": _quantiles([float(v) for v in k_series]),
            "final_commit_by_primary_frac": float(n_final_by_primary / n_scored) if n_scored else 0.0,
            "primary_argmin_flip_rate": float(n_primary_flip / n_scored) if n_scored else 0.0,

            "channel_scale_estimates": scale_est,
            "operator_engaged": bool(scale_est.get("engaged", False)),
        }

        cell.stamp(row)

    def _fmt(v: Optional[float]) -> str:
        return "None" if v is None else f"{v:.4f}"

    if dry_run:
        print(
            f"  [smoke] arm={arm_id} seed={seed} scored={n_scored} "
            f"content={content_channels} R_ON={_fmt(r_vals['ON'])} R_OFF={_fmt(r_vals['OFF'])} "
            f"R_ORACLE={_fmt(r_vals['ORACLE'])} resid_fail={n_residual_fail} "
            f"max_resid_rel={max_residual_rel:.2e} selfcheck_fail={n_selfcheck_fail} "
            f"offcheck_fail={n_offcheck_fail} i2_mismatch={n_i2_mismatch}/{n_shortlist_active} "
            f"engaged={row['operator_engaged']}",
            flush=True,
        )
        print(f"  [smoke] jbar={jbar} n_content={row['n_content_ticks']} "
              f"E_mean={row['eligible_size_mean']} E_live={row['eligible_size_live_float32_mean']:.2f} "
              f"fcp={row['final_commit_by_primary_frac']:.3f} r={row['r_realised_quantiles']}", flush=True)

    instrument_ok = (n_residual_fail == 0 and n_selfcheck_fail == 0 and n_offcheck_fail == 0
                     and row["i2_mismatch_rate"] <= I2_MAX_MISMATCH_RATE)
    print(f"verdict: {'PASS' if instrument_ok else 'FAIL'}", flush=True)
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
        "decomp_samples_sufficient": _worst("n_ticks_scored"),
        "operator_engaged": 1.0 if rows and all(bool(r["operator_engaged"]) for r in rows) else 0.0,
        "clamp_config_landed": 1.0 if rows and all(bool(r["clamp_config_landed"]) for r in rows) else 0.0,
        "residue_protocol_landed": 1.0 if rows and all(bool(r["residue_protocol_landed"]) for r in rows) else 0.0,
        "n_content_channels": _worst("n_content_channels"),
    }
    overrides = {spec.name: bool(measured[spec.name] >= spec.threshold)
                 for spec in PRECONDITION_SPECS if spec.name in GEQ_PRECONDITIONS}
    ctx = {"id": arm["id"], "feed_residue": arm["feed_residue"], "warmup": arm["warmup"]}
    gate = evaluate_arm_gate(arm["id"], ctx, PRECONDITION_SPECS, measured, met_overrides=overrides)
    for p in gate["preconditions"]:
        p["kind"] = "readiness"
    return gate


def _regime_reading(rows: List[Dict[str, Any]], arm_green: bool) -> Dict[str, Any]:
    def _n_pass(key: str) -> int:
        return sum(1 for r in rows if r[key] is not None and float(r[key]) >= R_BAR)

    n_on, n_off, n_or = _n_pass("R_ON"), _n_pass("R_OFF"), _n_pass("R_ORACLE")
    c1 = n_on >= SEEDS_REQUIRED
    moot = n_off >= SEEDS_REQUIRED
    oracle = n_or >= SEEDS_REQUIRED
    if not arm_green:
        cls = "not_ready"   # this arm's gate red: scoped out, never vacates the other arm
    elif moot:
        cls = "moot"
    elif c1:
        cls = "commensurate"
    elif oracle:
        cls = "ema_lag"
    else:
        cls = "shape_range"
    return {
        "class": cls,
        "arm_gate_green": bool(arm_green),
        "c1_passed": bool(c1 and arm_green),
        "n_seeds_R_ON_at_bar": int(n_on),
        "n_seeds_R_OFF_at_bar": int(n_off),
        "n_seeds_R_ORACLE_at_bar": int(n_or),
        "seeds_required": SEEDS_REQUIRED,
        "r_bar": R_BAR,
        "R_ON_per_seed": [r["R_ON"] for r in rows],
        "R_OFF_per_seed": [r["R_OFF"] for r in rows],
        "R_ORACLE_per_seed": [r["R_ORACLE"] for r in rows],
        "R_ON_median": (float(statistics.median([r["R_ON"] for r in rows if r["R_ON"] is not None]))
                        if any(r["R_ON"] is not None for r in rows) else None),
    }


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

    by_arm = {a["id"]: [r for r in rows if r["arm"] == a["id"]] for a in ARMS}
    arm_gates = [_arm_gate(a, by_arm[a["id"]]) for a in ARMS]
    gate = aggregate_arm_gates(arm_gates)
    green = set(gate["green_arms"])

    n_resid = sum(r["n_residual_failures"] for r in rows)
    n_self = sum(r["n_self_check_failures"] for r in rows)
    n_offc = sum(r["n_off_crosscheck_failures"] for r in rows)
    worst_i2 = max((r["i2_mismatch_rate"] for r in rows), default=0.0)
    instrument_ok = (n_resid == 0 and n_self == 0 and n_offc == 0 and worst_i2 <= I2_MAX_MISMATCH_RATE)
    # capture-collapse guard (red-team F1's worst case): R_ON identical to R_OFF in EVERY
    # cell means the reconstruction never separated the two scorings
    on_off_separated = any(
        r["R_ON"] is not None and r["R_OFF"] is not None and abs(r["R_ON"] - r["R_OFF"]) > 1e-9
        for r in rows
    )

    fed = _regime_reading(by_arm[FED_ARM], FED_ARM in green)
    starved = _regime_reading(by_arm[STARVED_ARM], STARVED_ARM in green)

    if not instrument_ok:
        outcome, label = "FAIL", "instrument_defect"
    elif FED_ARM not in green and STARVED_ARM not in green:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif fed["class"] == "commensurate" and starved["class"] == "commensurate":
        outcome, label = "PASS", "commensurate_at_eligibility_both_regimes"
    else:
        outcome, label = "FAIL", f"fed_{fed['class']}__starved_{starved['class']}"

    outcome_note = (
        f"{label}: fed R_ON per seed {fed['R_ON_per_seed']} ({fed['n_seeds_R_ON_at_bar']}/"
        f"{len(by_arm[FED_ARM])} at bar {R_BAR}), starved R_ON per seed {starved['R_ON_per_seed']} "
        f"({starved['n_seeds_R_ON_at_bar']}/{len(by_arm[STARVED_ARM])}); R_OFF fed "
        f"{fed['R_OFF_per_seed']} starved {starved['R_OFF_per_seed']}; R_ORACLE fed "
        f"{fed['R_ORACLE_per_seed']} starved {starved['R_ORACLE_per_seed']}. instrument: "
        f"residual_failures={n_resid} self_check_failures={n_self} off_crosscheck_failures={n_offc} "
        f"worst_i2_mismatch_rate={worst_i2:.4f}. gate green={sorted(green)}. Eligibility-stage "
        "validation of the E3 commensurability operator (substrate); MECH-439 direction does not move."
    )

    def _c1_nd(arm_id: str) -> bool:
        rs = by_arm[arm_id]
        sep = any(r["R_ON"] is not None and r["R_OFF"] is not None
                  and abs(r["R_ON"] - r["R_OFF"]) > 1e-9 for r in rs)
        return bool(instrument_ok and sep and arm_id in green)
    criteria = []
    for arm_id, rd in ((FED_ARM, fed), (STARVED_ARM, starved)):
        criteria.append({
            "name": f"C1_eligibility_authority_{'fed' if arm_id == FED_ARM else 'starved'}",
            "load_bearing": True,
            "role": "verdict",
            "passed": bool(rd["c1_passed"]),
            "measured": float(rd["n_seeds_R_ON_at_bar"]),
            "threshold": float(SEEDS_REQUIRED),
            "per_seed_measured": rd["R_ON_per_seed"],
            "per_seed_threshold": R_BAR,
            "detail": f"R_ON >= {R_BAR} in >= {SEEDS_REQUIRED} of 4 seeds",
        })
    criteria.append({
        "name": "instrument_clean",
        "load_bearing": True,
        "role": "instrument correctness gate",
        "passed": bool(instrument_ok),
        "measured": float(n_resid + n_self + n_offc),
        "threshold": 0.0,
        "direction": "upper",
        "i2_worst_mismatch_rate": float(worst_i2),
        "i2_threshold": I2_MAX_MISMATCH_RATE,
        "detail": (f"residual={n_resid} self_check={n_self} off_crosscheck={n_offc} failures; "
                   f"worst I2 mismatch rate {worst_i2:.4f} vs {I2_MAX_MISMATCH_RATE}"),
    })

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
                "C1_eligibility_authority_fed": _c1_nd(FED_ARM),
                "C1_eligibility_authority_starved": _c1_nd(STARVED_ARM),
            },
            "combination_rule": (
                "Precedence: instrument (residual, replay self-check, OFF cross-check, I2 rate) > "
                "per-regime class. Per regime exactly one class: not_ready (that arm's gate red; "
                "scoped out, never vacates the other regime), else moot (R_OFF >= bar in >= 3/4 seeds), else commensurate (C1: R_ON >= bar "
                "in >= 3/4), else ema_lag (R_ORACLE >= bar in >= 3/4), else shape_range. PASS iff "
                "both regimes commensurate; every other combination FAIL with label "
                "fed_<class>__starved_<class>. outcome is the rung-3 VALIDATION verdict, not a "
                "MECH-439 direction."
            ),
            "on_off_separated": bool(on_off_separated),
            "prior_note": (
                "PASS is the expected reading under Gaussian or sparse-HIGH channel shapes at "
                "r ~ 1; it fails gate-green for sparse-LOW shapes (R ~0.21 at perfect SD "
                "equalisation), sparse-HIGH at r >= 3, or a channel >= ~5x its EMA. Read R with "
                "shape_*_quantiles and r_realised_quantiles."
            ),
        },
        "criteria": criteria,
        "summary": {
            "label": label,
            "fed": fed,
            "starved": starved,
            "instrument_ok": bool(instrument_ok),
            "n_residual_failures_total": int(n_resid),
            "n_self_check_failures_total": int(n_self),
            "n_off_crosscheck_failures_total": int(n_offc),
            "worst_i2_mismatch_rate": float(worst_i2),
        },
        "diagnostics": {
            "R_per_cell": {f"{r['arm']}/seed{r['seed']}": {"ON": r["R_ON"], "OFF": r["R_OFF"],
                                                           "ORACLE": r["R_ORACLE"]} for r in rows},
            "final_commit_by_primary_frac_per_cell": {
                f"{r['arm']}/seed{r['seed']}": r["final_commit_by_primary_frac"] for r in rows},
            "primary_argmin_flip_rate_per_cell": {
                f"{r['arm']}/seed{r['seed']}": r["primary_argmin_flip_rate"] for r in rows},
        },
    }


def _flat(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None if v is None else (1.0 if v else 0.0)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-1012c: eligibility-stage validation of the E3 channel-commensurability operator"
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
            "R_BAR": R_BAR,
            "SEEDS_REQUIRED": SEEDS_REQUIRED,
            "CONTENT_FRAC": CONTENT_FRAC,
            "MIN_CONTENT_TICKS": MIN_CONTENT_TICKS,
            "RESIDUAL_REL_TOL": RESIDUAL_REL_TOL,
            "SELF_CHECK_TOLERANCE": SELF_CHECK_TOLERANCE,
            "I2_MAX_MISMATCH_RATE": I2_MAX_MISMATCH_RATE,
        },
        "arms": [dict(a) for a in ARMS],
        "arm_config_slices": {
            a["id"]: config_slice_for(a, p0_used, p1_cap_used, steps_used, fresh_used) for a in ARMS
        },
        "dry_run": bool(args.dry_run),
    }

    s = result["summary"]
    readout: Dict[str, float] = {}
    for tag, rd in (("fed", s["fed"]), ("starved", s["starved"])):
        for key in ("n_seeds_R_ON_at_bar", "n_seeds_R_OFF_at_bar", "n_seeds_R_ORACLE_at_bar",
                    "R_ON_median", "c1_passed"):
            v = _flat(rd.get(key))
            if v is not None:
                readout[f"{tag}_{key}"] = v
    for key in ("n_residual_failures_total", "n_self_check_failures_total",
                "n_off_crosscheck_failures_total", "worst_i2_mismatch_rate", "instrument_ok"):
        v = _flat(s.get(key))
        if v is not None:
            readout[key] = v

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": list(CLAIM_IDS),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "Substrate validation of f_dominance_conversion_ceiling rung 3 (the E3 channel-"
            "commensurability operator) at the eligibility stage; not a test of MECH-439's "
            "hypothesis. Non_contributory on every branch -- set explicitly so the indexer does "
            "not re-infer 'supports' from a PASS (1012a autopsy sec. 5g)."
        ),
        "evidence_direction_per_claim": {"MECH-439": "non_contributory"},
        "supersedes": None,
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
        "readout": readout,
        "custom_information": {
            "governance_flag": "GFLAG-0297",
            "design_doc": "evidence/planning/gflag0297_mech439_rung3_null_design.md",
            "predecessor": "V3-EXQ-1012a (primary-argmin flip; PASS, non_contributory)",
            "refused_sibling": "V3-EXQ-1012b (label-permutation placebo, refused Step 4.5)",
            "gov_reuse_1_check": (
                "Decisive readout: R_ON (eligibility-stage knockout authority ratio). No manifest "
                "carries per-tick per-candidate per-channel terms; 571c/1012a record per-cell "
                "aggregates only. Not recoverable -> run."
            ),
            "brake_count_note": (
                "MECH-439 re-derive brake fires for re-tests of its hypothesis; this is substrate "
                "validation (measurement axis), the work the brake routes to. No ceiling hit added."
            ),
            "step_2_5c_known_limitation": (
                "Runs under open corrupting entry contextmemory-write-path-addressing-degeneracy "
                "(e1_deep.py ContextMemory.write, live; fix default-off) -- upstream of the stage "
                "under test and shared with the regime the conversion falsifiers run in."
            ),
            "gflag0072_p1_note": (
                "support_preserving_min_first_action_classes kept at the lineage value 2: "
                "GFLAG-0072's ceiling binds committed-class-entropy DVs; this DV is candidate-level "
                "eligibility membership, and raising it would break regime identity with 571c/1012a."
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
    if manifest["supersedes"] is None:
        del manifest["supersedes"]
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
    print(f"LABEL: {result['interpretation']['label']}", flush=True)
    for c in result["criteria"]:
        print(f"  {c['name']}: {c['passed']} (measured {c['measured']}, threshold {c['threshold']})",
              flush=True)
    print(
        f"  per_arm_gate: green={result['per_arm_gate']['green_arms']} "
        f"red={result['per_arm_gate']['red_arms']}",
        flush=True,
    )

    if args.dry_run:
        assert s["n_residual_failures_total"] == 0, "SMOKE FAIL: I1 residual -- capture/reconstruction bug"
        assert s["n_self_check_failures_total"] == 0, "SMOKE FAIL: I1b replay self-check"
        assert s["n_off_crosscheck_failures_total"] == 0, "SMOKE FAIL: I1c OFF cross-check"
        print("DRY RUN complete.", flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=bool(args.dry_run),
    )
