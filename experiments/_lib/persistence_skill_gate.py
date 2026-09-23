"""Persistence-baseline readiness gate for residual / forward heads on slow latents.

WHAT THIS REPLACES, AND THE MEASUREMENT THAT CONDEMNS IT
---------------------------------------------------------
V3-EXQ-1062a gated axis 3 of MECH-055 on an ABSOLUTE R2 floor:

    harm_a_forward_r2_supra_floor : harm_a_forward_r2 >= FORWARD_R2_MIN (0.30)

Measured against its own control, on the 6 landed cells of
`v3_exq_1062a_mech055_affect_channel_separation_postshift_20260923T002356Z_v3`:

    the trivial persistence predictor (z_pred = z(t-1)) scores
    persistence_r2 = 0.96931 .. 0.99927 -- it CLEARS THE 0.30 FLOOR IN 6/6 CELLS,
    with 0.669 to 0.699 of headroom.

The model scored 0.8958 .. 0.9773 and passed the floor in 6/6 cells while sitting
BELOW its own control in 6/6 cells (model_r2 - persistence_r2 = -0.089 .. -0.022).
So the gate certified "substrate ready" for a head strictly worse than doing
nothing, and the dACC harm-PE that MECH-055 axis 3 routes on was model error
rather than world surprise. Source entry:
`SD-PP-B9-harm-forward-below-persistence-baseline` (substrate_queue.json),
from failure_autopsy_V3-EXQ-1062a_2026-09-23.

THIS IS THE MIRROR OF THE V3-EXQ-1075 DEFECT, NOT THE SAME ONE
---------------------------------------------------------------
V3-EXQ-1075's A3 bar (conv_rel_drop >= 0.99) sat ABOVE its own control
(0.98466/0.98486/0.99163): the PASS region was nearly empty, a FALSE-NEGATIVE
route. This one sits far BELOW its control: the PASS region contains the null
state, a FALSE-POSITIVE route. Both are the same root omission -- the bar was
never denominated on the run's own control arm. That is the shape
`GOV-CRITBAR-1` (draft, REE_assembly/evidence/planning/
claim_synthesis_dv_reachability_20260923.md sec. 6) names.

WHY THIS GATE IS NOT ANOTHER ABSOLUTE FLOOR
--------------------------------------------
The obvious repair -- "gate on skill >= 0.05" -- reintroduces the defect in a new
place. Skill = 1 - SSE_model/SSE_persistence is denominator-driven, and on the
STATIONARY arm at three seeds and an IDENTICAL config the denominator
(1 - persistence_r2) measured 0.00073 / 0.01194 / 0.00093 -- a 16.4x spread --
carrying skill to -71.67 / -3.98 / -23.48, an 18x spread. A magnitude bar on that
statistic is a bar on which seed you drew. The autopsy's own robust statement is
the SIGN, 6/6.

So the gate routes on a BOUNDED, DENOMINATOR-SYMMETRIC form of the same
comparison:

    d = (SSE_persistence - SSE_model) / (SSE_persistence + SSE_model)      in [-1, 1]

d has the same sign as skill always, and its CONTROL VALUE IS EXACTLY 0 BY
CONSTRUCTION -- persistence scored against itself gives SSE_model == SSE_persistence,
hence d = 0. It is not a measured control that might drift; it is analytic. That is
what "denominated on the control" means here, and it is why no literal threshold
appears in the decision.

Measured compression on the same 6 cells: skill spans 61.7x (-71.672 .. -1.162),
d spans 2.6x (-0.9729 .. -0.3674).

WHY THE DECISION IS A CONFIDENCE INTERVAL AND NOT `d > 0`
----------------------------------------------------------
A bare `d > 0` is V3-EXQ-1075's A1 shape: a zero-margin bar on a statistic with
sampling noise. And per the V3-EXQ-1075a stop handover section 3, a continuous
band-edge comparison is NOT PORTABLE across machine classes -- seed 42's ARM_OFF
ratio differed by 0.0005 between `linux-x86_64/torch2.12.0+cpu` and
`darwin-arm64/torch2.12.0`, which was 9% of that design's decisive margin.

So the margin is DERIVED from the run's own rows, never pinned: a paired
bootstrap over the per-row squared errors gives a CI on d, and

    CI lower  > 0  -> "ready"
    CI upper  < 0  -> "persistence_dominated"   (a real, attributable FAIL)
    CI straddles 0 -> "cannot_determine"

A decision inside the statistic's own noise therefore returns cannot_determine
instead of flipping on a float smaller than cross-machine reproduction error.

THIS IS A NEGATIVE INSTRUMENT -- READ CLAUDE.md "Negative instruments"
----------------------------------------------------------------------
It authorises STARTING work on a negative ("no persistence dominance found ->
proceed"), so numerator and denominator come from one computation. Three remedies,
strongest first:

1. STRUCTURAL cannot-determine CATEGORY. `PersistenceSkillVerdict.status` is a
   three-valued string, never a bool, and `cannot_determine` propagates into
   `to_dict()` / --json. `is_ready` exists but callers must branch on `status`.

2. KNOWN-BASELINE CANARY. `CANARY_V3_EXQ_1062A` pins that run's six landed cells.
   `check_canary()` reconstructs each cell's skill from the two R2 aggregates and
   requires it to reproduce the manifest's independently-computed
   `harm_forward_skill_vs_persistence` (the manifest derived it from raw SSE
   tensors; this module derives it from R2 -- two different routes, so agreement
   is evidence and not a tautology), and then drives the SHIPPED verdict path on a
   synthetic battery carrying each cell's numbers. Agreement is exact to ~1e-12.

3. PRINTED PRE-FILTER DENOMINATOR. Every verdict carries `n_rows`, `sse_model`,
   `sse_persistence` and `persistence_r2`, and `format_verdict()` prints them.

NO DEGENERATE-MOVEMENT THRESHOLD IS INVENTED HERE
--------------------------------------------------
A target that never moves makes the question unanswerable rather than answered,
so `ss_total == 0` is a cannot_determine. But "moved, though only a little" is an
OPERATING-POINT judgement, and pinning a literal for it would be the very defect
this module exists to remove. `min_persistence_headroom` therefore defaults to
None (OFF) and `persistence_r2` is surfaced for the reader instead. Note this is
NOT a fail-open hole in the verdict: the bootstrap CI already widens to
cannot_determine wherever the rows cannot separate the two predictors.

ASCII-only output (CLAUDE.md "ASCII-Only in Python Output").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "CANARY_V3_EXQ_1062A",
    "PersistenceSkillVerdict",
    "check_canary",
    "format_verdict",
    "per_row_squared_error",
    "persistence_verdict",
    "relative_skill",
    "relative_skill_from_r2",
    "skill_from_r2",
    "skill_vs_persistence",
]

# --------------------------------------------------------------------------- #
# Defaults. Everything here is a keyword argument at every call site. NONE of   #
# these is a bar on the statistic being gated -- the gate's own threshold is    #
# the analytic control value 0, which is not configurable and never appears as  #
# a literal in a decision.                                                      #
# --------------------------------------------------------------------------- #
MIN_ROWS = 32          # below this a paired bootstrap is not evidence
N_BOOTSTRAP = 2000
CI_LEVEL = 0.95
BOOTSTRAP_SEED = 20260923
# Used ONLY by check_canary, to tell a live bootstrap from one whose resamples
# are all identical. Not a gate on any experimental quantity. See the comment at
# its use site for the measurement that places it.
CI_LIVE_REL_WIDTH_MIN = 1e-4


@dataclass
class PersistenceSkillVerdict:
    """Three-valued readiness verdict. `status` is NEVER a bool.

    status:
      "ready"                 -- the head beats the persistence baseline, with the
                                 whole CI on the useful side of the control value.
      "persistence_dominated" -- MEASURED: the trivial z_pred = z(t-1) predictor is
                                 at least as good. An attributable FAIL, not noise.
      "cannot_determine"      -- the rows could not separate the two predictors, or
                                 the target did not move. NOT a negative result.
    """
    status: str
    reason: str
    relative_skill: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    skill: Optional[float] = None
    sse_model: Optional[float] = None
    sse_persistence: Optional[float] = None
    sse_total: Optional[float] = None
    model_r2: Optional[float] = None
    persistence_r2: Optional[float] = None
    n_rows: int = 0
    n_bootstrap: int = 0
    ci_level: float = CI_LEVEL
    control_value: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        """True ONLY for "ready". cannot_determine is False here -- but callers
        must branch on `status`, or they collapse the third value back into the
        bool this module exists to avoid."""
        return self.status == "ready"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Pure statistics. Each returns None -- never 0.0 -- when undefined, because a  #
# zero denominator is a cannot-determine and not a skill of zero.               #
# --------------------------------------------------------------------------- #

def skill_vs_persistence(sse_model: float, sse_persistence: float) -> Optional[float]:
    """1 - SSE_model / SSE_persistence. The statistic V3-EXQ-1062a recorded.

    Kept because it is what the landed manifests carry and what the autopsy
    quotes, NOT because it is the right thing to gate on -- see the module
    docstring: its denominator moved 16.4x across seeds at an identical config.
    Use `relative_skill` for a decision.
    """
    if not (_finite(sse_model) and _finite(sse_persistence)):
        return None
    if float(sse_persistence) <= 0.0:
        return None
    return 1.0 - (float(sse_model) / float(sse_persistence))


def skill_from_r2(model_r2: float, persistence_r2: float) -> Optional[float]:
    """The same skill, reconstructed from two R2 aggregates against a shared SS_tot.

    SSE/SS_tot = 1 - r2 for both predictors, so
        skill = 1 - (1 - model_r2) / (1 - persistence_r2).
    This is the route `check_canary` uses against manifests, which carry the two
    R2 values but not the raw residual tensors.
    """
    if not (_finite(model_r2) and _finite(persistence_r2)):
        return None
    den = 1.0 - float(persistence_r2)
    if den <= 0.0:
        return None
    return 1.0 - ((1.0 - float(model_r2)) / den)


def relative_skill(sse_model: float, sse_persistence: float) -> Optional[float]:
    """(SSE_persistence - SSE_model) / (SSE_persistence + SSE_model), in [-1, 1].

    THE STATISTIC THE GATE ROUTES ON. Same sign as `skill_vs_persistence` always,
    but bounded and denominator-symmetric, so a seed whose latent barely moved
    cannot inflate it. Its control value -- persistence scored against itself --
    is EXACTLY 0 by construction, which is what makes a sign test here a test
    against the run's own baseline rather than against an invented constant.
    """
    if not (_finite(sse_model) and _finite(sse_persistence)):
        return None
    den = float(sse_persistence) + float(sse_model)
    if den <= 0.0:
        return None
    return (float(sse_persistence) - float(sse_model)) / den


def relative_skill_from_r2(model_r2: float, persistence_r2: float) -> Optional[float]:
    """`relative_skill` from two R2 aggregates sharing one SS_tot (see skill_from_r2).

    DELIBERATELY DEFINED WHERE `skill_from_r2` IS NOT. At persistence_r2 == 1.0
    the baseline is exact, `skill_from_r2` divides by zero and returns None,
    while this returns -1.0 -- "persistence dominates completely", which is the
    true and useful answer. That asymmetry is the whole reason the gate routes
    on this form. The separate degenerate case, a target that never moved, is
    caught by the SS_tot check in `persistence_verdict` and not here.
    """
    if not (_finite(model_r2) and _finite(persistence_r2)):
        return None
    res = 1.0 - float(model_r2)
    per = 1.0 - float(persistence_r2)
    den = res + per
    if den <= 0.0:
        return None
    return (per - res) / den


def per_row_squared_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Row-wise sum of squared error, shape (n,). The bootstrap resamples these.

    Rows are the resampling unit because the two predictors are compared on the
    SAME rows -- a paired bootstrap, so the CI is on the difference and not on
    two independently noisy means.
    """
    with torch.no_grad():
        diff = (pred - target).reshape(int(pred.shape[0]), -1)
        return (diff ** 2).sum(dim=-1)


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def persistence_verdict(
    pred: torch.Tensor,
    target: torch.Tensor,
    prev: torch.Tensor,
    min_rows: int = MIN_ROWS,
    n_bootstrap: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
    min_persistence_headroom: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> PersistenceSkillVerdict:
    """The gate. Three-valued; never raises on a thin or degenerate battery.

    `pred`   -- the head's prediction of z(t), one row per evaluated pair.
    `target` -- the true z(t).
    `prev`   -- z(t-1), i.e. the trivial persistence predictor's output.

    All three must be the SAME rows in the SAME order, under one no-grad
    evaluation. This function cannot check that and callers must honour it --
    a mismatched `prev` silently measures a different baseline.

    `min_persistence_headroom`, when not None, additionally returns
    cannot_determine if (1 - persistence_r2) falls below it. It defaults to None
    on purpose; see the module docstring.
    """
    n_rows = int(pred.shape[0]) if pred.ndim > 0 else 0

    def cd(reason: str, **kw: Any) -> PersistenceSkillVerdict:
        return PersistenceSkillVerdict(
            status="cannot_determine", reason=reason, n_rows=n_rows,
            n_bootstrap=0, ci_level=ci_level, **kw)

    if n_rows < min_rows:
        return cd("battery too small: %d rows < %d" % (n_rows, min_rows))
    if int(target.shape[0]) != n_rows or int(prev.shape[0]) != n_rows:
        return cd("row-count mismatch: pred=%d target=%d prev=%d -- the paired "
                  "comparison is not defined on unaligned rows"
                  % (n_rows, int(target.shape[0]), int(prev.shape[0])))

    with torch.no_grad():
        e_model = per_row_squared_error(pred, target)
        e_per = per_row_squared_error(prev, target)
        sse_model = float(e_model.sum().item())
        sse_per = float(e_per.sum().item())
        flat = target.reshape(n_rows, -1)
        sse_tot = float(((flat - flat.mean(dim=0, keepdim=True)) ** 2).sum().item())

    if not (_finite(sse_model) and _finite(sse_per)):
        return cd("non-finite residuals: sse_model=%r sse_persistence=%r"
                  % (sse_model, sse_per))
    if sse_tot <= 0.0:
        return cd("the target did not move (sse_total=0) -- persistence is exactly "
                  "right and the question 'is there anything to predict' is "
                  "unanswerable, not answered",
                  sse_model=sse_model, sse_persistence=sse_per, sse_total=sse_tot)

    model_r2 = 1.0 - sse_model / sse_tot
    per_r2 = 1.0 - sse_per / sse_tot
    d = relative_skill(sse_model, sse_per)
    sk = skill_vs_persistence(sse_model, sse_per)
    common = dict(relative_skill=d, skill=sk, sse_model=sse_model,
                  sse_persistence=sse_per, sse_total=sse_tot,
                  model_r2=model_r2, persistence_r2=per_r2)

    if d is None:
        return cd("relative skill undefined: sse_model + sse_persistence = 0 "
                  "(both predictors exact; nothing to separate)", **common)

    if min_persistence_headroom is not None:
        headroom = 1.0 - per_r2
        if headroom < float(min_persistence_headroom):
            return cd("persistence headroom %.6g < %.6g -- the baseline already "
                      "explains the target, so this battery cannot show skill"
                      % (headroom, float(min_persistence_headroom)), **common)

    # Paired bootstrap over rows. The margin is DERIVED here; no literal bar.
    gen = generator
    if gen is None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(BOOTSTRAP_SEED)
    with torch.no_grad():
        idx = torch.randint(0, n_rows, (int(n_bootstrap), n_rows), generator=gen)
        bm = e_model.to("cpu")[idx].sum(dim=1)
        bp = e_per.to("cpu")[idx].sum(dim=1)
        den = bp + bm
        ok = den > 0
        vals = ((bp - bm)[ok] / den[ok]).tolist()

    if len(vals) < max(2, int(0.5 * n_bootstrap)):
        return cd("bootstrap degenerate: only %d of %d resamples had a non-zero "
                  "denominator" % (len(vals), int(n_bootstrap)), **common)

    vals.sort()
    alpha = (1.0 - float(ci_level)) / 2.0
    lo = _percentile(vals, alpha)
    hi = _percentile(vals, 1.0 - alpha)
    common.update(ci_low=lo, ci_high=hi)

    if lo > 0.0:
        return PersistenceSkillVerdict(
            status="ready",
            reason=("beats the persistence baseline: relative skill %.4f, "
                    "%.0f%% CI [%.4f, %.4f] entirely above the control value 0.0"
                    % (d, 100.0 * ci_level, lo, hi)),
            n_rows=n_rows, n_bootstrap=len(vals), ci_level=ci_level, **common)
    if hi < 0.0:
        return PersistenceSkillVerdict(
            status="persistence_dominated",
            reason=("the trivial z_pred = z(t-1) predictor is BETTER: relative "
                    "skill %.4f, %.0f%% CI [%.4f, %.4f] entirely below the "
                    "control value 0.0 (skill vs persistence %.4g). The residual "
                    "this head produces is model error, not world surprise."
                    % (d, 100.0 * ci_level, lo, hi,
                       float("nan") if sk is None else sk)),
            n_rows=n_rows, n_bootstrap=len(vals), ci_level=ci_level, **common)
    return cd("CI straddles the control value 0.0: relative skill %.4f, "
              "%.0f%% CI [%.4f, %.4f] -- these rows cannot separate the head "
              "from the persistence baseline in either direction"
              % (d, 100.0 * ci_level, lo, hi),
              **common)


# --------------------------------------------------------------------------- #
# Canary (remedy 2). Pinned from the LANDED manifest, not from this module.     #
# --------------------------------------------------------------------------- #
CANARY_V3_EXQ_1062A = {
    "source": ("REE_assembly/evidence/experiments/v3_exq_1062a_mech055_affect_"
               "channel_separation_postshift_20260923T002356Z_v3.json "
               "(arm_results, 6 cells; read 2026-09-23)"),
    "sd_id": "SD-PP-B9-harm-forward-below-persistence-baseline",
    # (seed, model_r2, persistence_r2, skill_recorded_in_manifest)
    "cells": [
        (42,   0.947201874120806,  0.9992734701951642, -71.6716585166474),
        (137,  0.9405317557368673, 0.9880552194894984,  -3.978596652391322),
        (2026, 0.9773246454335065, 0.9990736978258801, -23.479435760839465),
        (42,   0.8957915487241713, 0.9851004155231112,  -5.994050836616953),
        (137,  0.9409854968659925, 0.9820127668899814,  -2.2809105643456316),
        (2026, 0.9336487938931001, 0.9693063612669316,  -1.1617249972847064),
    ],
    # The gate this module replaces, and what the CONTROL read against it.
    "superseded_gate": "harm_a_forward_r2 >= FORWARD_R2_MIN (0.30)",
    "superseded_gate_control_pass_cells": 6,   # persistence_r2 cleared 0.30 in 6/6
    "superseded_gate_model_pass_cells": 6,
    # Every cell must classify persistence_dominated through the shipped path.
    "expected_status": ["persistence_dominated"] * 6,
}


def _synthetic_pairs(model_r2: float, persistence_r2: float,
                     n: int = 256, dim: int = 4):
    """Build (pred, target, prev) whose REAL measurement through this module
    reproduces a given (model_r2, persistence_r2).

    Exists so `check_canary` drives `persistence_verdict` itself rather than
    comparing two constants to two other constants -- CLAUDE.md, "a guard that
    supplies the thing it asserts is not a guard".

    THE TWO RESIDUALS MUST BE ROW-WISE INDEPENDENT, and this is load-bearing.
    An earlier draft built both predictors as scalings of the target
    (`pred = a*target`, `prev = b*target`). Those residuals are PROPORTIONAL in
    every row, so every paired-bootstrap resample returns the identical ratio and
    the CI collapses to zero width -- the canary then passes without ever
    exercising the interval logic it exists to guard. Measured on that draft: CI
    width 0.0000 on 6 of 6 cells. So the two residuals are drawn from INDEPENDENT
    noise here, each rescaled to hit its required SSE exactly, and
    `check_canary` asserts the resulting CI has non-zero width.

    Construction: `target` is zero-mean noise with a known SS_tot; each predictor
    is `target + e`, with `e` an independent noise tensor rescaled so
    sum(e^2) = (1 - r2) * SS_tot, which gives that predictor exactly `r2`.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(12345)
    base = torch.randn(n, dim, generator=g)
    base = base - base.mean(dim=0, keepdim=True)
    ss_tot = float((base ** 2).sum().item())

    def _resid(seed: int, r2: float) -> torch.Tensor:
        gg = torch.Generator(device="cpu")
        gg.manual_seed(seed)
        e = torch.randn(n, dim, generator=gg)
        cur = float((e ** 2).sum().item())
        want = max(1.0 - float(r2), 0.0) * ss_tot
        if cur <= 0.0:
            return torch.zeros_like(e)
        return e * math.sqrt(want / cur)

    return (base + _resid(101, model_r2), base, base + _resid(202, persistence_r2),
            ss_tot)


def check_canary(tol: float = 1e-6, n_bootstrap: int = 400) -> Dict[str, Any]:
    """Replay V3-EXQ-1062a's six landed cells through the SHIPPED path.

    Two independent checks per cell:

    (a) RECONSTRUCTION. `skill_from_r2(model_r2, persistence_r2)` must reproduce
        the manifest's `harm_forward_skill_vs_persistence`. The manifest computed
        that from raw SSE tensors inside the driver; this module computes it from
        two R2 aggregates. Different routes, so agreement is evidence -- and a
        flipped numerator/denominator or an inverted sign breaks it.

    (b) CLASSIFICATION. A synthetic battery really carrying that cell's two R2
        values is run through `persistence_verdict`, and must come back
        `persistence_dominated`. This is the DECISION, not a re-derivation of it.
        The cell additionally fails if the returned CI has ZERO WIDTH, because a
        degenerate interval would satisfy the classification without the interval
        logic having run at all.

    Also records the superseded gate's own reading, so the regression this module
    removes stays visible in the artifact: the 0.30 absolute R2 floor was cleared
    by the CONTROL in 6/6 cells.

    Returns a dict with "ok" plus per-cell detail. Never raises.
    """
    results: List[Dict[str, Any]] = []
    ok = True
    for (seed, m_r2, p_r2, skill_recorded), expected in zip(
            CANARY_V3_EXQ_1062A["cells"], CANARY_V3_EXQ_1062A["expected_status"]):
        rec = skill_from_r2(m_r2, p_r2)
        faithful = rec is not None and abs(rec - skill_recorded) <= max(
            tol, tol * abs(skill_recorded))
        try:
            pred, tgt, prev, _ = _synthetic_pairs(m_r2, p_r2)
            v = persistence_verdict(pred, tgt, prev, n_bootstrap=n_bootstrap)
            got = v.status
            # the synthetic battery must really carry the pinned R2s, or the
            # classification below is about something else
            carried = (v.model_r2 is not None and abs(v.model_r2 - m_r2) <= 1e-4
                       and v.persistence_r2 is not None
                       and abs(v.persistence_r2 - p_r2) <= 1e-4)
            # The interval logic must actually have been exercised. A
            # degenerate CI means every resample returned the same value, which
            # satisfies the classification check while testing nothing -- see
            # `_synthetic_pairs`. A bare `width > 0` does NOT detect that: the
            # proportional-residual battery measures width 1.79e-07 in float32
            # accumulation noise, which is strictly positive. MEASURED
            # separation: degenerate relative width ~1.8e-07, live relative
            # width 6.7e-03 .. 2.8e-01 across the six pinned cells -- four
            # orders apart, so CI_LIVE_REL_WIDTH_MIN sits between them.
            ci_live = (v.ci_low is not None and v.ci_high is not None
                       and v.relative_skill is not None
                       and abs(v.relative_skill) > 0.0
                       and ((v.ci_high - v.ci_low) / abs(v.relative_skill))
                       > CI_LIVE_REL_WIDTH_MIN)
        except Exception as exc:               # never raise out of a canary
            got, carried, ci_live, v = (
                "error:%s" % type(exc).__name__, False, False, None)
        match = (got == expected) and faithful and carried and ci_live
        ok = ok and match
        results.append({
            "seed": seed, "model_r2": m_r2, "persistence_r2": p_r2,
            "skill_recorded": skill_recorded,
            "skill_reconstructed": rec,
            "reconstruction_faithful": faithful,
            "battery_carries_pinned_r2": carried,
            "ci_non_degenerate": ci_live,
            "relative_skill": None if v is None else v.relative_skill,
            "ci": None if v is None else [v.ci_low, v.ci_high],
            "expected": expected, "got": got, "match": match,
            "superseded_absolute_floor_0p30_control_passes": p_r2 >= 0.30,
            "superseded_absolute_floor_0p30_model_passes": m_r2 >= 0.30,
        })
    n_ctrl = sum(1 for r in results
                 if r["superseded_absolute_floor_0p30_control_passes"])
    return {
        "ok": ok, "tol": tol, "n_cells": len(results), "cells": results,
        "source": CANARY_V3_EXQ_1062A["source"],
        "superseded_gate": CANARY_V3_EXQ_1062A["superseded_gate"],
        "superseded_gate_control_pass_cells": n_ctrl,
        "drives": "persistence_verdict (shipped path), not a re-implementation",
    }


def format_verdict(v: PersistenceSkillVerdict, label: str = "") -> str:
    """ASCII-only one-block summary. ALWAYS prints the denominators (remedy 3)."""
    head = "persistence-skill gate" + (" [%s]" % label if label else "")
    def f(x: Optional[float], spec: str = "%.4f") -> str:
        return "n/a" if x is None or not _finite(x) else spec % x
    lines = [
        "%s: %s" % (head, v.status.upper()),
        "  reason   : %s" % v.reason,
        "  rel skill: %s  (control value %.1f, bounded in [-1, 1])"
        % (f(v.relative_skill), v.control_value),
        "  CI       : [%s, %s] at %.0f%% over %d resamples"
        % (f(v.ci_low), f(v.ci_high), 100.0 * v.ci_level, v.n_bootstrap),
        "  skill    : %s  (1 - SSE_model/SSE_persistence; magnitude is "
        "denominator-driven, do not gate on it)" % f(v.skill, "%.4g"),
        "  r2       : model=%s persistence=%s" % (f(v.model_r2, "%.6g"),
                                                  f(v.persistence_r2, "%.6g")),
        "  DENOM    : n_rows=%d sse_model=%s sse_persistence=%s sse_total=%s"
        % (v.n_rows, f(v.sse_model, "%.6g"), f(v.sse_persistence, "%.6g"),
           f(v.sse_total, "%.6g")),
    ]
    if v.status == "cannot_determine":
        lines.append("  NOTE     : cannot_determine is NOT a negative result -- "
                     "these rows could not test the question.")
    return "\n".join(lines)
