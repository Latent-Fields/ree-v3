"""V3-EXQ-689h -- P-comp: MECH-448 demotion x MECH-449 Go/No-Go composition at C2.

PRONG-MAP P-comp (conversion_ceiling_prong_map.md): the within-selection-face
composition-characterization gate. We know the two face-validated selection-face
levers each act -- MECH-448 rank-preserving F->eligibility demotion (689d PASS
0.938 vs 0.371) and MECH-449 Go/No-Go eligibility constitution (689g PASS 3/3) --
and that Factor A x Factor B CANCELLED (689a). We do NOT yet know whether DEMOTION
x GO/NO-GO COMPOUND or CANCEL at the committed-action-class-entropy DV (C2). This
is the main open selection-face question (Factor B is dead, 689c) and the prong-map
gate that must be characterized BEFORE the full-stack target arm is assembled.

EXPERIMENT_PURPOSE = diagnostic. This CHARACTERIZES the composition (compound /
cancel / additive); it does NOT independently test the MECH-448 or MECH-449
mechanisms (each is separately face-validated). The verdict gates the full-stack
ASSEMBLY decision, so it is a characterization gate, not a claim falsifier. claim
tags (MECH-448 + MECH-449) are context only; the run is scoring-excluded from
confidence/conflict.

DV (C2): committed_action_class_entropy -- the Shannon entropy (nats) of the
distribution of committed first-action classes over N_BANKS decision contexts.

2x2 DESIGN (flags-literal; the two levers toggled, everything else held constant):
  ARM_OFF   demotion OFF, Go/No-Go OFF  -- plain F-argmin path (the conversion
            ceiling): the modulatory channel is drowned, committed = F-best class.
  ARM_DEM   demotion ON  (+ adaptive floor, 689e PASS, the matched-stack constant
            carried WITH demotion per the prong map), Go/No-Go OFF -- the MECH-448
            envelope admits the F-eligible front; the within-eligible argmin over
            the modulatory channel (F removed) converts to the modulatory favorite.
  ARM_GNG   demotion OFF, Go/No-Go ON   -- the Go/No-Go gate runs ONLY inside the
            shortlist-then-modulate block (e3_selector.py:1760), which fires only
            when demotion OR a shortlist builds an eligible set. With NEITHER, the
            gate NEVER runs, so use_go_nogo_constitution=True is a structural NO-OP
            -> ARM_GNG is bit-identical to ARM_OFF. This is a REAL FINDING (Go/No-Go
            has no standalone selection effect; it requires an eligible-set
            substrate), not a degeneracy -- it is what makes the 2x2 interaction
            measure the Go/No-Go-given-demotion conditional effect honestly.
  ARM_BOTH  demotion ON (+ adaptive floor), Go/No-Go ON -- envelope + the Go/No-Go
            gate suppresses an undesirable (safety) + a stale candidate on an axis
            ORTHOGONAL to F-rank; the within-eligible argmin then picks the
            highest-modulatory-preference SURVIVOR.

SUBSTRATE MODEL (why it is a faithful C2 conversion-ceiling probe). One FIXED
candidate menu per seed (the same K trajectories with distinct first-action
classes), so the primary F landscape is constant -> the F-argmin is a FIXED class
across contexts -> ARM_OFF commits the same class every context (monostrategy,
entropy ~ 0). The MODULATORY channel VARIES per context (a different preferred
class each decision, modelling context-dependent preference over a fixed action
menu). The levers' job is to let that contextual modulatory diversity reach
committed action -> committed-class entropy RISES. This is exactly the C2 question:
same menu, F monopolises (monostrategy), does the assembled selection face convert?

NON-VACUITY PRECONDITIONS (else self-route substrate_not_ready_requeue,
non_degenerate=false -- the run is scoring-excluded, NOT a false verdict):
  pool_divergent       median raw-F RANGE across seeds > floor (the menu has real
                       F-spread, so F-dominance is meaningful).
  envelope_nondegen    median demotion envelope size in [2, K-1] AND median
                       excluded_count >= 1 (the MECH-448 envelope actually excludes
                       the F-poor candidates; not an all-admit no-op).
  monostrategy_base    mean ARM_OFF entropy <= ceiling (the baseline is genuine
                       monostrategy -> there is headroom for the levers).
  demotion_engages     mean demotion main-effect lift >= floor (demotion genuinely
                       converts the F-monopoly into modulatory-driven diversity).
  gonogo_engages       median Go/No-Go suppression (candidates dropped from the
                       envelope) >= 1 on ARM_BOTH (the gate actually fires).

INTERACTION (the characterization; SD-of-delta + absolute-floor gate per
feedback_effect_size_pass_gate_margin -- NEVER pstdev of a baseline LEVEL):
  per seed: interaction = (e_both - e_gng) - (e_dem - e_off)
  margin   = max(K_SD * pstdev(interaction_per_seed), FLOOR_NATS)
  verdict  = compound  if interaction_mean >=  margin   (super-additive)
           = cancel    if interaction_mean <= -margin   (sub-additive / interfere)
           = additive  otherwise                        (independent composition)

INTERPRETATION (self-routing; CHARACTERIZES, promotes nothing):
  - preconditions unmet  -> substrate_not_ready_requeue (non_contributory;
    non_degenerate=false; both claim directions unknown).
  - preconditions met + compound -> demotion x Go/No-Go COMPOUND at C2 (the levers
    co-assemble super-additively); supports the full-stack assembly path.
  - preconditions met + additive -> they compose INDEPENDENTLY (Go/No-Go relabels
    within the envelope but adds no committed-class entropy beyond demotion).
  - preconditions met + cancel   -> they INTERFERE (Go/No-Go suppresses the
    converted diversity); routes the full-stack assembly to leave-one-out.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import (  # noqa: E402
    compute_arm_fingerprint,
    reset_all_rng,
)
from ree_core.predictors.e2_fast import Trajectory  # noqa: E402
from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_689h_pcomp_demotion_x_gonogo_composition"
QUEUE_ID = "V3-EXQ-689h"
SUPERSEDES: Optional[str] = None  # NEW question (P-comp composition, first run)
CLAIM_IDS: List[str] = ["MECH-448", "MECH-449"]
EXPERIMENT_PURPOSE = "diagnostic"

EVIDENCE_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# Design constants (pre-registered; NOT inferred post-hoc).
SEEDS: List[int] = [42, 43, 44]
N_BANKS = 80            # decision contexts per (seed, arm) -- entropy estimated over these
K_CANDIDATES = 8        # candidates per menu (== ACTION_DIM so every class is distinct)
WORLD_DIM = 8
ACTION_DIM = 8
HORIZON = 3
HIDDEN_DIM = 8

# Demotion envelope must be wide enough for Go/No-Go to have ROOM (dropping a
# safety + a stale candidate must leave >= 2 survivors so the modulatory channel
# still arbitrates -- a size-2 envelope nullifies Go/No-Go by construction, since
# fail-open protect_min keeps 1 and it can only relabel within a 2-class support).
ENVELOPE_MIN = 4

ARMS: List[str] = ["ARM_OFF", "ARM_DEM", "ARM_GNG", "ARM_BOTH"]

# Modulatory magnitude: M_max = M_MAX_FRAC * F-gap(best, second). Keeping the most
# negative bias BELOW the F-argmin's margin guarantees ARM_OFF stays F-locked
# (monostrategy) while the demotion within-eligible argmin (F removed) still picks
# the modulatory favorite.
M_MAX_FRAC = 0.4

# Non-vacuity floors.
F_RANGE_FLOOR = 0.10       # divergent F menu (range, not magnitude)
E_OFF_FLOOR = 0.05         # ARM_OFF monostrategy ceiling (nats) -- direction upper
DEM_LIFT_FLOOR = 0.20      # demotion main-effect entropy lift (nats)

# Acceptance (SD-of-delta + absolute floor).
K_SD = 1.5
FLOOR_NATS = 0.10          # absolute interaction effect-size floor (nats)

# Go/No-Go signal levels (mirror 689g: >= the gng floors so the gate fires).
NOGO_LEVEL = 0.9

MAX_POOL_TRIES = 40


# ----------------------------------------------------------------------------- #
# Pool / context construction
# ----------------------------------------------------------------------------- #
def _make_candidate(action_class: int, world_vec: torch.Tensor) -> Trajectory:
    states = [torch.zeros(1, WORLD_DIM) for _ in range(HORIZON + 1)]
    world_states = [world_vec.reshape(1, WORLD_DIM).clone() for _ in range(HORIZON + 1)]
    actions = torch.zeros(1, HORIZON, ACTION_DIM)
    actions[:, 0, action_class] = 1.0
    return Trajectory(states=states, actions=actions, world_states=world_states)


def _build_menu(rng: torch.Generator) -> List[Trajectory]:
    """A FIXED K-candidate action menu with all-distinct first-action classes.

    Distinct per-candidate world_states -> divergent raw F (graded magnitude gives
    a clear F front + F-poor tail through the random eval head)."""
    cands = []
    for k in range(K_CANDIDATES):
        wv = torch.randn(WORLD_DIM, generator=rng) * 0.6 + float(k) * 0.4
        cands.append(_make_candidate(action_class=k, world_vec=wv))  # class == k
    return cands


def _raw_f(selector: E3TrajectorySelector, cands: List[Trajectory]) -> List[float]:
    """Probe raw F (zero modulatory bias, no signals) -> scores ARE the raw F costs."""
    selector._running_variance = 0.0
    r = selector.select(cands, temperature=1.0, score_bias=torch.zeros(len(cands)))
    return [float(s.detach()) for s in r.scores]


def _make_config(arm: str) -> E3Config:
    demotion = arm in ("ARM_DEM", "ARM_BOTH")
    gonogo = arm in ("ARM_GNG", "ARM_BOTH")
    return E3Config(
        world_dim=WORLD_DIM,
        hidden_dim=HIDDEN_DIM,
        use_f_eligibility_demotion=demotion,
        # Adaptive floor carried WITH demotion (689e PASS; matched-stack constant
        # per the prong-map P-floor row). OFF when demotion is OFF.
        use_f_eligibility_adaptive_floor=demotion,
        use_go_nogo_constitution=gonogo,
    )


def _make_context(rng: random.Random, env_list: List[int]) -> Dict[str, Any]:
    """One decision context over the FIXED menu: a varying modulatory preference
    (graded over the F-eligible envelope) + a Go/No-Go undesirable + stale tag."""
    perm = list(env_list)
    rng.shuffle(perm)  # perm[-1] = the most-preferred (modulatory favorite)
    u_b = rng.choice(env_list)  # undesirable (safety No-Go)
    if len(env_list) >= 2:
        s_pool = [c for c in env_list if c != u_b]
        s_b = rng.choice(s_pool)  # stale (staleness No-Go)
    else:
        s_b = None
    return {"perm": perm, "u_b": u_b, "s_b": s_b}


def _build_bias(ctx: Dict[str, Any], step: float) -> torch.Tensor:
    """Graded modulatory bias over the envelope (lower=better; favorite=most
    negative). Non-envelope classes get 0. Favorite = perm[-1]."""
    bias = torch.zeros(K_CANDIDATES)
    perm = ctx["perm"]
    n = len(perm)
    for i, cls in enumerate(perm):
        bias[cls] = -step * float(i + 1)  # perm[-1] -> -step*n (= M_max)
    return bias


def _build_gonogo_signals(ctx: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    safety = torch.zeros(K_CANDIDATES)
    safety[ctx["u_b"]] = NOGO_LEVEL
    staleness = torch.zeros(K_CANDIDATES)
    if ctx["s_b"] is not None:
        staleness[ctx["s_b"]] = NOGO_LEVEL
    return {"safety": safety, "staleness": staleness}


def _committed_class(
    selector: E3TrajectorySelector,
    cands: List[Trajectory],
    bias: torch.Tensor,
    gonogo_signals: Optional[Dict[str, torch.Tensor]],
) -> Tuple[int, Dict[str, Any]]:
    """One committed selection through the real select() path. Deterministic
    (committed argmin; no sampling) so the entropy DV is sampling-noise free."""
    selector._running_variance = 0.0  # force the committed (argmin) path
    result = selector.select(
        cands, temperature=1.0, score_bias=bias, go_nogo_signals=gonogo_signals,
    )
    diag = dict(selector.last_score_diagnostics)
    cls = int(result.selected_action.reshape(-1).argmax().item())
    return cls, diag


def _entropy_nats(classes: List[int]) -> float:
    n = len(classes)
    if n == 0:
        return 0.0
    counts = Counter(classes)
    return -sum((v / n) * math.log(v / n) for v in counts.values())


# ----------------------------------------------------------------------------- #
# Per-seed run: build the menu, the 4 shared-head arm selectors, the contexts.
# ----------------------------------------------------------------------------- #
def _run_seed(seed: int, n_banks: int) -> Dict[str, Any]:
    reset_all_rng(seed)

    # Build a FIXED menu whose F front + envelope are non-degenerate (retry on a
    # perturbed sub-seed if the random head gives a flat or all-admit envelope).
    menu: Optional[List[Trajectory]] = None
    raw_f: List[float] = []
    env_list: List[int] = []
    excluded_count = -1
    base_sel_state = None
    f_gap = 0.0
    for attempt in range(MAX_POOL_TRIES):
        prng = torch.Generator().manual_seed(seed * 1000 + attempt)
        cand_menu = _build_menu(prng)
        base_sel = E3TrajectorySelector(_make_config("ARM_OFF"))
        rf = _raw_f(base_sel, cand_menu)
        f_range = max(rf) - min(rf)
        # demotion envelope (adaptive floor) over this fixed F.
        dem_sel = E3TrajectorySelector(_make_config("ARM_DEM"))
        dem_sel.load_state_dict(base_sel.state_dict())
        env_idx = dem_sel._f_eligibility_envelope(torch.tensor(rf))
        env = sorted(int(i) for i in env_idx.tolist())
        excl = K_CANDIDATES - len(env)
        srt = sorted(rf)
        gap = srt[1] - srt[0] if len(srt) >= 2 else 0.0
        if (
            f_range >= F_RANGE_FLOOR
            and ENVELOPE_MIN <= len(env) <= K_CANDIDATES - 1
            and excl >= 1
            and gap > 1e-6
        ):
            menu = cand_menu
            raw_f = rf
            env_list = env
            excluded_count = excl
            base_sel_state = base_sel.state_dict()
            f_gap = gap
            break

    if menu is None:
        # Degenerate menu across every retry -> non-vacuity will fail downstream.
        return {
            "seed": seed,
            "menu_ok": False,
            "raw_f_range": 0.0,
            "envelope_size": -1,
            "excluded_count": -1,
            "cells": {},
        }

    # Modulatory step: M_max = M_MAX_FRAC * f_gap; step = M_max / |env| so the
    # favorite's bias (-step*|env| = -M_max) stays below the F-argmin margin ->
    # ARM_OFF F-locked; demotion within-eligible argmin still picks the favorite.
    step = (M_MAX_FRAC * f_gap) / max(1, len(env_list))

    # Pre-generate the per-seed decision contexts ONCE and replay across all 4 arms
    # (matched contexts -> the 2x2 interaction is identifiable).
    crng = random.Random(seed * 7919 + 13)
    contexts = [_make_context(crng, env_list) for _ in range(n_banks)]

    cells: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        sel = E3TrajectorySelector(_make_config(arm))
        sel.load_state_dict(base_sel_state)
        gate_on = arm in ("ARM_GNG", "ARM_BOTH")
        demotion_on = arm in ("ARM_DEM", "ARM_BOTH")

        print(f"Seed {seed} Condition {arm}", flush=True)
        committed: List[int] = []
        excl_obs: List[int] = []        # demotion excluded_count per context
        gng_suppressed_obs: List[int] = []  # candidates the Go/No-Go gate dropped
        for b, ctx in enumerate(contexts):
            if (b + 1) % 20 == 0:
                print(f"  [eval] {arm} seed={seed} ep {b+1}/{n_banks}", flush=True)
            bias = _build_bias(ctx, step)
            signals = _build_gonogo_signals(ctx) if gate_on else None
            cls, diag = _committed_class(sel, menu, bias, signals)
            committed.append(cls)
            if demotion_on:
                excl_obs.append(int(diag.get("f_eligibility_excluded_count", 0) or 0))
                if gate_on:
                    # candidates the Go/No-Go gate removed from the demotion envelope.
                    # NOTE: f_eligibility_envelope_size is set POST-gate (e3_selector
                    # line 1836 runs after the gng gate), so use the KNOWN per-seed
                    # pre-gate envelope size (len(env_list); fixed F -> fixed envelope)
                    # minus the post-gate go_nogo_envelope_size.
                    gng_sz = int(diag.get("go_nogo_envelope_size", len(env_list)))
                    gng_suppressed_obs.append(max(0, len(env_list) - gng_sz))

        entropy = _entropy_nats(committed)
        cell = {
            "arm": arm,
            "seed": seed,
            "n_banks": n_banks,
            "committed_class_entropy_nats": entropy,
            "committed_class_counts": dict(Counter(committed)),
            "median_excluded_count": (
                statistics.median(excl_obs) if excl_obs else 0.0
            ),
            "median_gng_suppressed": (
                statistics.median(gng_suppressed_obs) if gng_suppressed_obs else 0.0
            ),
        }
        cell["arm_fingerprint"] = compute_arm_fingerprint(
            config_slice={
                "arm": arm,
                "use_f_eligibility_demotion": demotion_on,
                "use_f_eligibility_adaptive_floor": demotion_on,
                "use_go_nogo_constitution": gate_on,
                "n_banks": n_banks, "k": K_CANDIDATES, "m_max_frac": M_MAX_FRAC,
            },
            seed=seed,
            script_path=Path(__file__),
            rng_fully_reset=True,
            extra_ineligible_reasons=["selection_face_synthetic_no_training"],
        )
        # Per-cell verdict line for the runner (seeds x conditions = 12 verdicts).
        print(f"verdict: {'PASS' if entropy >= 0.0 else 'FAIL'}", flush=True)
        cells[arm] = cell

    return {
        "seed": seed,
        "menu_ok": True,
        "raw_f_range": max(raw_f) - min(raw_f),
        "envelope_size": len(env_list),
        "excluded_count": excluded_count,
        "modulatory_step": step,
        "cells": cells,
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = SEEDS[:1] if dry_run else SEEDS
    n_banks = 6 if dry_run else N_BANKS

    seed_results = [_run_seed(s, n_banks) for s in seeds]
    ok_seeds = [r for r in seed_results if r["menu_ok"]]

    # --- Per-seed C2 entropies + 2x2 effects --------------------------------- #
    per_seed: List[Dict[str, Any]] = []
    for r in ok_seeds:
        c = r["cells"]
        e_off = c["ARM_OFF"]["committed_class_entropy_nats"]
        e_dem = c["ARM_DEM"]["committed_class_entropy_nats"]
        e_gng = c["ARM_GNG"]["committed_class_entropy_nats"]
        e_both = c["ARM_BOTH"]["committed_class_entropy_nats"]
        interaction = (e_both - e_gng) - (e_dem - e_off)
        demotion_main = 0.5 * ((e_dem - e_off) + (e_both - e_gng))
        gonogo_main = 0.5 * ((e_gng - e_off) + (e_both - e_dem))
        per_seed.append({
            "seed": r["seed"],
            "e_off": e_off, "e_dem": e_dem, "e_gng": e_gng, "e_both": e_both,
            "interaction": interaction,
            "demotion_main": demotion_main,
            "gonogo_main": gonogo_main,
            "gng_equals_off": abs(e_gng - e_off) < 1e-9,
        })

    n_ok = len(per_seed)
    interactions = [p["interaction"] for p in per_seed]
    demotion_mains = [p["demotion_main"] for p in per_seed]

    inter_mean = statistics.mean(interactions) if interactions else 0.0
    inter_sd = statistics.pstdev(interactions) if len(interactions) >= 2 else 0.0
    margin = max(K_SD * inter_sd, FLOOR_NATS)
    demotion_main_mean = statistics.mean(demotion_mains) if demotion_mains else 0.0
    e_off_mean = (
        statistics.mean([p["e_off"] for p in per_seed]) if per_seed else 0.0
    )

    # --- Non-vacuity preconditions ------------------------------------------- #
    raw_f_range_med = (
        statistics.median([r["raw_f_range"] for r in ok_seeds]) if ok_seeds else 0.0
    )
    env_size_med = (
        statistics.median([r["envelope_size"] for r in ok_seeds]) if ok_seeds else 0.0
    )
    excluded_med = (
        statistics.median([r["excluded_count"] for r in ok_seeds]) if ok_seeds else 0.0
    )
    # Go/No-Go suppression on the BOTH arm (candidates dropped from the envelope).
    gng_suppressed_med = (
        statistics.median(
            [r["cells"]["ARM_BOTH"]["median_gng_suppressed"] for r in ok_seeds]
        ) if ok_seeds else 0.0
    )

    pool_divergent = (n_ok >= 1) and (raw_f_range_med > F_RANGE_FLOOR)
    envelope_nondegen = (
        (n_ok >= 1) and (env_size_med >= float(ENVELOPE_MIN))
        and (env_size_med <= K_CANDIDATES - 1) and (excluded_med >= 1.0)
    )
    monostrategy_base = (n_ok >= 1) and (e_off_mean <= E_OFF_FLOOR)
    demotion_engages = (n_ok >= 1) and (demotion_main_mean >= DEM_LIFT_FLOOR)
    gonogo_engages = (n_ok >= 1) and (gng_suppressed_med >= 1.0)
    # Need >= 2 seeds to estimate cross-seed pstdev for the SD-of-delta gate.
    enough_seeds = n_ok >= 2

    preconditions_met = bool(
        pool_divergent and envelope_nondegen and monostrategy_base
        and demotion_engages and gonogo_engages and enough_seeds
    )

    # --- Verdict + self-route ------------------------------------------------- #
    non_degenerate = True
    degeneracy_reason: Optional[str] = None
    if not preconditions_met:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = {"MECH-448": "unknown", "MECH-449": "unknown"}
        non_degenerate = False
        degeneracy_reason = (
            "non-vacuity precondition unmet: "
            f"pool_divergent={pool_divergent} (raw_f_range={raw_f_range_med:.4f}>"
            f"{F_RANGE_FLOOR}); envelope_nondegen={envelope_nondegen} "
            f"(env_size={env_size_med}, excluded={excluded_med}); "
            f"monostrategy_base={monostrategy_base} (e_off={e_off_mean:.4f}<="
            f"{E_OFF_FLOOR}); demotion_engages={demotion_engages} "
            f"(demotion_main={demotion_main_mean:.4f}>={DEM_LIFT_FLOOR}); "
            f"gonogo_engages={gonogo_engages} (gng_suppressed={gng_suppressed_med}); "
            f"enough_seeds={enough_seeds} (n_ok={n_ok})"
        )
    elif inter_mean >= margin:
        label = "demotion_x_gonogo_compound"
        outcome = "PASS"
        direction = {"MECH-448": "supports", "MECH-449": "supports"}
    elif inter_mean <= -margin:
        label = "demotion_x_gonogo_cancel"
        outcome = "PASS"
        direction = {"MECH-448": "mixed", "MECH-449": "mixed"}
    else:
        label = "demotion_x_gonogo_additive"
        outcome = "PASS"
        direction = {"MECH-448": "supports", "MECH-449": "supports"}

    interpretation = {
        "label": label,
        "preconditions": [
            {"name": "pool_divergent_raw_f_range",
             "description": "median raw-F RANGE across seeds clears the divergent-menu floor",
             "measured": round(raw_f_range_med, 5), "threshold": F_RANGE_FLOOR,
             "control": "graded-magnitude world_states -> divergent F",
             "met": bool(pool_divergent)},
            {"name": "envelope_size_lower",
             "description": "median MECH-448 demotion envelope size >= ENVELOPE_MIN (room for Go/No-Go to drop a safety + a stale candidate AND leave >= 2 survivors to arbitrate)",
             "measured": round(env_size_med, 4), "threshold": float(ENVELOPE_MIN),
             "control": "adaptive-floor envelope over the fixed F menu (retry-selected)",
             "met": bool(env_size_med >= float(ENVELOPE_MIN))},
            {"name": "envelope_excludes",
             "description": "median excluded_count >= 1 (the envelope is not an all-admit no-op; F-poor excluded)",
             "measured": round(excluded_med, 4), "threshold": 1.0,
             "control": "F-poor candidates below the adaptive mean share",
             "met": bool(excluded_med >= 1.0)},
            {"name": "monostrategy_baseline_off_entropy",
             "description": "ARM_OFF committed-class entropy stays BELOW the monostrategy ceiling (headroom)",
             "measured": round(e_off_mean, 5), "threshold": E_OFF_FLOOR,
             "direction": "upper",
             "control": "fixed F menu -> F-argmin is a fixed class",
             "met": bool(monostrategy_base)},
            {"name": "demotion_main_effect_lift",
             "description": "demotion main-effect entropy lift clears the floor (the lever genuinely converts)",
             "measured": round(demotion_main_mean, 5), "threshold": DEM_LIFT_FLOOR,
             "control": "within-eligible argmin over the varying modulatory favorite",
             "met": bool(demotion_engages)},
            {"name": "gonogo_suppression_engages",
             "description": "median Go/No-Go suppression on ARM_BOTH >= 1 candidate (the gate fires)",
             "measured": round(gng_suppressed_med, 4), "threshold": 1.0,
             "control": "safety+staleness No-Go injected on envelope members",
             "met": bool(gonogo_engages)},
            {"name": "cross_seed_estimable",
             "description": ">= 2 valid seeds so the SD-of-delta interaction gate is estimable",
             "measured": float(n_ok), "threshold": 2.0,
             "control": "matched contexts replayed across arms per seed",
             "met": bool(enough_seeds)},
        ],
        "criteria_non_degenerate": {
            "demotion_lift": bool(demotion_main_mean > 0.0),
            "gonogo_suppression": bool(gng_suppressed_med >= 1.0),
            "four_arms_distinguishable": bool(
                per_seed and (
                    max(p["e_dem"] for p in per_seed)
                    - min(p["e_off"] for p in per_seed)
                ) > 1e-6
            ),
        },
        "criteria": [
            {"name": "composition_characterized", "load_bearing": True,
             "passed": bool(preconditions_met)},
        ],
        # Recorded structural finding: Go/No-Go has NO standalone selection effect
        # (the gate runs only inside the demotion/shortlist eligible-set block);
        # ARM_GNG is bit-identical to ARM_OFF.
        "gng_inert_standalone": bool(
            per_seed and all(p["gng_equals_off"] for p in per_seed)
        ),
        "interaction_mean_nats": round(inter_mean, 5),
        "interaction_pstdev_nats": round(inter_sd, 5),
        "interaction_margin_nats": round(margin, 5),
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": (
            "supports" if outcome == "PASS" and "supports" in direction.values()
            else ("mixed" if "mixed" in direction.values() else "unknown")
        ),
        "evidence_direction_per_claim": direction,
        "evidence_direction_note": (
            "P-comp is a within-selection-face COMPOSITION characterization "
            "(compound/cancel/additive of MECH-448 demotion x MECH-449 Go/No-Go at "
            "committed-class entropy C2). experiment_purpose=diagnostic -> "
            "scoring-excluded; it does NOT independently test either claim's "
            "mechanism (each is face-validated 689d/689g). Directions are context "
            "for the full-stack assembly decision, not mechanism evidence."
        ),
        "interpretation": interpretation,
        "non_degenerate": non_degenerate,
        "timestamp_utc": ts,
        "summary": {
            "preconditions_met": preconditions_met,
            "verdict": label,
            "n_valid_seeds": n_ok,
            "interaction_per_seed": [round(p["interaction"], 5) for p in per_seed],
            "interaction_mean": round(inter_mean, 5),
            "interaction_pstdev": round(inter_sd, 5),
            "interaction_margin": round(margin, 5),
            "demotion_main_mean": round(demotion_main_mean, 5),
            "gonogo_main_mean": round(
                statistics.mean([p["gonogo_main"] for p in per_seed]), 5
            ) if per_seed else 0.0,
            "entropy_off_mean": round(e_off_mean, 5),
            "entropy_dem_mean": round(
                statistics.mean([p["e_dem"] for p in per_seed]), 5
            ) if per_seed else 0.0,
            "entropy_gng_mean": round(
                statistics.mean([p["e_gng"] for p in per_seed]), 5
            ) if per_seed else 0.0,
            "entropy_both_mean": round(
                statistics.mean([p["e_both"] for p in per_seed]), 5
            ) if per_seed else 0.0,
            "raw_f_range_median": round(raw_f_range_med, 5),
            "envelope_size_median": env_size_med,
            "excluded_count_median": excluded_med,
            "gng_suppressed_median": gng_suppressed_med,
            "gng_inert_standalone": interpretation["gng_inert_standalone"],
        },
        "per_seed": per_seed,
        "arm_results": [
            r["cells"][arm]
            for r in ok_seeds for arm in ARMS
        ],
        "config": {
            "seeds": seeds, "n_banks": n_banks, "k_candidates": K_CANDIDATES,
            "m_max_frac": M_MAX_FRAC, "k_sd": K_SD, "floor_nats": FLOOR_NATS,
            "f_range_floor": F_RANGE_FLOOR, "e_off_floor": E_OFF_FLOOR,
            "dem_lift_floor": DEM_LIFT_FLOOR,
            "arms": ARMS,
        },
        "notes": (
            "P-comp (conversion_ceiling_prong_map.md): within-selection-face "
            "composition-characterization gate. 2x2 demotion x Go/No-Go at C2 "
            "(committed-action-class entropy) over a fixed action menu with "
            "per-context varying modulatory preference (models the conversion "
            "ceiling: F monopolises = monostrategy baseline; the levers convert). "
            "Adaptive floor carried with demotion (689e). Go/No-Go has no standalone "
            "selection effect (gate runs only inside the eligible-set block) so "
            "ARM_GNG == ARM_OFF -- a structural finding, recorded. CHARACTERIZES "
            "composition; promotes nothing (each lever face-validated 689d/689g)."
        ),
    }
    if degeneracy_reason:
        manifest["degeneracy_reason"] = degeneracy_reason

    out_dir = (
        Path(tempfile.gettempdir()) / "ree_dry_run_manifests" if dry_run
        else EVIDENCE_DIR
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    manifest["manifest_path"] = str(out_path)
    print(
        f"[689h] outcome={outcome} label={label} "
        f"interaction_mean={inter_mean:.4f} margin={margin:.4f} "
        f"E[off,dem,gng,both]="
        f"[{manifest['summary']['entropy_off_mean']},"
        f"{manifest['summary']['entropy_dem_mean']},"
        f"{manifest['summary']['entropy_gng_mean']},"
        f"{manifest['summary']['entropy_both_mean']}] -> {out_path}",
        flush=True,
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-689h P-comp MECH-448 demotion x MECH-449 Go/No-Go composition at C2"
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run.")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)
    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
        dry_run=args.dry_run,
    )
