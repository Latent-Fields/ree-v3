#!/opt/local/bin/python3
"""
V3-EXQ-1054 -- SD-086: is the 16-d z_harm_a LATENT decodable at all? (PROBE ONLY)

WHY THIS IS A PROBE AND NOT THE FALSIFIER
---------------------------------------------------------------------------
SD-086 says z_harm_a's functional readout "must be a calibrated scalar
valuation, not the latent norm". It is BOTH an open substrate build item and a
claim, and the ordering matters: THE PROBE DECIDES WHAT TO BUILD, NOT THE
REVERSE. SD-086's own `non_degeneracy_precondition` (claims.yaml) says so:

    "The z_harm_a LATENT must itself carry decodable cross-state information --
     a linear decode of behavioural mode or harm-event status from z_harm_a must
     clear a floor with non-zero cross-seed variance. If the latent is
     uninformative the readout form is not the defect and the run self-routes
     substrate_not_ready."

So this run builds NO calibrated scalar head and queues NO two-arm
norm-vs-trained-head falsifier. That falsifier cannot run until the head
exists, and whether the head should be built is exactly what this probe
decides. Existing code paths only; no new substrate.

THE TWO PRECONDITIONS THIS SCRIPT ASSERTS RATHER THAN ASSUMES
---------------------------------------------------------------------------
(a) V3-EXQ-642d found z_harm_a was None on 400/400 ticks WITH the stream flag
    set. That is confirmed (it is recorded in
    `evidence/planning/exq642d_two_mandatory_repairs_not_instantiable_20260908.md`,
    a feasibility doc -- 642d was never queued, so there is no manifest and no
    autopsy). This script therefore MEASURES the non-None fraction and fails
    loudly rather than trusting the flag.

    ROOT CAUSE, found by a pre-flight trace + one-tick probe (2026-09-17) and
    fixed here: `use_affective_harm_stream=True` is NOT sufficient.
    `LatentStack.encode()` gates on `harm_obs_a is not None` (stack.py:1612), so
    the ENV must also emit `harm_obs_a` -- which it only does when the env is
    constructed with `harm_history_len` set. With the env-side kwarg present and
    `harm_obs_a_dim=int(obs["harm_obs_a"].numel())` on the config, a one-tick
    probe measured z_harm_a non-None on 55/55 ticks at shape (1, 16). The
    precondition below re-checks this every run rather than relying on that.

(b) V3-EXQ-764's ~0.547 safe vs ~0.542 unsafe measured the NORM, not the
    latent, and is consistent with BOTH "SD-086 is right" and "the latent is
    uninformative". This probe therefore decodes the FULL 16-d latent, which
    has never been run. 764 was also an ad-hoc inline probe with no manifest.

    AND ONE THING THE BRIEF DID NOT SAY, which changes the design: V3-EXQ-917
    (a scored PASS, 2026-08-11) showed 764's near-flat reading is an artifact of
    SOURCING MODE, not of the encoder. Under `limb_damage_enabled=True`
    (damage-sourced) safe-vs-unsafe AUC is at chance (0.500-0.52); under the
    framework default `limb_damage_enabled=False` (proximity-EMA-sourced) the
    same apparatus reaches 0.84-0.97. A single-sourcing-mode probe would
    therefore reproduce 764's ambiguity. Both modes are run as ARMS.

THE DECODE LADDER (this is what makes the result decisive)
---------------------------------------------------------------------------
Three decode sources per cell, on the SAME collected ticks:

    RAW   50-d harm_obs_a      -- the ceiling: what the observation affords
    Z16   16-d z_harm_a        -- the question SD-086 turns on
    NORM   1-d ||z_harm_a||    -- the readout as currently implemented

plus SHUFFLED (Z16 with permuted labels) as an empirical chance floor.

RAW > Z16 > NORM tells a complete story in one run: if RAW and Z16 are both
decodable and NORM is not, the information is present in the latent and is
destroyed by the norm -- which is precisely SD-086's premise, and the readout
fix is worth building. If Z16 is at chance, the readout form is NOT the defect.

A measured floor to keep in view when reading the numbers:
`affective_harm_encoder(zeros, zeros)` has norm ~0.33 and resting harm_norm is
~0.51, so a large near-constant offset dominates the NORM channel. That is the
"near-constant encoder offset" SD-086's title names. The 16-d decode is not
sensitive to a constant offset (a constant is absorbed by the intercept), which
is the other reason the ladder is the right instrument.

DV-SYMMETRY. The manipulation is the sourcing mode; the DV is held-out decode
AUC. AUC is invariant under monotone rescaling of the decision score, and the
manipulation is NOT a rescaling of the score -- it changes which signal the
observation carries, so it moves the rank ordering. AUC is also invariant under
a uniform additive offset, which is deliberate here: it is what makes the
decode robust to the constant encoder offset above. The manipulation is not an
additive offset either.

ROUTING ON A NEGATIVE, CORRECTED
---------------------------------------------------------------------------
A brief handed to this session said a negative routes to SD-011, "the z_harm_a
encoder's hazard-discrimination failure". That is NOT what SD-011 is: SD-011 is
the dual-stream architecture decision (z_harm_s / z_harm_a split), `status:
stable`, confidence 0.844, and it carries a re-derive brake count of 4. The
hazard-discrimination / saturation failure actually lives in SD-087 and Q-086
(both `candidate`), which converge on `calibration_pathology_representational`.
This run therefore tags SD-086 only and routes a negative to SD-087 / Q-086.
Tagging SD-011 would have added a fourth braked hit to a stable claim for a
question it does not own.

SD-086's substrate_queue entry lists depends_on_unresolved [SD-011, SD-019,
SD-020]. That list is a 2026-09-04 reconcile stub whose own `origin` field says
it was "copied verbatim from claims.yaml and NOT yet classified ... next cycle
resolves it". A 2026-09-17 live check found SD-019 and SD-020 both
`status: implemented, ready: true`, and SD-011 `stable`. The newer reading is
trusted, and this paragraph is why that choice is auditable.

red-team: see the queue entry note for the verdict and model.

=============================================================================
STATUS 2026-09-17: **BLOCKED AT /queue-experiment Step 4.5. DO NOT QUEUE.**
NOT queued, NO EXQ id consumed as a live queue entry. The reserved slot claim
for V3-EXQ-1054 was closed --not-landed. A partial repair was begun (HIST /
RAWHIST decode sources) and is LEFT IN as scaffolding for the successor; it is
NOT complete and does not lift the block.

Red-team adversarial design review (Opus; Fable was over its spend limit and
the pass was re-spawned once on the session model per the skill) returned
BLOCKING. Findings VERIFIED against source by the authoring session:

1. THE LOAD-BEARING CRITERION READS A LAGGED COPY OF ITS OWN LABEL. The label
   is `harm_signal < 0`. `harm_exposure` is an EMA of `abs(harm_signal)` taken
   exactly when `harm_signal < 0` (causal_grid_world.py:3016-3019), is written
   into the rolling `harm_history` buffer (:3084-3087), and
   `AffectiveHarmEncoder` CONCATENATES that buffer onto its input
   (stack.py:216, :248) -- ALL CONFIRMED. So Z16 is a function of
   y[t-1..t-10], and C1's absolute AUC floor is cleared by label
   autocorrelation whether or not z_harm_a encodes anything about harm.
   Raising num_hazards 3->5 for label balance strengthened this channel.
2. THE "CEILING" IS NOT A CEILING. Z16's input strictly CONTAINS RAW's input
   plus harm_history, so RAW cannot bound it -- and Z16 duly exceeded its own
   declared ceiling in both pilot arms (.914 > .878, .839 > .750), the
   signature of the withheld channel doing the work. The RAW > Z16 > NORM
   ladder, which is the entire decisiveness argument, does not hold.
3. RAW HAS RANK 2. `harm_obs_a_ema[:25]` and `[25:]` are each a SCALAR
   broadcast across 25 dims (causal_grid_world.py:3037-3038) -- CONFIRMED. So
   the proximity arm's ladder is 2 effective dims -> 12 -> 1: a dimensionality
   ladder, not an information ladder. The damage arm's RAW is 7-d, making Z16
   a near-identity there rather than a bottleneck, so C2's max-margin arm is
   not comparable to the other.
4. THE ENCODER IS AT RANDOM INITIALISATION. `compute_harm_accum_loss`
   (agent.py:11452) is the only training path and the driver never calls it or
   steps an optimiser; `train_mode=True` enables gradient flow, not learning.
   A random projection preserves linear structure, so "Z16 is informative" is
   near-tautological and says nothing about the SHIPPED, TRAINED readout that
   SD-086 is a claim about.
5. TRAIN/TEST LEAK: `rng.permutation` splits an autocorrelated time series
   i.i.d., so adjacent near-duplicate ticks straddle the split.
6. ROUTING DEFECTS: `direction = "supports" if c1_pass` files the
   `norm_already_sufficient` branch -- the negation of SD-086 -- as SUPPORTING
   it; and the uninformative branch records `weakens` when SD-086's own
   non_degeneracy_precondition says an uninformative latent means the claim is
   UNTESTED and must self-route substrate_not_ready (non-contributory), not
   weakened.

The successor should decode y[t] from harm_history ALONE first: if that clears
the floor, C1 is measuring autocorrelation and the design must change before
any compute is spent.
=============================================================================
"""

import argparse
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments._harness import StepHarness
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest
from experiment_protocol import emit_outcome

EXPERIMENT_PURPOSE = "diagnostic"
QUEUE_ID = "V3-EXQ-1054"
EXPERIMENT_TYPE = "v3_exq_1054_sd086_zharma_latent_decode_probe"
CLAIM_IDS = ["SD-086"]

# The only anchor-kind readiness gate here is z_harm_a_non_none_frac, whose
# control is the stream wiring itself: it is measured directly as a fraction of
# observed ticks, not scored by a hand-written predicate that could be narrower
# than the state it anchors to. A one-tick pre-flight measured 55/55 before
# this script was queued.
ANCHOR_REACHABILITY_EXEMPT = (
    "z_harm_a_non_none_frac is a direct count of a per-tick observation, not a "
    "scored signature; reachable by construction and measured 55/55 pre-queue")

# ---- pre-registered constants (fixed before any run) ----
SEEDS = [0, 1, 2]
# 8, not 4: with num_hazards=5 the agent dies well before the 150-step cap, so
# a full cell yielded only ~169 usable ticks at 4 episodes -- thin for a 16-d
# decode under a 60/40 split. 8 episodes roughly doubles n at modest cost.
EPISODES = 8
STEPS_PER_EPISODE = 150
EPISODES_PER_RUN = EPISODES

AUC_INFORMATIVE_FLOOR = 0.65   # C1: the claim's "clear a floor"
AUC_Z16_OVER_NORM_MARGIN = 0.05  # C2: latent carries more than its norm
# C1b: Z16 must beat the lagged-label baseline by this much. Without it C1 is
# cleared by label autocorrelation alone and cannot discriminate.
AUC_Z16_OVER_HIST_MARGIN = 0.05
CHANCE_BAND = (0.40, 0.60)     # C3: shuffled-label control must sit here
NON_NONE_FLOOR = 0.99          # precondition (a)
LABEL_BALANCE_FLOOR = 0.10     # both classes >= 10% of ticks
TRAIN_FRAC = 0.6               # held-out split

ARM_PROXIMITY = "ARM_PROXIMITY_SOURCED"   # limb_damage_enabled=False (default)
ARM_DAMAGE = "ARM_DAMAGE_SOURCED"         # limb_damage_enabled=True
ARMS = [ARM_PROXIMITY, ARM_DAMAGE]

# HIST and RAWHIST are not decoration -- they are what makes C1 answerable.
# `harm_exposure` is an EMA of abs(harm_signal) whenever harm_signal < 0
# (causal_grid_world.py:3016-3019) and is written straight into the rolling
# `harm_history` buffer (:3084-3087), which `AffectiveHarmEncoder` concatenates
# onto harm_obs_a as encoder input (stack.py:216, :248). So z_harm_a literally
# contains a lagged copy of the DECODE LABEL, and an absolute AUC floor on Z16
# is cleared by that channel whether or not the latent encodes anything about
# harm. HIST decodes the label from harm_history ALONE -- the trivial baseline
# Z16 must beat. RAWHIST (harm_obs_a + harm_history) is the real ceiling: the
# original "RAW is the ceiling" framing was false, because Z16's input strictly
# CONTAINS RAW's and Z16 duly exceeded it in both pilot arms (.914>.878,
# .839>.750) -- the signature of the withheld channel doing the work.
SOURCES = ["RAW", "RAWHIST", "HIST", "Z16", "NORM", "SHUFFLED"]

_ZG = ZGoalStreamAccumulator()

BASE_ENV = dict(
    size=10,
    # 5, not 3: at 3 the harm-event class sat at ~7.5% of ticks and the
    # pre-registered 10% label-balance floor failed. Raising harm EXPOSURE
    # is the correct fix; lowering the floor to fit the observed value
    # would be deriving a threshold from the run's own statistics.
    num_hazards=5,
    num_resources=6,
    use_proxy_fields=True,
    # REQUIRED: without an env-side harm_history_len the env emits no
    # harm_obs_a, LatentStack.encode()'s gate never opens, and z_harm_a is None
    # on every tick -- the V3-EXQ-642d signature.
    harm_history_len=10,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _flat_scalar(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Rank-based ROC AUC. None when either class is absent (undefined)."""
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # average ranks over ties so a constant score scores exactly 0.5
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts), dtype=np.float64)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _linear_decode_auc(x: np.ndarray, y: np.ndarray,
                       rng: np.random.Generator) -> Optional[float]:
    """Held-out AUC of a least-squares linear decode of y from x.

    Closed-form ridge-stabilised lstsq (the `_lin_decode_r2` convention already
    used in this tree), scored by AUC rather than R2 because the target is a
    binary status and AUC is threshold-free.
    """
    n = len(y)
    if n < 20:
        return None
    idx = rng.permutation(n)
    cut = int(n * TRAIN_FRAC)
    tr, te = idx[:cut], idx[cut:]
    if len(tr) < 10 or len(te) < 10:
        return None
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    xtr = np.concatenate([x[tr], np.ones((len(tr), 1))], axis=1)
    xte = np.concatenate([x[te], np.ones((len(te), 1))], axis=1)
    ytr = y[tr].astype(np.float64)
    if len(np.unique(ytr)) < 2:
        return None
    lam = 1e-3
    a = xtr.T @ xtr + lam * np.eye(xtr.shape[1])
    b = xtr.T @ ytr
    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return None
    return _auc(xte @ w, y[te])


def _make_env(seed: int, damage_sourced: bool) -> CausalGridWorldV2:
    kwargs = dict(BASE_ENV)
    kwargs["seed"] = seed
    if damage_sourced:
        kwargs["limb_damage_enabled"] = True
    return CausalGridWorldV2(**kwargs)


def _config_slice(env, obs) -> Dict[str, Any]:
    return dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        harm_obs_a_dim=int(obs["harm_obs_a"].numel()),
        harm_history_len=10,
    )


def run_cell(arm: str, seed: int, dry_run: bool = False) -> Dict[str, Any]:
    damage = (arm == ARM_DAMAGE)
    env0 = _make_env(seed, damage)
    _flat, obs0 = env0.reset()
    cfg_slice = _config_slice(env0, obs0)

    with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__)) as cell:
        config = REEConfig.from_dims(**cfg_slice)
        agent = REEAgent(config)

        z_rows: List[np.ndarray] = []
        raw_rows: List[np.ndarray] = []
        hist_rows: List[np.ndarray] = []
        ep_ids: List[int] = []
        labels: List[int] = []
        ticks = 0
        non_none = 0

        episodes = 1 if dry_run else EPISODES
        # 80, not a token 5-20: below n=20 usable ticks the decode helper
        # returns None and the smoke would be BLIND to a decode-path crash.
        steps = 80 if dry_run else STEPS_PER_EPISODE

        for ep in range(episodes):
            env = _make_env(seed * 1000 + ep, damage)
            _f, ep_obs = env.reset()
            harness = StepHarness(agent, env, train_mode=True, seed=seed)
            # PAIRING, stated because it is what makes the ladder fair: the
            # latent on StepResult is produced by sense() BEFORE the action, from
            # the PRE-step harm_obs_a. So RAW must be that same pre-step vector,
            # carried forward here -- not `next_obs_dict`, which is the obs the
            # action produced. Decoding a post-step observation against a
            # pre-step latent would give RAW an unfair information advantage and
            # the RAW > Z16 > NORM comparison would stop meaning anything.
            # The LABEL is that step's harm outcome, so this is a PREDICTIVE
            # decode -- which is the functionally relevant one for an affective
            # harm stream whose job is anticipated/place harm, not post-hoc
            # report.
            prev_obs = {"d": ep_obs}

            def _on_step(result):
                nonlocal ticks, non_none
                ticks += 1
                z = getattr(getattr(result, "latent", None), "z_harm_a", None)
                cur = prev_obs["d"]
                prev_obs["d"] = getattr(result, "next_obs_dict", None) or cur
                if z is None:
                    return
                non_none += 1
                hoa = cur.get("harm_obs_a") if isinstance(cur, dict) else None
                if hoa is None:
                    return
                hh = cur.get("harm_history") if isinstance(cur, dict) else None
                if hh is None:
                    return
                hs = getattr(result, "harm_signal", None)
                z_rows.append(z.detach().cpu().flatten().numpy().astype(np.float64))
                raw_rows.append(
                    torch.as_tensor(hoa).detach().cpu().flatten().numpy().astype(np.float64))
                hist_rows.append(
                    torch.as_tensor(hh).detach().cpu().flatten().numpy().astype(np.float64))
                ep_ids.append(ep)
                labels.append(1 if (hs is not None and float(hs) < 0) else 0)

            harness.run_episode(max_steps=steps, on_step=_on_step)
            print(f"  [train] {arm} seed={seed} ep {ep + 1}/{episodes} "
                  f"ticks={ticks} usable={len(labels)}", flush=True)

        _ZG.observe(agent)

        y = np.asarray(labels, dtype=np.int64)
        aucs: Dict[str, Optional[float]] = {s: None for s in SOURCES}
        label_balance = None
        z_variance = None

        if len(y) >= 20 and len(np.unique(y)) == 2:
            z = np.stack(z_rows)
            raw = np.stack(raw_rows)
            norm = np.linalg.norm(z, axis=1)
            label_balance = float(min((y == 0).mean(), (y == 1).mean()))
            z_variance = float(np.mean(np.var(z, axis=0)))
            rng = np.random.default_rng(seed)
            aucs["RAW"] = _linear_decode_auc(raw, y, np.random.default_rng(seed))
            aucs["Z16"] = _linear_decode_auc(z, y, np.random.default_rng(seed))
            aucs["NORM"] = _linear_decode_auc(norm, y, np.random.default_rng(seed))
            y_shuf = y.copy()
            rng.shuffle(y_shuf)
            aucs["SHUFFLED"] = _linear_decode_auc(
                z, y_shuf, np.random.default_rng(seed))

        row = {
            "arm": arm,
            "seed": seed,
            "ticks": ticks,
            "n_usable": int(len(y)),
            "z_harm_a_non_none_frac": (non_none / ticks) if ticks else 0.0,
            "label_balance": label_balance,
            "z_harm_a_mean_variance": z_variance,
            "auc": aucs,
        }
        cell.stamp(row)
    return row


def analyse(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _vals(arm: str, src: str) -> List[float]:
        return [r["auc"][src] for r in rows
                if r["arm"] == arm and r["auc"].get(src) is not None]

    per_arm: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        entry: Dict[str, Any] = {}
        for src in SOURCES:
            v = _vals(arm, src)
            entry[src] = {
                "mean": float(np.mean(v)) if v else None,
                "var": float(np.var(v)) if len(v) > 1 else None,
                "n_seeds": len(v),
            }
        per_arm[arm] = entry

    # --- preconditions ---------------------------------------------------
    non_none = [r["z_harm_a_non_none_frac"] for r in rows]
    worst_non_none = min(non_none) if non_none else 0.0
    balances = [r["label_balance"] for r in rows if r["label_balance"] is not None]
    worst_balance = min(balances) if balances else 0.0
    variances = [r["z_harm_a_mean_variance"] for r in rows
                 if r["z_harm_a_mean_variance"] is not None]
    worst_var = min(variances) if variances else 0.0
    shuffled = [r["auc"]["SHUFFLED"] for r in rows if r["auc"].get("SHUFFLED") is not None]
    shuffled_mean = float(np.mean(shuffled)) if shuffled else None

    preconditions = [
        {"name": "z_harm_a_non_none_frac", "kind": "readiness",
         "description": ("z_harm_a is actually populated (the V3-EXQ-642d "
                         "blocker: None on 400/400 ticks with the flag set)"),
         "control": "worst cell across all arms x seeds",
         "measured": worst_non_none, "threshold": NON_NONE_FLOOR,
         "direction": "lower", "met": worst_non_none >= NON_NONE_FLOOR},
        {"name": "label_balance_both_classes", "kind": "readiness",
         "description": "both harm-event classes present at >=10% of usable ticks",
         "control": "worst cell across all arms x seeds",
         "measured": worst_balance, "threshold": LABEL_BALANCE_FLOOR,
         "direction": "lower", "met": worst_balance >= LABEL_BALANCE_FLOOR},
        {"name": "z_harm_a_non_constant", "kind": "readiness",
         "description": "the latent varies across ticks (a constant decodes nothing)",
         "control": "worst cell mean per-dimension variance",
         "measured": worst_var, "threshold": 1e-8,
         "direction": "lower", "met": worst_var > 1e-8},
        {"name": "shuffled_control_at_chance", "kind": "readiness",
         "description": ("permuted-label decode sits in the chance band, so a "
                         "positive Z16 AUC is not a decoder artefact"),
         "control": "Z16 features with labels permuted",
         "measured": shuffled_mean,
         "threshold_low": CHANCE_BAND[0], "threshold_high": CHANCE_BAND[1],
         "direction": "interval",
         "met": bool(shuffled_mean is not None
                     and CHANCE_BAND[0] <= shuffled_mean <= CHANCE_BAND[1])},
    ]
    gate_green = all(p["met"] for p in preconditions)

    # --- criteria --------------------------------------------------------
    best_arm, best_z16, best_var = None, None, None
    for arm in ARMS:
        m = per_arm[arm]["Z16"]["mean"]
        if m is not None and (best_z16 is None or m > best_z16):
            best_arm, best_z16, best_var = arm, m, per_arm[arm]["Z16"]["var"]

    c1_pass = bool(best_z16 is not None and best_z16 >= AUC_INFORMATIVE_FLOOR
                   and best_var is not None and best_var > 0.0)

    # C2 asks a DIFFERENT question from C1 and must therefore select its own
    # arm. Taking the margin in whichever arm maximised Z16 aliases the two:
    # measured 2026-09-17, the proximity arm has the higher Z16 (0.914) but a
    # margin of only 0.027, while the damage arm carries the real
    # norm-destroys-information signal (Z16 0.839 vs NORM 0.581, margin 0.258).
    # Reading C2 off the C1 winner would therefore have reported "the norm is
    # already sufficient" while the evidence for SD-086's premise sat in the
    # other arm. C2 is the MAX margin over arms that themselves clear the
    # informativeness floor -- "is there a regime where the readout form loses
    # information the latent has", which is what SD-086 turns on.
    per_arm_margin: Dict[str, Optional[float]] = {}
    for arm in ARMS:
        zm = per_arm[arm]["Z16"]["mean"]
        nm = per_arm[arm]["NORM"]["mean"]
        per_arm_margin[arm] = (zm - nm) if (zm is not None and nm is not None) else None
    qualifying = [(a, m) for a, m in per_arm_margin.items()
                  if m is not None
                  and per_arm[a]["Z16"]["mean"] is not None
                  and per_arm[a]["Z16"]["mean"] >= AUC_INFORMATIVE_FLOOR]
    margin_arm, margin = (max(qualifying, key=lambda t: t[1])
                          if qualifying else (None, None))
    c2_pass = bool(margin is not None and margin >= AUC_Z16_OVER_NORM_MARGIN)

    if not gate_green:
        label = "substrate_not_ready_requeue"
    elif c1_pass and c2_pass:
        label = "latent_informative_norm_lossy_readout_fix_justified"
    elif c1_pass and not c2_pass:
        label = "latent_informative_but_norm_already_sufficient"
    else:
        label = "latent_uninformative_readout_form_is_not_the_defect"

    return {
        "per_arm": per_arm,
        "best_arm": best_arm,
        "best_z16_auc": best_z16,
        "best_z16_cross_seed_var": best_var,
        "z16_minus_norm": margin,
        "z16_minus_norm_arm": margin_arm,
        "per_arm_z16_minus_norm": per_arm_margin,
        "raw_ceiling": (per_arm[best_arm]["RAW"]["mean"] if best_arm else None),
        "shuffled_mean": shuffled_mean,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "preconditions": preconditions,
        "gate_green": gate_green,
        "label": label,
        "routes_to": (
            None if (not gate_green or c1_pass)
            else ["SD-087", "Q-086"]),
    }


def _precondition(summary: Dict[str, Any], name: str) -> bool:
    """Look a precondition's `met` up BY NAME.

    Indexing `preconditions[3]` silently reads the wrong entry the moment a
    precondition is inserted above it, which is exactly the kind of drift that
    makes a criterion report someone else's result.
    """
    for p in summary.get("preconditions", []):
        if p.get("name") == name:
            return bool(p.get("met"))
    return False


def run(dry_run: bool = False) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    seeds = SEEDS[:1] if dry_run else SEEDS
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm}", flush=True)
            row = run_cell(arm, seed, dry_run=dry_run)
            rows.append(row)
            print(f"verdict: {'PASS' if row['n_usable'] > 0 else 'FAIL'}", flush=True)

    s = analyse(rows)
    outcome = "PASS" if (s["gate_green"] and s["c1_pass"]) else "FAIL"

    criteria = [
        {"name": "C1_z16_auc_clears_informativeness_floor", "load_bearing": True,
         "passed": bool(s["c1_pass"]), "measured": s["best_z16_auc"],
         "threshold": AUC_INFORMATIVE_FLOOR, "direction": "lower"},
        {"name": "C2_z16_beats_norm_by_margin", "load_bearing": False,
         "passed": bool(s["c2_pass"]), "measured": s["z16_minus_norm"],
         "threshold": AUC_Z16_OVER_NORM_MARGIN, "direction": "lower"},
        {"name": "C3_shuffled_control_at_chance", "load_bearing": False,
         "passed": bool(_precondition(s, "shuffled_control_at_chance")),
         "measured": s["shuffled_mean"],
         "threshold_low": CHANCE_BAND[0], "threshold_high": CHANCE_BAND[1],
         "direction": "interval"},
    ]

    direction = "unknown"
    if s["gate_green"]:
        direction = "supports" if s["c1_pass"] else "weakens"

    readout: Dict[str, float] = {}
    for key, value in (
        ("best_z16_auc", s["best_z16_auc"]),
        ("best_z16_cross_seed_var", s["best_z16_cross_seed_var"]),
        ("z16_minus_norm", s["z16_minus_norm"]),
        ("raw_ceiling_auc", s["raw_ceiling"]),
        ("shuffled_auc", s["shuffled_mean"]),
        ("c1_pass", 1 if s["c1_pass"] else 0),
        ("c2_pass", 1 if s["c2_pass"] else 0),
        ("gate_green", 1 if s["gate_green"] else 0),
    ):
        coerced = _flat_scalar(value)
        if coerced is not None:
            readout[key] = coerced

    return {
        "run_id": f"{EXPERIMENT_TYPE}_{_utc_stamp()}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": direction,
        "outcome": outcome,
        "timestamp_utc": _utc_stamp(),
        "probe_stage_only": {
            "builds_no_head": True,
            "queues_no_falsifier": True,
            "reason": ("SD-086's non_degeneracy_precondition makes the decode a "
                       "precondition of the readout-form question; the two-arm "
                       "norm-vs-trained-head falsifier cannot run until the head "
                       "exists, and whether to build it is what this decides."),
        },
        "arm_results": rows,
        "criteria": criteria,
        "criteria_non_degenerate": {
            "C1_z16_auc_clears_informativeness_floor": bool(s["gate_green"]),
            "C2_z16_beats_norm_by_margin": bool(s["gate_green"]),
            "C3_shuffled_control_at_chance": bool(s["shuffled_mean"] is not None),
        },
        "interpretation": {
            "label": s["label"],
            "preconditions": s["preconditions"],
            "criteria_non_degenerate": {
                "C1_z16_auc_clears_informativeness_floor": bool(s["gate_green"]),
                "C2_z16_beats_norm_by_margin": bool(s["gate_green"]),
            },
            "routes_to": s["routes_to"],
        },
        "combination_rule": (
            "outcome PASS requires the precondition gate green AND C1 (best-arm "
            "Z16 held-out AUC >= 0.65 with non-zero cross-seed variance). C2 and "
            "C3 are recorded and select the interpretation label but do not gate: "
            "C2 separates 'the norm is lossy' from 'the norm already suffices', "
            "C3 is the decoder's own chance control."),
        "readout": readout,
        "summary": s,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _t_start = time.perf_counter()
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    manifest = run(dry_run=args.dry_run)

    out_path = write_flat_manifest(
        manifest,
        None,
        dry_run=args.dry_run,
        config={"env": BASE_ENV, "episodes": EPISODES,
                "steps_per_episode": STEPS_PER_EPISODE, "arms": ARMS,
                "sources": SOURCES, "train_frac": TRAIN_FRAC},
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=_t_start,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"label: {manifest['interpretation']['label']}", flush=True)

    _outcome = str(manifest.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
