#!/opt/local/bin/python3
"""
V3-EXQ-1053 -- MECH-042 sub-claim (2): does telemetry LEAD behaviour?

WHY THIS RUN TESTS ONLY HALF THE CLAIM, AND WHERE THE OTHER HALF WENT
---------------------------------------------------------------------------
MECH-042 ("Telemetry exposure channels report internal control-plane state for
diagnostics") has two CONFIRMING sub-claims, both required. Its own disposition
in claims.yaml (2026-09-16) routes them to different artifact types:

    "testable now and cheap; complicated (buildable). Sub-claim (1) alone is a
     contract-test-shaped check and could land as a ree-v3 contract test rather
     than an EXQ; sub-claim (2) is the experiment."

Sub-claim (1) -- READ-ONLY / NON-PARTICIPATION, "reading every telemetry channel
each tick versus never reading any of them yields a BIT-IDENTICAL action, commit
and weight trajectory" -- landed as
`tests/contracts/test_mech042_telemetry_readonly_nonparticipation.py` (5 checks,
including the claim's specifically-named `_last_control_vector` risk). It is a
bit-identity contract, so a contract test is a STRICTER instrument than an
experiment: it fails the build rather than recording a number. That file was
mutation-tested before landing (injecting one `torch.rand(1)` into the read path
broke C1 and C2), so its green is not vacuous.

THIS script is sub-claim (2), the DEVELOPMENTAL-SAFETY VALUE half:

    "for an injected control-plane pathology ... a pre-registered detector on
     the telemetry stream flags it with POSITIVE lead time over a matched
     detector on the behaviour stream (harm / reward), sign-consistent across
     >=3 seeds and >=2 pathology types, at matched false-alarm rate."

FALSIFYING (2): "telemetry gives ZERO lead time over behaviour for every
pathology (the channels expose nothing behaviour does not already show), in
which case the mechanism exists but its justification does not, and the claim
should be narrowed to 'diagnostic convenience'."

THE ONE DESIGN DECISION THAT DECIDES WHETHER THIS RUN MEANS ANYTHING
---------------------------------------------------------------------------
A telemetry detector pointed at the channel the pathology is injected INTO
would report a positive lead time by construction: you would be detecting the
injection at the injection site, on the tick it happens, and "telemetry leads
behaviour" would be an arithmetic identity rather than a measurement. That is
the aliasing failure family -- a criterion that cannot discriminate because the
manipulation reaches the DV through a trivial path.

So the LOAD-BEARING telemetry detector EXCLUDES the injected channel, per
pathology, pre-registered in INJECTED_CHANNELS below. It asks the genuinely
open question: does a control-plane pathology PROPAGATE to the rest of the
telemetry surface before it shows up in harm/reward? The full-surface detector
(injected channel included) is still computed and recorded as
`lead_time_full_surface`, explicitly NON-load-bearing, so a reader can see both
numbers and the difference between them.

MATCHED FALSE-ALARM RATE, which the claim makes a degeneracy condition:
    "Degenerate if the behaviour-stream detector is given a looser threshold
     than the telemetry detector (match false-alarm rates on control runs
     first)."
Both detectors are the same functional form (max abs z-score against a
pre-injection baseline, sustained CONSEC ticks). Neither threshold is chosen by
hand: each is calibrated on the CONTROL arm's own post-baseline window to the
same target false-alarm rate, TARGET_FA. The realised FA rates are recorded and
their gap is a precondition, so an unmatched pair vacates the run instead of
flattering the telemetry side.

DV-SYMMETRY. Manipulation = the injected pathology; DV = a DIFFERENCE of two
first-crossing tick indices. A uniform additive offset applied to every
telemetry channel would move both the baseline mean and the live value and is
absorbed by the z-score -- deliberately, since that is what makes the detector
scale-free -- but the pathologies are not uniform offsets: a precision collapse
and a mode lock each change a subset of channels relative to the others, which
is exactly what a per-channel z-score is sensitive to. The DV is a difference of
ranks in TIME, not of magnitudes, so a monotone rescaling of either stream
leaves it unchanged; the manipulation is not a rescaling.

SUBSTRATE-PATH OVERLAP (Step 2.5c), recorded rather than silently passed:
`substrate_queue.json` currently lists SD-082 as `severity: corrupting` with
`substrate_paths` including `ree_core/predictors/e3_selector.py`, which
`agent.get_state()` reads (current_precision, _running_variance). Disposed of as
NOT REACHABLE, and the reasoning is in the queue entry note -- the short version
is that SD-082's entry ALSO carries a `severity_note_2026_09_09` recording that
its footprint was deliberately emptied precisely because it was gating unrelated
work, and its 09-15 and 09-16 governance notes both assert `substrate_paths` is
`[]`, which the live value contradicts. That inconsistency is raised as a
governance flag rather than resolved here.

red-team: see the queue entry note for the verdict and model.

=============================================================================
STATUS 2026-09-17: **BLOCKED AT /queue-experiment Step 4.5. DO NOT QUEUE.**
NOT queued, NO EXQ id consumed as a live queue entry. The reserved slot claim
for V3-EXQ-1053 was closed --not-landed.

Red-team adversarial design review (Opus; Fable was over its spend limit and
the pass was re-spawned once on the session model per the skill) returned
BLOCKING. Findings VERIFIED against source by the authoring session:

1. THE MODE_LOCK ARM IS INERT -- it is a second control. `_inject` iterates
   `sal.config.enter_thresholds`, which is `field(default_factory=dict)` and
   is never populated by agent.py, so the `{"focused","diffuse"}` fallback
   fires and writes DEAD KEYS: the real mode names are `external_task`,
   `internal_planning`, `internal_replay`, `offline_consolidation`
   (salience_coordinator.py:75-80, bound to `self.mode_names` at :316) --
   CONFIRMED. The pilot corroborates it exactly: CONTROL and MODE_LOCK both
   ran 221 ticks, bit-identical. Worse, even a CORRECT injection would be
   invisible: `operating_mode` is a softmax computed at :489-490, upstream of
   and independent from the enter-threshold gate at :508-522, which acts only
   on `_current_mode` -- a quantity not in TELEMETRY_KEYS.
2. THE INJECTED-CHANNEL EXCLUSION IS INCOMPLETE, rebuilding the tautology it
   exists to prevent. `recalibrate_precision_to(0.01, step=1.0)` hard-sets
   `_running_variance` to ~1e2, and `committed = commit_variance <
   effective_threshold` reads that same variable (e3_selector.py:3843-3847) --
   CONFIRMED. So the RETAINED channel `is_committed` becomes the arithmetic
   identity 1[injected_value < threshold], with `beta_elevated` inheriting it.
   The pilot's t_telemetry = 163 = 150 + 13 is the signature of a step channel
   crossing a 15-tick trailing mean, not of propagation.
3. CENSORING IS AT A CELL-DEPENDENT n THAT THE MANIPULATION ITSELF SETS.
   Episodes end on `done`, so tick count is an OUTCOME: the never-committing
   precision arm survived 366 ticks vs the control's 221. lead = n - t_tel, so
   a cell that merely runs longer manufactures a larger positive lead (+203
   here would be +58 at the control's horizon). Leads are not commensurable
   across arms yet are pooled as if they were.
4. THE MATCHED-FALSE-ALARM GATE CANNOT FAIL. tau is the 0.95 quantile of a
   series and the FA rate is then measured on that SAME series, so both sides
   equal ~floor(0.05(m-1))/m by arithmetic -- the pilot's 0.0495/0.0495 gap of
   exactly 0.0 is a property of m, not of the detectors. The claim's own named
   degeneracy guard is therefore inert. It is also calibrated on the 9-column
   control statistic and applied to 7- and 8-column pathology statistics.

C1 requires positive leads on >=2 pathology arms; one arm is inert and the
other's lead is a definitional identity scaled by outcome-dependent censoring,
so no branch maps to a verdict the run can support.
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

EXPERIMENT_PURPOSE = "evidence"
QUEUE_ID = "V3-EXQ-1053"
EXPERIMENT_TYPE = "v3_exq_1053_mech042_telemetry_lead_time_falsifier"
CLAIM_IDS = ["MECH-042"]

# ---- pre-registered constants (fixed before any run) ----
SEEDS = [0, 1, 2]
EPISODES = 6
STEPS_PER_EPISODE = 150
EPISODES_PER_RUN = EPISODES

BASELINE_TICKS = 120      # pre-injection window the z-scores are built from
INJECT_AT_TICK = 150      # global tick index at which the pathology starts
CONSEC = 3                # sustained crossings required to call a detection
TARGET_FA = 0.05          # both detectors calibrated to this on CONTROL
FA_GAP_TOLERANCE = 0.03   # |FA_tel - FA_beh| must not exceed this
MIN_LEAD_SEEDS = 3        # claim: sign-consistent across >=3 seeds
MIN_PATHOLOGIES = 2       # claim: and >=2 pathology types
# Trailing-window smoothing applied to BOTH streams before z-scoring. The claim
# says the pathology must show as a departure in "harm RATE or reward", not in a
# single tick's harm signal, and an unsmoothed per-tick harm channel is so noisy
# that the behaviour detector never fired at all in the pre-queue pilot (0/2
# pathology cells) -- which vacates lead time by the claim's own precondition.
# Applied SYMMETRICALLY on purpose: smoothing adds lag, so applying it to the
# behaviour stream alone would hand the telemetry side an artificial head start.
# Both sides pay the same lag.
SMOOTH_W = 15

ARM_CONTROL = "ARM_CONTROL"
ARM_PRECISION = "ARM_PRECISION_COLLAPSE"
ARM_MODE_LOCK = "ARM_MODE_LOCK"
PATHOLOGY_ARMS = [ARM_PRECISION, ARM_MODE_LOCK]
ARMS = [ARM_CONTROL] + PATHOLOGY_ARMS

TELEMETRY_KEYS = [
    "precision", "running_variance", "is_committed", "beta_elevated",
    "e3_steps_per_tick", "commit_readiness", "salience_mode_prob",
    "residue_coverage_pct", "harm_benefit_ratio",
]

# Pre-registered per-pathology exclusion. The LOAD-BEARING detector must not
# read the channel the pathology is injected into -- see the module docstring.
INJECTED_CHANNELS = {
    ARM_PRECISION: ["precision", "running_variance"],  # both are E3-derived
    ARM_MODE_LOCK: ["salience_mode_prob"],
    ARM_CONTROL: [],
}

COLLAPSE_TARGET_PRECISION = 0.01   # the injected precision collapse
MODE_LOCK_THRESHOLD = 999.0        # unreachable enter-threshold => mode locked

_ZG = ZGoalStreamAccumulator()

ENV_KWARGS = dict(
    size=10, num_hazards=3, num_resources=6,
    use_proxy_fields=True, harm_history_len=10,
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


def _telemetry_vector(agent) -> Dict[str, float]:
    """Read every MECH-042 channel once. Read-only (sub-claim (1) is the
    contract test that pins that; this run relies on it)."""
    st = agent.get_state()
    out = {
        "precision": float(st.precision),
        "running_variance": float(st.running_variance),
        "is_committed": 1.0 if st.is_committed else 0.0,
        "beta_elevated": 1.0 if st.beta_elevated else 0.0,
        "e3_steps_per_tick": float(st.e3_steps_per_tick),
    }
    cr = getattr(agent, "commit_readiness", None)
    out["commit_readiness"] = float(cr.get_state().get("readiness", 0.0)) if cr else 0.0
    sal = getattr(agent, "salience", None)
    if sal is not None:
        try:
            mode = dict(sal.operating_mode)
            out["salience_mode_prob"] = float(max(mode.values())) if mode else 0.0
        except Exception:
            out["salience_mode_prob"] = 0.0
    else:
        out["salience_mode_prob"] = 0.0
    try:
        cov = agent.residue_field.get_coverage_telemetry()
        out["residue_coverage_pct"] = float(cov.get("residue_coverage_pct", 0.0))
        out["harm_benefit_ratio"] = float(cov.get("harm_benefit_ratio", 0.0))
    except Exception:
        out["residue_coverage_pct"] = 0.0
        out["harm_benefit_ratio"] = 0.0
    return out


def _first_crossing(series: np.ndarray, tau: float, start: int) -> Optional[int]:
    """First index >= start at which `series` exceeds tau for CONSEC ticks."""
    run = 0
    for i in range(start, len(series)):
        if series[i] > tau:
            run += 1
            if run >= CONSEC:
                return i - CONSEC + 1
        else:
            run = 0
    return None


def _smooth(matrix: np.ndarray, w: int = SMOOTH_W) -> np.ndarray:
    """Trailing-mean smoothing, column-wise. Applied to BOTH streams."""
    if matrix.size == 0 or w <= 1:
        return matrix
    out = np.empty_like(matrix, dtype=np.float64)
    for i in range(matrix.shape[0]):
        lo = max(0, i - w + 1)
        out[i] = matrix[lo:i + 1].mean(axis=0)
    return out


def _max_abs_z(matrix: np.ndarray, base_mu: np.ndarray,
               base_sd: np.ndarray) -> np.ndarray:
    """Per-tick max abs z-score across columns, against a baseline window."""
    if matrix.size == 0:
        return np.zeros(0)
    sd = np.where(base_sd < 1e-9, 1e-9, base_sd)
    return np.max(np.abs((matrix - base_mu) / sd), axis=1)


def _calibrate_tau(stat: np.ndarray, start: int, target_fa: float) -> float:
    """Threshold giving `target_fa` crossings on a CONTROL series.

    Calibrated, never hand-set: the claim makes an unmatched pair of thresholds
    a degeneracy condition, so both detectors are calibrated the same way on
    the same arm and the realised rates are recorded.
    """
    tail = stat[start:]
    if tail.size == 0:
        return float("inf")
    return float(np.quantile(tail, 1.0 - target_fa))


def _fa_rate(stat: np.ndarray, tau: float, start: int) -> float:
    tail = stat[start:]
    if tail.size == 0:
        return 0.0
    return float(np.mean(tail > tau))


def _inject(agent, arm: str) -> None:
    """Apply the control-plane pathology for this tick (held, not one-shot)."""
    if arm == ARM_PRECISION:
        try:
            agent.e3.recalibrate_precision_to(COLLAPSE_TARGET_PRECISION, step=1.0)
        except Exception:
            pass
    elif arm == ARM_MODE_LOCK:
        sal = getattr(agent, "salience", None)
        if sal is not None:
            try:
                for mode in list(getattr(sal.config, "enter_thresholds", {}) or
                                 {"focused": 0, "diffuse": 0}):
                    sal.set_enter_threshold(mode, MODE_LOCK_THRESHOLD)
            except Exception:
                pass


def _config_slice(env, obs) -> Dict[str, Any]:
    return dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32, world_dim=32, alpha_world=0.9,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        harm_obs_a_dim=int(obs["harm_obs_a"].numel()),
        harm_history_len=10,
        use_commit_readiness=True,
        use_salience_coordinator=True,
        use_dacc=True,
    )


def run_cell(arm: str, seed: int, dry_run: bool = False) -> Dict[str, Any]:
    env0 = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    _f, obs0 = env0.reset()
    cfg_slice = _config_slice(env0, obs0)

    inject_at = 40 if dry_run else INJECT_AT_TICK
    baseline = 30 if dry_run else BASELINE_TICKS
    episodes = 2 if dry_run else EPISODES
    steps = 60 if dry_run else STEPS_PER_EPISODE

    with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__)) as cell:
        config = REEConfig.from_dims(**cfg_slice)
        agent = REEAgent(config)

        tel_rows: List[List[float]] = []
        harm_rows: List[float] = []
        tick = {"n": 0}

        for ep in range(episodes):
            env = CausalGridWorldV2(seed=seed * 1000 + ep, **ENV_KWARGS)
            env.reset()
            harness = StepHarness(agent, env, train_mode=True, seed=seed)

            def _on_step(result):
                tick["n"] += 1
                if tick["n"] >= inject_at:
                    _inject(agent, arm)
                vec = _telemetry_vector(agent)
                tel_rows.append([vec[k] for k in TELEMETRY_KEYS])
                hs = getattr(result, "harm_signal", None)
                harm_rows.append(float(hs) if hs is not None else 0.0)

            harness.run_episode(max_steps=steps, on_step=_on_step)
            print(f"  [train] {arm} seed={seed} ep {ep + 1}/{episodes} "
                  f"ticks={tick['n']}", flush=True)

        _ZG.observe(agent)

        tel = np.asarray(tel_rows, dtype=np.float64)
        beh = np.asarray(harm_rows, dtype=np.float64).reshape(-1, 1)
        n = len(tel)
        row: Dict[str, Any] = {
            "arm": arm, "seed": seed, "n_ticks": n,
            "inject_at": inject_at, "baseline_ticks": baseline,
        }
        if n <= baseline + CONSEC:
            row["insufficient_ticks"] = True
            cell.stamp(row)
            return row

        keep = [i for i, k in enumerate(TELEMETRY_KEYS)
                if k not in INJECTED_CHANNELS.get(arm, [])]
        base = slice(0, baseline)
        tel_excl = tel[:, keep]

        tel_excl_s = _smooth(tel_excl)
        tel_s = _smooth(tel)
        beh_s = _smooth(beh)
        row["stat_tel_excl"] = _max_abs_z(
            tel_excl_s, tel_excl_s[base].mean(0), tel_excl_s[base].std(0)).tolist()
        row["stat_tel_full"] = _max_abs_z(
            tel_s, tel_s[base].mean(0), tel_s[base].std(0)).tolist()
        row["stat_beh"] = _max_abs_z(
            beh_s, beh_s[base].mean(0), beh_s[base].std(0)).tolist()
        row["telemetry_channel_variance"] = float(np.mean(np.var(tel, axis=0)))
        row["excluded_channels"] = INJECTED_CHANNELS.get(arm, [])
        cell.stamp(row)
    return row


def analyse(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [r for r in rows if not r.get("insufficient_ticks")]
    controls = [r for r in usable if r["arm"] == ARM_CONTROL]

    # --- calibrate BOTH detectors on the CONTROL arm, same procedure --------
    taus: Dict[int, Dict[str, float]] = {}
    fa: Dict[int, Dict[str, float]] = {}
    for c in controls:
        start = c["baseline_ticks"]
        t_excl = np.asarray(c["stat_tel_excl"])
        t_beh = np.asarray(c["stat_beh"])
        t_full = np.asarray(c["stat_tel_full"])
        tau_t = _calibrate_tau(t_excl, start, TARGET_FA)
        tau_b = _calibrate_tau(t_beh, start, TARGET_FA)
        tau_f = _calibrate_tau(t_full, start, TARGET_FA)
        taus[c["seed"]] = {"tel": tau_t, "beh": tau_b, "full": tau_f}
        fa[c["seed"]] = {"tel": _fa_rate(t_excl, tau_t, start),
                         "beh": _fa_rate(t_beh, tau_b, start)}

    leads: List[Dict[str, Any]] = []
    for r in usable:
        if r["arm"] == ARM_CONTROL or r["seed"] not in taus:
            continue
        tau = taus[r["seed"]]
        start = r["inject_at"]
        n = r["n_ticks"]
        t_excl = np.asarray(r["stat_tel_excl"])
        t_full = np.asarray(r["stat_tel_full"])
        t_beh = np.asarray(r["stat_beh"])
        d_tel = _first_crossing(t_excl, tau["tel"], start)
        d_full = _first_crossing(t_full, tau["full"], start)
        d_beh = _first_crossing(t_beh, tau["beh"], start)
        # A detector that never fires is CENSORED at the window end, not
        # dropped: dropping it would silently select for the cases that favour
        # whichever stream fired.
        tel_t = d_tel if d_tel is not None else n
        beh_t = d_beh if d_beh is not None else n
        full_t = d_full if d_full is not None else n
        leads.append({
            "arm": r["arm"], "seed": r["seed"],
            "t_telemetry": tel_t, "t_behaviour": beh_t,
            "telemetry_fired": d_tel is not None,
            "behaviour_fired": d_beh is not None,
            "lead_time": beh_t - tel_t,
            "lead_time_full_surface": beh_t - full_t,
        })

    # --- preconditions ----------------------------------------------------
    variances = [r.get("telemetry_channel_variance") for r in usable
                 if r.get("telemetry_channel_variance") is not None]
    worst_var = min(variances) if variances else 0.0
    fa_gaps = [abs(v["tel"] - v["beh"]) for v in fa.values()]
    worst_gap = max(fa_gaps) if fa_gaps else 1.0
    manifested = [l for l in leads if l["behaviour_fired"]]
    manifest_frac = (len(manifested) / len(leads)) if leads else 0.0

    preconditions = [
        {"name": "telemetry_channels_non_constant", "kind": "readiness",
         "description": "the telemetry surface varies over the probe window",
         "control": "worst cell mean per-channel variance",
         "measured": worst_var, "threshold": 1e-9, "direction": "lower",
         "met": worst_var > 1e-9},
        {"name": "pathology_manifests_behaviourally", "kind": "readiness",
         "description": ("the behaviour detector eventually fires; without it "
                         "lead time is undefined (the claim says so)"),
         "control": "fraction of pathology cells whose behaviour detector fired",
         "measured": manifest_frac, "threshold": 0.5, "direction": "lower",
         "met": manifest_frac >= 0.5},
        {"name": "false_alarm_rates_matched", "kind": "readiness",
         "description": ("both detectors calibrated to the same FA rate on "
                         "CONTROL; the claim makes an unmatched pair degenerate"),
         "control": "worst |FA_telemetry - FA_behaviour| across control seeds",
         "measured": worst_gap, "threshold": FA_GAP_TOLERANCE,
         "direction": "upper", "met": worst_gap <= FA_GAP_TOLERANCE},
    ]
    gate_green = all(p["met"] for p in preconditions)

    # --- C1 ---------------------------------------------------------------
    by_arm: Dict[str, List[int]] = {}
    for l in leads:
        by_arm.setdefault(l["arm"], []).append(l["lead_time"])
    positive_arms = [a for a, v in by_arm.items()
                     if len(v) >= MIN_LEAD_SEEDS and all(x > 0 for x in v)]
    c1_pass = bool(len(positive_arms) >= MIN_PATHOLOGIES)
    all_leads = [l["lead_time"] for l in leads]
    median_lead = float(np.median(all_leads)) if all_leads else None

    if not gate_green:
        label = "substrate_not_ready_requeue"
    elif c1_pass:
        label = "telemetry_leads_behaviour_developmental_safety_value_supported"
    elif all_leads and all(x <= 0 for x in all_leads):
        label = "zero_lead_time_narrow_to_diagnostic_convenience"
    else:
        label = "lead_time_inconsistent_across_seeds_or_pathologies"

    return {
        "leads": leads, "by_arm_lead_times": by_arm,
        "positive_arms": positive_arms, "median_lead": median_lead,
        "control_taus": {str(k): v for k, v in taus.items()},
        "control_fa": {str(k): v for k, v in fa.items()},
        "worst_fa_gap": worst_gap,
        "preconditions": preconditions, "gate_green": gate_green,
        "c1_pass": c1_pass, "label": label,
    }


def run(dry_run: bool = False) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    seeds = SEEDS[:1] if dry_run else SEEDS
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm}", flush=True)
            row = run_cell(arm, seed, dry_run=dry_run)
            rows.append(row)
            print(f"verdict: {'PASS' if not row.get('insufficient_ticks') else 'FAIL'}",
                  flush=True)

    s = analyse(rows)
    outcome = "PASS" if (s["gate_green"] and s["c1_pass"]) else "FAIL"

    criteria = [
        {"name": "C1_positive_lead_time_sign_consistent", "load_bearing": True,
         "passed": bool(s["c1_pass"]),
         "measured": len(s["positive_arms"]), "threshold": MIN_PATHOLOGIES,
         "direction": "lower"},
        {"name": "C2_median_lead_time", "load_bearing": False,
         "passed": bool(s["median_lead"] is not None and s["median_lead"] > 0),
         "measured": s["median_lead"], "threshold": 0.0, "direction": "lower"},
        {"name": "C3_false_alarm_gap", "load_bearing": False,
         "passed": bool(s["worst_fa_gap"] <= FA_GAP_TOLERANCE),
         "measured": s["worst_fa_gap"], "threshold": FA_GAP_TOLERANCE,
         "direction": "upper"},
    ]

    direction = "unknown"
    if s["gate_green"]:
        direction = "supports" if s["c1_pass"] else "weakens"

    readout: Dict[str, float] = {}
    for key, value in (
        ("n_positive_arms", len(s["positive_arms"])),
        ("median_lead_time", s["median_lead"]),
        ("worst_fa_gap", s["worst_fa_gap"]),
        ("c1_pass", 1 if s["c1_pass"] else 0),
        ("gate_green", 1 if s["gate_green"] else 0),
    ):
        coerced = _flat_scalar(value)
        if coerced is not None:
            readout[key] = coerced

    # per-seed detail kept, never collapsed to mean+/-std
    per_seed = [{k: v for k, v in r.items()
                 if k not in ("stat_tel_excl", "stat_tel_full", "stat_beh")}
                for r in rows]

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
        "sub_claim_1_landed_as": (
            "tests/contracts/test_mech042_telemetry_readonly_nonparticipation.py "
            "(per the claim's own 2026-09-16 disposition)"),
        "arm_results": rows,
        "per_seed_summary": per_seed,
        "criteria": criteria,
        "criteria_non_degenerate": {
            "C1_positive_lead_time_sign_consistent": bool(s["gate_green"]),
            "C2_median_lead_time": bool(s["leads"]),
            "C3_false_alarm_gap": bool(s["control_fa"]),
        },
        "non_degenerate": bool(s["gate_green"] and s["leads"]),
        "degeneracy_reason": (
            None if (s["gate_green"] and s["leads"])
            else "precondition gate not green, or no pathology cell produced a lead time"),
        "interpretation": {
            "label": s["label"],
            "preconditions": s["preconditions"],
            "criteria_non_degenerate": {
                "C1_positive_lead_time_sign_consistent": bool(s["gate_green"]),
            },
        },
        "combination_rule": (
            "outcome PASS requires the precondition gate green AND C1: strictly "
            "positive lead time on every seed of at least MIN_PATHOLOGIES (2) "
            "pathology arms, each with at least MIN_LEAD_SEEDS (3) seeds -- the "
            "claim's 'sign-consistent across >=3 seeds and >=2 pathology types'. "
            "C2 and C3 are recorded and do not gate: C2 is a pooled magnitude, "
            "C3 is the matched-false-alarm degeneracy check already enforced as "
            "a precondition."),
        "readout": readout,
        "summary": {k: v for k, v in s.items() if k != "leads"},
        "lead_times": s["leads"],
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
        manifest, None, dry_run=args.dry_run,
        config={"env": ENV_KWARGS, "episodes": EPISODES,
                "steps_per_episode": STEPS_PER_EPISODE, "arms": ARMS,
                "inject_at_tick": INJECT_AT_TICK,
                "baseline_ticks": BASELINE_TICKS, "consec": CONSEC,
                "target_fa": TARGET_FA,
                "injected_channels": INJECTED_CHANNELS},
        seeds=SEEDS, script_path=Path(__file__), started_at=_t_start,
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
