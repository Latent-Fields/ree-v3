"""V3-EXQ-777: MECH-063 orthogonal control-plane axes -- behavioural rank-2 falsifier.

Claim:    MECH-063 (control plane retains orthogonal axes rather than collapsing
          into one scalar).
Proposal: EVB-0204 (IGW-20260717-203).
Purpose:  evidence (tests the MECH-063 hypothesis directly).

WHY NOW (governance): MECH-063 is provisional but lit_only_above_cap /
low_exp_conf / missing_experimental_evidence / synthetic_signals_only. The one
prior experiment that touched it (V3-EXQ-164a, MECH-142 secondary) built a
SYNTHETIC control_vec out of raw harm_obs (EMA + abs-deviation). This run instead
drives the REAL PRODUCTION control plane -- two independently-toggleable
regulators that route to two DIFFERENT downstream targets at the E3 select()
call site (agent.py select_action):

  SCORE AXIS  -- MECH-320 tonic_vigor: an additive per-candidate directional
                 bias (dacc_score_bias) that favours action over no-op. Shifts
                 the *location* (mode) of the E3 softmax over candidates.
  TEMPERATURE -- MECH-313 noise_floor: lifts the E3 softmax *temperature*
     AXIS        (effective_temperature). Scales the *spread* of the softmax.

These are two mathematically distinct parameters of the same policy softmax
(probs = softmax(-(scores + score_bias) / T)). MECH-063 asserts REE keeps such
axes ORTHOGONAL rather than collapsing regulation into one arousal/precision
scalar. If the control plane were one scalar, the two regulators would move the
policy along a SINGLE direction (their behavioural effect vectors would be
parallel / rank-1). If the axes are genuinely orthogonal, the two regulators
move the policy in LINEARLY INDEPENDENT directions (rank-2).

WHY A RANK-2 TEST, NOT A SCALAR DOUBLE-DISSOCIATION: with a (bias, temperature)
softmax, NO single scalar summary is cleanly owned by one axis -- both knobs
perturb any marginal statistic (temperature pulls every summary toward the
uniform policy; a directional bias shifts location AND concentrates). A scalar
"axis A moves readout X, axis B moves readout Y" claim is therefore not
identifiable. The identifiable, faithful statement of "not one scalar" is that
the two axes move a 2-D policy-readout vector in linearly independent
directions. So we measure a 2-vector readout per arm and test the rank of the
2x2 main-effect matrix.

DESIGN: 2x2 factorial telemetry probe, NO gradient training (the regulators act
at selection time; tonic_vigor is forced live with tonic_vigor_v_t_floor so the
score axis does not depend on the EWMA charging -- the V3-EXQ-563 / 650 pattern).
  Factor A (SCORE axis)      : use_tonic_vigor in {OFF, ON}
  Factor B (TEMPERATURE axis): use_noise_floor  in {OFF, ON}
  4 arms x SEEDS seeds. Shared substrate settings across ALL arms:
    use_control_vector_logging=True (read-only telemetry; bit-identical
      behaviour -- V3-EXQ-650 C3), hippocampal.use_action_class_scaffold_candidates
      =True (prepends one one-hot candidate per action class so a no-op-class
      candidate is always present for the score axis to act on), same env.

READOUTS (per fresh E3 selection; candidates captured via a read-only wrapper on
agent.generate_trajectories, aligned with agent.e3.last_precommit_probs):
  E = normalised Shannon entropy of last_precommit_probs (spread; nats / ln K).
  D = precommit probability MASS on non-no-op candidates (action preference;
      first-action class != NOOP_CLASS). Continuous, so it carries signal even
      when the argmax never selects the no-op action.

AGGREGATION (per seed): cell means (E, D) for the 4 arms; standardise E and D by
their pooled across-cell SD so both readouts contribute comparably; form the two
main-effect vectors
  v_temp  = mean_B1(E,D) - mean_B0(E,D)      (noise_floor ON - OFF)
  v_score = mean_A1(E,D) - mean_A0(E,D)      (tonic_vigor ON - OFF)
and the normalised non-collinearity
  s = |det[v_temp, v_score]| / (||v_temp|| * ||v_score||)  = |sin(angle)| in [0,1].
s ~ 1 => orthogonal axes (rank-2, supports MECH-063).
s ~ 0 => parallel effect vectors (one scalar, weakens MECH-063).

P0 READINESS (self-routes substrate_not_ready_requeue -- NEVER a claim verdict):
  R1 score axis live   : A1 arms mean v_t > 0 and mean n_noop_candidates >= 1.
  R2 temperature live  : B1 arms mean noise_floor_temp_lift >= TEMP_LIFT_FLOOR.
  R3 sample sufficiency: every cell has >= MIN_SELECTS fresh E3 selections.
  R4 readout headroom  : baseline arm E < E_SAT_CEIL and D has non-zero spread.
  R5 axis authority    : ||v_temp|| and ||v_score|| both > EFFECT_FLOOR (a silent
                         regulator means the substrate did not exercise the axis,
                         not that control is one scalar).
Below any precondition -> outcome FAIL, evidence_direction non_contributory,
non_degenerate False, label substrate_not_ready_requeue.

VERDICT (pre-registered constants; not derived post-hoc):
  readiness unmet                    -> FAIL / non_contributory (requeue).
  readiness met AND C1 (s > SIN_MARGIN on >= MIN_SEEDS seeds, and robust:
     mean_s - SD_s > 0) -> PASS / supports  (rank-2: control is NOT one scalar).
  readiness met AND C1 unmet         -> FAIL / weakens (effect vectors parallel:
     evidence the control plane collapses toward one scalar).

MECH-094: trains nothing, writes no memory, no replay -- N/A (waking select only).
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_THIS = Path(__file__).resolve()
_REE_V3 = _THIS.parent.parent
if str(_REE_V3) not in sys.path:
    sys.path.insert(0, str(_REE_V3))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_777_mech063_orthogonal_control_axes_dissociation"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["MECH-063"]

# ---- Pre-registered constants (fixed before the run; not derived post-hoc) ----
SEEDS = [11, 17, 23, 29, 37]
MIN_SEEDS = 4                 # criteria must hold on >= this many of the 5 seeds
N_EPISODES = 3
STEPS_PER_EPISODE = 300
EPISODES_PER_RUN = N_EPISODES  # denominator for [train] progress prints

ENV_SIZE = 8
ENV_HAZARDS = 2
ENV_RESOURCES = 3
NOOP_CLASS = 4                # CausalGridWorldV2 ACTIONS[4] = (0,0) = stay/no-op
ACTION_DIM = 5                # CausalGridWorldV2 ACTIONS = {0..3 moves, 4 stay}

# Score axis (MECH-320 tonic_vigor) -- forced live via v_t_floor.
TV_V_T_FLOOR = 0.5
TV_W_ACTION = 0.5
TV_W_PASSIVE = 0.5
TV_BIAS_SCALE = 1.0
# Temperature axis (MECH-313 noise_floor). Baseline E3 temperature is 1.0
# (StepHarness passes temperature=1.0); effective = max(1.0 + alpha, min_T) = 2.0.
NF_ALPHA = 1.0
NF_MIN_T = 2.0

# Readiness thresholds.
MIN_SELECTS = 20             # R3: fresh E3 selections per cell
TEMP_LIFT_FLOOR = 0.5        # R2: effective_T - baseline_T in noise_floor arms
E_SAT_CEIL = 0.98            # R4: baseline normalised entropy must leave headroom
EFFECT_FLOOR = 0.05          # R5: min standardised main-effect vector norm

# Verdict threshold.
SIN_MARGIN = 0.5             # C1: |sin(angle)| between effect vectors (~30 deg)

ARMS = ["A0B0", "A1B0", "A0B1", "A1B1"]  # (tonic_vigor, noise_floor)
_ARM_FLAGS = {
    "A0B0": (False, False),
    "A1B0": (True, False),
    "A0B1": (False, True),
    "A1B1": (True, True),
}


def _mk_env() -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=ENV_SIZE, num_hazards=ENV_HAZARDS, num_resources=ENV_RESOURCES
    )


def _build_config(arm: str, env: CausalGridWorldV2) -> REEConfig:
    use_tv, use_nf = _ARM_FLAGS[arm]
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )
    # Shared substrate operating settings (identical across all four arms).
    cfg.use_control_vector_logging = True
    cfg.hippocampal.use_action_class_scaffold_candidates = True
    # Score axis.
    cfg.use_tonic_vigor = use_tv
    if use_tv:
        cfg.tonic_vigor_v_t_floor = TV_V_T_FLOOR
        cfg.tonic_vigor_w_action = TV_W_ACTION
        cfg.tonic_vigor_w_passive = TV_W_PASSIVE
        cfg.tonic_vigor_bias_scale = TV_BIAS_SCALE
        cfg.tonic_vigor_noop_class = NOOP_CLASS
        cfg.tonic_vigor_form = "additive"
    # Temperature axis.
    cfg.use_noise_floor = use_nf
    if use_nf:
        cfg.noise_floor_alpha = NF_ALPHA
        cfg.noise_floor_min_temperature = NF_MIN_T
    return cfg


def _config_slice(arm: str) -> Dict[str, Any]:
    """Fingerprint config slice: env + shared operating settings + this arm's
    control flags. Declared (only what the cell's build+collect path reads).
    No env instance needed -- action_dim is the fixed CausalGridWorldV2 value."""
    use_tv, use_nf = _ARM_FLAGS[arm]
    sl: Dict[str, Any] = {
        "env_size": ENV_SIZE,
        "env_hazards": ENV_HAZARDS,
        "env_resources": ENV_RESOURCES,
        "action_dim": ACTION_DIM,
        "n_episodes": N_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "use_control_vector_logging": True,
        "use_action_class_scaffold_candidates": True,
        "use_tonic_vigor": use_tv,
        "use_noise_floor": use_nf,
    }
    if use_tv:
        sl.update(
            tonic_vigor_v_t_floor=TV_V_T_FLOOR,
            tonic_vigor_w_action=TV_W_ACTION,
            tonic_vigor_w_passive=TV_W_PASSIVE,
            tonic_vigor_bias_scale=TV_BIAS_SCALE,
            tonic_vigor_noop_class=NOOP_CLASS,
            tonic_vigor_form="additive",
        )
    if use_nf:
        sl.update(noise_floor_alpha=NF_ALPHA, noise_floor_min_temperature=NF_MIN_T)
    return sl


def _norm_entropy(probs: torch.Tensor) -> Optional[float]:
    """Normalised Shannon entropy (nats / ln K) of a candidate softmax."""
    p = probs.detach().reshape(-1).float()
    k = int(p.numel())
    if k < 2:
        return None
    p = p[p > 0]
    h = float(-(p * p.log()).sum().item())
    return h / math.log(k)


def _candidate_noop_mask(candidates: Any) -> Optional[torch.Tensor]:
    """Per-candidate boolean mask: True where the first-action class == NOOP.
    First-action class = argmax over action_dim of candidate.actions[:, 0, :]."""
    classes: List[int] = []
    for c in candidates:
        acts = getattr(c, "actions", None)
        if acts is None:
            return None
        a = acts.reshape(-1, acts.shape[-1])[0]  # first-step action vector
        classes.append(int(a.argmax().item()))
    if not classes:
        return None
    return torch.tensor(classes) == NOOP_CLASS


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    """One (arm, seed) cell: telemetry rollout, no gradient training."""
    with arm_cell(
        seed,
        config_slice=_config_slice(arm),
        script_path=_THIS,
        config_slice_declared=True,
        include_driver_script_in_hash=False,  # cross-driver reusable mint
    ) as cell:
        env = _mk_env()
        cfg = _build_config(arm, env)
        agent = REEAgent(cfg)

        # Read-only wrapper to capture the candidate list E3 selects over, so the
        # per-candidate no-op mask aligns with agent.e3.last_precommit_probs.
        captured: Dict[str, Any] = {"cands": None}
        _orig_gen = agent.generate_trajectories

        def _gen_capture(*a: Any, **k: Any) -> Any:
            cands = _orig_gen(*a, **k)
            captured["cands"] = cands
            return cands

        agent.generate_trajectories = _gen_capture  # type: ignore[assignment]

        harness = StepHarness(agent, env, train_mode=True, seed=seed)

        e_vals: List[float] = []
        d_vals: List[float] = []
        vt_vals: List[float] = []
        templift_vals: List[float] = []
        nnoop_vals: List[int] = []
        n_selects = 0
        n_env_steps = 0
        prev_probs_id: Optional[int] = None

        for ep in range(N_EPISODES):
            _flat, obs_dict = env.reset()
            agent.reset()
            harness.reset()
            prev_probs_id = None
            for _step in range(STEPS_PER_EPISODE):
                r = harness.step(obs_dict)
                n_env_steps += 1
                probs = getattr(agent.e3, "last_precommit_probs", None)
                pid = id(probs) if probs is not None else None
                fresh = probs is not None and pid != prev_probs_id
                prev_probs_id = pid
                cands = captured.get("cands")
                if fresh and cands is not None and len(cands) == int(probs.numel()):
                    ent = _norm_entropy(probs)
                    mask = _candidate_noop_mask(cands)
                    if ent is not None and mask is not None:
                        n_selects += 1
                        e_vals.append(ent)
                        pp = probs.detach().reshape(-1).float()
                        d_vals.append(float(pp[~mask].sum().item()))  # action mass
                        nnoop_vals.append(int(mask.sum().item()))
                        cv = agent._last_control_vector or {}
                        sh = cv.get("shared", {})
                        gv = cv.get("G_vigor", {})
                        vt_vals.append(float(sh.get("tonic_vigor_v_t", 0.0)))
                        templift_vals.append(
                            float(gv.get("noise_floor_temp_lift", 0.0))
                        )
                obs_dict = r.next_obs_dict
                if r.done:
                    break
            print(
                f"  [train] {arm} seed={seed} ep {ep + 1}/{EPISODES_PER_RUN} "
                f"env_steps={n_env_steps} e3_selects={n_selects}",
                flush=True,
            )

        def _mean(xs: List[float]) -> float:
            return float(statistics.fmean(xs)) if xs else 0.0

        row: Dict[str, Any] = {
            "arm": arm,
            "seed": seed,
            "use_tonic_vigor": _ARM_FLAGS[arm][0],
            "use_noise_floor": _ARM_FLAGS[arm][1],
            "n_env_steps": n_env_steps,
            "n_e3_selects": n_selects,
            "E_norm_entropy_mean": _mean(e_vals),
            "D_action_mass_mean": _mean(d_vals),
            "D_action_mass_std": float(statistics.pstdev(d_vals)) if len(d_vals) > 1 else 0.0,
            "vt_mean": _mean(vt_vals),
            "noise_floor_temp_lift_mean": _mean(templift_vals),
            "n_noop_candidates_mean": _mean([float(x) for x in nnoop_vals]),
            # Per-selection series retained for generous recording / reanalysis.
            "E_series": e_vals,
            "D_series": d_vals,
        }
        cell.stamp(row)
    return row


# ----------------------------------------------------------------------
# Aggregation: 2x2 rank-2 non-collinearity of the control-axis effect vectors.
# ----------------------------------------------------------------------
def _pooled_std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return float(statistics.pstdev(vals))


def _seed_analysis(rows_by_arm: Dict[str, Dict[str, Any]], e_sd: float, d_sd: float) -> Dict[str, Any]:
    """Standardise (E, D), form the two main-effect vectors, compute |sin angle|."""
    def _std_ed(arm: str) -> tuple:
        r = rows_by_arm[arm]
        e = r["E_norm_entropy_mean"] / e_sd if e_sd > 0 else 0.0
        d = r["D_action_mass_mean"] / d_sd if d_sd > 0 else 0.0
        return (e, d)

    e00, d00 = _std_ed("A0B0")
    e10, d10 = _std_ed("A1B0")
    e01, d01 = _std_ed("A0B1")
    e11, d11 = _std_ed("A1B1")

    # Temperature main effect (noise_floor ON - OFF), averaged over factor A.
    vt_e = ((e01 + e11) / 2.0) - ((e00 + e10) / 2.0)
    vt_d = ((d01 + d11) / 2.0) - ((d00 + d10) / 2.0)
    # Score main effect (tonic_vigor ON - OFF), averaged over factor B.
    vs_e = ((e10 + e11) / 2.0) - ((e00 + e01) / 2.0)
    vs_d = ((d10 + d11) / 2.0) - ((d00 + d01) / 2.0)

    norm_t = math.hypot(vt_e, vt_d)
    norm_s = math.hypot(vs_e, vs_d)
    det = vt_e * vs_d - vt_d * vs_e
    sin_angle = abs(det) / (norm_t * norm_s) if (norm_t > 0 and norm_s > 0) else 0.0

    return {
        "v_temp": [vt_e, vt_d],
        "v_score": [vs_e, vs_d],
        "norm_v_temp": norm_t,
        "norm_v_score": norm_s,
        "sin_angle": sin_angle,
        # Raw (unstandardised) marginal deltas for interpretability.
        "dE_temp": ((rows_by_arm["A0B1"]["E_norm_entropy_mean"] + rows_by_arm["A1B1"]["E_norm_entropy_mean"]) / 2.0)
        - ((rows_by_arm["A0B0"]["E_norm_entropy_mean"] + rows_by_arm["A1B0"]["E_norm_entropy_mean"]) / 2.0),
        "dD_score": ((rows_by_arm["A1B0"]["D_action_mass_mean"] + rows_by_arm["A1B1"]["D_action_mass_mean"]) / 2.0)
        - ((rows_by_arm["A0B0"]["D_action_mass_mean"] + rows_by_arm["A0B1"]["D_action_mass_mean"]) / 2.0),
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    global N_EPISODES, STEPS_PER_EPISODE, EPISODES_PER_RUN
    seeds = SEEDS[:2] if dry_run else SEEDS
    if dry_run:
        N_EPISODES = 1
        STEPS_PER_EPISODE = 40
        EPISODES_PER_RUN = N_EPISODES

    t0 = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            row = _run_cell(arm, seed)
            rows.append(row)
            print(f"verdict: {'PASS' if row['n_e3_selects'] > 0 else 'FAIL'}", flush=True)

    # Pooled SDs for standardising E and D across all cells (comparable scaling).
    e_all = [r["E_norm_entropy_mean"] for r in rows]
    d_all = [r["D_action_mass_mean"] for r in rows]
    e_sd = _pooled_std(e_all)
    d_sd = _pooled_std(d_all)

    # Per-seed rank-2 analysis.
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        by_arm = {r["arm"]: r for r in rows if r["seed"] == seed}
        if len(by_arm) != len(ARMS):
            continue
        a = _seed_analysis(by_arm, e_sd, d_sd)
        a["seed"] = seed
        per_seed.append(a)

    # ---- Readiness preconditions (self-route requeue if unmet) ----
    a1_rows = [r for r in rows if r["use_tonic_vigor"]]
    b1_rows = [r for r in rows if r["use_noise_floor"]]
    baseline_rows = [r for r in rows if r["arm"] == "A0B0"]

    r1_score_live = bool(a1_rows) and all(
        r["vt_mean"] > 0.0 and r["n_noop_candidates_mean"] >= 1.0 for r in a1_rows
    )
    r2_temp_live = bool(b1_rows) and all(
        r["noise_floor_temp_lift_mean"] >= TEMP_LIFT_FLOOR for r in b1_rows
    )
    r3_samples = bool(rows) and all(r["n_e3_selects"] >= MIN_SELECTS for r in rows)
    r4_headroom = bool(baseline_rows) and all(
        r["E_norm_entropy_mean"] < E_SAT_CEIL and r["D_action_mass_std"] > 0.0
        for r in baseline_rows
    )
    r5_authority = bool(per_seed) and (
        statistics.fmean([a["norm_v_temp"] for a in per_seed]) > EFFECT_FLOOR
        and statistics.fmean([a["norm_v_score"] for a in per_seed]) > EFFECT_FLOOR
    )
    readiness_met = bool(
        r1_score_live and r2_temp_live and r3_samples and r4_headroom and r5_authority
    )

    # ---- Load-bearing criterion C1: rank-2 non-collinearity ----
    sins = [a["sin_angle"] for a in per_seed]
    seeds_c1 = sum(1 for s in sins if s > SIN_MARGIN)
    mean_sin = statistics.fmean(sins) if sins else 0.0
    sd_sin = _pooled_std(sins)
    c1_seed_count = seeds_c1 >= min(MIN_SEEDS, len(seeds))
    c1_robust = (mean_sin - sd_sin) > SIN_MARGIN  # effect exceeds its own noise
    c1 = bool(c1_seed_count and c1_robust)

    non_degenerate = bool(readiness_met and _pooled_std(sins) >= 0.0 and len(per_seed) >= 1)
    degeneracy_reason = None

    if not readiness_met:
        outcome = "FAIL"
        direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = "readiness precondition unmet (see interpretation.preconditions)"
    elif c1:
        outcome = "PASS"
        direction = "supports"
        label = "control_axes_orthogonal_rank2"
    else:
        outcome = "FAIL"
        direction = "weakens"
        label = "control_axes_collinear_toward_one_scalar"

    interpretation = {
        "label": label,
        "preconditions": [
            {"name": "score_axis_live", "control": "A1 arms: mean v_t and no-op candidate count",
             "measured": (statistics.fmean([r["vt_mean"] for r in a1_rows]) if a1_rows else 0.0),
             "threshold": 0.0, "met": bool(r1_score_live)},
            {"name": "temperature_axis_live", "control": "B1 arms: effective_T - baseline_T",
             "measured": (statistics.fmean([r["noise_floor_temp_lift_mean"] for r in b1_rows]) if b1_rows else 0.0),
             "threshold": TEMP_LIFT_FLOOR, "met": bool(r2_temp_live)},
            {"name": "sample_sufficiency", "control": "min fresh E3 selections over cells",
             "measured": (min(r["n_e3_selects"] for r in rows) if rows else 0),
             "threshold": MIN_SELECTS, "met": bool(r3_samples)},
            {"name": "baseline_entropy_headroom", "control": "A0B0 normalised entropy below ceiling",
             "measured": (statistics.fmean([r["E_norm_entropy_mean"] for r in baseline_rows]) if baseline_rows else 1.0),
             "threshold": E_SAT_CEIL, "direction": "upper", "met": bool(r4_headroom)},
            {"name": "axis_authority", "control": "both standardised main-effect vector norms",
             "measured": (min(statistics.fmean([a["norm_v_temp"] for a in per_seed]),
                              statistics.fmean([a["norm_v_score"] for a in per_seed])) if per_seed else 0.0),
             "threshold": EFFECT_FLOOR, "met": bool(r5_authority)},
        ],
        "criteria": [
            {"name": "C1_rank2_non_collinearity", "load_bearing": True, "passed": bool(c1)},
        ],
        "criteria_non_degenerate": {
            "C1_rank2_non_collinearity": bool(len(per_seed) >= 1 and _pooled_std(sins) >= 0.0),
        },
        "summary": (
            "MECH-063 rank-2 test on the production control plane. Two independently-"
            "toggled regulators route to two E3-softmax parameters (score-bias vs "
            "temperature). PASS = their standardised behavioural effect vectors are "
            "non-collinear (rank-2, control is NOT one scalar); FAIL/weakens = parallel "
            "(collapse toward one scalar); FAIL/non_contributory = a control axis was "
            "not exercised (substrate_not_ready_requeue)."
        ),
    }

    elapsed = time.perf_counter() - t0
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "evidence_direction": direction,
        "backlog_id": "EVB-0204",
        "proposal_id": "EVB-0204",
        "dry_run": bool(dry_run),
        "non_degenerate": non_degenerate,
        "interpretation": interpretation,
        "acceptance": {
            "readiness_met": readiness_met,
            "C1_rank2_non_collinearity": c1,
            "c1_seed_count": seeds_c1,
            "min_seeds": min(MIN_SEEDS, len(seeds)),
            "mean_sin_angle": mean_sin,
            "sd_sin_angle": sd_sin,
            "SIN_MARGIN": SIN_MARGIN,
        },
        "per_seed": per_seed,
        "arm_results": rows,
        "readout_standardisation": {"E_pooled_sd": e_sd, "D_pooled_sd": d_sd},
        "notes": (
            "MECH-063 orthogonal control-axes rank-2 falsifier. REAL production "
            "regulators (MECH-320 tonic_vigor score axis vs MECH-313 noise_floor "
            "temperature axis) at the E3 select() call site -- addresses the "
            "synthetic_signals_only weakness of the prior V3-EXQ-164a synthetic "
            "control_vec. GOV-REUSE-1: the decisive readout (behavioural rank of the "
            "2-axis effect matrix) is absent from all recorded manifests -- "
            "V3-EXQ-650 measured only the WITHIN-MECH-320 C_time/G_vigor collapse "
            "(claim_ids=[], weights nothing), a different question -> run. "
            "Re-derive brake: 0 substrate_ceiling/non_contributory autopsies on "
            "MECH-063 -> not braked."
        ),
    }
    if non_degenerate is False and degeneracy_reason:
        manifest["degeneracy_reason"] = degeneracy_reason

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t_start = time.perf_counter()
    manifest = run_experiment(dry_run=args.dry_run)

    out_dir = _REE_V3.parent / "REE_assembly" / "evidence" / "experiments"
    full_config = {
        "seeds": SEEDS,
        "n_episodes": N_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "env": {"size": ENV_SIZE, "num_hazards": ENV_HAZARDS, "num_resources": ENV_RESOURCES},
        "noop_class": NOOP_CLASS,
        "tonic_vigor": {"v_t_floor": TV_V_T_FLOOR, "w_action": TV_W_ACTION,
                        "w_passive": TV_W_PASSIVE, "bias_scale": TV_BIAS_SCALE, "form": "additive"},
        "noise_floor": {"alpha": NF_ALPHA, "min_temperature": NF_MIN_T},
        "thresholds": {"MIN_SEEDS": MIN_SEEDS, "MIN_SELECTS": MIN_SELECTS,
                       "TEMP_LIFT_FLOOR": TEMP_LIFT_FLOOR, "E_SAT_CEIL": E_SAT_CEIL,
                       "EFFECT_FLOOR": EFFECT_FLOOR, "SIN_MARGIN": SIN_MARGIN},
    }

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=_THIS,
        started_at=t_start,
    )

    # NOTE: use a non-"verdict:" prefix here -- the runner counts "verdict:" lines
    # as completed runs, and the per-cell prints already emit exactly
    # seeds x conditions of them.
    print(f"FINAL_OUTCOME: {manifest['outcome']}", flush=True)
    print(f"  label: {manifest['interpretation']['label']}", flush=True)
    print(f"  readiness_met: {manifest['acceptance']['readiness_met']}", flush=True)
    print(f"  mean_sin_angle: {manifest['acceptance']['mean_sin_angle']:.3f} "
          f"(margin {SIN_MARGIN})", flush=True)
    print(f"Result written to: {out_path}", flush=True)

    emit_outcome(
        outcome=manifest["outcome"] if manifest["outcome"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
        dry_run=args.dry_run,
    )
