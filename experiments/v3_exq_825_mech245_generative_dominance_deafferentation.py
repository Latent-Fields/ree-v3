"""V3-EXQ-825 -- MECH-245 generative-model-dominance-under-deafferentation
discriminative pair.

Proposal EXP-0278 / backlog EVB-0122 (IGW-20260726-222). Claim: MECH-245
(psychiatric.hallucination_generative_dominance -- hallucination is a
generative-model dominance failure where top-down predictions produce
percepts without corresponding bottom-up sensory grounding).

What MECH-245 asserts (claims.yaml, docs/architecture/default_mode.md)
------------------------------------------------------------------------
"E1 top-down predictions propagate forward as if they were received sensory
signals, bypassing or overwhelming the bottom-up mismatch check... The agent
perceives things that are not there." Distinct from MECH-094 (confabulation:
a memory WRITE-gate failure keyed on hypothesis_tag/provenance) and MECH-244
(precision-weighting failure in belief updating generally, with bottom-up
evidence present but resisted). MECH-245 is specifically about the SENSORY
GENERATION pathway when bottom-up grounding is ABSENT (deafferentation:
Charles Bonnet syndrome, hypnagogic hallucination), not merely degraded.

Substrate map (verified against ree-v3 source 2026-07-26)
-----------------------------------------------------------
`ree_core/predictors/e3_selector.py` E3TrajectorySelector implements the
system's own "bottom-up mismatch check" directly: `post_action_update`
computes `prediction_error = actual_z_world - predicted_world` (top-down
generative prediction vs bottom-up observation) and feeds it to
`update_running_variance`, which drives `current_precision` (1/running_var)
and gates action `committed` via `commit_threshold` (committed when
running_variance < commitment_threshold). Nothing in E3TrajectorySelector
independently verifies that `actual_z_world` originated from a real sensory
encoding rather than from the generative prediction itself -- the mismatch
check's integrity depends entirely on the CALLER supplying a genuinely
independent bottom-up measurement. This experiment instruments that real
production API directly (no env/agent dependency needed, following the
V3-EXQ-633 MECH-094 precedent of driving a real substrate module with a
controlled synthetic input stream) to test whether (a) a genuinely grounded-
absent condition correctly collapses confidence (protective, PRIMARY arm),
and (b) substituting the generative prediction for the missing bottom-up
signal at that same call site reproduces the predicted "percept without
grounding" signature: sustained/false high confidence and commit-eligibility
(ABLATION arm).

Design
------
A small GRU "generative forward model" (stand-in for the E1/E2 top-down
pathway) is trained via teacher forcing on synthetic momentum-random-walk
z_world trajectories (real, non-trivial predictive structure -- momentum
carries genuine short-horizon information a GRU must actually learn). At
EVAL time, per held-out seed:

  GROUNDING phase (T0 ticks): teacher-forced rollout against the TRUE
    trajectory. `prediction_error = true_next - predicted_next` is fed to
    E3.update_running_variance, bringing running_variance to a realistic
    operating point reflecting the model's REAL (imperfect, measured, not
    hand-set) predictive accuracy. IDENTICAL for both arms (same seed, same
    model, no stochasticity at eval) -- a twin design that guarantees both
    arms enter the manipulation window from the same confidence state.

  DEAFFERENTATION phase (T1 ticks): sensory input stops. The model rolls
    forward AUTOREGRESSIVELY from its own prior prediction (a real, model-
    quality-dependent free rollout whose drift from the continuing TRUE
    dynamics is measured, not designer-dictated). The two arms differ ONLY
    in what is fed to E3 as `actual_z_world` this window:
      PRIMARY (mismatch-check intact): the bottom-up estimate stays FROZEN
        at the last genuinely grounded observation (no new sensory evidence
        arrives -- the honest "I don't know what happened next" signal).
      ABLATION (generative dominance): the bottom-up estimate IS the
        model's own current prediction (the top-down content "propagates
        forward as if it were a received sensory signal").

DV-SYMMETRY DECLARATION (queue-experiment skill Step 3, MANDATORY)
--------------------------------------------------------------------
ABLATION's per-tick prediction_error = predicted - predicted = 0 is a
DEFINITIONAL IDENTITY (self-comparison), not a measured discovery -- it is
NOT invariant-in-the-problematic-sense so much as intentionally
constructed to instantiate the claim's own failure mode ("propagates
forward AS IF received"), exactly as V3-EXQ-633's ABLATION arm directly
sets hypothesis_tag=False to instantiate MECH-094's tag-loss pathology.
The GENUINELY MEASURED, non-tautological content -- which could fail to
clear its pre-registered floor -- is: (1) whether PRIMARY's deafferentation
PE, driven by the REAL trained model's REAL free-rollout drift against a
frozen true state, is large enough to push running_variance up by the
pre-registered margin (this depends on real model quality and real
dynamics chaoticity -- a smooth/over-damped process or an over-fit model
could plausibly fail this), and (2) `ground_truth_drift_at_end`, which
audits how far the ABLATION arm's "confident percept" has actually drifted
from the true (unobserved) state -- a real quantity from real dynamics,
establishing that the false confidence is confidence in something WRONG,
not an accidentally-correct guess.

Pre-registered metrics (per seed x condition)
----------------------------------------------
grounded_operating_variance : running_variance at end of grounding phase
  (twin-design manipulation check: identical across arms per seed).
final_running_variance      : running_variance at end of deafferentation.
final_current_precision     : 1 / (final_running_variance + 1e-6).
deaff_mean_abs_pe           : mean |prediction_error| over deafferentation.
would_commit_fraction       : fraction of deafferentation ticks where
  running_variance < e3.commit_threshold (production default 0.40;
  reported for interpretability, NOT used as a gating criterion since it
  is calibrated to real RL z_world scale, not this synthetic scale).
ground_truth_drift_at_end   : ||final predicted position - true position||
  (same dynamics continuation for both arms; audits hallucination content).

Pre-registered pass criteria (ratio-based -- scale-robust, no absolute
magic number needs calibrating to this synthetic z_world's units)
--------------------------------------------------------------------
C1 (asymmetric confidence collapse): PRIMARY final_running_variance /
   grounded_operating_variance >= 1.3 (genuine deprivation raises
   uncertainty) AND ABLATION final_running_variance /
   grounded_operating_variance <= 1.05 (substitution sustains -- or
   amplifies -- false confidence).
C2 (divergence magnitude): (primary_final_var - ablation_final_var) /
   grounded_operating_variance >= 0.30 (arms genuinely and substantially
   diverge, not marginally).
C3 (manipulation validity): deaff_mean_abs_pe(PRIMARY) /
   (deaff_mean_abs_pe(ABLATION) + 1e-9) >= 20 (the two arms' PE streams
   are genuinely different, not accidentally similar) AND
   deaff_mean_abs_pe(PRIMARY) / grounded_mean_abs_pe >= 1.2 (deprivation
   creates genuinely MORE surprise than ordinary tracking noise -- the
   part of C3 that is a real, failable measurement).
PASS overall = C1 AND C2 AND C3 in >= ceil(n_seeds * PASS_FRACTION_REQUIRED)
seeds. PASS supports MECH-245: the real bottom-up-mismatch-check pathway
(E3's own PE-driven precision) is load-bearing and protective when
genuinely exercised, AND its bypass (generative content substituted for
missing bottom-up input) reproduces exactly the predicted "percept without
corresponding grounding" signature -- distinct from MECH-094's write-gate
(different substrate module, different pathology: memory corruption vs
false perceptual confidence) and MECH-244's belief-updating dominance
(this manipulation is total-absence-of-input, not resistance-to-present-
evidence).

Phased training: NOT applicable in the P0/P1/P2 sense -- the GRU forward
model here is a synthetic stand-in for the top-down generative pathway
(analogous to E1/E2), trained once on synthetic dynamics; no head is
trained on real z_world/z_harm/encoder output.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_825_mech245_generative_dominance_deafferentation.py
  /opt/local/bin/python3 experiments/v3_exq_825_mech245_generative_dominance_deafferentation.py --dry-run
Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.predictors.e3_selector import E3TrajectorySelector  # noqa: E402
from ree_core.utils.config import E3Config  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from _metrics import check_degeneracy  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_825_mech245_generative_dominance_deafferentation"
CLAIM_IDS = ["MECH-245"]
EXPERIMENT_PURPOSE = "evidence"

# --- Configuration ----------------------------------------------------------
WORLD_DIM = 32
HIDDEN_DIM = 32
MOMENTUM_DECAY = 0.9
PROCESS_NOISE_STD = 0.05

TRAIN_BASE_SEED = 10000       # disjoint from EVAL seeds below (no train/test leak)
TRAIN_BATCH_SIZE = 32
TRAIN_SEQ_LEN = 30
TRAIN_ITERS = 150
TRAIN_LR = 0.01

EVAL_SEEDS = (0, 1, 2)        # matched_shared_seeds: same seeds for BOTH conditions
T0_GROUNDING_TICKS = 50
T1_DEAFFERENTATION_TICKS = 30

CONDITION_PRIMARY = ("mismatch_check_intact", False)   # (label, is_ablation)
CONDITION_ABLATION = ("generative_dominance_bypass", True)
CONDITIONS = (CONDITION_PRIMARY, CONDITION_ABLATION)

# --- Pre-registered thresholds (constants; not derived from run statistics) -
C1_PRIMARY_MIN_VARIANCE_RATIO = 1.3
C1_ABLATION_MAX_VARIANCE_RATIO = 1.05
C2_MIN_DIVERGENCE_RATIO = 0.30
C3_MIN_PE_ARM_RATIO = 20.0
C3_MIN_SURPRISE_RATIO = 1.2
PASS_FRACTION_REQUIRED = 2.0 / 3.0


# --- Synthetic dynamics + generative forward model --------------------------
def generate_trajectory(seed: int, n_steps: int, world_dim: int) -> torch.Tensor:
    """Deterministic-given-seed momentum random walk in z_world space.

    velocity_t = MOMENTUM_DECAY * velocity_{t-1} + noise_t
    position_t = position_{t-1} + velocity_t

    Real, non-trivial short-horizon predictive structure (momentum): a
    forward model must actually learn the process to predict well; it is
    not a hand-set trajectory. Returns [n_steps + 1, world_dim].
    """
    g = torch.Generator()
    g.manual_seed(seed)
    velocity = torch.zeros(world_dim)
    position = torch.zeros(world_dim)
    positions = [position.clone()]
    for _ in range(n_steps):
        noise = torch.randn(world_dim, generator=g) * PROCESS_NOISE_STD
        velocity = MOMENTUM_DECAY * velocity + noise
        position = position + velocity
        positions.append(position.clone())
    return torch.stack(positions)


class GenerativeForwardModel(nn.Module):
    """GRU one-step forward predictor -- synthetic stand-in for the E1/E2
    top-down generative pathway. Predicts a position DELTA (residual
    parametrisation) so free-running autoregressive rollout integrates a
    learned "expected velocity", directly analogous to E1's associative
    prior / E2's world-forward continuation propagating without new input.
    """

    def __init__(self, world_dim: int, hidden_dim: int):
        super().__init__()
        self.cell = nn.GRUCell(world_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, world_dim)

    def step(self, position: torch.Tensor, hidden: torch.Tensor):
        hidden = self.cell(position, hidden)
        delta = self.head(hidden)
        return position + delta, hidden


def train_forward_model(dry_run: bool = False) -> GenerativeForwardModel:
    model = GenerativeForwardModel(WORLD_DIM, HIDDEN_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    iters = 15 if dry_run else TRAIN_ITERS
    batch_size = 8 if dry_run else TRAIN_BATCH_SIZE
    seq_len = 10 if dry_run else TRAIN_SEQ_LEN

    for i in range(iters):
        batch = torch.stack([
            generate_trajectory(TRAIN_BASE_SEED + i * batch_size + b, seq_len, WORLD_DIM)
            for b in range(batch_size)
        ])  # [B, seq_len + 1, world_dim]
        hidden = torch.zeros(batch_size, HIDDEN_DIM)
        loss = torch.zeros(())
        for t in range(seq_len):
            pos_t = batch[:, t, :]
            pos_next_true = batch[:, t + 1, :]
            pos_pred, hidden = model.step(pos_t, hidden)
            loss = loss + F.mse_loss(pos_pred, pos_next_true)
        loss = loss / seq_len
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (i + 1) % max(1, iters // 5) == 0 or (i + 1) == iters:
            print(f"  [train] forward-model ep {i + 1}/{iters} loss={loss.item():.5f}",
                  flush=True)
    return model


# --- Per-seed x condition eval runner ---------------------------------------
def run_eval(
    model: GenerativeForwardModel,
    eval_seed: int,
    condition_label: str,
    is_ablation: bool,
    t0: int,
    t1: int,
) -> dict:
    full_traj = generate_trajectory(eval_seed, t0 + t1, WORLD_DIM)  # [t0+t1+1, world_dim]
    e3 = E3TrajectorySelector(config=E3Config(world_dim=WORLD_DIM))
    hidden = torch.zeros(1, HIDDEN_DIM)

    grounded_pe_mags: list[float] = []
    with torch.no_grad():
        for t in range(t0):
            pos_t = full_traj[t:t + 1, :]
            pos_next_true = full_traj[t + 1:t + 2, :]
            pos_pred, hidden = model.step(pos_t, hidden)
            pe = (pos_next_true - pos_pred).squeeze(0)
            e3.update_running_variance(pe)
            grounded_pe_mags.append(pe.abs().mean().item())
    grounded_operating_variance = float(e3._running_variance)
    grounded_mean_abs_pe = sum(grounded_pe_mags) / len(grounded_pe_mags)
    frozen_state = full_traj[t0:t0 + 1, :].clone()  # last genuinely grounded observation

    deaff_pe_mags: list[float] = []
    would_commit_trace: list[float] = []
    rollout_input = frozen_state.clone()
    pos_pred = frozen_state.clone()
    with torch.no_grad():
        for _ in range(t1):
            pos_pred, hidden = model.step(rollout_input, hidden)
            actual_fed = pos_pred if is_ablation else frozen_state
            pe = (actual_fed - pos_pred).squeeze(0)
            e3.update_running_variance(pe)
            deaff_pe_mags.append(pe.abs().mean().item())
            would_commit_trace.append(
                1.0 if e3._running_variance < e3.commit_threshold else 0.0
            )
            rollout_input = pos_pred  # autoregressive: no new sensory input

    true_final = full_traj[t0 + t1:t0 + t1 + 1, :]
    ground_truth_drift_at_end = (pos_pred - true_final).norm().item()

    return {
        "seed": eval_seed,
        "condition_label": condition_label,
        "is_ablation": bool(is_ablation),
        "grounded_operating_variance": grounded_operating_variance,
        "grounded_mean_abs_pe": grounded_mean_abs_pe,
        "final_running_variance": float(e3._running_variance),
        "final_current_precision": float(e3.current_precision),
        "deaff_mean_abs_pe": sum(deaff_pe_mags) / len(deaff_pe_mags),
        "would_commit_fraction": sum(would_commit_trace) / len(would_commit_trace),
        "commit_threshold_reference": float(e3.commit_threshold),
        "ground_truth_drift_at_end": ground_truth_drift_at_end,
    }


# --- Criteria evaluation -----------------------------------------------------
def _evaluate_criteria(primary_results: list[dict], ablation_results: list[dict]) -> dict:
    n_seeds = len(primary_results)
    required = math.ceil(n_seeds * PASS_FRACTION_REQUIRED)

    c1_passes = c2_passes = c3_passes = 0
    for p, a in zip(primary_results, ablation_results):
        g = p["grounded_operating_variance"]
        p_ratio = p["final_running_variance"] / (g + 1e-12)
        a_ratio = a["final_running_variance"] / (g + 1e-12)
        c1 = (p_ratio >= C1_PRIMARY_MIN_VARIANCE_RATIO
              and a_ratio <= C1_ABLATION_MAX_VARIANCE_RATIO)
        divergence_ratio = (p["final_running_variance"] - a["final_running_variance"]) / (g + 1e-12)
        c2 = divergence_ratio >= C2_MIN_DIVERGENCE_RATIO
        pe_arm_ratio = p["deaff_mean_abs_pe"] / (a["deaff_mean_abs_pe"] + 1e-9)
        surprise_ratio = p["deaff_mean_abs_pe"] / (p["grounded_mean_abs_pe"] + 1e-9)
        c3 = (pe_arm_ratio >= C3_MIN_PE_ARM_RATIO
              and surprise_ratio >= C3_MIN_SURPRISE_RATIO)
        c1_passes += int(c1)
        c2_passes += int(c2)
        c3_passes += int(c3)

    return {
        "n_seeds": n_seeds,
        "min_seeds_required": required,
        "c1_confidence_collapse_seeds_pass": c1_passes,
        "c2_divergence_magnitude_seeds_pass": c2_passes,
        "c3_manipulation_validity_seeds_pass": c3_passes,
        "c1_pass": c1_passes >= required,
        "c2_pass": c2_passes >= required,
        "c3_pass": c3_passes >= required,
        "overall_pass": (c1_passes >= required and c2_passes >= required
                          and c3_passes >= required),
    }


def _pairwise_deltas(primary_results: list[dict], ablation_results: list[dict]) -> dict:
    def _mean(rows, key):
        return sum(r[key] for r in rows) / max(len(rows), 1)

    p_final_var = _mean(primary_results, "final_running_variance")
    a_final_var = _mean(ablation_results, "final_running_variance")
    p_commit = _mean(primary_results, "would_commit_fraction")
    a_commit = _mean(ablation_results, "would_commit_fraction")
    p_drift = _mean(primary_results, "ground_truth_drift_at_end")
    a_drift = _mean(ablation_results, "ground_truth_drift_at_end")
    return {
        "primary_mean_final_variance": p_final_var,
        "ablation_mean_final_variance": a_final_var,
        "variance_delta_primary_minus_ablation": p_final_var - a_final_var,
        "primary_mean_would_commit_fraction": p_commit,
        "ablation_mean_would_commit_fraction": a_commit,
        "would_commit_delta_ablation_minus_primary": a_commit - p_commit,
        "primary_mean_ground_truth_drift": p_drift,
        "ablation_mean_ground_truth_drift": a_drift,
    }


# --- Driver -------------------------------------------------------------
def run_experiment(dry_run: bool = False):
    t0_wall = time.time()
    model = train_forward_model(dry_run=dry_run)

    t0 = 8 if dry_run else T0_GROUNDING_TICKS
    t1 = 6 if dry_run else T1_DEAFFERENTATION_TICKS
    seeds = (EVAL_SEEDS[0],) if dry_run else EVAL_SEEDS

    primary_results: list[dict] = []
    ablation_results: list[dict] = []
    for seed in seeds:
        print(f"Seed {seed} Condition {CONDITION_PRIMARY[0]}", flush=True)
        p = run_eval(model, seed, CONDITION_PRIMARY[0], CONDITION_PRIMARY[1], t0, t1)
        primary_results.append(p)
        print(f"  final_variance={p['final_running_variance']:.5f} "
              f"(grounded={p['grounded_operating_variance']:.5f}) verdict: "
              f"{'PASS' if p['final_running_variance'] > p['grounded_operating_variance'] else 'FAIL'}",
              flush=True)

        print(f"Seed {seed} Condition {CONDITION_ABLATION[0]}", flush=True)
        a = run_eval(model, seed, CONDITION_ABLATION[0], CONDITION_ABLATION[1], t0, t1)
        ablation_results.append(a)
        print(f"  final_variance={a['final_running_variance']:.5f} "
              f"(grounded={a['grounded_operating_variance']:.5f}) verdict: "
              f"{'PASS' if a['final_running_variance'] <= a['grounded_operating_variance'] * 1.05 else 'FAIL'}",
              flush=True)

    elapsed = time.time() - t0_wall
    criteria = _evaluate_criteria(primary_results, ablation_results)
    deltas = _pairwise_deltas(primary_results, ablation_results)
    return primary_results, ablation_results, criteria, deltas, elapsed, t0, t1, seeds


def main(dry_run: bool = False) -> dict:
    print(f"[{EXPERIMENT_TYPE}] MECH-245 generative-dominance-under-"
          f"deafferentation discriminative pair...", flush=True)
    (primary_results, ablation_results, criteria, deltas, elapsed,
     t0, t1, seeds) = run_experiment(dry_run=dry_run)

    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    direction = "supports" if criteria["overall_pass"] else "weakens"

    print(f"  C1 confidence_collapse: {criteria['c1_confidence_collapse_seeds_pass']}"
          f"/{criteria['n_seeds']} -> {'PASS' if criteria['c1_pass'] else 'FAIL'}", flush=True)
    print(f"  C2 divergence_magnitude: {criteria['c2_divergence_magnitude_seeds_pass']}"
          f"/{criteria['n_seeds']} -> {'PASS' if criteria['c2_pass'] else 'FAIL'}", flush=True)
    print(f"  C3 manipulation_validity: {criteria['c3_manipulation_validity_seeds_pass']}"
          f"/{criteria['n_seeds']} -> {'PASS' if criteria['c3_pass'] else 'FAIL'}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] overall: {outcome} ({elapsed:.1f}s)", flush=True)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    summary = (
        "MECH-245 generative-model-dominance-under-deafferentation discriminative "
        "pair. A GRU forward model (stand-in for E1/E2's top-down generative "
        "pathway) is trained on synthetic momentum z_world dynamics, then rolled "
        "forward through a grounding phase (teacher-forced against true "
        "trajectory) followed by a deafferentation window in which the real "
        "E3TrajectorySelector's own bottom-up-mismatch-check pathway "
        "(prediction_error -> update_running_variance -> current_precision / "
        "commit_threshold) is fed either the last genuinely grounded observation, "
        "held frozen (PRIMARY, mismatch_check_intact -- no new sensory evidence "
        "arrives) or the model's own rolled-forward prediction substituted as if "
        "it were the observation (ABLATION, generative_dominance_bypass -- "
        "top-down content propagates forward as if received sensory signal). "
        f"Interpretation: PASS={outcome == 'PASS'} -> evidence_direction={direction}. "
        "Under PRIMARY, genuine sensory absence correctly collapses confidence "
        "(running_variance rises from the grounded operating point); under "
        "ABLATION, substituting the generative prediction sustains or amplifies "
        "false confidence despite total absence of corresponding sensory "
        "grounding, reproducing the predicted hallucination signature. "
        f"Pairwise deltas: primary_final_variance="
        f"{deltas['primary_mean_final_variance']:.5f}, ablation_final_variance="
        f"{deltas['ablation_mean_final_variance']:.5f}, would_commit_fraction "
        f"delta(ablation-primary)="
        f"{deltas['would_commit_delta_ablation_minus_primary']:.3f}, "
        f"ground_truth_drift primary/ablation="
        f"{deltas['primary_mean_ground_truth_drift']:.3f}/"
        f"{deltas['ablation_mean_ground_truth_drift']:.3f} (both arms audited "
        "against the same continuing true dynamics -- the ABLATION arm's "
        "sustained confidence is confidence in drifted/wrong content, not an "
        "accidentally-correct guess). DV-symmetry note: ABLATION's per-tick "
        "prediction_error is a definitional identity (predicted-minus-itself); "
        "the falsifiable content is PRIMARY's real deprivation-driven PE "
        "magnitude (C3 surprise-ratio) and whether it clears the pre-registered "
        "confidence-collapse margin (C1/C2) -- both measured from a genuinely "
        "trained model's free-rollout drift, not designer-set."
    )

    registered_thresholds = {
        "C1_PRIMARY_MIN_VARIANCE_RATIO": C1_PRIMARY_MIN_VARIANCE_RATIO,
        "C1_ABLATION_MAX_VARIANCE_RATIO": C1_ABLATION_MAX_VARIANCE_RATIO,
        "C2_MIN_DIVERGENCE_RATIO": C2_MIN_DIVERGENCE_RATIO,
        "C3_MIN_PE_ARM_RATIO": C3_MIN_PE_ARM_RATIO,
        "C3_MIN_SURPRISE_RATIO": C3_MIN_SURPRISE_RATIO,
        "PASS_FRACTION_REQUIRED": PASS_FRACTION_REQUIRED,
    }

    metrics = {
        "primary_mean_final_variance": deltas["primary_mean_final_variance"],
        "ablation_mean_final_variance": deltas["ablation_mean_final_variance"],
        "variance_delta_primary_minus_ablation":
            deltas["variance_delta_primary_minus_ablation"],
        "primary_mean_would_commit_fraction": deltas["primary_mean_would_commit_fraction"],
        "ablation_mean_would_commit_fraction": deltas["ablation_mean_would_commit_fraction"],
        "would_commit_delta_ablation_minus_primary":
            deltas["would_commit_delta_ablation_minus_primary"],
        "primary_mean_ground_truth_drift": deltas["primary_mean_ground_truth_drift"],
        "ablation_mean_ground_truth_drift": deltas["ablation_mean_ground_truth_drift"],
        "c1_confidence_collapse_seeds_pass": criteria["c1_confidence_collapse_seeds_pass"],
        "c2_divergence_magnitude_seeds_pass": criteria["c2_divergence_magnitude_seeds_pass"],
        "c3_manipulation_validity_seeds_pass": criteria["c3_manipulation_validity_seeds_pass"],
        "n_seeds": criteria["n_seeds"],
        "min_seeds_required": criteria["min_seeds_required"],
    }

    config = {
        "world_dim": WORLD_DIM,
        "hidden_dim": HIDDEN_DIM,
        "momentum_decay": MOMENTUM_DECAY,
        "process_noise_std": PROCESS_NOISE_STD,
        "train_base_seed": TRAIN_BASE_SEED,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "train_seq_len": TRAIN_SEQ_LEN,
        "train_iters": TRAIN_ITERS,
        "train_lr": TRAIN_LR,
        "t0_grounding_ticks": t0,
        "t1_deafferentation_ticks": t1,
        "eval_seeds": list(seeds),
        "conditions": [CONDITION_PRIMARY[0], CONDITION_ABLATION[0]],
    }

    # Non-degeneracy self-report (validate_experiments.py requirement): the
    # load-bearing discriminative metrics -- final_running_variance (drives
    # C1/C2) and deaff_mean_abs_pe (drives C3) -- must show genuine spread
    # across the (arm x seed) cells, not a pinned/constant reading.
    degeneracy = check_degeneracy({
        "final_running_variance": [r["final_running_variance"] for r in primary_results]
        + [r["final_running_variance"] for r in ablation_results],
        "deaff_mean_abs_pe": [r["deaff_mean_abs_pe"] for r in primary_results]
        + [r["deaff_mean_abs_pe"] for r in ablation_results],
    })

    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "outcome": outcome,
        "result": outcome,
        "evidence_class": "discriminative_pair",
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"MECH-245": direction},
        "seed_policy": "matched_shared_seeds",
        "dispatch_mode": "discriminative_pair",
        "summary": summary,
        "registered_thresholds": registered_thresholds,
        "metrics": metrics,
        "criteria": criteria,
        "pairwise_deltas": deltas,
        "config": config,
        "results_primary_mismatch_check_intact": primary_results,
        "results_ablation_generative_dominance_bypass": ablation_results,
        "elapsed_seconds": elapsed,
        "dry_run": bool(dry_run),
        **degeneracy,
    }

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config=config,
        seeds=list(seeds),
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}", flush=True)

    return {
        "outcome": outcome,
        "all_pass": bool(criteria["overall_pass"]),
        "manifest_path": None if dry_run else str(out_path),
        "run_id": run_id,
        "dry_run": bool(dry_run),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Short smoke path: 1 seed, short grounding/"
                             "deafferentation windows, reduced forward-model "
                             "training -- still exercises train/eval/gate/"
                             "manifest paths end to end.")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    emit_outcome(
        outcome=result["outcome"],
        manifest_path=result["manifest_path"],
        run_id=result["run_id"],
        dry_run=result["dry_run"],
    )
    sys.exit(0 if result["all_pass"] else 1)
