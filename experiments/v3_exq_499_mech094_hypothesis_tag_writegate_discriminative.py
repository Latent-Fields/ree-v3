"""V3-EXQ-499 -- MECH-094 hypothesis-tag write-gate discriminative pair (EXP-0170).

Claim: MECH-094 (default_mode.simulation_real_distinction_write_profile)
Proposal: EXP-0170
Successor of (NOT supersede): V3-EXQ-140 (FAIL non_contributory) and the
V3-EXQ-465 diagnostic-probe family (which validated the gate at substrate level
but produced no evidence-grade discriminative signal).

What MECH-094 asserts
---------------------
The hypothesis tag is a categorical write gate, not a precision attenuation:
when hypothesis_tag=True (replay / simulation / DMN content), the post-commit
write profile is suppressed -- harm signals from simulated content do NOT
update the residue accumulation buffer (RBF weights, valence vectors).
Real-interaction events (hypothesis_tag=False) are the only inputs that
modify the consolidation substrate. Tag loss is the proposed PTSD /
confabulation mechanism: simulated harm leaks into the episodic substrate
as if it were real.

V3 substrate map (verified 2026-04-29):
  LatentState.hypothesis_tag      ree_core/latent/stack.py:699
  Set False in waking sense       ree_core/agent.py:1556, 2103
  Set True in replay              ree_core/agent.py:2747 (comment)
  Gate (accumulate)               ree_core/residue/field.py:282
  Gate (accumulate_benefit)       ree_core/residue/field.py:335
  Gate (update_valence)           ree_core/residue/field.py:372

Why the prior attempt failed
----------------------------
V3-EXQ-140 ran a full episode-loop discriminative pair with the gate
controlling which harm events trained the harm_eval head. It returned
non_contributory because grid-task harm density was too low: the
contamination signal was real but below the noise floor of the harm_eval
head's downstream training. EXQ-465 confirmed the gate works at substrate
level via a single-event probe (UC4) but its design was diagnostic, so the
PASS does not contribute evidence to MECH-094 confidence.

This experiment fixes both: it operates at the substrate level (direct
ResidueField.accumulate calls -- no env-loop noise) AND uses a forced-
injection design (deterministic N_REAL real events + N_SIM simulation
events at controlled distinct locations) so the contamination signal is
measurable per call. Pre-registered thresholds + 3-seed PASS criterion in
>= 2/3 seeds give it evidence-grade interpretability.

Discriminative pair
-------------------
ARM_A (TAG_ROUTING_ON, MECH-094 architecture intact):
  Real events  -> hypothesis_tag=False -> gate inactive -> RBF weights update
  Sim events   -> hypothesis_tag=True  -> gate active   -> RBF weights NOT updated

ARM_B (TAG_ROUTING_OFF, gate ablated by passing wrong tag):
  Real events  -> hypothesis_tag=False -> gate inactive -> RBF weights update
  Sim events   -> hypothesis_tag=False -> gate inactive -> RBF weights update
  (The tag is still set internally by the agent; the ablation is at the
  consumer, simulating a tag-loss / write-channel disinhibition pathology.)

Three pre-registered metrics (per the EXP-0170 objective)
---------------------------------------------------------
M1: write_channel_contamination_index
    = sum(|RBF_weight_delta at SIM locations|) / (sum(|RBF_weight_delta at
      REAL locations|) + 1e-9)
    Operationalises "fraction of episodic-memory entries originating from
    tagged simulation but classified as real". Ranges in [0, ~1].
    ARM_A predicts ~0; ARM_B predicts ~1.

M2: confabulation_rate
    = fraction of SIM locations whose post-experiment RBF weight magnitude
      exceeds the median weight at REAL locations.
    "Confabulation" here = a simulated content trace whose residue signature
    is indistinguishable from a real one.
    ARM_A predicts ~0; ARM_B predicts >= 0.30 (close to 0.5 in the limit
    of equal R / S weights, with measurement noise).

M3: routing_separation
    = mutual information between (hypothesis_tag, write_occurred) over the
      N_REAL + N_SIM events.
    ARM_A predicts MI close to log(2) ~= 0.693 (perfect anti-correlation:
    write_occurred=True iff tag=False).
    ARM_B predicts MI ~ 0 (writes happen regardless of tag value).

PASS criteria (>= 2/3 seeds for each):
  C1: ARM_A contamination_index < 0.05
      ARM_B contamination_index > 0.50
      ARM_B - ARM_A > 0.40
  C2: ARM_A confabulation_rate <= 0.05
      ARM_B confabulation_rate >= 0.30
  C3: ARM_A routing_separation >= 0.50  (close to log(2)~=0.693)
      ARM_B routing_separation <= 0.05

PASS overall = C1 AND C2 AND C3.
PASS supports MECH-094 (gate is active, load-bearing, and the predicted
tag-loss pathology is reproducible). FAIL with C1 alone passing -> gate is
present but not load-bearing in some downstream sense (governance walk).
FAIL with C1 also failing -> gate is broken / not wired (substrate
regression -- run /diagnose-errors).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_499_mech094_hypothesis_tag_writegate_discriminative.py
  /opt/local/bin/python3 experiments/v3_exq_499_mech094_hypothesis_tag_writegate_discriminative.py --dry-run

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.residue.field import (  # noqa: E402
    ResidueField,
    VALENCE_HARM_DISCRIMINATIVE,
)
from ree_core.utils.config import ResidueConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_499_mech094_hypothesis_tag_writegate_discriminative"
CLAIM_IDS = ["MECH-094"]
EXPERIMENT_PURPOSE = "evidence"

# --- Configuration --------------------------------------------------------
WORLD_DIM = 32
NUM_BASIS_FUNCTIONS = 64  # plenty of slots so REAL/SIM locations don't collide
N_REAL = 50               # real-interaction events per condition
N_SIM = 50                # simulation events per condition
HARM_MAGNITUDE = 0.5      # per-event accumulator input magnitude
SEEDS = (42, 43, 44)

# Pre-registered thresholds.
C1_ARM_A_MAX_CONTAMINATION = 0.05
C1_ARM_B_MIN_CONTAMINATION = 0.50
C1_MIN_ARM_DELTA = 0.40
C2_ARM_A_MAX_CONFABULATION = 0.05
C2_ARM_B_MIN_CONFABULATION = 0.30
C3_ARM_A_MIN_SEPARATION = 0.50
C3_ARM_B_MAX_SEPARATION = 0.05
PASS_FRACTION_REQUIRED = 2.0 / 3.0


# --- Helpers --------------------------------------------------------------
def _make_residue() -> ResidueField:
    cfg = ResidueConfig(
        world_dim=WORLD_DIM,
        hidden_dim=WORLD_DIM,
        num_basis_functions=NUM_BASIS_FUNCTIONS,
        valence_enabled=True,
    )
    return ResidueField(cfg)


def _generate_event_locations(seed: int, n_real: int, n_sim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Disjoint deterministic z_world locations for REAL and SIM event sets."""
    g = torch.Generator()
    g.manual_seed(seed)
    real_locs = torch.randn((n_real, WORLD_DIM), generator=g)
    sim_locs = torch.randn((n_sim, WORLD_DIM), generator=g) + 5.0  # offset so SIM not in REAL basin
    return real_locs, sim_locs


def _measure_rbf_weight_signature(
    residue: ResidueField,
    locations: torch.Tensor,
) -> torch.Tensor:
    """Per-location RBF activation x weight: a scalar per location.

    Reads the current weight magnitude as projected at each query location
    via the RBF kernel. This is the substrate-level analog of "how strong
    is the residue signature at this point."
    """
    centers = residue.rbf_field.centers           # [N_centers, world_dim]
    weights = residue.rbf_field.weights           # [N_centers]
    bandwidth = residue.rbf_field.bandwidth        # scalar
    # squared distances [N_loc, N_centers]
    diff = locations.unsqueeze(1) - centers.unsqueeze(0)
    sq = (diff * diff).sum(dim=-1)
    activations = torch.exp(-sq / (2.0 * bandwidth * bandwidth))
    return (activations * weights.unsqueeze(0)).sum(dim=-1).abs()


def _mutual_information_binary(tag_seq: list[bool], write_seq: list[bool]) -> float:
    """Discrete MI in bits (log base 2) between two binary sequences.

    log(2) ~= 0.693 in nats; here we report nats to match the manifest
    description in the docstring -- log_e(2). Use natural log for both
    joint and marginals so the unit cancels.
    """
    n = len(tag_seq)
    if n == 0:
        return 0.0
    counts = {(False, False): 0, (False, True): 0, (True, False): 0, (True, True): 0}
    for t, w in zip(tag_seq, write_seq):
        counts[(bool(t), bool(w))] += 1
    p_xy = {k: v / n for k, v in counts.items()}
    p_x = {False: p_xy[(False, False)] + p_xy[(False, True)],
           True:  p_xy[(True,  False)] + p_xy[(True,  True)]}
    p_y = {False: p_xy[(False, False)] + p_xy[(True, False)],
           True:  p_xy[(False, True)]  + p_xy[(True, True)]}
    mi = 0.0
    for (x, y), p in p_xy.items():
        if p <= 0.0:
            continue
        denom = p_x[x] * p_y[y]
        if denom <= 0.0:
            continue
        mi += p * math.log(p / denom)
    return mi


# --- Per-arm runner -------------------------------------------------------
def run_arm(
    seed: int,
    arm_label: str,
    sim_event_tag: bool,
    n_real: int = N_REAL,
    n_sim: int = N_SIM,
    harm_magnitude: float = HARM_MAGNITUDE,
) -> dict:
    """Run one arm.

    sim_event_tag=True   -> ARM_A: tag is True for sim events; gate active.
    sim_event_tag=False  -> ARM_B: tag is False for sim events; gate inactive.
    Real events always pass tag=False.
    """
    torch.manual_seed(seed)
    residue = _make_residue()
    real_locs, sim_locs = _generate_event_locations(seed, n_real, n_sim)

    # Baseline signature BEFORE any writes.
    real_sig_before = _measure_rbf_weight_signature(residue, real_locs).clone()
    sim_sig_before = _measure_rbf_weight_signature(residue, sim_locs).clone()

    tag_seq: list[bool] = []
    write_seq: list[bool] = []
    weights_history: list[torch.Tensor] = []

    # Interleave real and sim events to remove ordering effects.
    n_steps = max(n_real, n_sim)
    for i in range(n_steps):
        if i < n_real:
            w_pre = residue.rbf_field.weights.detach().clone()
            residue.accumulate(real_locs[i:i + 1], harm_magnitude=harm_magnitude,
                               hypothesis_tag=False)
            residue.update_valence(real_locs[i:i + 1], VALENCE_HARM_DISCRIMINATIVE,
                                   harm_magnitude, hypothesis_tag=False)
            wrote = (residue.rbf_field.weights - w_pre).abs().sum().item() > 1e-9
            tag_seq.append(False)
            write_seq.append(wrote)
        if i < n_sim:
            w_pre = residue.rbf_field.weights.detach().clone()
            residue.accumulate(sim_locs[i:i + 1], harm_magnitude=harm_magnitude,
                               hypothesis_tag=sim_event_tag)
            residue.update_valence(sim_locs[i:i + 1], VALENCE_HARM_DISCRIMINATIVE,
                                   harm_magnitude, hypothesis_tag=sim_event_tag)
            wrote = (residue.rbf_field.weights - w_pre).abs().sum().item() > 1e-9
            # Record the tag we passed to the gate, not the "true" provenance.
            # Under ARM_A this matches provenance; under ARM_B the gate sees
            # the wrong tag, which is the ablation.
            tag_seq.append(bool(sim_event_tag))
            write_seq.append(wrote)
        weights_history.append(residue.rbf_field.weights.detach().clone())

    # Post-experiment signatures.
    real_sig_after = _measure_rbf_weight_signature(residue, real_locs)
    sim_sig_after = _measure_rbf_weight_signature(residue, sim_locs)
    real_delta = (real_sig_after - real_sig_before).abs()
    sim_delta = (sim_sig_after - sim_sig_before).abs()

    # M1 contamination index.
    real_total = real_delta.sum().item()
    sim_total = sim_delta.sum().item()
    contamination_index = sim_total / (real_total + 1e-9)

    # M2 confabulation rate (sim-location signatures judged as real).
    real_median = real_sig_after.median().item()
    confabulation_rate = (sim_sig_after >= real_median).float().mean().item()

    # M3 routing separation = MI(tag, write_occurred) in nats.
    routing_separation = _mutual_information_binary(tag_seq, write_seq)

    return {
        "seed": seed,
        "arm_label": arm_label,
        "sim_event_tag_passed": bool(sim_event_tag),
        "n_real_events": n_real,
        "n_sim_events": n_sim,
        "real_weight_delta_total": real_total,
        "sim_weight_delta_total": sim_total,
        "real_signature_median_after": real_median,
        "sim_signature_median_after": sim_sig_after.median().item(),
        "contamination_index": contamination_index,
        "confabulation_rate": confabulation_rate,
        "routing_separation": routing_separation,
    }


# --- Aggregation + criteria ----------------------------------------------
def _evaluate_criteria(
    arm_a_results: list[dict],
    arm_b_results: list[dict],
) -> dict:
    """Per-seed pass counts compared against pre-registered thresholds."""
    n_seeds = len(arm_a_results)
    required = math.ceil(n_seeds * PASS_FRACTION_REQUIRED)

    c1_passes = 0
    c2_passes = 0
    c3_passes = 0
    for a, b in zip(arm_a_results, arm_b_results):
        c1 = (a["contamination_index"] < C1_ARM_A_MAX_CONTAMINATION
              and b["contamination_index"] > C1_ARM_B_MIN_CONTAMINATION
              and (b["contamination_index"] - a["contamination_index"]) > C1_MIN_ARM_DELTA)
        c2 = (a["confabulation_rate"] <= C2_ARM_A_MAX_CONFABULATION
              and b["confabulation_rate"] >= C2_ARM_B_MIN_CONFABULATION)
        c3 = (a["routing_separation"] >= C3_ARM_A_MIN_SEPARATION
              and b["routing_separation"] <= C3_ARM_B_MAX_SEPARATION)
        c1_passes += int(c1)
        c2_passes += int(c2)
        c3_passes += int(c3)

    return {
        "n_seeds": n_seeds,
        "min_seeds_required": required,
        "c1_contamination_seeds_pass": c1_passes,
        "c2_confabulation_seeds_pass": c2_passes,
        "c3_routing_separation_seeds_pass": c3_passes,
        "c1_pass": c1_passes >= required,
        "c2_pass": c2_passes >= required,
        "c3_pass": c3_passes >= required,
        "overall_pass": (c1_passes >= required
                         and c2_passes >= required
                         and c3_passes >= required),
    }


# --- Driver ---------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run a single seed and print results without writing manifest.")
    args = parser.parse_args()

    seeds = (SEEDS[0],) if args.dry_run else SEEDS
    t0 = time.time()
    arm_a_results = [run_arm(s, "ARM_A_tag_routing_on", sim_event_tag=True) for s in seeds]
    arm_b_results = [run_arm(s, "ARM_B_tag_routing_off", sim_event_tag=False) for s in seeds]
    elapsed = time.time() - t0

    criteria = _evaluate_criteria(arm_a_results, arm_b_results)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    direction = "supports" if criteria["overall_pass"] else "weakens"

    print(f"V3-EXQ-499 (MECH-094) -- {outcome} in {elapsed:.1f}s "
          f"({len(seeds)} seed(s))")
    for label, results in (("ARM_A_tag_routing_on", arm_a_results),
                           ("ARM_B_tag_routing_off", arm_b_results)):
        for r in results:
            print(f"  {label} seed={r['seed']}  "
                  f"contam={r['contamination_index']:.3f}  "
                  f"confab={r['confabulation_rate']:.3f}  "
                  f"routing_MI={r['routing_separation']:.3f}")
    print(f"  C1 contamination: {criteria['c1_contamination_seeds_pass']}/{criteria['n_seeds']} "
          f"-> {'PASS' if criteria['c1_pass'] else 'FAIL'}")
    print(f"  C2 confabulation: {criteria['c2_confabulation_seeds_pass']}/{criteria['n_seeds']} "
          f"-> {'PASS' if criteria['c2_pass'] else 'FAIL'}")
    print(f"  C3 routing_sep:   {criteria['c3_routing_separation_seeds_pass']}/{criteria['n_seeds']} "
          f"-> {'PASS' if criteria['c3_pass'] else 'FAIL'}")

    if args.dry_run:
        print("[--dry-run] manifest not written.")
        return

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "result": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"MECH-094": direction},
        "criteria": criteria,
        "registered_thresholds": {
            "C1_ARM_A_MAX_CONTAMINATION": C1_ARM_A_MAX_CONTAMINATION,
            "C1_ARM_B_MIN_CONTAMINATION": C1_ARM_B_MIN_CONTAMINATION,
            "C1_MIN_ARM_DELTA": C1_MIN_ARM_DELTA,
            "C2_ARM_A_MAX_CONFABULATION": C2_ARM_A_MAX_CONFABULATION,
            "C2_ARM_B_MIN_CONFABULATION": C2_ARM_B_MIN_CONFABULATION,
            "C3_ARM_A_MIN_SEPARATION": C3_ARM_A_MIN_SEPARATION,
            "C3_ARM_B_MAX_SEPARATION": C3_ARM_B_MAX_SEPARATION,
            "PASS_FRACTION_REQUIRED": PASS_FRACTION_REQUIRED,
        },
        "config": {
            "world_dim": WORLD_DIM,
            "num_basis_functions": NUM_BASIS_FUNCTIONS,
            "n_real": N_REAL,
            "n_sim": N_SIM,
            "harm_magnitude": HARM_MAGNITUDE,
            "seeds": list(seeds),
        },
        "results_arm_a_tag_routing_on": arm_a_results,
        "results_arm_b_tag_routing_off": arm_b_results,
        "elapsed_seconds": elapsed,
        "notes": (
            "Substrate-level discriminative pair for MECH-094. ARM_A passes "
            "hypothesis_tag=True for sim events (gate active); ARM_B passes "
            "False for sim events (gate inactive -- the tag-loss pathology "
            "predicted to underwrite confabulation / PTSD). Avoids the "
            "non_contributory failure mode of V3-EXQ-140 (insufficient harm "
            "density in env-loop) by operating directly on ResidueField "
            "with a forced-injection design: N_REAL deterministic real "
            "events + N_SIM deterministic sim events at disjoint z_world "
            "locations. Three pre-registered metrics: (M1) "
            "write_channel_contamination_index, (M2) confabulation_rate "
            "(sim signatures judged as real), (M3) routing_separation "
            "(MI between tag and write_occurred). PASS supports MECH-094 "
            "load-bearing status; FAIL with C1 alone passing routes "
            "evidence to a downstream consumer gap; FAIL with C1 also "
            "failing routes to substrate regression (run /diagnose-errors)."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}")


if __name__ == "__main__":
    main()
