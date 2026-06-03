"""V3-EXQ-633 -- MECH-094 simulation-vs-real write-gate discriminative pair.

Proposal EXP-0080 / backlog EVB-0234. Claim: MECH-094
(default_mode.simulation_real_distinction_write_profile -- the hypothesis-tag
categorical write gate; tag loss = PTSD / confabulation mechanism).

What MECH-094 asserts (the simulation-vs-real framing)
------------------------------------------------------
The system maintains a functional distinction between simulation-mode content
(replay / DMN / rehearsal, carrying hypothesis_tag=True) and real-experience
content (waking observation, hypothesis_tag=False), such that simulation
content does NOT corrupt the world model / stored representations. The
hypothesis tag is a categorical write gate (phi(z) write gate): when
hypothesis_tag=True the post-commit write profile is suppressed, so harm /
valence signals from simulated content do not update the persistent
representation substrate (ResidueField RBF weights + valence vectors).
Real-interaction events (hypothesis_tag=False) are the only inputs that
modify the substrate. Tag loss is the proposed PTSD / confabulation pathology:
simulated harm leaks into the episodic substrate as if it were real.

This experiment runs ONE claim-focused discriminative pair (dispatch_mode =
discriminative_pair):

  PRIMARY condition (GATE_ON, MECH-094 mechanism intact):
    Real events  -> hypothesis_tag=False -> gate inactive -> substrate updates.
    Sim events   -> hypothesis_tag=True  -> gate active   -> substrate NOT updated.

  ABLATION / CONTROL condition (GATE_OFF, write-gate disinhibited):
    Real events  -> hypothesis_tag=False -> gate inactive -> substrate updates.
    Sim events   -> hypothesis_tag=False -> gate inactive -> substrate updates.
    (The tag is still "really" a simulation event; the ablation strips the tag
    at the consumer, modelling the tag-loss / write-channel disinhibition
    pathology MECH-094 predicts underwrites confabulation / PTSD.)

The discriminating quantity is whether the simulation stream corrupts the
real-experience world model. We inject a controlled set of REAL events at one
disjoint set of z_world locations and a controlled set of SIMULATION events at
a second disjoint set of z_world locations, then measure how much of the
post-experiment substrate signature at the SIMULATION locations originated from
the simulation stream. Under GATE_ON this is ~0 (simulation cannot write);
under GATE_OFF the simulation stream writes freely and corrupts the substrate.

Substrate map (verified against ree-v3 source 2026-06-03):
  ResidueField.accumulate(z_world, harm_magnitude, world_delta, hypothesis_tag)
    -- ree_core/residue/field.py:385 (MECH-094 gate at line 410).
  ResidueField.update_valence(z_world, component, value, hypothesis_tag)
    -- ree_core/residue/field.py:522 (MECH-094 gate at line 544).
  RBFLayer.weights / .centers / .bandwidth / .active_mask
    -- ree_core/residue/field.py:74-92.
  ResidueConfig(world_dim, hidden_dim, num_basis_functions, valence_enabled)
    -- ree_core/utils/config.py:1017.
  VALENCE_HARM_DISCRIMINATIVE -- ree_core/residue/field.py:59.
ree-v3/CLAUDE.md confirms the MECH-094 write-gate substrate is implemented
("All replay/simulation content must carry hypothesis_tag=True (MECH-094)").

Pre-registered metrics (computed per seed x condition run)
----------------------------------------------------------
M1 sim_world_model_contamination_index
   = sum(|substrate signature delta at SIM locations|)
     / (sum(|substrate signature delta at REAL locations|) + 1e-9)
   Operationalises "fraction of world-model change at simulation locations
   relative to real locations". GATE_ON predicts ~0; GATE_OFF predicts ~1.

M2 confabulation_rate
   = fraction of SIM locations whose post-experiment substrate signature
     magnitude exceeds the median signature at REAL locations.
   A simulation trace whose substrate signature is indistinguishable from a
   real one is a confabulation. GATE_ON predicts ~0; GATE_OFF predicts high.

M3 write_routing_separation
   = mutual information (nats) between (hypothesis_tag_passed, write_occurred)
     over all real + sim events. GATE_ON predicts MI ~ log(2) ~= 0.693
     (write iff tag=False); GATE_OFF predicts MI ~ 0 (writes regardless of tag).

Pre-registered pass thresholds (constants below; NOT derived from run stats):
  C1 (contamination): GATE_ON M1 < 0.05 AND GATE_OFF M1 > 0.50
                      AND (GATE_OFF M1 - GATE_ON M1) > 0.40
  C2 (confabulation): GATE_ON M2 <= 0.05 AND GATE_OFF M2 >= 0.30
  C3 (routing sep):   GATE_ON M3 >= 0.50 AND GATE_OFF M3 <= 0.05
PASS overall = C1 AND C2 AND C3 in >= ceil(n_seeds * PASS_FRACTION_REQUIRED)
seeds. PASS supports MECH-094 (the simulation/real write gate is active,
load-bearing, and the predicted tag-loss confabulation pathology is
reproducible). FAIL with C1 alone failing -> gate broken / not wired
(substrate regression -> /diagnose-errors).

Phased training: NOT applicable. This is a substrate-level discriminative pair
operating by direct ResidueField writes; no head is trained on z_world / z_harm
/ encoder output, so the P0-encoder-warmup -> P1-frozen-encoder-detached-head
-> P2-eval protocol does not apply (matches the V3-EXQ-499 precedent that
produced a clean "supports" result).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_633_mech094_simulation_real_writegate_discriminative.py
  /opt/local/bin/python3 experiments/v3_exq_633_mech094_simulation_real_writegate_discriminative.py --dry-run
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

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.residue.field import (  # noqa: E402
    ResidueField,
    VALENCE_HARM_DISCRIMINATIVE,
)
from ree_core.utils.config import ResidueConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_633_mech094_simulation_real_writegate_discriminative"
CLAIM_IDS = ["MECH-094"]
EXPERIMENT_PURPOSE = "evidence"

# --- Configuration --------------------------------------------------------
WORLD_DIM = 32
NUM_BASIS_FUNCTIONS = 64        # plenty of slots so REAL/SIM locations don't collide
N_REAL = 50                     # real-experience events per condition run
N_SIM = 50                      # simulation-mode events per condition run
HARM_MAGNITUDE = 0.5            # per-event accumulator input magnitude
SEEDS = (42, 43, 44)            # matched_shared_seeds: same seeds for BOTH conditions

# The two arms of the discriminative pair. Each tuple is (label, sim_tag_passed).
# sim_tag_passed is the hypothesis_tag value the SIM events present to the gate.
#   PRIMARY (GATE_ON):  sim events tagged True  -> gate blocks the write.
#   ABLATION (GATE_OFF): sim events tagged False -> gate admits the write.
# Real events ALWAYS pass hypothesis_tag=False in both arms.
CONDITION_PRIMARY = ("gate_on_mech094_intact", True)
CONDITION_ABLATION = ("gate_off_tag_loss", False)
CONDITIONS = (CONDITION_PRIMARY, CONDITION_ABLATION)

# --- Pre-registered thresholds (constants; not derived from run statistics) -
C1_GATE_ON_MAX_CONTAMINATION = 0.05
C1_GATE_OFF_MIN_CONTAMINATION = 0.50
C1_MIN_ARM_DELTA = 0.40
C2_GATE_ON_MAX_CONFABULATION = 0.05
C2_GATE_OFF_MIN_CONFABULATION = 0.30
C3_GATE_ON_MIN_SEPARATION = 0.50
C3_GATE_OFF_MAX_SEPARATION = 0.05
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


def _generate_event_locations(
    seed: int, n_real: int, n_sim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Disjoint deterministic z_world locations for REAL and SIM event sets.

    SIM locations are offset by +5.0 so they do not sit in the REAL basin and
    the per-location signature reads cleanly separate which stream wrote.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    real_locs = torch.randn((n_real, WORLD_DIM), generator=g)
    sim_locs = torch.randn((n_sim, WORLD_DIM), generator=g) + 5.0
    return real_locs, sim_locs


def _measure_signature(residue: ResidueField, locations: torch.Tensor) -> torch.Tensor:
    """Per-location RBF activation x weight: a scalar substrate signature per loc.

    This is the substrate-level analog of "how strong is the world-model
    representation at this point". Reads the current weight magnitude projected
    at each query location through the RBF kernel.
    """
    centers = residue.rbf_field.centers          # [N_centers, world_dim]
    weights = residue.rbf_field.weights          # [N_centers]
    bandwidth = residue.rbf_field.bandwidth      # scalar
    diff = locations.unsqueeze(1) - centers.unsqueeze(0)
    sq = (diff * diff).sum(dim=-1)               # [N_loc, N_centers]
    activations = torch.exp(-sq / (2.0 * bandwidth * bandwidth))
    return (activations * weights.unsqueeze(0)).sum(dim=-1).abs()


def _mutual_information_binary(tag_seq: list[bool], write_seq: list[bool]) -> float:
    """Discrete MI in nats between two binary sequences (log base e)."""
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


# --- Per-condition runner -------------------------------------------------
def run_condition(
    seed: int,
    condition_label: str,
    sim_tag_passed: bool,
    n_real: int = N_REAL,
    n_sim: int = N_SIM,
    harm_magnitude: float = HARM_MAGNITUDE,
    n_train_events: int | None = None,
) -> dict:
    """Run one condition (one arm of the discriminative pair) for one seed.

    sim_tag_passed=True   -> PRIMARY (GATE_ON): sim events carry the tag, gate
                             blocks their write.
    sim_tag_passed=False  -> ABLATION (GATE_OFF): sim events present no tag, gate
                             admits their write (the tag-loss pathology).
    Real events always present hypothesis_tag=False.

    The "training loop" here is the controlled forced-injection of real and
    simulation events into the ResidueField substrate -- the per-event writes
    are the (gated) world-model updates under test.
    """
    torch.manual_seed(seed)
    residue = _make_residue()
    real_locs, sim_locs = _generate_event_locations(seed, n_real, n_sim)

    # Pre-injection signatures.
    real_sig_before = _measure_signature(residue, real_locs).clone()
    sim_sig_before = _measure_signature(residue, sim_locs).clone()

    tag_seq: list[bool] = []
    write_seq: list[bool] = []

    # M = number of injection (training) episodes for this run: interleave real
    # and sim events, one of each per episode where available.
    n_steps = max(n_real, n_sim)
    if n_train_events is not None:
        n_steps = min(n_steps, n_train_events)

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
            # Real provenance is simulation; the tag PRESENTED to the gate is
            # sim_tag_passed (True under GATE_ON, False under GATE_OFF ablation).
            residue.accumulate(sim_locs[i:i + 1], harm_magnitude=harm_magnitude,
                               hypothesis_tag=sim_tag_passed)
            residue.update_valence(sim_locs[i:i + 1], VALENCE_HARM_DISCRIMINATIVE,
                                   harm_magnitude, hypothesis_tag=sim_tag_passed)
            wrote = (residue.rbf_field.weights - w_pre).abs().sum().item() > 1e-9
            tag_seq.append(bool(sim_tag_passed))
            write_seq.append(wrote)
        if (i + 1) % 10 == 0 or (i + 1) == n_steps:
            print(f"[train] seed {seed} {condition_label} ep {i + 1}/{n_steps}",
                  flush=True)

    # Post-injection signatures.
    real_sig_after = _measure_signature(residue, real_locs)
    sim_sig_after = _measure_signature(residue, sim_locs)
    real_delta = (real_sig_after - real_sig_before).abs()
    sim_delta = (sim_sig_after - sim_sig_before).abs()

    # M1 sim_world_model_contamination_index.
    real_total = real_delta.sum().item()
    sim_total = sim_delta.sum().item()
    contamination_index = sim_total / (real_total + 1e-9)

    # M2 confabulation_rate (sim-location signatures judged real).
    real_median = real_sig_after.median().item()
    confabulation_rate = (sim_sig_after >= real_median).float().mean().item()

    # M3 write_routing_separation = MI(tag_passed, write_occurred) in nats.
    write_routing_separation = _mutual_information_binary(tag_seq, write_seq)

    return {
        "seed": seed,
        "condition_label": condition_label,
        "sim_tag_passed": bool(sim_tag_passed),
        "n_real_events": n_real,
        "n_sim_events": n_sim,
        "n_train_events": n_steps,
        "real_weight_delta_total": real_total,
        "sim_weight_delta_total": sim_total,
        "real_signature_median_after": real_median,
        "sim_signature_median_after": sim_sig_after.median().item(),
        "sim_world_model_contamination_index": contamination_index,
        "confabulation_rate": confabulation_rate,
        "write_routing_separation": write_routing_separation,
    }


# --- Aggregation + criteria ----------------------------------------------
def _evaluate_criteria(primary_results: list[dict], ablation_results: list[dict]) -> dict:
    """Per-seed pass counts compared against pre-registered thresholds."""
    n_seeds = len(primary_results)
    required = math.ceil(n_seeds * PASS_FRACTION_REQUIRED)

    c1_passes = 0
    c2_passes = 0
    c3_passes = 0
    for p, a in zip(primary_results, ablation_results):
        c1 = (p["sim_world_model_contamination_index"] < C1_GATE_ON_MAX_CONTAMINATION
              and a["sim_world_model_contamination_index"] > C1_GATE_OFF_MIN_CONTAMINATION
              and (a["sim_world_model_contamination_index"]
                   - p["sim_world_model_contamination_index"]) > C1_MIN_ARM_DELTA)
        c2 = (p["confabulation_rate"] <= C2_GATE_ON_MAX_CONFABULATION
              and a["confabulation_rate"] >= C2_GATE_OFF_MIN_CONFABULATION)
        c3 = (p["write_routing_separation"] >= C3_GATE_ON_MIN_SEPARATION
              and a["write_routing_separation"] <= C3_GATE_OFF_MAX_SEPARATION)
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


# --- Pairwise deltas (for the summary) ------------------------------------
def _pairwise_deltas(primary_results: list[dict], ablation_results: list[dict]) -> dict:
    def _mean(rows, key):
        return sum(r[key] for r in rows) / max(len(rows), 1)

    p_contam = _mean(primary_results, "sim_world_model_contamination_index")
    a_contam = _mean(ablation_results, "sim_world_model_contamination_index")
    p_confab = _mean(primary_results, "confabulation_rate")
    a_confab = _mean(ablation_results, "confabulation_rate")
    p_route = _mean(primary_results, "write_routing_separation")
    a_route = _mean(ablation_results, "write_routing_separation")
    return {
        "gate_on_mean_contamination": p_contam,
        "gate_off_mean_contamination": a_contam,
        "contamination_delta_off_minus_on": a_contam - p_contam,
        "gate_on_mean_confabulation": p_confab,
        "gate_off_mean_confabulation": a_confab,
        "confabulation_delta_off_minus_on": a_confab - p_confab,
        "gate_on_mean_routing_separation": p_route,
        "gate_off_mean_routing_separation": a_route,
        "routing_separation_delta_on_minus_off": p_route - a_route,
    }


# --- Driver ---------------------------------------------------------------
def run_experiment(dry_run: bool = False) -> tuple[list[dict], list[dict], dict, dict, float]:
    """Run both arms across the (matched, shared) seeds. Returns
    (primary_results, ablation_results, criteria, deltas, elapsed)."""
    if dry_run:
        seeds = (SEEDS[0],)
        n_train = 6           # short smoke path that still injects/gates/writes
    else:
        seeds = SEEDS
        n_train = None

    t0 = time.time()
    primary_results: list[dict] = []
    ablation_results: list[dict] = []
    for seed in seeds:
        print(f"Seed {seed} Condition {CONDITION_PRIMARY[0]}", flush=True)
        primary_results.append(
            run_condition(seed, CONDITION_PRIMARY[0], CONDITION_PRIMARY[1],
                          n_train_events=n_train))
        print(f"Seed {seed} Condition {CONDITION_ABLATION[0]}", flush=True)
        ablation_results.append(
            run_condition(seed, CONDITION_ABLATION[0], CONDITION_ABLATION[1],
                          n_train_events=n_train))
    elapsed = time.time() - t0

    criteria = _evaluate_criteria(primary_results, ablation_results)
    deltas = _pairwise_deltas(primary_results, ablation_results)
    return primary_results, ablation_results, criteria, deltas, elapsed


def main(dry_run: bool = False) -> dict:
    print(f"[{EXPERIMENT_TYPE}] MECH-094 simulation-vs-real write-gate "
          f"discriminative pair...", flush=True)
    (primary_results, ablation_results,
     criteria, deltas, elapsed) = run_experiment(dry_run=dry_run)

    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    direction = "supports" if criteria["overall_pass"] else "weakens"

    # One verdict line per seed x condition run.
    for label, rows in ((CONDITION_PRIMARY[0], primary_results),
                        (CONDITION_ABLATION[0], ablation_results)):
        for r in rows:
            v = "PASS" if (
                # informative per-run verdict: did this run behave as the arm
                # predicts? GATE_ON arm wants low contamination; GATE_OFF high.
                (r["sim_tag_passed"] and r["sim_world_model_contamination_index"]
                 < C1_GATE_ON_MAX_CONTAMINATION)
                or ((not r["sim_tag_passed"]) and r["sim_world_model_contamination_index"]
                    > C1_GATE_OFF_MIN_CONTAMINATION)
            ) else "FAIL"
            print(f"  {label} seed={r['seed']}  "
                  f"contam={r['sim_world_model_contamination_index']:.3f}  "
                  f"confab={r['confabulation_rate']:.3f}  "
                  f"routing_MI={r['write_routing_separation']:.3f}  "
                  f"verdict: {v}", flush=True)

    print(f"  C1 contamination: {criteria['c1_contamination_seeds_pass']}"
          f"/{criteria['n_seeds']} -> {'PASS' if criteria['c1_pass'] else 'FAIL'}",
          flush=True)
    print(f"  C2 confabulation: {criteria['c2_confabulation_seeds_pass']}"
          f"/{criteria['n_seeds']} -> {'PASS' if criteria['c2_pass'] else 'FAIL'}",
          flush=True)
    print(f"  C3 routing_sep:   {criteria['c3_routing_separation_seeds_pass']}"
          f"/{criteria['n_seeds']} -> {'PASS' if criteria['c3_pass'] else 'FAIL'}",
          flush=True)
    print(f"[{EXPERIMENT_TYPE}] overall: {outcome} ({elapsed:.1f}s)", flush=True)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    summary = (
        "MECH-094 simulation-vs-real write-gate discriminative pair. Scenario: "
        "N_REAL real-experience events (hypothesis_tag=False) and N_SIM "
        "simulation-mode events injected at disjoint z_world locations into the "
        "ResidueField world-model substrate, across matched shared seeds for "
        "both arms. PRIMARY arm (gate_on_mech094_intact) presents the "
        "hypothesis tag on simulation events so the MECH-094 categorical write "
        "gate blocks their substrate write; ABLATION arm (gate_off_tag_loss) "
        "strips the tag at the consumer so simulation content writes freely "
        "(the predicted tag-loss / confabulation pathology). Interpretation: "
        f"PASS={outcome == 'PASS'} -> evidence_direction={direction}. Under "
        "GATE_ON the simulation stream cannot corrupt the real-experience world "
        "model (contamination ~0, routing MI ~log2); under GATE_OFF it does "
        "(contamination ~1, routing MI ~0). PASS supports MECH-094 as an "
        "active, load-bearing write gate whose loss reproduces confabulation. "
        "FAIL with C1 failing -> gate broken / not wired (run /diagnose-errors). "
        f"Pairwise deltas: contamination(off-on)="
        f"{deltas['contamination_delta_off_minus_on']:.3f}, "
        f"confabulation(off-on)={deltas['confabulation_delta_off_minus_on']:.3f}, "
        f"routing_separation(on-off)="
        f"{deltas['routing_separation_delta_on_minus_off']:.3f}."
    )

    registered_thresholds = {
        "C1_GATE_ON_MAX_CONTAMINATION": C1_GATE_ON_MAX_CONTAMINATION,
        "C1_GATE_OFF_MIN_CONTAMINATION": C1_GATE_OFF_MIN_CONTAMINATION,
        "C1_MIN_ARM_DELTA": C1_MIN_ARM_DELTA,
        "C2_GATE_ON_MAX_CONFABULATION": C2_GATE_ON_MAX_CONFABULATION,
        "C2_GATE_OFF_MIN_CONFABULATION": C2_GATE_OFF_MIN_CONFABULATION,
        "C3_GATE_ON_MIN_SEPARATION": C3_GATE_ON_MIN_SEPARATION,
        "C3_GATE_OFF_MAX_SEPARATION": C3_GATE_OFF_MAX_SEPARATION,
        "PASS_FRACTION_REQUIRED": PASS_FRACTION_REQUIRED,
    }

    # Stable numeric metrics keys (means across seeds + per-criterion counts).
    metrics = {
        "gate_on_mean_contamination": deltas["gate_on_mean_contamination"],
        "gate_off_mean_contamination": deltas["gate_off_mean_contamination"],
        "contamination_delta_off_minus_on": deltas["contamination_delta_off_minus_on"],
        "gate_on_mean_confabulation": deltas["gate_on_mean_confabulation"],
        "gate_off_mean_confabulation": deltas["gate_off_mean_confabulation"],
        "confabulation_delta_off_minus_on": deltas["confabulation_delta_off_minus_on"],
        "gate_on_mean_routing_separation": deltas["gate_on_mean_routing_separation"],
        "gate_off_mean_routing_separation": deltas["gate_off_mean_routing_separation"],
        "routing_separation_delta_on_minus_off": deltas["routing_separation_delta_on_minus_off"],
        "c1_contamination_seeds_pass": criteria["c1_contamination_seeds_pass"],
        "c2_confabulation_seeds_pass": criteria["c2_confabulation_seeds_pass"],
        "c3_routing_separation_seeds_pass": criteria["c3_routing_separation_seeds_pass"],
        "n_seeds": criteria["n_seeds"],
        "min_seeds_required": criteria["min_seeds_required"],
    }

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
        "evidence_direction_per_claim": {"MECH-094": direction},
        "seed_policy": "matched_shared_seeds",
        "dispatch_mode": "discriminative_pair",
        "summary": summary,
        "registered_thresholds": registered_thresholds,
        "metrics": metrics,
        "criteria": criteria,
        "pairwise_deltas": deltas,
        "config": {
            "world_dim": WORLD_DIM,
            "num_basis_functions": NUM_BASIS_FUNCTIONS,
            "n_real": N_REAL,
            "n_sim": N_SIM,
            "harm_magnitude": HARM_MAGNITUDE,
            "seeds": list(SEEDS) if not dry_run else [SEEDS[0]],
            "conditions": [CONDITION_PRIMARY[0], CONDITION_ABLATION[0]],
        },
        "results_primary_gate_on": primary_results,
        "results_ablation_gate_off": ablation_results,
        "elapsed_seconds": elapsed,
        "dry_run": bool(dry_run),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Result written to: {out_path}", flush=True)

    return {
        "outcome": outcome,
        "all_pass": bool(criteria["overall_pass"]),
        "manifest_path": str(out_path),
        "run_id": run_id,
        "dry_run": bool(dry_run),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Short smoke path: one seed, few injection events, "
                             "still exercises init/inject/gate/write/manifest.")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    # Runner-conformance sentinel reached on every manifest-writing path
    # (the manifest is written in both real and dry-run modes). Crashes
    # propagate as a non-zero exit + missing sentinel -> runner classifies ERROR.
    emit_outcome(
        outcome=result["outcome"],
        manifest_path=result["manifest_path"],
        run_id=result["run_id"],
    )
    sys.exit(0 if result["all_pass"] else 1)
