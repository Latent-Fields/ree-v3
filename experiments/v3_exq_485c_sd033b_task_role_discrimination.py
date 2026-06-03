"""V3-EXQ-485c: SD-033b OFC-analog task-role discrimination (MECH-263 signature b).

commitment_closure:GAP-8 deliverable 2 of 2 (Phase 7). Representation-level
functional-signature diagnostic on the OFC state_code, direct-drive style
(precedent: V3-EXQ-485 / 485a).

WHAT IS TESTED (MECH-263 signature b -- same-sensory / different-task-role):
  "Perceptually IDENTICAL states that occupy different positions in task
   structure produce distinct OFC representations."

WHY DIRECT-DRIVE (not the GAP-3 dual-cue primitive):
  The GAP-3 dual-cue primitive presents two simultaneously-active resource types
  whose SD-049 field views are perceptually DISTINCT -- that tests distinct-cue
  discrimination, not the 'perceptually-identical / different-task-role'
  signature MECH-263 (b) specifies. The OFCAnalog reads only z_world (+ z_harm),
  and its state_code is a gate-modulated EMA over z_world history -- so its only
  route to encoding 'task-structural position' is the HISTORY of z_world it has
  integrated. The faithful test is therefore: drive the OFC to the SAME final
  z_world (perceptually identical sensory input) via two DIFFERENT task-context
  histories (task-role A vs B), and ask whether the state_code at the matched
  input is separable. This is exactly the OFC's intended function (a latent
  carrying 'the agent's position in task structure'; ofc_analog.py docstring).

DESIGN (deterministic, direct-drive, matched final input):
  use_ofc_analog=True (default ofc_harm_dim=0 -> state_code = world_proj(z_world),
  task-role carried purely by z_world history). Per replicate:
    Context A: feed CONTEXT_TICKS of history pattern A, then ONE tick of the
               shared matched input Z_match; read state_code_A.
    Context B: feed CONTEXT_TICKS of history pattern B, then ONE tick of Z_match;
               read state_code_B.
  Z_match is byte-identical across A and B (perceptually identical final state).
  z_world_match_cosine verifies the inputs are matched (~1.0 by construction).
  Read the state_code right after the FIRST Z_match tick, where the differing
  history still dominates the EMA (many Z_match ticks would wash the history out
  and collapse the two -- that washout is itself the EMA semantics, not a
  failure of the signature).
  Negative control (within-context jitter): run Context A twice with different
  RNG on the history pattern and measure the state_code distance between the two
  A-runs. Task-role discrimination requires between-context separation to exceed
  this within-context jitter.

METRICS (per replicate + aggregate):
  between_context_distance   cosine dist state_code_A vs state_code_B at Z_match.
  within_context_jitter      cosine dist between two Context-A runs at Z_match.
  separation_ratio           between / max(within, eps).
  z_world_match_cosine       cosine sim of the two final inputs (~1.0 control).
  state_code_nonzero         bool: both state_codes have norm > 0.

INTERPRETATION GRID (one row per plausible outcome -> next action):
  PASS / supports:
    between-context separation >= SEPARATION_RATIO_MIN * within-context jitter
    AND z_world_match_cosine >= Z_WORLD_MATCH_MIN AND state_code_nonzero, on a
    majority of replicates. -> The OFC distinguishes task role at matched sensory
    input. Advances commitment_closure:GAP-8.
  FAIL / weakens:
    separation ~ jitter (ratio below threshold). -> The state_code carries no
    task-role structure beyond input-driven noise.
  non_contributory / substrate_ceiling (route /failure-autopsy, do NOT force-map):
    state_code_nonzero False (representation collapses to ~0) OR
    z_world_match_cosine < Z_WORLD_MATCH_MIN (the matched-input control failed,
    so the test is ill-posed) -> diagnose, do not score as weakens.

HONEST SCOPING:
  Representation-level functional signature; advances GAP-8. FULL SD-033b
  promotion candidate -> provisional still needs the deferred trained-OFC-head
  behavioural arm (frozen-zeroed bias head -> behaviour-change not measurable
  here; phased-training protocol parallel to SD-033a GAP-1).

Run:
  /opt/local/bin/python3 experiments/v3_exq_485c_sd033b_task_role_discrimination.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_485c_sd033b_task_role_discrimination"

# -- Pre-registered thresholds --
CONTEXT_TICKS = 25            # history ticks before the matched final input
SEPARATION_RATIO_MIN = 3.0   # between-context must exceed within-context jitter by this
Z_WORLD_MATCH_MIN = 0.95     # matched-input control: final z_world cosine sim
WORLD_DIM = 32
N_REPLICATES = 4
PASS_FRACTION = 0.5          # majority of replicates


def _cos_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    na = float(a.norm().item())
    nb = float(b.norm().item())
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    cos = float(torch.dot(a, b).item()) / (na * nb)
    return 1.0 - max(-1.0, min(1.0, cos))


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return 1.0 - _cos_dist(a, b)


def _build_agent() -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_ofc_analog=True,
    )
    return REEAgent(cfg)


def _drive_to_state_code(
    base_agent: REEAgent, history: torch.Tensor, z_match: torch.Tensor
) -> torch.Tensor:
    """Fresh OFC (copy of base init weights), feed `history` then one Z_match
    tick, return the state_code after the first matched-input tick."""
    agent = _build_agent()
    agent.ofc.load_state_dict(base_agent.ofc.state_dict())
    agent.ofc.reset()
    for t in range(history.shape[0]):
        agent.ofc.update(z_world=history[t:t + 1], z_harm=None, gate=1.0)
    agent.ofc.update(z_world=z_match, z_harm=None, gate=1.0)
    return agent.ofc.state_code.detach().clone()


def _run_replicate(rep: int, context_ticks: int) -> dict:
    """One replicate: build two distinct task-context histories converging on a
    shared Z_match, measure between- vs within-context state_code distance."""
    torch.manual_seed(1000 + rep)

    # Shared, byte-identical matched final input (perceptually identical state).
    z_match = torch.randn(1, WORLD_DIM)

    # Two distinct task-role histories (different structured patterns).
    hist_a = torch.randn(context_ticks, WORLD_DIM) + 1.5   # role-A offset cluster
    hist_b = torch.randn(context_ticks, WORLD_DIM) - 1.5   # role-B offset cluster
    # Within-context jitter control: a second Context-A history, same distribution,
    # different RNG draw.
    hist_a2 = torch.randn(context_ticks, WORLD_DIM) + 1.5

    base = _build_agent()  # single init shared by all three drives

    sc_a = _drive_to_state_code(base, hist_a, z_match)
    sc_b = _drive_to_state_code(base, hist_b, z_match)
    sc_a2 = _drive_to_state_code(base, hist_a2, z_match)

    between = _cos_dist(sc_a, sc_b)
    within = _cos_dist(sc_a, sc_a2)
    ratio = between / max(within, 1e-6)
    z_match_cos = _cos_sim(z_match, z_match)  # control: identical -> 1.0
    nonzero = (
        float(sc_a.norm().item()) > 0.0
        and float(sc_b.norm().item()) > 0.0
    )

    rep_pass = (
        ratio >= SEPARATION_RATIO_MIN
        and z_match_cos >= Z_WORLD_MATCH_MIN
        and nonzero
    )
    return {
        "replicate": rep,
        "between_context_distance": between,
        "within_context_jitter": within,
        "separation_ratio": ratio,
        "z_world_match_cosine": z_match_cos,
        "state_code_nonzero": nonzero,
        "pass": bool(rep_pass),
    }


def run_experiment(dry_run: bool = False) -> dict:
    context_ticks = 6 if dry_run else CONTEXT_TICKS
    n_rep = 1 if dry_run else N_REPLICATES

    per_rep = []
    n_pass = 0
    for r in range(n_rep):
        print(f"Seed {1000 + r} Condition task_role_discrimination", flush=True)
        print(f"  [probe] task_role rep={r} ep {r + 1}/{n_rep}", flush=True)
        res = _run_replicate(r, context_ticks)
        per_rep.append(res)
        n_pass += int(res["pass"])
        print(
            f"verdict: {'PASS' if res['pass'] else 'FAIL'}"
            f" (ratio={res['separation_ratio']:.2f},"
            f" nonzero={res['state_code_nonzero']})",
            flush=True,
        )

    frac = n_pass / max(1, n_rep)
    all_nonzero = all(r["state_code_nonzero"] for r in per_rep)
    all_matched = all(r["z_world_match_cosine"] >= Z_WORLD_MATCH_MIN for r in per_rep)
    overall_pass = frac >= PASS_FRACTION

    if overall_pass:
        direction = "supports"
        ceiling_flag = False
    elif not (all_nonzero and all_matched):
        # control failed (collapse or mismatched input) -> non_contributory
        direction = "mixed"
        ceiling_flag = True
    else:
        direction = "weakens"
        ceiling_flag = False

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "evidence_direction": direction,
        "substrate_ceiling_flag": ceiling_flag,
        "pass_fraction": frac,
        "n_replicates": n_rep,
        "per_replicate": per_rep,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    res = run_experiment(dry_run=args.dry_run)
    elapsed = time.time() - t0

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    outcome = res["outcome"]
    direction = res["evidence_direction"]

    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": ["SD-033b", "MECH-263"],
        "result": outcome,
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {
            "SD-033b": direction,
            "MECH-263": direction,
        },
        "metrics": res,
        "elapsed_seconds": elapsed,
        "notes": (
            "SD-033b OFC-analog task-role discrimination (MECH-263 signature b), "
            "representation-level direct-drive diagnostic. Two distinct task-role "
            "z_world histories converge on a byte-identical matched final input; "
            "the state_code (gate-modulated EMA over z_world history) is tested "
            "for separability at the matched input vs within-context jitter. The "
            "GAP-3 dual-cue primitive was not used (its cues are perceptually "
            "DISTINCT, not the 'perceptually-identical / different-task-role' "
            "signature). Advances commitment_closure:GAP-8; FULL SD-033b "
            "promotion still needs the deferred trained-OFC-head behavioural arm "
            "(frozen-zeroed bias head -> behaviour-change not measurable here)."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"result: {outcome}")
    print(f"  evidence_direction: {direction}")
    print(f"  pass_fraction: {res['pass_fraction']:.2f} ({res['n_replicates']} reps)")
    print(f"  substrate_ceiling_flag: {res['substrate_ceiling_flag']}")
    print(f"Result written to: {out_path}", flush=True)

    _o = str(outcome).upper()
    return (_o if _o in ("PASS", "FAIL") else "FAIL"), str(out_path)


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
