"""V3-EXQ-485b: SD-033b OFC-analog devaluation sensitivity (MECH-263 signature a).

commitment_closure:GAP-8 deliverable 1 of 2 (Phase 7). The GAP-3 env extensions
that the original plan named (landed 2026-05-17) are NOT used here -- see the
SUBSTRATE FINDING below for why -- so this is a representation-level functional-
signature diagnostic on the OFC state_code, in the same direct-drive style as
V3-EXQ-485 / 485a (both PASS substrate diagnostics that drove ofc.update()
directly with constructed inputs).

WHAT IS TESTED (MECH-263 signature a -- devaluation sensitivity):
  "If an outcome's value changes while the state-action -> outcome mapping does
   not, the OFC representation updates appropriately, within bounded ticks."

SUBSTRATE FINDING (why the env-based appetitive instrument was not used):
  OFCAnalog.update(z_world, z_harm, gate) reads ONLY z_world and (when
  harm_dim > 0) z_harm. It has NO appetitive value / drive / benefit input. The
  user-preferred instrument (SD-049 sensory-specific satiety, Critchley & Rolls
  1996) changes the agent's INTERNAL drive state (body / z_self), which the OFC
  never reads -- so appetitive satiety is invisible to the OFC state_code (with
  matched actions: zero signal; with free behaviour: a pure scene-change
  confound). The GAP-3 counter-evidence primitive was also ruled out: it mutates
  only the committed target's reward-validity and leaves grid / hazards /
  resources / agent position invariant (env contract C3), so it never reaches an
  obs-reading OFC either. The faithful realization given the OFC's ACTUAL inputs
  is an AVERSIVE outcome devaluation in the z_harm domain (threat removed at a
  fixed state): the value of the outcome changes, the state (z_world) is held
  fixed, and the change is genuinely visible to the OFC via outcome_proj(z_harm).
  This is a legitimate aversive devaluation (Rudebeck & Murray 2014; Dickinson &
  Balleine 1994 outcome revaluation).

DESIGN (deterministic, direct-drive, matched arms):
  use_ofc_analog=True, ofc_harm_dim>0 so state_code = world_proj(z_world)
  + outcome_pool_weight * outcome_proj(z_harm). Hold z_world fixed (the agent is
  "at" a state). For the first PRE_ONSET ticks both arms receive an identical
  HIGH-threat z_harm (the aversive outcome is present and stable -- state-action
  -> outcome mapping established). At onset:
    DEVALUE arm: z_harm -> LOW (threat removed; outcome devalued).
    CONTROL arm: z_harm stays HIGH (no change).
  Both arms keep z_world identical throughout, so any state_code divergence is
  attributable to the outcome (z_harm) change, NOT to a state/scene change.
  Measure cosine distance between the two arms' state_code at each post-onset
  tick. The OFC EMA time-constant is ~1/update_eta (~20 ticks at eta=0.05), so a
  devaluation-sensitive representation diverges and crosses the margin within a
  bounded number of ticks.

METRICS:
  pre_onset_max_distance        cosine dist DEVALUE-vs-CONTROL state_code, pre-onset
                                (must be ~0 -- the arms are identical until onset).
  post_onset_final_distance     distance at the last post-onset tick.
  ticks_to_divergence           first post-onset tick where distance > DIVERGENCE_MARGIN
                                (and stays above) -- onset latency; -1 if never.
  devaluation_engaged           bool: post_onset_final_distance > DIVERGENCE_MARGIN.

INTERPRETATION GRID (one row per plausible outcome -> next action):
  PASS / supports:
    post-onset divergence >> pre-onset baseline AND ticks_to_divergence within
    BOUNDED_TICKS, on a majority of seeds. -> The OFC representation IS
    devaluation-sensitive (aversive domain). Advances commitment_closure:GAP-8.
  FAIL / weakens:
    no divergence beyond baseline (devaluation_engaged False everywhere). -> The
    state_code is insensitive to the outcome-value change.
  non_contributory / substrate_ceiling (route /failure-autopsy, do NOT force-map):
    divergence present but NOT bounded (ticks_to_divergence > BOUNDED_TICKS on a
    majority) -- the representation updates but too slowly to count as the
    bounded-tick signature; this is an eta-calibration / substrate-timescale
    question, not a yes/no on sensitivity.

HONEST SCOPING:
  This delivers the MECH-263 representation-level functional signature and
  advances GAP-8, but FULL promotion of SD-033b candidate -> provisional still
  requires the deferred trained-OFC-head BEHAVIOURAL arm (the frozen-random,
  last-layer-zeroed bias head means behaviour-change is not measurable here;
  the phased-training protocol is the same deferred work as SD-033a GAP-1).

Run:
  /opt/local/bin/python3 experiments/v3_exq_485b_sd033b_devaluation_sensitivity.py [--dry-run]
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_485b_sd033b_devaluation_sensitivity"

# -- Pre-registered thresholds (NOT derived from the run's own statistics) --
PRE_ONSET_TICKS = 25          # ticks of stable HIGH-threat before devaluation
POST_ONSET_TICKS = 40         # ticks observed after the outcome change
DIVERGENCE_MARGIN = 0.02      # cosine-distance floor counting as "diverged"
BOUNDED_TICKS = 20            # onset must occur within this many post-onset ticks
HARM_DIM = 32                 # z_harm dim fed to OFC outcome_proj
SEEDS = (0, 1, 2)
PASS_FRACTION = 0.5           # majority of seeds


def _cos_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    na = float(a.norm().item())
    nb = float(b.norm().item())
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    cos = float(torch.dot(a, b).item()) / (na * nb)
    return 1.0 - max(-1.0, min(1.0, cos))


def _build_agent() -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_ofc_analog=True,
        ofc_harm_dim=HARM_DIM,
    )
    return REEAgent(cfg)


def _run_seed(seed: int, pre_ticks: int, post_ticks: int) -> dict:
    """One seed: drive two arms (DEVALUE / CONTROL) with matched z_world and a
    matched HIGH-threat z_harm until onset, then drop z_harm in DEVALUE only."""
    torch.manual_seed(seed)
    world_dim = 32

    # Fixed state (z_world) and fixed outcome latents, shared across arms.
    z_world = torch.randn(1, world_dim)
    z_harm_high = torch.randn(1, HARM_DIM) * 1.0          # threat present
    z_harm_low = torch.zeros(1, HARM_DIM)                  # threat removed (devalued)

    dev = _build_agent()
    ctl = _build_agent()
    # Identical initial OFC parameters across arms (same seed -> same init) so any
    # divergence is purely from the differing z_harm input, not weight init.
    ctl.ofc.load_state_dict(dev.ofc.state_dict())
    dev.ofc.reset()
    ctl.ofc.reset()

    pre_dists = []
    post_dists = []
    ticks_to_div = -1

    # Pre-onset: both arms see HIGH threat.
    for _ in range(pre_ticks):
        dev.ofc.update(z_world=z_world, z_harm=z_harm_high, gate=1.0)
        ctl.ofc.update(z_world=z_world, z_harm=z_harm_high, gate=1.0)
        pre_dists.append(_cos_dist(dev.ofc.state_code, ctl.ofc.state_code))

    # Onset + post-onset: DEVALUE drops threat, CONTROL holds it.
    above_run = 0
    for t in range(post_ticks):
        dev.ofc.update(z_world=z_world, z_harm=z_harm_low, gate=1.0)
        ctl.ofc.update(z_world=z_world, z_harm=z_harm_high, gate=1.0)
        d = _cos_dist(dev.ofc.state_code, ctl.ofc.state_code)
        post_dists.append(d)
        if d > DIVERGENCE_MARGIN:
            above_run += 1
            if above_run >= 2 and ticks_to_div < 0:
                ticks_to_div = t  # first tick of a sustained crossing
        else:
            above_run = 0

    pre_max = max(pre_dists) if pre_dists else 0.0
    post_final = post_dists[-1] if post_dists else 0.0
    engaged = post_final > DIVERGENCE_MARGIN
    bounded = 0 <= ticks_to_div <= BOUNDED_TICKS
    seed_pass = engaged and bounded
    return {
        "seed": seed,
        "pre_onset_max_distance": pre_max,
        "post_onset_final_distance": post_final,
        "ticks_to_divergence": ticks_to_div,
        "devaluation_engaged": engaged,
        "onset_within_bounded_ticks": bounded,
        "pass": bool(seed_pass),
    }


def run_experiment(dry_run: bool = False) -> dict:
    pre_ticks = 6 if dry_run else PRE_ONSET_TICKS
    post_ticks = 12 if dry_run else POST_ONSET_TICKS
    seeds = (0,) if dry_run else SEEDS

    per_seed = []
    n_pass = 0
    for i, s in enumerate(seeds):
        print(f"Seed {s} Condition devaluation_sensitivity", flush=True)
        print(f"  [probe] devaluation seed={s} ep {i + 1}/{len(seeds)}", flush=True)
        r = _run_seed(s, pre_ticks, post_ticks)
        per_seed.append(r)
        n_pass += int(r["pass"])
        print(
            f"verdict: {'PASS' if r['pass'] else 'FAIL'}"
            f" (engaged={r['devaluation_engaged']},"
            f" ticks_to_div={r['ticks_to_divergence']})",
            flush=True,
        )

    frac = n_pass / max(1, len(seeds))
    any_engaged = any(r["devaluation_engaged"] for r in per_seed)
    # Substrate-ceiling branch: representation updates (engaged) but not within
    # bounded ticks on a majority -> non_contributory, route /failure-autopsy.
    bounded_majority = (
        sum(int(r["onset_within_bounded_ticks"]) for r in per_seed)
        / max(1, len(seeds))
    ) >= PASS_FRACTION
    overall_pass = frac >= PASS_FRACTION

    if overall_pass:
        direction = "supports"
        ceiling_flag = False
    elif any_engaged and not bounded_majority:
        # diverges but too slowly -> not a clean weakens; flag for autopsy.
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
        "n_seeds": len(seeds),
        "per_seed": per_seed,
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
            "SD-033b OFC-analog devaluation sensitivity (MECH-263 signature a), "
            "representation-level direct-drive diagnostic. AVERSIVE devaluation "
            "(z_harm dropped at a fixed z_world state) used because the OFC reads "
            "only z_world + z_harm and has no appetitive value/drive input -- "
            "SD-049 satiety and the GAP-3 counter-evidence primitive are both "
            "invisible to the OFC state_code (substrate finding documented in the "
            "module docstring). Advances commitment_closure:GAP-8; FULL SD-033b "
            "promotion still needs the deferred trained-OFC-head behavioural arm "
            "(frozen-zeroed bias head -> behaviour-change not measurable here)."
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

    print(f"result: {outcome}")
    print(f"  evidence_direction: {direction}")
    print(f"  pass_fraction: {res['pass_fraction']:.2f} ({res['n_seeds']} seeds)")
    print(f"  substrate_ceiling_flag: {res['substrate_ceiling_flag']}")
    print(f"Result written to: {out_path}", flush=True)

    _o = str(outcome).upper()
    return (_o if _o in ("PASS", "FAIL") else "FAIL"), str(out_path)


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
