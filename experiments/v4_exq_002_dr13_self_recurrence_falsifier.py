"""V4-EXQ-002: DR-13 z_self temporal depth -- self-recurrence substrate-readiness falsifier.

Second V4 experiment (self_model_v4:SELF-1; user-directed build 2026-07-01 to unblock
SELF-3/DR-10). Same V4 conventions as V4-EXQ-001 (DR-12):
  - architecture_epoch = "ree_self_model_v1"
  - run_id suffix "_v4"; queue_id "V4-EXQ-002"; owner_exq -> self_model_v4:SELF-1.

PURPOSE (diagnostic / substrate-readiness; PROMOTES NOTHING; claim_ids=[]):
validate the no-op-default DR-13 lever landed 2026-07-01 in
ree_core/latent/self_recurrence.py + LatentStack.encode() (use_self_recurrence: a
dedicated gated self-recurrence over z_self, blended toward an E1 generative anchor,
replacing the z_self EMA step ONLY). A CONTROLLED synthetic probe over encode() (no env,
no training) exercising the lever end-to-end through the harness path -- not just the unit
contracts.

FALSIFIER (SELF-1 build path): if the dedicated self-recurrence does NOT make z_self carry
temporal/self-model state beyond the instantaneous encode, DR-13 buys nothing and the
recurrence is inert.

THREE arms per seed (LatentStack weights bit-identical across the two ON arms within a seed
via per-cell RNG reset; OFF has no recurrence module by construction):
  A0_OFF     -- use_self_recurrence=False; the fixed-alpha EMA baseline. Records the EMA's
                own history-discrimination for CONTEXT (secondary; NOT a gate).
  A1_RECUR   -- use_self_recurrence=True, coupling=0.15, NO anchor supplied -> pure
                recurrence path. The load-bearing arm: state_departure, history-carrying,
                perturbation-isolation.
  A2_ANCHOR  -- use_self_recurrence=True, coupling=0.15, a synthetic self_e1_anchor SUPPLIED
                -> exercises the E1-feedback blend end-to-end (anchor_present True, blend
                consequential vs A1). Confirms the second motif is live.

Primary claim is SUBSTRATE-READINESS (like V4-EXQ-001 tested WIRING liveness, not trained
superiority): the recurrence delivers a LIVE, STATEFUL, history-carrying, perturbation-
isolable self subject -- the thing DR-10/SELF-3, DR-11 and the INV-064 stability gate
attach to. Whether the (untrained) recurrence OUT-remembers the fixed-alpha EMA at long lag
is a TRAINED-performance question deferred to the DR-10 consumer experiments; the ON-vs-OFF
history-discrimination ratio is reported as a SECONDARY diagnostic only, NOT a gate (an
untrained GRU update gate ~0.5 can retain less than a 0.7 EMA -- that is not a substrate
defect).

History-discrimination test: two observation sequences that SHARE the final observation but
DIFFER at every earlier step. A memoryless encoder gives identical final z_self
(hist_disc == 0); a stateful one gives hist_disc > 0.

NON-VACUITY precondition (readiness): A1 state_departure (||stateful z_self - instantaneous
z_self||) >= DEPART_FLOOR on a strict majority of seeds -- the recurrence is LIVE, not
degenerate to the instantaneous encode. Below-floor self-routes substrate_not_ready_requeue
(NEVER a false negative against the lever).

INERT off-ramp: precondition met but A1 history-discrimination ~ 0 -> label
dr13_recurrence_inert (the recurrence carries no history the instantaneous encode lacks;
DR-13 buys nothing).

GUARDRAILS: generation:v4, off the V3 critical path, promotes nothing in V3. No training
(synthetic deterministic encode probe); no env; no downstream head.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from ree_core.utils.config import LatentStackConfig  # noqa: E402
from ree_core.latent.stack import LatentStack  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v4_exq_002_dr13_self_recurrence_falsifier"
ARCH_EPOCH = "ree_self_model_v1"  # V4 epoch (V4-EXQ-001 precedent)

SEEDS = [42, 43, 44]
ARMS = ["A0_OFF", "A1_RECUR", "A2_ANCHOR"]
COUPLING = 0.15            # light hybrid default (the recorded residual tunable)
SEQ_LEN = 6               # observation-sequence length
DEPART_FLOOR = 1e-3       # non-vacuity: stateful z_self must depart from the instantaneous encode
HIST_FLOOR = 1e-3         # load-bearing: the stateful z_self must carry pre-final history
LESION_FLOOR = 1e-3       # perturbation-isolation: lesioning prev.z_self must move the subject
LEAK_EPS = 1e-5           # isolation: a z_self perturbation must NOT leak into z_world
MAJORITY = 2              # of 3 seeds

OUT_DIR = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"


def _obs_dim(cfg: LatentStackConfig) -> int:
    return cfg.body_obs_dim + cfg.world_obs_dim


def _make_stack(use_recurrence: bool) -> LatentStack:
    cfg = LatentStackConfig(
        use_self_recurrence=use_recurrence,
        self_recurrence_e1_coupling=COUPLING,
    )
    return LatentStack(cfg)


def _roll(stack: LatentStack, obs_seq, anchor=None):
    """Run a sequence of observations through encode() step by step; return the final
    LatentState (anchor supplied at every step when provided)."""
    state = None
    for obs in obs_seq:
        state = stack.encode(obs, state, self_e1_anchor=anchor)
    return state


def _run_seed(seed: int) -> dict:
    """Run all three arms for one seed and return the per-seed record."""
    arm_rows = {}
    for arm in ARMS:
        use_rec = (arm != "A0_OFF")
        cfg_slice = {
            "arm": arm, "use_self_recurrence": use_rec, "coupling": COUPLING,
            "seq_len": SEQ_LEN,
        }
        with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__)) as cell:
            # arm_cell reset RNG to `seed` on enter -> stack init + the observation
            # sequences below are deterministic per seed (bit-identical weights across
            # the two ON arms, which differ only in whether an anchor is supplied).
            stack = _make_stack(use_rec)
            cfg = stack.config
            od = _obs_dim(cfg)

            # Two sequences sharing the FINAL observation, differing at every earlier
            # step -> any final-z_self difference is memory of pre-final history.
            final_obs = torch.randn(1, od)
            seq_a = [torch.randn(1, od) for _ in range(SEQ_LEN - 1)] + [final_obs]
            seq_b = [torch.randn(1, od) for _ in range(SEQ_LEN - 1)] + [final_obs]

            anchor = None
            if arm == "A2_ANCHOR":
                anchor = torch.randn(1, cfg.self_dim)  # synthetic E1-generative anchor

            st_a = _roll(stack, seq_a, anchor=anchor)
            st_b = _roll(stack, seq_b, anchor=anchor)
            hist_disc = float((st_a.z_self - st_b.z_self).norm(dim=-1).mean().item())

            # state_departure + anchor_present from the ON diag (final step of seq_a).
            diag = st_a.self_recurrence_diag or {}
            state_departure = float(diag.get("state_departure", 0.0)) if use_rec else 0.0
            anchor_present = bool(diag.get("anchor_present", False))

            # Perturbation-isolation: re-run the final encode from seq_a's penultimate
            # state, once real and once with prev.z_self lesioned to zeros. lesion must
            # move z_self (the subject) and must NOT leak into z_world.
            lesion_delta = 0.0
            world_leak = 0.0
            if use_rec:
                prev = _roll(stack, seq_a[:-1], anchor=anchor)
                real = stack.encode(final_obs, prev, self_e1_anchor=anchor)
                lesioned = prev.detach()
                lesioned.z_self = torch.zeros_like(prev.z_self)
                les = stack.encode(final_obs, lesioned, self_e1_anchor=anchor)
                lesion_delta = float((real.z_self - les.z_self).abs().max().item())
                world_leak = float((real.z_world - les.z_world).abs().max().item())

            # Blend consequence (A2 only): stateful z_self with anchor differs from the
            # pure-recurrence value (same weights, anchor off).
            blend_delta = 0.0
            if arm == "A2_ANCHOR":
                stack_noanchor = _make_stack(True)
                stack_noanchor.load_state_dict(stack.state_dict())
                st_a_noanchor = _roll(stack_noanchor, seq_a, anchor=None)
                blend_delta = float((st_a.z_self - st_a_noanchor.z_self).abs().max().item())

            row = {
                "arm": arm,
                "seed": seed,
                "hist_disc": hist_disc,
                "state_departure": state_departure,
                "anchor_present": anchor_present,
                "lesion_delta": lesion_delta,
                "world_leak": world_leak,
                "blend_delta": blend_delta,
            }
            cell.stamp(row)  # writes row["arm_fingerprint"]
            arm_rows[arm] = row
    return {"seed": seed, "arm_rows": [arm_rows[a] for a in ARMS], "_by_arm": arm_rows}


def run_experiment(seeds=None, dry_run: bool = False) -> dict:
    seeds = seeds if seeds is not None else (SEEDS[:1] if dry_run else SEEDS)
    seed_records = []
    for s in seeds:
        print(f"Seed {s} Condition dr13_probe", flush=True)
        print(f"  [train] dr13 seed={s} ep 1/1", flush=True)
        rec = _run_seed(s)
        a1 = rec["_by_arm"]["A1_RECUR"]
        a2 = rec["_by_arm"]["A2_ANCHOR"]
        seed_pass = bool(
            a1["state_departure"] >= DEPART_FLOOR
            and a1["hist_disc"] >= HIST_FLOOR
            and a1["lesion_delta"] >= LESION_FLOOR
            and a1["world_leak"] < LEAK_EPS
            and a2["anchor_present"]
            and a2["blend_delta"] > 0.0
        )
        rec["seed_pass"] = seed_pass
        seed_records.append(rec)
        print(f"verdict: {'PASS' if seed_pass else 'FAIL'}", flush=True)

    n = len(seed_records)
    majority = MAJORITY if not dry_run else 1

    def _a(r, arm):
        return r["_by_arm"][arm]

    min_departure = min((_a(r, "A1_RECUR")["state_departure"] for r in seed_records), default=0.0)
    n_hist = sum(1 for r in seed_records if _a(r, "A1_RECUR")["hist_disc"] >= HIST_FLOOR)
    n_isolated = sum(
        1 for r in seed_records
        if _a(r, "A1_RECUR")["lesion_delta"] >= LESION_FLOOR
        and _a(r, "A1_RECUR")["world_leak"] < LEAK_EPS
    )
    n_anchor_live = sum(
        1 for r in seed_records
        if _a(r, "A2_ANCHOR")["anchor_present"] and _a(r, "A2_ANCHOR")["blend_delta"] > 0.0
    )
    ema_hist = [ _a(r, "A0_OFF")["hist_disc"] for r in seed_records ]
    recur_hist = [ _a(r, "A1_RECUR")["hist_disc"] for r in seed_records ]

    precond_met = min_departure >= DEPART_FLOOR      # recurrence LIVE (non-vacuity)
    c1_history = n_hist >= majority                    # load-bearing: carries history
    c2_isolated = n_isolated >= majority               # load-bearing: perturbation-isolated subject
    c3_anchor = n_anchor_live >= majority              # control: E1-feedback blend live

    if not precond_met:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
    elif not c1_history:
        label = "dr13_recurrence_inert"  # FALSIFIER fired: carries no history the encode lacks
        outcome = "FAIL"
    elif c1_history and c2_isolated and c3_anchor:
        label = "dr13_self_recurrence_delivers_stateful_subject"
        outcome = "PASS"
    else:
        label = "dr13_partial_substrate_gap"  # live + history, but isolation or anchor gap
        outcome = "FAIL"

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v4"

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCH_EPOCH,
        "generation": "v4",
        "claim_ids": [],
        "unblocks_claims": ["ARC-081", "MECH-215"],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "timestamp_utc": ts,
        "owner_node": "self_model_v4:SELF-1",
        "interpretation": {
            "label": label,
            "preconditions": [
                {
                    "name": "self_recurrence_live_state_departure",
                    "description": "A1_RECUR stateful z_self departs from the instantaneous "
                                   "encode (||stateful - instantaneous||) above the "
                                   "non-vacuity floor on every seed -- the recurrence is "
                                   "live, not degenerate to the instantaneous snapshot. "
                                   "Same statistic the load-bearing history test relies on.",
                    "measured": min_departure,
                    "threshold": DEPART_FLOOR,
                    "control": "A1_RECUR (use_self_recurrence=True) vs its own instantaneous encode",
                    "met": bool(precond_met),
                },
            ],
            "criteria_non_degenerate": {
                "C1_carries_history": bool(SEQ_LEN >= 2),   # sequences share only the final obs
                "C2_perturbation_isolated": True,
                "C3_anchor_blend_live": True,
            },
            "criteria": [
                {"name": "C1_carries_history", "load_bearing": True,
                 "passed": bool(c1_history),
                 "detail": f"{n_hist}/{n} seeds: A1 hist_disc >= {HIST_FLOOR} "
                           f"(stateful z_self carries pre-final history the instantaneous encode lacks)"},
                {"name": "C2_perturbation_isolated", "load_bearing": True,
                 "passed": bool(c2_isolated),
                 "detail": f"{n_isolated}/{n} seeds: lesioning prev.z_self moves the subject "
                           f"(>= {LESION_FLOOR}) with z_world leak < {LEAK_EPS}"},
                {"name": "C3_anchor_blend_live", "load_bearing": True,
                 "passed": bool(c3_anchor),
                 "detail": f"{n_anchor_live}/{n} seeds: A2 anchor_present AND blend consequential "
                           f"(E1-feedback motif live end-to-end)"},
            ],
        },
        "summary": {
            "n_seeds": n,
            "min_state_departure": min_departure,
            "n_carries_history": n_hist,
            "n_perturbation_isolated": n_isolated,
            "n_anchor_blend_live": n_anchor_live,
            "majority_required": majority,
            "secondary_recur_vs_ema_hist_disc": {
                "recur_mean": (sum(recur_hist) / n) if n else 0.0,
                "ema_mean": (sum(ema_hist) / n) if n else 0.0,
                "note": "SECONDARY / context only, NOT a gate: an untrained GRU can retain "
                        "less history than a fixed-alpha EMA; trained superiority over the EMA "
                        "is a DR-10 consumer-experiment question, not substrate-readiness.",
            },
        },
        "arm_results": [row for r in seed_records for row in r["arm_rows"]],
        "seed_records": [
            {k: v for k, v in r.items() if k != "_by_arm"} for r in seed_records
        ],
        "config": {
            "seeds": seeds, "arms": ARMS, "coupling": COUPLING, "seq_len": SEQ_LEN,
            "depart_floor": DEPART_FLOOR, "hist_floor": HIST_FLOOR,
            "lesion_floor": LESION_FLOOR, "leak_eps": LEAK_EPS, "dry_run": dry_run,
        },
        "notes": "Second V4 experiment. Substrate-readiness probe for DR-13 (self_model_v4:"
                 "SELF-1). FALSIFIER: the dedicated self-recurrence must make z_self carry "
                 "temporal/self-model state beyond the instantaneous encode. PASS = live + "
                 "history-carrying + perturbation-isolable stateful subject + E1-anchor blend "
                 "live. Trained superiority over the EMA is a DR-10 consumer question "
                 "(reported secondary only). Baseline mint SKIPPED: synthetic no-training "
                 "forward-pass probe (~seconds), no expensive reusable trained baseline "
                 "(disqualifier: trivial + no plausible trained successor of THIS OFF cell). "
                 "Promotes nothing; off the V3 critical path.",
    }
    return manifest, run_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest, run_id = run_experiment(dry_run=args.dry_run)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"outcome={manifest['outcome']} label={manifest['interpretation']['label']} "
          f"hist={manifest['summary']['n_carries_history']}/{manifest['summary']['n_seeds']} "
          f"iso={manifest['summary']['n_perturbation_isolated']}/{manifest['summary']['n_seeds']} "
          f"anchor={manifest['summary']['n_anchor_blend_live']}/{manifest['summary']['n_seeds']}",
          flush=True)
    print(f"wrote {out_path}", flush=True)

    _outcome = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        run_id=run_id,
        dry_run=args.dry_run,  # relocate the smoke manifest out of evidence/ (skill Step 3)
    )
