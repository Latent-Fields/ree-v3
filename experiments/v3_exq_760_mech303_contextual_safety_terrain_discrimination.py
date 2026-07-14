#!/opt/local/bin/python3
"""V3-EXQ-760 -- MECH-303 contextual passive safety terrain DISCRIMINATION (representation-level).

Claim: MECH-303 (safety_prediction.contextual_passive_substrate)
Status: candidate (v3_pending=false). Substrate IMPLEMENTED 2026-05-04 as SD-052.
Purpose: evidence (confirming experiment). Single claim -> no per-claim direction needed.

Why this experiment exists
--------------------------
V3-EXQ-520 was the SUBSTRATE-READINESS diagnostic for SD-052 and PASSed (2026-05-04).
It verified the accumulate_safety/evaluate_safety mechanics and the agent wiring, but
EXPLICITLY DEFERRED the downstream contextual-safety DISCRIMINATION test ("tested later
via discriminative pair gated on EXQ-520 PASS"). This is that discriminative test, at
the REPRESENTATION level. It was referenced in the behavioural-avoidance autopsies
(603h / 569c / 603i) but never tested as a field readout.

WALL-INDEPENDENT by construction
--------------------------------
The V3 program is bottlenecked on the competence wall (behavioral_diversity_isolation:GAP-I;
V3-EXQ-752..756 in flight). Any committed-behaviour DV is wall-bound. This experiment's DV
is a PASSIVE FIELD READOUT -- ResidueField.evaluate_safety(z_world) -- with NO planning,
NO action-commitment, NO goal-directed policy. Trajectories are pure random walks used only
to expose z_world context geometry; the safety terrain is a non-parametric RBF accumulator.
So the verdict is independent of the competence wall (precedent: functional-signature DVs
passed in V3-EXQ-455/447/448 while the behavioural baseline was monostrategy-locked;
failure_autopsy 455a).

No training occurs
------------------
The encoder is an UNTRAINED, frozen REEAgent used purely as a deterministic fixed z_world
feature map (torch.no_grad forward passes). The safety terrain is non-parametric
(RBFLayer.add_residue -- no gradients). Therefore:
  * Phased-training protocol is N/A (no head is trained on z_world / z_harm / any encoder
    output).
  * alpha_world (a training-loss weight) does NOT affect a no-grad forward pass, so its
    default value is irrelevant here. The readiness gate below guards the only way an
    untrained encoder could invalidate the test (context collapse in z_world).

Design
------
Contexts (per seed): K_SAFE distinct SAFE grid layouts (num_hazards=0 -> harm proximity ~0
everywhere) and K_UNSAFE distinct UNSAFE grid layouts (num_hazards high -> frequent harm
proximity). MECH-303 is CONTEXT-bound ("this environment predicts absence of suffering"),
so the discrimination unit is the whole environment, not a cue/cell (that is MECH-304).

Two arms (grid = seed x arm), sharing the SAME z_world trajectories per seed (deterministic),
differing ONLY in the accumulation gate:
  ARM gated_mech303 : accumulate_safety(z_world) ONLY on harm-ABSENT steps
                      (ground-truth env hazard_field_view proximity < quiescent threshold;
                      exactly 0 in num_hazards=0 safe contexts). This IS the MECH-303
                      mechanism: sustained NON-OCCURRENCE of harm builds the store.
  ARM ungated_control: accumulate_safety(z_world) on EVERY visited step (a pure
                      visitation/familiarity field). With balanced exposure across safe and
                      unsafe contexts, this field CANNOT discriminate safe from unsafe --
                      it isolates the harm-absence gate as the source of any discrimination,
                      removing the "it is just density" triviality.

Then, per context, a HELD-OUT random walk (different action seed, same layout) provides
query points never used for accumulation -> tests RBF generalisation within-context.

DV / metrics
------------
  auc_gated   = ROC-AUC of evaluate_safety(held-out z_world) predicting safe-context (1)
                vs unsafe-context (0), pooled across contexts.
  auc_ungated = same for the ungated control arm.
  margin      = auc_gated - auc_ungated (the mechanism-attributable discrimination).
  readout_range = spread of evaluate_safety over held-out points (non-vacuity).

Readiness preconditions (self-route, NOT a weakens -- non_contributory)
-----------------------------------------------------------------------
Same-statistic positive controls, measured before trusting the load-bearing AUC:
  P1 safety_field_responds_to_accumulation: AUC(evaluate_safety at accumulated safe points
     vs never-touched random z_world) >= AUC_RESPONSE_FLOOR. Positive control -- high by
     construction if the RBF reads back accumulated mass. Below floor => RBF misconfigured
     / degenerate.
  P2 contexts_separable_in_zworld: in-sample linear separability (centroid-projection AUC)
     of safe vs unsafe held-out z_world >= INPUT_SEP_FLOOR. If contexts collapse in z_world
     (encoder pathology), the field mechanism CANNOT be tested for reasons unrelated to
     MECH-303.
If either precondition is unmet -> self-route substrate_not_ready_requeue (outcome FAIL,
evidence_direction non_contributory). Explicitly arms the substrate
(safety_terrain_enabled=True; num_basis_functions sized to avoid ring-buffer eviction) --
lesson from V3-EXQ-688's vacuous null: do NOT rely on default flag paths.

Pre-registered PASS (load-bearing)
----------------------------------
  C1 (load_bearing): mean auc_gated >= AUC_PASS
  C2 (load_bearing): mean margin >= MARGIN_ABS AND mean margin - sd(margin) > 0
     (effect-size gate: scale on SD of the DELTA + absolute floor)
  non-vacuity: mean readout_range_gated > RANGE_FLOOR
PASS = C1 and C2 and non-vacuity, with preconditions met => supports MECH-303.
Preconditions met but criteria not met => weakens (real null: harm-absence accumulation
failed to produce a context-discriminating field despite separable inputs).

experiment_purpose = "evidence"
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig, ResidueConfig  # noqa: E402
from ree_core.residue.field import ResidueField  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_760_mech303_contextual_safety_terrain_discrimination"
CLAIM_IDS = ["MECH-303"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# ---- Environment / context structure ----
GRID_SIZE = 7
NUM_HAZARDS_SAFE = 0          # SAFE context: hazard proximity exactly 0 everywhere
NUM_HAZARDS_UNSAFE = 8        # UNSAFE context: hazard proximity uniformly elevated
NUM_RESOURCES = 2
K_SAFE = 3                    # distinct SAFE layouts per seed
K_UNSAFE = 3                  # distinct UNSAFE layouts per seed
ACCUM_STEPS = 120            # accumulation-walk steps per context
HELDOUT_STEPS = 60          # held-out query-walk steps per context

# ---- Safety terrain / field ----
WORLD_DIM = 32
NUM_BASIS = 1024             # > max accumulations (ungated: K*ACCUM = 720) -> no ring-buffer eviction
KERNEL_BANDWIDTH = 1.0
SAFETY_MAGNITUDE = 0.01      # per-step increment (substrate default; "diffuse/passive")
# harm-ABSENT iff ground-truth env hazard proximity (hazard_field_view abs-sum) < this.
# hazard_field_view is EXACTLY 0 with num_hazards=0 (safe) and ~10-21 with dense hazards
# (unsafe), so any threshold in (0, 10) cleanly separates the two context classes. Gating on
# the ENV ground-truth (not the untrained agent's z_harm_a) isolates the ResidueField
# mechanism from harm-encoder fidelity (a separate claim, SD-011).
QUIESCENT_THRESHOLD = 0.5

# ---- Pre-registered thresholds ----
AUC_RESPONSE_FLOOR = 0.90    # readiness P1 (positive control): field reads back accumulated mass
INPUT_SEP_FLOOR = 0.70       # readiness P2: safe/unsafe contexts separable in z_world
AUC_PASS = 0.75              # C1 load-bearing: gated arm discriminates
MARGIN_ABS = 0.15            # C2 load-bearing: gate-attributable discrimination
RANGE_FLOOR = 1e-3           # non-vacuity: field readout not flat

SEEDS: Tuple[int, ...] = (42, 43, 44)
ARMS = ("gated_mech303", "ungated_control")


# ------------------------------------------------------------------ helpers

def roc_auc(scores: List[float], labels: List[int]) -> float:
    """ROC-AUC of scores predicting label==1 (tie-aware, O(n_pos*n_neg))."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def build_agent(env: CausalGridWorldV2) -> REEAgent:
    """Untrained frozen REEAgent used only as a deterministic z_world feature map."""
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        world_dim=WORLD_DIM,
    )
    agent = REEAgent(config)
    agent.eval()
    return agent


def make_env(seed: int, num_hazards: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=num_hazards,
        num_resources=NUM_RESOURCES,
        seed=seed,
        use_proxy_fields=True,
    )


def hazard_proximity(obs_dict: Dict) -> float:
    """Ground-truth env hazard proximity: abs-sum of hazard_field_view.

    Exactly 0.0 when num_hazards=0 (safe context); uniformly elevated when hazards
    are present. Independent of the (untrained) agent's harm encoder.
    """
    h = obs_dict.get("hazard_field_view")
    if h is None:
        return 0.0
    return float(torch.as_tensor(h).float().flatten().abs().sum())


def walk_collect(
    env: CausalGridWorldV2,
    agent: REEAgent,
    n_steps: int,
    action_seed: int,
) -> List[Tuple[torch.Tensor, float]]:
    """Random-walk (no planning); return list of (z_world[1,world_dim], harm_norm).

    Resets the shared agent's recurrent latent state at walk start (and on any
    mid-walk episode end) so each walk is a fresh trajectory through the SAME
    frozen encoder -- no cross-walk / cross-context state leakage. agent.reset()
    does NOT touch residue (and we use an external field anyway).
    """
    gen = torch.Generator().manual_seed(action_seed)
    out: List[Tuple[torch.Tensor, float]] = []
    agent.reset()
    _, obs_dict = env.reset()
    with torch.no_grad():
        for _ in range(n_steps):
            hn = hazard_proximity(obs_dict)
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            z = latent.z_world.detach().clone()
            out.append((z, hn))
            action = int(torch.randint(0, env.action_dim, (1,), generator=gen).item())
            _, _, done, _, obs_dict = env.step(torch.tensor(action))
            if done:
                agent.reset()
                _, obs_dict = env.reset()
    return out


def run_cell(seed: int, arm: str, n_ctx_safe: int, n_ctx_unsafe: int,
             accum_steps: int, heldout_steps: int) -> Dict:
    """One (seed, arm) cell. RNG already reset by arm_cell on enter."""
    gated = (arm == "gated_mech303")

    field = ResidueField(ResidueConfig(
        world_dim=WORLD_DIM,
        num_basis_functions=NUM_BASIS,
        kernel_bandwidth=KERNEL_BANDWIDTH,
        safety_terrain_enabled=True,   # explicitly arm the substrate
    ))
    field.eval()

    # ONE shared frozen encoder across ALL contexts, so z_world is a consistent
    # feature map and safe/unsafe context structure is preserved. obs dims are
    # num_hazards-independent (hazard_field_view is a fixed 5x5 regardless), so a
    # reference env's dims apply to every context (asserted per context below).
    ref_env = make_env(seed * 1000 + 999, NUM_HAZARDS_SAFE)
    ref_dims = (ref_env.body_obs_dim, ref_env.world_obs_dim, ref_env.action_dim)
    agent = build_agent(ref_env)

    contexts: List[Tuple[int, int]] = (
        [(NUM_HAZARDS_SAFE, 1)] * n_ctx_safe + [(NUM_HAZARDS_UNSAFE, 0)] * n_ctx_unsafe
    )
    total_accum_steps = len(contexts) * accum_steps
    step_counter = 0

    accum_safe_points: List[torch.Tensor] = []   # z_world where safety was accumulated in SAFE contexts
    gate_pass_frac: List[float] = []
    heldout: List[Tuple[torch.Tensor, int]] = []  # (z_world, safe_label)

    for ci, (num_hazards, safe_label) in enumerate(contexts):
        ctx_seed = seed * 1000 + ci
        env = make_env(ctx_seed, num_hazards)
        assert (env.body_obs_dim, env.world_obs_dim, env.action_dim) == ref_dims, (
            "context obs dims must match the shared encoder"
        )

        # --- accumulation walk ---
        accum = walk_collect(env, agent, accum_steps, action_seed=ctx_seed + 1)
        n_pass = 0
        for z, hn in accum:
            step_counter += 1
            harm_absent = hn < QUIESCENT_THRESHOLD
            do_accum = harm_absent if gated else True
            if do_accum:
                field.accumulate_safety(z, safety_magnitude=SAFETY_MAGNITUDE)
                n_pass += 1
                if safe_label == 1 and harm_absent:
                    accum_safe_points.append(z)
            if step_counter % 50 == 0:
                print(f"  [accum] seed={seed} arm={arm} ep {step_counter}/{total_accum_steps}",
                      flush=True)
        gate_pass_frac.append(n_pass / max(1, len(accum)))

        # --- held-out query walk (never used for accumulation) ---
        held = walk_collect(env, agent, heldout_steps, action_seed=ctx_seed + 7777)
        for z, _hn in held:
            heldout.append((z, safe_label))

    # --- evaluate field over held-out points ---
    with torch.no_grad():
        held_z = torch.cat([z for z, _ in heldout], dim=0)          # [N, world_dim]
        held_labels = [lbl for _, lbl in heldout]
        held_scores = field.evaluate_safety(held_z).tolist()        # [N]

        auc_ctx = roc_auc(held_scores, held_labels)
        readout_range = float(max(held_scores) - min(held_scores)) if held_scores else 0.0

        # readiness P1 (positive control): accumulated-safe vs never-touched random z
        if accum_safe_points:
            acc_z = torch.cat(accum_safe_points, dim=0)
            rand_z = torch.randn(acc_z.shape[0], WORLD_DIM)
            resp_scores = (field.evaluate_safety(acc_z).tolist()
                           + field.evaluate_safety(rand_z).tolist())
            resp_labels = [1] * acc_z.shape[0] + [0] * rand_z.shape[0]
            auc_response = roc_auc(resp_scores, resp_labels)
        else:
            auc_response = float("nan")

        # readiness P2: in-sample linear separability of safe/unsafe in z_world
        safe_z = held_z[[i for i, l in enumerate(held_labels) if l == 1]]
        unsafe_z = held_z[[i for i, l in enumerate(held_labels) if l == 0]]
        if len(safe_z) and len(unsafe_z):
            direction = safe_z.mean(0) - unsafe_z.mean(0)            # [world_dim]
            proj = (held_z * direction.unsqueeze(0)).sum(-1).tolist()
            input_sep = roc_auc(proj, held_labels)
        else:
            input_sep = float("nan")

    total_safety = float(field.total_safety)
    active_centers = int(field.safety_terrain_rbf_field.active_mask.sum().item())

    verdict = "PASS" if (auc_ctx >= AUC_PASS) else "FAIL"
    print(f"verdict: {verdict}")

    return {
        "arm": arm,
        "seed": seed,
        "auc_context_discrimination": float(auc_ctx),
        "readout_range": readout_range,
        "auc_response_positive_control": float(auc_response),
        "input_separability_zworld": float(input_sep),
        "total_safety": total_safety,
        "active_centers": active_centers,
        "num_basis_functions": NUM_BASIS,
        "mean_gate_pass_frac": float(sum(gate_pass_frac) / max(1, len(gate_pass_frac))),
        "gate_pass_frac_per_context": [float(x) for x in gate_pass_frac],
        "n_heldout_points": len(heldout),
        "n_accum_safe_points": len(accum_safe_points),
        # recorded generously (recording standard): the raw per-point field readouts
        # + labels feed the pooled non-degeneracy check and post-hoc reanalysis.
        "heldout_safety_scores": [round(float(s), 6) for s in held_scores],
        "heldout_safe_labels": [int(l) for l in held_labels],
    }


def evaluate(rows: List[Dict]) -> Dict:
    """Aggregate per-seed, apply readiness gate then load-bearing criteria."""
    def per_seed(arm: str, field: str) -> List[float]:
        return [r[field] for r in rows if r["arm"] == arm]

    seeds_sorted = sorted({r["seed"] for r in rows})
    g_auc = per_seed("gated_mech303", "auc_context_discrimination")
    u_auc = per_seed("ungated_control", "auc_context_discrimination")
    g_range = per_seed("gated_mech303", "readout_range")
    g_resp = per_seed("gated_mech303", "auc_response_positive_control")
    g_sep = per_seed("gated_mech303", "input_separability_zworld")
    g_total = per_seed("gated_mech303", "total_safety")

    # align by seed order (per_seed preserves row order == seed x arm loop order)
    margins = [a - b for a, b in zip(g_auc, u_auc)]

    def mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    def sd(xs: List[float]) -> float:
        if len(xs) < 2:
            return 0.0
        m = mean(xs)
        return float((sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5)

    mean_auc_gated = mean(g_auc)
    mean_auc_ungated = mean(u_auc)
    mean_margin = mean(margins)
    sd_margin = sd(margins)
    mean_range = mean(g_range)
    mean_resp = mean(g_resp)
    mean_sep = mean(g_sep)

    # non-degeneracy net (backstop; feeds indexer scoring exclusion). Pool the raw
    # per-point gated field readouts across seeds: a working field spans 0..high
    # (real spread -> non-degenerate); a flat field collapses to ~0 (zero spread /
    # floor-pinned -> degenerate). This is semantically the "is the field flat?"
    # check and does not misfire on low seed count (unlike per-seed range spread).
    pooled_gated_scores: List[float] = []
    for r in rows:
        if r["arm"] == "gated_mech303":
            pooled_gated_scores.extend(r.get("heldout_safety_scores", []))
    degen = check_degeneracy({
        "gated_safety_readout": {"values": pooled_gated_scores, "floor": RANGE_FLOOR},
    })

    # readiness gate (self-route non_contributory, not a weakens)
    preconditions = None
    substrate_not_ready = False
    not_ready_reason = ""
    try:
        preconditions = p0_readiness_gate([
            {"name": "safety_field_responds_to_accumulation",
             "measured": mean_resp, "threshold": AUC_RESPONSE_FLOOR, "direction": "lower"},
            {"name": "contexts_separable_in_zworld",
             "measured": mean_sep, "threshold": INPUT_SEP_FLOOR, "direction": "lower"},
        ])
    except P0NotReady as exc:
        substrate_not_ready = True
        not_ready_reason = exc.reason
        preconditions = exc.preconditions

    # load-bearing criteria
    c1 = mean_auc_gated >= AUC_PASS
    c2 = (mean_margin >= MARGIN_ABS) and (mean_margin - sd_margin > 0.0)
    non_vacuous = mean_range > RANGE_FLOOR

    if substrate_not_ready or not degen["non_degenerate"]:
        outcome = "FAIL"
        direction = "non_contributory"
        label = "substrate_not_ready_requeue"
    elif c1 and c2 and non_vacuous:
        outcome = "PASS"
        direction = "supports"
        label = "context_specific_safety_terrain_confirmed"
    else:
        outcome = "FAIL"
        direction = "weakens"
        label = "no_context_discrimination_despite_separable_inputs"

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria": [
                {"name": "C1_auc_gated_above_pass", "load_bearing": True, "passed": bool(c1)},
                {"name": "C2_margin_gate_attributable", "load_bearing": True, "passed": bool(c2)},
                {"name": "non_vacuity_field_not_flat", "load_bearing": False, "passed": bool(non_vacuous)},
            ],
            "criteria_non_degenerate": {
                "C1_auc_gated_above_pass": bool(sd(g_auc) > 0 or mean_auc_gated != 0.5),
                "C2_margin_gate_attributable": bool(sd(margins) >= 0.0 and mean_range > RANGE_FLOOR),
            },
        },
        "aggregates": {
            "seeds": seeds_sorted,
            "mean_auc_gated": mean_auc_gated,
            "mean_auc_ungated": mean_auc_ungated,
            "mean_margin": mean_margin,
            "sd_margin": sd_margin,
            "per_seed_auc_gated": g_auc,
            "per_seed_auc_ungated": u_auc,
            "per_seed_margin": margins,
            "mean_readout_range_gated": mean_range,
            "mean_auc_response_positive_control": mean_resp,
            "mean_input_separability_zworld": mean_sep,
            "c1_auc_gated_above_pass": bool(c1),
            "c2_margin_gate_attributable": bool(c2),
            "non_vacuity_field_not_flat": bool(non_vacuous),
        },
        "thresholds": {
            "AUC_RESPONSE_FLOOR": AUC_RESPONSE_FLOOR,
            "INPUT_SEP_FLOOR": INPUT_SEP_FLOOR,
            "AUC_PASS": AUC_PASS,
            "MARGIN_ABS": MARGIN_ABS,
            "RANGE_FLOOR": RANGE_FLOOR,
        },
        "non_degenerate": degen["non_degenerate"],
        "degeneracy_reason": degen["degeneracy_reason"],
    }


def main(dry_run: bool = False) -> int:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})")
    t0 = time.time()
    t0_perf = time.perf_counter()   # for stamp_recording_core elapsed_seconds

    seeds = (SEEDS[0],) if dry_run else SEEDS
    k_safe = 1 if dry_run else K_SAFE
    k_unsafe = 1 if dry_run else K_UNSAFE
    accum_steps = 20 if dry_run else ACCUM_STEPS
    heldout_steps = 10 if dry_run else HELDOUT_STEPS

    full_config = {
        "grid_size": GRID_SIZE,
        "num_hazards_safe": NUM_HAZARDS_SAFE,
        "num_hazards_unsafe": NUM_HAZARDS_UNSAFE,
        "num_resources": NUM_RESOURCES,
        "k_safe": k_safe,
        "k_unsafe": k_unsafe,
        "accum_steps": accum_steps,
        "heldout_steps": heldout_steps,
        "world_dim": WORLD_DIM,
        "num_basis_functions": NUM_BASIS,
        "kernel_bandwidth": KERNEL_BANDWIDTH,
        "safety_magnitude": SAFETY_MAGNITUDE,
        "quiescent_threshold": QUIESCENT_THRESHOLD,
        "seeds": list(seeds),
        "arms": list(ARMS),
    }

    rows: List[Dict] = []
    for seed in seeds:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}")
            config_slice = dict(full_config)
            config_slice["arm"] = arm
            config_slice["seed"] = seed
            with arm_cell(
                seed,
                config_slice=config_slice,
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=False,
            ) as cell:
                row = run_cell(seed, arm, k_safe, k_unsafe, accum_steps, heldout_steps)
                cell.stamp(row)
            rows.append(row)
            print(
                f"  seed={seed} {arm:<16} auc={row['auc_context_discrimination']:.3f} "
                f"range={row['readout_range']:.4f} resp={row['auc_response_positive_control']:.3f} "
                f"sep={row['input_separability_zworld']:.3f} gatefrac={row['mean_gate_pass_frac']:.3f} "
                f"centers={row['active_centers']}"
            )

    ev = evaluate(rows)
    outcome = ev["outcome"]
    elapsed = time.time() - t0

    print(f"[{EXPERIMENT_TYPE}] label={ev['interpretation']['label']} "
          f"direction={ev['evidence_direction']}")
    agg = ev["aggregates"]
    print(f"  mean_auc_gated={agg['mean_auc_gated']:.3f} "
          f"mean_auc_ungated={agg['mean_auc_ungated']:.3f} "
          f"mean_margin={agg['mean_margin']:.3f} sd_margin={agg['sd_margin']:.3f}")
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s")

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "started_utc": datetime.utcfromtimestamp(t0).isoformat() + "Z",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "evidence_direction": ev["evidence_direction"],
        "interpretation": ev["interpretation"],
        "aggregates": ev["aggregates"],
        "thresholds": ev["thresholds"],
        "non_degenerate": ev["non_degenerate"],
        "degeneracy_reason": ev["degeneracy_reason"],
        "arm_results": rows,
        "wall_independent": True,
        "notes": (
            "Read-only field readout (evaluate_safety); no planning/action-commitment. "
            "Untrained frozen encoder as fixed z_world feature map; safety terrain "
            "non-parametric. Follows V3-EXQ-520 PASS (readiness). num_basis_functions "
            "sized to avoid RBF ring-buffer eviction (representation-level test)."
        ),
    }

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=dry_run,
        config=full_config,
        seeds=list(seeds),
        script_path=Path(__file__),
        started_at=t0_perf,
    )
    print(f"Result written to: {out_path}")
    print(f"Done. Outcome: {outcome}")
    return {"outcome": outcome, "manifest_path": out_path, "run_id": run_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run (reduced scope).")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    _outcome = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=result["manifest_path"],
        run_id=result["run_id"],
        dry_run=args.dry_run,
    )
    sys.exit(0)
