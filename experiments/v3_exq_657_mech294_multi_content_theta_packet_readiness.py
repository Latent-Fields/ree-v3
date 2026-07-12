"""V3-EXQ-657 -- MECH-294 multi-content theta-burst packet substrate-readiness diagnostic.

LINEAGE / ROUTING
-----------------
- Originating gate : /queue-experiment substrate-readiness gate found MECH-294
  blocked_substrate on 2026-06-09 (proposal EVB-0075) -- V3 implemented only
  single-content temporal averaging (MECH-089 ThetaBuffer), so any MECH-294
  joint-binding experiment was vacuous.
- Substrate landed (this session, /implement-substrate): sibling
  ree_core/latent/multi_content_theta_packet.py (MultiContentThetaPacket) binds a
  joint {goal_latent, action_proposal, risk_estimate (z_harm_s + z_harm_a),
  state_summary} packet within one theta cycle (E3-heartbeat interval), with
  MECH-269b per-stream V_s vintaging + a joint-read interface, behind the
  no-op-default flag use_multi_content_theta_packet (bit-identical OFF).
- THIS diagnostic validates that the substrate can build a NON-DEGENERATE joint
  packet AND that the joint regime is separable from the matched alternation
  (Kay-2020 one-stream-per-cycle) and shuffled (independent-content) controls.

CLAIM HANDLING
--------------
claim_ids = []  (substrate-readiness diagnostic). evidence_direction =
non_contributory. Does NOT validate or weaken MECH-294; the 2026-04-26 governance
hold stands until this PASSes per the design-memo S7.3 grid, after which the
MECH-294 behavioural-evidence successor (which DOES weight confidence) is queued.

DESIGN (4 arms x 3 seeds [42,43,44]; real CausalGridWorldV2; matched seeds + content)
-------------------------------------------------------------------------------------
All arms share the SAME seeds, env, and content streams. The packet is wired
READ-ONLY (theta_packet_compose_into_e3_bias=False), so it does NOT touch action
selection -- the action stream (hence the env trajectory + per-cycle content) is
IDENTICAL across arms for a matched seed. Only the binding-mode transform of the
sealed packet differs. A forced supra-threshold benefit + drive is fed to
update_z_goal each step (a CONTROLLED positive control for goal activation,
decoupled from goal_pipeline:GAP-2 -- same decoupling as V3-EXQ-636/637/603j) so
the goal_latent slot populates and G0 is testable.

  ARM_0_OFF        : use_multi_content_theta_packet=False (homogeneous-latent null).
  ARM_1_JOINT      : packet ON, binding_mode="joint"        (the MECH-294 hypothesis).
  ARM_2_ALTERNATION: packet ON, binding_mode="alternation"  (Kay-2020 control).
  ARM_3_SHUFFLED   : packet ON, binding_mode="shuffled"     (independent-content control).

READOUTS
--------
- Packet completeness fraction (all four content slots populated per cycle).
- Vintage heterogeneity fraction (cycles with >= 2 distinct component vintages).
- Conjunctive within-cycle coherence (mean pairwise cosine among the four content
  components, each L2-normalised on a common truncated dim) -- the read-only,
  parameter-free "goal-and-risk-consistency score" (memo S7.2). It is genuinely
  CONJUNCTIVE: joint binds same-cycle content (correlated by the agent's own
  dynamics); alternation/shuffled draw components from different cycles
  (decorrelated). The C2-FAIL row of the memo grid demands exactly such a readout.
- Per-cycle component-norm signature (for the structural non-vacuity gate).

GATES (substrate-readiness; thresholds pre-registered; memo S7.2/S7.3)
----------------------------------------------------------------------
  G0 packet completeness : ARM_1 completeness frac >= 0.8 on >= 2/3 seeds.
       FAIL -> substrate_not_ready_requeue (goal-pipeline blocker if goal slot
       chronically missing -- NOT a MECH-294 result).
  G1 vintage heterogeneity : ARM_1 het frac > 0 on >= 2/3 seeds.
       FAIL -> substrate_not_ready_requeue (V_s / MECH-269 wiring; substrate-ceiling).
  NON-VACUITY (the shuffle control must actually differ from intact) :
       mean matched-cycle component-norm L1 distance(ARM_1, ARM_3) > NONVAC_FLOOR
       on >= 2/3 seeds. FAIL -> substrate_not_ready_requeue (degenerate packets).
  C1 joint != alternation : |coherence(ARM_1) - coherence(ARM_2)| > C_MARGIN AND
       > the ARM_1 cross-seed baseline, on >= 2/3 seeds.
  C2 joint != shuffled    : |coherence(ARM_1) - coherence(ARM_3)| > C_MARGIN AND
       > the ARM_1 cross-seed baseline, on >= 2/3 seeds.

PASS = G0 AND G1 AND NON-VACUITY AND C1 AND C2 (label joint_binding_load_bearing).
Interpretation grid (memo S7.3) applied at review for the partial-pass rows.

Usage:
  /opt/local/bin/python3 experiments/v3_exq_657_mech294_multi_content_theta_packet_readiness.py --dry-run
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_657_mech294_multi_content_theta_packet_readiness"
QUEUE_ID = "V3-EXQ-657"
CLAIM_IDS: List[str] = []  # claim-free substrate-readiness diagnostic
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
N_EPISODES = 4
STEPS_PER_EP = 40
DRY_RUN_SEEDS = [42]
DRY_RUN_EPISODES = 2
DRY_RUN_STEPS = 12

GRID_SIZE = 8
NUM_HAZARDS = 3
NUM_RESOURCES = 3

# Controlled positive control for goal activation (decoupled from GAP-2).
FORCED_BENEFIT = 1.0
FORCED_DRIVE = 0.9

# Pre-registered thresholds.
G0_COMPLETENESS_FLOOR = 0.8     # ARM_1 fraction of cycles with all 4 slots
NONVAC_FLOOR = 1e-3             # matched-cycle component-norm L1 distance joint-vs-shuffled
C_MARGIN = 0.05                # coherence margin for C1 / C2
MIN_SEEDS_2OF3 = 2

ARMS = [
    {"arm_id": "ARM_0_OFF", "packet": False, "mode": None},
    {"arm_id": "ARM_1_JOINT", "packet": True, "mode": "joint"},
    {"arm_id": "ARM_2_ALTERNATION", "packet": True, "mode": "alternation"},
    {"arm_id": "ARM_3_SHUFFLED", "packet": True, "mode": "shuffled"},
]

_CONTENT_FIELDS = ("goal_latent", "risk_sensory", "risk_affective", "state_summary")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _coherence(pk) -> Optional[float]:
    """Conjunctive within-cycle coherence: mean pairwise cosine among the four
    content components, each L2-normalised on a common truncated dim. None when
    any content slot is missing (incomplete packet)."""
    comps = []
    for f in _CONTENT_FIELDS:
        t = getattr(pk, f)
        if t is None:
            return None
        comps.append(t.reshape(-1))
    d = min(int(c.shape[0]) for c in comps)
    if d < 1:
        return None
    us = [torch.nn.functional.normalize(c[:d], dim=0, eps=1e-8) for c in comps]
    s, n = 0.0, 0
    for i in range(len(us)):
        for j in range(i + 1, len(us)):
            s += float((us[i] * us[j]).sum().item())
            n += 1
    return s / n if n else None


def _norm_signature(pk) -> Optional[List[float]]:
    """Per-cycle component-norm signature (4 floats) for the structural
    non-vacuity gate. None when incomplete."""
    sig = []
    for f in _CONTENT_FIELDS:
        t = getattr(pk, f)
        if t is None:
            return None
        sig.append(float(t.reshape(-1).norm().item()))
    return sig


def _build_agent(arm: Dict[str, Any], harm_dim: int, harm_a_dim: int,
                 body_dim: int, world_dim: int, action_dim: int) -> REEAgent:
    # arm_cell.__enter__ already reset RNG for this seed -> distinct weight init.
    cfg = REEConfig.from_dims(
        body_obs_dim=body_dim, world_obs_dim=world_dim,
        harm_obs_dim=harm_dim, harm_obs_a_dim=harm_a_dim,
        action_dim=action_dim,
        self_dim=16, world_dim=16,
        # Goal pipeline (forced-benefit positive control activates it).
        z_goal_enabled=True, drive_weight=2.0,
        # MECH-269 per-stream V_s (required by the packet) + MECH-294 packet.
        use_per_stream_vs=True,
        use_multi_content_theta_packet=bool(arm["packet"]),
        theta_packet_binding_mode=(arm["mode"] or "joint"),
        theta_packet_compose_into_e3_bias=False,  # READ-ONLY-FIRST (memo S5)
    )
    # SD-010 / SD-011 harm streams so z_harm_s / z_harm_a populate (risk slots).
    cfg.latent.use_harm_stream = True
    cfg.latent.use_affective_harm_stream = True
    return REEAgent(cfg)


def _sense_kwargs(obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    kw: Dict[str, torch.Tensor] = {}
    for key, attr in (("harm_obs", "obs_harm"), ("harm_obs_a", "obs_harm_a")):
        v = obs.get(key)
        if v is not None:
            if v.dim() == 1:
                v = v.unsqueeze(0)
            kw[attr] = v
    return kw


def _benefit_drive(obs: Dict[str, Any]) -> Any:
    # Forced supra-threshold positive control (goal activation, decoupled GAP-2).
    return FORCED_BENEFIT, FORCED_DRIVE


def _run_cell(arm: Dict[str, Any], seed: int, n_ep: int, steps: int) -> Dict[str, Any]:
    arm_id = arm["arm_id"]
    env = CausalGridWorldV2(seed=seed, size=GRID_SIZE, num_hazards=NUM_HAZARDS,
                            num_resources=NUM_RESOURCES, use_proxy_fields=True)
    _flat, obs = env.reset()
    body = obs["body_state"]
    world = obs["world_state"]
    harm_dim = int(obs["harm_obs"].reshape(-1).shape[0]) if obs.get("harm_obs") is not None else 50
    harm_a_dim = int(obs["harm_obs_a"].reshape(-1).shape[0]) if obs.get("harm_obs_a") is not None else 50
    body_dim = int(body.reshape(-1).shape[0])
    world_dim = int(world.reshape(-1).shape[0])
    action_dim = 4

    agent = _build_agent(arm, harm_dim, harm_a_dim, body_dim, world_dim, action_dim)

    completeness_hits = 0
    vintage_het_hits = 0
    n_packets = 0
    coh_vals: List[float] = []
    norm_sigs: List[List[float]] = []  # per-cycle norm signature (complete cycles)

    for ep in range(n_ep):
        _flat, obs = env.reset()
        agent.reset()
        for _ in range(steps):
            body = obs["body_state"]
            world = obs["world_state"]
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)
            kw = _sense_kwargs(obs)
            with torch.no_grad():
                lat = agent.sense(body, world, **kw)
                # Forced-benefit goal activation (controlled positive control).
                b, drv = _benefit_drive(obs)
                agent.update_z_goal(b, drv)
                ticks = agent.clock.advance()
                e1p = (agent._e1_tick(lat) if ticks["e1_tick"]
                       else torch.zeros(1, agent.config.latent.world_dim, device=agent.device))
                cands = agent.generate_trajectories(lat, e1p, ticks)
                act = agent.select_action(cands, ticks, 1.0)
            agent._step_count += 1
            # Collect a freshly-sealed packet (E3-tick cycles only).
            pkt = agent.last_theta_packet
            if pkt is not None and ticks.get("e3_tick", False):
                n_packets += 1
                if pkt.is_complete():
                    completeness_hits += 1
                if pkt.n_distinct_vintages() >= 2:
                    vintage_het_hits += 1
                c = _coherence(pkt)
                if c is not None:
                    coh_vals.append(c)
                sig = _norm_signature(pkt)
                if sig is not None:
                    norm_sigs.append(sig)
            ai = int(act.reshape(-1).argmax().item())
            _flat, _r, _d, _i, obs = env.step(ai)
        print(f"  [run] arm={arm_id} seed={seed} ep {ep + 1}/{n_ep} "
              f"packets={n_packets}", flush=True)

    completeness_frac = (completeness_hits / n_packets) if n_packets else 0.0
    vintage_het_frac = (vintage_het_hits / n_packets) if n_packets else 0.0
    mean_coh = (sum(coh_vals) / len(coh_vals)) if coh_vals else 0.0

    return {
        "arm_id": arm_id,
        "seed": int(seed),
        "packet_on": bool(arm["packet"]),
        "binding_mode": arm["mode"],
        "n_packets": int(n_packets),
        "completeness_frac": round(completeness_frac, 6),
        "vintage_het_frac": round(vintage_het_frac, 6),
        "mean_coherence": round(mean_coh, 6),
        "n_coh_samples": len(coh_vals),
        # Per-cycle norm signatures (complete cycles only) for the non-vacuity gate.
        "norm_sigs": norm_sigs,
    }


def _matched_norm_l1(a_sigs: List[List[float]], b_sigs: List[List[float]]) -> float:
    """Mean matched-cycle component-norm L1 distance between two arms (same seed)."""
    n = min(len(a_sigs), len(b_sigs))
    if n == 0:
        return 0.0
    tot = 0.0
    for k in range(n):
        tot += sum(abs(x - y) for x, y in zip(a_sigs[k], b_sigs[k]))
    return tot / n


def _evaluate(rows: List[Dict[str, Any]], n_seeds: int) -> Dict[str, Any]:
    by = {}
    for r in rows:
        by.setdefault(r["arm_id"], {})[r["seed"]] = r

    seeds = sorted({r["seed"] for r in rows})

    # G0 + G1 on ARM_1_JOINT.
    g0_seeds = [s for s in seeds
                if by.get("ARM_1_JOINT", {}).get(s, {}).get("completeness_frac", 0.0)
                >= G0_COMPLETENESS_FLOOR]
    g1_seeds = [s for s in seeds
                if by.get("ARM_1_JOINT", {}).get(s, {}).get("vintage_het_frac", 0.0) > 0.0]

    # NON-VACUITY: matched-cycle norm distance ARM_1 vs ARM_3, per seed.
    nonvac_seeds = []
    nonvac_dist = {}
    for s in seeds:
        j = by.get("ARM_1_JOINT", {}).get(s, {}).get("norm_sigs", [])
        sh = by.get("ARM_3_SHUFFLED", {}).get(s, {}).get("norm_sigs", [])
        d = _matched_norm_l1(j, sh)
        nonvac_dist[s] = round(d, 6)
        if d > NONVAC_FLOOR:
            nonvac_seeds.append(s)

    # ARM_1 cross-seed coherence baseline (max pairwise |coh_i - coh_j|).
    j_cohs = [by.get("ARM_1_JOINT", {}).get(s, {}).get("mean_coherence", 0.0) for s in seeds]
    baseline = 0.0
    for i in range(len(j_cohs)):
        for k in range(i + 1, len(j_cohs)):
            baseline = max(baseline, abs(j_cohs[i] - j_cohs[k]))

    # C1 (joint vs alternation), C2 (joint vs shuffled): per-seed coherence delta
    # exceeds C_MARGIN AND the cross-seed baseline.
    def _disc_seeds(ctrl_arm: str) -> List[int]:
        out = []
        for s in seeds:
            cj = by.get("ARM_1_JOINT", {}).get(s, {}).get("mean_coherence", 0.0)
            cc = by.get(ctrl_arm, {}).get(s, {}).get("mean_coherence", 0.0)
            delta = abs(cj - cc)
            if delta > C_MARGIN and delta > baseline:
                out.append(s)
        return out

    c1_seeds = _disc_seeds("ARM_2_ALTERNATION")
    c2_seeds = _disc_seeds("ARM_3_SHUFFLED")

    g0 = len(g0_seeds) >= MIN_SEEDS_2OF3
    g1 = len(g1_seeds) >= MIN_SEEDS_2OF3
    nonvac = len(nonvac_seeds) >= MIN_SEEDS_2OF3
    c1 = len(c1_seeds) >= MIN_SEEDS_2OF3
    c2 = len(c2_seeds) >= MIN_SEEDS_2OF3

    # Verdict per the memo S7.3 interpretation grid.
    if not g0:
        label = "substrate_not_ready_requeue"
        reading = ("ARM_1 packets chronically incomplete (G0 fail) -- most likely "
                   "the goal_latent slot is missing (goal pipeline inert). Routes to "
                   "the goal-pipeline blocker, NOT a MECH-294 result.")
        overall = False
    elif not g1:
        label = "substrate_not_ready_requeue"
        reading = ("Single-vintage packets (G1 fail) -- the MECH-269b V_s vintaging "
                   "is inert on this run. Substrate-ceiling / V_s wiring; not a pass.")
        overall = False
    elif not nonvac:
        label = "substrate_not_ready_requeue"
        reading = ("Shuffled packets do not structurally differ from joint "
                   "(non-vacuity fail) -- the controls are degenerate; the experiment "
                   "would be vacuous. Re-queue with a deeper history / longer run.")
        overall = False
    elif g0 and g1 and c1 and c2:
        label = "joint_binding_load_bearing"
        reading = ("PASS -- the substrate builds a non-degenerate joint packet AND "
                   "within-cycle co-binding is behaviourally separable from BOTH the "
                   "alternation and shuffled controls. Substrate-readiness gate cleared; "
                   "queue the MECH-294 behavioural-evidence successor.")
        overall = True
    elif g0 and g1 and c1 and not c2:
        label = "multi_content_yes_joint_not_isolated"
        reading = ("Joint beats alternation but ties shuffled: 'four streams present' "
                   "carries the signal, not their co-observation. Refine to a more "
                   "conjunctive readout before claiming joint binding. Do NOT promote "
                   "MECH-294's joint clause.")
        overall = False
    elif g0 and g1 and not c1:
        label = "kay_parsimonious_survives"
        reading = ("Joint indistinguishable from alternation (C1 fail) -- the Kay-2020 "
                   "parsimonious outcome survives on REE's own substrate. Route to "
                   "/failure-autopsy; the claim's joint clause is NOT promotable and "
                   "the 2026-04-26 governance hold stands.")
        overall = False
    else:
        label = "indeterminate"
        reading = "Unhandled gate combination; review the gate table."
        overall = False

    return {
        "label": label,
        "reading": reading,
        "overall_pass": overall,
        "gate_pass": {"G0": g0, "G1": g1, "non_vacuity": nonvac, "C1": c1, "C2": c2},
        "gate_seeds": {
            "G0_completeness": g0_seeds, "G1_vintage_het": g1_seeds,
            "non_vacuity": nonvac_seeds, "C1_joint_vs_alt": c1_seeds,
            "C2_joint_vs_shuf": c2_seeds,
        },
        "coherence_by_seed": {
            a: {s: by.get(a, {}).get(s, {}).get("mean_coherence", 0.0) for s in seeds}
            for a in ("ARM_0_OFF", "ARM_1_JOINT", "ARM_2_ALTERNATION", "ARM_3_SHUFFLED")
        },
        "arm1_cross_seed_baseline": round(baseline, 6),
        "nonvacuity_distance_by_seed": nonvac_dist,
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    n_ep = DRY_RUN_EPISODES if dry_run else N_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EP

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            with arm_cell(
                seed,
                config_slice={
                    "arm": arm, "grid_size": GRID_SIZE, "num_hazards": NUM_HAZARDS,
                    "num_resources": NUM_RESOURCES, "n_episodes": n_ep,
                    "steps_per_ep": steps, "forced_benefit": FORCED_BENEFIT,
                    "forced_drive": FORCED_DRIVE,
                },
                script_path=Path(__file__),
                extra_ineligible_reasons=["substrate_readiness_per_cell_agent_build"],
            ) as cell:
                row = _run_cell(arm, seed, n_ep, steps)
                cell.stamp(row)
            # Drop the bulky norm_sigs from the persisted row (kept only for the
            # in-memory non-vacuity computation); store a compact count instead.
            arm_results.append(row)

    summary = _evaluate(arm_results, n_seeds=len(seeds))
    outcome = "PASS" if summary["overall_pass"] else "FAIL"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    # Compact the persisted arm rows (drop per-cycle norm_sigs arrays).
    persisted_rows = []
    for r in arm_results:
        rr = dict(r)
        rr["n_norm_sig_cycles"] = len(rr.pop("norm_sigs", []))
        persisted_rows.append(rr)

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "non_contributory",
        "outcome": outcome,
        "title": "MECH-294 multi-content theta-burst packet substrate-readiness (joint vs alternation vs shuffled)",
        "hypothesis": (
            "The MultiContentThetaPacket builds a non-degenerate joint packet and "
            "the joint within-cycle co-binding is separable from the matched "
            "alternation (Kay-2020) and shuffled (independent-content) controls."
        ),
        "interpretation": {
            "label": summary["label"],
            "reading": summary["reading"],
            "grid": {
                "joint_binding_load_bearing": (
                    "PASS -> queue the MECH-294 behavioural-evidence successor; the "
                    "substrate-readiness gate that found MECH-294 blocked_substrate is "
                    "cleared. Do NOT promote MECH-294 here (governance evidence is the "
                    "successor's job)."
                ),
                "multi_content_yes_joint_not_isolated": (
                    "C2 FAIL -> 'four streams present' carries the signal, not the "
                    "co-observation. Refine the readout to a genuinely conjunctive one; "
                    "do NOT promote MECH-294's joint clause."
                ),
                "kay_parsimonious_survives": (
                    "C1 FAIL -> Kay-2020 parsimonious outcome survives on REE's substrate; "
                    "/failure-autopsy; joint clause NOT promotable; 2026-04-26 hold stands."
                ),
                "substrate_not_ready_requeue": (
                    "G0/G1/non-vacuity FAIL -> packet degenerate (missing streams / "
                    "single-vintage / shuffle-ties-joint). Route to the upstream blocker "
                    "(goal pipeline / V_s wiring) or re-queue; NOT a MECH-294 result."
                ),
            },
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds, "n_episodes": n_ep, "steps_per_ep": steps,
            "grid_size": GRID_SIZE, "num_hazards": NUM_HAZARDS,
            "num_resources": NUM_RESOURCES,
            "forced_benefit": FORCED_BENEFIT, "forced_drive": FORCED_DRIVE,
            "arms": [a["arm_id"] for a in ARMS],
            "read_only_first": True,
            "thresholds": {
                "G0_completeness_floor": G0_COMPLETENESS_FLOOR,
                "nonvacuity_floor": NONVAC_FLOOR,
                "coherence_margin": C_MARGIN,
                "min_seeds_2of3": MIN_SEEDS_2OF3,
            },
        },
        "acceptance_criteria": {
            **summary["gate_pass"],
            "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": persisted_rows,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = out_dir / f"{run_id}.json"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome} (label={summary['label']})", flush=True)
    print(f"  gate_pass: {summary['gate_pass']}", flush=True)
    print(f"  coherence_by_seed: {summary['coherence_by_seed']}", flush=True)
    print(f"  nonvacuity_distance_by_seed: {summary['nonvacuity_distance_by_seed']}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-657 MECH-294 multi-content theta-burst packet readiness diagnostic"
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run.")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
    )
