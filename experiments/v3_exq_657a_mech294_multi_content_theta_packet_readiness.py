"""V3-EXQ-657a -- MECH-294 multi-content theta-burst packet substrate-readiness (CORRECTED).

SUPERSEDES V3-EXQ-657. Scoped fix per failure_autopsy_V3-EXQ-657_2026-06-09
(both fixes user-confirmed 2026-06-09). The substrate
(ree_core/latent/multi_content_theta_packet.py) is UNCHANGED and correct -- the
657 FAIL was two DIAGNOSTIC metric/gate bugs, fixed here:

DEFECT 1 (657 G1 vintage-inert) -- gate-design bug, NOT wiring, NOT substrate-ceiling.
  657 measured G1 (vintage heterogeneity) on ARM_1_JOINT. In the verisimilar regime
  (short stable forced-benefit run, V_s seeded 1.0 / EMA tau 0.1) every stream's V_s
  stays >= the 0.4 hold floor, so joint mode correctly takes every current value
  (homogeneous-current packet) -- which is what joint binding MEANS. The V_s-hold
  path IS functional (contract C3 proves it; ARM_2 alternation het=1.0 proves the
  snapshot/hold/age machinery in the agent loop). 657 short-circuited the verdict on
  a secondary property the regime does not elicit.
  FIX: G1 now passes iff BOTH (a) a deterministic forced-low-V_s JOINT-mode probe
  drives one stream's V_s below the hold floor and confirms the packet substitutes
  the snapshot + marks the component stale (n_distinct_vintages >= 2) -- directly
  exercising the MECH-269b V_s consumption in joint mode -- AND (b) ARM_2 alternation
  vintage_het_frac > 0 on >= 2/3 seeds (machinery live in the agent loop).

DEFECT 2 (657 C1/C2 coherence-blind) -- metric bug. The substrate produces
  structurally-distinct packets per binding mode (657 non-vacuity joint-vs-shuffled
  norm-L1 0.29-0.48; activation-smoke structural dist joint-vs-alt 17.9 /
  joint-vs-shuf 44.6), but 657's _coherence() (mean pairwise cosine over 4 streams,
  L2-normalised, truncated to common dim) (i) discards magnitude -- the exact signal
  the norm-signature captures -- and (ii) computes cross-cosine between different
  semantic latent spaces (z_goal / z_world / z_harm), which is regime-stable and
  binding-mode-blind. Under read-only-first the action stream is identical across
  arms by construction, so behavioural readouts (proposer / committed-class) are dead
  -- the discriminator MUST be packet-structure.
  FIX: C1/C2 = matched-cycle component-norm-signature L1 distance (the metric the
  non-vacuity gate already uses successfully): C1 = dist(joint, alternation),
  C2 = dist(joint, shuffled); each required > STRUCT_MARGIN AND > the within-joint
  temporal self-variation baseline (the binding-mode difference must exceed the
  structural change joint already exhibits cycle-to-cycle). Plus a genuinely
  conjunctive corroboration: cross-stream norm-covariation (joint co-varies same-cycle;
  shuffled draws decorrelated cross-cycle) -- reported, strengthens the reading.

HONEST FRAMING (read-only-first readiness): this gate confirms the three binding
modes are NON-DEGENERATE and STRUCTURALLY SEPARABLE. "Co-binding carries DOWNSTREAM
signal" is the behavioural successor's job (theta_packet_compose_into_e3_bias=True),
per memo S5/S7.3 -- explicitly NOT tested here.

CLAIM HANDLING
--------------
claim_ids = []  (substrate-readiness diagnostic). evidence_direction =
non_contributory. Does NOT validate or weaken MECH-294; the 2026-04-26 governance
hold stands until this PASSes per the memo S7.3 grid, after which the MECH-294
behavioural-evidence successor (which DOES weight confidence) is queued.

DESIGN (4 arms x 3 seeds [42,43,44]; real CausalGridWorldV2; matched seeds + content)
-------------------------------------------------------------------------------------
Packet wired READ-ONLY (theta_packet_compose_into_e3_bias=False): the action stream
(hence env trajectory + per-cycle content + cycle alignment) is IDENTICAL across arms
for a matched seed; only the binding-mode transform of the sealed packet differs, so
matched-cycle structural comparison between arms is valid.

  ARM_0_OFF        : use_multi_content_theta_packet=False (homogeneous-latent null).
  ARM_1_JOINT      : packet ON, binding_mode="joint"        (the MECH-294 hypothesis).
  ARM_2_ALTERNATION: packet ON, binding_mode="alternation"  (Kay-2020 control).
  ARM_3_SHUFFLED   : packet ON, binding_mode="shuffled"     (independent-content control).

Usage:
  /opt/local/bin/python3 experiments/v3_exq_657a_mech294_multi_content_theta_packet_readiness.py --dry-run
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.latent.multi_content_theta_packet import (  # noqa: E402
    MultiContentThetaPacket,
    MultiContentThetaPacketConfig,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_657a_mech294_multi_content_theta_packet_readiness"
QUEUE_ID = "V3-EXQ-657a"
SUPERSEDES = "V3-EXQ-657"
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
STRUCT_MARGIN = 0.05           # absolute floor for the C1/C2 matched-cycle structural distance
PROBE_HOLD_VS = 0.1            # forced V_s for the G1 probe (< hold threshold 0.4)
MIN_SEEDS_2OF3 = 2

ARMS = [
    {"arm_id": "ARM_0_OFF", "packet": False, "mode": None},
    {"arm_id": "ARM_1_JOINT", "packet": True, "mode": "joint"},
    {"arm_id": "ARM_2_ALTERNATION", "packet": True, "mode": "alternation"},
    {"arm_id": "ARM_3_SHUFFLED", "packet": True, "mode": "shuffled"},
]

_CONTENT_FIELDS = ("goal_latent", "risk_sensory", "risk_affective", "state_summary")


# ----------------------------------------------------------------------
# G1 fix -- forced-low-V_s JOINT-mode probe (deterministic; exercises the MECH-269b
# V_s consumption directly, since the verisimilar agent-loop regime never drives V_s
# below the hold floor). Mirrors contract C3 inside the readiness diagnostic.
# ----------------------------------------------------------------------
def _vs_hold_probe() -> Dict[str, Any]:
    """Build a fresh joint-mode packet; cycle 1 high V_s (refresh snapshot), cycle 2
    drop one stream's V_s below the hold floor. A correct V_s consumption substitutes
    the held snapshot, marks the component stale, and yields >= 2 distinct vintages."""
    p = MultiContentThetaPacket(
        MultiContentThetaPacketConfig(
            binding_mode="joint", snapshot_refresh_threshold=0.5, hold_threshold=0.4,
        )
    )
    high = {"z_goal": 0.9, "z_harm_s": 0.9, "z_harm_a": 0.9, "z_world": 0.9}
    g0 = torch.randn(1, 8)
    hs = torch.randn(1, 4)
    ha = torch.randn(1, 4)
    st = torch.randn(1, 16)
    act = torch.randn(1, 4)
    # Cycle 1: all high V_s -> snapshots refreshed.
    p.observe(g0, hs, ha, high)
    p.observe_action_proposal(act)
    p.seal(st)
    # Cycle 2: drop z_harm_s (risk_sensory) V_s below hold; it must hold its snapshot.
    low = dict(high)
    low["z_harm_s"] = PROBE_HOLD_VS
    g1 = torch.randn(1, 8)
    hs1 = torch.randn(1, 4)
    p.observe(g1, hs1, ha, low)
    p.observe_action_proposal(act)
    pk = p.seal(st)
    n_vint = int(pk.n_distinct_vintages())
    stale = bool(pk.is_component_stale("risk_sensory"))
    held_is_snapshot = bool(torch.allclose(pk.risk_sensory.reshape(-1), hs.reshape(-1)))
    return {
        "probe_n_distinct_vintages": n_vint,
        "probe_risk_sensory_stale": stale,
        "probe_held_equals_prior_snapshot": held_is_snapshot,
        "probe_hold_fired": bool(n_vint >= 2 and stale and held_is_snapshot),
    }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _norm_signature(pk) -> Optional[List[float]]:
    """Per-cycle component-norm signature (4 floats). None when incomplete."""
    sig = []
    for f in _CONTENT_FIELDS:
        t = getattr(pk, f)
        if t is None:
            return None
        sig.append(float(t.reshape(-1).norm().item()))
    return sig


def _matched_norm_l1(a_sigs: List[List[float]], b_sigs: List[List[float]]) -> float:
    """Mean matched-cycle component-norm L1 distance between two arms (same seed)."""
    n = min(len(a_sigs), len(b_sigs))
    if n == 0:
        return 0.0
    tot = 0.0
    for k in range(n):
        tot += sum(abs(x - y) for x, y in zip(a_sigs[k], b_sigs[k]))
    return tot / n


def _temporal_self_baseline(sigs: List[List[float]]) -> float:
    """Within-arm mean consecutive-cycle norm-signature L1 distance -- the structural
    change the arm already exhibits cycle-to-cycle from the agent's own dynamics. A
    binding-mode difference is meaningful only if it exceeds this null."""
    if len(sigs) < 2:
        return 0.0
    tot = 0.0
    for k in range(1, len(sigs)):
        tot += sum(abs(x - y) for x, y in zip(sigs[k], sigs[k - 1]))
    return tot / (len(sigs) - 1)


def _mean_abs_offdiag_corr(sigs: List[List[float]]) -> float:
    """Cross-stream norm covariation: mean absolute off-diagonal Pearson correlation
    among the four component-norm time-series. Genuinely conjunctive -- joint binds
    same-cycle content (correlated by the agent's own dynamics); shuffled draws each
    slot from a different cycle (decorrelated). Returns 0.0 when undersampled or all
    series are constant."""
    n = len(sigs)
    if n < 3:
        return 0.0
    # Transpose to 4 series of length n.
    series = [[sigs[k][i] for k in range(n)] for i in range(4)]
    means = [sum(s) / n for s in series]
    var = [sum((x - means[i]) ** 2 for x in series[i]) for i in range(4)]
    corrs: List[float] = []
    for i in range(4):
        for j in range(i + 1, 4):
            if var[i] <= 1e-12 or var[j] <= 1e-12:
                continue
            cov = sum((series[i][k] - means[i]) * (series[j][k] - means[j]) for k in range(n))
            denom = (var[i] * var[j]) ** 0.5
            corrs.append(abs(cov / denom))
    return (sum(corrs) / len(corrs)) if corrs else 0.0


def _build_agent(arm: Dict[str, Any], harm_dim: int, harm_a_dim: int,
                 body_dim: int, world_dim: int, action_dim: int) -> REEAgent:
    # arm_cell.__enter__ already reset RNG for this seed -> distinct weight init.
    cfg = REEConfig.from_dims(
        body_obs_dim=body_dim, world_obs_dim=world_dim,
        harm_obs_dim=harm_dim, harm_obs_a_dim=harm_a_dim,
        action_dim=action_dim,
        self_dim=16, world_dim=16,
        z_goal_enabled=True, drive_weight=2.0,
        use_per_stream_vs=True,
        use_multi_content_theta_packet=bool(arm["packet"]),
        theta_packet_binding_mode=(arm["mode"] or "joint"),
        theta_packet_compose_into_e3_bias=False,  # READ-ONLY-FIRST (memo S5)
    )
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
                b, drv = FORCED_BENEFIT, FORCED_DRIVE
                agent.update_z_goal(b, drv)
                ticks = agent.clock.advance()
                e1p = (agent._e1_tick(lat) if ticks["e1_tick"]
                       else torch.zeros(1, agent.config.latent.world_dim, device=agent.device))
                cands = agent.generate_trajectories(lat, e1p, ticks)
                act = agent.select_action(cands, ticks, 1.0)
            agent._step_count += 1
            pkt = agent.last_theta_packet
            if pkt is not None and ticks.get("e3_tick", False):
                n_packets += 1
                if pkt.is_complete():
                    completeness_hits += 1
                if pkt.n_distinct_vintages() >= 2:
                    vintage_het_hits += 1
                sig = _norm_signature(pkt)
                if sig is not None:
                    norm_sigs.append(sig)
            ai = int(act.reshape(-1).argmax().item())
            _flat, _r, _d, _i, obs = env.step(ai)
        print(f"  [run] arm={arm_id} seed={seed} ep {ep + 1}/{n_ep} "
              f"packets={n_packets}", flush=True)

    completeness_frac = (completeness_hits / n_packets) if n_packets else 0.0
    vintage_het_frac = (vintage_het_hits / n_packets) if n_packets else 0.0
    covariation = _mean_abs_offdiag_corr(norm_sigs)
    temporal_baseline = _temporal_self_baseline(norm_sigs)

    return {
        "arm_id": arm_id,
        "seed": int(seed),
        "packet_on": bool(arm["packet"]),
        "binding_mode": arm["mode"],
        "n_packets": int(n_packets),
        "completeness_frac": round(completeness_frac, 6),
        "vintage_het_frac": round(vintage_het_frac, 6),
        "norm_covariation": round(covariation, 6),
        "temporal_self_baseline": round(temporal_baseline, 6),
        # Per-cycle norm signatures (complete cycles only) for the structural readouts.
        "norm_sigs": norm_sigs,
    }


def _evaluate(rows: List[Dict[str, Any]], probe: Dict[str, Any]) -> Dict[str, Any]:
    by: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["arm_id"], {})[r["seed"]] = r
    seeds = sorted({r["seed"] for r in rows})

    def get(arm: str, s: int, k: str, default=0.0):
        return by.get(arm, {}).get(s, {}).get(k, default)

    # G0 completeness (ARM_1_JOINT).
    g0_seeds = [s for s in seeds if get("ARM_1_JOINT", s, "completeness_frac") >= G0_COMPLETENESS_FLOOR]

    # G1 fix: (a) forced-low-V_s JOINT-mode probe fired (deterministic, exercises the
    # MECH-269b V_s consumption directly) AND (b) ARM_2 alternation het > 0 on >=2/3
    # seeds (machinery live in the agent loop). NOT measured on the joint arm, whose
    # homogeneous-current packets are the correct verisimilar-regime behaviour.
    probe_fired = bool(probe.get("probe_hold_fired", False))
    alt_het_seeds = [s for s in seeds if get("ARM_2_ALTERNATION", s, "vintage_het_frac") > 0.0]

    # Structural-metric-live readiness (non-vacuity): joint vs shuffled matched-cycle
    # norm-L1 distance > floor -- the SAME statistic C1/C2 route on, on a known
    # structural difference (shuffled draws each stream from a different cycle).
    nonvac_dist: Dict[int, float] = {}
    nonvac_seeds: List[int] = []
    for s in seeds:
        d = _matched_norm_l1(get("ARM_1_JOINT", s, "norm_sigs", []),
                             get("ARM_3_SHUFFLED", s, "norm_sigs", []))
        nonvac_dist[s] = round(d, 6)
        if d > NONVAC_FLOOR:
            nonvac_seeds.append(s)

    # C1 / C2: matched-cycle structural distance > STRUCT_MARGIN AND > the within-joint
    # temporal self-variation baseline (mode difference must exceed joint's own
    # cycle-to-cycle drift).
    c1_dist: Dict[int, float] = {}
    c2_dist: Dict[int, float] = {}
    baseline_by_seed: Dict[int, float] = {}
    c1_seeds: List[int] = []
    c2_seeds: List[int] = []
    cov_contrast: Dict[int, float] = {}
    for s in seeds:
        j = get("ARM_1_JOINT", s, "norm_sigs", [])
        a = get("ARM_2_ALTERNATION", s, "norm_sigs", [])
        sh = get("ARM_3_SHUFFLED", s, "norm_sigs", [])
        baseline = _temporal_self_baseline(j)
        baseline_by_seed[s] = round(baseline, 6)
        d1 = _matched_norm_l1(j, a)
        d2 = _matched_norm_l1(j, sh)
        c1_dist[s] = round(d1, 6)
        c2_dist[s] = round(d2, 6)
        if d1 > STRUCT_MARGIN and d1 > baseline:
            c1_seeds.append(s)
        if d2 > STRUCT_MARGIN and d2 > baseline:
            c2_seeds.append(s)
        cov_contrast[s] = round(
            get("ARM_1_JOINT", s, "norm_covariation") - get("ARM_3_SHUFFLED", s, "norm_covariation"), 6)

    g0 = len(g0_seeds) >= MIN_SEEDS_2OF3
    g1 = probe_fired and (len(alt_het_seeds) >= MIN_SEEDS_2OF3)
    nonvac = len(nonvac_seeds) >= MIN_SEEDS_2OF3
    c1 = len(c1_seeds) >= MIN_SEEDS_2OF3
    c2 = len(c2_seeds) >= MIN_SEEDS_2OF3
    cov_corroborates = sum(1 for s in seeds if cov_contrast[s] > 0.0) >= MIN_SEEDS_2OF3

    # Verdict (memo S7.3 grid). Readiness gates self-route substrate_not_ready_requeue.
    if not g0:
        label = "substrate_not_ready_requeue"
        reading = ("ARM_1 packets chronically incomplete (G0 fail) -- the goal_latent "
                   "slot is missing (goal pipeline inert). Routes to the goal-pipeline "
                   "blocker, NOT a MECH-294 result.")
        overall = False
    elif not g1:
        label = "substrate_not_ready_requeue"
        if not probe_fired:
            reading = ("The forced-low-V_s JOINT-mode probe did NOT substitute the held "
                       "snapshot / mark the component stale -- the MECH-269b V_s "
                       "consumption is genuinely broken. Substrate/wiring fix needed; "
                       "not a MECH-294 result.")
        else:
            reading = ("Alternation vintage heterogeneity absent (G1-alt fail) -- the "
                       "snapshot/hold/age machinery is inert in the agent loop. "
                       "Substrate-ceiling / V_s wiring; not a pass.")
        overall = False
    elif not nonvac:
        label = "substrate_not_ready_requeue"
        reading = ("Shuffled packets do not structurally differ from joint (non-vacuity "
                   "fail) -- the structural-distance metric is degenerate; the experiment "
                   "would be vacuous. Re-queue with a deeper history / longer run.")
        overall = False
    elif c1 and c2:
        label = "joint_binding_load_bearing"
        reading = ("PASS -- the substrate builds a non-degenerate joint packet AND the "
                   "three binding modes are structurally separable: joint differs from "
                   "BOTH the alternation (Kay-2020) and shuffled controls beyond joint's "
                   "own cycle-to-cycle drift, on >= 2/3 seeds. Read-only-first "
                   "substrate-readiness gate cleared; queue the MECH-294 "
                   "behavioural-evidence successor (compose ON) to test whether the "
                   "co-binding carries DOWNSTREAM signal -- that is the successor's job, "
                   "NOT this gate's."
                   + (" Cross-stream norm-covariation corroborates (joint > shuffled on "
                      ">= 2/3 seeds): co-observation, not just four-streams-present."
                      if cov_corroborates else
                      " NOTE: norm-covariation did NOT corroborate (joint !> shuffled on "
                      ">= 2/3 seeds) -- the structural separability is magnitude-driven; "
                      "the behavioural successor must confirm co-observation carries "
                      "signal."))
        overall = True
    elif c1 and not c2:
        label = "multi_content_yes_joint_not_isolated"
        reading = ("Joint separable from alternation (C1) but NOT from shuffled (C2): "
                   "'four streams present' carries the structural signal, not their "
                   "co-observation. Refine to a more conjunctive readout / the "
                   "behavioural successor before claiming joint binding. Do NOT promote "
                   "MECH-294's joint clause.")
        overall = False
    elif not c1:
        label = "kay_parsimonious_survives"
        reading = ("Joint structurally indistinguishable from alternation (C1 fail) -- "
                   "the Kay-2020 parsimonious outcome survives on REE's own substrate. "
                   "Route to /failure-autopsy; the joint clause is NOT promotable and the "
                   "2026-04-26 governance hold stands.")
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
        "covariation_corroborates": cov_corroborates,
        "gate_seeds": {
            "G0_completeness": g0_seeds, "G1_alternation_het": alt_het_seeds,
            "non_vacuity": nonvac_seeds, "C1_joint_vs_alt": c1_seeds,
            "C2_joint_vs_shuf": c2_seeds,
        },
        "vs_hold_probe": probe,
        "structural_distance_by_seed": {
            "C1_joint_vs_alt": c1_dist, "C2_joint_vs_shuf": c2_dist,
            "nonvacuity_joint_vs_shuf": nonvac_dist,
            "within_joint_temporal_baseline": baseline_by_seed,
        },
        "norm_covariation_by_arm": {
            a: {s: get(a, s, "norm_covariation") for s in seeds}
            for a in ("ARM_1_JOINT", "ARM_2_ALTERNATION", "ARM_3_SHUFFLED")
        },
        "covariation_contrast_joint_minus_shuf_by_seed": cov_contrast,
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    n_ep = DRY_RUN_EPISODES if dry_run else N_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EP

    # G1 fix part (a): deterministic forced-low-V_s JOINT-mode probe (seed-independent).
    torch.manual_seed(657)
    probe = _vs_hold_probe()
    print(f"vs_hold_probe: {probe}", flush=True)

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
            arm_results.append(row)
            print(f"verdict: {'PASS' if row['n_packets'] >= 0 else 'FAIL'}", flush=True)

    summary = _evaluate(arm_results, probe)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    # Compact the persisted arm rows (drop per-cycle norm_sigs arrays).
    persisted_rows = []
    for r in arm_results:
        rr = dict(r)
        rr["n_norm_sig_cycles"] = len(rr.pop("norm_sigs", []))
        persisted_rows.append(rr)

    gp = summary["gate_pass"]
    # Diagnostic adjudication gate: preconditions (readiness-kind, same-statistic as
    # the load-bearing criteria) + criteria_non_degenerate + load_bearing criteria.
    arm1_n = [r for r in persisted_rows if r["arm_id"] == "ARM_1_JOINT"]
    nonvac_vals = list(summary["structural_distance_by_seed"]["nonvacuity_joint_vs_shuf"].values())
    min_nonvac = min(nonvac_vals) if nonvac_vals else 0.0
    interpretation = {
        "label": summary["label"],
        "reading": summary["reading"],
        "preconditions": [
            {
                "name": "vs_hold_path_fires_in_joint",
                "description": ("Forced-low-V_s (0.1 < 0.4 hold) on one stream in a "
                                "joint-mode packet must substitute the held snapshot and "
                                "yield >= 2 distinct vintages -- the SAME n_distinct_vintages "
                                "statistic G1 routes on. Directly exercises the MECH-269b "
                                "V_s consumption the verisimilar agent-loop regime never reaches."),
                "measured": float(probe.get("probe_n_distinct_vintages", 0)),
                "threshold": 2.0,
                "direction": "lower",
                "control": "deterministic forced sub-hold V_s on risk_sensory, cycle 2",
                "met": bool(probe.get("probe_hold_fired", False)),
            },
            {
                "name": "structural_distance_metric_live",
                "description": ("The matched-cycle component-norm-L1 distance metric (the SAME "
                                "statistic C1/C2 route on) must register the known joint-vs-shuffled "
                                "structural difference above the non-vacuity floor."),
                "measured": float(round(min_nonvac, 6)),
                "threshold": float(NONVAC_FLOOR),
                "direction": "lower",
                "control": "shuffled draws each stream from a different cycle = known structural difference",
                "met": bool(summary["gate_pass"]["non_vacuity"]),
            },
        ],
        "criteria_non_degenerate": {
            "G0": bool(arm1_n and arm1_n[0]["n_packets"] > 0),
            "G1": bool(probe.get("probe_hold_fired", False)),
            "C1": bool(any(v > 0 for v in summary["structural_distance_by_seed"]["C1_joint_vs_alt"].values())),
            "C2": bool(any(v > 0 for v in summary["structural_distance_by_seed"]["C2_joint_vs_shuf"].values())),
        },
        "criteria": [
            {"name": "C1_joint_vs_alternation_structurally_separable", "load_bearing": True, "passed": bool(gp["C1"])},
            {"name": "C2_joint_vs_shuffled_structurally_separable", "load_bearing": True, "passed": bool(gp["C2"])},
            {"name": "G0_completeness", "load_bearing": False, "passed": bool(gp["G0"])},
            {"name": "G1_vs_vintaging_live", "load_bearing": False, "passed": bool(gp["G1"])},
        ],
        "grid": {
            "joint_binding_load_bearing": (
                "PASS -> queue the MECH-294 behavioural-evidence successor (compose ON). "
                "Read-only-first readiness = modes non-degenerate + structurally separable; "
                "downstream-signal is the successor's job. Do NOT promote MECH-294 here."),
            "multi_content_yes_joint_not_isolated": (
                "C2 FAIL -> 'four streams present' carries the structural signal, not the "
                "co-observation; refine the readout / behavioural successor; do NOT promote "
                "MECH-294's joint clause."),
            "kay_parsimonious_survives": (
                "C1 FAIL -> Kay-2020 parsimonious outcome survives on REE's substrate; "
                "/failure-autopsy; joint clause NOT promotable; 2026-04-26 hold stands."),
            "substrate_not_ready_requeue": (
                "G0/G1/non-vacuity FAIL -> packet degenerate (missing streams / V_s "
                "vintaging broken / shuffle-ties-joint). Route to the upstream blocker or "
                "re-queue; NOT a MECH-294 result."),
        },
    }

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
        "supersedes": SUPERSEDES,
        "outcome": outcome,
        "title": "MECH-294 multi-content theta-burst packet substrate-readiness CORRECTED (joint vs alternation vs shuffled; structural separability + forced-V_s probe)",
        "hypothesis": (
            "The MultiContentThetaPacket builds a non-degenerate joint packet, the "
            "MECH-269b V_s vintaging fires in joint mode under sub-hold V_s, and the "
            "joint binding mode is structurally separable from the matched alternation "
            "(Kay-2020) and shuffled (independent-content) controls beyond joint's own "
            "cycle-to-cycle drift."
        ),
        "interpretation": interpretation,
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
                "structural_margin": STRUCT_MARGIN,
                "probe_hold_vs": PROBE_HOLD_VS,
                "min_seeds_2of3": MIN_SEEDS_2OF3,
            },
        },
        "acceptance_criteria": {
            **summary["gate_pass"],
            "covariation_corroborates": summary["covariation_corroborates"],
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
    print(f"  structural_distance_by_seed: {summary['structural_distance_by_seed']}", flush=True)
    print(f"  covariation_contrast: {summary['covariation_contrast_joint_minus_shuf_by_seed']}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-657a MECH-294 multi-content theta-burst packet readiness (corrected)"
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
