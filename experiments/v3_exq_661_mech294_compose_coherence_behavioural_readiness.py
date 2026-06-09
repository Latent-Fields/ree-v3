"""V3-EXQ-661 -- MECH-294 compose-coherence substrate-readiness (behavioural wiring).

EXPERIMENT_PURPOSE = diagnostic (claim_ids = []). Validates the 2026-06-09
compose-coherence wiring landed in ree-v3 53d4c75: compose_e3_bias now GATES the
per-candidate action-grounding bias by ThetaPacket.currency_coherence() (the
fraction of co-temporally-current content streams), so the binding mode causally
reaches E3 behaviour. Before this wiring, compose_e3_bias read ONLY action_proposal
(identical across binding modes), so compose-ON produced behaviourally identical
action streams across joint/alternation/shuffled -- the design-doc S6 C1-FAIL
"wiring" case. THIS run is the substrate-readiness gate that, on PASS, unblocks the
MECH-294 behavioural-EVIDENCE successor (claim_ids=["MECH-294"]) as a SEPARATE
/queue-experiment session. It does NOT weight MECH-294 confidence.

WHAT IT MEASURES (compose ON; the readout is the committed/selected action-class
distribution per arm; the discriminator is whether the binding mode CHANGES it):

  ARM_0_OFF        : packet OFF (no compose) -- unbiased baseline distribution.
  ARM_1_JOINT      : packet ON, joint, compose ON, coherence-gating ON.
  ARM_2_ALTERNATION: packet ON, alternation, compose ON, coherence-gating ON.
  ARM_3_SHUFFLED   : packet ON, shuffled, compose ON, coherence-gating ON.
  ARM_4_ALT_COH_OFF: packet ON, alternation, compose ON, coherence-gating OFF
                     (gate==1.0, full bias -- recovers the legacy action-only path).

Coherence gate by mode (the substrate mechanism): joint ~1.0 (all streams current),
alternation ~0.25 (one live, three held -- Kay-2020), shuffled 0.0 (all stale),
alt-coh-OFF forces gate==1.0 regardless. Because action_proposal is identical across
modes, ARM_4 (alt, coh OFF) applies the SAME full bias as ARM_1 (joint, gate~1.0) and
should behave like JOINT -- the contrast ARM_2 (alt, gate~0.25) vs ARM_4 (alt, gate
1.0) isolates the coherence gating as the mechanism (not mere packet presence).

PRE-REGISTERED GATES (TV = total-variation distance of normalised 4-class committed
action histograms; margin 0.10 per design-memo S7.2; >= 2/3 seeds):

  Non-vacuity readiness (matched to the load-bearing criterion -- the committed-action
  TV needs the proposer to offer >= 2 first-action classes for the gate to express a
  difference; if the CEM pool collapses to one class the gate has nothing to rank and
  JOINT==ALTERNATION trivially):
    R0 candidate first-action diversity (JOINT arm): mean distinct first-action
       classes in the CEM candidate pool per E3 tick >= 2 on >= 2/3 seeds.
    R1 compose fired (JOINT arm): n_compose_calls > 0 on >= 2/3 seeds.
    R2 coherence gate mode-distinct: JOINT coherence - SHUFFLED coherence >= 0.5 on
       >= 2/3 seeds (the gate is read mode-dependently).
  Below floor on any readiness gate -> self-route substrate_not_ready_requeue (NOT a
  verdict label).

  Discriminative (the wiring works):
    C1 TV(JOINT, ALTERNATION)            >= 0.10 on >= 2/3 seeds  [load-bearing]
    C2 TV(JOINT, SHUFFLED)               >= 0.10 on >= 2/3 seeds  [load-bearing]
    C3 TV(ALT_COH_ON, ALT_COH_OFF)       >= 0.10 on >= 2/3 seeds  [load-bearing]
       (the coherence GATING -- not packet presence -- changes behaviour;
        corroboration: TV(ALT_COH_OFF, JOINT) < TV(ALT_COH_ON, JOINT))

  PASS = R0 & R1 & R2 met AND C1 AND C2 AND C3.

Usage:
  /opt/local/bin/python3 experiments/v3_exq_661_mech294_compose_coherence_behavioural_readiness.py --dry-run
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
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

EXPERIMENT_TYPE = "v3_exq_661_mech294_compose_coherence_behavioural_readiness"
QUEUE_ID = "V3-EXQ-661"
CLAIM_IDS: List[str] = []  # substrate-readiness diagnostic (wiring validation)
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
N_EPISODES = 12
STEPS_PER_EP = 60
DRY_RUN_SEEDS = [42]
DRY_RUN_EPISODES = 2
DRY_RUN_STEPS = 12

GRID_SIZE = 8
NUM_HAZARDS = 3
NUM_RESOURCES = 3
ACTION_DIM = 4

FORCED_BENEFIT = 1.0
FORCED_DRIVE = 0.9

# Pre-registered thresholds.
TV_MARGIN = 0.10            # committed-action-distribution TV margin (memo S7.2)
DIVERSITY_FLOOR = 2.0       # mean distinct CEM first-action classes / E3 tick (R0)
COHERENCE_ORDER_MARGIN = 0.5  # joint coherence - shuffled coherence (R2)
MIN_SEEDS_2OF3 = 2

ARMS = [
    {"arm_id": "ARM_0_OFF", "packet": False, "mode": "joint", "coh": True, "compose": False},
    {"arm_id": "ARM_1_JOINT", "packet": True, "mode": "joint", "coh": True, "compose": True},
    {"arm_id": "ARM_2_ALTERNATION", "packet": True, "mode": "alternation", "coh": True, "compose": True},
    {"arm_id": "ARM_3_SHUFFLED", "packet": True, "mode": "shuffled", "coh": True, "compose": True},
    {"arm_id": "ARM_4_ALT_COH_OFF", "packet": True, "mode": "alternation", "coh": False, "compose": True},
]


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
        theta_packet_binding_mode=arm["mode"],
        theta_packet_compose_into_e3_bias=bool(arm["compose"]),
        theta_packet_compose_use_joint_coherence=bool(arm["coh"]),
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


def _candidate_first_action_classes(cands) -> Optional[int]:
    """Number of DISTINCT first-action argmax classes in the CEM candidate pool.
    None when no candidates / no actions (excluded from the diversity mean)."""
    classes = set()
    for c in cands or []:
        a = getattr(c, "actions", None)
        if a is not None and a.shape[1] > 0:
            classes.add(int(a[:, 0, :].reshape(-1).argmax().item()))
    return len(classes) if classes else None


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

    agent = _build_agent(arm, harm_dim, harm_a_dim, body_dim, world_dim, ACTION_DIM)

    action_hist = Counter()       # selected action class -> count (the behavioural readout)
    diversity_samples: List[int] = []  # distinct CEM first-action classes per E3 tick

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
                agent.update_z_goal(FORCED_BENEFIT, FORCED_DRIVE)
                ticks = agent.clock.advance()
                e1p = (agent._e1_tick(lat) if ticks["e1_tick"]
                       else torch.zeros(1, agent.config.latent.world_dim, device=agent.device))
                cands = agent.generate_trajectories(lat, e1p, ticks)
                act = agent.select_action(cands, ticks, 1.0)
            agent._step_count += 1
            # Candidate first-action diversity (the non-vacuity readiness statistic):
            # sampled at E3 ticks where the proposer refit the candidate pool.
            if ticks.get("e3_tick", False):
                nd = _candidate_first_action_classes(cands)
                if nd is not None:
                    diversity_samples.append(nd)
            ai = int(act.reshape(-1).argmax().item())
            action_hist[ai] += 1
            _flat, _r, _d, _i, obs = env.step(ai)
        print(f"  [run] arm={arm_id} seed={seed} ep {ep + 1}/{n_ep} "
              f"acts={dict(action_hist)}", flush=True)

    diag = (agent.multi_content_theta_packet.get_diagnostics()
            if agent.multi_content_theta_packet is not None else {})
    mean_div = (sum(diversity_samples) / len(diversity_samples)) if diversity_samples else 0.0
    total = sum(action_hist.values())
    norm_hist = [round(action_hist.get(a, 0) / total, 6) if total else 0.0 for a in range(ACTION_DIM)]

    return {
        "arm_id": arm_id,
        "seed": int(seed),
        "packet_on": bool(arm["packet"]),
        "binding_mode": arm["mode"],
        "compose_on": bool(arm["compose"]),
        "coherence_gating": bool(arm["coh"]),
        "action_hist": {str(a): int(action_hist.get(a, 0)) for a in range(ACTION_DIM)},
        "action_dist": norm_hist,
        "n_actions": int(total),
        "mean_cem_first_action_diversity": round(mean_div, 6),
        "n_e3_diversity_samples": len(diversity_samples),
        "n_compose_calls": int(diag.get("mech294_n_compose_calls", 0.0)),
        "last_compose_coherence": round(float(diag.get("mech294_last_compose_coherence", 0.0)), 6),
        "last_currency_coherence": round(float(diag.get("mech294_last_currency_coherence", 0.0)), 6),
        "last_compose_bias_absmax": round(float(diag.get("mech294_last_compose_bias_absmax", 0.0)), 6),
    }


def _tv(a: List[float], b: List[float]) -> float:
    """Total-variation distance between two normalised distributions."""
    n = min(len(a), len(b))
    return 0.5 * sum(abs(a[i] - b[i]) for i in range(n))


def _evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["arm_id"], {})[r["seed"]] = r
    seeds = sorted({r["seed"] for r in rows})

    def dist(arm: str, s: int) -> List[float]:
        return by.get(arm, {}).get(s, {}).get("action_dist", [0.0] * ACTION_DIM)

    def get(arm: str, s: int, k: str, default=0.0):
        return by.get(arm, {}).get(s, {}).get(k, default)

    # Readiness statistics (JOINT arm).
    div_by_seed = {s: float(get("ARM_1_JOINT", s, "mean_cem_first_action_diversity")) for s in seeds}
    compose_by_seed = {s: int(get("ARM_1_JOINT", s, "n_compose_calls")) for s in seeds}
    coh_order_by_seed = {
        s: round(float(get("ARM_1_JOINT", s, "last_currency_coherence"))
                 - float(get("ARM_3_SHUFFLED", s, "last_currency_coherence")), 6)
        for s in seeds
    }
    r0_seeds = [s for s in seeds if div_by_seed[s] >= DIVERSITY_FLOOR]
    r1_seeds = [s for s in seeds if compose_by_seed[s] > 0]
    r2_seeds = [s for s in seeds if coh_order_by_seed[s] >= COHERENCE_ORDER_MARGIN]
    r0 = len(r0_seeds) >= MIN_SEEDS_2OF3
    r1 = len(r1_seeds) >= MIN_SEEDS_2OF3
    r2 = len(r2_seeds) >= MIN_SEEDS_2OF3
    readiness_met = r0 and r1 and r2

    # Discriminative TV gates.
    c1_tv = {s: round(_tv(dist("ARM_1_JOINT", s), dist("ARM_2_ALTERNATION", s)), 6) for s in seeds}
    c2_tv = {s: round(_tv(dist("ARM_1_JOINT", s), dist("ARM_3_SHUFFLED", s)), 6) for s in seeds}
    c3_tv = {s: round(_tv(dist("ARM_2_ALTERNATION", s), dist("ARM_4_ALT_COH_OFF", s)), 6) for s in seeds}
    # Corroboration: alt-coh-OFF (full bias) should sit closer to JOINT than alt-coh-ON.
    corr = {s: bool(_tv(dist("ARM_4_ALT_COH_OFF", s), dist("ARM_1_JOINT", s))
                    < _tv(dist("ARM_2_ALTERNATION", s), dist("ARM_1_JOINT", s))) for s in seeds}

    c1_seeds = [s for s in seeds if c1_tv[s] >= TV_MARGIN]
    c2_seeds = [s for s in seeds if c2_tv[s] >= TV_MARGIN]
    c3_seeds = [s for s in seeds if c3_tv[s] >= TV_MARGIN]
    c1 = len(c1_seeds) >= MIN_SEEDS_2OF3
    c2 = len(c2_seeds) >= MIN_SEEDS_2OF3
    c3 = len(c3_seeds) >= MIN_SEEDS_2OF3
    corr_ok = sum(1 for s in seeds if corr[s]) >= MIN_SEEDS_2OF3

    # Verdict / self-route.
    if not readiness_met:
        label = "substrate_not_ready_requeue"
        if not r0:
            reading = ("CEM candidate first-action diversity below floor (R0 fail): the "
                       "proposer offers < 2 first-action classes per E3 tick on the majority "
                       "of seeds, so the coherence-gated action bias has nothing to rank and "
                       "JOINT == ALTERNATION trivially. NOT a wiring result -- re-queue with a "
                       "candidate-diversity-restoring regime (SP-CEM / larger pool). The "
                       "compose-coherence wiring is not falsified.")
        elif not r1:
            reading = ("Compose never fired on the JOINT arm (R1 fail): the packet did not "
                       "seal / the compose hook did not run. Wiring/precondition issue, not a "
                       "MECH-294 result. Re-queue after diagnosing the seal/compose path.")
        else:
            reading = ("Coherence gate not mode-distinct (R2 fail): JOINT coherence does not "
                       "exceed SHUFFLED by the margin, so the gate is not being read mode-"
                       "dependently. Substrate/V_s wiring issue, not a MECH-294 result.")
        overall = False
    elif c1 and c2 and c3:
        label = "wiring_validated_binding_reaches_behaviour"
        reading = ("PASS -- with compose ON the binding mode CHANGES the committed-action "
                   "distribution: JOINT differs from BOTH the alternation (Kay-2020) and "
                   "shuffled controls (C1, C2), AND the coherence GATING is the mechanism -- "
                   "ALTERNATION with the gate ON (weak bias) differs from the same arm with the "
                   "gate OFF / full bias (C3)"
                   + (", and alt-coh-OFF sits closer to JOINT than alt-coh-ON (corroboration)."
                      if corr_ok else
                      " (NOTE: the alt-coh-OFF-near-JOINT corroboration did not hold on >= 2/3 "
                      "seeds; the gating is load-bearing by C3 regardless).")
                   + " The compose path now carries the joint binding into E3 behaviour, so the "
                   "MECH-294 behavioural-evidence successor (claim_ids=[\"MECH-294\"]) is valid "
                   "to queue. This run does NOT weight MECH-294 confidence.")
        overall = True
    elif not c3:
        label = "coherence_gating_not_load_bearing"
        reading = ("C3 FAIL -- ALTERNATION with the coherence gate ON behaves the same as with "
                   "it OFF (full bias), so the mode-discrimination (if any in C1/C2) is carried "
                   "by mere packet presence / the action-only path, NOT the co-binding coherence "
                   "gating. The wiring does not establish that joint binding (as co-binding) "
                   "reaches behaviour. Route to /failure-autopsy; do NOT queue the behavioural "
                   "successor on the joint-specificity reading.")
        overall = False
    else:
        label = "wiring_fires_but_committed_distribution_unchanged"
        reading = ("Readiness met (compose fires, gate mode-distinct, candidates diverse) but "
                   "the committed-action distribution does not differ across binding modes by "
                   "the margin (C1 and/or C2 fail). The bias is present but sub-threshold at "
                   "this budget / bias_scale. Re-tune (longer run, larger theta_packet_bias_scale) "
                   "before the behavioural successor; not a falsification of the wiring.")
        overall = False

    return {
        "label": label,
        "reading": reading,
        "overall_pass": overall,
        "readiness": {"R0_candidate_diversity": r0, "R1_compose_fired": r1,
                      "R2_coherence_mode_distinct": r2, "readiness_met": readiness_met},
        "gate_pass": {"C1_joint_vs_alt": c1, "C2_joint_vs_shuf": c2,
                      "C3_gating_load_bearing": c3, "corroboration_alt_off_near_joint": corr_ok},
        "readiness_by_seed": {
            "joint_cem_first_action_diversity": div_by_seed,
            "joint_n_compose_calls": compose_by_seed,
            "coherence_order_joint_minus_shuf": coh_order_by_seed,
        },
        "tv_by_seed": {"C1_joint_vs_alt": c1_tv, "C2_joint_vs_shuf": c2_tv,
                       "C3_alt_on_vs_alt_off": c3_tv},
        "gate_seeds": {"R0": r0_seeds, "R1": r1_seeds, "R2": r2_seeds,
                       "C1": c1_seeds, "C2": c2_seeds, "C3": c3_seeds},
        "coherence_by_arm": {
            a: {s: round(float(get(a, s, "last_currency_coherence")), 6) for s in seeds}
            for a in ("ARM_1_JOINT", "ARM_2_ALTERNATION", "ARM_3_SHUFFLED", "ARM_4_ALT_COH_OFF")
        },
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
                    "num_resources": NUM_RESOURCES, "n_episodes": n_ep, "steps_per_ep": steps,
                    "forced_benefit": FORCED_BENEFIT, "forced_drive": FORCED_DRIVE,
                    "tv_margin": TV_MARGIN, "diversity_floor": DIVERSITY_FLOOR,
                },
                script_path=Path(__file__),
                extra_ineligible_reasons=["compose_coherence_per_cell_agent_build"],
            ) as cell:
                row = _run_cell(arm, seed, n_ep, steps)
                cell.stamp(row)
            arm_results.append(row)
            print(f"verdict: {'PASS' if row['n_actions'] > 0 else 'FAIL'}", flush=True)

    summary = _evaluate(arm_results)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    rb = summary["readiness_by_seed"]
    tv = summary["tv_by_seed"]
    min_div = min(rb["joint_cem_first_action_diversity"].values()) if rb["joint_cem_first_action_diversity"] else 0.0
    min_compose = min(rb["joint_n_compose_calls"].values()) if rb["joint_n_compose_calls"] else 0
    min_coh_order = min(rb["coherence_order_joint_minus_shuf"].values()) if rb["coherence_order_joint_minus_shuf"] else 0.0
    gp = summary["gate_pass"]

    # Diagnostic adjudication gate: readiness-kind preconditions matched to the
    # load-bearing TV criteria (the committed-action-distribution TV needs candidate
    # first-action diversity to be meaningful) + criteria_non_degenerate + load_bearing.
    interpretation = {
        "label": summary["label"],
        "reading": summary["reading"],
        "preconditions": [
            {
                "name": "joint_cem_first_action_diversity",
                "description": ("Mean distinct CEM first-action classes per E3 tick on the JOINT "
                                "arm. The compose bias is action-alignment GATED by coherence; if "
                                "the proposer offers < 2 first-action classes the gate cannot "
                                "change selection, so the committed-action-distribution TV "
                                "criteria (C1/C2/C3) are starved, not falsified. SAME multiplicity "
                                "axis the TV criteria route on."),
                "measured": float(round(min_div, 6)),
                "threshold": float(DIVERSITY_FLOOR),
                "direction": "lower",
                "control": "JOINT-arm CEM candidate pool (SP-CEM main-path proposer)",
                "met": bool(summary["readiness"]["R0_candidate_diversity"]),
            },
            {
                "name": "joint_compose_fired",
                "description": ("compose_e3_bias was invoked on the JOINT arm (packet sealed and "
                                "the compose hook ran). If zero, the packet never reached E3 and "
                                "no readout is meaningful."),
                "measured": float(min_compose),
                "threshold": 1.0,
                "direction": "lower",
                "control": "theta_packet_compose_into_e3_bias=True on JOINT",
                "met": bool(summary["readiness"]["R1_compose_fired"]),
            },
            {
                "name": "coherence_gate_mode_distinct",
                "description": ("JOINT currency_coherence minus SHUFFLED currency_coherence -- the "
                                "gate must be read mode-dependently (joint co-temporal ~1.0 vs "
                                "shuffled all-stale 0.0) for the binding mode to reach behaviour."),
                "measured": float(round(min_coh_order, 6)),
                "threshold": float(COHERENCE_ORDER_MARGIN),
                "direction": "lower",
                "control": "joint vs shuffled sealed-packet vintage currency",
                "met": bool(summary["readiness"]["R2_coherence_mode_distinct"]),
            },
        ],
        "criteria_non_degenerate": {
            "C1": bool(any(v > 0 for v in tv["C1_joint_vs_alt"].values())),
            "C2": bool(any(v > 0 for v in tv["C2_joint_vs_shuf"].values())),
            "C3": bool(any(v > 0 for v in tv["C3_alt_on_vs_alt_off"].values())),
        },
        "criteria": [
            {"name": "C1_joint_vs_alternation_committed_dist", "load_bearing": True, "passed": bool(gp["C1_joint_vs_alt"])},
            {"name": "C2_joint_vs_shuffled_committed_dist", "load_bearing": True, "passed": bool(gp["C2_joint_vs_shuf"])},
            {"name": "C3_coherence_gating_load_bearing", "load_bearing": True, "passed": bool(gp["C3_gating_load_bearing"])},
        ],
        "grid": {
            "wiring_validated_binding_reaches_behaviour": (
                "R0/R1/R2 met AND C1 AND C2 AND C3 -> the binding mode reaches E3 behaviour and "
                "the coherence gating is the mechanism. Queue the MECH-294 behavioural-evidence "
                "successor (claim_ids=[\"MECH-294\"]). Does NOT weight MECH-294 here."),
            "wiring_fires_but_committed_distribution_unchanged": (
                "Readiness met but C1/C2 sub-threshold -> bias present but too weak at this budget; "
                "re-tune (longer run / larger bias_scale) before the successor; not a falsification."),
            "coherence_gating_not_load_bearing": (
                "C3 FAIL -> alt-coh-ON == alt-coh-OFF; any C1/C2 difference is packet-presence / "
                "action-only, not co-binding coherence. /failure-autopsy; do NOT queue the "
                "joint-specificity successor."),
            "substrate_not_ready_requeue": (
                "R0/R1/R2 unmet -> candidate pool collapsed / compose inert / gate not mode-"
                "distinct. Re-queue with a diversity-restoring regime; NOT a MECH-294 result."),
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
        "outcome": outcome,
        "title": ("MECH-294 compose-coherence substrate-readiness (behavioural wiring): does the "
                  "binding mode now change the committed-action distribution; is the coherence "
                  "gating load-bearing"),
        "hypothesis": (
            "With compose ON, ThetaPacket.currency_coherence() gating makes the joint / "
            "alternation / shuffled binding modes produce distinct committed-action "
            "distributions (C1, C2), and the coherence gating -- not mere packet presence -- "
            "is the mechanism (C3: alternation gate-ON vs gate-OFF), provided the CEM proposer "
            "offers >= 2 first-action classes (R0 non-vacuity)."
        ),
        "interpretation": interpretation,
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds, "n_episodes": n_ep, "steps_per_ep": steps,
            "grid_size": GRID_SIZE, "num_hazards": NUM_HAZARDS, "num_resources": NUM_RESOURCES,
            "forced_benefit": FORCED_BENEFIT, "forced_drive": FORCED_DRIVE,
            "arms": [a["arm_id"] for a in ARMS], "compose_on": True,
            "thresholds": {
                "tv_margin": TV_MARGIN, "diversity_floor": DIVERSITY_FLOOR,
                "coherence_order_margin": COHERENCE_ORDER_MARGIN, "min_seeds_2of3": MIN_SEEDS_2OF3,
            },
        },
        "acceptance_criteria": {
            **summary["readiness"], **summary["gate_pass"], "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = out_dir / f"{run_id}.json"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome} (label={summary['label']})", flush=True)
    print(f"  readiness: {summary['readiness']}", flush=True)
    print(f"  gate_pass: {summary['gate_pass']}", flush=True)
    print(f"  tv_by_seed: {summary['tv_by_seed']}", flush=True)
    print(f"  coherence_by_arm: {summary['coherence_by_arm']}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-661 MECH-294 compose-coherence behavioural-wiring readiness"
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
