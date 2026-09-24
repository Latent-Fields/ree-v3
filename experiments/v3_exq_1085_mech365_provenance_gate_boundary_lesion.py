#!/opt/local/bin/python3
"""V3-EXQ-1085 -- MECH-365 provenance-bearing one-way commit-status gate: boundary-lesion falsifier.
SLEEP DRIVER: manual-cycle-loop (run_sleep_cycle() called once per cycle in a dedicated N_CYCLES wake-sleep-test loop)
RED-TEAM (fable, pass 1): BLOCKING -> fixed; pass 2: CLEAR (2 notes applied: cycle-1
M_img recorded; amplification mechanism corrected to the terminus READ). (1) D pools imagined writes with real-writer
writes amplified through a contaminated terminus read -> C2 now routes on the imagined-sourced
share D_img; D and D_realamp recorded as descriptives. (2) C4 selectivity cannot fail given C1a
and P3 -> demoted to a recorded structural audit (S4), F2 removed from the grid. (4) C1 split:
C1b (bit-identity) failing with C1a holding -> instrument_invalid, not weakens. P5 hardened to
center POSITIONS, not count. Pass 2 verdict: see queue entry note.

Claim under test (MECH-365, V3 half after the 2026-09-24 split): imagined /
simulated event representations carry an explicit committed_vs_imagined
provenance label, and a ONE-WAY gate lets them be used (presented, read,
scored) without ever acquiring committed-history status. In V3 the label is
Trajectory.hypothesis_tag, carried ON the representation object; the
"committed history" analogue is the ResidueField VALENCE_WANTING terrain that
later reads as experienced value. WWA: REE_assembly claims.yaml MECH-365
(merged e6424df604); design: REE_assembly/evidence/planning/
mech365_wwa_and_lesion_design_20260924.md.

The boundary. The only V3 replay -> consolidation writer is MECH-217's
HippocampalModule.spread_reverse_replay_wanting(), called from
REEAgent.run_rem_attribution_pass(). As shipped, only REVERSE (real, recorded)
replay reaches it; FORWARD replay (an E2 rollout of random actions = imagined
content) is scored read-only. So as shipped imagined content is kept out of
committed history by ROUTING EXCLUSION, not by a label. This experiment uses
three default-OFF knobs landed with it (ree-v3, HippocampalConfig):
  rem_route_forward_replay_to_consolidation -- present forward (imagined)
      replay to the SAME writer, so only the label stands between it and a
      committed write;
  mech365_provenance_lesion -- "drop_at_consolidation" drops the label in the
      TRANSLATION into update_valence while the sender Trajectory keeps it
      (a boundary lesion, not a source lesion); "sham_real_only" runs the same
      override assignment on untagged (real) reverse trajectories, a no-op;
  mech365_suppress_replay_provenance_stamp -- reproduces the pre-2026-09-24
      defect (replay() output left untagged) as a SOURCE-side canary.

Arms (5; same seeds; the spread consumes no RNG, so arms differ only through
what the writer accepts):
  R0_unrouted        routing OFF (as shipped)          -- reference committed-history map
  A1_gate_intact     routing ON, label stamped, off    -- the claim's gate
  A2_boundary_lesion routing ON, drop_at_consolidation -- lesion
  A3_sham            routing ON, sham_real_only        -- matched negative control
  A4_canary          routing ON, stamp SUPPRESSED      -- P4 positive control: the
                     instrument must detect a label failure at source

DVs (per seed):
  M_img  = summed |spread| ACCEPTED by the writer from forward (imagined)
           trajectories (rem_fwd_spread_accepted_mass), gate side.
  D_img  = M_img(A2) / ||W_A1||_1 -- the IMAGINED-SOURCED committed mass relative
           to the intact committed map. The routed C2 DV.
  D(X)   = relative L1 divergence of the end-of-run VALENCE_WANTING vector over
           the active RBF centers, ||W_X - W_A1||_1 / ||W_A1||_1 (C1b, C3). For A2 it
           is DESCRIPTIVE only: it also contains D_realamp = (real reverse-writer
           mass A2 - A1) / ||W_A1||_1, because forward replay starts at the
           wanting-seeded terminus, imagined waypoints write onto nearby active
           centers, and the kernel-weighted terminus READ (evaluate_valence) of
           later spreads -- real and imagined -- then sees an inflated value; the
           contamination self-amplifies across cycles to a seed-dependent degree.
           (Measured by the red-team: imagined states do not snap onto the
           terminus center itself; the loop closes through the READ.) D_img's
           magnitude is loop-driven for the same reason, so cycle-1 M_img(A2) --
           a single crossing, ~config-constant -- is recorded as its descriptive.
           The waking path is scripted and identical across arms, so the active
           centers are identical at a matched seed (checked by position, P5).

Preconditions (unmet -> substrate_not_ready_requeue, never a verdict):
  P1 writer live: R0 total rem_wanting_spread_n_steps > 0 AND A1 reverse accepted
     writes > 0, every seed.
  P2 routing live: forward trajectories presented to the writer >= 1 per REM pass
     (total >= N_CYCLES) in every routed cell.
  P3 gate actually reached: >= 3 forward trajectories per seed with start-state
     wanting > 0 (rem_fwd_spread_n_reached_gate) in A1, A2 and A4.
  P4 canary: M_img(A4) > 0 every seed.
  P5 center set matched: the active-center POSITIONS are identical (max abs diff 0)
     across all 5 arms at every seed (paired per-center comparison valid).
  Note: valence_bounding_enabled is False at this config (no clamp), so the
  clamp-saturation precondition from the design is structurally met and is
  recorded as a diagnostic, not a gate.

Pre-registered criteria (load-bearing; PASS = P1..P5 AND C1a AND C1b AND C2 AND C3):
  C1a gate holds: every seed M_img(A1) == 0.
  C1b committed map unchanged by routing: every seed D(R0 vs A1) <= 1e-6.
  C2 lesion contaminates: every seed M_img(A2) > 0 AND D_img(A2) > theta,
     theta = max(3 * SD_seeds(D(A3)), 0.05).
  C3 sham inert (control): every seed D(A3) <= 0.01.
  S4 (recorded, NOT gated -- structural): rem_n_rollouts(A1) == R0 > 0,
     |sum rem_mean_harm_terrain A1 - R0| <= 1e-6, rem_fwd_n_scored(A1) > 0 and
     rem_fwd_spread_n_refused(A1) > 0. Given C1a and P3 this cannot fail (scoring
     precedes routing; no code couples the label to the read path), so the
     "gate, not discard" leg is a source-audit fact, not an experimental result.
Routing on failure: C3 fail, or C1b fail with C1a holding -> instrument_invalid
(no verdict, non_degenerate false); C1a fail -> F1 label_does_not_travel
(weakens); C2 fail in >= 3 of 5 seeds -> F3 label_not_load_bearing (weakens);
C2 fail in 1-2 seeds -> mixed.

Scope, stated so it is not over-read: claim_ids = [MECH-365] ONLY. This is ONE
relation (reality-status) lesioned at ONE boundary with ONE predicted
signature. It is NOT MECH-545's five-way contract-lesion assay (no identity /
temporal / agency / confidence lesion, no random-projection floor, no
acute/adapted separation, no contract-admissibility verdict) and NOT MECH-271's
differential-routing test (one destination). C1, C3 and S4 are structurally
determined once the label is stamped and honoured, and C2 is near-guaranteed
given P3 -- this is AUDIT-STRENGTH evidence (like INV-011's). The informative
content is the canary (does the instrument see a label failure at source) and
the recorded magnitudes (D_img, D_realamp).

Harness: derived from V3-EXQ-842 (MECH-217 writer demonstrated live, PASS
2026-07-30): scripted epsilon-greedy walk to a fixed resource, terminus
VALENCE_WANTING seeded directly on cycle 1 only (MECH-203 off), near-zero RBF
centers at visited waypoints, kernel_bandwidth 0.03, num_basis_functions 256.
Sleep runs BEFORE agent.reset() so theta_buffer is populated and REM forward
replay starts at the (wanting-seeded) contact terminus.

experiment_purpose = "evidence".
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.residue.field import VALENCE_WANTING  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1085_mech365_provenance_gate_boundary_lesion"
QUEUE_ID = "V3-EXQ-1085"
CLAIM_IDS = ["MECH-365"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = (42, 43, 44, 45, 46)
# arm -> (route_forward, lesion, suppress_stamp)
ARMS = {
    "R0_unrouted": (False, "off", False),
    "A1_gate_intact": (True, "off", False),
    "A2_boundary_lesion": (True, "drop_at_consolidation", False),
    "A3_sham": (True, "sham_real_only", False),
    "A4_canary": (True, "off", True),
}

GRID_SIZE = 8
AGENT_START = (1, 1)
RESOURCE_POS = (6, 6)
CONTROL_POS = (1, 6)

N_CYCLES = 6
MAX_STEPS_PER_EPISODE = 25
EPSILON = 0.1
SWS_CONSOLIDATION_STEPS = 5
REM_ATTRIBUTION_STEPS = 10   # 5 forward + 5 reverse per REM pass
WAYPOINT_CENTER_SEED = 1e-3
CONTACT_SEED_WANTING = 2.0
NUM_BASIS_FUNCTIONS = 256
KERNEL_BANDWIDTH = 0.03

# Pre-registered thresholds
P3_MIN_REACHED_GATE = 3
C1_BIT_IDENTITY_TOL = 1e-6
C2_ABS_FLOOR = 0.05
C2_SD_MULT = 3.0
C3_SHAM_CEIL = 0.01
C4_SCORE_TOL = 1e-6
F3_MAJORITY = 3

_ZG = ZGoalStreamAccumulator()


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=0,
        num_resources=1,
        resource_benefit=1.0,
        proximity_harm_scale=0.0,
        proximity_benefit_scale=0.0,
        env_drift_prob=0.0,
        use_proxy_fields=False,
        resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2, seed: int, route: bool, lesion: str,
                suppress: bool) -> REEAgent:
    torch.manual_seed(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        replay_diversity_enabled=True,
        sws_enabled=True,
        sws_consolidation_steps=SWS_CONSOLIDATION_STEPS,
        rem_enabled=True,
        rem_attribution_steps=REM_ATTRIBUTION_STEPS,
        use_offline_wanting_spread=True,
        rem_route_forward_replay_to_consolidation=route,
        mech365_provenance_lesion=lesion,
        mech365_suppress_replay_provenance_stamp=suppress,
    )
    cfg.residue.num_basis_functions = NUM_BASIS_FUNCTIONS
    cfg.residue.kernel_bandwidth = KERNEL_BANDWIDTH  # see V3-EXQ-842 for the measured-scale rationale
    # from_dims reachability assert: the three MECH-365 knobs must land.
    assert cfg.hippocampal.rem_route_forward_replay_to_consolidation is bool(route)
    assert cfg.hippocampal.mech365_provenance_lesion == lesion
    assert cfg.hippocampal.mech365_suppress_replay_provenance_stamp is bool(suppress)
    assert cfg.hippocampal.use_offline_wanting_spread is True
    return REEAgent(cfg)


def _one_hot_action(action_idx: int, action_dim: int) -> torch.Tensor:
    action = torch.zeros(1, action_dim)
    action[0, int(action_idx)] = 1.0
    return action


def _greedy_action_toward(env, target, epsilon: float, rng: torch.Generator) -> int:
    if torch.rand(1, generator=rng).item() < epsilon:
        return int(torch.randint(0, 4, (1,), generator=rng).item())
    ax, ay = env.agent_x, env.agent_y
    tx, ty = target
    dx, dy = tx - ax, ty - ay
    if dx == 0 and dy == 0:
        return 4
    if abs(dx) >= abs(dy) and dx != 0:
        return 0 if dx < 0 else 1
    return 2 if dy < 0 else 3


def _sense_tick(agent: REEAgent, obs_dict: dict):
    latent = agent.sense(
        obs_dict["body_state"],
        obs_dict["world_state"],
        obs_harm=obs_dict.get("harm_obs"),
        obs_harm_a=obs_dict.get("harm_obs_a"),
        obs_harm_history=obs_dict.get("harm_history"),
    )
    ticks = agent.clock.advance()
    if ticks.get("e1_tick", False):
        agent._e1_tick(latent)
    return latent


_SUM_KEYS = (
    "rem_n_rollouts", "rem_n_reverse", "rem_wanting_spread_n_steps",
    "rem_fwd_n_scored", "rem_fwd_spread_n_presented", "rem_fwd_spread_n_reached_gate",
    "rem_fwd_spread_n_accepted", "rem_fwd_spread_n_refused", "rem_fwd_spread_accepted_mass",
    "rem_fwd_spread_n_lesion_overrides", "rem_rev_spread_n_accepted",
    "rem_rev_spread_accepted_mass", "rem_rev_spread_n_sham_overrides",
)


def run_cycle(agent: REEAgent, env, rng: torch.Generator, seed_wanting: bool) -> Dict:
    _flat, obs_dict = env.reset_to(
        agent_pos=AGENT_START, hazard_positions=[], resource_positions=[RESOURCE_POS]
    )
    agent.reset()
    agent.e1.reset_hidden_state()
    contact = False
    n_steps = 0
    for _ in range(MAX_STEPS_PER_EPISODE):
        latent = _sense_tick(agent, obs_dict)
        agent.residue_field.rbf_field.add_residue(latent.z_world, WAYPOINT_CENTER_SEED)
        action_idx = _greedy_action_toward(env, RESOURCE_POS, EPSILON, rng)
        action = _one_hot_action(action_idx, env.action_dim)
        agent._record_exploration_action(action)
        _flat, harm, done, _info, obs_dict = env.step(action)
        n_steps += 1
        if harm > 0.0:
            contact = True
            break
        if done:
            break
    if contact:
        latent = _sense_tick(agent, obs_dict)
        agent.residue_field.rbf_field.add_residue(latent.z_world, WAYPOINT_CENTER_SEED)
        if seed_wanting:
            agent.residue_field.update_valence(
                latent.z_world, VALENCE_WANTING, CONTACT_SEED_WANTING, hypothesis_tag=False
            )
        agent._record_exploration_action(_one_hot_action(4, env.action_dim))
    sleep_metrics = agent.run_sleep_cycle()
    agent.reset()
    row = {"contact": contact, "n_steps": n_steps}
    for k in _SUM_KEYS:
        row[k] = float(sleep_metrics.get(k, 0.0))
    row["rem_mean_harm_terrain"] = float(sleep_metrics.get("rem_mean_harm_terrain", 0.0))
    return row


def _wanting_vector(agent: REEAgent) -> List[float]:
    rbf = agent.residue_field.rbf_field
    mask = rbf.active_mask.bool()
    vals = rbf.valence_vecs[mask][:, VALENCE_WANTING]
    return [float(v) for v in vals.detach().cpu().tolist()]


def _read_wanting(agent: REEAgent, z_world: torch.Tensor) -> float:
    with torch.no_grad():
        valence = agent.residue_field.evaluate_valence(z_world)
    return float(valence[..., VALENCE_WANTING].mean().item())


def run_cell(arm: str, seed: int, n_cycles: int) -> Dict:
    route, lesion, suppress = ARMS[arm]
    env = _make_env(seed)
    agent = _make_agent(env, seed, route, lesion, suppress)
    rng = torch.Generator().manual_seed(seed)
    print(f"Seed {seed} Condition {arm}", flush=True)
    cycles: List[Dict] = []
    for c in range(n_cycles):
        print(f"  [train] {arm} seed={seed} ep {c + 1}/{n_cycles}", flush=True)
        cycles.append(run_cycle(agent, env, rng, seed_wanting=(c == 0)))
    _ZG.observe(agent)

    totals = {k: sum(cy[k] for cy in cycles) for k in _SUM_KEYS}
    totals["rem_mean_harm_terrain_sum"] = sum(cy["rem_mean_harm_terrain"] for cy in cycles)
    wvec = _wanting_vector(agent)
    _rbf = agent.residue_field.rbf_field
    centers = _rbf.centers.detach()[_rbf.active_mask.bool()].clone()

    # Descriptive MECH-217-shape readouts (842's near/far/control).
    near_vals, far_vals = [], []
    for traj in agent.hippocampal._exploration_buffer:
        ws = traj.world_states
        if ws is None or len(ws) < 2:
            continue
        near_vals.append(_read_wanting(agent, ws[-2]))
        far_vals.append(_read_wanting(agent, ws[0]))
    _flat, control_obs = env.reset_to(
        agent_pos=CONTROL_POS, hazard_positions=[], resource_positions=[RESOURCE_POS]
    )
    control_latent = _sense_tick(agent, control_obs)
    control_wanting = _read_wanting(agent, control_latent.z_world)
    agent.reset()

    cell_ready = bool(totals["rem_n_rollouts"] > 0 and len(wvec) > 0)
    print(f"verdict: {'PASS' if cell_ready else 'FAIL'}", flush=True)
    return {
        "arm": arm,
        "seed": seed,
        "rem_route_forward_replay_to_consolidation": route,
        "mech365_provenance_lesion": lesion,
        "mech365_suppress_replay_provenance_stamp": suppress,
        "n_cycles": n_cycles,
        "n_contacts": sum(1 for cy in cycles if cy["contact"]),
        "buffer_len": len(agent.hippocampal._exploration_buffer),
        "totals": totals,
        "n_active_centers": len(wvec),
        "wanting_vector": wvec,
        "_centers": centers,
        "wanting_l1": float(sum(abs(v) for v in wvec)),
        "wanting_max_abs": float(max((abs(v) for v in wvec), default=0.0)),
        "near_wanting_mean": (sum(near_vals) / len(near_vals)) if near_vals else 0.0,
        "far_wanting_mean": (sum(far_vals) / len(far_vals)) if far_vals else 0.0,
        "control_wanting": control_wanting,
        "valence_bounding_enabled": bool(
            getattr(agent.residue_field.config, "valence_bounding_enabled", False)
        ),
        "cycle_results": cycles,
        "cell_ready": cell_ready,
    }


def _rel_l1(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) != len(b) or len(b) == 0:
        return None
    den = sum(abs(x) for x in b)
    if den <= 0.0:
        return None
    return sum(abs(x - y) for x, y in zip(a, b)) / den


def _sd(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def evaluate(cells: List[Dict], seeds) -> Dict:
    by = {(c["arm"], c["seed"]): c for c in cells}
    per_seed: List[Dict] = []
    for s in seeds:
        r0, a1, a2, a3, a4 = (by[(a, s)] for a in ARMS)
        d_r0 = _rel_l1(r0["wanting_vector"], a1["wanting_vector"])
        d_a2 = _rel_l1(a2["wanting_vector"], a1["wanting_vector"])
        d_a3 = _rel_l1(a3["wanting_vector"], a1["wanting_vector"])
        d_a4 = _rel_l1(a4["wanting_vector"], a1["wanting_vector"])
        l1_a1 = a1["wanting_l1"]
        cmax = 0.0
        for a in ARMS:
            ca, c1_ = by[(a, s)]["_centers"], a1["_centers"]
            if ca.shape != c1_.shape:
                cmax = float("inf")
                break
            if ca.numel():
                cmax = max(cmax, float((ca - c1_).abs().max().item()))
        per_seed.append({
            "seed": s,
            "center_pos_maxabs_diff_vs_A1": cmax,
            "D_img_A2": (a2["totals"]["rem_fwd_spread_accepted_mass"] / l1_a1) if l1_a1 > 0 else None,
            "D_realamp_A2": ((a2["totals"]["rem_rev_spread_accepted_mass"]
                              - a1["totals"]["rem_rev_spread_accepted_mass"]) / l1_a1)
                            if l1_a1 > 0 else None,
            "D_R0_vs_A1": d_r0, "D_A2": d_a2, "D_A3": d_a3, "D_A4": d_a4,
            "M_img_A1": a1["totals"]["rem_fwd_spread_accepted_mass"],
            "M_img_A2": a2["totals"]["rem_fwd_spread_accepted_mass"],
            "M_img_A3": a3["totals"]["rem_fwd_spread_accepted_mass"],
            "M_img_A4": a4["totals"]["rem_fwd_spread_accepted_mass"],
            "M_img_A2_cycle1": a2["cycle_results"][0]["rem_fwd_spread_accepted_mass"],
            "n_centers": {a: by[(a, s)]["n_active_centers"] for a in ARMS},
            "R0_spread_steps": r0["totals"]["rem_wanting_spread_n_steps"],
            "A1_rev_accepted": a1["totals"]["rem_rev_spread_n_accepted"],
            "presented_min_routed": min(by[(a, s)]["totals"]["rem_fwd_spread_n_presented"]
                                        for a in ARMS if ARMS[a][0]),
            "reached_gate_min": min(by[(a, s)]["totals"]["rem_fwd_spread_n_reached_gate"]
                                    for a in ("A1_gate_intact", "A2_boundary_lesion", "A4_canary")),
            "rollouts_A1": a1["totals"]["rem_n_rollouts"],
            "rollouts_R0": r0["totals"]["rem_n_rollouts"],
            "terrain_absdiff_A1_R0": abs(a1["totals"]["rem_mean_harm_terrain_sum"]
                                         - r0["totals"]["rem_mean_harm_terrain_sum"]),
            "fwd_scored_A1": a1["totals"]["rem_fwd_n_scored"],
            "fwd_refused_A1": a1["totals"]["rem_fwd_spread_n_refused"],
            "lesion_overrides_A2": a2["totals"]["rem_fwd_spread_n_lesion_overrides"],
            "sham_overrides_A3": a3["totals"]["rem_rev_spread_n_sham_overrides"],
        })

    def _worst(key, fn=min):
        vals = [(p[key], p["seed"]) for p in per_seed]
        v, sd = fn(vals, key=lambda t: t[0])
        return v, sd

    # ---- preconditions (worst cell reported; floors) ----
    p1_r0, p1_r0_seed = _worst("R0_spread_steps")
    p1_a1, p1_a1_seed = _worst("A1_rev_accepted")
    p2, p2_seed = _worst("presented_min_routed")
    p3, p3_seed = _worst("reached_gate_min")
    p4, p4_seed = _worst("M_img_A4")
    center_mismatch = sum(1 for p in per_seed
                          if len(set(p["n_centers"].values())) != 1
                          or p["center_pos_maxabs_diff_vs_A1"] != 0.0)
    n_cyc = cells[0]["n_cycles"]
    preconditions = [
        {"name": "P1a_writer_live_R0_spread_steps", "measured": p1_r0, "threshold": 0.0,
         "comparator": ">", "offending_cell": p1_r0_seed,
         "control": "MECH-217 reverse spread on real replay (V3-EXQ-842 positive control)",
         "met": bool(p1_r0 > 0.0)},
        {"name": "P1b_writer_live_A1_reverse_accepted", "measured": p1_a1, "threshold": 0.0,
         "comparator": ">", "offending_cell": p1_a1_seed, "met": bool(p1_a1 > 0.0)},
        {"name": "P2_forward_presented_to_writer", "measured": p2, "threshold": float(n_cyc),
         "direction": "lower", "offending_cell": p2_seed, "met": bool(p2 >= n_cyc)},
        {"name": "P3_forward_reached_gate", "measured": p3, "threshold": float(P3_MIN_REACHED_GATE),
         "direction": "lower", "offending_cell": p3_seed, "met": bool(p3 >= P3_MIN_REACHED_GATE)},
        {"name": "P4_canary_detects_source_label_loss", "measured": p4, "threshold": 0.0,
         "comparator": ">", "offending_cell": p4_seed,
         "control": "A4: replay() stamp suppressed = the pre-2026-09-24 as-built defect",
         "met": bool(p4 > 0.0)},
        {"name": "P5_center_set_matched_across_arms", "measured": float(center_mismatch),
         "threshold": 0.0, "direction": "upper", "met": bool(center_mismatch == 0)},
    ]
    pre_ok = all(p["met"] for p in preconditions)

    d_a3_vals = [p["D_A3"] for p in per_seed if p["D_A3"] is not None]
    theta = max(C2_SD_MULT * _sd(d_a3_vals), C2_ABS_FLOOR)

    def _ok(v, pred):
        return v is not None and pred(v)

    # C1 split (red-team finding 4): C1a is the gate (imagined content accepted
    # with the label intact = F1); C1b is bit-identity of the committed map to
    # the unrouted substrate -- failing C1b while C1a holds is a side-effect /
    # nondeterminism signature, routed with C3 to instrument_invalid.
    c1a_seed = [p["M_img_A1"] == 0.0 for p in per_seed]
    c1b_seed = [_ok(p["D_R0_vs_A1"], lambda v: v <= C1_BIT_IDENTITY_TOL) for p in per_seed]
    # C2 reads the IMAGINED-SOURCED share only (red-team finding 1): D_A2 pools
    # imagined writes with real-writer writes amplified through a contaminated
    # terminus read, so its magnitude is descriptive, never the routed DV.
    c2_seed = [p["M_img_A2"] > 0.0 and _ok(p["D_img_A2"], lambda v: v > theta) for p in per_seed]
    c3_seed = [_ok(p["D_A3"], lambda v: v <= C3_SHAM_CEIL) for p in per_seed]
    c4_seed = [
        p["rollouts_A1"] == p["rollouts_R0"] and p["rollouts_A1"] > 0
        and p["terrain_absdiff_A1_R0"] <= C4_SCORE_TOL
        and p["fwd_scored_A1"] > 0 and p["fwd_refused_A1"] > 0
        for p in per_seed
    ]
    c1a, c1b = all(c1a_seed), all(c1b_seed)
    c1 = c1a and c1b
    c2, c3, c4 = all(c2_seed), all(c3_seed), all(c4_seed)
    n_c2_fail = sum(1 for x in c2_seed if not x)

    def _mx(key):
        vals = [p[key] for p in per_seed if p[key] is not None]
        return max(vals) if vals else None

    def _mn(key):
        vals = [p[key] for p in per_seed if p[key] is not None]
        return min(vals) if vals else None

    criteria = [
        {"name": "C1a_gate_holds_Mimg_A1_zero", "load_bearing": True,
         "passed": c1a, "measured": _mx("M_img_A1"), "threshold": 0.0,
         "comparator": "<=", "n_seeds_passed": sum(c1a_seed), "seeds_required": len(per_seed)},
        {"name": "C1b_committed_map_bit_identical_R0_vs_A1", "load_bearing": True,
         "passed": c1b, "measured": _mx("D_R0_vs_A1"), "threshold": C1_BIT_IDENTITY_TOL,
         "comparator": "<=", "n_seeds_passed": sum(c1b_seed), "seeds_required": len(per_seed)},
        {"name": "C2_boundary_lesion_contaminates_imagined_sourced", "load_bearing": True,
         "passed": c2, "measured": _mn("D_img_A2"), "threshold": theta,
         "threshold_floor": C2_ABS_FLOOR, "threshold_sd_mult": C2_SD_MULT,
         "measured_M_img_A2_min": _mn("M_img_A2"),
         "descriptive_D_total_A2_min": _mn("D_A2"),
         "descriptive_D_realamp_A2_min": _mn("D_realamp_A2"),
         "n_seeds_passed": sum(c2_seed), "seeds_required": len(per_seed)},
        {"name": "C3_sham_inert", "load_bearing": True, "passed": c3,
         "measured": _mx("D_A3"), "threshold": C3_SHAM_CEIL,
         "n_seeds_passed": sum(c3_seed), "seeds_required": len(per_seed)},
        {"name": "S4_selectivity_structural_audit", "load_bearing": False,
         "structural": True,
         "structural_note": ("red-team (fable) finding 2, verified: given C1a and P3 this cannot "
                             "fail -- scoring runs before routing in run_rem_attribution_pass and "
                             "no ree_core code couples the label to the read path. Recorded as a "
                             "source-audit fact (gate, not discard), NOT an experimental criterion."),
         "passed": c4, "measured": _mx("terrain_absdiff_A1_R0"), "threshold": C4_SCORE_TOL,
         "measured_fwd_refused_A1_min": _mn("fwd_refused_A1"), "threshold_fwd_refused_A1": 0.0,
         "n_seeds_passed": sum(c4_seed), "seeds_required": len(per_seed)},
    ]
    combination_rule = ("PASS = P1..P5 all met AND C1a AND C1b AND C2 AND C3 (each every seed); "
                        "S4 is a recorded structural audit, not gated. "
                        "C3 fail or C1b fail -> instrument_invalid (no verdict); "
                        "C1a fail -> F1 weakens; C2 fail in >=3 seeds -> F3 weakens; "
                        "C2 fail in 1-2 seeds -> mixed.")

    non_degenerate = True
    degeneracy_reason = None
    if not pre_ok:
        label, outcome, direction = "substrate_not_ready_requeue", "FAIL", "unknown"
        non_degenerate = False
        degeneracy_reason = "precondition(s) unmet: " + ",".join(
            p["name"] for p in preconditions if not p["met"])
    elif not c3 or (c1a and not c1b):
        label, outcome, direction = "instrument_invalid_nondeterminism", "FAIL", "unknown"
        non_degenerate = False
        degeneracy_reason = ("sham control moved the committed-history map, or the gate-intact map "
                             "differs from the unrouted map with zero imagined writes accepted "
                             "(instrument nondeterminism / side effect)")
    elif not c1a:
        label, outcome, direction = "F1_label_does_not_travel_gate_not_one_way", "FAIL", "weakens"
    elif not c2 and n_c2_fail >= F3_MAJORITY:
        label, outcome, direction = "F3_label_not_load_bearing", "FAIL", "weakens"
    elif not c2:
        label, outcome, direction = "C2_partial_contamination", "FAIL", "mixed"
    else:
        label, outcome, direction = "mech365_gate_holds_and_boundary_lesion_contaminates", "PASS", "supports"

    return {
        "label": label,
        "outcome": outcome,
        "evidence_direction": direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "preconditions": preconditions,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "criteria_non_degenerate": {
            "C1a": bool(p4 > 0.0),   # the canary shows the instrument CAN see a nonzero M_img
            "C1b": True,
            "C2": bool(len(d_a3_vals) == len(per_seed)),
            "C3": bool(_mn("sham_overrides_A3") is not None and _mn("sham_overrides_A3") > 0),
            "S4": False,  # structural: cannot fail given C1a and P3 (red-team finding 2)
        },
        "theta_C2": theta,
        "per_seed": per_seed,
    }


def run_experiment(dry_run: bool = False):
    seeds = (SEEDS[0],) if dry_run else SEEDS
    n_cycles = 3 if dry_run else N_CYCLES
    t0 = time.perf_counter()
    cells: List[Dict] = []
    for arm, (route, lesion, suppress) in ARMS.items():
        for seed in seeds:
            slice_ = {
                "arm": arm, "rem_route_forward_replay_to_consolidation": route,
                "mech365_provenance_lesion": lesion,
                "mech365_suppress_replay_provenance_stamp": suppress,
                "use_offline_wanting_spread": True, "grid_size": GRID_SIZE,
                "agent_start": AGENT_START, "resource_pos": RESOURCE_POS,
                "n_cycles": n_cycles, "rem_attribution_steps": REM_ATTRIBUTION_STEPS,
                "sws_consolidation_steps": SWS_CONSOLIDATION_STEPS,
                "num_basis_functions": NUM_BASIS_FUNCTIONS,
                "kernel_bandwidth": KERNEL_BANDWIDTH,
                "contact_seed_wanting": CONTACT_SEED_WANTING,
            }
            with arm_cell(seed, config_slice=slice_, script_path=Path(__file__),
                          config_slice_declared=True) as cell:
                row = run_cell(arm, seed, n_cycles)
                cell.stamp(row)
            cells.append(row)

    elapsed = time.perf_counter() - t0
    ev = evaluate(cells, seeds)
    for c in cells:
        c.pop("_centers", None)
    outcome = ev["outcome"]
    print(f"{QUEUE_ID} MECH-365 provenance-gate boundary lesion -- {outcome} ({ev['label']}) "
          f"in {elapsed:.1f}s", flush=True)
    for p in ev["preconditions"]:
        print(f"  pre {p['name']}: measured={p['measured']} thr={p['threshold']} met={p['met']}",
              flush=True)
    for c in ev["criteria"]:
        print(f"  crit {c['name']}: passed={c['passed']} measured={c['measured']} "
              f"thr={c['threshold']}", flush=True)
    for p in ev["per_seed"]:
        print(f"  seed {p['seed']}: D_R0={p['D_R0_vs_A1']} D_img_A2={p['D_img_A2']} "
              f"D_realamp_A2={p['D_realamp_A2']} D_A2={p['D_A2']} D_A3={p['D_A3']} "
              f"D_A4={p['D_A4']} Mimg A1={p['M_img_A1']:.5f} A2={p['M_img_A2']:.5f} "
              f"A4={p['M_img_A4']:.5f} reached={p['reached_gate_min']}", flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    def _num(v):
        if v is None or isinstance(v, bool):
            return None if v is None else int(v)
        v = float(v)
        return v if math.isfinite(v) else None

    readout = {}
    for p in ev["per_seed"]:
        for k in ("D_R0_vs_A1", "D_img_A2", "D_realamp_A2", "D_A2", "D_A3", "D_A4",
                  "M_img_A1", "M_img_A2", "M_img_A2_cycle1", "M_img_A4"):
            val = _num(p[k])
            if val is not None:
                readout[f"{k}_seed{p['seed']}"] = val
    readout["theta_C2"] = _num(ev["theta_C2"])
    for c in ev["criteria"]:
        readout[f"{c['name'].split('_')[0]}_passed"] = int(bool(c["passed"]))
    readout["overall_pass"] = int(outcome == "PASS")
    readout = {k: v for k, v in readout.items() if v is not None}

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "evidence_direction": ev["evidence_direction"],
        "non_degenerate": ev["non_degenerate"],
        "degeneracy_reason": ev["degeneracy_reason"],
        "interpretation": {
            "label": ev["label"],
            "preconditions": ev["preconditions"],
            "criteria_non_degenerate": ev["criteria_non_degenerate"],
            "combination_rule": ev["combination_rule"],
        },
        "criteria": ev["criteria"],
        "combination_rule": ev["combination_rule"],
        "per_seed": ev["per_seed"],
        "readout": readout,
        "arm_results": cells,
        "sleep_driver_pattern": "manual-cycle-loop",
        "registered_thresholds": {
            "P3_MIN_REACHED_GATE": P3_MIN_REACHED_GATE,
            "C1_BIT_IDENTITY_TOL": C1_BIT_IDENTITY_TOL,
            "C2_ABS_FLOOR": C2_ABS_FLOOR, "C2_SD_MULT": C2_SD_MULT,
            "C3_SHAM_CEIL": C3_SHAM_CEIL, "C4_SCORE_TOL": C4_SCORE_TOL,
            "F3_MAJORITY": F3_MAJORITY,
        },
        "scope_note": (
            "Single-relation (reality-status), single-boundary lesion for MECH-365 only. "
            "Not MECH-545's five-way contract-lesion assay; not MECH-271's routing test. "
            "C1/C3 are structurally determined once the label is stamped and honoured "
            "(audit-strength); the canary A4, selectivity C4 and the magnitude of D carry "
            "the information."
        ),
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False, "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False, "decision": "allow",
        },
    }
    full_config = {
        "seeds": list(seeds), "n_cycles": n_cycles, "arms": {k: list(v) for k, v in ARMS.items()},
        "grid_size": GRID_SIZE, "agent_start": AGENT_START, "resource_pos": RESOURCE_POS,
        "control_pos": CONTROL_POS, "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
        "epsilon": EPSILON, "sws_consolidation_steps": SWS_CONSOLIDATION_STEPS,
        "rem_attribution_steps": REM_ATTRIBUTION_STEPS,
        "waypoint_center_seed": WAYPOINT_CENTER_SEED,
        "contact_seed_wanting": CONTACT_SEED_WANTING,
        "num_basis_functions": NUM_BASIS_FUNCTIONS, "kernel_bandwidth": KERNEL_BANDWIDTH,
        "alpha_world": 0.9, "alpha_self": 0.3, "replay_diversity_enabled": True,
        "use_offline_wanting_spread": True,
    }
    out_path = write_flat_manifest(
        manifest, dry_run=dry_run, config=full_config, seeds=list(seeds),
        script_path=Path(__file__), started_at=t0, z_goal_stream_stats=_ZG.stats(),
    )
    print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run: 1 seed, 3 cycles.")
    args = parser.parse_args()
    _outcome, _manifest_path = run_experiment(dry_run=args.dry_run)
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_manifest_path) if _manifest_path is not None else None,
        dry_run=args.dry_run,
    )
    sys.exit(0)
