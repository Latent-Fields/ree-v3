"""
V3-EXQ-1061 -- MECH-131

!! NOT QUEUEABLE AS WRITTEN -- MEASURED BLOCKER, 2026-09-19. READ THIS FIRST. !!

This driver is complete and its dry-run passes end to end (9/9 cells, all
preconditions met, C1 and C2 met). It must NOT be queued yet, because the DV was
measured on the hub and CANNOT DETECT ITS OWN MANIPULATION on an untrained
substrate:

    between-candidate residue SD        0.18%  of the pool mean
    intact vs complete-lesion effect   0.002% of the pool mean   (~100x smaller)

and the ratio is scale-invariant -- it does not improve with more accumulated
residue (measured at 60 and 200 charge steps: 31.40 +/- 0.056 with a 0.0005
lesion effect; 99.48 +/- 0.181 with a 0.0018 effect). All 32 candidates land in
essentially the same residue region, so residue-based selection has nothing to
exploit and C1's per-seed ordinal test would be deciding on differences ~100x
below candidate-level noise: a coin flip wearing a criterion's clothes.

ROOT CAUSE, and it is the codebase's own documented expectation rather than a new
finding -- V3-EXQ-042 (hippocampal terrain training) says it outright:
"if terrain_prior is random, proposals are uninformed (equivalent to random
candidates)". Residue avoidance is a LEARNED competence here; this driver runs an
UNTRAINED agent, so lesioning the anticipatory channel removes a capability the
substrate never had. V3-EXQ-042's own eval metric is this driver's DV1 almost
verbatim ("hippo_quality_gap = mean_residue_random - mean_residue_hippo").

Adding a training phase changes what gets measured (trained vs untrained
substrate) and is not in the ratified design, so it was NOT done unilaterally.
Raised as a decision chip; see the design of record below.

V3-EXQ-1061 -- MECH-131: is stored aversive residue ACTIVATED as an anticipatory
forward-biasing signal before candidate generation?

Claims: MECH-131 (diagnostic -- see EXPERIMENT_PURPOSE)

MECH-131 asserts that a vmPFC-analog must activate stored aversive residue as an
anticipatory forward-biasing signal at trajectory-evaluation time, and that residue
"correctly stored but not so activated will fail to suppress harm-associated
trajectory re-selection". V3 has TWO live pre-candidate-generation residue reads:

  CH1  HippocampalModule._get_terrain_action_object_mean
       residue_field.evaluate -> residue_val -> terrain_prior -> initial
       action-object proposal mean.
       Knob: HippocampalConfig.terrain_prior_residue_channel_enabled

  CH2  HippocampalModule._score_trajectory
       residue_field.evaluate_trajectory -> per-candidate terrain score ->
       torch.argsort(scores)[:num_elite] elite selection + distribution refit.
       Knob: HippocampalConfig.score_trajectory_residue_terrain_enabled

Both knobs default True (bit-identical to the pre-instrument substrate) and were
built for this experiment (ree-v3, 2026-09-18). Setting BOTH False is the complete
anticipatory lesion; either alone is partial -- with CH1 alone off, the CEM
elite-ranking score still carried a cross-candidate spread of 0.892 over 32
candidates (contract C6), i.e. residue still steered selection.

The POST-HOC path (E3.compute_residue_cost, scaled by rho_residue) is NEVER
lesioned by either knob, and residue ACCUMULATION (ResidueField.accumulate) is
never touched. That is what makes this the claim's own arm: storage intact,
post-hoc readout intact, anticipatory activation removed.

ARMS
  ARM_1_intact          CH1 True  CH2 True    full anticipatory activation
  ARM_2_ch1_lesion      CH1 False CH2 True    the partial lesion, kept deliberately
  ARM_3_complete_lesion CH1 False CH2 False   no anticipatory read at all

ARM_3 is the internal residue-blind reference for GENERATION -- a measured
structural fact (its CEM score spread is exactly 0.0, contract C7/C8), not an
assumption. So no fourth control arm is needed. ARM_2 partitions the effect
between the two channels and records whether the originally-proposed single-knob
design could have detected anything at all.

DVs, each mapped to one of MECH-131's three recorded predictions (claims.yaml notes)
  DV1 <- prediction (1) "...failing to shift trajectory distribution away from them
         in generation."  residue_avoidance = mean residue over the PROPOSED
         candidate pool (pre-E3-selection). LOWER = more avoidance.
         Predicted: ARM_1 < ARM_3.
  DV2 <- prediction (3) "The activation signal should be graded by residue
         curvature magnitude."  gradedness_rho = Spearman rank correlation, across
         candidates, between a candidate's residue magnitude in the residue-blind
         reference and how much its proposal mass is suppressed in ARM_1.
         Predicted: positive. REPORTED WITH ITS SIGN, no magnitude floor.
  DV3 <- prediction (2) "Residue activation must precede trajectory ranking, not
         follow it... a post-hoc filter is not an active state constraint."
         Recorded as a structural dissociation: in ARM_3 generation is
         residue-blind (cem_score_spread == 0.0) WHILE compute_residue_cost is
         non-zero. Threshold-free.

CRITERIA -- ORDINAL, pre-registered, NO invented numeric floor
(user decision 2026-09-19T00:49Z; MECH-131 has no what_would_answer and EXP-0847
sets require_pre_registered_thresholds: false, so no effect-size floor or alpha is
derivable from recorded text and none is invented here)
  C1  DV1 ordering holds PER SEED: residue_avoidance(ARM_1) < residue_avoidance(ARM_3)
      in every seed. Effect sizes are REPORTED, never thresholded.
  C2  DV3 dissociation holds in every seed: ARM_3 cem_score_spread == 0.0 AND
      ARM_3 post_hoc_residue_cost != 0.0.
  C3  Preconditions hold in every arm and seed (see below).
  Gradedness (DV2) is REPORTED, not gated -- its sign is the finding.

PRECONDITIONS (a failure here makes the probe vacuous, not negative)
  P1  num_harm_events > 0 in every cell -- residue actually accumulated.
  P2  total_residue identical across arms at matched seed -- "correctly stored".
  P3  post_hoc_residue_cost != 0 in every cell -- the null path survives.
  P4  cem_score_spread > 0 in ARM_1 and ARM_2, and == 0.0 in ARM_3.

PURPOSE: DIAGNOSTIC. This run does NOT move MECH-131's status and is excluded from
governance confidence scoring. Its stated job is to hand governance the numbers
needed to author MECH-131's missing `what_would_answer` -- the per-arm
residue_avoidance values, their per-seed ordering, the observed effect sizes, and
the gradedness sign -- so that a later governance-grade run can pre-register
against measured quantities instead of invented ones.

GOV-ECOL-1: replication is reported as TWO counts. This design uses ONE world
family (CausalGridWorldV2 at fixed size/hazard/resource counts), so the result must
be read as `ecological transfer untested`. Seeds alone never license "general".

Substrate-path note (queue-experiment Step 2.5c): four OPEN `corrupting`
substrate_queue entries overlap ree_core modules this driver imports, and every one
is INERT at this config -- contextmemory-write-path-addressing-degeneracy (both call
sites gated: sd016_writepath_mode defaults "off",
contextmemory_write_addressing_loss_weight defaults 0.0), MECH-320
(use_tonic_vigor=False), sd_blocked_agency_mismatch_floor_calibration
(use_blocked_agency=False), sd105_frozen_shared_entropy_floor_multiplier
(use_selection_entropy_floor=False). This driver asserts all four are off at
runtime (assert_defect_paths_inert) rather than trusting the defaults, because the
gate's validity depends on it.

Design of record:
REE_assembly/evidence/planning/mech131_three_arm_lesion_design_staged_20260918.md
"""

import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import (
    write_flat_manifest,
    resolve_evidence_experiments_dir,
    flat_readout,
)
from experiments._lib.arm_fingerprint import arm_cell
from experiment_protocol import emit_outcome

EXPERIMENT_TYPE    = "v3_exq_1061_mech131_anticipatory_residue_lesion"
QUEUE_ID           = "V3-EXQ-1061"
CLAIM_IDS          = ["MECH-131"]
EXPERIMENT_PURPOSE = "diagnostic"  # excluded from governance confidence scoring

ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

SEEDS = [11, 23, 37]              # >= 3 distinct seeds (EXP-0847 seed_policy)
GRID_SIZE      = 5
NUM_HAZARDS    = 1
NUM_RESOURCES  = 2
SELF_DIM       = 16
WORLD_DIM      = 16
ACTION_DIM     = 4

CHARGE_STEPS   = 60               # rollout ticks that lay down residue
N_PROBE_STATES = 12               # probe states per cell (proposal pools measured)

# (arm_name, CH1 terrain_prior channel, CH2 score_trajectory terrain score)
ARMS: List[Tuple[str, bool, bool]] = [
    ("ARM_1_intact",          True,  True),
    ("ARM_2_ch1_lesion",      False, True),
    ("ARM_3_complete_lesion", False, False),
]

# substrate_queue defects whose paths this driver imports; each must be inert.
_DEFECT_FLAGS_MUST_BE_OFF = (
    ("sd016_writepath_mode", "off"),                            # on e1 config
    ("contextmemory_write_addressing_loss_weight", 0.0),        # on top config
    ("use_tonic_vigor", False),
    ("use_blocked_agency", False),
    ("use_selection_entropy_floor", False),
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _obs(obs_dict) -> Tuple[torch.Tensor, torch.Tensor]:
    body, world = obs_dict["body_state"], obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return body, world


def _config_slice(ch1: bool, ch2: bool, seed: int) -> Dict[str, Any]:
    """Declared config slice for the arm fingerprint (config_slice_declared=True)."""
    return {
        "terrain_prior_residue_channel_enabled": ch1,
        "score_trajectory_residue_terrain_enabled": ch2,
        "seed": seed,
        "grid_size": GRID_SIZE,
        "num_hazards": NUM_HAZARDS,
        "num_resources": NUM_RESOURCES,
        "self_dim": SELF_DIM,
        "world_dim": WORLD_DIM,
        "charge_steps": CHARGE_STEPS,
        "n_probe_states": N_PROBE_STATES,
    }


def assert_defect_paths_inert(agent: REEAgent, cfg: REEConfig) -> Dict[str, Any]:
    """Step 2.5c validity condition, asserted at RUNTIME rather than assumed.

    The overlap gate was cleared because each open `corrupting` entry's code path
    is unreachable at this config. If any of these ever defaults differently, the
    gate's clearance is void and this run must not be trusted -- so it fails loudly
    here instead of producing a quietly-invalid manifest.
    """
    observed: Dict[str, Any] = {}
    wp = getattr(agent.e1.config, "sd016_writepath_mode", "off")
    observed["sd016_writepath_mode"] = wp
    assert wp == "off", (
        f"substrate_queue contextmemory-write-path-addressing-degeneracy is OPEN and "
        f"corrupting; this run's 2.5c clearance requires sd016_writepath_mode='off', "
        f"got {wp!r}"
    )
    for name, want in _DEFECT_FLAGS_MUST_BE_OFF[1:]:
        got = getattr(cfg, name, want)
        observed[name] = got
        assert got == want, (
            f"2.5c clearance requires {name}=={want!r} (an open corrupting "
            f"substrate_queue entry covers its path); got {got!r}"
        )
    return observed


def build_agent(ch1: bool, ch2: bool, seed: int):
    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=GRID_SIZE, num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES, use_proxy_fields=True,
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        terrain_prior_residue_channel_enabled=ch1,
        score_trajectory_residue_terrain_enabled=ch2,
    )
    # from_dims silently swallows unknown kwargs (the MECH-307 failure shape), so
    # the arms are asserted reachable rather than assumed. Without this a lesion
    # arm could silently run intact and the whole experiment would read as a null.
    assert cfg.hippocampal.terrain_prior_residue_channel_enabled is ch1, (
        "from_dims did not route terrain_prior_residue_channel_enabled"
    )
    assert cfg.hippocampal.score_trajectory_residue_terrain_enabled is ch2, (
        "from_dims did not route score_trajectory_residue_terrain_enabled"
    )
    agent = REEAgent(cfg)
    agent.reset()
    return agent, env, cfg


def charge_residue(agent, env, steps: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Lay residue down at REAL visited z_world locations via a real rollout.

    Not a hand-built favourable batch: the locations are wherever the agent
    actually went. Returns the final obs plus the number of accumulate calls.
    """
    _flat, obs_dict = env.reset()
    body, world = _obs(obs_dict)
    writes = 0
    for _ in range(steps):
        latent = agent.sense(body, world)
        agent.residue_field.accumulate(latent.z_world.detach(), harm_magnitude=1.0)
        writes += 1
        action = int(torch.randint(0, ACTION_DIM, (1,)).item())
        _flat, _harm, done, _info, obs_dict = env.step(action)
        body, world = _obs(obs_dict)
        if done:
            _flat, obs_dict = env.reset()
            body, world = _obs(obs_dict)
    return body, world, writes


def measure_pool(agent, env, body, world, seed: int, n_states: int) -> Dict[str, Any]:
    """Propose candidates from n_states probe states; measure the DVs on the pool.

    residue_avoidance is the mean residue of the PROPOSED trajectories -- what E3
    is handed. Lower = the generator steered away from harm-associated regions.
    """
    rf = agent.residue_field
    per_state: List[Dict[str, Any]] = []
    for s in range(n_states):
        latent = agent.sense(body, world)
        z_world = latent.z_world.detach()
        z_self = latent.z_self.detach()
        # Same sampling noise across arms at matched (seed, probe index): the arms
        # must differ by the lesion, not by their random draws.
        torch.manual_seed(seed * 1000 + s)
        trajs = agent.hippocampal.propose_trajectories(z_world, z_self=z_self)
        if not trajs:
            continue
        residues = [float(rf.evaluate_trajectory(t.get_world_state_sequence()).detach().sum())
                    for t in trajs if t.get_world_state_sequence() is not None]
        scores = [float(agent.hippocampal._score_trajectory(t).detach()) for t in trajs]
        post_hoc = float(agent.e3.compute_residue_cost(trajs[0]).detach().sum())
        if residues:
            per_state.append({
                "probe_index": s,
                "n_candidates": len(trajs),
                "mean_candidate_residue": sum(residues) / len(residues),
                "min_candidate_residue": min(residues),
                "cem_score_spread": max(scores) - min(scores),
                "post_hoc_residue_cost": post_hoc,
                "candidate_residues": residues,
            })
        # advance the world so probe states are not all identical
        action = int(torch.randint(0, ACTION_DIM, (1,)).item())
        _flat, _harm, done, _info, obs_dict = env.step(action)
        body, world = _obs(obs_dict)
        if done:
            _flat, obs_dict = env.reset()
            body, world = _obs(obs_dict)

    assert per_state, "no candidates proposed at any probe state -- probe is vacuous"
    n = len(per_state)
    return {
        "n_probe_states_scored": n,
        "residue_avoidance": sum(p["mean_candidate_residue"] for p in per_state) / n,
        "mean_min_candidate_residue": sum(p["min_candidate_residue"] for p in per_state) / n,
        "cem_score_spread": sum(p["cem_score_spread"] for p in per_state) / n,
        "cem_score_spread_max": max(p["cem_score_spread"] for p in per_state),
        "post_hoc_residue_cost": sum(p["post_hoc_residue_cost"] for p in per_state) / n,
        "post_hoc_residue_cost_min_abs": min(abs(p["post_hoc_residue_cost"]) for p in per_state),
        "per_probe_state": per_state,
    }


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    """Spearman rank correlation. None when undefined (n < 3, or no variance)."""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None

    def ranks(v: List[float]) -> Optional[List[float]]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n))
    dy = sum((ry[i] - my) ** 2 for i in range(n))
    if dx <= 0 or dy <= 0:
        return None
    return num / ((dx * dy) ** 0.5)


def gradedness_rho(intact: Dict[str, Any], reference: Dict[str, Any]) -> Optional[float]:
    """DV2 <- prediction (3): is suppression GRADED by residue magnitude?

    x = candidate residue in the residue-blind reference arm (ARM_3), i.e. how
        harm-associated that region is, measured by an arm that is not steering.
    y = suppression = reference residue - intact residue, per matched candidate
        rank position.
    A positive rho means higher-residue candidates were suppressed more, which is
    what the claim predicts. Sign is the finding; no magnitude floor is applied.
    """
    xs: List[float] = []
    ys: List[float] = []
    ref_by_idx = {p["probe_index"]: p for p in reference["per_probe_state"]}
    for p in intact["per_probe_state"]:
        r = ref_by_idx.get(p["probe_index"])
        if r is None:
            continue
        a = sorted(r["candidate_residues"])
        b = sorted(p["candidate_residues"])
        for i in range(min(len(a), len(b))):
            xs.append(a[i])
            ys.append(a[i] - b[i])
    return _spearman(xs, ys)


def run_cell(arm: str, ch1: bool, ch2: bool, seed: int, dry_run: bool) -> Dict[str, Any]:
    steps = 8 if dry_run else CHARGE_STEPS
    n_states = 3 if dry_run else N_PROBE_STATES
    agent, env, cfg = build_agent(ch1, ch2, seed)
    defect_flags = assert_defect_paths_inert(agent, cfg)
    body, world, writes = charge_residue(agent, env, steps)
    rf = agent.residue_field
    pool = measure_pool(agent, env, body, world, seed, n_states)
    row: Dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "ch1_terrain_prior_residue_channel_enabled": bool(ch1),
        "ch2_score_trajectory_residue_terrain_enabled": bool(ch2),
        "accumulate_calls": writes,
        "total_residue": float(rf.total_residue),
        "num_harm_events": float(rf.num_harm_events),
        "defect_flags_observed": defect_flags,
    }
    row.update({k: v for k, v in pool.items()})
    row["agent"] = agent
    return row


# ----------------------------------------------------------------------
def main(dry_run: bool = False):
    t0 = time.time()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    arm_results: List[Dict[str, Any]] = []
    all_agents: List[Any] = []
    total_cells = len(ARMS) * len(SEEDS)
    cell_num = 0

    for arm, ch1, ch2 in ARMS:
        for seed in SEEDS:
            cell_num += 1
            print(f"[cell {cell_num}/{total_cells}] arm={arm} seed={seed} "
                  f"ch1={ch1} ch2={ch2}", flush=True)
            with arm_cell(
                seed,
                config_slice=_config_slice(ch1, ch2, seed),
                script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                row = run_cell(arm, ch1, ch2, seed, dry_run=dry_run)
                all_agents.append(row.pop("agent"))
                cell.stamp(row)
            arm_results.append(row)

    def _cell(arm: str, seed: int) -> Dict[str, Any]:
        return next(r for r in arm_results if r["arm"] == arm and r["seed"] == seed)

    # ---- Preconditions -------------------------------------------------
    p1_harm = all(r["num_harm_events"] > 0 for r in arm_results)
    p2_storage = all(
        _cell("ARM_1_intact", s)["total_residue"]
        == _cell("ARM_2_ch1_lesion", s)["total_residue"]
        == _cell("ARM_3_complete_lesion", s)["total_residue"]
        for s in SEEDS
    )
    p3_post_hoc = all(r["post_hoc_residue_cost_min_abs"] > 0.0 for r in arm_results)
    p4_spread = (
        all(_cell(a, s)["cem_score_spread_max"] > 0.0
            for a in ("ARM_1_intact", "ARM_2_ch1_lesion") for s in SEEDS)
        and all(_cell("ARM_3_complete_lesion", s)["cem_score_spread_max"] == 0.0
                for s in SEEDS)
    )
    preconditions_met = bool(p1_harm and p2_storage and p3_post_hoc and p4_spread)

    # ---- C1: DV1 ordinal ordering, PER SEED ----------------------------
    per_seed: List[Dict[str, Any]] = []
    for s in SEEDS:
        a1 = _cell("ARM_1_intact", s)["residue_avoidance"]
        a2 = _cell("ARM_2_ch1_lesion", s)["residue_avoidance"]
        a3 = _cell("ARM_3_complete_lesion", s)["residue_avoidance"]
        per_seed.append({
            "seed": s,
            "residue_avoidance_arm1_intact": a1,
            "residue_avoidance_arm2_ch1_lesion": a2,
            "residue_avoidance_arm3_complete_lesion": a3,
            # effect sizes are REPORTED, never thresholded (user decision)
            "effect_arm3_minus_arm1": a3 - a1,
            "effect_arm3_minus_arm2": a3 - a2,
            "effect_arm2_minus_arm1": a2 - a1,
            "relative_effect_arm3_vs_arm1": ((a3 - a1) / abs(a3)) if a3 else None,
            "c1_arm1_below_arm3": int(a1 < a3),
            "arm2_below_arm3": int(a2 < a3),
        })
    c1_met = all(p["c1_arm1_below_arm3"] == 1 for p in per_seed)
    c1_seeds_clearing = sum(p["c1_arm1_below_arm3"] for p in per_seed)

    # ---- C2: DV3 ordering dissociation, per seed -----------------------
    c2_flags = []
    for s in SEEDS:
        r3 = _cell("ARM_3_complete_lesion", s)
        c2_flags.append(int(r3["cem_score_spread_max"] == 0.0
                            and r3["post_hoc_residue_cost_min_abs"] > 0.0))
    c2_met = all(f == 1 for f in c2_flags)

    # ---- DV2: gradedness, reported with its sign -----------------------
    rhos: List[Dict[str, Any]] = []
    for s in SEEDS:
        rho = gradedness_rho(_cell("ARM_1_intact", s), _cell("ARM_3_complete_lesion", s))
        rhos.append({"seed": s, "gradedness_rho": rho})
    rho_vals = [r["gradedness_rho"] for r in rhos if r["gradedness_rho"] is not None]
    rho_mean = (sum(rho_vals) / len(rho_vals)) if rho_vals else None
    rho_positive_seeds = sum(1 for v in rho_vals if v > 0)

    outcome = "PASS" if (preconditions_met and c1_met and c2_met) else "FAIL"

    interpretation = (
        "DIAGNOSTIC -- excluded from governance confidence scoring; does not move "
        "MECH-131's status. Purpose: supply the measured quantities governance needs "
        "to author MECH-131's missing what_would_answer. "
        f"C1 (residue_avoidance ARM_1 < ARM_3 per seed): {c1_seeds_clearing}/{len(SEEDS)} seeds. "
        f"C2 (ARM_3 generation residue-blind while post-hoc live): {'met' if c2_met else 'NOT met'}. "
        f"Gradedness rho mean {rho_mean} ({rho_positive_seeds}/{len(rho_vals)} seeds positive) "
        "-- reported, not gated. "
        "Stochastic replication: 3 seeds. World-family replication: 1 family -- "
        "ECOLOGICAL TRANSFER UNTESTED (GOV-ECOL-1); seeds alone do not license "
        "'general' or 'robust'."
    )

    readout = flat_readout({
        "c1_arm1_below_arm3_all_seeds": int(c1_met),
        "c1_seeds_clearing": c1_seeds_clearing,
        "c1_seeds_required": len(SEEDS),
        "c2_ordering_dissociation_all_seeds": int(c2_met),
        "preconditions_met": int(preconditions_met),
        "p1_harm_accumulated": int(p1_harm),
        "p2_storage_identical_across_arms": int(p2_storage),
        "p3_post_hoc_scorer_live": int(p3_post_hoc),
        "p4_spread_signature": int(p4_spread),
        "residue_avoidance_arm1_intact_mean":
            sum(p["residue_avoidance_arm1_intact"] for p in per_seed) / len(per_seed),
        "residue_avoidance_arm2_ch1_lesion_mean":
            sum(p["residue_avoidance_arm2_ch1_lesion"] for p in per_seed) / len(per_seed),
        "residue_avoidance_arm3_complete_lesion_mean":
            sum(p["residue_avoidance_arm3_complete_lesion"] for p in per_seed) / len(per_seed),
        "effect_arm3_minus_arm1_mean":
            sum(p["effect_arm3_minus_arm1"] for p in per_seed) / len(per_seed),
        "effect_arm2_minus_arm1_mean":
            sum(p["effect_arm2_minus_arm1"] for p in per_seed) / len(per_seed),
        "gradedness_rho_mean": rho_mean if rho_mean is not None else 0.0,
        "gradedness_rho_seeds_positive": rho_positive_seeds,
        "gradedness_rho_seeds_defined": len(rho_vals),
        "n_seeds": len(SEEDS),
        "n_world_families": 1,
        "n_arms": len(ARMS),
    })

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_class": "diagnostic",
        "evidence_direction": "unknown",
        "outcome": outcome,
        "status": outcome,
        "readout": readout,
        "criteria": {
            "C1_dv1_ordinal_arm1_below_arm3_per_seed": {
                "measured": c1_seeds_clearing, "threshold": len(SEEDS),
                "met": int(c1_met),
                "note": "ordinal; effect sizes reported, no invented floor",
            },
            "C2_dv3_ordering_dissociation_per_seed": {
                "measured": sum(c2_flags), "threshold": len(SEEDS), "met": int(c2_met),
            },
            "C3_preconditions": {
                "measured": int(preconditions_met), "threshold": 1,
                "met": int(preconditions_met),
            },
            "DV2_gradedness": {
                "measured_rho": rho_mean, "threshold": None,
                "note": "REPORTED with its sign; no magnitude floor (user decision)",
            },
        },
        "replication": {
            "stochastic_seeds": len(SEEDS),
            "world_families": 1,
            "ecological_transfer": "untested",
        },
        "per_seed": per_seed,
        "gradedness": rhos,
        "arm_results": arm_results,
        "params": {
            "seeds": SEEDS, "arms": [a[0] for a in ARMS],
            "grid_size": GRID_SIZE, "num_hazards": NUM_HAZARDS,
            "num_resources": NUM_RESOURCES, "self_dim": SELF_DIM,
            "world_dim": WORLD_DIM, "action_dim": ACTION_DIM,
            "charge_steps": CHARGE_STEPS, "n_probe_states": N_PROBE_STATES,
            "dry_run": bool(dry_run),
        },
        "interpretation": interpretation,
        "design_of_record": (
            "REE_assembly/evidence/planning/"
            "mech131_three_arm_lesion_design_staged_20260918.md"
        ),
    }

    out_dir = resolve_evidence_experiments_dir(Path(__file__))
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=dry_run,
        config=manifest["params"],
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        agent=all_agents,
    )
    print(f"Outcome: {outcome}", flush=True)
    print(f"wrote: {out_path}", flush=True)
    manifest["_manifest_path"] = str(out_path)
    return manifest, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-1061: MECH-131 three-arm anticipatory-residue lesion "
                    "(intact / CH1-lesion / CH1+CH2-lesion) -- DIAGNOSTIC"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Minimal cells (8 charge steps, 3 probe states) to verify wiring")
    args = parser.parse_args()

    manifest, out_path = main(dry_run=args.dry_run)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        run_id=manifest["run_id"],
        queue_id=QUEUE_ID,
        dry_run=args.dry_run,
    )
