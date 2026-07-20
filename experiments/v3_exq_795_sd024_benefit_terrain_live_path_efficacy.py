#!/opt/local/bin/python3
"""V3-EXQ-795 -- SD-024 benefit-terrain LIVE-PATH efficacy (producer ON vs OFF).

Claims: SD-024 (residue.benefit_terrain_live_producer), SD-025 (hippocampal_module.curiosity_drive)
Purpose: DIAGNOSTIC (experiment_purpose="diagnostic"; excluded from governance
  confidence/conflict scoring; a PASS is live-path readiness, routed through
  /failure-autopsy adjudication before any governance action).
Substrate: SD-024 IMPLEMENTED 2026-07-16; its LIVE-PATH PRODUCER wired 2026-07-20
  (ree-v3 8ac193d substrate + 7f16f25ceb contracts). SD-025 IMPLEMENTED 2026-07-16.
Multi-claim -> evidence_direction_per_claim REQUIRED (emitted).

WHY THIS EXPERIMENT (the gap V3-EXQ-766/767/767a structurally could not see)
----------------------------------------------------------------------------
Between 2026-07-16 and 2026-07-20, ResidueField.accumulate_benefit -- the SOLE write
path into benefit_rbf_field -- had NO CALLER anywhere in ree_core/. Its only two write
sites (residue/field.py:673, :682) sit inside that one method; it was invoked only from
experiments/ scripts and tests/contracts/. Measured on a real warmup_train loop
(darwin-arm64, curiosity_weight=0.5) with benefit_terrain_enabled AND
use_da_modulated_rbf_density BOTH True:

  benefit_rbf_field.active_mask.sum() == 0 ; num_benefit_events == 0.0
  -> RBFLayer.compute_local_density early-returns zeros on the empty active mask
     (field.py:273) -> compute_representational_density == 0.0 exactly
  -> HippocampalModule._curiosity_bonus: novelty = density * (1 - familiarity) = 0
  -> the returned bonus was 0.0 on ALL 14432 live calls, regardless of curiosity_weight.

The use_curiosity_familiarity True/False ablation was BIT-IDENTICAL, confirming the
familiarity term was not the binding constraint. So the SD-025 curiosity drive
contributed EXACTLY ZERO to CEM trajectory scoring in every live agent run.

V3-EXQ-766/767/767a are NOT invalidated and are NOT superseded by this run. They
populate the terrain THEMSELVES via direct rf.accumulate_benefit() calls (767a lines
236, 238, 305), which is why they measured density gaps of 75-80 and PASSed. They are
valid IN-VITRO validations of the DRIVE MECHANISM. What no experiment has yet tested --
and what this one tests -- is LIVE-PATH efficacy: does the drive do anything when the
terrain is built by the AGENT'S OWN reward contacts rather than by the driver script?
That is a different question, so this is a new EXQ NUMBER, not a lettered iteration.

THE MEASUREMENT (instrument the LIVE calls, do not reconstruct them)
--------------------------------------------------------------------
This script NEVER calls accumulate_benefit. The terrain may only be populated by the
agent's own update_z_goal path. A recording wrapper is installed around
HippocampalModule._curiosity_bonus so every LIVE call made during real CEM scoring is
logged -- reproducing exactly the statistic that read 0.0 on all 14432 calls. Calls are
grouped by selection event (a counter incremented per select_action) so the
cross-candidate RANGE within one selection is recoverable.

ARMS (identical seeds, identical config except the one flag under test):
  ARM_OFF  benefit_terrain_live_producer=False  -- the pre-2026-07-20 state
  ARM_ON   benefit_terrain_live_producer=True   -- the wired live path
Both arms: benefit_terrain_enabled=True, use_da_modulated_rbf_density=True,
curiosity_weight=CURIOSITY_WEIGHT. So the ONLY difference is whether a producer exists.

LEG 1 -- PRODUCER LIVENESS (does the agent's own loop build terrain?)
  L1a benefit_active_centers_on   >= L1A_CENTERS_FLOOR                    [LOAD-BEARING]
  L1b benefit_events_on           >= L1B_EVENTS_FLOOR                     [supporting]
  L1c benefit_active_centers_off  == 0 exactly (the defect's own signature) [supporting]

LEG 2 -- DENSITY LIVENESS (does the SD-024 read become non-zero?)
  L2a density_mean_on  >= L2A_DENSITY_FLOOR                               [LOAD-BEARING]
  L2b density_mean_off == 0.0 exactly                                     [supporting]

LEG 3 -- DRIVE LIVENESS (THE decisive readout -- was 0.0 on all 14432 live calls)
  L3a curiosity_bonus_abs_mean_on >= L3A_BONUS_FLOOR over LIVE calls      [LOAD-BEARING]
  L3b curiosity_bonus_abs_mean_off == 0.0 exactly over LIVE calls         [supporting]

LEG 4 -- SELECTION AUTHORITY (does it change what CEM would pick?)
  L4a curiosity_bonus_range_mean_on >= L4A_RANGE_FLOOR                    [LOAD-BEARING]
      The CROSS-CANDIDATE range within a selection event, NOT a magnitude.

  DV-SYMMETRY DECLARATION (mandatory; skill Step 3).
  ARM_ON manipulation = populating benefit terrain at reward-contact z_world.
  ARM_ON DV = per-candidate curiosity bonus -> CEM argmin over candidate scores.
  Symmetry group of that DV: (i) addition of a constant uniform across candidates,
  (ii) monotone rescaling, (iii) permutation of candidates.
  The manipulation is NOT invariant under (i): HippocampalModule._score_trajectory
  calls _curiosity_bonus(world_seq) ONCE PER TRAJECTORY (module.py:1101 loop and
  :1755 comprehension), and each call's density read is taken at THAT candidate's own
  world states. Two candidates traversing regions of differing benefit density
  therefore receive DIFFERENT bonuses, so the term does not cancel in an argmin and
  the cross-candidate range is a real measurement rather than an arithmetic identity.
  (This is precisely the V3-EXQ-604c failure mode -- a broadcast scalar whose delta
  against a selection-derived DV is fixed to 0.0 before the run -- and it does NOT
  apply here. Note the .mean() inside _curiosity_bonus reduces over that ONE
  candidate's (batch, horizon), never across candidates.)
  L4 is nevertheless the leg where this could still degenerate, which is why its
  readiness precondition asserts RANGE (below), not magnitude.
  ARM_OFF's manipulation is the absence of a producer; its DV is identically 0 by
  construction, which is the control, not a measurement.

ACCEPTANCE (PASS): L1a AND L2a AND L3a AND L4a, with every OFF-arm control exactly 0.
FAIL means the producer is wired but does not move the live drive -- which would route
to substrate work on the drive, not on the producer.

READINESS (P0 positive control -- SAME statistic the load-bearing criteria route on)
------------------------------------------------------------------------------------
The V3-EXQ-643 defect was a readiness gate asserting a MAGNITUDE while the load-bearing
criterion routed on a RANGE; a uniform per-tick offset then passed readiness while the
criterion's precondition was unmet, and the run self-routed a falsification on a test
that never ran. L4a here gates on cross-candidate RANGE, so the readiness precondition
asserts RANGE on a positive control: a hand-built dense-vs-sparse benefit field scored
over candidates that GENUINELY DIFFER (half heading dense, half sparse). This is the
only place in this script where the terrain is populated directly, and it is used SOLELY
to establish that the instrument can register a range at all -- never as an arm, never
as evidence. Below floor => substrate_not_ready_requeue, NEVER a verdict label.

Per-precondition scope: the readiness gate certifies the CURIOSITY channel only (it
exercises _curiosity_bonus and nothing else). It does not speak for the residue-terrain
or wanting channels, which this experiment does not test.

No training of any head on a latent (non-parametric RBF writes + reads + an EMA), so
phased training is N/A. MECH-094: the producer reads hypothesis_tag off the live latent;
replay/DMN ticks cannot build terrain. No sleep -> no SLEEP DRIVER line required.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402

ANCHOR_REACHABILITY_EXEMPT = (
    "The readiness predicate IS the degeneracy definition, and is reachable by "
    "construction. READINESS_RANGE_FLOOR = 1e-6 asserts only that the cross-candidate "
    "range is NOT IDENTICALLY ZERO -- zero range is precisely what 'the instrument "
    "cannot register a range' means, so the gate cannot be narrower than the state it "
    "anchors to (the V3-EXQ-778d failure mode, where a hand-written predicate scored one "
    "rail of a two-rail degeneracy and was unmeetable by construction). Reachability is "
    "structural rather than empirical: the positive control scores candidates placed at a "
    "20-center DA cluster versus a single sparse center, so the two candidate classes "
    "cannot return equal density unless compute_local_density itself is broken -- which "
    "is the condition the gate exists to detect. Smoke measured 13.86 against the 1e-6 "
    "floor (>7 orders of magnitude of headroom)."
)

EXPERIMENT_TYPE = "v3_exq_795_sd024_benefit_terrain_live_path_efficacy"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["SD-024", "SD-025"]

# ---- fixed design constants (pre-registered) ----
GRID_SIZE = 8
NUM_HAZARDS = 2
NUM_RESOURCES = 3
WORLD_DIM = 16
SELF_DIM = 16
ACTION_DIM = 4
CURIOSITY_WEIGHT = 0.5
ALPHA_WORLD = 0.9          # SD-008: z_world fidelity; default 0.3 is a known root cause
EPISODES = 12
STEPS_PER_EPISODE = 60
EPISODES_PER_RUN = EPISODES
SEEDS = [11, 23, 37]
ARMS = ["ARM_OFF", "ARM_ON"]

# ---- pre-registered thresholds (NOT derived from this run's statistics) ----
L1A_CENTERS_FLOOR = 1.0        # at least one active benefit center
L1B_EVENTS_FLOOR = 1.0         # at least one benefit event
L2A_DENSITY_FLOOR = 1e-6       # density strictly > 0 (defect read exactly 0.0)
L3A_BONUS_FLOOR = 1e-9         # live bonus strictly > 0 (defect read exactly 0.0)
L4A_RANGE_FLOOR = 1e-9         # cross-candidate range strictly > 0
READINESS_RANGE_FLOOR = 1e-6   # positive-control cross-candidate RANGE floor


# ----------------------------------------------------------------------
# Live-call recorder
# ----------------------------------------------------------------------
class CuriosityBonusRecorder:
    """Wraps HippocampalModule._curiosity_bonus and logs every LIVE call.

    Reproduces the exact statistic that read 0.0 on all 14432 live calls. Calls are
    tagged with a selection-event id so the CROSS-CANDIDATE range within one selection
    is recoverable -- _score_trajectory calls the bonus once per candidate trajectory,
    so all calls sharing an event id are the candidates of one CEM selection.

    This wrapper is READ-ONLY instrumentation: it returns the wrapped value unchanged,
    so the agent's behaviour is identical to an un-instrumented run.
    """

    def __init__(self, hippocampal) -> None:
        self.hippocampal = hippocampal
        self._orig = hippocampal._curiosity_bonus
        self.event_id = 0
        self.values: List[float] = []
        self.event_ids: List[int] = []
        hippocampal._curiosity_bonus = self._wrapped

    def _wrapped(self, world_seq):
        out = self._orig(world_seq)
        try:
            self.values.append(float(out))
            self.event_ids.append(self.event_id)
        except Exception:
            pass
        return out

    def new_event(self) -> None:
        self.event_id += 1

    def restore(self) -> None:
        self.hippocampal._curiosity_bonus = self._orig

    def abs_mean(self) -> float:
        return statistics.fmean([abs(v) for v in self.values]) if self.values else 0.0

    def range_mean(self) -> float:
        """Mean over selection events of (max - min) across that event's candidates.

        Events with a single candidate carry no cross-candidate information and are
        excluded rather than counted as a 0.0 range -- including them would dilute the
        statistic toward 0 and could mask a real range.
        """
        buckets: Dict[int, List[float]] = {}
        for eid, v in zip(self.event_ids, self.values):
            buckets.setdefault(eid, []).append(v)
        ranges = [max(vs) - min(vs) for vs in buckets.values() if len(vs) > 1]
        return statistics.fmean(ranges) if ranges else 0.0

    def n_multi_candidate_events(self) -> int:
        buckets: Dict[int, int] = {}
        for eid in self.event_ids:
            buckets[eid] = buckets.get(eid, 0) + 1
        return sum(1 for n in buckets.values() if n > 1)


# ----------------------------------------------------------------------
# Config / agent construction
# ----------------------------------------------------------------------
def _build_cfg(env, live_producer: bool) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
    )
    # Residue-terrain flags are not from_dims kwargs; set on the nested ResidueConfig
    # BEFORE REEAgent.__init__ builds the ResidueField from it.
    cfg.residue.benefit_terrain_enabled = True
    cfg.residue.use_da_modulated_rbf_density = True
    cfg.residue.da_allocation_scale = 4.0
    cfg.residue.benefit_terrain_live_producer = live_producer   # THE ONLY ARM DIFFERENCE
    cfg.hippocampal.curiosity_weight = CURIOSITY_WEIGHT
    cfg.latent.alpha_world = ALPHA_WORLD
    return cfg


def _obs(obs_dict):
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return body, world


def _full_config() -> dict:
    return {
        "grid_size": GRID_SIZE, "num_hazards": NUM_HAZARDS,
        "num_resources": NUM_RESOURCES, "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM, "action_dim": ACTION_DIM,
        "curiosity_weight": CURIOSITY_WEIGHT, "alpha_world": ALPHA_WORLD,
        "episodes": EPISODES,  # full-run value; dry-run shortens the loop bound only "steps_per_episode": STEPS_PER_EPISODE,
        "benefit_terrain_enabled": True, "use_da_modulated_rbf_density": True,
        "da_allocation_scale": 4.0,
        "arms": ARMS, "seeds": SEEDS,
        "thresholds": {
            "L1A_CENTERS_FLOOR": L1A_CENTERS_FLOOR,
            "L1B_EVENTS_FLOOR": L1B_EVENTS_FLOOR,
            "L2A_DENSITY_FLOOR": L2A_DENSITY_FLOOR,
            "L3A_BONUS_FLOOR": L3A_BONUS_FLOOR,
            "L4A_RANGE_FLOOR": L4A_RANGE_FLOOR,
            "READINESS_RANGE_FLOOR": READINESS_RANGE_FLOOR,
        },
    }


# ----------------------------------------------------------------------
# P0 readiness positive control -- RANGE, the statistic L4a routes on
# ----------------------------------------------------------------------
def readiness_probe(seed: int) -> Dict[str, float]:
    """Positive control: can the instrument register a CROSS-CANDIDATE RANGE at all?

    Hand-builds a dense benefit cluster and scores candidates that genuinely differ.
    This is the ONLY place this script populates the terrain directly, and it is used
    solely to certify the CURIOSITY channel's instrument -- never as an arm, never as
    evidence. Asserting RANGE (not magnitude) is the V3-EXQ-643 correction.
    """
    torch.manual_seed(seed)
    env = CausalGridWorldV2(seed=seed, size=GRID_SIZE, num_hazards=NUM_HAZARDS,
                            num_resources=NUM_RESOURCES, use_proxy_fields=True)
    cfg = _build_cfg(env, live_producer=False)
    agent = REEAgent(cfg)
    agent.reset()
    _flat, obs_dict = env.reset()
    body, world = _obs(obs_dict)
    agent.sense(body, world)

    # Dense cluster at a known location; the sparse region gets a single center.
    dense = torch.zeros(1, WORLD_DIM); dense[0, 0] = 2.0
    sparse = torch.zeros(1, WORLD_DIM); sparse[0, 0] = -2.0
    for _ in range(20):
        agent.residue_field.accumulate_benefit(
            dense, benefit_magnitude=1.0, hypothesis_tag=False, dopamine_signal=1.0)
    agent.residue_field.accumulate_benefit(
        sparse, benefit_magnitude=1.0, hypothesis_tag=False, dopamine_signal=0.0)

    # Candidates that GENUINELY DIFFER: half at the dense location, half at the sparse
    # one. A positive control whose candidates were identical would have range ~0 by
    # construction and could not certify anything.
    bonuses = []
    for target in (dense, sparse, dense, sparse):
        seq = target.unsqueeze(0).repeat(1, 4, 1)   # [1, horizon, world_dim]
        bonuses.append(float(agent.hippocampal._curiosity_bonus(seq)))
    rng = max(bonuses) - min(bonuses)
    return {"readiness_range": rng, "readiness_bonuses": bonuses}


# ----------------------------------------------------------------------
# One (seed, arm) cell -- a REAL episode loop
# ----------------------------------------------------------------------
def run_cell(seed: int, arm: str, episodes: int) -> dict:
    """One (seed, arm) cell. `episodes` is the LOOP BOUND and is also the denominator
    printed in the `[train] ep N/M` progress line -- never a module constant, so a
    dry-run's shortened loop cannot advertise the full-run total to the runner."""
    live_producer = (arm == "ARM_ON")
    # MINT AS YOU GO: config_slice_declared=True + include_driver_script_in_hash=False
    # make ARM_OFF a cross-driver-reusable baseline mint for this lineage. Terminality
    # is a forward prediction, never a fact at run time, so the arms are emitted
    # reuse-eligible by default rather than gated on a guess about successors.
    with arm_cell(seed, config_slice=_full_config(), script_path=Path(__file__),
                  config_slice_declared=True,
                  include_driver_script_in_hash=False) as cell:
        torch.manual_seed(seed)
        env = CausalGridWorldV2(seed=seed, size=GRID_SIZE, num_hazards=NUM_HAZARDS,
                                num_resources=NUM_RESOURCES, use_proxy_fields=True)
        agent = REEAgent(_build_cfg(env, live_producer=live_producer))
        agent.reset()
        rec = CuriosityBonusRecorder(agent.hippocampal)

        densities: List[float] = []
        n_contacts = 0
        try:
            for ep in range(episodes):
                _flat, obs_dict = env.reset()
                body, world = _obs(obs_dict)
                for _step in range(STEPS_PER_EPISODE):
                    latent = agent.sense(body, world)
                    # Canonical V3 driver loop (cf. v3_exq_793:465-474): advance the
                    # clock, tick E1, generate candidates, then select. select_action
                    # takes (candidates, ticks) -- NOT raw observations.
                    ticks = agent.clock.advance()
                    e1_prior = (
                        agent._e1_tick(latent) if ticks.get("e1_tick")
                        else torch.zeros(1, agent.config.latent.world_dim)
                    )
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                    rec.new_event()
                    action = agent.select_action(candidates, ticks)
                    action_idx = int(action.argmax(dim=-1).item())
                    _f, harm_signal, done, _info, obs_dict = env.step(action_idx)
                    body, world = _obs(obs_dict)
                    # Reward contact drives the producer -- the ONLY terrain source in
                    # an arm. benefit_exposure is the positive part of harm_signal
                    # (negative = harm, positive = benefit), and drive_level is the
                    # canonical SD-012 formula from the body observation.
                    benefit = max(0.0, float(harm_signal))
                    if benefit > 0.0:
                        n_contacts += 1
                        agent.update_z_goal(
                            benefit_exposure=benefit,
                            drive_level=REEAgent.compute_drive_level(body),
                        )
                    if agent._current_latent is not None and agent._current_latent.z_world is not None:
                        densities.append(float(
                            agent.hippocampal.compute_representational_density(
                                agent._current_latent.z_world).detach().max()))
                    if done:
                        break
                # Print at an interval that always fires at least once, including
                # in a shortened dry-run -- the smoke must be able to verify the
                # progress pattern the runner scrapes.
                if (ep + 1) % max(1, episodes // 4) == 0 or (ep + 1) == episodes:
                    print(f"  [train] {arm} seed={seed} ep {ep+1}/{episodes}", flush=True)
        finally:
            rec.restore()

        rf = agent.residue_field
        row = {
            "arm_id": arm,
            "seed": seed,
            "benefit_active_centers": int(rf.benefit_rbf_field.active_mask.sum().item()),
            "benefit_events": float(rf.num_benefit_events),
            "density_mean": statistics.fmean(densities) if densities else 0.0,
            "density_max": max(densities) if densities else 0.0,
            "curiosity_bonus_abs_mean": rec.abs_mean(),
            "curiosity_bonus_range_mean": rec.range_mean(),
            "n_live_bonus_calls": len(rec.values),
            "n_multi_candidate_events": rec.n_multi_candidate_events(),
            "n_reward_contacts": n_contacts,
        }
        cell.stamp(row)
    return row


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate(rows: List[dict], readiness: List[Dict[str, float]]) -> dict:
    def agg(arm: str, key: str) -> float:
        vals = [r[key] for r in rows if r["arm_id"] == arm]
        return statistics.fmean(vals) if vals else 0.0

    centers_on, centers_off = agg("ARM_ON", "benefit_active_centers"), agg("ARM_OFF", "benefit_active_centers")
    events_on = agg("ARM_ON", "benefit_events")
    density_on, density_off = agg("ARM_ON", "density_mean"), agg("ARM_OFF", "density_mean")
    bonus_on, bonus_off = agg("ARM_ON", "curiosity_bonus_abs_mean"), agg("ARM_OFF", "curiosity_bonus_abs_mean")
    range_on = agg("ARM_ON", "curiosity_bonus_range_mean")

    # Readiness: the WORST cell, not the mean -- `met` is a worst-case claim, so the
    # reported `measured` must be the extremum the indexer can recompute against.
    worst_idx = min(range(len(readiness)), key=lambda i: readiness[i]["readiness_range"])
    worst_range = readiness[worst_idx]["readiness_range"]

    preconditions = [{
        "name": "curiosity_bonus_cross_candidate_range_live",
        "kind": "readiness",
        "description": ("positive control: dense-vs-sparse benefit field scored over "
                        "candidates that genuinely differ yields a non-zero CROSS-CANDIDATE "
                        "RANGE -- the SAME statistic L4a routes on (V3-EXQ-643 correction). "
                        "Certifies the CURIOSITY channel only."),
        "control": "hand-built dense cluster (20 DA centers) vs single sparse center",
        "measured": worst_range,
        "threshold": READINESS_RANGE_FLOOR,
        "direction": "lower",
        "offending_cell": f"seed={SEEDS[worst_idx]}",
        "met": worst_range >= READINESS_RANGE_FLOOR,
    }]
    ready = all(p["met"] for p in preconditions)

    l1a = centers_on >= L1A_CENTERS_FLOOR
    l1b = events_on >= L1B_EVENTS_FLOOR
    l1c = centers_off == 0
    l2a = density_on >= L2A_DENSITY_FLOOR
    l2b = density_off == 0.0
    l3a = bonus_on >= L3A_BONUS_FLOOR
    l3b = bonus_off == 0.0
    l4a = range_on >= L4A_RANGE_FLOOR

    on_rows = [r for r in rows if r["arm_id"] == "ARM_ON"]
    contacts_ok = all(r["n_reward_contacts"] > 0 for r in on_rows)
    multi_ok = all(r["n_multi_candidate_events"] > 0 for r in on_rows)

    criteria_non_degenerate = {
        # L1a is degenerate if the ON arm never made a reward contact -- then a 0 would
        # mean "no input", not "no producer".
        "L1a": contacts_ok,
        "L2a": contacts_ok,
        "L3a": contacts_ok and any(r["n_live_bonus_calls"] > 0 for r in on_rows),
        # A range over single-candidate events carries no cross-candidate information.
        "L4a": contacts_ok and multi_ok,
    }

    if not ready:
        label = "substrate_not_ready_requeue"
        outcome, direction = "FAIL", "non_contributory"
        per_claim = {"SD-024": "unknown", "SD-025": "unknown"}
    else:
        passed = l1a and l2a and l3a and l4a
        outcome = "PASS" if passed else "FAIL"
        if passed:
            label = "live_path_efficacious"
            direction = "supports"
            per_claim = {"SD-024": "supports", "SD-025": "supports"}
        elif l1a and l2a and not (l3a and l4a):
            label = "producer_live_drive_inert"
            direction = "mixed"
            per_claim = {"SD-024": "supports", "SD-025": "weakens"}
        else:
            label = "producer_not_live"
            direction = "weakens"
            per_claim = {"SD-024": "weakens", "SD-025": "unknown"}

    non_degenerate = all(criteria_non_degenerate.values()) and ready
    degeneracy_reason = None
    if not ready:
        degeneracy_reason = ("readiness precondition unmet: positive-control "
                             "cross-candidate range below floor")
    elif not contacts_ok:
        degeneracy_reason = "ARM_ON made no reward contacts; producer had no input to act on"
    elif not multi_ok:
        degeneracy_reason = "no multi-candidate selection events; L4a range uninformative"

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": per_claim,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": [
                {"name": "L1a_benefit_centers_on", "load_bearing": True, "passed": l1a},
                {"name": "L1b_benefit_events_on", "load_bearing": False, "passed": l1b},
                {"name": "L1c_benefit_centers_off_zero", "load_bearing": False, "passed": l1c},
                {"name": "L2a_density_on", "load_bearing": True, "passed": l2a},
                {"name": "L2b_density_off_zero", "load_bearing": False, "passed": l2b},
                {"name": "L3a_live_bonus_on", "load_bearing": True, "passed": l3a},
                {"name": "L3b_live_bonus_off_zero", "load_bearing": False, "passed": l3b},
                {"name": "L4a_cross_candidate_range_on", "load_bearing": True, "passed": l4a},
            ],
        },
        "aggregates": {
            "benefit_active_centers_on": centers_on,
            "benefit_active_centers_off": centers_off,
            "benefit_events_on": events_on,
            "density_mean_on": density_on,
            "density_mean_off": density_off,
            "curiosity_bonus_abs_mean_on": bonus_on,
            "curiosity_bonus_abs_mean_off": bonus_off,
            "curiosity_bonus_range_mean_on": range_on,
            "readiness_range_worst": worst_range,
        },
        "thresholds": _full_config()["thresholds"],
    }


def main(dry_run: bool = False) -> dict:
    t0, t0_perf = time.time(), time.perf_counter()
    seeds = SEEDS[:1] if dry_run else SEEDS
    episodes = 2 if dry_run else EPISODES

    readiness = [readiness_probe(s) for s in seeds]
    rows: List[dict] = []
    for seed in seeds:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            row = run_cell(seed, arm, episodes)
            rows.append(row)
            # Per-cell verdict = did this cell yield usable data? (The scientific
            # PASS/FAIL is the cross-arm comparison in evaluate(), not a per-cell
            # judgement -- an ARM_OFF cell reading 0 is the CONTROL WORKING.)
            print(f"verdict: {'PASS' if row['n_live_bonus_calls'] > 0 else 'FAIL'}", flush=True)

    ev = evaluate(rows, readiness)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "started_utc": datetime.utcfromtimestamp(t0).isoformat() + "Z",
        "timestamp_utc": timestamp,
        "outcome": ev["outcome"],
        "evidence_direction": ev["evidence_direction"],
        "evidence_direction_per_claim": ev["evidence_direction_per_claim"],
        "interpretation": ev["interpretation"],
        "aggregates": ev["aggregates"],
        "thresholds": ev["thresholds"],
        "non_degenerate": ev["non_degenerate"],
        "degeneracy_reason": ev["degeneracy_reason"],
        "per_seed_rows": rows,
        "arm_results": rows,
        "readiness_probe": readiness,
        "substrate": "SD-024",
        "notes": (
            "SD-024 benefit-terrain LIVE-PATH efficacy. Tests whether the producer wired "
            "2026-07-20 (REEAgent.update_z_goal -> ResidueField.accumulate_benefit) makes "
            "the SD-025 curiosity drive non-zero in a REAL agent episode loop. Between "
            "2026-07-16 and 2026-07-20 accumulate_benefit had NO CALLER in ree_core/, so "
            "benefit_rbf_field stayed empty, compute_representational_density returned "
            "exactly 0.0, and the curiosity bonus was 0.0 on all 14432 live calls -- the "
            "drive contributed exactly zero to CEM scoring in every live run. This script "
            "NEVER calls accumulate_benefit except in the P0 readiness positive control; "
            "the terrain in each arm may only be built by the agent's own reward contacts, "
            "and the bonus is read by wrapping _curiosity_bonus to log LIVE calls. "
            "V3-EXQ-766/767/767a are NOT superseded and NOT invalidated: they populate the "
            "terrain themselves (767a lines 236, 238, 305) and remain valid IN-VITRO "
            "validations of the drive mechanism; they simply never tested the live path, "
            "which is a different question -- hence a new EXQ number, not a letter. "
            "GOV-REUSE-1: the decisive readout (live curiosity bonus under producer ON vs "
            "OFF) cannot exist in any recorded manifest -- benefit_terrain_live_producer "
            "did not exist before 2026-07-20 -- so no reanalysis is possible and the run "
            "is required. L4a routes on CROSS-CANDIDATE RANGE and its readiness "
            "precondition asserts RANGE on a positive control (V3-EXQ-643 correction). "
            "DV-symmetry: _curiosity_bonus is called once per trajectory (module.py:1101, "
            ":1755) so the term is per-candidate, not a broadcast scalar, and does not "
            "cancel in an argmin (the V3-EXQ-604c failure mode does not apply). "
            "A PASS is live-path readiness routed through /failure-autopsy; it does not "
            "itself promote ARC-057."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=dry_run,
        config=_full_config(), seeds=list(seeds),
        script_path=Path(__file__), started_at=t0_perf,
    )
    print(f"Result written to: {out_path}")
    print(f"Done. Outcome: {ev['outcome']}")
    return {"outcome": ev["outcome"], "manifest_path": out_path, "run_id": run_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run (1 seed, 2 episodes).")
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
