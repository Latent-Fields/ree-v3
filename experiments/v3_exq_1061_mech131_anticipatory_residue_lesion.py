"""
V3-EXQ-1061 -- MECH-131

WARMUP IS LOAD-BEARING -- read before changing anything about it (2026-09-19).

An earlier revision of this driver measured an UNTRAINED agent and could not detect
its own manipulation: the intact-vs-complete-lesion effect was ~100x SMALLER than the
between-candidate residue SD (0.002% vs 0.18% of the pool mean), scale-invariantly
(measured on the hub at 60 and 200 residue-charge steps). Root cause is the
codebase's own documented expectation -- V3-EXQ-042: "if terrain_prior is random,
proposals are uninformed (equivalent to random candidates)". Residue avoidance is a
LEARNED competence here, so lesioning an untrained channel removes a capability the
substrate never had, and the resulting null would mean "an untrained generate-rollout
loop cannot express residue avoidance", NOT "MECH-131 is false".

Fix, per user decision OPTION A (2026-09-19T02:32Z): train terrain_prior FIRST using
V3-EXQ-042's existing protocol -- E3 behavioural cloning,
MSE(terrain_prior_ao_mean, selected_trajectory_ao_sequence.detach()), Adam(lr=5e-4),
600 warmup episodes x 200 steps (042's own numbers, not new ones) -- and gate on
042's hippo_quality_gap (hippocampal vs RANDOM proposal residue) before any arm is
lesioned. If the gate does not clear, the generator never acquired avoidance and the
run reports THAT rather than a claim-negative.

So: do not reduce the warmup, and do not remove the readiness gate. They are what
make a null here interpretable.

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

LESION AT EVALUATION, ON ONE TRAINED AGENT PER SEED -- not three trainings.
The user's OPTION A wording is explicit that the readiness gate must clear "before
any arm is lesioned", which fixes the design: warm up an INTACT agent (both channels
on), gate it, then read the three arms off THAT SAME agent by toggling the two read
gates at evaluation time. Weights are therefore identical across arms and the arms
differ only by the lesion -- a within-subject lesion, which is both the cleaner
comparison and a third of the compute. It is also the biologically apt framing: the
capability is acquired first, then the anticipatory read is removed.

This is only possible because both knobs are read LIVE off HippocampalConfig on every
call (getattr in _get_terrain_action_object_mean / _score_trajectory) rather than
cached at construction. Do not "optimise" them into cached attributes.

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

import math

import torch
import torch.nn.functional as F
import torch.optim as optim

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

# V3-EXQ-042's own numbers -- mirrored, not re-chosen (user decision OPTION A).
WARMUP_EPISODES    = 600
STEPS_PER_EPISODE  = 200
TERRAIN_LR         = 5e-4
N_RANDOM_COMPARE   = 8            # 042's hippo-vs-random candidate count
CANDIDATE_HORIZON  = 5            # 042's random-candidate horizon
GAP_SAMPLE_EVERY   = 10           # 042 samples the quality gap every 10 steps

N_PROBE_STATES = 12               # probe states per arm (proposal pools measured)

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


def warmup_train(agent, env, seed: int, episodes: int, steps_per_episode: int
                 ) -> Dict[str, Any]:
    """V3-EXQ-042's terrain_prior warmup, mirrored -- E3 behavioural cloning.

    terrain_prior learns to predict the action-object sequence E3 actually selected:
        MSE(terrain_prior_ao_mean, selected_trajectory_ao_sequence.detach())
    E3 prefers low-residue trajectories (compute_residue_cost * rho_residue), so over
    episodes the PROPOSALS become residue-avoidant. That acquired avoidance is the
    capability MECH-131's lesion is supposed to remove -- without it there is nothing
    to lesion, which is exactly what the pre-warmup revision of this driver measured.

    Residue accumulates at locations where the environment ACTUALLY reports harm, so
    the terrain the generator learns to avoid is the real hazard landscape rather than
    an injected one.

    Trains with BOTH channels intact (the lesion is applied later, at evaluation).
    """
    terrain_optimizer = optim.Adam(
        list(agent.hippocampal.terrain_prior.parameters())
        + list(agent.hippocampal.action_object_decoder.parameters()),
        lr=TERRAIN_LR,
    )
    agent.train()
    losses_early: List[float] = []
    losses_late: List[float] = []
    e3_ticks = 0
    harm_events = 0
    late_from = max(0, episodes - 100)

    for ep in range(episodes):
        _flat, obs_dict = env.reset()
        agent.reset()
        for _step in range(steps_per_episode):
            body, world = _obs(obs_dict)
            latent = agent.sense(body, world)
            ticks = agent.clock.advance()
            e1_prior = (agent._e1_tick(latent) if ticks["e1_tick"]
                        else torch.zeros(1, WORLD_DIM, device=agent.device))
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z = agent.theta_buffer.summary()

            action = None
            if ticks.get("e3_tick", False) and candidates:
                e3_ticks += 1
                result = agent.e3.select(candidates, temperature=1.0)
                action = result.selected_action.detach()
                agent._last_action = action
                selected_ao = result.selected_trajectory.get_action_object_sequence()
                if selected_ao is not None:
                    ao_mean_pred = agent.hippocampal._get_terrain_action_object_mean(
                        theta_z, e1_prior=e1_prior.detach()
                    )
                    terrain_loss = F.mse_loss(ao_mean_pred, selected_ao.detach())
                    terrain_optimizer.zero_grad()
                    terrain_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.hippocampal.terrain_prior.parameters())
                        + list(agent.hippocampal.action_object_decoder.parameters()),
                        1.0,
                    )
                    terrain_optimizer.step()
                    lv = float(terrain_loss.item())
                    if ep < 100:
                        losses_early.append(lv)
                    if ep >= late_from:
                        losses_late.append(lv)
            if action is None:
                action = getattr(agent, "_last_action", None)
            act_idx = (int(torch.argmax(action).item()) if action is not None
                       else int(torch.randint(0, ACTION_DIM, (1,)).item()))

            _flat, harm, done, _info, obs_dict = env.step(act_idx)
            # Residue marks REAL harm contacts, not injected ones.
            if float(harm) > 0.0:
                agent.residue_field.accumulate(
                    latent.z_world.detach(), harm_magnitude=1.0
                )
                harm_events += 1
            if done:
                _flat, obs_dict = env.reset()
                agent.reset()

    agent.eval()

    def _mean(v: List[float]) -> Optional[float]:
        return (sum(v) / len(v)) if v else None

    return {
        "warmup_episodes": episodes,
        "steps_per_episode": steps_per_episode,
        "e3_ticks": e3_ticks,
        "harm_events_during_warmup": harm_events,
        "terrain_loss_early": _mean(losses_early),
        "terrain_loss_late": _mean(losses_late),
        "final_obs": obs_dict,
    }


def _mean_residue_042(agent, trajs) -> float:
    """042's own mean_residue: per-trajectory MEAN over the state sequence."""
    vals = []
    for t in trajs:
        ws = t.get_world_state_sequence()
        if ws is not None and not torch.isnan(ws).any():
            v = float(agent.residue_field.evaluate_trajectory(ws).detach().mean().item())
            if not math.isnan(v):
                vals.append(v)
    return float(sum(vals) / len(vals)) if vals else 0.0


def readiness_gate(agent, env, obs_dict, seed: int, n_samples: int) -> Dict[str, Any]:
    """042's hippo_quality_gap -- the gate that must clear BEFORE any arm is lesioned.

    gap = mean_residue(random proposals) - mean_residue(hippocampal proposals)
    Positive = the trained generator navigates to LOWER-residue regions than chance,
    i.e. it actually acquired residue avoidance. A non-positive gap means there is no
    avoidance to lesion, so the whole probe would be vacuous -- and this driver reports
    that as a readiness failure rather than as evidence about MECH-131.

    Measured with BOTH channels intact (the lesion has not been applied yet).
    """
    hippo: List[float] = []
    rand: List[float] = []
    for i in range(n_samples):
        body, world = _obs(obs_dict)
        latent = agent.sense(body, world)
        theta_z = agent.theta_buffer.summary()
        z_self = latent.z_self.detach()
        torch.manual_seed(seed * 7919 + i)
        h = agent.hippocampal.propose_trajectories(
            z_world=theta_z, z_self=z_self, num_candidates=N_RANDOM_COMPARE,
        )
        torch.manual_seed(seed * 7919 + i)
        r = agent.e2.generate_candidates_random(
            initial_z_self=z_self, initial_z_world=theta_z,
            num_candidates=N_RANDOM_COMPARE, horizon=CANDIDATE_HORIZON,
            compute_action_objects=False,
        )
        hippo.append(_mean_residue_042(agent, h))
        rand.append(_mean_residue_042(agent, r))
        _flat, _harm, done, _info, obs_dict = env.step(
            int(torch.randint(0, ACTION_DIM, (1,)).item())
        )
        if done:
            _flat, obs_dict = env.reset()
    mh = sum(hippo) / len(hippo)
    mr = sum(rand) / len(rand)
    return {
        "hippo_mean_residue": mh,
        "random_mean_residue": mr,
        "hippo_quality_gap": mr - mh,
        "gap_clears": int((mr - mh) > 0.0),
        "n_gap_samples": len(hippo),
        "final_obs": obs_dict,
    }


def measure_arm(agent, env, obs_dict, arm: str, ch1: bool, ch2: bool,
                seed: int, n_states: int) -> Dict[str, Any]:
    """Read one arm off the ALREADY-TRAINED agent by toggling the live read gates.

    Weights are untouched -- only the two config flags move, and both are read live
    per call, so this is the same network with the anticipatory read removed.
    """
    hc = agent.hippocampal.config
    hc.terrain_prior_residue_channel_enabled = ch1
    hc.score_trajectory_residue_terrain_enabled = ch2
    assert hc.terrain_prior_residue_channel_enabled is ch1
    assert hc.score_trajectory_residue_terrain_enabled is ch2

    rf = agent.residue_field
    per_state: List[Dict[str, Any]] = []
    for s in range(n_states):
        body, world = _obs(obs_dict)
        latent = agent.sense(body, world)
        theta_z = agent.theta_buffer.summary()
        z_self = latent.z_self.detach()
        # Identical sampling noise across arms at matched (seed, probe index): the
        # arms must differ by the lesion, not by their random draws.
        torch.manual_seed(seed * 1000 + s)
        trajs = agent.hippocampal.propose_trajectories(theta_z, z_self=z_self)
        if trajs:
            sums, means = [], []
            for t in trajs:
                ws = t.get_world_state_sequence()
                if ws is None or torch.isnan(ws).any():
                    continue
                ev = rf.evaluate_trajectory(ws).detach()
                sums.append(float(ev.sum()))
                means.append(float(ev.mean()))
            scores = [float(agent.hippocampal._score_trajectory(t).detach())
                      for t in trajs]
            post_hoc = float(agent.e3.compute_residue_cost(trajs[0]).detach().sum())
            if sums:
                n = len(sums)
                mu = sum(sums) / n
                var = sum((x - mu) ** 2 for x in sums) / n
                per_state.append({
                    "probe_index": s,
                    "n_candidates": len(trajs),
                    "mean_candidate_residue": mu,
                    "mean_candidate_residue_normed": sum(means) / len(means),
                    # between-candidate SD: the noise the lesion effect must exceed
                    "between_candidate_sd": var ** 0.5,
                    "min_candidate_residue": min(sums),
                    "cem_score_spread": max(scores) - min(scores),
                    "post_hoc_residue_cost": post_hoc,
                    "candidate_residues": sums,
                })
        _flat, _harm, done, _info, obs_dict = env.step(
            int(torch.randint(0, ACTION_DIM, (1,)).item())
        )
        if done:
            _flat, obs_dict = env.reset()

    assert per_state, f"{arm}: no candidates proposed at any probe state -- vacuous"
    n = len(per_state)
    return {
        "arm": arm,
        "seed": seed,
        "ch1_terrain_prior_residue_channel_enabled": bool(ch1),
        "ch2_score_trajectory_residue_terrain_enabled": bool(ch2),
        "n_probe_states_scored": n,
        "residue_avoidance": sum(p["mean_candidate_residue"] for p in per_state) / n,
        "residue_avoidance_normed":
            sum(p["mean_candidate_residue_normed"] for p in per_state) / n,
        "between_candidate_sd": sum(p["between_candidate_sd"] for p in per_state) / n,
        "cem_score_spread": sum(p["cem_score_spread"] for p in per_state) / n,
        "cem_score_spread_max": max(p["cem_score_spread"] for p in per_state),
        "post_hoc_residue_cost": sum(p["post_hoc_residue_cost"] for p in per_state) / n,
        "post_hoc_residue_cost_min_abs":
            min(abs(p["post_hoc_residue_cost"]) for p in per_state),
        "per_probe_state": per_state,
        "final_obs": obs_dict,
    }


def run_seed(seed: int, dry_run: bool, warmup_episodes: int,
             steps_per_episode: int) -> Dict[str, Any]:
    """Train ONE intact agent, gate it, then read all three arms off it."""
    n_states = 3 if dry_run else N_PROBE_STATES
    n_gap = 3 if dry_run else 12
    agent, env, cfg = build_agent(True, True, seed)   # trained INTACT
    defect_flags = assert_defect_paths_inert(agent, cfg)

    warm = warmup_train(agent, env, seed, warmup_episodes, steps_per_episode)
    obs_dict = warm.pop("final_obs")

    gate = readiness_gate(agent, env, obs_dict, seed, n_gap)
    obs_dict = gate.pop("final_obs")

    arms: List[Dict[str, Any]] = []
    for arm, ch1, ch2 in ARMS:
        row = measure_arm(agent, env, obs_dict, arm, ch1, ch2, seed, n_states)
        obs_dict = row.pop("final_obs")
        row["total_residue"] = float(agent.residue_field.total_residue)
        row["num_harm_events"] = float(agent.residue_field.num_harm_events)
        row["defect_flags_observed"] = defect_flags
        arms.append(row)

    return {"seed": seed, "warmup": warm, "readiness": gate, "arms": arms,
            "agent": agent}


# ----------------------------------------------------------------------
def main(dry_run: bool = False, warmup_episodes: int = WARMUP_EPISODES,
         steps_per_episode: int = STEPS_PER_EPISODE):
    t0 = time.time()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    if dry_run:
        warmup_episodes, steps_per_episode = 3, 20

    seed_rows: List[Dict[str, Any]] = []
    all_agents: List[Any] = []
    for i, seed in enumerate(SEEDS):
        print(f"[seed {i+1}/{len(SEEDS)}] seed={seed} warmup={warmup_episodes}x"
              f"{steps_per_episode}", flush=True)
        with arm_cell(
            seed,
            config_slice=_config_slice(True, True, seed),
            script_path=Path(__file__),
            config_slice_declared=True,
        ) as cell:
            row = run_seed(seed, dry_run, warmup_episodes, steps_per_episode)
            all_agents.append(row.pop("agent"))
            cell.stamp(row)
        seed_rows.append(row)
        g = row["readiness"]
        print(f"    readiness hippo_quality_gap={g['hippo_quality_gap']:.6g} "
              f"(clears={g['gap_clears']})", flush=True)

    arm_results = [a for r in seed_rows for a in r["arms"]]

    def _arm(arm: str, seed: int) -> Dict[str, Any]:
        return next(a for a in arm_results if a["arm"] == arm and a["seed"] == seed)

    # ---- READINESS GATE (must clear before any arm is lesioned) --------
    gate_clears = all(r["readiness"]["gap_clears"] == 1 for r in seed_rows)
    gaps = [r["readiness"]["hippo_quality_gap"] for r in seed_rows]

    # ---- Preconditions --------------------------------------------------
    p1_harm = all(a["num_harm_events"] > 0 for a in arm_results)
    p2_storage = all(
        _arm("ARM_1_intact", s)["total_residue"]
        == _arm("ARM_2_ch1_lesion", s)["total_residue"]
        == _arm("ARM_3_complete_lesion", s)["total_residue"]
        for s in SEEDS
    )   # one trained agent per seed, so this is exact by construction -- asserted anyway
    p3_post_hoc = all(a["post_hoc_residue_cost_min_abs"] > 0.0 for a in arm_results)
    p4_spread = (
        all(_arm(a, s)["cem_score_spread_max"] > 0.0
            for a in ("ARM_1_intact", "ARM_2_ch1_lesion") for s in SEEDS)
        and all(_arm("ARM_3_complete_lesion", s)["cem_score_spread_max"] == 0.0
                for s in SEEDS)
    )
    preconditions_met = bool(p1_harm and p2_storage and p3_post_hoc and p4_spread)

    # ---- C1: DV1 ordinal ordering, PER SEED -----------------------------
    per_seed: List[Dict[str, Any]] = []
    for s in SEEDS:
        r1, r2, r3 = (_arm("ARM_1_intact", s), _arm("ARM_2_ch1_lesion", s),
                      _arm("ARM_3_complete_lesion", s))
        a1, a2, a3 = (r1["residue_avoidance"], r2["residue_avoidance"],
                      r3["residue_avoidance"])
        noise = max(r1["between_candidate_sd"], r3["between_candidate_sd"])
        eff = a3 - a1
        per_seed.append({
            "seed": s,
            "residue_avoidance_arm1_intact": a1,
            "residue_avoidance_arm2_ch1_lesion": a2,
            "residue_avoidance_arm3_complete_lesion": a3,
            "effect_arm3_minus_arm1": eff,
            "effect_arm3_minus_arm2": a3 - a2,
            "effect_arm2_minus_arm1": a2 - a1,
            "relative_effect_arm3_vs_arm1": (eff / abs(a3)) if a3 else None,
            "between_candidate_sd": noise,
            # The diagnosis that stopped the pre-warmup revision, now a recorded
            # readout rather than a one-off measurement.
            "effect_over_noise": (eff / noise) if noise else None,
            "hippo_quality_gap": _seed_gap(seed_rows, s),
            "c1_arm1_below_arm3": int(a1 < a3),
            "arm2_below_arm3": int(a2 < a3),
        })
    c1_met = all(p["c1_arm1_below_arm3"] == 1 for p in per_seed)
    c1_seeds_clearing = sum(p["c1_arm1_below_arm3"] for p in per_seed)

    # ---- C2: DV3 ordering dissociation ---------------------------------
    c2_flags = [int(_arm("ARM_3_complete_lesion", s)["cem_score_spread_max"] == 0.0
                    and _arm("ARM_3_complete_lesion", s)["post_hoc_residue_cost_min_abs"] > 0.0)
                for s in SEEDS]
    c2_met = all(f == 1 for f in c2_flags)

    # ---- DV2: gradedness, reported with its sign ------------------------
    rhos = [{"seed": s,
             "gradedness_rho": gradedness_rho(_arm("ARM_1_intact", s),
                                              _arm("ARM_3_complete_lesion", s))}
            for s in SEEDS]
    rho_vals = [r["gradedness_rho"] for r in rhos if r["gradedness_rho"] is not None]
    rho_mean = (sum(rho_vals) / len(rho_vals)) if rho_vals else None
    rho_positive_seeds = sum(1 for v in rho_vals if v > 0)

    eon = [p["effect_over_noise"] for p in per_seed if p["effect_over_noise"] is not None]
    eon_mean = (sum(eon) / len(eon)) if eon else None

    # A readiness failure is NOT a claim-negative: it means the generator never
    # acquired the avoidance the lesion is supposed to remove.
    if not gate_clears:
        outcome = "FAIL"
        verdict_reason = "readiness_gate_failed"
    elif not preconditions_met:
        outcome = "FAIL"
        verdict_reason = "preconditions_failed"
    else:
        outcome = "PASS" if (c1_met and c2_met) else "FAIL"
        verdict_reason = "criteria" if outcome == "PASS" else "criteria_not_met"

    interpretation = (
        "DIAGNOSTIC -- excluded from governance confidence scoring; does not move "
        "MECH-131's status. Purpose: supply the measured quantities governance needs "
        "to author MECH-131's missing what_would_answer. "
        f"Readiness (042 hippo_quality_gap, hippocampal vs random proposal residue): "
        f"{gaps} -- {'CLEARS' if gate_clears else 'DOES NOT CLEAR'}. "
        f"C1 (residue_avoidance ARM_1 < ARM_3 per seed): {c1_seeds_clearing}/{len(SEEDS)}. "
        f"C2 (ARM_3 generation residue-blind while post-hoc live): "
        f"{'met' if c2_met else 'NOT met'}. "
        f"Effect/noise (|ARM_3-ARM_1| over between-candidate SD) mean {eon_mean} -- "
        "the pre-warmup revision of this driver measured ~0.01 here, which is why the "
        "warmup exists; a value below ~1 means the DV still cannot resolve its own "
        "manipulation and the verdict should not be read as being about MECH-131. "
        f"Gradedness rho mean {rho_mean} ({rho_positive_seeds}/{len(rho_vals)} seeds "
        "positive) -- reported, not gated. "
        f"Verdict reason: {verdict_reason}. "
        "Stochastic replication: 3 seeds. World-family replication: 1 family -- "
        "ECOLOGICAL TRANSFER UNTESTED (GOV-ECOL-1); seeds alone do not license "
        "'general' or 'robust'."
    )

    readout = flat_readout({
        "readiness_gate_clears_all_seeds": int(gate_clears),
        "hippo_quality_gap_mean": sum(gaps) / len(gaps),
        "hippo_quality_gap_min": min(gaps),
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
        "between_candidate_sd_mean":
            sum(p["between_candidate_sd"] for p in per_seed) / len(per_seed),
        "effect_over_noise_mean": eon_mean if eon_mean is not None else 0.0,
        "gradedness_rho_mean": rho_mean if rho_mean is not None else 0.0,
        "gradedness_rho_seeds_positive": rho_positive_seeds,
        "gradedness_rho_seeds_defined": len(rho_vals),
        "terrain_loss_early_mean": _loss_mean(seed_rows, "terrain_loss_early"),
        "terrain_loss_late_mean": _loss_mean(seed_rows, "terrain_loss_late"),
        "warmup_episodes": warmup_episodes,
        "steps_per_episode": steps_per_episode,
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
        "verdict_reason": verdict_reason,
        "readout": readout,
        "criteria": {
            "READINESS_hippo_quality_gap_per_seed": {
                "measured": min(gaps), "threshold": 0.0, "met": int(gate_clears),
                "note": "042's hippo-vs-random proposal residue; must clear BEFORE any "
                        "arm is lesioned. A failure here is a readiness result, NOT "
                        "evidence about MECH-131.",
            },
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
            "DIAGNOSTIC_effect_over_noise": {
                "measured": eon_mean, "threshold": None,
                "note": "|ARM_3-ARM_1| / between-candidate SD. Reported, not gated. "
                        "Pre-warmup revision measured ~0.01; below ~1 the DV cannot "
                        "resolve its own manipulation.",
            },
        },
        "replication": {
            "stochastic_seeds": len(SEEDS), "world_families": 1,
            "ecological_transfer": "untested",
        },
        "per_seed": per_seed,
        "gradedness": rhos,
        "warmup": [{"seed": r["seed"], **r["warmup"]} for r in seed_rows],
        "readiness": [{"seed": r["seed"], **r["readiness"]} for r in seed_rows],
        "arm_results": arm_results,
        "params": {
            "seeds": SEEDS, "arms": [a[0] for a in ARMS],
            "grid_size": GRID_SIZE, "num_hazards": NUM_HAZARDS,
            "num_resources": NUM_RESOURCES, "self_dim": SELF_DIM,
            "world_dim": WORLD_DIM, "action_dim": ACTION_DIM,
            "warmup_episodes": warmup_episodes,
            "steps_per_episode": steps_per_episode,
            "terrain_lr": TERRAIN_LR,
            "n_random_compare": N_RANDOM_COMPARE,
            "candidate_horizon": CANDIDATE_HORIZON,
            "n_probe_states": N_PROBE_STATES,
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
        manifest, out_dir, dry_run=dry_run, config=manifest["params"],
        seeds=SEEDS, script_path=Path(__file__), started_at=t0, agent=all_agents,
    )
    print(f"Outcome: {outcome} ({verdict_reason})", flush=True)
    print(f"wrote: {out_path}", flush=True)
    manifest["_manifest_path"] = str(out_path)
    return manifest, out_path


def _seed_gap(seed_rows: List[Dict[str, Any]], seed: int) -> float:
    return next(r["readiness"]["hippo_quality_gap"] for r in seed_rows
                if r["seed"] == seed)


def _loss_mean(seed_rows: List[Dict[str, Any]], key: str) -> float:
    vals = [r["warmup"][key] for r in seed_rows if r["warmup"].get(key) is not None]
    return (sum(vals) / len(vals)) if vals else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-1061: MECH-131 three-arm anticipatory-residue lesion "
                    "(intact / CH1-lesion / CH1+CH2-lesion) -- DIAGNOSTIC"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Minimal wiring check (3 warmup episodes x 20 steps)")
    parser.add_argument("--warmup-episodes", type=int, default=WARMUP_EPISODES,
                        help="042's number by default; lower ONLY for a scale probe")
    parser.add_argument("--steps-per-episode", type=int, default=STEPS_PER_EPISODE)
    args = parser.parse_args()

    manifest, out_path = main(dry_run=args.dry_run,
                              warmup_episodes=args.warmup_episodes,
                              steps_per_episode=args.steps_per_episode)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        run_id=manifest["run_id"],
        queue_id=QUEUE_ID,
        dry_run=args.dry_run,
    )
