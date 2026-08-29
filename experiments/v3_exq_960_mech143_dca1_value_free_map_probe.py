#!/opt/local/bin/python3
"""
V3-EXQ-960: MECH-143 Dorsal CA1 Value-Free Map Probe -- Discriminative Pair

experiment_purpose: evidence

MECH-143 claim: dorsal CA1 implements value-free spatial mapping -- the
hippocampal map representation (z_world) is insensitive to a change in GOAL
VALUE at a fixed goal LOCATION, supporting ARC-007's no-new-value-computation
constraint for the trajectory proposal module (Duvelle et al., Curr Biol 2019:
place fields are geometrically stable and insensitive to reward-value changes
at a constant goal location). Under Q-020 Resolution A, value/wanting lives in
the frontal z_goal channel (REEAgent.update_z_goal, fed from benefit_exposure),
architecturally separate from the ResidueField/z_world "map" HippocampalModule
navigates (SD-005: ResidueField operates over z_world, not z_gamma/z_self).
MECH-143 is the specific prediction that this separation holds at the
representation level: z_world sampled at fixed spatial positions should not
drift when goal VALUE changes, while it SHOULD (and structurally must) drift
when the goal LOCATION itself changes -- the latter is the positive control
proving the probe is sensitive to genuine structural change, not merely inert.

Discriminative pair design (EVB-0598, EXP-0058):

  VALUE_CHANGE (primary condition):
    Goal LOCATION fixed for the whole run (hazard + resource cells never
    move). Goal VALUE (env.resource_benefit, the contact-reward magnitude)
    is LOW in phase A, then raised to HIGH in phase B. Raw observations at
    every probe position are BIT-IDENTICAL between phase A and phase B (the
    environment layout never changes) -- so the ONLY possible source of
    z_world drift here is learned-weight drift induced by exposure to the
    higher-value contact events during phase B training. This isolates
    exactly the channel MECH-143 makes a prediction about.
    Prediction: LOW map_drift (map geometry stays stable under value change).

  LOCATION_CHANGE (control/ablation condition):
    Goal VALUE fixed (constant, LOW, matching VALUE_CHANGE's phase-A value)
    for the whole run. Goal LOCATION (the resource cell) moves to a new
    fixed position at the same phase transition point. Raw observations at
    several probe positions genuinely differ between phase A and phase B, so
    this arm demonstrates the probe CAN detect a real structural map change
    -- without this arm, a null VALUE_CHANGE result would be uninterpretable
    (value-invariant, or just an insensitive probe?).
    Prediction: HIGH map_drift (probe is sensitive; the map DOES reorganise
    when the physical goal location changes).

Probe mechanism: CausalGridWorldV2.reset_to(agent_pos, hazards, resources) is
the SD-029/EXQ-433a deterministic scripted-eval entry point -- it rebuilds the
full obs_dict (including proximity fields) at an exact grid cell with no RNG.
At a fixed set of 8 interior probe positions (none coincide with the hazard
or either resource location), we snapshot z_world at the end of phase A (PRE)
and again at the end of phase B (POST) via agent.reset() + one agent.sense()
call per position (no stepping -- agent.reset() re-initialises the latent
stack's z_world prior to zero via latent_stack.init_state, so the snapshot
reflects only current encoder weights + the raw observation at that cell, not
trajectory history). map_drift(seed, arm) = mean cosine distance between the
PRE and POST z_world vector at each probe position, averaged across the 8
positions.

DV-symmetry-invariance declaration (per CLAUDE.md /queue-experiment Step 3):
  VALUE_CHANGE: raw observations are bit-identical pre/post by construction
    (layout never changes), so map_drift here is NOT a null-by-construction
    DV -- it is exactly the intended lever: any nonzero reading is entirely
    attributable to weight drift from value-differentiated training, which is
    the mechanism MECH-143 makes a claim about. Not invariant under any
    symmetry; a zero reading is a genuine (not vacuous) finding.
  LOCATION_CHANGE: raw observations at several probe cells differ pre/post
    (the environment structurally changed), so this arm is NOT expected to be
    invariant under the manipulation -- it is the positive control.

Pre-registered thresholds (no prior V3 run of this probe exists to calibrate
against -- GOV-REUSE-1 checked: no manifest in evidence/experiments/ records a
z_world-drift-under-value-vs-location-change readout on this substrate, see
Step 2.4 note in the queue entry; thresholds below are the author's
best-effort priors, not empirically tuned):
  C1: VALUE_CHANGE map_drift <  0.20 in >= 2/3 seeds (map stays stable).
  C2: LOCATION_CHANGE map_drift > 0.35 in >= 2/3 seeds (probe is sensitive).
  C3: LOCATION_CHANGE map_drift > VALUE_CHANGE map_drift, paired per seed,
      in >= 2/3 seeds (within-seed discriminative ordering; more robust to
      absolute-scale differences in z_world magnitude across seeds than C1/C2
      alone).

PASS: C1 AND C2 AND C3.

Non-degeneracy guard (evidence-purpose runs are still subject to the
Non-degeneracy scoring net, CLAUDE.md /queue-experiment Step 3): each cell's
PRE-snapshot cross-position dispersion (mean pairwise cosine distance among
the 8 probe positions' z_world vectors) must clear MIN_DISPERSION, else the 8
probe positions are not even distinguishable from one another and any low
map_drift reading would be a representation-collapse artifact, not evidence
of value-invariance. Written to the manifest as non_degenerate / per-cell
zworld_probe_dispersion via _metrics.check_degeneracy.

Phase structure (per seed x arm cell):
  P0   (60 ep): encoder warmup, fixed layout, LOW value. No probe snapshot.
  P1A  (60 ep): continued training, phase-A layout/value held. PRE snapshot
                taken at the end of this phase (no gradient step during the
                snapshot itself).
  P1B  (60 ep): continued training, phase-B manipulation applied at entry
                (VALUE_CHANGE: resource_benefit raised; LOCATION_CHANGE:
                resource repositioned). POST snapshot taken at the end.

No downstream head is trained on z_world in this script (the probe reads raw
z_world vectors and computes a closed-form cosine-distance statistic, not a
learned discriminator), so the phased-training encoder-freeze rule does not
apply here in the same shape it does for EXQ-375's AUROC head; P0/P1A/P1B
still separate warmup from the pre/post manipulation windows as the natural
structure for this design.

Claims: MECH-143
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_960_mech143_dca1_value_free_map_probe"
CLAIM_IDS          = ["MECH-143"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEEDS = [42, 7, 13]
ARMS  = ["VALUE_CHANGE", "LOCATION_CHANGE"]

P0_EPISODES  = 60   # encoder warmup
P1A_EPISODES = 60   # phase A (pre-manipulation) -- PRE snapshot at the end
P1B_EPISODES = 60   # phase B (post-manipulation) -- POST snapshot at the end
STEPS_PER_EP = 150

GRID_SIZE = 8

# Fixed deterministic layout (reset_to -- no RNG placement).
AGENT_START   = (1, 1)
HAZARD_POS    = [(4, 4)]          # never moves in either arm
RESOURCE_POS_A = (6, 6)           # phase-A goal location
RESOURCE_POS_B = (2, 2)           # phase-B goal location, LOCATION_CHANGE only
LOW_VALUE  = 0.20                 # phase-A resource_benefit, both arms
HIGH_VALUE = 1.00                 # phase-B resource_benefit, VALUE_CHANGE only

# Probe grid -- interior cells distinct from AGENT_START / HAZARD_POS /
# RESOURCE_POS_A / RESOURCE_POS_B.
PROBE_POSITIONS = [
    (1, 3), (1, 5), (3, 1), (3, 3), (3, 5), (5, 1), (5, 3), (5, 5),
]

LR = 3e-4

# Pre-registered thresholds
C1_VALUE_DRIFT_MAX    = 0.20
C2_LOCATION_DRIFT_MIN = 0.35
MIN_SEEDS_PASS         = 2   # out of 3
MIN_DISPERSION         = 0.02

DRY_RUN_EPISODES = 3
DRY_RUN_STEPS    = 15


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=1,
        num_hazards=1,
        hazard_harm=0.15,
        resource_benefit=LOW_VALUE,
        resource_respawn_on_consume=False,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.05,
        proximity_approach_threshold=0.15,
        use_proxy_fields=True,
    )


def _make_agent(env: CausalGridWorldV2, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        use_event_classifier=True,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=0.5,
        drive_weight=2.0,
        z_goal_enabled=True,
    )
    return REEAgent(config)


def _resource_pos_for(arm: str, phase: str) -> tuple:
    if arm == "LOCATION_CHANGE" and phase == "P1B":
        return RESOURCE_POS_B
    return RESOURCE_POS_A


def _resource_benefit_for(arm: str, phase: str) -> float:
    if arm == "VALUE_CHANGE" and phase == "P1B":
        return HIGH_VALUE
    return LOW_VALUE


# ---------------------------------------------------------------------------
# Probe: snapshot z_world at fixed positions
# ---------------------------------------------------------------------------

def _probe_zworld(env: CausalGridWorldV2, agent: REEAgent, arm: str, phase: str) -> torch.Tensor:
    """Snapshot z_world at each PROBE_POSITIONS cell for the CURRENT layout of
    (arm, phase). No stepping, no gradient. agent.reset() clears the latent
    stack's z_world prior each call so the reading depends only on current
    encoder weights + the raw observation at that cell.
    """
    resource_pos = _resource_pos_for(arm, phase)
    device = agent.device
    vecs: List[torch.Tensor] = []
    for pos in PROBE_POSITIONS:
        _, obs_dict = env.reset_to(pos, HAZARD_POS, [resource_pos])
        agent.reset()
        obs_body  = obs_dict["body_state"].to(device)
        obs_world = obs_dict["world_state"].to(device)
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        vecs.append(latent.z_world.detach().cpu().squeeze(0).clone())
    return torch.stack(vecs)  # [n_probe, world_dim]


def _mean_cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    sims = nn.functional.cosine_similarity(a, b, dim=-1)  # [n_probe]
    return float((1.0 - sims).mean().item())


def _mean_pairwise_cosine_distance(vecs: torch.Tensor) -> float:
    n = vecs.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = nn.functional.cosine_similarity(
                vecs[i].unsqueeze(0), vecs[j].unsqueeze(0), dim=-1
            ).item()
            total += (1.0 - sim)
            count += 1
    return total / count if count else 0.0


# ---------------------------------------------------------------------------
# Single run cell (seed x arm)
# ---------------------------------------------------------------------------

def run_cell(seed: int, arm: str, zg_acc: ZGoalStreamAccumulator, dry_run: bool = False) -> Dict:
    total_p0  = DRY_RUN_EPISODES if dry_run else P0_EPISODES
    total_p1a = DRY_RUN_EPISODES if dry_run else P1A_EPISODES
    total_p1b = DRY_RUN_EPISODES if dry_run else P1B_EPISODES
    steps_per = DRY_RUN_STEPS    if dry_run else STEPS_PER_EP

    print(f"  Seed {seed} Condition {arm}", flush=True)

    env   = _make_env(seed)
    agent = _make_agent(env, seed)
    device = agent.device
    optimizer = optim.Adam(list(agent.parameters()), lr=LR)

    prev_ttype = "none"
    total_eps = total_p0 + total_p1a + total_p1b
    ep_counter = 0

    def _train_phase(phase: str, n_eps: int):
        nonlocal prev_ttype, ep_counter
        resource_pos = _resource_pos_for(arm, phase)
        benefit = _resource_benefit_for(arm, phase)
        env.resource_benefit = benefit
        for _ in range(n_eps):
            _, obs_dict = env.reset_to(AGENT_START, HAZARD_POS, [resource_pos])
            agent.reset()
            for _step in range(steps_per):
                obs_body  = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)

                z_self_prev: Optional[torch.Tensor] = None
                if agent._current_latent is not None:
                    z_self_prev = agent._current_latent.z_self.detach().clone()

                latent = agent.sense(obs_body, obs_world)
                ticks  = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks.get("e1_tick", True)
                    else torch.zeros(1, 32, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action     = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())

                benefit_raw = float(obs_body.flatten()[11].item()) if obs_body.shape[-1] > 11 else 0.0
                drive_level = REEAgent.compute_drive_level(obs_body)
                agent.update_z_goal(benefit_raw, drive_level)

                flat_next, harm_signal, done, info, obs_dict_next = env.step(action_idx)
                ttype = info.get("transition_type", "none")
                agent.update_residue(float(harm_signal))

                if z_self_prev is not None:
                    agent.record_transition(z_self_prev, action, latent.z_self.detach())

                optimizer.zero_grad()
                e1_loss = agent.compute_prediction_loss()
                e2_loss = agent.compute_e2_loss()
                loss    = e1_loss + e2_loss

                rfv = obs_dict.get("resource_field_view", None)
                if rfv is not None:
                    rp_t = float(rfv.max().item())
                    loss = loss + agent.compute_resource_proximity_loss(rp_t, latent)

                latent2 = agent.sense(obs_body, obs_world)
                ec_loss = agent.compute_event_contrastive_loss(prev_ttype, latent2)
                loss    = loss + ec_loss

                benefit_val = max(0.0, float(info.get("benefit_exposure", 0.0)))
                if benefit_val > 0:
                    benefit_t = torch.tensor([[benefit_val]], dtype=torch.float32, device=device)
                    loss = loss + agent.compute_benefit_eval_loss(benefit_t)

                if loss.requires_grad:
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(agent.parameters()), 1.0)
                    optimizer.step()

                prev_ttype = ttype
                obs_dict   = obs_dict_next
                if done:
                    break

            ep_counter += 1
            if ep_counter % 30 == 0 or ep_counter == total_eps:
                print(
                    f"    [train] seed={seed} {arm} ep {ep_counter}/{total_eps} phase={phase}",
                    flush=True,
                )

    _train_phase("P0", total_p0)
    _train_phase("P1A", total_p1a)

    pre_vecs = _probe_zworld(env, agent, arm, "P1A")
    pre_dispersion = _mean_pairwise_cosine_distance(pre_vecs)

    _train_phase("P1B", total_p1b)

    post_vecs = _probe_zworld(env, agent, arm, "P1B")
    map_drift = _mean_cosine_distance(pre_vecs, post_vecs)

    zg_acc.observe(agent)

    cell_ok = pre_dispersion >= MIN_DISPERSION
    print(
        f"  verdict: {'PASS' if cell_ok else 'FAIL'} "
        f"(map_drift={map_drift:.4f} pre_dispersion={pre_dispersion:.4f})",
        flush=True,
    )

    return {
        "seed": seed,
        "arm": arm,
        "map_drift": map_drift,
        "pre_dispersion": pre_dispersion,
        "n_probe_positions": len(PROBE_POSITIONS),
    }


# ---------------------------------------------------------------------------
# Criteria evaluation
# ---------------------------------------------------------------------------

def evaluate_criteria(all_results: List[Dict]) -> Dict:
    by_arm: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_results:
        by_arm[r["arm"]].append(r)

    value_list    = sorted(by_arm.get("VALUE_CHANGE", []),    key=lambda x: x["seed"])
    location_list = sorted(by_arm.get("LOCATION_CHANGE", []), key=lambda x: x["seed"])

    c1_seeds = sum(r["map_drift"] < C1_VALUE_DRIFT_MAX for r in value_list)
    c1_pass  = c1_seeds >= MIN_SEEDS_PASS

    c2_seeds = sum(r["map_drift"] > C2_LOCATION_DRIFT_MIN for r in location_list)
    c2_pass  = c2_seeds >= MIN_SEEDS_PASS

    by_seed_value    = {r["seed"]: r["map_drift"] for r in value_list}
    by_seed_location = {r["seed"]: r["map_drift"] for r in location_list}
    common_seeds = sorted(set(by_seed_value) & set(by_seed_location))
    c3_seeds = sum(
        by_seed_location[s] > by_seed_value[s] for s in common_seeds
    )
    c3_pass = c3_seeds >= MIN_SEEDS_PASS

    overall_pass = c1_pass and c2_pass and c3_pass

    pairwise_deltas = [
        {"seed": s, "location_minus_value_drift": by_seed_location[s] - by_seed_value[s]}
        for s in common_seeds
    ]

    return {
        "c1_value_stable_pass": c1_pass,
        "c1_seeds_pass": c1_seeds,
        "c1_value_drifts": [r["map_drift"] for r in value_list],
        "c1_threshold_max": C1_VALUE_DRIFT_MAX,
        "c2_location_sensitive_pass": c2_pass,
        "c2_seeds_pass": c2_seeds,
        "c2_location_drifts": [r["map_drift"] for r in location_list],
        "c2_threshold_min": C2_LOCATION_DRIFT_MIN,
        "c3_paired_ordering_pass": c3_pass,
        "c3_seeds_pass": c3_seeds,
        "pairwise_deltas": pairwise_deltas,
        "overall_pass": overall_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"{EXPERIMENT_TYPE}_dry_{ts}_v3" if args.dry_run else f"{EXPERIMENT_TYPE}_{ts}_v3"
    )
    print(f"V3-EXQ-960 start: {run_id}", flush=True)

    zg_acc = ZGoalStreamAccumulator()
    all_results: List[Dict] = []

    for seed in SEEDS:
        print(f"\n--- seed {seed} ---", flush=True)
        for arm in ARMS:
            result = run_cell(seed, arm, zg_acc, dry_run=args.dry_run)
            all_results.append(result)

    criteria = evaluate_criteria(all_results)
    outcome  = "PASS" if criteria["overall_pass"] else "FAIL"

    print(f"\n=== V3-EXQ-960 {outcome} ===", flush=True)
    print(
        f"C1 value_stable: {criteria['c1_value_stable_pass']} "
        f"({criteria['c1_seeds_pass']}/{len(SEEDS)} seeds) "
        f"drifts={criteria['c1_value_drifts']} threshold<{criteria['c1_threshold_max']}",
        flush=True,
    )
    print(
        f"C2 location_sensitive: {criteria['c2_location_sensitive_pass']} "
        f"({criteria['c2_seeds_pass']}/{len(SEEDS)} seeds) "
        f"drifts={criteria['c2_location_drifts']} threshold>{criteria['c2_threshold_min']}",
        flush=True,
    )
    print(
        f"C3 paired_ordering: {criteria['c3_paired_ordering_pass']} "
        f"({criteria['c3_seeds_pass']}/{len(SEEDS)} seeds) "
        f"deltas={criteria['pairwise_deltas']}",
        flush=True,
    )

    degeneracy = check_degeneracy({
        "zworld_probe_dispersion": {
            "values": [r["pre_dispersion"] for r in all_results],
            "floor": MIN_DISPERSION,
        },
    })

    interpretation_label = (
        "value_free_map_supported" if criteria["overall_pass"]
        else "value_free_map_not_supported"
    )

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_class": "discriminative_pair",
        "evidence_direction": "supports" if criteria["overall_pass"] else "does_not_support",
        "outcome": outcome,
        "registered_thresholds": {
            "c1_value_drift_max": C1_VALUE_DRIFT_MAX,
            "c2_location_drift_min": C2_LOCATION_DRIFT_MIN,
            "min_seeds_pass": MIN_SEEDS_PASS,
            "min_dispersion": MIN_DISPERSION,
        },
        "criteria": criteria,
        "results_per_arm": all_results,
        "interpretation": {
            "label": interpretation_label,
        },
        **degeneracy,
        "summary": (
            f"MECH-143 dCA1 value-free map probe (discriminative pair). "
            f"Scenario: goal LOCATION fixed while goal VALUE changes "
            f"(VALUE_CHANGE, map_drift={criteria['c1_value_drifts']}) vs goal VALUE "
            f"fixed while goal LOCATION changes "
            f"(LOCATION_CHANGE, map_drift={criteria['c2_location_drifts']}). "
            f"Pairwise deltas (location_minus_value_drift): {criteria['pairwise_deltas']}. "
            f"Outcome: {outcome}. Interpretation: "
            + (
                "the hippocampal map (z_world) stays geometrically stable under a "
                "goal-value change at fixed location, and the probe is demonstrably "
                "sensitive to a genuine location change -- consistent with MECH-143's "
                "value-free spatial mapping prediction and ARC-007's no-new-value-"
                "computation constraint."
                if criteria["overall_pass"] else
                "either the map drifted under a value-only change, or the probe failed "
                "to demonstrate sensitivity to a genuine location change (or both) -- "
                "MECH-143 not supported at this scale/design."
            )
        ),
        "config": {
            "seeds": SEEDS,
            "arms": ARMS,
            "p0_episodes": P0_EPISODES,
            "p1a_episodes": P1A_EPISODES,
            "p1b_episodes": P1B_EPISODES,
            "steps_per_ep": STEPS_PER_EP,
            "grid_size": GRID_SIZE,
            "agent_start": AGENT_START,
            "hazard_pos": HAZARD_POS,
            "resource_pos_a": RESOURCE_POS_A,
            "resource_pos_b": RESOURCE_POS_B,
            "low_value": LOW_VALUE,
            "high_value": HIGH_VALUE,
            "probe_positions": PROBE_POSITIONS,
        },
        "timestamp_utc": ts,
    }

    out_path = write_flat_manifest(
        output,
        dry_run=bool(args.dry_run),
        config=output.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
        z_goal_stream_stats=zg_acc.stats(),
    )
    print(f"Results -> {out_path}", flush=True)

    return outcome, out_path, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _out_path, _dry_run = main()
    from experiment_protocol import emit_outcome
    emit_outcome(outcome=_outcome, manifest_path=str(_out_path), dry_run=_dry_run)
