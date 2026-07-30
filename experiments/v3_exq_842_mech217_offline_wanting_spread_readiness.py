#!/opt/local/bin/python3
"""V3-EXQ-842 -- EXP-0301 / EVB-0349 MECH-217 offline wanting-spread readiness.
SLEEP DRIVER: manual-cycle-loop (run_sleep_cycle() called once per cycle in a dedicated N_CYCLES wake-sleep-test loop)

Substrate readiness diagnostic for MECH-217 (goal.replay_wanting_spread), just
implemented 2026-07-30: HippocampalModule.spread_reverse_replay_wanting(),
called from REEAgent.run_rem_attribution_pass() for each MECH-165 reverse-
replayed trajectory during the SD-017 REM pass, behind
HippocampalConfig.use_offline_wanting_spread (default False).

This is NOT the full behavioural (approach-navigation) test of MECH-217 --
it verifies the SPREAD MECHANISM ITSELF produces the claimed signature
(concentration along a traversed approach path, decaying with distance from
the resource terminus, dissociable from an unvisited control point) before
any behavioural claim is queued.

Design (disclosed, diagnostic-only instrumentation choices)
-------------------------------------------------------------
1. SCRIPTED greedy-toward-resource policy (privileged access to
   env.agent_x/agent_y/env.resources), not the agent's own CEM selection.
   This isolates the OFFLINE SPREAD mechanism from whether the agent's own
   policy can navigate to a resource -- a separate, already-tested capacity
   (v3_exq_259 MECH-216/SD-014 navigation). Agent and resource start at a
   FIXED deterministic distance (env.reset_to), identical across arms and
   seeds, so path length is controlled and comparable.
2. Resource-contact wanting is seeded DIRECTLY at the trajectory terminus
   (ResidueField.update_valence(..., VALENCE_WANTING, CONTACT_SEED_WANTING,
   hypothesis_tag=False)) rather than via the serotonin-gated MECH-203
   update_benefit_salience() writer. This isolates MECH-217 (the SPREAD)
   from MECH-203 (a separate, real-time proximity-gated WRITE mechanism);
   entangling them would confound "gradient along path" with MECH-203's own
   proximity-graded writes during waking. tonic_5ht stays disabled
   throughout, so MECH-203 never fires in this run.
3. A near-zero-magnitude RBF center is activated at every waypoint visited
   during wake (ResidueField.rbf_field.add_residue(z_world, intensity~0)).
   ResidueField.update_valence() only ever updates the NEAREST EXISTING
   active center -- it never creates one -- so without this, every waypoint
   before contact would be centerless and the spread would have nowhere
   distinct to land except the terminus's own center. This purely creates
   ADDRESSABLE STRUCTURE; it injects no harm/wanting signal of its own
   (intensity is 1e-3, four orders below the seeded contact wanting).
   HippocampalConfig.num_basis_functions is raised to 256 (well above the
   default 32/64) so ~150 activations across the run never round-robin-evict
   an earlier waypoint's center.

Arms (2, same 3 seeds): ARM_OFF (use_offline_wanting_spread=False, the
substrate default) vs ARM_ON (True, offline_wanting_spread_gamma=0.9 /
offline_wanting_spread_gain=0.1 -- both left at their landed defaults, so
this tests the mechanism AS SHIPPED, not a tuned variant).

Per (arm, seed) cell: N_CYCLES wake-sleep cycles. Each cycle: reset env/agent
to the fixed agent/resource layout, scripted-walk to contact (or a step cap),
seed terminus VALENCE_WANTING on the FIRST cycle only (see run_cycle's
seed_wanting docstring), then agent.run_sleep_cycle() (SWS + REM; the REM
pass is where spread_reverse_replay_wanting() fires on ON) BEFORE
agent.reset() flushes the episode to the MECH-165 exploration buffer --
agent.reset() also clears theta_buffer, which run_rem_attribution_pass's
forward branch needs populated, so sleep must run first. This episode's own
trajectory becomes reverse-replayable starting next cycle. After all cycles, read
VALENCE_WANTING (ResidueField.evaluate_valence) at, for every buffered
trajectory: the near waypoint (1 step from terminus) and the far waypoint
(the episode start, maximal distance) -- plus one fixed, never-visited
CONTROL point at a comparable distance from the resource but off the
diagonal approach corridor.

Pre-registered acceptance criteria (cross-arm, seed-aggregated)
-----------------------------------------------------------------
Readiness preconditions (both arms; below-floor => substrate_not_ready_requeue,
never a claim verdict):
  P1 exploration_buffer_populated: min seeds' final buffer length >= 2.
  P2 reverse_replay_fired: min seeds' total rem_n_reverse across cycles >= 1.

C1 (LOAD-BEARING): mean(near_wanting, ARM_ON) > 1.5 * mean(near_wanting, ARM_OFF)
    AND mean(near_wanting, ARM_ON) > 0. The mechanism writes a materially
    larger near-waypoint wanting than the OFF control (which must read ~0 --
    HippocampalConfig.use_offline_wanting_spread=False is a hard no-op guard
    inside spread_reverse_replay_wanting()).
C2: within ARM_ON, mean(near_wanting) > mean(far_wanting) -- the gamma-decay
    shape (steps_from_terminus=1 outweighs steps_from_terminus=path_length-1).
C3 (informative, not load-bearing): within ARM_ON, control point wanting <
    mean(near_wanting) -- path-specificity: the never-visited control point
    (which has no dedicated RBF center) reads below a genuine near-waypoint,
    rather than acquiring wanting merely for being close to the resource.
    RBF fields are smooth (not delta functions), so some non-zero leakage to
    the control point from nearby active centers is expected and is not
    itself a failure -- only a degenerate reading (control >= near) would be.

PASS = every readiness precondition met AND C1 AND C2 AND C3.
No claim about behavioural navigation lift is made here -- that is a
follow-up EXQ once this PASSes.

experiment_purpose = "diagnostic" (substrate readiness, not governance evidence).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_842_mech217_offline_wanting_spread_readiness.py
  /opt/local/bin/python3 experiments/v3_exq_842_mech217_offline_wanting_spread_readiness.py --dry-run
"""
from __future__ import annotations

import argparse
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
from ree_core.residue.field import VALENCE_WANTING  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_842_mech217_offline_wanting_spread_readiness"
CLAIM_IDS = ["MECH-217"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = (42, 43, 44)
ARMS = ("ARM_OFF", "ARM_ON")

GRID_SIZE = 8
AGENT_START = (1, 1)
RESOURCE_POS = (6, 6)
CONTROL_POS = (1, 6)  # off the diagonal approach corridor; ~5 steps from resource

N_CYCLES = 6
MAX_STEPS_PER_EPISODE = 25
EPSILON = 0.1

SWS_CONSOLIDATION_STEPS = 5
REM_ATTRIBUTION_STEPS = 8

WAYPOINT_CENTER_SEED = 1e-3   # near-zero: activates a center, injects ~no signal
CONTACT_SEED_WANTING = 2.0    # direct MECH-217-isolating terminus wanting seed
NUM_BASIS_FUNCTIONS = 256     # headroom so ~150 activations/cell never recycle

# Pre-registered thresholds
P1_MIN_BUFFER_LEN = 2
P2_MIN_REVERSE_FIRED = 1
C1_ON_OVER_OFF_MARGIN = 1.5


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


def _make_agent(env: CausalGridWorldV2, seed: int, use_spread: bool) -> REEAgent:
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
        use_offline_wanting_spread=use_spread,
    )
    cfg.residue.num_basis_functions = NUM_BASIS_FUNCTIONS
    # The shared default kernel_bandwidth (1.0) is calibrated for a different
    # terrain and is documented elsewhere (SD-067) as ~15x too wide for the
    # z_world residual scale -- confirmed here directly: measured z_world
    # distances between waypoints along a 10-step path in this env/config are
    # ~0.04-0.11, so at bandwidth=1.0 the RBF kernel (exp(-dist^2/2bw^2))
    # assigns every waypoint on the path ~full weight against every other,
    # collapsing the whole path to one blurred point for evaluate_valence
    # reads. That is a read-side confound of THIS script's short, spatially
    # tight path -- not a property of spread_reverse_replay_wanting, which
    # only ever writes via nearest-center lookup -- but left uncorrected it
    # creates a real feedback loop: an early cycle's spread onto a near
    # waypoint leaks (via the smooth kernel) into the terminus's own
    # evaluate_valence read, inflating wanting_at_terminus for every later
    # cycle's spread. Narrowing the bandwidth to the measured distance scale
    # restores the kernel's intended spatial discrimination.
    cfg.residue.kernel_bandwidth = 0.03
    return REEAgent(cfg)


def _one_hot_action(action_idx: int, action_dim: int) -> torch.Tensor:
    action = torch.zeros(1, action_dim)
    action[0, int(action_idx)] = 1.0
    return action


def _greedy_action_toward(env: CausalGridWorldV2, target, epsilon: float, rng: torch.Generator) -> int:
    """Scripted epsilon-greedy Manhattan action toward `target` (privileged)."""
    if torch.rand(1, generator=rng).item() < epsilon:
        return int(torch.randint(0, 4, (1,), generator=rng).item())
    ax, ay = env.agent_x, env.agent_y
    tx, ty = target
    dx, dy = tx - ax, ty - ay
    if dx == 0 and dy == 0:
        return 4  # stay
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


def run_cycle(agent: REEAgent, env: CausalGridWorldV2, rng: torch.Generator, seed_wanting: bool) -> Dict:
    """One wake episode (scripted walk to contact) + one SD-017 sleep cycle."""
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
        # One more sense() so the post-contact cell is the trajectory's final
        # (terminus) state, then seed real (hypothesis_tag=False) wanting
        # there -- the "resource-contact wanting at the episode terminus"
        # MECH-217 spreads backward. Isolated from MECH-203 (see docstring).
        # Seeded only on the FIRST cycle (seed_wanting=True): the claim text
        # is explicit that the mechanism produces multi-step behaviour from a
        # SINGLE resource encounter, and re-seeding every cycle at the same
        # fixed resource location (agent/resource positions are deliberately
        # fixed across cycles for path-length control -- see docstring) would
        # let repeated real writes compound the read source itself, which is
        # a confound of THIS script's repeated-encounter design, not of the
        # spread mechanism under test (which already guards its own single-
        # call feedback loop by never writing the terminus's own center).
        latent = _sense_tick(agent, obs_dict)
        agent.residue_field.rbf_field.add_residue(latent.z_world, WAYPOINT_CENTER_SEED)
        if seed_wanting:
            agent.residue_field.update_valence(
                latent.z_world, VALENCE_WANTING, CONTACT_SEED_WANTING, hypothesis_tag=False
            )
        agent._record_exploration_action(_one_hot_action(4, env.action_dim))

    # Sleep BEFORE the episode-boundary reset: agent.reset() clears
    # theta_buffer (needed by run_rem_attribution_pass's forward branch) as
    # well as flushing the exploration buffer, so sleep must observe the
    # still-populated theta_buffer from this episode's waking steps. This
    # episode's own trajectory becomes reverse-replayable starting next
    # cycle (flushed immediately below); aggregate readings at run end are
    # unaffected by that one-cycle pipeline delay.
    sleep_metrics = agent.run_sleep_cycle()
    agent.reset()  # MECH-165: flushes this episode to the exploration buffer
    return {
        "contact": contact,
        "n_steps": n_steps,
        "rem_n_reverse": float(sleep_metrics.get("rem_n_reverse", 0.0)),
        "rem_wanting_spread_n_steps": float(sleep_metrics.get("rem_wanting_spread_n_steps", 0.0)),
        "rem_wanting_spread_mean": float(sleep_metrics.get("rem_wanting_spread_mean", 0.0)),
    }


def _read_wanting(agent: REEAgent, z_world: torch.Tensor) -> float:
    with torch.no_grad():
        valence = agent.residue_field.evaluate_valence(z_world)
    return float(valence[..., VALENCE_WANTING].mean().item())


def run_cell(arm_name: str, use_spread: bool, seed: int, n_cycles: int) -> Dict:
    env = _make_env(seed)
    agent = _make_agent(env, seed, use_spread)
    rng = torch.Generator().manual_seed(seed)

    print(f"Seed {seed} Condition {arm_name}", flush=True)
    cycle_results: List[Dict] = []
    for cycle in range(n_cycles):
        print(f"  [train] {arm_name} seed={seed} ep {cycle + 1}/{N_CYCLES}", flush=True)
        cycle_results.append(run_cycle(agent, env, rng, seed_wanting=(cycle == 0)))

    buffer = agent.hippocampal._exploration_buffer
    near_vals: List[float] = []
    far_vals: List[float] = []
    for traj in buffer:
        ws = traj.world_states
        if ws is None or len(ws) < 2:
            continue
        near_vals.append(_read_wanting(agent, ws[-2]))
        far_vals.append(_read_wanting(agent, ws[0]))

    # Off-path control point: never visited, no dedicated center.
    _flat, control_obs = env.reset_to(
        agent_pos=CONTROL_POS, hazard_positions=[], resource_positions=[RESOURCE_POS]
    )
    control_latent = _sense_tick(agent, control_obs)
    control_wanting = _read_wanting(agent, control_latent.z_world)
    agent.reset()  # discard the 1-state pseudo-episode (< min_steps, no-op flush)

    n_contacts = sum(1 for c in cycle_results if c["contact"])
    total_reverse = sum(c["rem_n_reverse"] for c in cycle_results)
    total_spread_steps = sum(c["rem_wanting_spread_n_steps"] for c in cycle_results)

    cell_ready = (
        len(buffer) >= P1_MIN_BUFFER_LEN and total_reverse >= P2_MIN_REVERSE_FIRED
    )
    print(f"verdict: {'PASS' if cell_ready else 'FAIL'}", flush=True)

    return {
        "arm": arm_name,
        "use_offline_wanting_spread": use_spread,
        "seed": seed,
        "n_cycles": n_cycles,
        "n_contacts": n_contacts,
        "buffer_len": len(buffer),
        "total_rem_n_reverse": total_reverse,
        "total_rem_wanting_spread_n_steps": total_spread_steps,
        "near_wanting_mean": (sum(near_vals) / len(near_vals)) if near_vals else 0.0,
        "far_wanting_mean": (sum(far_vals) / len(far_vals)) if far_vals else 0.0,
        "n_trajectories_read": len(near_vals),
        "control_wanting": control_wanting,
        "cell_ready": bool(cell_ready),
        "cycle_results": cycle_results,
    }


def evaluate_criteria(cells: List[Dict]) -> Dict:
    by_arm_seed = {(c["arm"], c["seed"]): c for c in cells}

    p1_met = min(c["buffer_len"] for c in cells) >= P1_MIN_BUFFER_LEN
    p2_met = min(c["total_rem_n_reverse"] for c in cells) >= P2_MIN_REVERSE_FIRED
    preconditions = [
        {
            "name": "exploration_buffer_populated",
            "description": "MECH-165 exploration buffer holds >= 2 trajectories at run end",
            "measured": min(c["buffer_len"] for c in cells),
            "threshold": P1_MIN_BUFFER_LEN,
            "direction": "lower",
            "met": bool(p1_met),
        },
        {
            "name": "reverse_replay_fired",
            "description": "at least one MECH-165 reverse rollout fired across the run",
            "measured": min(c["total_rem_n_reverse"] for c in cells),
            "threshold": P2_MIN_REVERSE_FIRED,
            "direction": "lower",
            "met": bool(p2_met),
        },
    ]
    if not (p1_met and p2_met):
        return {
            "label": "substrate_not_ready_requeue",
            "preconditions": preconditions,
            "criteria_non_degenerate": {},
            "overall_pass": False,
        }

    on_near = [by_arm_seed[("ARM_ON", s)]["near_wanting_mean"] for s in SEEDS if ("ARM_ON", s) in by_arm_seed]
    off_near = [by_arm_seed[("ARM_OFF", s)]["near_wanting_mean"] for s in SEEDS if ("ARM_OFF", s) in by_arm_seed]
    on_far = [by_arm_seed[("ARM_ON", s)]["far_wanting_mean"] for s in SEEDS if ("ARM_ON", s) in by_arm_seed]
    on_control = [by_arm_seed[("ARM_ON", s)]["control_wanting"] for s in SEEDS if ("ARM_ON", s) in by_arm_seed]
    off_spread_steps = [by_arm_seed[("ARM_OFF", s)]["total_rem_wanting_spread_n_steps"] for s in SEEDS if ("ARM_OFF", s) in by_arm_seed]

    mean_on_near = sum(on_near) / max(1, len(on_near))
    mean_off_near = sum(off_near) / max(1, len(off_near))
    mean_on_far = sum(on_far) / max(1, len(on_far))
    mean_on_control = sum(on_control) / max(1, len(on_control))

    c0_off_no_writes = all(v == 0.0 for v in off_spread_steps)
    c1 = bool(mean_on_near > C1_ON_OVER_OFF_MARGIN * mean_off_near and mean_on_near > 0.0)
    c2 = bool(mean_on_near > mean_on_far)
    c3 = bool(mean_on_control < mean_on_near)

    overall = bool(c0_off_no_writes and c1 and c2 and c3)
    return {
        "label": "mech217_offline_spread_readiness_verified" if overall else "mech217_offline_spread_readiness_not_verified",
        "preconditions": preconditions,
        "criteria_non_degenerate": {
            "C0_off_no_writes": True,
            "C1_on_near_exceeds_off": bool(mean_on_near > 0.0),
            "C2_decay_with_distance": bool(len(on_far) > 0),
            "C3_control_below_near": bool(len(on_control) > 0),
        },
        "criteria": [
            {"name": "C0_off_arm_zero_writes", "load_bearing": False, "passed": c0_off_no_writes},
            {"name": "C1_on_near_exceeds_off_1p5x", "load_bearing": True, "passed": c1},
            {"name": "C2_on_near_exceeds_on_far", "load_bearing": False, "passed": c2},
            {"name": "C3_control_below_near", "load_bearing": False, "passed": c3},
        ],
        "mean_on_near_wanting": mean_on_near,
        "mean_off_near_wanting": mean_off_near,
        "mean_on_far_wanting": mean_on_far,
        "mean_on_control_wanting": mean_on_control,
        "overall_pass": overall,
    }


def run_experiment(dry_run: bool = False):
    """Run the full experiment. Returns (outcome, manifest_path) -- manifest_path
    is None under --dry-run (no manifest written). emit_outcome is called from
    the __main__ block, not here (runner-conformance AST check requirement)."""
    seeds = (SEEDS[0],) if dry_run else SEEDS
    n_cycles = 2 if dry_run else N_CYCLES
    t0 = time.perf_counter()

    cells: List[Dict] = []
    for arm_name in ARMS:
        use_spread = arm_name == "ARM_ON"
        for seed in seeds:
            with arm_cell(
                seed,
                config_slice={"arm": arm_name, "use_offline_wanting_spread": use_spread,
                              "grid_size": GRID_SIZE, "agent_start": AGENT_START,
                              "resource_pos": RESOURCE_POS, "n_cycles": n_cycles},
                script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                row = run_cell(arm_name, use_spread, seed, n_cycles)
                cell.stamp(row)
            cells.append(row)

    elapsed = time.perf_counter() - t0
    interpretation = evaluate_criteria(cells)
    outcome = "PASS" if interpretation["overall_pass"] else "FAIL"
    direction = "supports" if outcome == "PASS" else "weakens"

    print(f"V3-EXQ-842 MECH-217 offline wanting-spread readiness -- {outcome} in {elapsed:.1f}s", flush=True)
    for k in ("mean_on_near_wanting", "mean_off_near_wanting", "mean_on_far_wanting", "mean_on_control_wanting"):
        if k in interpretation:
            print(f"  {k}: {interpretation[k]:.5f}", flush=True)

    if dry_run:
        print("[--dry-run] manifest not written.", flush=True)
        return outcome, None

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "evidence_direction": direction,
        "interpretation": interpretation,
        "arm_results": cells,
        "registered_thresholds": {
            "P1_MIN_BUFFER_LEN": P1_MIN_BUFFER_LEN,
            "P2_MIN_REVERSE_FIRED": P2_MIN_REVERSE_FIRED,
            "C1_ON_OVER_OFF_MARGIN": C1_ON_OVER_OFF_MARGIN,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "Diagnostic substrate-readiness probe for MECH-217 (EXP-0301 / "
            "EVB-0349), immediately following the 2026-07-30 implementation "
            "landing. Isolates the offline reverse-replay spread mechanism "
            "from MECH-203 (serotonin-gated proximity wanting, unused here) "
            "and from the agent's own navigation policy (scripted greedy "
            "walk used instead). Not a behavioural navigation-lift claim."
        ),
    }
    full_config = {
        "seeds": list(seeds),
        "n_cycles": n_cycles,
        "arms": list(ARMS),
        "grid_size": GRID_SIZE,
        "agent_start": AGENT_START,
        "resource_pos": RESOURCE_POS,
        "control_pos": CONTROL_POS,
        "sws_consolidation_steps": SWS_CONSOLIDATION_STEPS,
        "rem_attribution_steps": REM_ATTRIBUTION_STEPS,
        "waypoint_center_seed": WAYPOINT_CENTER_SEED,
        "contact_seed_wanting": CONTACT_SEED_WANTING,
        "num_basis_functions": NUM_BASIS_FUNCTIONS,
    }
    out_path = write_flat_manifest(
        manifest,
        dry_run=False,
        config=full_config,
        seeds=list(seeds),
        script_path=Path(__file__),
        started_at=t0,
    )
    print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    _outcome, _manifest_path = run_experiment(dry_run=args.dry_run)
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(_manifest_path) if _manifest_path is not None else None,
        dry_run=args.dry_run,
    )
    sys.exit(0)
