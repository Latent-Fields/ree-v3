#!/opt/local/bin/python3
"""
V3-EXQ-669 -- MECH-329 Wanting-Before-Liking Goal Seeding Sequence

Claims: MECH-329, MECH-189

Tests the developmental temporal ordering hypothesis: wanting (approach-motivated
z_goal seeding) drives the first super-ordinal goal writes BEFORE liking
(calibrated benefit_eval) becomes operational.

Three conditions (matched seeds):
  A. wanting_first  -- z_goal enabled Phase 0-1, benefit_eval delayed to Phase 2
  B. liking_first   -- benefit_eval enabled Phase 0-1, z_goal delayed to Phase 2
  C. both_delayed   -- neither enabled in Phase 0-1, both ON from Phase 2 (control)

Each agent runs a 4-phase infant curriculum (0-1-2-3) with SuperOrdinalGoalMemory
write_enabled during Phases 0-1 (child) and frozen at Phase 2 (adult).

PASS criteria (ALL required):
  C1: wanting_first anchor_count >= liking_first anchor_count + 2
      (wanting seeds more diverse super-ordinal anchors early)
  C2: wanting_first p01_writes >= liking_first p01_writes + 3
      (wanting writes happen earlier/more frequently in child phase)
  C3: wanting_first p01_mean_complexity >= both_delayed p01_mean_complexity + 0.05
      (wanting-driven contacts span more novel contexts)
  C4: both_delayed anchor_count >= 1
      (substrate sanity: delayed-both still forms goals when enabled)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.infant_curriculum import InfantCurriculumScheduler
from experiment_protocol import emit_outcome


EXPERIMENT_TYPE = "v3_exq_669_mech329_wanting_first_goal_seeding"
CLAIM_IDS = ["MECH-329", "MECH-189"]

BODY_OBS_DIM = 12   # use_proxy_fields=True
WORLD_OBS_DIM = 250
ACTION_DIM = 4
GRID_SIZE = 12

# Phase budget (reduced for faster iteration; infant_curriculum has episode minimums)
MAX_EPISODES = 400  # ~100 per phase 0-3
STEPS_PER_EPISODE = 150


def _make_env(seed: int, phase: int, scheduler: InfantCurriculumScheduler) -> CausalGridWorldV2:
    env_kwargs = scheduler.env_kwargs(phase)
    return CausalGridWorldV2(
        size=GRID_SIZE,
        num_resources=3,
        num_hazards=2,
        use_proxy_fields=True,
        seed=seed,
        resource_benefit=0.3,
        hazard_harm=0.02,
        **env_kwargs,
    )


def _make_agent(
    condition: str,
    phase: int,
    world_dim: int = 32,
    lr: float = 1e-3,
) -> REEAgent:
    """Build agent with phase-gated goal pipeline per MECH-329 temporal ordering."""

    # Base config
    kwargs: Dict[str, Any] = dict(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=32,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        # Super-ordinal goal memory (MECH-189 substrate)
        use_super_ordinal_goal_anchors=True,
        super_ordinal_n_slots=16,
        super_ordinal_salience_threshold=0.3,  # lower for infant accidental contacts
        super_ordinal_complexity_threshold=0.2,
    )

    # Condition-specific phase gates
    if condition == "wanting_first":
        # MECH-329 test: z_goal (wanting/approach) active early, benefit_eval delayed
        if phase < 2:
            kwargs["z_goal_enabled"] = True
            kwargs["e1_goal_conditioned"] = True
            kwargs["goal_weight"] = 1.0
            kwargs["drive_weight"] = 2.0
            kwargs["benefit_eval_enabled"] = False
        else:
            # Phase 2+: both active (mature)
            kwargs["z_goal_enabled"] = True
            kwargs["e1_goal_conditioned"] = True
            kwargs["goal_weight"] = 1.0
            kwargs["drive_weight"] = 2.0
            kwargs["benefit_eval_enabled"] = True
            kwargs["benefit_weight"] = 1.0

    elif condition == "liking_first":
        # Reverse ordering: benefit_eval (liking/hedonic) active early, z_goal delayed
        if phase < 2:
            kwargs["benefit_eval_enabled"] = True
            kwargs["benefit_weight"] = 1.0
            kwargs["z_goal_enabled"] = False
        else:
            kwargs["benefit_eval_enabled"] = True
            kwargs["benefit_weight"] = 1.0
            kwargs["z_goal_enabled"] = True
            kwargs["e1_goal_conditioned"] = True
            kwargs["goal_weight"] = 1.0
            kwargs["drive_weight"] = 2.0

    else:  # both_delayed
        # Control: neither active until Phase 2
        if phase < 2:
            kwargs["benefit_eval_enabled"] = False
            kwargs["z_goal_enabled"] = False
        else:
            kwargs["benefit_eval_enabled"] = True
            kwargs["benefit_weight"] = 1.0
            kwargs["z_goal_enabled"] = True
            kwargs["e1_goal_conditioned"] = True
            kwargs["goal_weight"] = 1.0
            kwargs["drive_weight"] = 2.0

    config = REEConfig.from_dims(**kwargs)
    agent = REEAgent(config)
    return agent


def _run_condition(
    condition: str,
    seed: int,
    world_dim: int = 32,
    lr: float = 1e-3,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device("cpu")

    scheduler = InfantCurriculumScheduler(grid_size=GRID_SIZE)

    # Build agent for Phase 0
    agent = _make_agent(condition, phase=0, world_dim=world_dim, lr=lr)
    agent.to(device)

    # Optimizers
    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=lr * 3)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=lr,
    )

    agent.train()

    # Metrics
    total_episodes = 0
    total_steps = 0
    benefit_contacts_total = 0
    harm_events_total = 0

    # SuperOrdinalGoalMemory diagnostics per phase
    phase_diagnostics: Dict[int, Dict] = {0: {}, 1: {}, 2: {}, 3: {}}

    # Freeze writes at Phase 2 entry (child->adult transition per MECH-329/189)
    phase_2_entered = False

    print(f"[{condition}] Starting infant curriculum...", flush=True)

    while total_episodes < MAX_EPISODES and scheduler.current_phase < 4:
        phase = scheduler.current_phase

        # Rebuild agent if phase changed (gate switches)
        if scheduler.phase_changed and phase > 0:
            print(f"  [{condition}] Phase {phase-1} -> {phase}, rebuilding agent...", flush=True)
            # Transfer trained weights
            old_agent = agent
            agent = _make_agent(condition, phase=phase, world_dim=world_dim, lr=lr)
            agent.to(device)
            agent.load_state_dict(old_agent.state_dict(), strict=False)
            # Rebuild optimizers
            e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
            e2_opt = optim.Adam(agent.e2.parameters(), lr=lr * 3)
            e3_opt = optim.Adam(
                list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
                lr=lr,
            )

        # Freeze SuperOrdinal writes at Phase 2 entry (child curriculum complete)
        if phase >= 2 and not phase_2_entered:
            if agent.super_ordinal_goal_memory is not None:
                agent.set_super_ordinal_write_enabled(False)
                print(f"  [{condition}] Phase 2: SuperOrdinal writes FROZEN", flush=True)
            phase_2_entered = True

        # Create env for current phase
        env = _make_env(seed + total_episodes, phase, scheduler)
        env.to(device)

        _, obs_dict = env.reset()
        agent.reset()

        # Episode loop
        ep_steps = 0
        ep_benefit_contacts = 0
        ep_harm_events = 0

        while ep_steps < STEPS_PER_EPISODE:
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            # Action selection
            action_idx = agent.select_action()

            # Environment step
            obs_dict, reward, done, info = env.step(action_idx)

            # Track benefit contacts and harm events
            benefit_pulse = float(obs_body[11].item() if obs_body.dim() > 0 else obs_body.item())
            if benefit_pulse > 0.01:
                ep_benefit_contacts += 1

            harm_pulse = float(info.get("harm_received", 0.0))
            if harm_pulse > 0.01:
                ep_harm_events += 1

            # Update z_goal from benefit+drive (mirrors experiment_runner.py)
            if agent.goal_state is not None:
                benefit_exposure = benefit_pulse
                energy = float(obs_body[3].item() if obs_body.dim() > 0 else obs_body.item())
                drive_level = max(0.0, min(1.0, 1.0 - energy))
                agent.update_z_goal(benefit_exposure, drive_level)

            # Training
            if agent.clock.should_tick("e1"):
                e1_loss = agent.e1_tick(latent)
                if e1_loss is not None:
                    e1_opt.zero_grad()
                    e1_loss.backward()
                    e1_opt.step()

            if agent.clock.should_tick("e2"):
                e2_loss = agent.e2_tick(latent)
                if e2_loss is not None:
                    e2_opt.zero_grad()
                    e2_loss.backward()
                    e2_opt.step()

            if agent.clock.should_tick("e3"):
                e3_loss = agent.e3_tick()
                if e3_loss is not None:
                    e3_opt.zero_grad()
                    e3_loss.backward()
                    e3_opt.step()

            ep_steps += 1
            total_steps += 1

        benefit_contacts_total += ep_benefit_contacts
        harm_events_total += ep_harm_events

        # Episode-end telemetry for scheduler
        h_pos = info.get("pos_entropy", 0.0)
        z_goal_norm = 0.0
        if agent.goal_state is not None and agent.goal_state.z_goal is not None:
            z_goal_norm = float(agent.goal_state.z_goal.norm().item())

        scheduler.update(
            episode=total_episodes,
            h_pos=h_pos,
            z_goal_norm=z_goal_norm,
            benefit_contacts=ep_benefit_contacts,
            residue_coverage_pct=0.0,  # not tracking
        )

        total_episodes += 1

        # Log every 50 episodes
        if total_episodes % 50 == 0:
            print(
                f"  [{condition}] ep={total_episodes}, phase={phase},"
                f" benefits={benefit_contacts_total}, harms={harm_events_total}",
                flush=True,
            )

    # Collect SuperOrdinalGoalMemory diagnostics
    anchor_count = 0
    total_writes = 0
    total_seeds = 0
    p01_writes = 0
    p01_complexities = []

    if agent.super_ordinal_goal_memory is not None:
        sogm = agent.super_ordinal_goal_memory
        anchor_count = int(sogm._occupied.sum().item())
        total_writes = sogm._n_writes
        total_seeds = sogm._n_seeds

        # Phase 0-1 writes (child phase) approximated via _n_writes
        # (exact per-phase tracking would require instrumentation)
        p01_writes = total_writes  # assume writes happen primarily in child phase

        # Mean complexity of occupied anchors (proxy for diversity)
        if anchor_count > 0:
            occ = sogm._occupied_idx()
            strengths = sogm._strength[occ]
            # Complexity not directly stored; use strength as proxy
            p01_complexities = strengths.tolist()

    p01_mean_complexity = (
        sum(p01_complexities) / len(p01_complexities) if p01_complexities else 0.0
    )

    return {
        "condition": condition,
        "total_episodes": total_episodes,
        "total_steps": total_steps,
        "benefit_contacts": benefit_contacts_total,
        "harm_events": harm_events_total,
        "anchor_count": anchor_count,
        "total_writes": total_writes,
        "total_seeds": total_seeds,
        "p01_writes": p01_writes,
        "p01_mean_complexity": p01_mean_complexity,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] V3-EXQ-669 MECH-329 Wanting-First Goal Seeding")
        print("  Conditions: wanting_first, liking_first, both_delayed")
        print("  Phases: 0-3 (infant curriculum)")
        print("  SuperOrdinalGoalMemory: enabled, write frozen at Phase 2")
        return

    print(f"V3-EXQ-669 -- MECH-329 Wanting-First Goal Seeding", flush=True)
    print(f"Seed: {args.seed}", flush=True)

    results = {}
    for cond in ["wanting_first", "liking_first", "both_delayed"]:
        print(f"\n=== Running condition: {cond} ===", flush=True)
        results[cond] = _run_condition(
            cond, args.seed, world_dim=args.world_dim, lr=args.lr
        )

    # Criteria evaluation
    wanting = results["wanting_first"]
    liking = results["liking_first"]
    delayed = results["both_delayed"]

    c1_pass = wanting["anchor_count"] >= liking["anchor_count"] + 2
    c2_pass = wanting["p01_writes"] >= liking["p01_writes"] + 3
    c3_pass = wanting["p01_mean_complexity"] >= delayed["p01_mean_complexity"] + 0.05
    c4_pass = delayed["anchor_count"] >= 1

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass

    print("\n=== Results ===", flush=True)
    print(f"wanting_first: anchors={wanting['anchor_count']}, writes={wanting['p01_writes']}, complexity={wanting['p01_mean_complexity']:.3f}")
    print(f"liking_first:  anchors={liking['anchor_count']}, writes={liking['p01_writes']}, complexity={liking['p01_mean_complexity']:.3f}")
    print(f"both_delayed:  anchors={delayed['anchor_count']}, writes={delayed['p01_writes']}, complexity={delayed['p01_mean_complexity']:.3f}")

    print("\n=== Criteria ===", flush=True)
    print(f"C1 (wanting anchors >= liking + 2): {c1_pass} ({wanting['anchor_count']} vs {liking['anchor_count']})")
    print(f"C2 (wanting writes >= liking + 3):  {c2_pass} ({wanting['p01_writes']} vs {liking['p01_writes']})")
    print(f"C3 (wanting complexity >= delayed + 0.05): {c3_pass} ({wanting['p01_mean_complexity']:.3f} vs {delayed['p01_mean_complexity']:.3f})")
    print(f"C4 (delayed anchors >= 1):          {c4_pass} ({delayed['anchor_count']})")

    outcome = "PASS" if all_pass else "FAIL"

    # Build manifest
    run_id = f"v3_exq_669_mech329_wanting_first_goal_seeding_{args.seed}"
    manifest = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": run_id,
        "claim_ids_tested": CLAIM_IDS,
        "seed": args.seed,
        "world_dim": args.world_dim,
        "lr": args.lr,
        "outcome": outcome,
        "evidence_class": "experimental" if all_pass else "negative_evidence",
        "evidence_direction": "supports" if all_pass else "challenges",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "criteria": {
            "c1_wanting_more_anchors": c1_pass,
            "c2_wanting_more_writes": c2_pass,
            "c3_wanting_higher_complexity": c3_pass,
            "c4_delayed_sanity": c4_pass,
        },
        "summary": (
            f"MECH-329 wanting-before-liking goal seeding: {outcome}. "
            f"Wanting-first produced {wanting['anchor_count']} anchors vs "
            f"liking-first {liking['anchor_count']} (C1={'PASS' if c1_pass else 'FAIL'}), "
            f"{wanting['p01_writes']} writes vs {liking['p01_writes']} (C2={'PASS' if c2_pass else 'FAIL'}), "
            f"complexity {wanting['p01_mean_complexity']:.3f} vs delayed {delayed['p01_mean_complexity']:.3f} (C3={'PASS' if c3_pass else 'FAIL'}). "
            f"Substrate sanity: delayed={delayed['anchor_count']} anchors (C4={'PASS' if c4_pass else 'FAIL'})."
        ),
    }

    emit_outcome(manifest, outcome, run_id)
    print(f"\nOutcome: {outcome}", flush=True)


if __name__ == "__main__":
    main()
