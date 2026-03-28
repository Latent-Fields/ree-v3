#!/opt/local/bin/python3
"""
V3-EXQ-120 -- ARC-018: Viability Map Discriminative Pair

Claim: ARC-018
Proposal: EXP-0017 / EVB-0014
Dispatch mode: discriminative_pair
Min shared seeds: 2

ARC-018 asserts: "Hippocampus generates explicit rollouts and post-commitment
viability mapping." Specifically (corrected framing, 2026-03-15):
  - The hippocampal module builds a viability map indexed by E2 action-object
    coordinates, updated by E3 harm/goal error.
  - Precision is dynamically updated from prediction error variance (ARC-016
    component of the claim).
  - Terrain-guided navigation exploiting this viability map reduces harm exposure
    compared to an agent with no viability map.

Prior evidence: EXQ-053 PASS (terrain_guided: harm_per_step=0.00147 vs random
0.0724, viability_advantage=0.071, 5/5 criteria). That was a single-condition
run (not a discriminative pair with matched seeds). This experiment provides
the matched-seed pair required by EVB-0014.

This experiment implements a discriminative pair:
  VIABILITY_MAP_ON    -- Residue field accumulation ACTIVE: harm events populate
                         the viability map (RBF terrain). HippocampalModule
                         terrain-guided proposals used for action selection.
                         Dynamic precision updated via E3.post_action_update()
                         after each harm event.
  VIABILITY_MAP_ABLATED -- Residue field accumulation DISABLED: terrain never
                           forms; random action selection (no hippocampal
                           proposals). Dynamic precision frozen at init value
                           (no E3.post_action_update() calls).

The ablation disables both components of ARC-018:
  (a) viability map construction (residue accumulation off)
  (b) dynamic precision update (post_action_update not called)

Design notes:
  - Both conditions run identical training episodes with identical seeds.
  - Navigation bias: nav_bias=0.40 (40% chance to approach hazard) ensures
    harm events occur and the viability map can accumulate signal.
  - The ablated condition accumulates ZERO residue -- the terrain is flat.
    This is a stronger ablation than merely disabling proposals; it removes
    the information substrate entirely.
  - SD-005 split latent mode: z_self != z_world (residue on z_world only).
  - SD-008: alpha_world=0.9 (event-responsive z_world encoding).
  - SD-007 reafference disabled to isolate ARC-018 mechanism.

PRE-REGISTERED THRESHOLDS (hardcoded, not inferred post-hoc):

  C1 (harm rate advantage):
    harm_rate_ON <= harm_rate_ABLATED * (1 - THRESH_C1_REDUCTION)
    VIABILITY_MAP_ON must achieve >=15% fractional reduction in harm rate.
    Threshold: THRESH_C1_REDUCTION = 0.15.

  C2 (viability map populates):
    residue_field_harm_events_ON >= THRESH_C2_MIN_RESIDUE_EVENTS per seed.
    The terrain must actually accumulate: at least 20 harm events per seed.
    Threshold: THRESH_C2_MIN_RESIDUE_EVENTS = 20.

  C3 (consistency across seeds):
    harm_rate_ON < harm_rate_ABLATED for BOTH seeds.
    Direction must be consistent across all seeds.

  C4 (data quality):
    n_harm_events_min >= THRESH_C4_MIN_EVENTS per cell.
    Sufficient harm events to estimate harm rate reliably.
    Threshold: THRESH_C4_MIN_EVENTS = 20.

  C5 (residue field active centers, MAP_ON only):
    residue_field_active_centers_on >= THRESH_C5_MIN_ACTIVE_CENTERS per seed.
    The viability map must have spread across the latent space: at least 5 RBF
    centers must be active after training, confirming the map indexed actual states.
    Threshold: THRESH_C5_MIN_ACTIVE_CENTERS = 5.
    Note: inactive centers (weights=0) don't contribute to terrain scoring.

PASS criteria:
  C1 AND C2 AND C3 AND C4 AND C5 -> PASS -> supports ARC-018
  C1 AND C3 AND C4, NOT C2       -> mixed (harm advantage without confirmed map)
  C2 AND C4, harm_rate_ON > harm_rate_ABLATED -> FAIL -> retire (map HURTS)
  NOT C4                          -> FAIL -> data quality (inconclusive)

Decision mapping:
  PASS -> retain_ree
  partial (C1+C3+C4, C2 optional, C5 optional) -> hybridize
  harm_rate_ON > harm_rate_ABLATED AND C4 -> retire_ree_claim
  NOT C4 -> inconclusive
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_120_arc018_viability_map_pair"
CLAIM_IDS = ["ARC-018"]

# Pre-registered thresholds
THRESH_C1_REDUCTION       = 0.15   # VIABILITY_MAP_ON harm_rate <= ABLATED * (1 - 0.15)
THRESH_C2_MIN_RESIDUE_EVENTS = 20  # min harm events accumulated in residue field per seed
THRESH_C4_MIN_EVENTS      = 20     # min harm events per (seed, condition) for data quality
THRESH_C5_MIN_ACTIVE_CENTERS = 5   # min active RBF centers in residue field after training


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _hazard_approach_action(env: CausalGridWorldV2, n_actions: int) -> int:
    """Return action that moves toward nearest hazard gradient. Fallback: random."""
    obs_dict = env._get_observation_dict()
    world_state = obs_dict.get("world_state", None)
    if world_state is None or not env.use_proxy_fields:
        return random.randint(0, n_actions - 1)
    # world_state[225:250] = hazard_field_view (5x5 flattened, proxy channel)
    field_view = world_state[225:250].numpy().reshape(5, 5)
    # actions: 0=up(-1,0), 1=down(+1,0), 2=left(0,-1), 3=right(0,+1)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    vals = []
    for dr, dc in deltas:
        r, c = 2 + dr, 2 + dc
        if 0 <= r < 5 and 0 <= c < 5:
            vals.append(float(field_view[r, c]))
        else:
            vals.append(-1.0)
    return int(np.argmax(vals))


def _run_single(
    seed: int,
    viability_map_ablated: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
    nav_bias: float,
    dry_run: bool,
) -> Dict:
    """Run one (seed, condition) cell; return harm rate and viability map metrics."""
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "VIABILITY_MAP_ABLATED" if viability_map_ablated else "VIABILITY_MAP_ON"

    env = CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=4,
        num_resources=3,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
    )
    n_actions = env.action_dim

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,  # SD-007 disabled -- isolate ARC-018
    )
    # SD-005: split latent mode (z_self != z_world)
    config.latent.unified_latent_mode = False

    agent = REEAgent(config)
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    harm_events_train = 0
    total_steps_train = 0

    # Track residue field events (in ON condition only)
    residue_events_accumulated = 0

    if dry_run:
        actual_warmup = min(3, warmup_episodes)
        actual_eval = min(2, eval_episodes)
    else:
        actual_warmup = warmup_episodes
        actual_eval = eval_episodes

    # --- TRAIN ---
    agent.train()

    for ep in range(actual_warmup):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()
            z_self_curr = latent.z_self.detach()

            # Action selection: condition determines whether viability map is used
            if viability_map_ablated:
                # ABLATED: random action, no terrain guidance
                # Nav bias still applies (ensures comparable harm exposure, so
                # the ablated condition still encounters hazards)
                if random.random() < nav_bias:
                    action_idx = _hazard_approach_action(env, n_actions)
                else:
                    action_idx = random.randint(0, n_actions - 1)
            else:
                # VIABILITY_MAP_ON: terrain-guided proposals when available
                # Also apply nav_bias by occasionally forcing approach
                try:
                    with torch.no_grad():
                        candidates = agent.hippocampal.propose_trajectories(
                            z_world_curr,
                            z_self=z_self_curr,
                            num_candidates=4,
                        )
                    if candidates and random.random() > nav_bias:
                        # Use hippocampal proposal (low-residue trajectory)
                        best_traj = candidates[0]
                        world_seq = best_traj.get_world_state_sequence()
                        if world_seq is not None:
                            ao_seq = best_traj.get_action_object_sequence()
                            if ao_seq is not None and ao_seq.shape[1] > 0:
                                first_ao = ao_seq[:, 0, :]
                                raw_logits = agent.hippocampal.action_object_decoder(
                                    first_ao
                                )
                                action_idx = int(
                                    torch.argmax(raw_logits, dim=-1).item()
                                )
                            else:
                                action_idx = random.randint(0, n_actions - 1)
                        else:
                            action_idx = random.randint(0, n_actions - 1)
                    else:
                        # nav_bias fraction: force approach to guarantee map populates
                        action_idx = _hazard_approach_action(env, n_actions)
                except Exception:
                    action_idx = random.randint(0, n_actions - 1)

            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            total_steps_train += 1

            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_events_train += 1

                if not viability_map_ablated:
                    # VIABILITY_MAP_ON: accumulate harm into residue field
                    agent.residue_field.accumulate(
                        z_world_curr,
                        harm_magnitude=abs(float(harm_signal)),
                    )
                    residue_events_accumulated += 1

                    # Dynamic precision: update E3 running variance on harm
                    try:
                        agent.e3.post_action_update(
                            actual_z_world=latent.z_world,
                            harm_occurred=True,
                        )
                    except Exception:
                        pass
                # ABLATED: neither accumulate residue nor update precision

            # Standard E1 + E2 losses (both conditions)
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == actual_warmup - 1:
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{actual_warmup}"
                f" harm_events={harm_events_train}"
                f" harm_rate={harm_events_train / max(1, total_steps_train):.4f}"
                f" residue_events={residue_events_accumulated}",
                flush=True,
            )

    # --- EVAL ---
    agent.eval()

    harm_events_eval = 0
    total_steps_eval = 0

    for _ in range(actual_eval):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world_curr = latent.z_world
                z_self_curr = latent.z_self

            # Same action selection logic as training (eval: random only -- no nav_bias)
            if viability_map_ablated:
                action_idx = random.randint(0, n_actions - 1)
            else:
                try:
                    candidates = agent.hippocampal.propose_trajectories(
                        z_world_curr,
                        z_self=z_self_curr,
                        num_candidates=4,
                    )
                    if candidates:
                        best_traj = candidates[0]
                        world_seq = best_traj.get_world_state_sequence()
                        if world_seq is not None:
                            ao_seq = best_traj.get_action_object_sequence()
                            if ao_seq is not None and ao_seq.shape[1] > 0:
                                first_ao = ao_seq[:, 0, :]
                                raw_logits = agent.hippocampal.action_object_decoder(
                                    first_ao
                                )
                                action_idx = int(
                                    torch.argmax(raw_logits, dim=-1).item()
                                )
                            else:
                                action_idx = random.randint(0, n_actions - 1)
                        else:
                            action_idx = random.randint(0, n_actions - 1)
                    else:
                        action_idx = random.randint(0, n_actions - 1)
                except Exception:
                    action_idx = random.randint(0, n_actions - 1)

            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            total_steps_eval += 1

            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_events_eval += 1

            if done:
                break

    harm_rate_eval = harm_events_eval / max(1, total_steps_eval)

    # C5: active RBF centers in residue field (only non-zero for MAP_ON)
    try:
        stats = agent.residue_field.get_statistics()
        active_centers = int(stats.get("active_centers", torch.tensor(0)).item())
    except Exception:
        active_centers = 0

    # Residue field stats
    residue_total = float(agent.residue_field.total_residue.item())
    residue_field_harm_events = int(agent.residue_field.num_harm_events.item())

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" harm_rate={harm_rate_eval:.4f}"
        f" harm_events={harm_events_eval}/{total_steps_eval}"
        f" residue_events={residue_events_accumulated}"
        f" residue_total={residue_total:.4f}"
        f" active_centers={active_centers}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "viability_map_ablated": viability_map_ablated,
        # Training stats
        "harm_events_train": int(harm_events_train),
        "total_steps_train": int(total_steps_train),
        "harm_rate_train": float(harm_events_train / max(1, total_steps_train)),
        # Eval stats
        "harm_events_eval": int(harm_events_eval),
        "total_steps_eval": int(total_steps_eval),
        "harm_rate_eval": float(harm_rate_eval),
        # ARC-018 specific: viability map construction
        "residue_events_accumulated": int(residue_events_accumulated),
        "residue_field_total": float(residue_total),
        "residue_field_harm_events": int(residue_field_harm_events),
        "active_centers": int(active_centers),
    }


def run(
    seeds: Tuple = (42, 123),
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    nav_bias: float = 0.40,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """
    Discriminative pair: VIABILITY_MAP_ON vs VIABILITY_MAP_ABLATED.
    Tests ARC-018: hippocampal viability map (E2 rollout scored via residue field
    + dynamic precision) reduces harm exposure vs no-map ablation.
    """
    results_on: List[Dict] = []
    results_ablated: List[Dict] = []

    for seed in seeds:
        for viability_map_ablated in [False, True]:
            label = "VIABILITY_MAP_ABLATED" if viability_map_ablated else "VIABILITY_MAP_ON"
            print(
                f"\n[V3-EXQ-120] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" alpha_world={alpha_world} nav_bias={nav_bias}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                viability_map_ablated=viability_map_ablated,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
                nav_bias=nav_bias,
                dry_run=dry_run,
            )
            if viability_map_ablated:
                results_ablated.append(r)
            else:
                results_on.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    # Primary metrics
    harm_rate_on      = _avg(results_on,      "harm_rate_eval")
    harm_rate_ablated = _avg(results_ablated, "harm_rate_eval")

    # Fractional reduction
    if harm_rate_ablated > 0:
        harm_reduction_frac = (harm_rate_ablated - harm_rate_on) / harm_rate_ablated
    else:
        harm_reduction_frac = 0.0

    # Residue events: did the viability map actually populate?
    residue_events_min_on = min(
        r["residue_events_accumulated"] for r in results_on
    )

    # Per-seed consistency (C3)
    seed_pair_pass_harm = sum(
        1 for n, a in zip(results_on, results_ablated)
        if n["harm_rate_eval"] < a["harm_rate_eval"]
    )

    # Data quality: min harm events across all cells (C4)
    n_harm_min = min(r["harm_events_eval"] for r in results_on + results_ablated)

    # Active centers: min across ON seeds (C5)
    active_centers_min_on = min(r["active_centers"] for r in results_on)

    # Pre-registered PASS criteria
    c1_pass = harm_reduction_frac >= THRESH_C1_REDUCTION
    c2_pass = residue_events_min_on >= THRESH_C2_MIN_RESIDUE_EVENTS
    c3_pass = seed_pair_pass_harm >= len(seeds)
    c4_pass = n_harm_min >= THRESH_C4_MIN_EVENTS
    c5_pass = active_centers_min_on >= THRESH_C5_MIN_ACTIVE_CENTERS

    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status = "PASS" if all_pass else "FAIL"

    # Decision scoring
    if all_pass:
        decision = "retain_ree"
    elif c1_pass and c3_pass and c4_pass:
        # Harm advantage demonstrated with consistent direction
        decision = "hybridize"
    elif c4_pass and harm_rate_on > harm_rate_ablated:
        # Map actively hurts navigation
        decision = "retire_ree_claim"
    else:
        decision = "hybridize"

    print(f"\n[V3-EXQ-120] Results:", flush=True)
    print(
        f"  harm_rate: MAP_ON={harm_rate_on:.4f}"
        f" MAP_ABLATED={harm_rate_ablated:.4f}"
        f" reduction_frac={harm_reduction_frac:+.4f}",
        flush=True,
    )
    print(
        f"  residue_events_min_on={residue_events_min_on}"
        f"  active_centers_min_on={active_centers_min_on}"
        f"  seed_pair_pass={seed_pair_pass_harm}/{len(seeds)}"
        f"  n_harm_min={n_harm_min}",
        flush=True,
    )
    print(
        f"  decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: harm_reduction_frac={harm_reduction_frac:.4f}"
            f" < {THRESH_C1_REDUCTION}"
            f" (MAP_ON does not reduce harm by >={int(THRESH_C1_REDUCTION*100)}%"
            f" vs MAP_ABLATED)"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: residue_events_min_on={residue_events_min_on}"
            f" < {THRESH_C2_MIN_RESIDUE_EVENTS}"
            " (viability map did not accumulate enough harm signal)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: seed_pair_pass_harm={seed_pair_pass_harm}/{len(seeds)}"
            " (inconsistent direction across seeds)"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: n_harm_min={n_harm_min}"
            f" < {THRESH_C4_MIN_EVENTS}"
            " (insufficient harm events -- data quality gate)"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: active_centers_min_on={active_centers_min_on}"
            f" < {THRESH_C5_MIN_ACTIVE_CENTERS}"
            " (viability map did not spread into enough RBF centers)"
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    # Interpretation
    if all_pass:
        interpretation = (
            "ARC-018 SUPPORTED: Hippocampal viability map provides measurable"
            f" harm reduction: MAP_ON={harm_rate_on:.4f}"
            f" vs MAP_ABLATED={harm_rate_ablated:.4f}"
            f" ({harm_reduction_frac*100:.1f}% reduction)."
            f" Residue field populated (min events={residue_events_min_on},"
            f" active centers={active_centers_min_on})."
            f" Consistent across all {len(seeds)} seeds."
            " E2 rollout scoring (terrain prior) enables the viability map"
            " navigation advantage."
        )
    elif c1_pass and c3_pass and c4_pass:
        interpretation = (
            "Partial support: harm rate reduction achieved"
            f" ({harm_reduction_frac*100:.1f}%, C1 PASS, C3 PASS)."
            f" Map populated: C2={'PASS' if c2_pass else 'FAIL'}"
            f" (residue_events_min={residue_events_min_on})."
            f" Map coverage: C5={'PASS' if c5_pass else 'FAIL'}"
            f" (active_centers_min={active_centers_min_on})."
            " ARC-018 harm advantage directionally supported."
        )
    elif c4_pass and harm_rate_on > harm_rate_ablated:
        interpretation = (
            "ARC-018 CONTRADICTED: viability-map-guided navigation"
            f" INCREASES harm rate (MAP_ON={harm_rate_on:.4f}"
            f" > MAP_ABLATED={harm_rate_ablated:.4f})."
            " Terrain prior may be misfiring: residue field populated but"
            " proposals are leading agent toward, not away from, hazards."
            " Check terrain scoring polarity and proposal-action decoding."
        )
    else:
        interpretation = (
            "ARC-018 inconclusive: data quality gate (C4) failed"
            f" (n_harm_min={n_harm_min} < {THRESH_C4_MIN_EVENTS})."
            " Insufficient harm events to evaluate viability map advantage."
            " Consider increasing num_hazards or harm_scale."
        )

    per_on_rows = "\n".join(
        f"  seed={r['seed']}: harm_rate={r['harm_rate_eval']:.4f}"
        f" harm_events={r['harm_events_eval']}/{r['total_steps_eval']}"
        f" residue_events={r['residue_events_accumulated']}"
        f" active_centers={r['active_centers']}"
        for r in results_on
    )
    per_ablated_rows = "\n".join(
        f"  seed={r['seed']}: harm_rate={r['harm_rate_eval']:.4f}"
        f" harm_events={r['harm_events_eval']}/{r['total_steps_eval']}"
        for r in results_ablated
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-120 -- ARC-018 Viability Map Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claim:** ARC-018\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Conditions:** VIABILITY_MAP_ON (residue+hippo+precision)"
        f" vs VIABILITY_MAP_ABLATED (no terrain, no precision update)\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps"
        f"  **Steps/ep:** {steps_per_episode}\n"
        f"**Env:** CausalGridWorldV2 size=6, 4 hazards, use_proxy_fields=True,"
        f" nav_bias={nav_bias}\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: harm_reduction_frac >= {THRESH_C1_REDUCTION}"
        f"  (MAP_ON reduces harm by >={int(THRESH_C1_REDUCTION*100)}%)\n"
        f"C2: residue_events_min_on >= {THRESH_C2_MIN_RESIDUE_EVENTS}"
        f"  (viability map actually populated)\n"
        f"C3: consistent harm reduction for ALL seeds"
        f"  (seed_pair_pass >= {len(seeds)})\n"
        f"C4: n_harm_min >= {THRESH_C4_MIN_EVENTS}"
        f"  (data quality gate)\n"
        f"C5: active_centers_min_on >= {THRESH_C5_MIN_ACTIVE_CENTERS}"
        f"  (viability map covers enough latent space)\n\n"
        f"## Results\n\n"
        f"| Condition | harm_rate | harm_events | steps |\n"
        f"|-----------|----------|------------|-------|\n"
        f"| MAP_ON  | {harm_rate_on:.4f} |"
        f" {sum(r['harm_events_eval'] for r in results_on)} |"
        f" {sum(r['total_steps_eval'] for r in results_on)} |\n"
        f"| MAP_ABLATED | {harm_rate_ablated:.4f} |"
        f" {sum(r['harm_events_eval'] for r in results_ablated)} |"
        f" {sum(r['total_steps_eval'] for r in results_ablated)} |\n\n"
        f"**harm_reduction_frac: {harm_reduction_frac:+.4f}**"
        f"  ({harm_rate_ablated:.4f} -> {harm_rate_on:.4f})\n"
        f"**residue_events_min_on: {residue_events_min_on}**"
        f"  **active_centers_min_on: {active_centers_min_on}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: harm_reduction >= {THRESH_C1_REDUCTION} |"
        f" {'PASS' if c1_pass else 'FAIL'} | {harm_reduction_frac:.4f} |\n"
        f"| C2: residue_events_min >= {THRESH_C2_MIN_RESIDUE_EVENTS} |"
        f" {'PASS' if c2_pass else 'FAIL'} | {residue_events_min_on} |\n"
        f"| C3: consistent across seeds |"
        f" {'PASS' if c3_pass else 'FAIL'} | {seed_pair_pass_harm}/{len(seeds)} |\n"
        f"| C4: n_harm_min >= {THRESH_C4_MIN_EVENTS} |"
        f" {'PASS' if c4_pass else 'FAIL'} | {n_harm_min} |\n"
        f"| C5: active_centers_min >= {THRESH_C5_MIN_ACTIVE_CENTERS} |"
        f" {'PASS' if c5_pass else 'FAIL'} | {active_centers_min_on} |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed\n\n"
        f"VIABILITY_MAP_ON:\n{per_on_rows}\n\n"
        f"VIABILITY_MAP_ABLATED:\n{per_ablated_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "harm_rate_on":              float(harm_rate_on),
        "harm_rate_ablated":         float(harm_rate_ablated),
        "harm_reduction_frac":       float(harm_reduction_frac),
        "residue_events_min_on":     float(residue_events_min_on),
        "active_centers_min_on":     float(active_centers_min_on),
        "seed_pair_pass_harm":       float(seed_pair_pass_harm),
        "n_harm_min":                float(n_harm_min),
        "n_seeds":                   float(len(seeds)),
        "alpha_world":               float(alpha_world),
        "nav_bias":                  float(nav_bias),
        "crit1_pass":                1.0 if c1_pass else 0.0,
        "crit2_pass":                1.0 if c2_pass else 0.0,
        "crit3_pass":                1.0 if c3_pass else 0.0,
        "crit4_pass":                1.0 if c4_pass else 0.0,
        "crit5_pass":                1.0 if c5_pass else 0.0,
        "criteria_met":              float(criteria_met),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int, nargs="+", default=[42, 123])
    parser.add_argument("--warmup",          type=int,   default=400)
    parser.add_argument("--eval-eps",        type=int,   default=50)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument("--nav-bias",        type=float, default=0.40)
    parser.add_argument("--dry-run",         action="store_true",
                        help="Run 3 warmup + 2 eval episodes per cell to check for errors.")
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        nav_bias=args.nav_bias,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["registered_thresholds"] = {
        "C1_harm_reduction_frac":          THRESH_C1_REDUCTION,
        "C2_min_residue_events_on":        THRESH_C2_MIN_RESIDUE_EVENTS,
        "C3_seed_pair_pass":               len(args.seeds),
        "C4_n_harm_min":                   THRESH_C4_MIN_EVENTS,
        "C5_min_active_centers_on":        THRESH_C5_MIN_ACTIVE_CENTERS,
    }
    result["seeds"] = list(args.seeds)
    result["conditions"] = ["VIABILITY_MAP_ON", "VIABILITY_MAP_ABLATED"]
    result["dispatch_mode"] = "discriminative_pair"
    result["backlog_id"] = "EVB-0014"

    if args.dry_run:
        print("\n[dry-run] Skipping file output.", flush=True)
        sys.exit(0)

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
