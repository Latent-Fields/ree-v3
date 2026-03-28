#!/opt/local/bin/python3
"""
V3-EXQ-136 -- MECH-092 Quiescent Replay: QUIESCENT_REPLAY_ON vs QUIESCENT_REPLAY_ABLATED

Claims: MECH-092
Proposal: EXP-0036 / EVB-0028

MECH-092 asserts:
  "Quiescent E3 heartbeat cycles trigger hippocampal SWR-equivalent replay for
   viability map consolidation."

Functional restatement (V3 testable proxy):
  During quiescent windows -- consecutive heartbeat cycles with no salient (harm)
  event -- the hippocampal module performs offline batch consolidation of recent
  trajectory experience. This consolidation improves the viability map (harm_eval_head
  quality) beyond what online updates alone achieve, because it integrates experiences
  that arrived too fast for real-time E3 processing.

What distinguishes this from EXQ-127 (MECH-030 post-episode consolidation):
  EXQ-127 fires replay once per episode boundary (post-episode). MECH-092 specifically
  claims that replay is triggered by quiescent INTRA-EPISODE windows -- K consecutive
  steps with no harm signal -- simulating the quiescent E3 heartbeat cycle trigger.
  The replay fires WITHIN episodes during quiet stretches, not only at episode end.
  This is the micro-quiescence analog described in MECH-092 notes.

Mechanism under test:
  If MECH-092 is correct, replay triggered specifically during quiescent intra-episode
  windows should provide consolidated harm representation that online-only updates miss,
  because quiescent windows represent idle E3 cycles where consolidation is safe and
  the hypothesis tag (MECH-094) is active.

Discriminative pair:
  QUIESCENT_REPLAY_ON    -- online updates + intra-episode quiescent replay
                            (N replay steps fire when K consecutive quiescent steps
                             accumulate, i.e., no harm_signal < 0 in last K steps)
  QUIESCENT_REPLAY_ABLATED -- online updates only; quiescence counter runs but
                            consolidation never fires

After equal total episode experience, evaluate whether QUIESCENT_REPLAY_ON achieves
higher harm_eval_head quality (gap between harm-positive and harm-negative scores),
and whether the quiescent windows were actually used (manipulation check: n_quiescent_triggers).

If quiescent-triggered replay genuinely helps:
  QUIESCENT_REPLAY_ON should produce a larger harm_eval gap than ABLATED.
  Per-seed delta (ON - ABLATED) should be positive and above threshold.
  Quiescent triggers should have fired at least THRESH_C5 times (manipulation check).

Experimental design:
  - Seeds:        [42, 123] (matched seeds, same environments per seed)
  - Env:          CausalGridWorldV2, size=6, 4 hazards, nav_bias=0.40
  - Warmup:       400 eps x 200 steps (online training, with/without quiescent replay)
  - Eval:         50 eps x 200 steps (no further training, measure harm_eval gap)
  - Conditions:   QUIESCENT_REPLAY_ON, QUIESCENT_REPLAY_ABLATED (x 2 seeds = 4 cells)
  - K (quiescence threshold): 10 consecutive steps with no harm signal
  - N (replay steps per trigger): 8 steps per quiescent trigger
  - replay_batch: 16 (8 harm + 8 safe samples per step)

Pre-registered acceptance criteria:
  C1: gap_replay_on >= 0.04  (QUIESCENT_REPLAY_ON harm eval gap above floor -- all seeds)
  C2: per-seed delta (ON - ABLATED) >= 0.02  (quiescent consolidation adds >=2pp -- all seeds)
  C3: gap_ablated >= 0.0  (ablation still learns; data quality check)
  C4: n_harm_eval_min >= 20  (sufficient harm events in eval phase -- all seeds)
  C5: n_quiescent_triggers_min >= 15  (quiescent replay fired enough times -- REPLAY_ON seeds)

Decision scoring:
  PASS (all 5):  retain_ree -- quiescent consolidation advantage demonstrated
  C1+C4+C5:      hybridize  -- mechanism fires but delta threshold not met
  C2 fails:      retire_ree_claim -- no quiescent consolidation advantage at current scale
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_136_mech092_quiescent_replay_pair"
CLAIM_IDS = ["MECH-092"]

# Pre-registered thresholds
THRESH_C1 = 0.04   # gap_replay_on >= 0.04 (harm eval above floor)
THRESH_C2 = 0.02   # per-seed delta (ON - ABLATED) >= 0.02
THRESH_C3 = 0.0    # gap_ablated >= 0.0 (ablation learns something)
THRESH_C4 = 20     # n_harm_eval_min >= 20 (data quality)
THRESH_C5 = 15     # n_quiescent_triggers_min >= 15 (manipulation check: replay fired)


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _hazard_approach_action(env: CausalGridWorldV2, n_actions: int) -> int:
    """Return action index that moves toward the nearest hazard gradient.
    Falls back to random if proxy fields are unavailable."""
    obs_dict = env._get_observation_dict()
    world_state = obs_dict.get("world_state", None)
    if world_state is None or not env.use_proxy_fields:
        return random.randint(0, n_actions - 1)
    # world_state[225:250] = hazard_field_view (5x5 flattened, proxy channel)
    field_view = world_state[225:250].numpy().reshape(5, 5)
    # Agent at center (2,2); actions: 0=up(-1,0), 1=down(+1,0), 2=left(0,-1), 3=right(0,+1)
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
    quiescent_replay_on: bool,
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
    quiescence_k: int,
    n_replay_steps: int,
    replay_batch: int,
    dry_run: bool,
) -> Dict:
    """Run one (seed, condition) cell; return harm eval gap metrics."""
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "QUIESCENT_REPLAY_ON" if quiescent_replay_on else "QUIESCENT_REPLAY_ABLATED"

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
        reafference_action_dim=0,  # isolate MECH-092 quiescent replay effect
    )

    agent = REEAgent(config)

    # Rolling consolidation buffer: (z_world latent, label) pairs
    # label=1.0 for harm-positive, 0.0 for harm-negative
    consol_buf_zw: List[torch.Tensor] = []
    consol_buf_labels: List[float] = []
    MAX_BUF = 2000

    # Separate optimizers: standard (E1+E2) and harm_eval
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    counts: Dict[str, int] = {
        "hazard_approach": 0,
        "env_caused_hazard": 0,
        "agent_caused_hazard": 0,
        "none": 0,
    }
    n_quiescent_triggers = 0

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

        # Quiescence counter: tracks consecutive steps with no harm
        quiescence_counter = 0

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            # Biased navigation: nav_bias chance to move toward nearest hazard
            if random.random() < nav_bias:
                action_idx = _hazard_approach_action(env, n_actions)
            else:
                action_idx = random.randint(0, n_actions - 1)
            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if ttype in counts:
                counts[ttype] += 1

            is_harm = float(harm_signal) < 0

            # Update quiescence counter:
            # Salient event (harm) resets it; quiescent step increments it
            if is_harm:
                quiescence_counter = 0
            else:
                quiescence_counter += 1

            # Collect into rolling buffer for online and quiescent replay
            consol_buf_zw.append(z_world_curr)
            consol_buf_labels.append(1.0 if is_harm else 0.0)
            if len(consol_buf_zw) > MAX_BUF:
                consol_buf_zw = consol_buf_zw[-MAX_BUF:]
                consol_buf_labels = consol_buf_labels[-MAX_BUF:]

            # Standard E1 + E2 losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # Online harm_eval step (4-sample mini-batch if sufficient data)
            n_harm_buf = sum(1 for lbl in consol_buf_labels if lbl > 0.5)
            n_safe_buf = sum(1 for lbl in consol_buf_labels if lbl <= 0.5)
            if n_harm_buf >= 4 and n_safe_buf >= 4:
                harm_idxs = [i for i, lbl in enumerate(consol_buf_labels) if lbl > 0.5]
                safe_idxs = [i for i, lbl in enumerate(consol_buf_labels) if lbl <= 0.5]
                k = min(8, min(len(harm_idxs), len(safe_idxs)))
                selected_harm = random.sample(harm_idxs, k)
                selected_safe = random.sample(safe_idxs, k)
                selected = selected_harm + selected_safe
                zw_b = torch.cat([consol_buf_zw[i] for i in selected], dim=0)
                labels_b = torch.tensor(
                    [consol_buf_labels[i] for i in selected],
                    dtype=torch.float32,
                ).unsqueeze(1)
                pred_online = agent.e3.harm_eval(zw_b)
                loss_online = F.mse_loss(pred_online, labels_b)
                if loss_online.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    loss_online.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_eval_optimizer.step()

            # --- QUIESCENT INTRA-EPISODE REPLAY (QUIESCENT_REPLAY_ON only) ---
            # Fires when quiescence_counter reaches quiescence_k:
            # K consecutive steps with no harm event = quiescent E3 heartbeat window.
            # Simulates hippocampal SWR-equivalent offline batch consolidation during
            # idle inter-action periods (MECH-092 micro-quiescence mechanism).
            # After firing, reset counter to avoid re-triggering every step.
            # QUIESCENT_REPLAY_ABLATED: counter advances but replay never fires.
            if quiescent_replay_on and quiescence_counter >= quiescence_k:
                if n_harm_buf >= 2 and n_safe_buf >= 2:
                    # Run n_replay_steps consolidation updates from rolling buffer
                    for _ in range(n_replay_steps):
                        harm_idxs_r = [
                            i for i, lbl in enumerate(consol_buf_labels) if lbl > 0.5
                        ]
                        safe_idxs_r = [
                            i for i, lbl in enumerate(consol_buf_labels) if lbl <= 0.5
                        ]
                        k_pos = min(replay_batch // 2, len(harm_idxs_r))
                        k_neg = min(replay_batch // 2, len(safe_idxs_r))
                        if k_pos < 1 or k_neg < 1:
                            break
                        pos_idx = random.sample(harm_idxs_r, k_pos)
                        neg_idx = random.sample(safe_idxs_r, k_neg)
                        zw_pos = torch.cat([consol_buf_zw[i] for i in pos_idx], dim=0)
                        zw_neg = torch.cat([consol_buf_zw[i] for i in neg_idx], dim=0)
                        zw_b_q = torch.cat([zw_pos, zw_neg], dim=0)
                        target_b_q = torch.cat([
                            torch.ones(k_pos, 1, device=agent.device),
                            torch.zeros(k_neg, 1, device=agent.device),
                        ], dim=0)
                        pred_quiescent = agent.e3.harm_eval(zw_b_q)
                        loss_quiescent = F.mse_loss(pred_quiescent, target_b_q)
                        if loss_quiescent.requires_grad:
                            harm_eval_optimizer.zero_grad()
                            loss_quiescent.backward()
                            torch.nn.utils.clip_grad_norm_(
                                agent.e3.harm_eval_head.parameters(), 0.5
                            )
                            harm_eval_optimizer.step()
                    n_quiescent_triggers += 1
                    # Reset counter: quiescent window consumed; wait for next K steps
                    quiescence_counter = 0

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == actual_warmup - 1:
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{actual_warmup}"
                f" approach={counts['hazard_approach']}"
                f" contact={counts['env_caused_hazard']+counts['agent_caused_hazard']}"
                f" quiescent_triggers={n_quiescent_triggers}",
                flush=True,
            )

    # --- EVAL ---
    agent.eval()

    eval_scores_harm: List[float] = []   # harm_eval scores at harm events
    eval_scores_safe: List[float] = []   # harm_eval scores at safe events
    n_fatal = 0

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

            action_idx = random.randint(0, n_actions - 1)
            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)

            is_harm = float(harm_signal) < 0

            try:
                with torch.no_grad():
                    score = float(agent.e3.harm_eval(z_world_curr).item())
                if is_harm:
                    eval_scores_harm.append(score)
                else:
                    eval_scores_safe.append(score)
            except Exception:
                n_fatal += 1

            if done:
                break

    n_harm_eval = len(eval_scores_harm)
    n_safe_eval = len(eval_scores_safe)

    mean_harm = float(sum(eval_scores_harm) / max(1, n_harm_eval))
    mean_safe = float(sum(eval_scores_safe) / max(1, n_safe_eval))
    gap = mean_harm - mean_safe

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" gap={gap:.4f} mean_harm={mean_harm:.4f} mean_safe={mean_safe:.4f}"
        f" n_harm={n_harm_eval} n_safe={n_safe_eval}"
        f" quiescent_triggers={n_quiescent_triggers}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "quiescent_replay_on": quiescent_replay_on,
        "harm_eval_gap": float(gap),
        "mean_harm_score": float(mean_harm),
        "mean_safe_score": float(mean_safe),
        "n_harm_eval": int(n_harm_eval),
        "n_safe_eval": int(n_safe_eval),
        "n_fatal": int(n_fatal),
        "n_quiescent_triggers": int(n_quiescent_triggers),
        "train_approach": int(counts["hazard_approach"]),
        "train_contact": int(counts["env_caused_hazard"] + counts["agent_caused_hazard"]),
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
    quiescence_k: int = 10,
    n_replay_steps: int = 8,
    replay_batch: int = 16,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """Discriminative pair: QUIESCENT_REPLAY_ON vs QUIESCENT_REPLAY_ABLATED."""
    results_on:      List[Dict] = []
    results_ablated: List[Dict] = []

    for seed in seeds:
        for quiescent_replay_on in [True, False]:
            label = "QUIESCENT_REPLAY_ON" if quiescent_replay_on else "QUIESCENT_REPLAY_ABLATED"
            print(
                f"\n[V3-EXQ-136] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" quiescence_k={quiescence_k} n_replay_steps={n_replay_steps}"
                f" replay_batch={replay_batch}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                quiescent_replay_on=quiescent_replay_on,
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
                quiescence_k=quiescence_k,
                n_replay_steps=n_replay_steps,
                replay_batch=replay_batch,
                dry_run=dry_run,
            )
            if quiescent_replay_on:
                results_on.append(r)
            else:
                results_ablated.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    gap_on      = _avg(results_on,      "harm_eval_gap")
    gap_ablated = _avg(results_ablated, "harm_eval_gap")
    delta_gap   = gap_on - gap_ablated

    n_harm_min = min(r["n_harm_eval"] for r in results_on + results_ablated)
    n_quiescent_min = min(r["n_quiescent_triggers"] for r in results_on)

    # Pre-registered PASS criteria
    # C1: QUIESCENT_REPLAY_ON gap >= threshold in ALL seeds
    c1_pass = all(r["harm_eval_gap"] >= THRESH_C1 for r in results_on)
    # C2: per-seed delta (ON - ABLATED) >= threshold in ALL seeds
    per_seed_deltas = []
    for r_on in results_on:
        matching = [r for r in results_ablated if r["seed"] == r_on["seed"]]
        if matching:
            per_seed_deltas.append(r_on["harm_eval_gap"] - matching[0]["harm_eval_gap"])
    c2_pass = len(per_seed_deltas) > 0 and all(d >= THRESH_C2 for d in per_seed_deltas)
    # C3: ABLATED gap >= 0 (ablation learns something; data quality)
    c3_pass = gap_ablated >= THRESH_C3
    # C4: sufficient harm events in eval phase
    c4_pass = n_harm_min >= THRESH_C4
    # C5: quiescent replay fired at least THRESH_C5 times per seed (manipulation check)
    c5_pass = n_quiescent_min >= THRESH_C5

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    # Decision scoring
    if all_pass:
        decision = "retain_ree"
    elif c1_pass and c4_pass and c5_pass:
        # Quiescent mechanism fires, gap above floor, but delta threshold not met
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-136] Results:", flush=True)
    print(
        f"  gap_on={gap_on:.4f}"
        f" gap_ablated={gap_ablated:.4f}"
        f" delta_gap={delta_gap:+.4f}",
        flush=True,
    )
    print(
        f"  n_harm_min={n_harm_min}"
        f"  n_quiescent_min={n_quiescent_min}"
        f"  per_seed_deltas={[f'{d:.4f}' for d in per_seed_deltas]}"
        f"  decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failing_seeds = [r["seed"] for r in results_on if r["harm_eval_gap"] < THRESH_C1]
        failure_notes.append(
            f"C1 FAIL: QUIESCENT_REPLAY_ON gap below {THRESH_C1} in seeds {failing_seeds}"
            " -- harm_eval_head not learning meaningful representation even with quiescent replay"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: per-seed deltas {[f'{d:.4f}' for d in per_seed_deltas]} < {THRESH_C2}"
            " -- quiescent consolidation does not add >=2pp over online-only training"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: gap_ablated={gap_ablated:.4f} < {THRESH_C3}"
            " -- ablation fails to learn harm representation at all (confound)"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: n_harm_min={n_harm_min} < {THRESH_C4}"
            " -- insufficient harm events in eval phase for reliable estimate"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: n_quiescent_triggers_min={n_quiescent_min} < {THRESH_C5}"
            " -- quiescent replay fired too rarely (manipulation check fail);"
            f" consider reducing quiescence_k (currently {quiescence_k})"
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    # Interpretation
    if all_pass:
        interpretation = (
            "MECH-092 SUPPORTED at V3 proxy level: quiescent-triggered intra-episode"
            f" replay improves harm_eval_head quality (gap_on={gap_on:.4f} vs"
            f" gap_ablated={gap_ablated:.4f}, delta={delta_gap:+.4f} across"
            f" {len(seeds)} seeds, n_quiescent_min={n_quiescent_min})."
            " Consolidation triggered during quiescent E3 heartbeat windows (K="
            f"{quiescence_k} consecutive non-harm steps) drives better harm"
            " representation than online-only updates. Supports the core MECH-092"
            " thesis that idle E3 cycles trigger offline viability map consolidation."
        )
    elif c1_pass and c4_pass and c5_pass:
        interpretation = (
            f"Partial support: QUIESCENT_REPLAY_ON achieves gap={gap_on:.4f} (C1 PASS)"
            f" and quiescent triggers fired {n_quiescent_min}+ times (C5 PASS),"
            f" but per-seed delta={delta_gap:+.4f} (C2 FAIL, threshold={THRESH_C2})."
            " The quiescent mechanism is operational but the advantage over online-only"
            " training is below the pre-registered threshold. Possible explanations:"
            " online updates alone are sufficient at this scale, or n_replay_steps="
            f"{n_replay_steps} is too few to produce a detectable advantage per trigger."
        )
    elif not c5_pass:
        interpretation = (
            f"C5 FAIL: quiescent replay triggered only {n_quiescent_min} times"
            f" (threshold={THRESH_C5}). With nav_bias=0.40 the agent frequently"
            " approaches hazards, reducing quiescent window duration. The quiescent"
            " mechanism may need higher nav_bias or lower quiescence_k to accumulate"
            " enough idle windows. Harm_eval gap (C1 {'PASS' if c1_pass else 'FAIL'})."
        )
    else:
        interpretation = (
            f"MECH-092 V3 proxy NOT supported: gap_on={gap_on:.4f} (C1"
            f" {'PASS' if c1_pass else 'FAIL'}), delta={delta_gap:+.4f} (C2"
            f" {'PASS' if c2_pass else 'FAIL'}). Quiescent intra-episode replay"
            " does not demonstrate improvement in harm_eval_head quality at this"
            " training scale."
        )

    per_on_rows = "\n".join(
        f"  seed={r['seed']}: gap={r['harm_eval_gap']:.4f}"
        f" n_harm={r['n_harm_eval']}"
        f" n_quiescent_triggers={r['n_quiescent_triggers']}"
        for r in results_on
    )
    per_ablated_rows = "\n".join(
        f"  seed={r['seed']}: gap={r['harm_eval_gap']:.4f}"
        f" n_harm={r['n_harm_eval']}"
        for r in results_ablated
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-136 -- MECH-092 Quiescent Replay Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-092\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Conditions:** QUIESCENT_REPLAY_ON vs QUIESCENT_REPLAY_ABLATED\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n"
        f"**Env:** CausalGridWorldV2 size=6, 4 hazards, nav_bias=0.40\n"
        f"**quiescence_k:** {quiescence_k} (consecutive non-harm steps to trigger replay)\n"
        f"**n_replay_steps:** {n_replay_steps}  **replay_batch:** {replay_batch}\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: gap_replay_on >= {THRESH_C1} in all seeds  (QUIESCENT_REPLAY_ON harm eval above floor)\n"
        f"C2: per-seed delta (ON - ABLATED) >= {THRESH_C2} in all seeds  (quiescent consolidation adds >=2pp)\n"
        f"C3: gap_ablated >= {THRESH_C3}  (ablation learns something; data quality)\n"
        f"C4: n_harm_eval_min >= {THRESH_C4}  (sufficient harm events in eval)\n"
        f"C5: n_quiescent_triggers_min >= {THRESH_C5}  (manipulation check: quiescent windows fire)\n\n"
        f"## Results\n\n"
        f"| Condition | gap (avg) | mean_harm | mean_safe |\n"
        f"|-----------|-----------|-----------|----------|\n"
        f"| QUIESCENT_REPLAY_ON      | {gap_on:.4f}"
        f" | {_avg(results_on,      'mean_harm_score'):.4f}"
        f" | {_avg(results_on,      'mean_safe_score'):.4f} |\n"
        f"| QUIESCENT_REPLAY_ABLATED | {gap_ablated:.4f}"
        f" | {_avg(results_ablated, 'mean_harm_score'):.4f}"
        f" | {_avg(results_ablated, 'mean_safe_score'):.4f} |\n\n"
        f"**delta_gap (ON - ABLATED): {delta_gap:+.4f}**\n"
        f"**n_quiescent_triggers (ON, min across seeds): {n_quiescent_min}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: gap_on >= {THRESH_C1} (all seeds) | {'PASS' if c1_pass else 'FAIL'}"
        f" | {gap_on:.4f} |\n"
        f"| C2: per-seed delta >= {THRESH_C2} (all seeds) | {'PASS' if c2_pass else 'FAIL'}"
        f" | {[round(d, 4) for d in per_seed_deltas]} |\n"
        f"| C3: gap_ablated >= {THRESH_C3} | {'PASS' if c3_pass else 'FAIL'}"
        f" | {gap_ablated:.4f} |\n"
        f"| C4: n_harm_min >= {THRESH_C4} | {'PASS' if c4_pass else 'FAIL'}"
        f" | {n_harm_min} |\n"
        f"| C5: n_quiescent_triggers_min >= {THRESH_C5} | {'PASS' if c5_pass else 'FAIL'}"
        f" | {n_quiescent_min} |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed\n\n"
        f"QUIESCENT_REPLAY_ON:\n{per_on_rows}\n\n"
        f"QUIESCENT_REPLAY_ABLATED:\n{per_ablated_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "gap_replay_on":              float(gap_on),
        "gap_replay_ablated":         float(gap_ablated),
        "delta_gap":                  float(delta_gap),
        "n_harm_eval_min":            float(n_harm_min),
        "n_quiescent_triggers_min":   float(n_quiescent_min),
        "n_seeds":                    float(len(seeds)),
        "quiescence_k":               float(quiescence_k),
        "n_replay_steps":             float(n_replay_steps),
        "replay_batch":               float(replay_batch),
        "alpha_world":                float(alpha_world),
        "nav_bias":                   float(nav_bias),
        "per_seed_delta_min":         float(min(per_seed_deltas)) if per_seed_deltas else 0.0,
        "per_seed_delta_max":         float(max(per_seed_deltas)) if per_seed_deltas else 0.0,
        "crit1_pass":                 1.0 if c1_pass else 0.0,
        "crit2_pass":                 1.0 if c2_pass else 0.0,
        "crit3_pass":                 1.0 if c3_pass else 0.0,
        "crit4_pass":                 1.0 if c4_pass else 0.0,
        "crit5_pass":                 1.0 if c5_pass else 0.0,
        "criteria_met":               float(criteria_met),
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
        "fatal_error_count": sum(r["n_fatal"] for r in results_on + results_ablated),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",            type=int, nargs="+", default=[42, 123])
    parser.add_argument("--warmup",           type=int,   default=400)
    parser.add_argument("--eval-eps",         type=int,   default=50)
    parser.add_argument("--steps",            type=int,   default=200)
    parser.add_argument("--alpha-world",      type=float, default=0.9)
    parser.add_argument("--alpha-self",       type=float, default=0.3)
    parser.add_argument("--harm-scale",       type=float, default=0.02)
    parser.add_argument("--proximity-scale",  type=float, default=0.05)
    parser.add_argument("--nav-bias",         type=float, default=0.40)
    parser.add_argument("--quiescence-k",     type=int,   default=10)
    parser.add_argument("--n-replay-steps",   type=int,   default=8)
    parser.add_argument("--replay-batch",     type=int,   default=16)
    parser.add_argument("--dry-run",          action="store_true",
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
        quiescence_k=args.quiescence_k,
        n_replay_steps=args.n_replay_steps,
        replay_batch=args.replay_batch,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["registered_thresholds"] = {
        "C1_gap_replay_on":              THRESH_C1,
        "C2_per_seed_delta":             THRESH_C2,
        "C3_gap_ablated":                THRESH_C3,
        "C4_n_harm_eval_min":            THRESH_C4,
        "C5_n_quiescent_triggers_min":   THRESH_C5,
    }
    result["seeds"] = list(args.seeds)
    result["conditions"] = ["QUIESCENT_REPLAY_ON", "QUIESCENT_REPLAY_ABLATED"]
    result["dispatch_mode"] = "discriminative_pair"
    result["backlog_id"] = "EVB-0028"
    result["proposal_id"] = "EXP-0036"

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
