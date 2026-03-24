#!/opt/local/bin/python3
"""
V3-EXQ-047j -- MECH-095 TPJ Routing via CE Head (fix for EXQ-047i)

Claims: MECH-095, SD-005

Bugs found in EXQ-047i (2026-03-24):

Bug 1 -- Routing loss near-zero (route_loss=0.0000):
  EXQ-047i used MSE(z_world_curr, z_world_prev) for SELF_CAUSED_LOCO transitions.
  With alpha_world=0.9, consecutive z_world values differ by only ~10% of the new
  encoding delta: dz_world ~= 0.1*(encode(obs) - z_world_prev). MSE is ~0.01,
  times lambda_route=0.1 -> ~0.001, printing as 0.0000 at 4 decimal places.
  The routing loss applied essentially zero gradient to the encoder.

Bug 2 -- Contact probe majority-class collapse (contact_world=contact_self=0.911):
  n_contact=164 out of ~2000 probe steps -> 8.2% positive rate.
  A probe that always predicts "no contact" achieves ~91.8% accuracy.
  Both z_world and z_self probes collapsed to the majority predictor, giving
  identical "accuracy" of 0.911 and contact_dissociation=0.000 trivially.
  The dissociation metric was degenerate regardless of the routing mechanism.

Fixes in EXQ-047j:

Fix 1 -- CE routing head replaces MSE stability loss:
  A small binary classification head predicts is_world_caused from z_world.
  Training signal: at each step, the head must predict whether the PREVIOUS
  transition was world-caused (WORLD_CAUSED set) or not. Backprop flows through
  z_world_curr, pushing z_world to encode world-causation information.
  This is a direct gradient signal, not a near-zero MSE difference.
  The head is a 2-layer MLP: world_dim -> 16 -> 1 (sigmoid output).
  Optimizer is shared with agent standard_params for simplicity.

Fix 2 -- Balanced contact probe replaces majority-class probe:
  Contact probe uses balanced undersampling: subsample no-contact steps to
  match the number of contact steps (n_contact). Fit the probe on balanced data.
  Report contact_recall_world = recall on the contact (positive) class only.
  This removes the majority-class baseline bias entirely.
  A contact_recall of 0.5 = chance; > 0.5 means z_world actually predicts contacts.

Updated PASS criteria:
  C1: contact_recall_world > 0.55
      (z_world probe recall on contact class > chance by meaningful margin)
  C2: contact_recall_world > contact_recall_baseline + 0.04
      (routing improves contact recall -- validates the mechanism)
  C3: action_dissoc_self_world > 0
      (z_self still better than z_world at predicting actions -- split is intact)
  C4: n_contact_probe >= 20
  C5: no fatal errors

  Also report mean_routing_loss (must be > 0.001 to confirm CE head is active).

Decision:
  retain_ree (MECH-095 validated):   C1+C2+C3 pass + routing_loss > 0.001
  hybridize (partial routing signal): C1+C2 pass but C3 marginal
  retire_ree_claim:                   C1 OR C2 fail
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_047j_tpj_routing_ce"
CLAIM_IDS = ["MECH-095", "SD-005"]

# Transition types considered world-caused (routing head should predict 1.0)
WORLD_CAUSED = frozenset({"env_caused_hazard", "hazard_approach"})


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _fit_balanced_probe(
    X_pos: torch.Tensor,
    X_neg: torch.Tensor,
    n_steps: int = 400,
    lr: float = 1e-2,
) -> float:
    """
    Fit a binary probe on balanced data. Returns recall on the positive class.
    X_pos: [n_pos, dim] -- contact (positive) examples
    X_neg: [n_neg, dim] -- no-contact (negative) examples, will be undersampled
    Returns: recall on positive class (fraction of positives correctly classified)
    """
    n = min(len(X_pos), len(X_neg))
    if n < 5:
        return 0.0

    # Undersample to balance
    idx_pos = torch.randperm(len(X_pos))[:n]
    idx_neg = torch.randperm(len(X_neg))[:n]
    X = torch.cat([X_pos[idx_pos], X_neg[idx_neg]], dim=0).float()
    y = torch.cat([
        torch.ones(n, dtype=torch.long),
        torch.zeros(n, dtype=torch.long),
    ])

    probe = nn.Linear(X.shape[1], 2)
    opt = optim.Adam(probe.parameters(), lr=lr)
    for _ in range(n_steps):
        logits = probe(X)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = probe(X_pos[idx_pos]).argmax(dim=1)
        recall = float((preds == 1).float().mean().item())
    return recall


def _fit_action_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    n_steps: int = 400,
    lr: float = 1e-2,
) -> float:
    """Standard accuracy probe for action prediction."""
    X = X.detach().float()
    y = y.detach().long()
    probe = nn.Linear(X.shape[1], n_classes)
    opt = optim.Adam(probe.parameters(), lr=lr)
    for _ in range(n_steps):
        logits = probe(X)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = probe(X).argmax(dim=1)
        acc = float((preds == y).float().mean().item())
    return acc


def _run_single(
    seed: int,
    use_routing: bool,
    warmup_episodes: int,
    probe_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
    lambda_route: float,
) -> Dict:
    """Run one (seed, routing condition) cell."""
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "ROUTED" if use_routing else "BASELINE"

    env = CausalGridWorldV2(
        seed=seed,
        size=12,
        num_hazards=4,
        num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,
    )
    # Always use SPLIT mode (SD-005) -- isolate routing effect
    config.latent.unified_latent_mode = False

    agent = REEAgent(config)

    # CE routing head: world_dim -> 16 -> 1 (predicts is_world_caused from z_world)
    # Only used in ROUTED condition; included in standard_params for simplicity.
    routing_head: nn.Module = nn.Sequential(
        nn.Linear(world_dim, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    # Include routing_head in standard_params so it trains with the agent
    standard_params = (
        list(agent.parameters()) + list(routing_head.parameters())
        if use_routing else list(agent.parameters())
    )
    # Exclude harm_eval_head from standard optimizer
    standard_params = [
        p for p in standard_params
        if not any(
            p is ph
            for ph in agent.e3.harm_eval_head.parameters()
        )
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    counts: Dict[str, int] = {
        "hazard_approach": 0, "env_caused_hazard": 0,
        "agent_caused_hazard": 0, "none": 0,
    }
    total_routing_loss = 0.0
    routing_steps = 0

    # --- WARMUP TRAINING ---
    agent.train()
    routing_head.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        prev_ttype: str = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world   # [1, world_dim], has grad

            # --- CE routing loss (applied BEFORE action, using prev transition) ---
            routing_loss = torch.zeros(1)
            if use_routing and prev_ttype is not None:
                is_world = 1.0 if prev_ttype in WORLD_CAUSED else 0.0
                label = torch.tensor([[is_world]])
                routing_loss = lambda_route * F.binary_cross_entropy_with_logits(
                    routing_head(z_world_curr),
                    label,
                )
                total_routing_loss += routing_loss.item()
                routing_steps += 1

            z_world_track = z_world_curr.detach()

            # Take action
            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            prev_ttype = info.get("transition_type", "none")
            ttype = prev_ttype
            if ttype in counts:
                counts[ttype] += 1

            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_track)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_track)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss + routing_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b = torch.cat([zw_pos, zw_neg], dim=0)
                target = torch.cat([
                    torch.ones(k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5,
                    )
                    harm_eval_optimizer.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            mean_route = (
                total_routing_loss / max(1, routing_steps) if use_routing else 0.0
            )
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" approach={counts['hazard_approach']}"
                f" contact={counts['env_caused_hazard']+counts['agent_caused_hazard']}"
                f" route_loss={mean_route:.4f}",
                flush=True,
            )

    # --- PROBE COLLECTION ---
    agent.eval()
    routing_head.eval()

    probe_self_contact: List[torch.Tensor] = []
    probe_self_no_contact: List[torch.Tensor] = []
    probe_world_contact: List[torch.Tensor] = []
    probe_world_no_contact: List[torch.Tensor] = []
    probe_self_all: List[torch.Tensor] = []
    probe_world_all: List[torch.Tensor] = []
    probe_actions: List[int] = []
    n_fatal = 0

    for _ in range(probe_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, _, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            try:
                with torch.no_grad():
                    obs_body_next  = obs_dict["body_state"]
                    obs_world_next = obs_dict["world_state"]
                    latent_next = agent.sense(obs_body_next, obs_world_next)
                    agent.clock.advance()

                    zs = latent_next.z_self.detach()
                    zw = latent_next.z_world.detach()
                    is_contact = ttype in ("hazard_approach", "agent_caused_hazard",
                                           "env_caused_hazard")

                    probe_self_all.append(zs)
                    probe_world_all.append(zw)
                    probe_actions.append(action_idx)

                    if is_contact:
                        probe_self_contact.append(zs)
                        probe_world_contact.append(zw)
                    else:
                        probe_self_no_contact.append(zs)
                        probe_world_no_contact.append(zw)

                    _, _, done2, _, obs_dict = env.step(
                        _action_to_onehot(
                            random.randint(0, env.action_dim - 1),
                            env.action_dim, agent.device,
                        )
                    )
                    if done2:
                        break
            except Exception:
                n_fatal += 1

            if done:
                break

    # --- FIT PROBES ---
    action_acc_self  = 0.0
    action_acc_world = 0.0
    contact_recall_self  = 0.0
    contact_recall_world = 0.0
    n_contact_probe = len(probe_self_contact)

    if len(probe_self_all) >= 20:
        X_self_all  = torch.cat(probe_self_all,  dim=0).float()
        X_world_all = torch.cat(probe_world_all, dim=0).float()
        y_action    = torch.tensor(probe_actions, dtype=torch.long)

        action_acc_self  = _fit_action_probe(X_self_all,  y_action, env.action_dim)
        action_acc_world = _fit_action_probe(X_world_all, y_action, env.action_dim)

    if n_contact_probe >= 5 and len(probe_self_no_contact) >= 5:
        X_self_pos  = torch.cat(probe_self_contact,     dim=0).float()
        X_self_neg  = torch.cat(probe_self_no_contact,  dim=0).float()
        X_world_pos = torch.cat(probe_world_contact,    dim=0).float()
        X_world_neg = torch.cat(probe_world_no_contact, dim=0).float()

        contact_recall_self  = _fit_balanced_probe(X_self_pos,  X_self_neg)
        contact_recall_world = _fit_balanced_probe(X_world_pos, X_world_neg)

    action_dissociation = action_acc_self - action_acc_world

    mean_route_loss = total_routing_loss / max(1, routing_steps) if use_routing else 0.0

    print(
        f"  [probe] seed={seed} cond={cond_label}"
        f" action: self={action_acc_self:.3f} world={action_acc_world:.3f}"
        f" dissoc={action_dissociation:+.3f}"
        f"  contact_recall: world={contact_recall_world:.3f}"
        f" self={contact_recall_self:.3f}"
        f" n_contact={n_contact_probe}"
        f" route_loss={mean_route_loss:.4f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "use_routing": use_routing,
        "action_acc_self":        float(action_acc_self),
        "action_acc_world":       float(action_acc_world),
        "action_dissociation":    float(action_dissociation),
        "contact_recall_self":    float(contact_recall_self),
        "contact_recall_world":   float(contact_recall_world),
        "n_contact_probe":        int(n_contact_probe),
        "n_probe_steps":          int(len(probe_self_all)),
        "mean_routing_loss":      float(mean_route_loss),
        "train_approach_events":  int(counts["hazard_approach"]),
        "train_contact_events":   int(
            counts["env_caused_hazard"] + counts["agent_caused_hazard"]
        ),
        "n_fatal": int(n_fatal),
    }


def run(
    seeds: Tuple = (42, 7),
    warmup_episodes: int = 400,
    probe_episodes: int = 20,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    lambda_route: float = 0.1,
    **kwargs,
) -> dict:
    """ROUTED (CE head) vs BASELINE (both SPLIT) -- balanced contact probe."""
    results_routed:   List[Dict] = []
    results_baseline: List[Dict] = []

    for seed in seeds:
        for use_routing in [True, False]:
            label = "ROUTED" if use_routing else "BASELINE"
            print(
                f"\n[V3-EXQ-047j] {label} seed={seed}"
                f" warmup={warmup_episodes} lambda_route={lambda_route}"
                f" alpha_world={alpha_world}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                use_routing=use_routing,
                warmup_episodes=warmup_episodes,
                probe_episodes=probe_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
                lambda_route=lambda_route,
            )
            if use_routing:
                results_routed.append(r)
            else:
                results_baseline.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    contact_recall_routed   = _avg(results_routed,   "contact_recall_world")
    contact_recall_baseline = _avg(results_baseline, "contact_recall_world")
    action_dissoc_routed    = _avg(results_routed,   "action_dissociation")
    mean_route_loss         = _avg(results_routed,   "mean_routing_loss")
    n_contact_min = min(
        r["n_contact_probe"] for r in results_routed + results_baseline
    )

    c1_pass = contact_recall_routed > 0.55
    c2_pass = (contact_recall_routed - contact_recall_baseline) > 0.04
    c3_pass = action_dissoc_routed > 0.0
    c4_pass = n_contact_min >= 20
    c5_pass = all(r["n_fatal"] == 0 for r in results_routed + results_baseline)

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    routing_active = mean_route_loss > 0.001

    if all_pass and routing_active:
        decision = "retain_ree"
    elif c1_pass and c2_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-047j] Final results:", flush=True)
    print(
        f"  contact_recall: routed={contact_recall_routed:.3f}"
        f"  baseline={contact_recall_baseline:.3f}"
        f"  improvement={contact_recall_routed - contact_recall_baseline:+.3f}",
        flush=True,
    )
    print(
        f"  action_dissoc_routed={action_dissoc_routed:+.3f}"
        f"  mean_route_loss={mean_route_loss:.4f}"
        f"  routing_active={routing_active}",
        flush=True,
    )
    print(
        f"  decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not routing_active:
        failure_notes.append(
            f"WARN: mean_route_loss={mean_route_loss:.4f} <= 0.001"
            " (CE routing head not producing gradient -- check implementation)"
        )
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: contact_recall_world_routed={contact_recall_routed:.3f} <= 0.55"
        )
    if not c2_pass:
        gap = contact_recall_routed - contact_recall_baseline
        failure_notes.append(
            f"C2 FAIL: recall improvement={gap:+.3f} <= 0.04"
            f" (routed={contact_recall_routed:.3f}"
            f" vs baseline={contact_recall_baseline:.3f})"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: action_dissoc_routed={action_dissoc_routed:+.3f} <= 0"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: n_contact_min={n_contact_min} < 20")
    if not c5_pass:
        failure_notes.append("C5 FAIL: fatal errors occurred")

    for note in failure_notes:
        print(f"  {note}", flush=True)

    per_routed_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" contact_recall_world={r['contact_recall_world']:.3f}"
        f" contact_recall_self={r['contact_recall_self']:.3f}"
        f" action_dissoc={r['action_dissociation']:+.3f}"
        f" route_loss={r['mean_routing_loss']:.4f}"
        f" n_contact={r['n_contact_probe']}"
        for r in results_routed
    )
    per_baseline_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" contact_recall_world={r['contact_recall_world']:.3f}"
        f" contact_recall_self={r['contact_recall_self']:.3f}"
        f" action_dissoc={r['action_dissociation']:+.3f}"
        f" n_contact={r['n_contact_probe']}"
        for r in results_baseline
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-047j -- MECH-095 TPJ Routing via CE Head\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-095, SD-005\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**lambda_route:** {lambda_route}\n"
        f"**Warmup:** {warmup_episodes} eps  **Probe:** {probe_episodes} eps\n"
        f"**mean_routing_loss (ROUTED):** {mean_route_loss:.4f}"
        f" ({'ACTIVE' if routing_active else 'INACTIVE'})\n"
        f"**Bug fixes vs EXQ-047i:**\n"
        f"  (1) CE routing head replaces MSE-stability loss (was near-zero due to alpha=0.9 EMA).\n"
        f"  (2) Balanced contact probe (undersampled) replaces majority-class-collapse probe.\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: contact_recall_world_routed > 0.55\n"
        f"C2: recall improvement (routed - baseline) > 0.04\n"
        f"C3: action_dissoc_routed > 0 (z_self better than z_world for actions)\n"
        f"C4: n_contact_probe >= 20\n"
        f"C5: no fatal errors\n\n"
        f"## Results\n\n"
        f"| Condition | contact_recall | action_dissoc | route_loss |\n"
        f"|-----------|----------------|---------------|------------|\n"
        f"| ROUTED    | {contact_recall_routed:.3f}          |"
        f" {action_dissoc_routed:+.3f}        | {mean_route_loss:.4f}     |\n"
        f"| BASELINE  | {contact_recall_baseline:.3f}          | --            | --         |\n\n"
        f"**Recall improvement: {contact_recall_routed - contact_recall_baseline:+.3f}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: contact_recall_world > 0.55 | {'PASS' if c1_pass else 'FAIL'}"
        f" | {contact_recall_routed:.3f} |\n"
        f"| C2: improvement > 0.04 | {'PASS' if c2_pass else 'FAIL'}"
        f" | {contact_recall_routed - contact_recall_baseline:+.3f} |\n"
        f"| C3: action_dissoc > 0 | {'PASS' if c3_pass else 'FAIL'}"
        f" | {action_dissoc_routed:+.3f} |\n"
        f"| C4: n_contact >= 20 | {'PASS' if c4_pass else 'FAIL'}"
        f" | {n_contact_min} |\n"
        f"| C5: no fatal errors | {'PASS' if c5_pass else 'FAIL'} | -- |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Per-Seed\n\n"
        f"ROUTED:\n{per_routed_rows}\n\n"
        f"BASELINE:\n{per_baseline_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "contact_recall_routed":   float(contact_recall_routed),
        "contact_recall_baseline": float(contact_recall_baseline),
        "recall_improvement":      float(contact_recall_routed - contact_recall_baseline),
        "action_dissoc_routed":    float(action_dissoc_routed),
        "mean_routing_loss":       float(mean_route_loss),
        "n_contact_min":           float(n_contact_min),
        "lambda_route":            float(lambda_route),
        "n_seeds":                 float(len(seeds)),
        "alpha_world":             float(alpha_world),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(criteria_met),
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
        "fatal_error_count": sum(
            r["n_fatal"] for r in results_routed + results_baseline
        ),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int,   nargs="+", default=[42, 7])
    parser.add_argument("--warmup",          type=int,   default=400)
    parser.add_argument("--probe-eps",       type=int,   default=20)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument("--lambda-route",    type=float, default=0.1)
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        probe_episodes=args.probe_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        lambda_route=args.lambda_route,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

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
