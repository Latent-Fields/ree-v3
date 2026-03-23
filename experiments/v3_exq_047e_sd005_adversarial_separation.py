#!/opt/local/bin/python3
"""
V3-EXQ-047e -- SD-005: Latent Information Separation (Adversarial Training Probe)

Claims: SD-005

Context:
  EXQ-047d FAIL (x2, identical runs) -- design failure, not a genuine negative.
  Root cause: probed frozen representations from training that never applied
  separation pressure. Actions are equally decodable from z_self and z_world
  (split_action_dissociation = 0.008) because neither encoder is penalised for
  carrying the other domain's information. Collecting two runs with the same
  seed produced byte-identical results (deterministic frozen probe).

Fix -- adversarial fine-tuning phase (Phase 2):
  After warmup, run 80 additional episodes with round-robin actions and a
  gradient-reversal probe on z_world. The probe learns to predict the dominant
  action from z_world; its gradient is REVERSED through z_world, so the encoder
  is penalised for action decodability from z_world. z_self is not penalised --
  it retains action information through motor-sensory correlation in the WF loss.

  This is standard domain-adversarial training (Ganin et al. 2015). If the
  split architecture is correct, the adversarial phase should force z_self to
  concentrate action signal and z_world to concentrate world-state / contact
  signal. The unified condition cannot achieve this -- both z_self and z_world
  are the same tensor, so adversarial suppression of action from z_world also
  suppresses it from z_self.

Design:
  Phase 1 -- warmup (300 eps, random actions): train E1 + world_forward.
  Phase 2 -- adversarial (80 eps, round-robin): GRL probe on z_world.
  Phase 3 -- probe eval (150 eps, round-robin): same probe set as EXQ-047d.
  Two conditions: SPLIT vs UNIFIED. Both receive identical Phase 1 + 2 training.

PASS criteria (ALL must hold):
  C1: action_acc_self_split > action_acc_world_split + 0.10
      (z_self carries more action signal than z_world after adversarial training)
  C2: contact_acc_world_split > contact_acc_self_split + 0.10
      (z_world carries more contact signal than z_self)
  C3: action_acc_self_split > action_acc_self_unified + 0.05
      (split z_self is more action-selective than merged z)
  C4: contact_acc_world_split > contact_acc_world_unified + 0.05
      (split z_world is more contact-selective than merged z)
  C5: No fatal errors
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_047e_sd005_adversarial_separation"
CLAIM_IDS = ["SD-005"]

WARMUP_EPS = 300
ADV_EPS = 80
EVAL_EPS = 150
STEPS_PER_EP = 200
PROBE_STEPS = 500
ADV_LAMBDA = 0.1   # gradient reversal scale
ADV_WEIGHT = 0.5   # adversarial loss weight relative to WF loss


class GradientReversal(torch.autograd.Function):
    """Gradient reversal layer (Ganin et al. 2015).

    Forward: identity. Backward: multiply gradient by -lambda.
    The encoder is trained to CONFUSE the adversarial probe by reversing
    the gradient signal when it flows back through z_world.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=0.02,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )


def _build_agent(env: CausalGridWorldV2, seed: int, alpha_world: float,
                 self_dim: int, world_dim: int, unified: bool) -> REEAgent:
    torch.manual_seed(seed)
    random.seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
    )
    config.latent.unified_latent_mode = unified
    return REEAgent(config)


def _train_warmup(
    agent: REEAgent,
    env: CausalGridWorldV2,
    world_dim: int,
    num_episodes: int,
    steps: int,
    label: str,
) -> None:
    """Phase 1: standard warmup (random actions), same as EXQ-047d."""
    agent.train()
    optimizer = optim.Adam(
        list(agent.e1.parameters()) +
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        a_prev: Optional[torch.Tensor] = None

        for _ in range(steps):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()
            z_world_curr = latent.z_world

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action
            flat_obs, _, done, _, obs_dict = env.step(action)

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if z_world_prev is not None and a_prev is not None:
                wf_buf.append((z_world_prev.cpu(), a_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 3000:
                    wf_buf = wf_buf[-3000:]

            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    optimizer.step()

            z_world_prev = z_world_curr
            a_prev = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            print(f"  [{label}|warmup] ep {ep+1}/{num_episodes}", flush=True)


def _train_adversarial(
    agent: REEAgent,
    env: CausalGridWorldV2,
    adv_probe: nn.Module,
    world_dim: int,
    num_episodes: int,
    steps: int,
    adv_lambda: float,
    adv_weight: float,
    label: str,
) -> None:
    """Phase 2: adversarial fine-tuning with gradient reversal on z_world.

    Round-robin action assignment (same as probe collection) creates a
    decodable action signal. GRL on z_world forces the encoder to push
    action information OUT of z_world while z_self retains it.

    Two separate optimizers per step:
      Step A: adv_probe update using detached z_world (probe learns to predict action).
      Step B: encoder update using GRL z_world (encoder learns to confuse probe).
    """
    agent.train()
    n_actions = env.action_dim

    enc_optimizer = optim.Adam(
        list(agent.e1.parameters()) +
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=5e-4,
    )
    adv_optimizer = optim.Adam(adv_probe.parameters(), lr=1e-3)

    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    probe_accs: List[float] = []

    for ep in range(num_episodes):
        dominant_action_idx = ep % n_actions
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        a_prev: Optional[torch.Tensor] = None

        for _ in range(steps):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            # Sense WITHOUT no_grad so z_world has grad_fn for adversarial step
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()
            z_world_curr_detached = latent.z_world.detach()

            # Round-robin action
            if random.random() < 0.8:
                action_idx = dominant_action_idx
            else:
                action_idx = random.randint(0, n_actions - 1)
            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action
            flat_obs, _, done, _, obs_dict = env.step(action)

            dominant_label = torch.tensor(
                [dominant_action_idx], dtype=torch.long, device=agent.device
            )

            # Step A: train adv_probe on detached z_world
            logits_probe = adv_probe(z_world_curr_detached)
            adv_probe_loss = F.cross_entropy(logits_probe, dominant_label)
            adv_optimizer.zero_grad()
            adv_probe_loss.backward()
            adv_optimizer.step()
            probe_accs.append(
                float((logits_probe.detach().argmax(dim=1) == dominant_label).float().mean())
            )

            # Step B: update encoder adversarially (reversed gradient through z_world)
            z_world_rev = GradientReversal.apply(latent.z_world, adv_lambda)
            logits_enc = adv_probe(z_world_rev)
            adv_enc_loss = F.cross_entropy(logits_enc, dominant_label)

            # WF loss for encoder stability (uses detached buffer, updates E2 transitions)
            total_enc_loss = adv_weight * adv_enc_loss
            if z_world_prev is not None and a_prev is not None:
                wf_buf.append((z_world_prev, a_prev, z_world_curr_detached.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]
            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    total_enc_loss = total_enc_loss + wf_loss

            enc_optimizer.zero_grad()
            total_enc_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(agent.e1.parameters()) +
                list(agent.e2.world_transition.parameters()) +
                list(agent.e2.world_action_encoder.parameters()),
                1.0,
            )
            enc_optimizer.step()

            z_world_prev = z_world_curr_detached.cpu()
            a_prev = action.detach().cpu()
            if done:
                break

        if (ep + 1) % 20 == 0 or ep == num_episodes - 1:
            recent_probe_acc = _mean_safe(probe_accs[-200:])
            print(
                f"  [{label}|adv] ep {ep+1}/{num_episodes}"
                f"  probe_acc(recent)={recent_probe_acc:.3f}",
                flush=True,
            )


def _collect_probe_data(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps: int,
    label: str,
) -> Dict:
    """Collect probe data with round-robin dominant actions (same as EXQ-047d)."""
    agent.eval()
    n_actions = env.action_dim
    z_selfs, z_worlds, dominant_actions, contact_flags = [], [], [], []

    for ep in range(num_episodes):
        dominant_action_idx = ep % n_actions
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            if random.random() < 0.8:
                action_idx = dominant_action_idx
            else:
                action_idx = random.randint(0, n_actions - 1)
            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action
            flat_obs, _, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            is_contact = 1 if ttype in ("env_caused_hazard", "agent_caused_hazard") else 0

            z_selfs.append(latent.z_self.detach().cpu())
            z_worlds.append(latent.z_world.detach().cpu())
            dominant_actions.append(dominant_action_idx)
            contact_flags.append(is_contact)
            if done:
                break

    n = len(z_selfs)
    n_contacts = sum(contact_flags)
    print(
        f"  [{label}|collect] {n} samples  contacts={n_contacts}"
        f" ({100*n_contacts/max(1,n):.1f}%)",
        flush=True,
    )
    return {
        "z_self":          torch.cat(z_selfs, dim=0),
        "z_world":         torch.cat(z_worlds, dim=0),
        "dominant_action": torch.tensor(dominant_actions, dtype=torch.long),
        "contact":         torch.tensor(contact_flags, dtype=torch.long),
        "n_contacts":      n_contacts,
        "n_total":         n,
    }


def _fit_probe(features: torch.Tensor, labels: torch.Tensor,
               n_classes: int, n_steps: int, lr: float = 5e-3,
               balanced: bool = False, label: str = "") -> float:
    """Fit linear probe and return test accuracy. Optionally balance classes."""
    if balanced:
        pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            print(f"    [{label}] SKIP: degenerate classes", flush=True)
            return 0.0
        n_min = min(len(pos_idx), len(neg_idx))
        idx = torch.cat([
            pos_idx[torch.randperm(len(pos_idx))[:n_min]],
            neg_idx[torch.randperm(len(neg_idx))[:n_min]],
        ])
        idx = idx[torch.randperm(len(idx))]
    else:
        idx = torch.randperm(len(labels))

    n_train = int(len(idx) * 0.8)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    if len(test_idx) < 4:
        print(f"    [{label}] SKIP: too few test samples", flush=True)
        return 0.0

    x_tr, y_tr = features[train_idx], labels[train_idx]
    x_te, y_te = features[test_idx], labels[test_idx]

    probe = nn.Linear(features.shape[1], n_classes)
    opt = optim.Adam(probe.parameters(), lr=lr)
    probe.train()
    for _ in range(n_steps):
        opt.zero_grad()
        F.cross_entropy(probe(x_tr), y_tr).backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        acc = float((probe(x_te).argmax(1) == y_te).float().mean())
    print(f"    [{label}] acc={acc:.3f}  n_test={len(y_te)}", flush=True)
    return acc


def _run_probes(data: Dict, n_actions: int, label: str, probe_steps: int) -> Dict[str, float]:
    print(f"\n  [{label}] Fitting probes...", flush=True)
    return {
        "action_acc_self":   _fit_probe(
            data["z_self"], data["dominant_action"], n_actions, probe_steps,
            label=f"{label}|action<-z_self"),
        "action_acc_world":  _fit_probe(
            data["z_world"], data["dominant_action"], n_actions, probe_steps,
            label=f"{label}|action<-z_world"),
        "contact_acc_world": _fit_probe(
            data["z_world"], data["contact"], 2, probe_steps,
            balanced=True, label=f"{label}|contact<-z_world"),
        "contact_acc_self":  _fit_probe(
            data["z_self"], data["contact"], 2, probe_steps,
            balanced=True, label=f"{label}|contact<-z_self"),
    }


def run(
    seed: int = 0,
    warmup_episodes: int = WARMUP_EPS,
    adv_episodes: int = ADV_EPS,
    eval_episodes: int = EVAL_EPS,
    steps_per_episode: int = STEPS_PER_EP,
    probe_steps: int = PROBE_STEPS,
    alpha_world: float = 0.9,
    adv_lambda: float = ADV_LAMBDA,
    adv_weight: float = ADV_WEIGHT,
    self_dim: int = 32,
    world_dim: int = 32,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    print(
        f"[V3-EXQ-047e] SD-005: Latent Information Separation (Adversarial Training)\n"
        f"  Phase 1: {warmup_episodes} warmup eps (random actions)\n"
        f"  Phase 2: {adv_episodes} adversarial eps (GRL on z_world, round-robin)\n"
        f"  Phase 3: {eval_episodes} eval eps (probe collection + fitting)\n"
        f"  adv_lambda={adv_lambda}  adv_weight={adv_weight}\n"
        f"  seed={seed}  alpha_world={alpha_world}",
        flush=True,
    )

    probe_results: Dict[str, Dict] = {}
    n_contacts_by_condition: Dict[str, int] = {}

    for label, unified in [("split", False), ("unified", True)]:
        print(f"\n{'='*60}", flush=True)
        print(f"[V3-EXQ-047e] CONDITION: {label.upper()} (unified_latent={unified})", flush=True)
        print('='*60, flush=True)

        env = _make_env(seed)
        agent = _build_agent(env, seed, alpha_world, self_dim, world_dim, unified)
        adv_probe = nn.Linear(world_dim, env.action_dim)

        # Phase 1: warmup
        print(f"\n[Phase 1] Warmup ({warmup_episodes} eps)...", flush=True)
        _train_warmup(agent, env, world_dim, warmup_episodes, steps_per_episode, label)

        # Phase 2: adversarial fine-tuning
        print(f"\n[Phase 2] Adversarial fine-tuning ({adv_episodes} eps)...", flush=True)
        _train_adversarial(
            agent, env, adv_probe, world_dim, adv_episodes,
            steps_per_episode, adv_lambda, adv_weight, label,
        )

        # Phase 3: probe collection + eval
        print(f"\n[Phase 3] Probe collection ({eval_episodes} eps)...", flush=True)
        data = _collect_probe_data(agent, env, eval_episodes, steps_per_episode, label)
        n_contacts_by_condition[label] = data["n_contacts"]
        probe_results[label] = _run_probes(data, env.action_dim, label, probe_steps)

    r_s = probe_results["split"]
    r_u = probe_results["unified"]
    n_contacts_split = n_contacts_by_condition["split"]

    # -- PASS / FAIL ---------------------------------------------------------
    c1_pass = r_s["action_acc_self"]   > r_s["action_acc_world"]  + 0.10
    c2_pass = r_s["contact_acc_world"] > r_s["contact_acc_self"]  + 0.10
    c3_pass = r_s["action_acc_self"]   > r_u["action_acc_self"]   + 0.05
    c4_pass = r_s["contact_acc_world"] > r_u["contact_acc_world"] + 0.05
    c5_pass = True

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: action_acc_self={r_s['action_acc_self']:.3f}"
            f" not > action_acc_world={r_s['action_acc_world']:.3f} + 0.10"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: contact_acc_world={r_s['contact_acc_world']:.3f}"
            f" not > contact_acc_self={r_s['contact_acc_self']:.3f} + 0.10"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: action_acc_self_split={r_s['action_acc_self']:.3f}"
            f" not > action_acc_self_unified={r_u['action_acc_self']:.3f} + 0.05"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: contact_acc_world_split={r_s['contact_acc_world']:.3f}"
            f" not > contact_acc_world_unified={r_u['contact_acc_world']:.3f} + 0.05"
        )

    print(f"\nV3-EXQ-047e verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "split_action_acc_self":    float(r_s["action_acc_self"]),
        "split_action_acc_world":   float(r_s["action_acc_world"]),
        "split_contact_acc_world":  float(r_s["contact_acc_world"]),
        "split_contact_acc_self":   float(r_s["contact_acc_self"]),
        "split_action_dissociation":  float(r_s["action_acc_self"] - r_s["action_acc_world"]),
        "split_contact_dissociation": float(r_s["contact_acc_world"] - r_s["contact_acc_self"]),
        "unified_action_acc_self":    float(r_u["action_acc_self"]),
        "unified_action_acc_world":   float(r_u["action_acc_world"]),
        "unified_contact_acc_world":  float(r_u["contact_acc_world"]),
        "unified_contact_acc_self":   float(r_u["contact_acc_self"]),
        "action_self_improvement":    float(r_s["action_acc_self"] - r_u["action_acc_self"]),
        "contact_world_improvement":  float(r_s["contact_acc_world"] - r_u["contact_acc_world"]),
        "n_contacts_split":    float(n_contacts_split),
        "adv_lambda":          float(adv_lambda),
        "adv_weight":          float(adv_weight),
        "crit1_pass":  1.0 if c1_pass else 0.0,
        "crit2_pass":  1.0 if c2_pass else 0.0,
        "crit3_pass":  1.0 if c3_pass else 0.0,
        "crit4_pass":  1.0 if c4_pass else 0.0,
        "crit5_pass":  1.0,
        "criteria_met":       float(criteria_met),
        "fatal_error_count":  0.0,
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-047e -- SD-005: Latent Information Separation (Adversarial Training)

**Status:** {status}
**Claim:** SD-005 -- z_gamma split into z_self (motor-sensory) and z_world (environmental)
**Design:** Phase 1 warmup + Phase 2 adversarial GRL fine-tuning + Phase 3 linear probe eval
**adv_lambda:** {adv_lambda}  |  **adv_weight:** {adv_weight}  |  **Seed:** {seed}
**Warmup:** {warmup_episodes} eps  |  **Adv:** {adv_episodes} eps  |  **Eval:** {eval_episodes} eps

## EXQ-047d Failure Analysis

EXQ-047d probed frozen representations with no separation training objective.
Both z_self and z_world carried equal action information (dissociation=0.008).
Two "runs" were byte-identical (same seed, deterministic frozen probe).
Root cause: without adversarial pressure, the encoder has no incentive to
concentrate action information in z_self and expel it from z_world.

EXQ-047e adds gradient reversal on z_world during adversarial Phase 2:
the probe learns to predict action from z_world; the reversed gradient forces
the encoder to push action information out of z_world. In the unified condition,
z_self = z_world, so adversarial suppression affects both -- the split is required.

## Probe Results

| Probe | Split | Unified | delta |
|---|---|---|---|
| Action accuracy <- z_self | {r_s['action_acc_self']:.3f} | {r_u['action_acc_self']:.3f} | {r_s['action_acc_self'] - r_u['action_acc_self']:+.3f} |
| Action accuracy <- z_world | {r_s['action_acc_world']:.3f} | {r_u['action_acc_world']:.3f} | {r_s['action_acc_world'] - r_u['action_acc_world']:+.3f} |
| Contact accuracy <- z_world | {r_s['contact_acc_world']:.3f} | {r_u['contact_acc_world']:.3f} | {r_s['contact_acc_world'] - r_u['contact_acc_world']:+.3f} |
| Contact accuracy <- z_self | {r_s['contact_acc_self']:.3f} | {r_u['contact_acc_self']:.3f} | {r_s['contact_acc_self'] - r_u['contact_acc_self']:+.3f} |

Action dissociation (split):   {r_s['action_acc_self'] - r_s['action_acc_world']:+.3f}  (target: > 0.10)
Contact dissociation (split):  {r_s['contact_acc_world'] - r_s['contact_acc_self']:+.3f}  (target: > 0.10)
Contacts collected (split): {n_contacts_split}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: action_self > action_world + 0.10 (split z_self action-selective) | {"PASS" if c1_pass else "FAIL"} | {r_s['action_acc_self']:.3f} vs {r_s['action_acc_world']:.3f} |
| C2: contact_world > contact_self + 0.10 (split z_world contact-selective) | {"PASS" if c2_pass else "FAIL"} | {r_s['contact_acc_world']:.3f} vs {r_s['contact_acc_self']:.3f} |
| C3: action_self_split > action_self_unified + 0.05 | {"PASS" if c3_pass else "FAIL"} | {r_s['action_acc_self']:.3f} vs {r_u['action_acc_self']:.3f} |
| C4: contact_world_split > contact_world_unified + 0.05 | {"PASS" if c4_pass else "FAIL"} | {r_s['contact_acc_world']:.3f} vs {r_u['contact_acc_world']:.3f} |
| C5: No fatal errors | PASS | 0 |

Criteria met: {criteria_met}/5 -> **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0.0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--warmup",      type=int,   default=WARMUP_EPS)
    parser.add_argument("--adv-eps",     type=int,   default=ADV_EPS)
    parser.add_argument("--eval-eps",    type=int,   default=EVAL_EPS)
    parser.add_argument("--steps",       type=int,   default=STEPS_PER_EP)
    parser.add_argument("--probe-steps", type=int,   default=PROBE_STEPS)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--adv-lambda",  type=float, default=ADV_LAMBDA)
    parser.add_argument("--adv-weight",  type=float, default=ADV_WEIGHT)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        adv_episodes=args.adv_eps,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        probe_steps=args.probe_steps,
        alpha_world=args.alpha_world,
        adv_lambda=args.adv_lambda,
        adv_weight=args.adv_weight,
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
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}", flush=True)
