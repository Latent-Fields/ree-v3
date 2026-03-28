#!/opt/local/bin/python3
"""
V3-EXQ-137 -- MECH-097 Peripersonal Space Commit Locus: PPS_LOCUS_ON vs PPS_LOCUS_ABLATED

Claims: MECH-097
Proposal: EXP-0037 / EVB-0029

MECH-097 asserts:
  "Peripersonal space geometry defines the commit locus -- the spatial boundary of
   committed action."

Specifically: the commit boundary coincides with the agent's peripersonal space (PPS)
boundary -- the dynamically-maintained near-body region where self-directed and
world-directed action consequences first interact. Actions crossing this boundary
(stepping onto a hazard cell, contacting an object) constitute the class of
z_self -> z_world causal transitions that generate moral residue.

Without a PPS representation, the harm_eval_head must learn the commit boundary
implicitly from harm outcomes alone (z_world content). With a PPS representation --
a compact proximity signal encoding how close the agent is to the nearest hazard --
the harm_eval_head has explicit spatial grounding for where self ends and world begins,
which should improve its ability to predict harm at or before the contact boundary.

V3-accessible proxy design
--------------------------
The PPS signal is operationalised as a scalar proximity-to-hazard feature derived
from the hazard gradient field already available in world_obs (the hazard_field_view
subvector). This is consistent with the MECH-097 claim: PPS is the near-body spatial
envelope, and the hazard proximity field captures exactly the gradient intensity at
the agent's current position -- it is the agent's current distance from the commit locus.

PPS_LOCUS_ON (MECH-097 architecture):
  - harm_eval_head receives: cat(z_world, pps_feature) where pps_feature is a
    1-dimensional scalar: max(hazard_field_view) over the 3x3 neighbourhood around
    the agent (from world_obs[225:250], the 5x5 hazard field view, central 3x3).
    This encodes "how close is the commit boundary" as a continuous spatial signal.
  - pps_feature is concatenated to z_world before harm_eval_head forward pass.
  - harm_eval_head input_dim = world_dim + 1.

PPS_LOCUS_ABLATED (ablation -- no spatial boundary encoding):
  - harm_eval_head receives: z_world only (same as standard pipeline).
  - harm_eval_head input_dim = world_dim.
  - No PPS proximity signal; the head must infer boundary proximity from z_world alone.

Both conditions use the same z_world encoder, same E1/E2, same training loss, same
environment. Only the harm_eval_head input differs: with or without the PPS feature.

Testable prediction:
  PPS_LOCUS_ON should produce higher harm_eval_gap than PPS_LOCUS_ABLATED because:
  - The PPS feature provides a direct graded signal of proximity to the commit
    boundary (contact hazard cell), enabling the harm_eval_head to predict harm
    at the approach stage before contact.
  - Without this signal, z_world encodes post-contact world state changes but does
    not carry a clean pre-contact proximity gradient; the harm_eval_head is limited
    to reactive harm detection.
  - MECH-097's claim that PPS geometry *defines* the commit locus predicts that
    adding the PPS encoding to the harm_eval input should yield a discriminable
    advantage in harm prediction quality.

This is a targeted isolation test: MECH-097 predicts the PPS feature is necessary.
If PPS_LOCUS_ON >= PPS_LOCUS_ABLATED in harm_eval_gap by the pre-registered threshold,
that directly supports the MECH-097 claim that peripersonal space geometry matters.

Note on evidence_quality_note (claims.yaml):
  MECH-097 depends on SD-005, MECH-091, ARC-016. This experiment isolates the spatial
  PPS component independently of heartbeat (MECH-091) and dynamic precision (ARC-016).
  It tests the core spatial grounding claim: does encoding proximity to the commit
  boundary improve harm prediction? A PASS here is partial support for MECH-097
  (spatial advantage confirmed at V3 proxy level) pending full SD-006 and ARC-016
  integration.

Pre-registered acceptance criteria:
  C1: gap_locus_on >= 0.04       (PPS_LOCUS_ON harm eval above floor -- both seeds)
  C2: per-seed delta (ON - ABLATED) >= 0.02  (PPS feature adds >=2pp -- both seeds)
  C3: gap_ablated >= 0.0         (ablation still learns something -- data quality)
  C4: n_harm_eval_min >= 20      (sufficient harm events in eval -- both seeds)
  C5: n_approach_contacts >= 10  (manipulation check: enough proximity approach events
                                  during eval to distinguish near vs far harm)

Decision scoring:
  PASS (all 5):   retain_ree -- PPS spatial grounding improves harm eval quality;
                  supports MECH-097 commit locus thesis
  C1+C4+C5, C2 fails:
                  hybridize  -- PPS mechanism works but delta below threshold;
                  increase n_approach_contacts or warmup and rerun
  C2 fails (C5 FAIL):
                  data_quality -- insufficient proximity events; increase nav_bias
  C2 fails (C5 PASS):
                  retire_ree_claim -- PPS feature does not add >=2pp at current scale
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


EXPERIMENT_TYPE = "v3_exq_137_mech097_pps_commit_locus_pair"
CLAIM_IDS = ["MECH-097"]

# Pre-registered thresholds
THRESH_C1 = 0.04   # gap_locus_on >= 0.04 (harm eval above floor, both seeds)
THRESH_C2 = 0.02   # per-seed delta (ON - ABLATED) >= 0.02 (PPS adds >=2pp)
THRESH_C3 = 0.0    # gap_ablated >= 0.0 (ablation learns something)
THRESH_C4 = 20     # n_harm_eval_min >= 20 (data quality)
THRESH_C5 = 10     # n_approach_contacts >= 10 (manipulation check)


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _hazard_approach_action(env: CausalGridWorldV2, n_actions: int) -> int:
    """Return action that moves toward the nearest hazard gradient."""
    obs_dict = env._get_observation_dict()
    world_state = obs_dict.get("world_state", None)
    if world_state is None or not env.use_proxy_fields:
        return random.randint(0, n_actions - 1)
    # world_state[225:250] = hazard_field_view (5x5 flattened)
    field_view = world_state[225:250].numpy().reshape(5, 5)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    vals = []
    for dr, dc in deltas:
        r, c = 2 + dr, 2 + dc
        if 0 <= r < 5 and 0 <= c < 5:
            vals.append(float(field_view[r, c]))
        else:
            vals.append(-1.0)
    return int(np.argmax(vals))


def _extract_pps_feature(obs_world: torch.Tensor) -> torch.Tensor:
    """
    Extract PPS scalar: max hazard field intensity in the 3x3 central neighbourhood
    of the 5x5 hazard_field_view subvector (world_obs[225:250]).

    world_obs[225:250] = 5x5 hazard field view, flattened row-major.
    Central 3x3 corresponds to indices (1,1)-(3,3) in the 5x5 grid:
      rows 1-3, cols 1-3 -> linear indices [6,7,8, 11,12,13, 16,17,18].
    This encodes how close the agent is to the commit boundary (hazard cell contact).
    Returns shape (1, 1) tensor on CPU.
    """
    if obs_world.dim() == 1:
        haz_5x5 = obs_world[225:250].cpu().numpy().reshape(5, 5)
    else:
        haz_5x5 = obs_world[0, 225:250].cpu().numpy().reshape(5, 5)
    # Central 3x3 neighbourhood
    central_vals = haz_5x5[1:4, 1:4].flatten()
    pps_scalar = float(np.max(central_vals))
    return torch.tensor([[pps_scalar]], dtype=torch.float32)


class HarmEvalHeadExtended(nn.Module):
    """
    harm_eval_head with optional PPS feature concatenated to z_world input.
    PPS_LOCUS_ON: input_dim = world_dim + 1.
    PPS_LOCUS_ABLATED: input_dim = world_dim.
    """
    def __init__(self, world_dim: int, use_pps: bool):
        super().__init__()
        self.use_pps = use_pps
        in_dim = world_dim + 1 if use_pps else world_dim
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, z_world: torch.Tensor, pps_feat: torch.Tensor = None) -> torch.Tensor:
        if self.use_pps and pps_feat is not None:
            if z_world.shape[0] != pps_feat.shape[0]:
                # Broadcast pps_feat if batched z_world
                pps_feat = pps_feat.expand(z_world.shape[0], -1)
            x = torch.cat([z_world, pps_feat], dim=-1)
        else:
            x = z_world
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


def _run_single(
    seed: int,
    pps_locus_on: bool,
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
    """Run one (seed, condition) cell; return harm eval gap and approach contact metrics."""
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "PPS_LOCUS_ON" if pps_locus_on else "PPS_LOCUS_ABLATED"

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
        reafference_action_dim=0,
    )

    agent = REEAgent(config)

    # Replace agent's harm_eval_head with our extended version (with or without PPS)
    harm_eval_head = HarmEvalHeadExtended(world_dim=world_dim, use_pps=pps_locus_on)

    # Separate optimizers: standard pipeline (E1+E2) and harm_eval
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(harm_eval_head.parameters())

    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    # Rolling buffer for harm_eval training
    buf_zw: List[torch.Tensor] = []
    buf_pps: List[torch.Tensor] = []
    buf_labels: List[float] = []
    MAX_BUF = 2000

    counts: Dict[str, int] = {
        "hazard_approach": 0,
        "env_caused_hazard": 0,
        "agent_caused_hazard": 0,
        "none": 0,
    }

    if dry_run:
        actual_warmup = min(3, warmup_episodes)
        actual_eval = min(2, eval_episodes)
    else:
        actual_warmup = warmup_episodes
        actual_eval = eval_episodes

    # --- TRAIN ---
    agent.train()
    harm_eval_head.train()

    for ep in range(actual_warmup):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            # PPS feature from obs_world (hazard field view, central 3x3 max)
            pps_feat = _extract_pps_feature(obs_world)

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

            # Collect into rolling buffer
            buf_zw.append(z_world_curr)
            buf_pps.append(pps_feat)
            buf_labels.append(1.0 if is_harm else 0.0)
            if len(buf_zw) > MAX_BUF:
                buf_zw = buf_zw[-MAX_BUF:]
                buf_pps = buf_pps[-MAX_BUF:]
                buf_labels = buf_labels[-MAX_BUF:]

            # Standard E1 + E2 losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # Online harm_eval training (mini-batch from rolling buffer)
            n_harm_buf = sum(1 for lbl in buf_labels if lbl > 0.5)
            n_safe_buf = sum(1 for lbl in buf_labels if lbl <= 0.5)
            if n_harm_buf >= 4 and n_safe_buf >= 4:
                harm_idxs = [i for i, lbl in enumerate(buf_labels) if lbl > 0.5]
                safe_idxs = [i for i, lbl in enumerate(buf_labels) if lbl <= 0.5]
                k = min(8, min(len(harm_idxs), len(safe_idxs)))
                sel_harm = random.sample(harm_idxs, k)
                sel_safe = random.sample(safe_idxs, k)
                selected = sel_harm + sel_safe

                zw_b = torch.cat([buf_zw[i] for i in selected], dim=0)
                pps_b = torch.cat([buf_pps[i] for i in selected], dim=0)
                labels_b = torch.tensor(
                    [buf_labels[i] for i in selected],
                    dtype=torch.float32,
                ).unsqueeze(1)

                pred = harm_eval_head(zw_b, pps_b)
                loss_he = F.mse_loss(pred, labels_b)
                if loss_he.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    loss_he.backward()
                    torch.nn.utils.clip_grad_norm_(harm_eval_head.parameters(), 0.5)
                    harm_eval_optimizer.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == actual_warmup - 1:
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{actual_warmup}"
                f" approach={counts['hazard_approach']}"
                f" contact={counts['env_caused_hazard'] + counts['agent_caused_hazard']}",
                flush=True,
            )

    # --- EVAL ---
    agent.eval()
    harm_eval_head.eval()

    eval_scores_harm: List[float] = []
    eval_scores_safe: List[float] = []
    n_approach_contacts = 0  # C5 manipulation check: proximity events in eval
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

            pps_feat = _extract_pps_feature(obs_world)

            action_idx = random.randint(0, n_actions - 1)
            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            is_harm = float(harm_signal) < 0
            # Approach contacts: hazard_approach OR env_caused/agent_caused (proximity)
            if ttype in ("hazard_approach", "env_caused_hazard", "agent_caused_hazard"):
                n_approach_contacts += 1

            try:
                with torch.no_grad():
                    score = float(harm_eval_head(z_world_curr, pps_feat).item())
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
        f" n_approach_contacts={n_approach_contacts}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "pps_locus_on": pps_locus_on,
        "harm_eval_gap": float(gap),
        "mean_harm_score": float(mean_harm),
        "mean_safe_score": float(mean_safe),
        "n_harm_eval": int(n_harm_eval),
        "n_safe_eval": int(n_safe_eval),
        "n_approach_contacts": int(n_approach_contacts),
        "n_fatal": int(n_fatal),
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
    nav_bias: float = 0.45,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """Discriminative pair: PPS_LOCUS_ON vs PPS_LOCUS_ABLATED."""
    results_on:      List[Dict] = []
    results_ablated: List[Dict] = []

    for seed in seeds:
        for pps_locus_on in [True, False]:
            label = "PPS_LOCUS_ON" if pps_locus_on else "PPS_LOCUS_ABLATED"
            print(
                f"\n[V3-EXQ-137] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" nav_bias={nav_bias}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                pps_locus_on=pps_locus_on,
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
            if pps_locus_on:
                results_on.append(r)
            else:
                results_ablated.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    gap_on      = _avg(results_on,      "harm_eval_gap")
    gap_ablated = _avg(results_ablated, "harm_eval_gap")
    delta_gap   = gap_on - gap_ablated

    n_harm_min        = min(r["n_harm_eval"] for r in results_on + results_ablated)
    n_approach_min    = min(r["n_approach_contacts"] for r in results_on + results_ablated)

    # Pre-registered PASS criteria
    # C1: PPS_LOCUS_ON gap >= THRESH_C1 in ALL seeds
    c1_pass = all(r["harm_eval_gap"] >= THRESH_C1 for r in results_on)
    # C2: per-seed delta (ON - ABLATED) >= THRESH_C2 in ALL seeds
    per_seed_deltas = []
    for r_on in results_on:
        matching = [r for r in results_ablated if r["seed"] == r_on["seed"]]
        if matching:
            per_seed_deltas.append(r_on["harm_eval_gap"] - matching[0]["harm_eval_gap"])
    c2_pass = len(per_seed_deltas) > 0 and all(d >= THRESH_C2 for d in per_seed_deltas)
    # C3: ABLATED gap >= 0 (ablation still learns something)
    c3_pass = gap_ablated >= THRESH_C3
    # C4: sufficient harm events in eval
    c4_pass = n_harm_min >= THRESH_C4
    # C5: enough proximity/approach contacts (manipulation check)
    c5_pass = n_approach_min >= THRESH_C5

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    # Decision scoring
    if all_pass:
        decision = "retain_ree"
    elif c1_pass and c4_pass and c5_pass:
        decision = "hybridize"
    elif not c5_pass:
        decision = "data_quality"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-137] Results:", flush=True)
    print(
        f"  gap_on={gap_on:.4f}"
        f" gap_ablated={gap_ablated:.4f}"
        f" delta_gap={delta_gap:+.4f}",
        flush=True,
    )
    print(
        f"  n_harm_min={n_harm_min}"
        f"  n_approach_min={n_approach_min}"
        f"  per_seed_deltas={[f'{d:.4f}' for d in per_seed_deltas]}"
        f"  decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failing_seeds = [r["seed"] for r in results_on if r["harm_eval_gap"] < THRESH_C1]
        failure_notes.append(
            f"C1 FAIL: PPS_LOCUS_ON gap below {THRESH_C1} in seeds {failing_seeds}"
            " -- harm_eval_head not learning harm representation even with PPS feature"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: per-seed deltas {[f'{d:.4f}' for d in per_seed_deltas]} < {THRESH_C2}"
            " -- PPS feature does not add >=2pp over ablated condition"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: gap_ablated={gap_ablated:.4f} < {THRESH_C3}"
            " -- ablation fails to learn harm representation (confound)"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: n_harm_min={n_harm_min} < {THRESH_C4}"
            " -- insufficient harm events in eval phase for reliable estimate"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: n_approach_min={n_approach_min} < {THRESH_C5}"
            " -- too few proximity approach contacts; increase nav_bias or warmup"
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    # Interpretation
    if all_pass:
        interpretation = (
            "MECH-097 SUPPORTED at V3 proxy level: PPS spatial encoding of commit"
            f" locus improves harm_eval_head quality (gap_on={gap_on:.4f} vs"
            f" gap_ablated={gap_ablated:.4f}, delta={delta_gap:+.4f} across"
            f" {len(seeds)} seeds, n_approach_min={n_approach_min})."
            " Adding the peripersonal space proximity feature to the harm_eval_head"
            " input provides spatial grounding for the commit boundary, consistent"
            " with MECH-097's claim that PPS geometry defines the commit locus."
        )
    elif c1_pass and c4_pass and c5_pass:
        interpretation = (
            f"Partial support: PPS_LOCUS_ON achieves gap={gap_on:.4f} (C1 PASS)"
            f" and sufficient approach contacts (C5 PASS), but per-seed delta="
            f"{delta_gap:+.4f} (C2 FAIL, threshold={THRESH_C2})."
            " The PPS mechanism is operational but the advantage over ablated"
            " is below the pre-registered threshold at current training scale."
        )
    elif not c5_pass:
        interpretation = (
            f"C5 FAIL: only {n_approach_min} approach/proximity contacts in eval"
            f" (threshold={THRESH_C5}). Insufficient data to distinguish near-"
            " vs far-boundary harm discrimination. Increase nav_bias or"
            " evaluation episodes and rerun."
        )
    else:
        interpretation = (
            f"MECH-097 V3 proxy NOT supported: gap_on={gap_on:.4f} (C1"
            f" {'PASS' if c1_pass else 'FAIL'}), delta={delta_gap:+.4f} (C2"
            f" {'PASS' if c2_pass else 'FAIL'}). Peripersonal space proximity"
            " encoding does not demonstrate improvement in harm eval quality at"
            " this training scale."
        )

    per_on_rows = "\n".join(
        f"  seed={r['seed']}: gap={r['harm_eval_gap']:.4f}"
        f" n_harm={r['n_harm_eval']}"
        f" n_approach_contacts={r['n_approach_contacts']}"
        for r in results_on
    )
    per_ablated_rows = "\n".join(
        f"  seed={r['seed']}: gap={r['harm_eval_gap']:.4f}"
        f" n_harm={r['n_harm_eval']}"
        f" n_approach_contacts={r['n_approach_contacts']}"
        for r in results_ablated
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-137 -- MECH-097 Peripersonal Space Commit Locus Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-097\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Conditions:** PPS_LOCUS_ON vs PPS_LOCUS_ABLATED\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n"
        f"**Env:** CausalGridWorldV2 size=6, 4 hazards, nav_bias={nav_bias}\n\n"
        f"## Experimental Design\n\n"
        f"PPS_LOCUS_ON: harm_eval_head receives cat(z_world, pps_feature)\n"
        f"  pps_feature = max hazard_field intensity in central 3x3 of 5x5 view\n"
        f"PPS_LOCUS_ABLATED: harm_eval_head receives z_world only\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: gap_locus_on >= {THRESH_C1} in all seeds  (PPS_LOCUS_ON harm eval above floor)\n"
        f"C2: per-seed delta (ON - ABLATED) >= {THRESH_C2} in all seeds  (PPS adds >=2pp)\n"
        f"C3: gap_ablated >= {THRESH_C3}  (ablation learns something; data quality)\n"
        f"C4: n_harm_eval_min >= {THRESH_C4}  (sufficient harm events in eval)\n"
        f"C5: n_approach_contacts_min >= {THRESH_C5}  (manipulation check: proximity events fired)\n\n"
        f"## Results\n\n"
        f"| Condition | gap (avg) | mean_harm | mean_safe |\n"
        f"|-----------|-----------|-----------|----------|\n"
        f"| PPS_LOCUS_ON      | {gap_on:.4f}"
        f" | {_avg(results_on,      'mean_harm_score'):.4f}"
        f" | {_avg(results_on,      'mean_safe_score'):.4f} |\n"
        f"| PPS_LOCUS_ABLATED | {gap_ablated:.4f}"
        f" | {_avg(results_ablated, 'mean_harm_score'):.4f}"
        f" | {_avg(results_ablated, 'mean_safe_score'):.4f} |\n\n"
        f"**delta_gap (ON - ABLATED): {delta_gap:+.4f}**\n"
        f"**n_approach_contacts (min across all cells): {n_approach_min}**\n\n"
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
        f"| C5: n_approach_contacts_min >= {THRESH_C5} | {'PASS' if c5_pass else 'FAIL'}"
        f" | {n_approach_min} |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed\n\n"
        f"PPS_LOCUS_ON:\n{per_on_rows}\n\n"
        f"PPS_LOCUS_ABLATED:\n{per_ablated_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "gap_locus_on":              float(gap_on),
        "gap_locus_ablated":         float(gap_ablated),
        "delta_gap":                 float(delta_gap),
        "n_harm_eval_min":           float(n_harm_min),
        "n_approach_contacts_min":   float(n_approach_min),
        "n_seeds":                   float(len(seeds)),
        "alpha_world":               float(alpha_world),
        "nav_bias":                  float(nav_bias),
        "per_seed_delta_min":        float(min(per_seed_deltas)) if per_seed_deltas else 0.0,
        "per_seed_delta_max":        float(max(per_seed_deltas)) if per_seed_deltas else 0.0,
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
        "fatal_error_count": sum(r["n_fatal"] for r in results_on + results_ablated),
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
    parser.add_argument("--nav-bias",        type=float, default=0.45)
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
        "C1_gap_locus_on":             THRESH_C1,
        "C2_per_seed_delta":           THRESH_C2,
        "C3_gap_ablated":              THRESH_C3,
        "C4_n_harm_eval_min":          THRESH_C4,
        "C5_n_approach_contacts_min":  THRESH_C5,
    }
    result["seeds"] = list(args.seeds)
    result["conditions"] = ["PPS_LOCUS_ON", "PPS_LOCUS_ABLATED"]
    result["dispatch_mode"] = "discriminative_pair"
    result["backlog_id"] = "EVB-0029"
    result["proposal_id"] = "EXP-0037"

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
