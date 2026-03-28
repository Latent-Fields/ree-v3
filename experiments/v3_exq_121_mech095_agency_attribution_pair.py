#!/opt/local/bin/python3
"""
V3-EXQ-121 -- MECH-095: Agency Attribution Discriminative Pair (EXP-0019 / EVB-0015)

Claim: MECH-095
Proposal: EXP-0019 / EVB-0015
Dispatch mode: discriminative_pair
Min shared seeds: 2

MECH-095 asserts: "Temporoparietal junction (TPJ) acts as agency-detection comparator
distinguishing self-caused from other-caused change."

Specifically:
  At the z_self/z_world interface, an explicit agency-detection comparator (TPJ-equivalent)
  compares efference-copy-predicted z_self change with observed z_self change.
  - When predicted and observed z_self changes match: state change is self-caused (no
    residue contribution).
  - When they diverge: cause is attributed to z_world (potential residue).
  This is the mechanism for SD-003 counterfactual attribution: self_delta and world_delta
  are only cleanly separable if this comparator exists.

This experiment implements a discriminative pair:

  AGENCY_ATTRIBUTION_ON:
    AgencyComparator module computes:
      predicted_z_self_delta = E2.self_forward(z_self_prev, action)
      observed_z_self_delta  = z_self_current - z_self_prev
      mismatch               = ||observed - predicted||
    When mismatch is HIGH: cause attributed to world (world_delta signal strong).
    When mismatch is LOW:  cause attributed to self (self_delta signal; world_delta gated).
    Attribution probe trained on mismatch vector to classify agent_caused vs env_caused
    hazard contacts. Gradient flows through AgencyComparator only -- clean z_self/z_world
    encoder not contaminated.

  AGENCY_ATTRIBUTION_ABLATED:
    No AgencyComparator. Attribution probe trained directly on concatenated
    [z_self_delta, z_world] without mismatch computation. This is the two-component
    fusion baseline: same information available, no structured comparison.

Both conditions:
  - 2 matched seeds: [42, 123]
  - 400 warmup + 50 eval episodes x 200 steps
  - CausalGridWorldV2: size=8, 4 hazards, 3 resources, env_drift_prob=0.3
  - SD-005 split latent: z_self != z_world (unified_latent_mode=False)
  - SD-008: alpha_world=0.9
  - SD-007 reafference disabled (reafference_action_dim=0) -- isolate MECH-095 mechanism
  - Attribution labels from CausalGridWorldV2 transition_type:
      "agent_caused_hazard"  -> label 1.0 (agent walked into own contamination)
      "env_caused_hazard"    -> label 0.0 (agent walked into env-placed hazard)

CLAIM_IDS RATIONALE:
  The only claim directly tested is MECH-095: does a structured efference-copy
  mismatch comparator improve agency attribution over an unstructured fusion baseline?
  MECH-095 directly predicts YES (comparator mechanism is necessary).
  MECH-099 (lateral stream) is NOT tagged -- that tests three-stream routing, not
  the comparator per se.

PRE-REGISTERED THRESHOLDS (hardcoded, not inferred post-hoc):

  C1 (relative attribution advantage):
    auc_ON >= auc_ABLATED + THRESH_C1_AUC_DELTA
    The comparator must provide a meaningful AUC advantage over fusion baseline.
    Threshold: THRESH_C1_AUC_DELTA = 0.05.

  C2 (absolute attribution learning):
    auc_ON >= THRESH_C2_AUC_ABS
    The ON condition must achieve reliable above-chance attribution.
    Threshold: THRESH_C2_AUC_ABS = 0.60.

  C3 (consistency across seeds):
    auc_ON > auc_ABLATED for BOTH seeds independently.
    Direction must replicate.

  C4 (data quality -- sufficient attribution events):
    n_contact_steps_eval >= THRESH_C4_MIN_CONTACTS per seed per condition.
    Threshold: THRESH_C4_MIN_CONTACTS = 20.

  C5 (mismatch signal is informative):
    mean_mismatch_agent_caused >= mean_mismatch_env_caused * THRESH_C5_MISMATCH_RATIO
    The AgencyComparator mismatch must be larger for agent-caused contacts
    than env-caused contacts (structural: moving toward your contamination
    = predictable self-motion = low mismatch, NOT high. Env hazard approach
    = unexpected world change = high mismatch). So the comparator output
    is a REVERSED signal: high mismatch -> env_caused; low mismatch -> agent_caused.
    NOTE: AUC is computed with mismatch NEGATED for agent_caused scoring.
    Threshold: THRESH_C5_MISMATCH_RATIO = 1.05  (env mismatch >= 5% above agent mismatch).

PASS criteria:
  C1 AND C2 AND C3 AND C4 -> PASS -> supports MECH-095
  C1 AND C3, NOT C2       -> mixed (relative advantage without absolute learning)
  C2 AND C4, auc_ON < auc_ABLATED -> FAIL -> comparator hypothesis refuted
  NOT C4                  -> inconclusive (data quality failure)

Decision mapping:
  PASS        -> retain_ree
  C1+C3+C4    -> hybridize
  auc_ON < auc_ABLATED AND C4 -> retire_ree_claim (comparator fails)
  NOT C4      -> inconclusive
"""

import math
import random
import sys
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


EXPERIMENT_TYPE = "v3_exq_121_mech095_agency_attribution_pair"
CLAIM_IDS = ["MECH-095"]

# Hazard contact transition types that carry attribution labels
ATTRIBUTION_TYPES = {"agent_caused_hazard", "env_caused_hazard"}

# Pre-registered thresholds
THRESH_C1_AUC_DELTA     = 0.05   # ON must beat ABLATED by >= 5pp AUC
THRESH_C2_AUC_ABS       = 0.60   # ON must reach >= 0.60 absolute AUC
THRESH_C4_MIN_CONTACTS  = 20     # >= 20 contact steps in eval per (seed, condition)
THRESH_C5_MISMATCH_RATIO = 1.05  # env mismatch >= 5% above agent mismatch (structural check)


# ---------------------------------------------------------------------------
# AgencyComparator: TPJ-equivalent efference-copy mismatch module
# ---------------------------------------------------------------------------

class AgencyComparator(nn.Module):
    """
    TPJ-equivalent agency-detection comparator (MECH-095).

    Computes predicted z_self delta from efference copy (action embedding),
    then measures mismatch with observed z_self delta.

    Architecture:
      1. E2_self_predictor: Linear(self_dim + action_dim -> self_dim)
         Predicts z_self_{t+1} from z_self_t and action_t (efference copy).
      2. Mismatch vector: observed_z_self_delta - predicted_z_self_delta
      3. Mismatch magnitude and direction encode agency signal.

    The mismatch has the following structural property:
      - Agent-caused events: agent moved toward own contamination -> predictable
        self-motion -> LOW mismatch (self-caused, no world surprise).
      - Env-caused events: hazard drifted into agent's cell -> unexpected world
        change -> HIGH mismatch (cause is external, not self-motion).
    Therefore: high mismatch predicts env_caused; low mismatch predicts agent_caused.
    Attribution probe must learn this mapping (negated mismatch for agent_caused).

    Gradient:
      The comparator trains ONLY via the attribution probe loss.
      Agent (z_self/z_world encoder) parameters are NOT in the comparator optimizer.
      This keeps the z_self/z_world encoding clean -- the comparator is a read-only
      consumer of latent states (MECH-095 architectural requirement: comparator sits
      at the interface, not inside the encoder).
    """

    def __init__(self, self_dim: int, action_dim: int) -> None:
        super().__init__()
        self.self_dim = self_dim
        self.action_dim = action_dim
        # E2_self_predictor: predict delta_z_self from z_self + action
        self.e2_self = nn.Sequential(
            nn.Linear(self_dim + action_dim, self_dim * 2),
            nn.ReLU(),
            nn.Linear(self_dim * 2, self_dim),
        )
        # Mismatch encoder: [mismatch_vector(self_dim) + mismatch_magnitude(1)] -> agency_dim
        self.mismatch_encoder = nn.Sequential(
            nn.Linear(self_dim + 1, self_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        z_self_prev: torch.Tensor,   # [1, self_dim]
        z_self_curr: torch.Tensor,   # [1, self_dim]
        action_onehot: torch.Tensor, # [1, action_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          agency_features: [1, self_dim] -- mismatch-encoded agency signal
          mismatch_magnitude: scalar -- for diagnostics
        """
        if z_self_prev.dim() == 1:
            z_self_prev = z_self_prev.unsqueeze(0)
        if z_self_curr.dim() == 1:
            z_self_curr = z_self_curr.unsqueeze(0)
        if action_onehot.dim() == 1:
            action_onehot = action_onehot.unsqueeze(0)

        # Predict z_self delta from efference copy (action)
        inp = torch.cat([z_self_prev, action_onehot], dim=-1)  # [1, self_dim + action_dim]
        predicted_delta = self.e2_self(inp)                    # [1, self_dim]

        # Observed delta
        observed_delta = z_self_curr - z_self_prev             # [1, self_dim]

        # Mismatch: unexplained delta (what the efference copy cannot account for)
        mismatch_vec = observed_delta - predicted_delta        # [1, self_dim]
        mismatch_mag = mismatch_vec.norm(dim=-1, keepdim=True) # [1, 1]

        # Encode mismatch as agency signal
        enc_inp = torch.cat([mismatch_vec, mismatch_mag], dim=-1)  # [1, self_dim+1]
        agency_features = self.mismatch_encoder(enc_inp)           # [1, self_dim]

        return agency_features, mismatch_mag.squeeze().item()


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    return t.unsqueeze(0) if t.dim() == 1 else t


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _compute_auc(scores: List[float], labels: List[float]) -> float:
    """Rank-based (Wilcoxon-Mann-Whitney) AUC. Returns 0.5 on degenerate input."""
    if not scores:
        return 0.5
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    auc = 0.0
    running_neg = 0.0
    for _, label in pairs:
        if label == 0.0:
            running_neg += 1.0
        else:
            auc += running_neg
    return auc / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Single cell runner
# ---------------------------------------------------------------------------

def _run_single(
    seed: int,
    agency_on: bool,
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
    env_drift_prob: float,
    env_drift_interval: int,
    dry_run: bool = False,
) -> Dict:
    """Run one (seed, condition) cell. Returns per-cell metrics."""
    torch.manual_seed(seed)
    random.seed(seed)

    cond = "AGENCY_ATTRIBUTION_ON" if agency_on else "AGENCY_ATTRIBUTION_ABLATED"

    env = CausalGridWorldV2(
        seed=seed,
        size=8,
        num_hazards=4,
        num_resources=3,
        hazard_harm=harm_scale,
        env_drift_interval=env_drift_interval,
        env_drift_prob=env_drift_prob,
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
        reafference_action_dim=0,  # SD-007 disabled -- isolate MECH-095
    )
    # SD-005: split latent mode (z_self != z_world)
    config.latent.unified_latent_mode = False

    agent = REEAgent(config)
    device = agent.device

    if agency_on:
        # AGENCY_ON: AgencyComparator + linear probe on comparator output
        comparator = AgencyComparator(
            self_dim=self_dim,
            action_dim=n_actions,
        ).to(device)
        attribution_probe = nn.Linear(self_dim, 1).to(device)
        # Agent optimizer: E1+E2 only (comparator has its own optimizer)
        agent_opt = optim.Adam(agent.parameters(), lr=lr)
        # Comparator optimizer: comparator + probe only (does NOT touch agent encoder)
        comp_opt = optim.Adam(
            list(comparator.parameters()) + list(attribution_probe.parameters()),
            lr=lr,
        )
    else:
        # AGENCY_ABLATED: direct fusion of z_self_delta + z_world without comparator
        # Probe input: [z_self_delta (self_dim) + z_world (world_dim)] = self_dim + world_dim
        comparator = None
        fusion_dim = self_dim + world_dim
        attribution_probe = nn.Linear(fusion_dim, 1).to(device)
        # Probe optimizer: probe only (agent encoder still has own opt for E1+E2)
        agent_opt = optim.Adam(agent.parameters(), lr=lr)
        comp_opt = optim.Adam(list(attribution_probe.parameters()), lr=lr)

    train_counts: Dict[str, int] = {
        "agent_caused": 0,
        "env_caused": 0,
        "total_steps": 0,
    }
    mismatch_agent: List[float] = []
    mismatch_env:   List[float] = []

    actual_warmup = min(3, warmup_episodes) if dry_run else warmup_episodes
    actual_eval   = min(2, eval_episodes)   if dry_run else eval_episodes

    # ---- TRAINING PHASE ----
    agent.train()
    attribution_probe.train()
    if comparator is not None:
        comparator.train()

    for ep in range(actual_warmup):
        _, obs_dict = env.reset()
        agent.reset()
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_self_curr  = _ensure_2d(latent.z_self.detach())
            z_world_curr = _ensure_2d(latent.z_world.detach())

            action_idx = random.randint(0, n_actions - 1)
            action_oh  = _action_to_onehot(action_idx, n_actions, device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")

            if ttype == "agent_caused_hazard":
                train_counts["agent_caused"] += 1
            elif ttype == "env_caused_hazard":
                train_counts["env_caused"] += 1
            train_counts["total_steps"] += 1

            # E1 + E2 loss (both conditions)
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_agent_loss = e1_loss + e2_loss
            if total_agent_loss.requires_grad:
                agent_opt.zero_grad()
                total_agent_loss.backward()
                nn.utils.clip_grad_norm_(list(agent.parameters()), 1.0)
                agent_opt.step()

            # Attribution loss: sparse -- only on hazard contact steps
            if ttype in ATTRIBUTION_TYPES and z_self_prev is not None and action_prev is not None:
                attr_label_val = 1.0 if ttype == "agent_caused_hazard" else 0.0
                attr_label = torch.tensor([[attr_label_val]], device=device)

                if agency_on:
                    # Use comparator mismatch as agency signal
                    agency_feats, mismatch_val = comparator(
                        z_self_prev, z_self_curr, action_prev
                    )
                    attr_pred = attribution_probe(agency_feats)
                    if attr_label_val == 1.0:
                        mismatch_agent.append(mismatch_val)
                    else:
                        mismatch_env.append(mismatch_val)
                else:
                    # Direct fusion: z_self_delta + z_world (no comparator)
                    z_self_delta = (z_self_curr - z_self_prev).detach()
                    fusion = torch.cat([z_self_delta, z_world_curr], dim=-1)
                    attr_pred = attribution_probe(fusion)

                attr_loss = F.binary_cross_entropy_with_logits(attr_pred, attr_label)
                comp_opt.zero_grad()
                attr_loss.backward()
                if agency_on:
                    nn.utils.clip_grad_norm_(
                        list(comparator.parameters()) + list(attribution_probe.parameters()),
                        1.0,
                    )
                else:
                    nn.utils.clip_grad_norm_(list(attribution_probe.parameters()), 1.0)
                comp_opt.step()

            z_self_prev = z_self_curr.detach()
            action_prev = action_oh.detach()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == actual_warmup - 1:
            print(
                f"  [train] seed={seed} cond={cond}"
                f" ep {ep + 1}/{actual_warmup}"
                f" agent_caused={train_counts['agent_caused']}"
                f" env_caused={train_counts['env_caused']}"
                f" total_steps={train_counts['total_steps']}",
                flush=True,
            )

    n_min_train = min(train_counts["agent_caused"], train_counts["env_caused"])
    if n_min_train < 30:
        print(
            f"  [WARN] seed={seed} cond={cond}"
            f" sparse attribution events: agent_caused={train_counts['agent_caused']}"
            f" env_caused={train_counts['env_caused']}"
            f" (min_class={n_min_train} < 30) -- AUC may be unreliable",
            flush=True,
        )

    # ---- EVAL PHASE ----
    agent.eval()
    attribution_probe.eval()
    if comparator is not None:
        comparator.eval()

    attr_scores:  List[float] = []   # probe scores on hazard-contact steps
    attr_labels:  List[float] = []   # 1.0=agent_caused, 0.0=env_caused
    harm_contacts = 0
    total_eval_steps = 0
    eval_mismatch_agent: List[float] = []
    eval_mismatch_env:   List[float] = []

    for _ in range(actual_eval):
        _, obs_dict = env.reset()
        agent.reset()
        z_self_prev = None
        action_prev = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()

                z_self_curr  = _ensure_2d(latent.z_self.detach())
                z_world_curr = _ensure_2d(latent.z_world.detach())

                if agency_on and z_self_prev is not None and action_prev is not None:
                    agency_feats, mismatch_val = comparator(
                        z_self_prev, z_self_curr, action_prev
                    )
                    raw_score = attribution_probe(agency_feats).item()
                elif (not agency_on) and z_self_prev is not None:
                    z_self_delta = (z_self_curr - z_self_prev)
                    fusion = torch.cat([z_self_delta, z_world_curr], dim=-1)
                    raw_score = attribution_probe(fusion).item()
                    mismatch_val = 0.0
                else:
                    raw_score = 0.0
                    mismatch_val = 0.0

            action_idx = random.randint(0, n_actions - 1)
            action_oh  = _action_to_onehot(action_idx, n_actions, device)
            agent._last_action = action_oh
            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")
            total_eval_steps += 1

            if ttype in ATTRIBUTION_TYPES and z_self_prev is not None:
                lbl = 1.0 if ttype == "agent_caused_hazard" else 0.0
                attr_scores.append(raw_score)
                attr_labels.append(lbl)
                if lbl == 1.0:
                    eval_mismatch_agent.append(mismatch_val)
                else:
                    eval_mismatch_env.append(mismatch_val)

            if float(harm_signal) < 0:
                harm_contacts += 1

            z_self_prev = z_self_curr.detach()
            action_prev = action_oh.detach()

            if done:
                break

    # AUC: for agency_on, high mismatch -> env_caused (0.0).
    # Low mismatch -> agent_caused (1.0). Negate scores so AUC is maximized
    # when low-mismatch = high-score = agent_caused = 1.0.
    if agency_on:
        auc_scores = [-s for s in attr_scores]
    else:
        auc_scores = attr_scores

    attribution_auc = _compute_auc(auc_scores, attr_labels)
    harm_rate = harm_contacts / max(1, total_eval_steps)
    n_contact = len(attr_scores)
    n_agent_caused = sum(1 for l in attr_labels if l > 0.5)
    n_env_caused   = sum(1 for l in attr_labels if l < 0.5)

    mean_mm_agent = (
        float(sum(eval_mismatch_agent) / len(eval_mismatch_agent))
        if eval_mismatch_agent else 0.0
    )
    mean_mm_env = (
        float(sum(eval_mismatch_env) / len(eval_mismatch_env))
        if eval_mismatch_env else 0.0
    )

    print(
        f"  [eval] seed={seed} cond={cond}"
        f" attr_auc={attribution_auc:.4f}"
        f" harm_rate={harm_rate:.4f}"
        f" contacts={n_contact} (a={n_agent_caused} e={n_env_caused})"
        f" mm_agent={mean_mm_agent:.4f} mm_env={mean_mm_env:.4f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond,
        "agency_on": agency_on,
        "attribution_auc": float(attribution_auc),
        "harm_rate": float(harm_rate),
        "harm_contacts": int(harm_contacts),
        "total_eval_steps": int(total_eval_steps),
        "n_contact_steps_eval": int(n_contact),
        "n_agent_caused_eval": int(n_agent_caused),
        "n_env_caused_eval": int(n_env_caused),
        "mean_mismatch_agent_eval": float(mean_mm_agent),
        "mean_mismatch_env_eval": float(mean_mm_env),
        "n_agent_caused_train": int(train_counts["agent_caused"]),
        "n_env_caused_train": int(train_counts["env_caused"]),
        "total_train_steps": int(train_counts["total_steps"]),
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

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
    proximity_harm_scale: float = 0.01,
    env_drift_prob: float = 0.3,
    env_drift_interval: int = 3,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """Discriminative pair: AGENCY_ATTRIBUTION_ON vs AGENCY_ATTRIBUTION_ABLATED."""
    results_on:  List[Dict] = []
    results_off: List[Dict] = []

    for seed in seeds:
        for agency_on in [True, False]:
            cond = "AGENCY_ATTRIBUTION_ON" if agency_on else "AGENCY_ATTRIBUTION_ABLATED"
            print(
                f"\n[V3-EXQ-121] {cond} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" steps={steps_per_episode}"
                f" alpha_world={alpha_world}"
                f" drift_prob={env_drift_prob}"
                f" {'DRY_RUN' if dry_run else ''}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                agency_on=agency_on,
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
                env_drift_prob=env_drift_prob,
                env_drift_interval=env_drift_interval,
                dry_run=dry_run,
            )
            if agency_on:
                results_on.append(r)
            else:
                results_off.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    auc_on   = _avg(results_on,  "attribution_auc")
    auc_off  = _avg(results_off, "attribution_auc")
    auc_delta = auc_on - auc_off

    # Per-seed C1 check (consistency)
    per_seed_c1 = [
        ron["attribution_auc"] > roff["attribution_auc"]
        for ron, roff in zip(results_on, results_off)
    ]
    c3_pass = all(per_seed_c1)

    # C4: data quality -- min contact steps across all cells
    min_contacts = min(
        r["n_contact_steps_eval"]
        for r in results_on + results_off
    )

    # C5: mismatch structural check (ON condition only)
    mm_agent_vals = [r["mean_mismatch_agent_eval"] for r in results_on]
    mm_env_vals   = [r["mean_mismatch_env_eval"]   for r in results_on]
    mean_mm_agent = float(sum(mm_agent_vals) / max(1, len(mm_agent_vals)))
    mean_mm_env   = float(sum(mm_env_vals)   / max(1, len(mm_env_vals)))
    c5_pass = (
        mean_mm_env >= mean_mm_agent * THRESH_C5_MISMATCH_RATIO
        if mean_mm_agent > 0 else False
    )

    c1_pass = auc_delta  >= THRESH_C1_AUC_DELTA
    c2_pass = auc_on     >= THRESH_C2_AUC_ABS
    c4_pass = min_contacts >= THRESH_C4_MIN_CONTACTS
    all_pass = c1_pass and c2_pass and c3_pass and c4_pass

    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    print(f"\n[V3-EXQ-121] Final results:", flush=True)
    print(
        f"  auc_ON={auc_on:.4f}  auc_ABLATED={auc_off:.4f}"
        f"  delta={auc_delta:+.4f}  (C1 thresh >={THRESH_C1_AUC_DELTA})"
        f"  C1={'PASS' if c1_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  auc_ON={auc_on:.4f}  (C2 absolute thresh >={THRESH_C2_AUC_ABS})"
        f"  C2={'PASS' if c2_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  per_seed_ON_vs_ABLATED: {per_seed_c1}"
        f"  C3={'PASS' if c3_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  min_contacts={min_contacts}  (C4 thresh >={THRESH_C4_MIN_CONTACTS})"
        f"  C4={'PASS' if c4_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  mm_agent={mean_mm_agent:.4f}  mm_env={mean_mm_env:.4f}"
        f"  ratio={mean_mm_env / max(1e-6, mean_mm_agent):.3f}"
        f"  (C5 thresh >={THRESH_C5_MISMATCH_RATIO})"
        f"  C5={'PASS' if c5_pass else 'FAIL'} (diagnostic only)",
        flush=True,
    )
    print(f"  status={status}  ({criteria_met}/5 criteria met, 4 required)", flush=True)

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: auc_ON={auc_on:.4f} vs auc_ABLATED={auc_off:.4f}"
            f" (delta={auc_delta:+.4f}, needs >={THRESH_C1_AUC_DELTA})."
            " AgencyComparator did not provide relative attribution advantage."
            " Possible causes: (1) z_self delta signal too noisy at self_dim=32,"
            " (2) env_drift_prob too low (too few env_caused events),"
            " (3) action embedding insufficient for efference copy at scale."
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: auc_ON={auc_on:.4f} (needs >={THRESH_C2_AUC_ABS})."
            " ON condition did not achieve reliable attribution above chance."
            " Check n_agent_caused_train and n_env_caused_train -- sparse signal."
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: per_seed direction inconsistent ({per_seed_c1})."
            " ON did not consistently beat ABLATED across seeds."
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: min_contacts={min_contacts} < {THRESH_C4_MIN_CONTACTS}."
            " Insufficient hazard contact events in eval -- data quality issue."
            " Increase env_drift_prob or warmup_episodes."
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 (diagnostic): mm_agent={mean_mm_agent:.4f} mm_env={mean_mm_env:.4f}"
            f" ratio={mean_mm_env / max(1e-6, mean_mm_agent):.3f}"
            f" < {THRESH_C5_MISMATCH_RATIO}."
            " Env-caused contacts did not produce higher z_self mismatch than agent-caused."
            " The efference copy may not capture enough predictable self-motion signal."
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    if all_pass:
        interpretation = (
            "MECH-095 SUPPORTED: AgencyComparator (efference-copy mismatch)"
            f" achieves AUC={auc_on:.4f} (>={THRESH_C2_AUC_ABS}) and outperforms"
            f" direct fusion by delta={auc_delta:+.4f} (>={THRESH_C1_AUC_DELTA})."
            " Direction consistent across all seeds (C3). Data quality confirmed (C4)."
            " TPJ-equivalent efference-copy mismatch comparator provides structured"
            " agency attribution advantage over unstructured latent fusion."
        )
    elif c1_pass and not c2_pass:
        interpretation = (
            "PARTIAL: C1 passes (relative delta={:.4f}) but C2 fails"
            " (absolute AUC={:.4f} < {:.2f}). Comparator provides relative advantage"
            " but neither condition learns reliable attribution."
            " Likely data sparsity -- check attribution event counts."
        ).format(auc_delta, auc_on, THRESH_C2_AUC_ABS)
    elif c2_pass and not c1_pass:
        interpretation = (
            "PARTIAL: C2 passes (AUC_ON={:.4f} >= {:.2f}) but C1 fails"
            " (delta={:+.4f} < {:.2f}). AgencyComparator learns attribution but"
            " direct fusion baseline is nearly as good. The z_self delta + z_world"
            " fusion may already encode agency signal at this scale."
            " Consider deeper comparator or larger self_dim."
        ).format(auc_on, THRESH_C2_AUC_ABS, auc_delta, THRESH_C1_AUC_DELTA)
    else:
        interpretation = (
            "MECH-095 NOT SUPPORTED at current operationalisation."
            f" auc_ON={auc_on:.4f}, auc_ABLATED={auc_off:.4f}, delta={auc_delta:+.4f}."
            " AgencyComparator (efference-copy mismatch) did not outperform direct"
            " fusion baseline. Possible causes: (1) self_dim=32 z_self delta too noisy"
            " for reliable efference-copy prediction; (2) action embedding (one-hot 5)"
            " insufficient to predict z_self dynamics; (3) env_drift_prob=0.3 generates"
            " sufficient env_caused events but they may not produce discriminable"
            " z_self mismatch patterns."
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    per_on_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" auc={r['attribution_auc']:.4f}"
        f" harm_rate={r['harm_rate']:.4f}"
        f" contacts_eval={r['n_contact_steps_eval']}"
        f" (a={r['n_agent_caused_eval']} e={r['n_env_caused_eval']})"
        f" mm_agent={r['mean_mismatch_agent_eval']:.4f}"
        f" mm_env={r['mean_mismatch_env_eval']:.4f}"
        f" train: a={r['n_agent_caused_train']} e={r['n_env_caused_train']}"
        for r in results_on
    )
    per_off_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" auc={r['attribution_auc']:.4f}"
        f" harm_rate={r['harm_rate']:.4f}"
        f" contacts_eval={r['n_contact_steps_eval']}"
        f" (a={r['n_agent_caused_eval']} e={r['n_env_caused_eval']})"
        f" train: a={r['n_agent_caused_train']} e={r['n_env_caused_train']}"
        for r in results_off
    )

    summary_markdown = (
        f"# V3-EXQ-121 -- MECH-095 Agency Attribution Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claim:** MECH-095\n"
        f"**Proposal:** EXP-0019 / EVB-0015\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n"
        f"**env_drift_prob:** {env_drift_prob}  **env_drift_interval:** {env_drift_interval}\n\n"
        f"## Design\n\n"
        f"AGENCY_ATTRIBUTION_ON: AgencyComparator computes efference-copy mismatch"
        f" (predicted z_self delta vs observed), encodes as agency signal.\n"
        f"AGENCY_ATTRIBUTION_ABLATED: Direct fusion of [z_self_delta, z_world] without"
        f" structured comparison.\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: auc_ON - auc_ABLATED >= {THRESH_C1_AUC_DELTA}  (relative advantage)\n"
        f"C2: auc_ON >= {THRESH_C2_AUC_ABS}  (absolute attribution learning)\n"
        f"C3: auc_ON > auc_ABLATED for ALL seeds  (consistency)\n"
        f"C4: min_contacts >= {THRESH_C4_MIN_CONTACTS}  (data quality)\n"
        f"C5 (diagnostic): env mismatch >= {THRESH_C5_MISMATCH_RATIO}x agent mismatch\n\n"
        f"## Aggregate Results\n\n"
        f"| Metric | AGENCY_ON | AGENCY_ABLATED | Delta | Pass |\n"
        f"|--------|-----------|----------------|-------|------|\n"
        f"| attribution_AUC (C1) | {auc_on:.4f} | {auc_off:.4f}"
        f" | {auc_delta:+.4f} | {'YES' if c1_pass else 'NO'} |\n"
        f"| attribution_AUC >= {THRESH_C2_AUC_ABS} (C2) | {auc_on:.4f} | -- | --"
        f" | {'YES' if c2_pass else 'NO'} |\n"
        f"| seed consistency (C3) | {per_seed_c1} | -- | --"
        f" | {'YES' if c3_pass else 'NO'} |\n"
        f"| min_contacts (C4) | {min_contacts} | -- | --"
        f" | {'YES' if c4_pass else 'NO'} |\n"
        f"| mm_env/mm_agent (C5 diag) | {mean_mm_env / max(1e-6, mean_mm_agent):.3f} | -- | --"
        f" | {'YES' if c5_pass else 'NO'} |\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed (AGENCY_ATTRIBUTION_ON)\n\n"
        f"{per_on_rows}\n\n"
        f"## Per-Seed (AGENCY_ATTRIBUTION_ABLATED)\n\n"
        f"{per_off_rows}\n"
        f"{failure_section}\n"
    )

    return {
        "status": status,
        "metrics": {
            "attribution_auc_on":       float(auc_on),
            "attribution_auc_ablated":  float(auc_off),
            "auc_delta":                float(auc_delta),
            "min_contacts_eval":        float(min_contacts),
            "mean_mismatch_agent":      float(mean_mm_agent),
            "mean_mismatch_env":        float(mean_mm_env),
            "crit1_pass":               1.0 if c1_pass else 0.0,
            "crit2_pass":               1.0 if c2_pass else 0.0,
            "crit3_pass":               1.0 if c3_pass else 0.0,
            "crit4_pass":               1.0 if c4_pass else 0.0,
            "crit5_pass":               1.0 if c5_pass else 0.0,
            "criteria_met":             float(criteria_met),
            "n_seeds":                  float(len(seeds)),
            "alpha_world":              float(alpha_world),
            "env_drift_prob":           float(env_drift_prob),
        },
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if (c1_pass or c2_pass) else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "per_seed_on":  results_on,
        "per_seed_off": results_off,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int,   nargs="+", default=[42, 123])
    parser.add_argument("--warmup",          type=int,   default=400)
    parser.add_argument("--eval-eps",        type=int,   default=50)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--self-dim",        type=int,   default=32)
    parser.add_argument("--world-dim",       type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.01)
    parser.add_argument("--drift-prob",      type=float, default=0.3)
    parser.add_argument("--drift-interval",  type=int,   default=3)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick dry-run: 3 warmup, 2 eval, 50 steps per cell. Writes JSON.",
    )
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        self_dim=args.self_dim,
        world_dim=args.world_dim,
        lr=args.lr,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        env_drift_prob=args.drift_prob,
        env_drift_interval=args.drift_interval,
        dry_run=args.dry_run,
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

    print(f"\n[V3-EXQ-121] Result written to {out_path}", flush=True)
    print(f"[V3-EXQ-121] status={result['status']}", flush=True)
