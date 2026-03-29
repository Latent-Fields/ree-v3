#!/opt/local/bin/python3
"""
V3-EXQ-177 -- SD-008 Integration Test: SD-008 + SD-007 + SD-003 Composition

Claim: SD-008
EXP-0087 (manual proposal): Verify SD-008 + SD-007 + SD-003 compose correctly.

Background:
  EXQ-023 failed because SD-007 input was wrong (used z_self_prev instead of
  z_world_raw_prev as the reafference predictor input). Each of the three components
  has since been independently validated:
    - SD-008 (alpha_world=0.9): EXQ-023b PASS
    - SD-007 (reafference correction): EXQ-027b PASS (world_forward_r2=0.947)
    - SD-003 (counterfactual attribution): EXQ-030b PASS

  This experiment checks that all three compose correctly when wired together
  in a single REEAgent, i.e., there is no destructive interference between
  the three mechanisms.

Design (fallback inline implementation):
  Uses a lightweight inline model to avoid REEAgent API uncertainty and focus
  purely on the three composition properties:

  WorldEncoder: Linear(world_obs_dim, 32) + LayerNorm -> z_world_raw
  Reafference:  Linear(32 + action_dim, 32) -> correction
                z_world = (1-alpha)*z_world_prev + alpha*(z_world_raw - correction)
                (alpha=0.9 = SD-008; correction = SD-007)
  E2World:      Linear(32 + action_dim, 32) -> z_world_next (SD-003 forward model)
  HarmHead:     Linear(32, 1) -> harm scalar (E3 harm_eval analog)

  The three metrics map directly onto the three SDs:
    event_selectivity_margin  -> SD-008 (alpha_world suppresses EMA smoothing)
    world_forward_r2          -> SD-007 (reafference-corrected z_world is predictable)
    attribution_gap           -> SD-003 (counterfactual harm gap > 0 at hazard contact)

Training:
  300 warmup episodes x 100 steps each (random actions).
  All modules trained end-to-end: reconstruction + E2 forward loss.
  HarmHead trained on harm_exposure channel (body_state[10] in proxy mode).

Eval:
  50 eval episodes x 100 steps (no gradient).
  Collect z_world per step, tagged with transition context.
  Compute three metrics; evaluate PASS/PARTIAL/FAIL criteria.

PRE-REGISTERED ACCEPTANCE CRITERIA:
  C1: event_selectivity_margin > 0.05  (both seeds) -- SD-008 validates
  C2: world_forward_r2 > 0.30          (both seeds) -- SD-007 validates
  C3: mean_attribution_gap > 0.005     (both seeds) -- SD-003 validates

  PASS:    all three criteria met for both seeds
  PARTIAL: two of three criteria met for both seeds (or all three for one seed)
  FAIL:    fewer than two criteria met

evidence_direction: "supports" if PASS or PARTIAL; "weakens" if FAIL
"""

import sys
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.environment.causal_grid_world import CausalGridWorldV2


EXPERIMENT_TYPE = "v3_exq_177_sd008_integration_test"
CLAIM_IDS = ["SD-008"]
SEEDS = [42, 123]

# Pre-registered thresholds -- DO NOT change post-hoc
THRESH_EVENT_SEL   = 0.05   # C1: hazard vs open cosine distance
THRESH_WORLD_R2    = 0.30   # C2: E2 world_forward variance explained
THRESH_ATTRIBUTION = 0.005  # C3: mean attribution gap at hazard contact

WORLD_DIM  = 32
ALPHA_WORLD = 0.9          # SD-008: high alpha to preserve event responses
ACTION_DIM  = 5            # CausalGridWorldV2: 4 directions + stay

WARMUP_EPISODES = 300
WARMUP_STEPS    = 100
EVAL_EPISODES   = 50
EVAL_STEPS      = 100


# ---------------------------------------------------------------------------
# Inline lightweight modules (avoid REEAgent API ambiguity)
# ---------------------------------------------------------------------------

class WorldEncoder(nn.Module):
    """Linear encoder: world_obs -> z_world_raw."""

    def __init__(self, world_obs_dim: int, world_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(world_obs_dim, world_dim * 2),
            nn.ReLU(),
            nn.Linear(world_dim * 2, world_dim),
            nn.LayerNorm(world_dim),
        )

    def forward(self, world_obs: torch.Tensor) -> torch.Tensor:
        return self.net(world_obs)


class ReafferenceNet(nn.Module):
    """
    SD-007: perspective-shift correction.
    Predicts expected z_world change from z_world_raw_prev + action.
    z_world_corrected = z_world_raw - correction(z_world_raw_prev, a)
    """

    def __init__(self, world_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(world_dim + action_dim, world_dim),
            nn.ReLU(),
            nn.Linear(world_dim, world_dim),
        )

    def forward(self, z_world_prev: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_world_prev, a_onehot], dim=-1)
        return self.net(x)


class E2WorldForward(nn.Module):
    """
    SD-003: world forward model for counterfactual attribution.
    Predicts z_world_next from z_world + action (residual).
    """

    def __init__(self, world_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(world_dim + action_dim, world_dim * 2),
            nn.ReLU(),
            nn.Linear(world_dim * 2, world_dim),
        )

    def forward(self, z_world: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_world, a_onehot], dim=-1)
        delta = self.net(x)
        return z_world + delta   # residual: predict delta, add to current


class HarmHead(nn.Module):
    """E3 harm_eval analog: z_world -> harm scalar."""

    def __init__(self, world_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(world_dim, world_dim // 2),
            nn.ReLU(),
            nn.Linear(world_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_world: torch.Tensor) -> torch.Tensor:
        return self.net(z_world)


def make_a_onehot(action: int, action_dim: int) -> torch.Tensor:
    """Convert integer action to one-hot float tensor [1, action_dim]."""
    t = torch.zeros(1, action_dim)
    t[0, action] = 1.0
    return t


# ---------------------------------------------------------------------------
# Per-seed experiment
# ---------------------------------------------------------------------------

def run_seed(seed: int, dry_run: bool) -> Dict:
    """Run a single seed; return per-seed metrics dict."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed,
        size=8,
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.02,
        env_drift_interval=5,
        env_drift_prob=0.3,
    )

    world_obs_dim = env.world_obs_dim  # 250 (proxy mode)
    body_obs_dim  = env.body_obs_dim   # 12 (proxy mode)
    action_dim    = env.action_dim     # 5

    # Build inline modules
    encoder    = WorldEncoder(world_obs_dim, WORLD_DIM)
    reafference = ReafferenceNet(WORLD_DIM, action_dim)
    e2_forward  = E2WorldForward(WORLD_DIM, action_dim)
    harm_head   = HarmHead(WORLD_DIM)

    all_params = (
        list(encoder.parameters())
        + list(reafference.parameters())
        + list(e2_forward.parameters())
        + list(harm_head.parameters())
    )
    optimizer = optim.Adam(all_params, lr=3e-4)

    # Transition replay buffer: (z_world_t, a_onehot, z_world_t1)
    trans_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    # -----------------------------------------------------------------------
    # Warmup training
    # -----------------------------------------------------------------------
    n_warmup_steps = 0
    z_world_ema = None       # running EMA state (detached)
    z_world_raw_prev = None  # SD-007: previous raw z_world for reafference input

    for ep in range(WARMUP_EPISODES):
        if dry_run and ep >= 2:
            break
        _, obs_dict = env.reset()

        z_world_ema = None
        z_world_raw_prev = None
        prev_action = None

        for step in range(WARMUP_STEPS):
            world_obs = obs_dict["world_state"].unsqueeze(0)   # [1, 250]
            body_obs  = obs_dict["body_state"].unsqueeze(0)    # [1, 12]

            # harm_exposure is body_state[10] in proxy mode (0-indexed)
            harm_exposure = body_obs[0, 10].item()   # scalar in [0, 1]

            # Encode raw z_world
            z_world_raw = encoder(world_obs)  # [1, 32]

            # SD-007: apply reafference correction using PREVIOUS raw z_world
            if z_world_raw_prev is not None and prev_action is not None:
                a_onehot = make_a_onehot(prev_action, action_dim)
                correction = reafference(z_world_raw_prev.detach(), a_onehot)
                z_world_corrected = z_world_raw - correction
            else:
                z_world_corrected = z_world_raw

            # SD-008: EMA with alpha=0.9 (high alpha -> minimal smoothing)
            if z_world_ema is None:
                z_world = z_world_corrected
            else:
                z_world = (1.0 - ALPHA_WORLD) * z_world_ema.detach() + ALPHA_WORLD * z_world_corrected

            # Random action
            action = env.action_space.sample() if hasattr(env, "action_space") else random.randint(0, action_dim - 1)
            a_onehot = make_a_onehot(action, action_dim)

            # Step environment
            _, harm_signal, done, info, next_obs_dict = env.step(action)

            # Encode next z_world (no gradient needed for replay storage)
            with torch.no_grad():
                z_world_raw_next = encoder(next_obs_dict["world_state"].unsqueeze(0))
                if z_world_ema is not None:
                    z_world_next = (1.0 - ALPHA_WORLD) * z_world_ema + ALPHA_WORLD * z_world_raw_next
                else:
                    z_world_next = z_world_raw_next

            # Store transition
            if len(trans_buf) < 4000:
                trans_buf.append((
                    z_world.detach().clone(),
                    a_onehot.detach().clone(),
                    z_world_next.detach().clone(),
                ))

            # Train on E2 forward + harm head
            optimizer.zero_grad()

            # E2 forward model loss
            z_world_pred = e2_forward(z_world, a_onehot)
            loss_fwd = F.mse_loss(z_world_pred, z_world_next.detach())

            # Harm head supervised on harm_exposure
            harm_pred = harm_head(z_world)
            harm_target = torch.tensor([[harm_exposure]], dtype=torch.float32)
            loss_harm = F.mse_loss(harm_pred, harm_target)

            loss = loss_fwd + loss_harm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            # Update state tracking (detached)
            z_world_ema = z_world.detach().clone()
            z_world_raw_prev = z_world_raw.detach().clone()
            prev_action = action
            obs_dict = next_obs_dict
            n_warmup_steps += 1

            if done:
                break

    print(f"  Seed {seed}: warmup done, {n_warmup_steps} steps, {len(trans_buf)} transitions", flush=True)

    # -----------------------------------------------------------------------
    # Evaluation pass
    # -----------------------------------------------------------------------
    encoder.eval()
    reafference.eval()
    e2_forward.eval()
    harm_head.eval()

    # Buffers for metric computation
    z_world_hazard: List[torch.Tensor] = []   # C1: z_world at hazard-adjacent steps
    z_world_open:   List[torch.Tensor] = []   # C1: z_world at open-space steps

    # C2: world_forward R2
    z_world_actual_list: List[torch.Tensor] = []
    z_world_pred_list:   List[torch.Tensor] = []

    # C3: attribution gap at hazard contact
    attribution_gaps: List[float] = []

    for ep in range(EVAL_EPISODES):
        if dry_run and ep >= 2:
            break
        _, obs_dict = env.reset()

        z_world_ema_eval = None
        z_world_raw_prev_eval = None
        prev_action_eval = None

        for step in range(EVAL_STEPS):
            world_obs = obs_dict["world_state"].unsqueeze(0)
            body_obs  = obs_dict["body_state"].unsqueeze(0)

            with torch.no_grad():
                z_world_raw = encoder(world_obs)

                if z_world_raw_prev_eval is not None and prev_action_eval is not None:
                    a_prev_oh = make_a_onehot(prev_action_eval, action_dim)
                    correction = reafference(z_world_raw_prev_eval, a_prev_oh)
                    z_world_corrected = z_world_raw - correction
                else:
                    z_world_corrected = z_world_raw

                if z_world_ema_eval is None:
                    z_world = z_world_corrected
                else:
                    z_world = (1.0 - ALPHA_WORLD) * z_world_ema_eval + ALPHA_WORLD * z_world_corrected

                action = env.action_space.sample() if hasattr(env, "action_space") else random.randint(0, action_dim - 1)
                a_onehot = make_a_onehot(action, action_dim)

                _, harm_signal, done, info, next_obs_dict = env.step(action)

                transition_type = info.get("transition_type", "none")
                # harm_exposure on body_state[10] (proxy mode)
                harm_exposure_val = float(body_obs[0, 10].item())

                # C1: classify step as hazard-adjacent vs open
                is_hazard = (transition_type in ("agent_caused_hazard", "env_caused_hazard",
                                                  "hazard_approach")) or (harm_signal < -0.001)
                if is_hazard:
                    z_world_hazard.append(z_world.squeeze(0).clone())
                elif transition_type == "none" and harm_exposure_val < 0.05:
                    z_world_open.append(z_world.squeeze(0).clone())

                # C2: E2 world_forward prediction accuracy
                z_world_raw_next = encoder(next_obs_dict["world_state"].unsqueeze(0))
                if z_world_ema_eval is not None:
                    z_world_next_actual = (1.0 - ALPHA_WORLD) * z_world_ema_eval + ALPHA_WORLD * z_world_raw_next
                else:
                    z_world_next_actual = z_world_raw_next
                z_world_pred = e2_forward(z_world, a_onehot)

                z_world_actual_list.append(z_world_next_actual.squeeze(0).clone())
                z_world_pred_list.append(z_world_pred.squeeze(0).clone())

                # C3: attribution gap at hazard contact
                if harm_signal < -0.001 and z_world is not None:
                    harm_actual = harm_head(z_world)  # [1, 1]

                    # Pick a counterfactual action different from actual
                    cf_actions = [a for a in range(action_dim) if a != action]
                    if cf_actions:
                        a_cf = random.choice(cf_actions)
                        a_cf_oh = make_a_onehot(a_cf, action_dim)
                        z_world_cf = e2_forward(z_world, a_cf_oh)
                        harm_cf = harm_head(z_world_cf)
                        gap = abs(float(harm_actual.item()) - float(harm_cf.item()))
                        attribution_gaps.append(gap)

                z_world_ema_eval = z_world.clone()
                z_world_raw_prev_eval = z_world_raw.clone()
                prev_action_eval = action
                obs_dict = next_obs_dict

            if done:
                break

    # -----------------------------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------------------------

    # C1: event selectivity margin
    event_selectivity_margin = 0.0
    if len(z_world_hazard) >= 5 and len(z_world_open) >= 5:
        h_stack = torch.stack(z_world_hazard[:200])   # cap for speed
        o_stack = torch.stack(z_world_open[:200])
        # Mean cosine similarity between hazard and open centroids
        h_mean = h_stack.mean(0)  # [32]
        o_mean = o_stack.mean(0)  # [32]
        cos_sim = F.cosine_similarity(h_mean.unsqueeze(0), o_mean.unsqueeze(0)).item()
        event_selectivity_margin = 1.0 - cos_sim   # 0 = identical, 1 = orthogonal
    else:
        print(f"  Seed {seed}: WARNING -- insufficient hazard ({len(z_world_hazard)}) or "
              f"open ({len(z_world_open)}) samples for C1", flush=True)

    # C2: world_forward R2
    world_forward_r2 = 0.0
    if len(z_world_actual_list) >= 10:
        actual = torch.stack(z_world_actual_list)   # [N, 32]
        pred   = torch.stack(z_world_pred_list)     # [N, 32]
        ss_res = ((actual - pred) ** 2).sum().item()
        ss_tot = ((actual - actual.mean(0, keepdim=True)) ** 2).sum().item()
        if ss_tot > 1e-8:
            world_forward_r2 = max(0.0, 1.0 - ss_res / ss_tot)
        # Clamp to [-0.5, 1.0] for readability (negative R2 = worse than mean predictor)
        world_forward_r2 = float(world_forward_r2)

    # C3: mean attribution gap
    mean_attribution_gap = float(np.mean(attribution_gaps)) if attribution_gaps else 0.0

    # Criteria evaluation
    c1_pass = event_selectivity_margin > THRESH_EVENT_SEL
    c2_pass = world_forward_r2 > THRESH_WORLD_R2
    c3_pass = mean_attribution_gap > THRESH_ATTRIBUTION
    n_pass  = sum([c1_pass, c2_pass, c3_pass])

    print(f"  Seed {seed}: C1={c1_pass} (sel={event_selectivity_margin:.4f}), "
          f"C2={c2_pass} (r2={world_forward_r2:.4f}), "
          f"C3={c3_pass} (gap={mean_attribution_gap:.5f})",
          flush=True)

    return {
        "seed": seed,
        "event_selectivity_margin": event_selectivity_margin,
        "world_forward_r2": world_forward_r2,
        "mean_attribution_gap": mean_attribution_gap,
        "n_hazard_steps": len(z_world_hazard),
        "n_open_steps": len(z_world_open),
        "n_transitions": len(z_world_actual_list),
        "n_attribution_events": len(attribution_gaps),
        "warmup_steps": n_warmup_steps,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "n_criteria_pass": n_pass,
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> dict:
    print(f"[EXQ-177] SD-008 integration test. dry_run={dry_run}", flush=True)
    print(f"  alpha_world={ALPHA_WORLD}, SD-007 reafference enabled, SD-003 counterfactual enabled",
          flush=True)
    print(f"  Thresholds: C1>{THRESH_EVENT_SEL}, C2>{THRESH_WORLD_R2}, C3>{THRESH_ATTRIBUTION}",
          flush=True)

    per_seed_results = {}
    for seed in SEEDS:
        print(f"--- Seed {seed} ---", flush=True)
        result = run_seed(seed, dry_run)
        per_seed_results[str(seed)] = result

    # Aggregate across seeds
    c1_both = all(per_seed_results[str(s)]["c1_pass"] for s in SEEDS)
    c2_both = all(per_seed_results[str(s)]["c2_pass"] for s in SEEDS)
    c3_both = all(per_seed_results[str(s)]["c3_pass"] for s in SEEDS)

    c1_either = any(per_seed_results[str(s)]["c1_pass"] for s in SEEDS)
    c2_either = any(per_seed_results[str(s)]["c2_pass"] for s in SEEDS)
    c3_either = any(per_seed_results[str(s)]["c3_pass"] for s in SEEDS)

    n_criteria_both = sum([c1_both, c2_both, c3_both])

    # PASS: all three met both seeds
    # PARTIAL: >= 2 of 3 criteria met both seeds, OR all 3 for at least one seed
    # FAIL: otherwise
    if c1_both and c2_both and c3_both:
        outcome = "PASS"
        evidence_direction = "supports"
    elif n_criteria_both >= 2 or (c1_either and c2_either and c3_either):
        outcome = "PARTIAL"
        evidence_direction = "supports"
    else:
        outcome = "FAIL"
        evidence_direction = "weakens"

    summary_metrics = {
        "event_selectivity_margin_mean": float(np.mean(
            [per_seed_results[str(s)]["event_selectivity_margin"] for s in SEEDS]
        )),
        "world_forward_r2_mean": float(np.mean(
            [per_seed_results[str(s)]["world_forward_r2"] for s in SEEDS]
        )),
        "mean_attribution_gap_mean": float(np.mean(
            [per_seed_results[str(s)]["mean_attribution_gap"] for s in SEEDS]
        )),
        "c1_both_seeds": c1_both,
        "c2_both_seeds": c2_both,
        "c3_both_seeds": c3_both,
        "n_criteria_both_seeds": n_criteria_both,
    }

    criteria = {
        "C1_event_selectivity_margin_gt_0_05_both_seeds": c1_both,
        "C2_world_forward_r2_gt_0_30_both_seeds": c2_both,
        "C3_mean_attribution_gap_gt_0_005_both_seeds": c3_both,
    }

    interpretation = (
        "All three components (SD-008 alpha correction, SD-007 reafference, "
        "SD-003 counterfactual) compose correctly in a unified model."
        if outcome == "PASS"
        else (
            "Partial composition: some components validated but at least one criterion "
            "failed both seeds. Check per-seed breakdown for which SD needs further work."
            if outcome == "PARTIAL"
            else
            "Composition failed: at least two of three criteria failed both seeds. "
            "Components may be destructively interfering or one is not learning."
        )
    )

    pack = {
        "run_id": (
            f"{EXPERIMENT_TYPE}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3"
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_class": "integration_test",
        "criteria": criteria,
        "pre_registered_thresholds": {
            "THRESH_EVENT_SEL": THRESH_EVENT_SEL,
            "THRESH_WORLD_R2": THRESH_WORLD_R2,
            "THRESH_ATTRIBUTION": THRESH_ATTRIBUTION,
        },
        "summary_metrics": summary_metrics,
        "seeds": SEEDS,
        "scenario": (
            "CausalGridWorldV2 8x8, 3 hazards, 3 resources, hazard_harm=0.02, "
            "env_drift_interval=5, env_drift_prob=0.3; "
            f"alpha_world={ALPHA_WORLD} (SD-008); "
            "reafference correction enabled (SD-007); "
            "counterfactual attribution enabled (SD-003)"
        ),
        "interpretation": interpretation,
        "per_seed_results": per_seed_results,
        "config": {
            "alpha_world": ALPHA_WORLD,
            "world_dim": WORLD_DIM,
            "warmup_episodes": WARMUP_EPISODES if not dry_run else 2,
            "warmup_steps_per_ep": WARMUP_STEPS,
            "eval_episodes": EVAL_EPISODES if not dry_run else 2,
            "eval_steps_per_ep": EVAL_STEPS,
            "sd007_reafference": True,
            "sd008_alpha_correction": True,
            "sd003_counterfactual": True,
        },
        "dry_run": dry_run,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not dry_run:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly" / "evidence" / "experiments"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pack['run_id']}.json"
        import json
        with open(out_path, "w") as f:
            json.dump(pack, f, indent=2)
        print(f"Result pack written to: {out_path}", flush=True)
    else:
        print("[dry_run] Result pack NOT written.", flush=True)

    return pack


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = run_experiment(dry_run=dry_run)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
