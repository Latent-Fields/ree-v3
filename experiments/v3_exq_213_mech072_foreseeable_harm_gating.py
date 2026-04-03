#!/opt/local/bin/python3
"""
V3-EXQ-213 -- MECH-072: Foreseeable-Harm Gating Discriminative Pair

Claim: MECH-072 (candidate, v3_pending)
  "Foreseeable-harm gating on residue accumulation reduces false attribution
  without degrading harm avoidance."

Mechanism under test:
  When the agent encounters harm, MECH-072 predicts that gating residue
  accumulation on E2 harm foreseeability (does E2.world_forward predict high
  harm for this action?) reduces false attribution: residue should accumulate
  only at locations where the agent could have foreseen the harm (agent-caused
  footprint). Without gating, env-caused harm events (e.g. moving hazards,
  ambient contamination) add residue indiscriminately, polluting the terrain.

Prior failures: EXQ-054 (FAIL V2), EXQ-028 (FAIL V2).
Root cause of V2 failures: z_self/z_world split (SD-005) not yet in place,
making E2 attribution structurally impossible. SD-005 is now implemented.

Experimental design -- discriminative_pair:
  Single shared training phase (both conditions train identically).
  Eval phase diverges on residue accumulation gate only.

  Condition A: GATED
    Residue accumulates at z_world ONLY IF the E2 world-forward model predicted
    the harm was foreseeable: E3.harm_eval(E2.world_forward(z_world, a)) > GATE_THRESH.
    Interpretation: "I should have predicted this -- it goes on my record."

  Condition B: UNGATED
    Residue accumulates at z_world for ALL harm events (old, unfiltered behavior).
    Interpretation: "Any harm near me accumulates regardless of cause."

  Both conditions share:
    - Same trained encoder (WorldEncoder), E2 (world_forward), E3 (harm_eval)
    - CausalGridWorldV2 with use_proxy_fields=True (gradient fields visible)
    - transition_type labels from info dict (for false_attribution ground truth)
    - "agent_caused_hazard" = agent-caused harm (should accumulate in both)
    - "env_caused_hazard" = env-caused harm (should be gated out in GATED)

  False attribution metric:
    false_attr_rate = residue_at_env_caused / (residue_at_env_caused + residue_at_agent_caused)
    Lower is better for GATED condition.

  Harm avoidance metric:
    harm_rate = n_harm_steps / total_steps during eval
    Should not increase significantly with gating.

Pre-registered acceptance criteria:
  C1: false_attr_rate_GATED < false_attr_rate_UNGATED (each seed independently)
      Gating must reduce false attribution -- the direction is the primary claim.

  C2: mean delta_false_attr (UNGATED - GATED) >= THRESH_C2_FAR_DELTA
      The reduction must exceed a minimum threshold (not just noise).
      THRESH_C2_FAR_DELTA = 0.05 (5 pp reduction in false attribution rate).

  C3: harm_rate_GATED <= harm_rate_UNGATED + THRESH_C3_HARM_TOLERANCE
      Gating must not degrade harm avoidance beyond tolerance.
      THRESH_C3_HARM_TOLERANCE = 0.03 (3 pp absolute).

  C4: n_env_caused_events >= THRESH_C4_MIN_ENV_EVENTS (each seed, data quality)
      Need sufficient env-caused events to measure false attribution rate.
      THRESH_C4_MIN_ENV_EVENTS = 5

  C5: n_agent_caused_events >= THRESH_C5_MIN_AGENT_EVENTS (each seed, data quality)
      Need sufficient agent-caused events for baseline residue.
      THRESH_C5_MIN_AGENT_EVENTS = 5

  C6: e2_world_r2 >= THRESH_C6_E2_QUALITY
      E2 world_forward must be adequately trained for gating to work.
      THRESH_C6_E2_QUALITY = 0.10

Outcome:
  PASS: C1 + C2 + C3 + C4 + C5 + C6 (all seeds)
  INCONCLUSIVE: NOT C4 or NOT C5 or NOT C6 (data/training quality gate)
  FAIL: otherwise

Evidence direction:
  "supports"     if PASS
  "weakens"      if C4+C5+C6 pass but C1 fails in all seeds
  "mixed"        otherwise

Decision scoring:
  PASS -> retain_ree
  FAIL (weakens) -> hybridize (false attribution not reduced -- mechanism unclear)
  INCONCLUSIVE -> inconclusive

Estimated runtime: ~120 min on DLAPTOP-4.local (3 seeds x 300 train + 100 eval eps x 150 steps)
"""

import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import E2Config, E3Config, ResidueConfig
from ree_core.predictors.e2_fast import E2FastPredictor
from ree_core.predictors.e3_selector import E3TrajectorySelector
from ree_core.residue.field import ResidueField

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_TYPE    = "v3_exq_213_mech072_foreseeable_harm_gating"
CLAIM_IDS          = ["MECH-072"]
EXPERIMENT_PURPOSE = "evidence"
SEEDS = [42, 7, 13]

# Environment
ENV_SIZE      = 10
NUM_HAZARDS   = 3
NUM_RESOURCES = 3
HAZARD_HARM   = 0.5

# Architecture dims
WORLD_DIM  = 32
SELF_DIM   = 16
ACTION_DIM = 5       # up/down/left/right/stay

# Note: body_obs_dim=12, world_obs_dim=250 with use_proxy_fields=True
BODY_OBS_DIM  = 12
WORLD_OBS_DIM = 250

# Training
LR               = 3e-4
N_TRAIN_EPISODES = 300
STEPS_PER_TRAIN  = 150

# Eval
N_EVAL_EPISODES = 100
STEPS_PER_EVAL  = 150

# Gating threshold: E3.harm_eval > GATE_THRESH -> foreseeable harm -> accumulate
GATE_THRESH = 0.2

# Pre-registered thresholds
THRESH_C2_FAR_DELTA     = 0.05
THRESH_C3_HARM_TOLERANCE = 0.03
THRESH_C4_MIN_ENV_EVENTS  = 5
THRESH_C5_MIN_AGENT_EVENTS = 5
THRESH_C6_E2_QUALITY       = 0.10


# ---------------------------------------------------------------------------
# Inline modules (lightweight standalone versions)
# ---------------------------------------------------------------------------

class WorldEncoder(nn.Module):
    """Encode world_obs -> z_world."""
    def __init__(self, in_dim: int = WORLD_OBS_DIM, out_dim: int = WORLD_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Build modules
# ---------------------------------------------------------------------------

def build_modules(seed: int) -> dict:
    torch.manual_seed(seed)

    e2_cfg = E2Config(
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=64,
    )
    e2 = E2FastPredictor(e2_cfg)

    res_cfg = ResidueConfig(
        world_dim=WORLD_DIM,
        hidden_dim=32,
        accumulation_rate=0.2,
        num_basis_functions=32,
        kernel_bandwidth=1.0,
    )
    residue_field = ResidueField(res_cfg)

    e3_cfg = E3Config(
        world_dim=WORLD_DIM,
        hidden_dim=64,
        lambda_ethical=1.0,
        rho_residue=0.5,
    )
    e3 = E3TrajectorySelector(e3_cfg, residue_field=residue_field)

    world_enc = WorldEncoder()

    return {
        "e2": e2,
        "e3": e3,
        "residue_field": residue_field,
        "world_enc": world_enc,
    }


# ---------------------------------------------------------------------------
# Training phase (shared between conditions)
# ---------------------------------------------------------------------------

def train_modules(modules: dict, seed: int, dry_run: bool = False) -> dict:
    """
    Train e2 (world_forward) and e3 (harm_eval) jointly.
    During training, accumulate residue at ALL harm events (ungated baseline).
    Returns training diagnostics including e2_world_r2.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    n_eps   = 10 if dry_run else N_TRAIN_EPISODES
    n_steps = 30 if dry_run else STEPS_PER_TRAIN

    env = CausalGridWorld(
        size=ENV_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=HAZARD_HARM,
        use_proxy_fields=True,
        seed=seed,
    )

    e2          = modules["e2"]
    e3          = modules["e3"]
    residue     = modules["residue_field"]
    world_enc   = modules["world_enc"]

    all_params = (
        list(e2.parameters())
        + list(e3.parameters())
        + list(world_enc.parameters())
    )
    optimizer = optim.Adam(all_params, lr=LR)

    buffer: List = []
    MAX_BUF = 20000

    for ep in range(n_eps):
        flat_obs, obs_dict = env.reset()
        ws = obs_dict["world_state"].float()

        for _ in range(n_steps):
            a = rng.randint(0, ACTION_DIM - 1)
            a_onehot = torch.zeros(ACTION_DIM)
            a_onehot[a] = 1.0

            flat_next, harm_signal, done, info, obs_dict_next = env.step(a)
            ws_next = obs_dict_next["world_state"].float()
            harm_val = abs(float(harm_signal)) if harm_signal < 0 else 0.0
            transition_type = info.get("transition_type", "none")

            buffer.append((ws.clone(), a_onehot.clone(), ws_next.clone(), harm_val, transition_type))
            if len(buffer) > MAX_BUF:
                buffer.pop(0)

            # Accumulate residue during training (ungated -- for terrain formation)
            if harm_val > 0.01:
                with torch.no_grad():
                    z_w = world_enc(ws.unsqueeze(0))
                residue.accumulate(z_w, harm_magnitude=harm_val, hypothesis_tag=False)

            ws = ws_next
            if done:
                break

        if len(buffer) >= 64:
            batch = random.sample(buffer, min(128, len(buffer)))
            ws_b      = torch.stack([b[0] for b in batch])
            a_b       = torch.stack([b[1] for b in batch])
            ws_next_b = torch.stack([b[2] for b in batch])
            harm_b    = torch.tensor([b[3] for b in batch], dtype=torch.float32).unsqueeze(1)

            z_w      = world_enc(ws_b)
            z_w_next = world_enc(ws_next_b).detach()

            # E2 world_forward loss
            z_w_pred     = e2.world_forward(z_w, a_b)
            loss_e2_world = F.mse_loss(z_w_pred, z_w_next)

            # E3 harm_eval loss (supervised from harm signal)
            harm_pred = e3.harm_eval(z_w)
            loss_harm = F.mse_loss(harm_pred, harm_b.clamp(0, 1))

            loss = loss_e2_world + loss_harm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Compute e2_world_r2 on buffer tail
    e2_world_r2 = 0.0
    world_enc.eval()
    e2.eval()
    e3.eval()

    if len(buffer) >= 100:
        eval_buf  = buffer[-200:]
        ws_e      = torch.stack([b[0] for b in eval_buf])
        a_e       = torch.stack([b[1] for b in eval_buf])
        ws_next_e = torch.stack([b[2] for b in eval_buf])
        with torch.no_grad():
            z_e      = world_enc(ws_e)
            z_pred   = e2.world_forward(z_e, a_e)
            z_tgt    = world_enc(ws_next_e)
        ss_res = ((z_pred - z_tgt) ** 2).sum().item()
        ss_tot = ((z_tgt - z_tgt.mean(0)) ** 2).sum().item()
        e2_world_r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    return {
        "e2_world_r2": e2_world_r2,
        "residue_total": float(residue.total_residue.item()),
        "num_harm_events": int(residue.num_harm_events.item()),
    }


# ---------------------------------------------------------------------------
# Eval phase
# ---------------------------------------------------------------------------

def eval_condition(
    condition: str,
    modules: dict,
    seed: int,
    dry_run: bool = False,
) -> dict:
    """
    Evaluate GATED or UNGATED condition.

    Collects:
      - harm_rate (primary harm-avoidance metric)
      - n_env_caused_events: steps where transition_type indicates env-caused harm
      - n_agent_caused_events: steps where transition_type = agent_caused_hazard
      - residue_at_env_caused: total residue accumulated at env-caused events
      - residue_at_agent_caused: total residue accumulated at agent-caused events
      - false_attr_rate: residue_at_env / (residue_at_env + residue_at_agent)
    """
    rng = random.Random(seed + 9999)
    torch.manual_seed(seed + 9999)

    n_eps   = 5  if dry_run else N_EVAL_EPISODES
    n_steps = 30 if dry_run else STEPS_PER_EVAL

    env = CausalGridWorld(
        size=ENV_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=HAZARD_HARM,
        use_proxy_fields=True,
        seed=seed + 9999,
    )

    # Use a fresh residue field for eval accumulation tracking (do not pollute training field)
    from ree_core.utils.config import ResidueConfig as RC
    eval_res_cfg = RC(
        world_dim=WORLD_DIM,
        hidden_dim=32,
        accumulation_rate=0.2,
        num_basis_functions=32,
        kernel_bandwidth=1.0,
    )
    eval_residue = ResidueField(eval_res_cfg)

    e2        = modules["e2"]
    e3        = modules["e3"]
    world_enc = modules["world_enc"]

    world_enc.eval()
    e2.eval()
    e3.eval()

    harm_steps  = 0
    total_steps = 0
    residue_env   = 0.0   # residue accumulated at env-caused events
    residue_agent = 0.0   # residue accumulated at agent-caused events
    n_env_events    = 0
    n_agent_events  = 0

    # ENV-caused transition types (CausalGridWorld labels)
    ENV_CAUSED = {"env_caused_hazard", "contaminated"}
    AGENT_CAUSED = {"agent_caused_hazard"}

    for ep in range(n_eps):
        flat_obs, obs_dict = env.reset()
        ws = obs_dict["world_state"].float()

        for _ in range(n_steps):
            a = rng.randint(0, ACTION_DIM - 1)
            a_onehot = torch.zeros(ACTION_DIM)
            a_onehot[a] = 1.0

            flat_next, harm_signal, done, info, obs_dict_next = env.step(a)
            ws_next = obs_dict_next["world_state"].float()
            harm_val = abs(float(harm_signal)) if harm_signal < 0 else 0.0
            transition_type = info.get("transition_type", "none")

            total_steps += 1
            if harm_signal < 0:
                harm_steps += 1

            # Residue accumulation decision
            if harm_val > 0.01:
                with torch.no_grad():
                    z_w   = world_enc(ws.unsqueeze(0))
                    # E2 prediction for current action -- measures foreseeability
                    z_w_pred = e2.world_forward(z_w, a_onehot.unsqueeze(0))
                    harm_forecast = float(e3.harm_eval(z_w_pred).item())

                is_foreseeable = (harm_forecast >= GATE_THRESH)

                # Determine if we should accumulate based on condition
                should_accumulate = (
                    True if condition == "UNGATED"
                    else is_foreseeable  # GATED: only if foreseeable
                )

                if should_accumulate:
                    eval_residue.accumulate(z_w, harm_magnitude=harm_val, hypothesis_tag=False)

                # Track by transition type for false_attr_rate
                if transition_type in ENV_CAUSED:
                    n_env_events += 1
                    if should_accumulate:
                        residue_env += harm_val * eval_residue.config.accumulation_rate
                elif transition_type in AGENT_CAUSED:
                    n_agent_events += 1
                    if should_accumulate:
                        residue_agent += harm_val * eval_residue.config.accumulation_rate

            ws = ws_next
            if done:
                break

    harm_rate = harm_steps / max(total_steps, 1)
    total_residue_accumulated = residue_env + residue_agent
    false_attr_rate = (
        residue_env / total_residue_accumulated
        if total_residue_accumulated > 1e-8
        else 0.0
    )

    return {
        "harm_rate":          harm_rate,
        "harm_steps":         harm_steps,
        "total_steps":        total_steps,
        "n_env_events":       n_env_events,
        "n_agent_events":     n_agent_events,
        "residue_env":        residue_env,
        "residue_agent":      residue_agent,
        "false_attr_rate":    false_attr_rate,
        "total_residue_eval": total_residue_accumulated,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> dict:
    per_seed = {}
    e2_r2_vals   = []
    gated_far    = []   # false_attr_rate per seed, GATED
    ungated_far  = []
    gated_harm   = []
    ungated_harm = []

    for seed in SEEDS:
        print(f"[EXQ-213] seed={seed} Building modules...", flush=True)
        modules = build_modules(seed)

        print(f"[EXQ-213] seed={seed} Training...", flush=True)
        train_diag = train_modules(modules, seed, dry_run=dry_run)
        e2_r2_vals.append(train_diag["e2_world_r2"])
        print(
            f"[EXQ-213] seed={seed} e2_r2={train_diag['e2_world_r2']:.4f} "
            f"residue_events={train_diag['num_harm_events']}",
            flush=True,
        )

        # Eval GATED condition (uses trained modules as-is)
        print(f"[EXQ-213] seed={seed} Eval GATED...", flush=True)
        gated_res = eval_condition("GATED", modules, seed, dry_run=dry_run)

        # Deep-copy trained modules for UNGATED condition (same weights)
        modules_copy = copy.deepcopy(modules)
        print(f"[EXQ-213] seed={seed} Eval UNGATED...", flush=True)
        ungated_res = eval_condition("UNGATED", modules_copy, seed, dry_run=dry_run)

        g_far = gated_res["false_attr_rate"]
        u_far = ungated_res["false_attr_rate"]
        gated_far.append(g_far)
        ungated_far.append(u_far)
        gated_harm.append(gated_res["harm_rate"])
        ungated_harm.append(ungated_res["harm_rate"])

        print(
            f"[EXQ-213] seed={seed} "
            f"GATED_far={g_far:.4f} UNGATED_far={u_far:.4f} delta={u_far-g_far:.4f} "
            f"GATED_harm={gated_res['harm_rate']:.4f} UNGATED_harm={ungated_res['harm_rate']:.4f}",
            flush=True,
        )

        per_seed[str(seed)] = {
            "train": train_diag,
            "GATED":   gated_res,
            "UNGATED": ungated_res,
            "c1_direction_pass": g_far < u_far,
            "delta_false_attr":  u_far - g_far,
            "harm_difference":   gated_res["harm_rate"] - ungated_res["harm_rate"],
        }

    # -----------------------------------------------------------------------
    # Criteria evaluation
    # -----------------------------------------------------------------------
    e2_r2_mean = sum(e2_r2_vals) / len(e2_r2_vals)

    c1_pass = all(per_seed[str(s)]["c1_direction_pass"] for s in SEEDS)
    c2_pass = (sum(ungated_far[i] - gated_far[i] for i in range(len(SEEDS))) / len(SEEDS)) >= THRESH_C2_FAR_DELTA
    c3_pass = all(
        (gated_harm[i] - ungated_harm[i]) <= THRESH_C3_HARM_TOLERANCE
        for i in range(len(SEEDS))
    )
    c4_pass = all(per_seed[str(s)]["GATED"]["n_env_events"] >= THRESH_C4_MIN_ENV_EVENTS for s in SEEDS)
    c5_pass = all(per_seed[str(s)]["GATED"]["n_agent_events"] >= THRESH_C5_MIN_AGENT_EVENTS for s in SEEDS)
    c6_pass = e2_r2_mean >= THRESH_C6_E2_QUALITY

    criteria = {
        "C1_direction_all_seeds": c1_pass,
        "C2_mean_delta_far":      c2_pass,
        "C3_harm_not_degraded":   c3_pass,
        "C4_env_events":          c4_pass,
        "C5_agent_events":        c5_pass,
        "C6_e2_quality":          c6_pass,
    }

    mean_delta_far = (sum(ungated_far[i] - gated_far[i] for i in range(len(SEEDS))) / len(SEEDS))
    print(f"[EXQ-213] Criteria: {criteria}", flush=True)
    print(f"[EXQ-213] mean_delta_far={mean_delta_far:.4f} e2_r2_mean={e2_r2_mean:.4f}", flush=True)

    # -----------------------------------------------------------------------
    # Outcome
    # -----------------------------------------------------------------------
    if not c4_pass or not c5_pass or not c6_pass:
        outcome            = "INCONCLUSIVE"
        evidence_direction = "inconclusive"
        decision           = "inconclusive"
    elif c1_pass and c2_pass and c3_pass:
        outcome            = "PASS"
        evidence_direction = "supports"
        decision           = "retain_ree"
    elif not c1_pass and c4_pass and c5_pass:
        outcome            = "FAIL"
        evidence_direction = "weakens"
        decision           = "hybridize"
    else:
        outcome            = "FAIL"
        evidence_direction = "mixed"
        decision           = "hybridize"

    print(f"[EXQ-213] outcome={outcome} evidence_direction={evidence_direction}", flush=True)

    # -----------------------------------------------------------------------
    # Result pack
    # -----------------------------------------------------------------------
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts_str}_v3"

    pack = {
        "run_id":             run_id,
        "experiment_type":    EXPERIMENT_TYPE,
        "claim_ids":          CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome":            outcome,
        "evidence_direction": evidence_direction,
        "evidence_class":     "discriminative_pair",
        "decision":           decision,
        "criteria":           criteria,
        "pre_registered_thresholds": {
            "GATE_THRESH":           GATE_THRESH,
            "THRESH_C2_FAR_DELTA":   THRESH_C2_FAR_DELTA,
            "THRESH_C3_HARM_TOLERANCE": THRESH_C3_HARM_TOLERANCE,
            "THRESH_C4_MIN_ENV_EVENTS":  THRESH_C4_MIN_ENV_EVENTS,
            "THRESH_C5_MIN_AGENT_EVENTS": THRESH_C5_MIN_AGENT_EVENTS,
            "THRESH_C6_E2_QUALITY":      THRESH_C6_E2_QUALITY,
        },
        "summary_metrics": {
            "gated_false_attr_rate_mean":   sum(gated_far) / len(gated_far),
            "ungated_false_attr_rate_mean": sum(ungated_far) / len(ungated_far),
            "mean_delta_false_attr":        mean_delta_far,
            "gated_harm_rate_mean":         sum(gated_harm) / len(gated_harm),
            "ungated_harm_rate_mean":       sum(ungated_harm) / len(ungated_harm),
            "e2_world_r2_mean":             e2_r2_mean,
        },
        "seeds": SEEDS,
        "config": {
            "env_size":      ENV_SIZE,
            "num_hazards":   NUM_HAZARDS,
            "hazard_harm":   HAZARD_HARM,
            "world_dim":     WORLD_DIM,
            "action_dim":    ACTION_DIM,
            "n_train_eps":   N_TRAIN_EPISODES,
            "n_eval_eps":    N_EVAL_EPISODES,
            "steps_per_ep":  STEPS_PER_TRAIN,
        },
        "scenario": (
            "discriminative_pair: GATED (residue accumulates only when E2.world_forward "
            "predicts foreseeable harm: E3.harm_eval(z_w_pred) > GATE_THRESH) vs "
            "UNGATED (residue accumulates at all harm events regardless of cause). "
            "Shared training phase (300 eps). Eval phase diverges on accumulation gate only. "
            "False attribution rate = residue_at_env_caused / total_residue. "
            f"CausalGridWorld size={ENV_SIZE} hazards={NUM_HAZARDS}. "
            f"Gating threshold={GATE_THRESH}."
        ),
        "interpretation": (
            "PASS supports MECH-072: foreseeable-harm gating reduces residue accumulation "
            "at env-caused events while preserving harm avoidance. The E2 world-forward "
            "model provides a causal filter: only foreseeable (agent-attributable) harm "
            "accumulates in the ethical residue record. "
            "FAIL weakens MECH-072: gating does not reduce false attribution rate, "
            "suggesting E2 cannot discriminate foreseeable from unforeseeable harm at "
            "this scale. INCONCLUSIVE: insufficient env-caused events or E2 quality too "
            "low for attribution signal."
        ),
        "per_seed_results": per_seed,
        "supersedes": None,
        "dry_run":    dry_run,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not dry_run:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly" / "evidence" / "experiments"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.json"
        with open(out_path, "w") as f:
            json.dump(pack, f, indent=2)
        print(f"[EXQ-213] Result written to: {out_path}", flush=True)
    else:
        print("[EXQ-213] dry_run: result NOT written.", flush=True)

    return pack


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = run_experiment(dry_run=dry_run)
    print(f"[EXQ-213] Done. outcome={result['outcome']}", flush=True)
