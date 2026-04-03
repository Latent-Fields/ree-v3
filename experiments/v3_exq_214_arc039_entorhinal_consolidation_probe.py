#!/opt/local/bin/python3
"""
V3-EXQ-214 -- ARC-039: Entorhinal Offline Consolidation Targeted Probe

Claim: ARC-039 (candidate, implementation_phase: v4)
  "Durable long-term storage of the hippocampal viability map requires a
  hippocampal-entorhinal loop that engages during offline (sleep/DMN-equivalent)
  consolidation and provides a coordinate-invariant grid representation; this
  loop does not operate during waking planning or waking immobility
  consolidation."

Evidence state: 0 experimental entries, 2 literature entries (both 'supports').
  EVB-0054 / EXP-0054: targeted probe, priority medium.

Experimental approach (targeted_probe):
  ARC-039 is a V4 claim -- it depends on mechanisms not yet implemented.
  This probe tests a V3-accessible proxy: whether periodic offline residue
  integration (a V3 approximation of hippocampal-entorhinal consolidation)
  produces better viability map retention over multi-episode sequences
  compared to no consolidation.

  The V3 approximation:
    ResidueField.integrate() performs offline gradient updates that contextualise
    (but do not erase) the residue field. This is a proxy for the consolidation
    function ARC-039 posits for the hippocampal-entorhinal loop.

  Design -- targeted_probe with 2 conditions (discriminative, matched seeds):
    Condition A: WITH_CONSOLIDATION
      After every N_CONSOLIDATION_INTERVAL episodes, call residue.integrate()
      (N_CONSOLIDATION_STEPS gradient steps). Simulates offline consolidation.

    Condition B: NO_CONSOLIDATION
      Same training and eval; residue.integrate() is never called.
      Raw RBF field only -- no offline contextualisation.

  Both conditions accumulate residue identically during waking episodes.
  The probe measures viability map quality and retention:
    1. Residue accuracy: Pearson r(residue_score_at_position, hazard_proximity)
       measured over eval episodes. Higher = better terrain representation.
    2. Viability lift: mean harm rate WITH_CONSOLIDATION vs NO_CONSOLIDATION.
       ARC-039 predicts consolidation improves long-term terrain quality,
       which should translate to better harm avoidance.

Note on scope: This probe tests PERSISTENCE/QUALITY of residue maps in a
V3 proxy regime. It does NOT test the full hippocampal-entorhinal biological
circuit (grid cells, coordinate-invariant representation) which is V4 scope.
A positive result would provide weak supporting evidence for the principle
that offline consolidation improves viability map quality. A null result
does NOT refute ARC-039 (the probe may be too weak). Classify as
EXPERIMENT_PURPOSE = "diagnostic" (proxy measure, not direct claim test).

Pre-registered acceptance criteria:
  C1: residue_accuracy_CONSOL > residue_accuracy_NOCONSOL (each seed)
      Consolidation should improve terrain-hazard correlation.

  C2: mean delta_accuracy (CONSOL - NOCONSOL) >= THRESH_C2_ACC_DELTA
      THRESH_C2_ACC_DELTA = 0.05 (5 pp improvement in residue accuracy).

  C3: n_harm_events_eval >= THRESH_C3_MIN_HARM (each seed, data quality)
      THRESH_C3_MIN_HARM = 5

  C4: residue_total_CONSOL > THRESH_C4_MIN_RESIDUE (each seed, residue populated)
      THRESH_C4_MIN_RESIDUE = 0.01

Outcome:
  PASS: C1 + C2 + C3 + C4 (all seeds) -> "weak support" (diagnostic probe)
  INCONCLUSIVE: NOT C3 or NOT C4
  FAIL: otherwise

Note: This is a diagnostic probe. A PASS provides weak supporting evidence
for the principle that offline consolidation benefits map quality. The
evidence_direction is thus "mixed" at best for ARC-039 directly (since it
tests only a V3 proxy, not the full V4 mechanism).

Evidence direction:
  "mixed"  if PASS (weak V3 proxy support; does not confirm V4 mechanism)
  "weakens" if C3+C4 pass but C1 fails in all seeds (no consolidation benefit)
  "inconclusive" otherwise

Estimated runtime: ~90 min on DLAPTOP-4.local (3 seeds x 400 train eps + 100 eval eps)
"""

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import E2Config, ResidueConfig
from ree_core.predictors.e2_fast import E2FastPredictor
from ree_core.residue.field import ResidueField

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_TYPE    = "v3_exq_214_arc039_entorhinal_consolidation_probe"
CLAIM_IDS          = ["ARC-039"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7, 13]

# Environment
ENV_SIZE      = 10
NUM_HAZARDS   = 3
NUM_RESOURCES = 3
HAZARD_HARM   = 0.5

# Architecture
WORLD_DIM     = 32
SELF_DIM      = 16
ACTION_DIM    = 5
WORLD_OBS_DIM = 250

# Training
LR               = 3e-4
N_TRAIN_EPISODES = 300
STEPS_PER_EP     = 150

# Consolidation (offline residue integration)
N_CONSOLIDATION_INTERVAL = 20   # integrate every N episodes
N_CONSOLIDATION_STEPS    = 10   # gradient steps per consolidation

# Eval
N_EVAL_EPISODES = 100
STEPS_PER_EVAL  = 150

# Pre-registered thresholds
THRESH_C2_ACC_DELTA  = 0.05
THRESH_C3_MIN_HARM   = 5
THRESH_C4_MIN_RESIDUE = 0.01


# ---------------------------------------------------------------------------
# Inline WorldEncoder
# ---------------------------------------------------------------------------

class WorldEncoder(nn.Module):
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
# Build shared modules
# ---------------------------------------------------------------------------

def build_modules(seed: int) -> dict:
    torch.manual_seed(seed)

    res_cfg = ResidueConfig(
        world_dim=WORLD_DIM,
        hidden_dim=32,
        accumulation_rate=0.2,
        num_basis_functions=32,
        kernel_bandwidth=1.0,
    )

    residue_consol = ResidueField(res_cfg)
    residue_noconsol = ResidueField(res_cfg)  # separate field, no integrate() calls

    world_enc = WorldEncoder()

    return {
        "world_enc":       world_enc,
        "residue_consol":  residue_consol,
        "residue_noconsol": residue_noconsol,
    }


# ---------------------------------------------------------------------------
# Training + consolidation phase
# ---------------------------------------------------------------------------

def train_condition(
    condition: str,
    modules: dict,
    seed: int,
    dry_run: bool = False,
) -> dict:
    """
    Train encoder on world prediction, accumulate residue in the condition-specific
    residue field. WITH_CONSOLIDATION calls integrate() every N_CONSOLIDATION_INTERVAL eps.
    NO_CONSOLIDATION never calls integrate().

    Both conditions share the SAME world_enc (trained jointly in advance), but
    accumulate residue independently in their own ResidueField instances.
    """
    rng = random.Random(seed + (0 if condition == "WITH_CONSOLIDATION" else 100))
    torch.manual_seed(seed + (0 if condition == "WITH_CONSOLIDATION" else 100))

    n_eps   = 10 if dry_run else N_TRAIN_EPISODES
    n_steps = 30 if dry_run else STEPS_PER_EP

    env = CausalGridWorld(
        size=ENV_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=HAZARD_HARM,
        use_proxy_fields=True,
        seed=seed,
    )

    residue = (
        modules["residue_consol"] if condition == "WITH_CONSOLIDATION"
        else modules["residue_noconsol"]
    )
    world_enc = modules["world_enc"]

    optimizer = optim.Adam(list(world_enc.parameters()), lr=LR)

    # E2 for simple world_forward (used for probing only -- not trained here)
    e2_cfg = E2Config(
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=64,
    )
    e2 = E2FastPredictor(e2_cfg)
    e2_opt = optim.Adam(list(e2.parameters()), lr=LR)

    buffer: List = []
    MAX_BUF = 15000

    for ep in range(n_eps):
        flat_obs, obs_dict = env.reset()
        ws = obs_dict["world_state"].float()

        for _ in range(n_steps):
            a = rng.randint(0, ACTION_DIM - 1)
            a_onehot = torch.zeros(ACTION_DIM)
            a_onehot[a] = 1.0

            flat_next, harm_signal, done, info, obs_dict_next = env.step(a)
            ws_next = obs_dict_next["world_state"].float()
            harm_val  = abs(float(harm_signal)) if harm_signal < 0 else 0.0
            hazard_field_val = obs_dict_next.get(
                "hazard_field_view",
                torch.zeros(25),
            ).float()

            buffer.append((ws.clone(), a_onehot.clone(), ws_next.clone(),
                           harm_val, hazard_field_val.clone()))
            if len(buffer) > MAX_BUF:
                buffer.pop(0)

            # Accumulate residue at harm events
            if harm_val > 0.01:
                with torch.no_grad():
                    z_w = world_enc(ws.unsqueeze(0))
                residue.accumulate(z_w, harm_magnitude=harm_val, hypothesis_tag=False)

            ws = ws_next
            if done:
                break

        # Train encoder + E2 on buffer
        if len(buffer) >= 64:
            batch = random.sample(buffer, min(128, len(buffer)))
            ws_b      = torch.stack([b[0] for b in batch])
            a_b       = torch.stack([b[1] for b in batch])
            ws_next_b = torch.stack([b[2] for b in batch])

            z_w      = world_enc(ws_b)
            z_w_next_tgt = world_enc(ws_next_b).detach()
            z_w_pred = e2.world_forward(z_w, a_b)
            loss = F.mse_loss(z_w_pred, z_w_next_tgt)

            optimizer.zero_grad()
            e2_opt.zero_grad()
            loss.backward()
            optimizer.step()
            e2_opt.step()

        # Offline consolidation for WITH_CONSOLIDATION condition
        if (condition == "WITH_CONSOLIDATION"
                and (ep + 1) % N_CONSOLIDATION_INTERVAL == 0):
            residue.integrate(num_steps=N_CONSOLIDATION_STEPS)
            print(
                f"  [EXQ-214] {condition} seed={seed} ep={ep+1}: consolidation done",
                flush=True,
            )

    return {
        "residue_total": float(residue.total_residue.item()),
        "num_harm_events": int(residue.num_harm_events.item()),
    }


# ---------------------------------------------------------------------------
# Eval: measure residue accuracy and harm avoidance
# ---------------------------------------------------------------------------

def eval_condition(
    condition: str,
    modules: dict,
    seed: int,
    dry_run: bool = False,
) -> dict:
    """
    Evaluate residue accuracy (correlation with hazard proximity) and harm rate.
    residue_accuracy = Pearson r(residue_score, hazard_field_max_at_position).
    """
    rng = random.Random(seed + 5555)
    torch.manual_seed(seed + 5555)

    n_eps   = 5  if dry_run else N_EVAL_EPISODES
    n_steps = 30 if dry_run else STEPS_PER_EVAL

    env = CausalGridWorld(
        size=ENV_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=HAZARD_HARM,
        use_proxy_fields=True,
        seed=seed + 5555,
    )

    residue   = (
        modules["residue_consol"] if condition == "WITH_CONSOLIDATION"
        else modules["residue_noconsol"]
    )
    world_enc = modules["world_enc"]
    world_enc.eval()

    harm_steps  = 0
    total_steps = 0
    residue_scores: List[float] = []
    hazard_proxies: List[float] = []

    for ep in range(n_eps):
        flat_obs, obs_dict = env.reset()
        ws = obs_dict["world_state"].float()

        for _ in range(n_steps):
            a = rng.randint(0, ACTION_DIM - 1)

            flat_next, harm_signal, done, info, obs_dict_next = env.step(a)
            ws_next = obs_dict_next["world_state"].float()
            total_steps += 1

            if harm_signal < 0:
                harm_steps += 1

            # Measure residue score at current position
            with torch.no_grad():
                z_w = world_enc(ws.unsqueeze(0))
                r_score = float(residue.evaluate(z_w).item())

            # Hazard proximity proxy: max of hazard_field_view
            hfv = obs_dict_next.get("hazard_field_view", torch.zeros(25)).float()
            hazard_proxy = float(hfv.max().item())

            residue_scores.append(r_score)
            hazard_proxies.append(hazard_proxy)

            ws = ws_next
            if done:
                break

    harm_rate = harm_steps / max(total_steps, 1)

    # Pearson correlation between residue score and hazard proximity
    residue_accuracy = 0.0
    if len(residue_scores) >= 10:
        try:
            rs = np.array(residue_scores, dtype=float)
            hp = np.array(hazard_proxies, dtype=float)
            rs_std = rs.std()
            hp_std = hp.std()
            if rs_std > 1e-8 and hp_std > 1e-8:
                r = float(np.corrcoef(rs, hp)[0, 1])
                residue_accuracy = r if not (r != r) else 0.0  # NaN check
        except Exception:
            residue_accuracy = 0.0

    return {
        "harm_rate":         harm_rate,
        "harm_steps":        harm_steps,
        "total_steps":       total_steps,
        "residue_accuracy":  residue_accuracy,
        "n_data_points":     len(residue_scores),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> dict:
    per_seed = {}
    consol_acc   = []
    noconsol_acc = []
    consol_harm  = []
    noconsol_harm = []
    residue_totals = []
    env_harm_counts = []

    for seed in SEEDS:
        print(f"[EXQ-214] seed={seed} Building modules...", flush=True)
        modules = build_modules(seed)

        # Train both conditions (shared world_enc; separate residue fields)
        print(f"[EXQ-214] seed={seed} Training WITH_CONSOLIDATION...", flush=True)
        train_c = train_condition("WITH_CONSOLIDATION", modules, seed, dry_run=dry_run)

        print(f"[EXQ-214] seed={seed} Training NO_CONSOLIDATION...", flush=True)
        train_nc = train_condition("NO_CONSOLIDATION", modules, seed, dry_run=dry_run)

        print(
            f"[EXQ-214] seed={seed} "
            f"residue_consol={train_c['residue_total']:.4f} "
            f"residue_noconsol={train_nc['residue_total']:.4f}",
            flush=True,
        )

        # Eval
        print(f"[EXQ-214] seed={seed} Eval WITH_CONSOLIDATION...", flush=True)
        eval_c  = eval_condition("WITH_CONSOLIDATION",  modules, seed, dry_run=dry_run)
        print(f"[EXQ-214] seed={seed} Eval NO_CONSOLIDATION...", flush=True)
        eval_nc = eval_condition("NO_CONSOLIDATION", modules, seed, dry_run=dry_run)

        c_acc  = eval_c["residue_accuracy"]
        nc_acc = eval_nc["residue_accuracy"]
        consol_acc.append(c_acc)
        noconsol_acc.append(nc_acc)
        consol_harm.append(eval_c["harm_rate"])
        noconsol_harm.append(eval_nc["harm_rate"])
        residue_totals.append(train_c["residue_total"])
        env_harm_counts.append(eval_c["harm_steps"])

        print(
            f"[EXQ-214] seed={seed} "
            f"CONSOL_acc={c_acc:.4f} NOCONSOL_acc={nc_acc:.4f} delta={c_acc-nc_acc:.4f} "
            f"CONSOL_harm={eval_c['harm_rate']:.4f} NOCONSOL_harm={eval_nc['harm_rate']:.4f}",
            flush=True,
        )

        per_seed[str(seed)] = {
            "train_consol":  train_c,
            "train_noconsol": train_nc,
            "eval_consol":   eval_c,
            "eval_noconsol": eval_nc,
            "c1_direction_pass": c_acc > nc_acc,
            "delta_accuracy": c_acc - nc_acc,
        }

    # -----------------------------------------------------------------------
    # Criteria
    # -----------------------------------------------------------------------
    c1_pass = all(per_seed[str(s)]["c1_direction_pass"] for s in SEEDS)
    mean_delta_acc = sum(consol_acc[i] - noconsol_acc[i] for i in range(len(SEEDS))) / len(SEEDS)
    c2_pass = mean_delta_acc >= THRESH_C2_ACC_DELTA
    c3_pass = all(env_harm_counts[i] >= THRESH_C3_MIN_HARM for i in range(len(SEEDS)))
    c4_pass = all(residue_totals[i] >= THRESH_C4_MIN_RESIDUE for i in range(len(SEEDS)))

    criteria = {
        "C1_direction_all_seeds": c1_pass,
        "C2_mean_delta_accuracy": c2_pass,
        "C3_harm_data_quality":   c3_pass,
        "C4_residue_populated":   c4_pass,
    }

    print(f"[EXQ-214] Criteria: {criteria}", flush=True)
    print(f"[EXQ-214] mean_delta_acc={mean_delta_acc:.4f}", flush=True)

    # -----------------------------------------------------------------------
    # Outcome
    # -----------------------------------------------------------------------
    if not c3_pass or not c4_pass:
        outcome            = "INCONCLUSIVE"
        evidence_direction = "inconclusive"
        decision           = "inconclusive"
    elif c1_pass and c2_pass:
        # PASS but this is a diagnostic probe -- weak support only
        outcome            = "PASS"
        evidence_direction = "mixed"   # proxy only; does not confirm V4 mechanism
        decision           = "retain_ree"
    elif not c1_pass and c3_pass and c4_pass:
        outcome            = "FAIL"
        evidence_direction = "weakens"
        decision           = "hybridize"
    else:
        outcome            = "FAIL"
        evidence_direction = "mixed"
        decision           = "hybridize"

    print(f"[EXQ-214] outcome={outcome} evidence_direction={evidence_direction}", flush=True)

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
        "evidence_class":     "targeted_probe",
        "decision":           decision,
        "criteria":           criteria,
        "pre_registered_thresholds": {
            "THRESH_C2_ACC_DELTA":   THRESH_C2_ACC_DELTA,
            "THRESH_C3_MIN_HARM":    THRESH_C3_MIN_HARM,
            "THRESH_C4_MIN_RESIDUE": THRESH_C4_MIN_RESIDUE,
            "N_CONSOLIDATION_INTERVAL": N_CONSOLIDATION_INTERVAL,
            "N_CONSOLIDATION_STEPS":    N_CONSOLIDATION_STEPS,
        },
        "summary_metrics": {
            "consol_residue_acc_mean":   sum(consol_acc) / len(consol_acc),
            "noconsol_residue_acc_mean": sum(noconsol_acc) / len(noconsol_acc),
            "mean_delta_accuracy":       mean_delta_acc,
            "consol_harm_rate_mean":     sum(consol_harm) / len(consol_harm),
            "noconsol_harm_rate_mean":   sum(noconsol_harm) / len(noconsol_harm),
        },
        "seeds": SEEDS,
        "config": {
            "env_size":         ENV_SIZE,
            "num_hazards":      NUM_HAZARDS,
            "hazard_harm":      HAZARD_HARM,
            "world_dim":        WORLD_DIM,
            "action_dim":       ACTION_DIM,
            "n_train_eps":      N_TRAIN_EPISODES,
            "n_eval_eps":       N_EVAL_EPISODES,
            "consol_interval":  N_CONSOLIDATION_INTERVAL,
            "consol_steps":     N_CONSOLIDATION_STEPS,
        },
        "scope_note": (
            "V3 proxy probe for ARC-039 (V4 claim). Tests whether periodic offline "
            "residue integration improves viability map accuracy. Does NOT test the "
            "full hippocampal-entorhinal grid cell mechanism (V4 scope). A positive "
            "result provides weak diagnostic support for the consolidation principle."
        ),
        "scenario": (
            "targeted_probe: WITH_CONSOLIDATION (ResidueField.integrate() called every "
            f"{N_CONSOLIDATION_INTERVAL} training episodes, {N_CONSOLIDATION_STEPS} steps) "
            "vs NO_CONSOLIDATION (no integrate() calls). Shared WorldEncoder trained jointly "
            "on world prediction. Separate residue fields accumulate identically during "
            f"training. Eval: residue accuracy = Pearson r(residue_score, hazard_field_max). "
            f"CausalGridWorld size={ENV_SIZE} hazards={NUM_HAZARDS}."
        ),
        "interpretation": (
            "PASS (mixed direction): offline consolidation (ResidueField.integrate()) "
            "improves residue terrain correlation with hazard proximity. Provides weak "
            "diagnostic support for ARC-039's principle that offline processing improves "
            "map quality. Does not confirm the hippocampal-entorhinal mechanism (V4). "
            "FAIL weakens ARC-039: offline integration does not improve terrain quality, "
            "suggesting either the V3 residue.integrate() proxy is too weak, or the "
            "consolidation benefit requires the full grid-cell circuit (V4). "
            "INCONCLUSIVE: insufficient harm events or residue not populated."
        ),
        "per_seed_results": per_seed,
        "supersedes":     None,
        "dry_run":        dry_run,
        "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
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
        print(f"[EXQ-214] Result written to: {out_path}", flush=True)
    else:
        print("[EXQ-214] dry_run: result NOT written.", flush=True)

    return pack


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = run_experiment(dry_run=dry_run)
    print(f"[EXQ-214] Done. outcome={result['outcome']}", flush=True)
