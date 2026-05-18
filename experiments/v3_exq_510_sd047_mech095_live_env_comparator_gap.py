#!/opt/local/bin/python3
"""V3-EXQ-510 -- MECH-095 agency-detection comparator on the live multi-source env.

Successor to V3-EXQ-506 (substrate-readiness probe with synthetic data, FAIL on
C1-C3 with C4-only PASS, 2026-05-03 -- the C4-only-PASS substrate-ceiling
signature that motivated SD-047). SD-047 substrate landed 2026-05-03; the
substrate-readiness diagnostic V3-EXQ-509 PASSed (calibration ratio 2.03 in
target band [0.5, 2.5], 2026-05-03). This experiment is the behavioural
validation: the same E2_harm_s + SD-013 interventional comparator pipeline
from V3-EXQ-506, but with data from the live CausalGridWorldV2 across the four
pre-registered noise arms (OFF / 0.25x / 1.0x / 4.0x).

Claims tested
-------------
- MECH-095 (tpj.agency_detection_comparator) -- substrate-ceiling lift
- SD-047 (environment.multi_source_dynamics) -- behavioural validation

Why this experiment exists
--------------------------
SD-047's pre-registered prediction is that the V3-EXQ-506 C1-C3 FAIL pattern
flips to PASS at ARM_2 (default calibration), validating the substrate-ceiling
diagnosis on MECH-095 (Asai 2016 non-monotonic agency S/N). The Woo & Spelke
2023 falsifier branch routes MECH-095 from substrate_ceiling to
substrate_conditional (V4 multi-agent dependency) on flat-FAIL across all four
arms. The experiment must produce one of five distinct signatures so that each
maps to a distinct architectural conclusion.

Pipeline
--------
For each ARM in [ARM_0_off, ARM_1_low_0p25, ARM_2_default, ARM_3_high_4p0]:
  For each seed in SEEDS:
    1. Build CausalGridWorldV2 with multi_source_dynamics_enabled at the arm's
       intensity_scale; identical seed across arms within a given seed-row so
       the env stochastic process is the only manipulated variable.
    2. Initialize a frozen-random HarmEncoder (51 -> 32) with the per-seed RNG
       so encoder weights are identical across arms within a seed-row.
    3. Roll out a uniform-random-policy agent for N_COLLECT_TICKS, recording
       per-tick (harm_obs_t, action_t, harm_obs_{t+1}, transition_type,
       multi_source_n_env_events). Encode harm_obs through the frozen
       HarmEncoder to produce z_harm_s tensors.
    4. Tag each transition by causal type:
         agent_caused      -- transition_type in {"agent_caused_hazard",
                              "resource", "hazard_approach", "benefit_approach"}
                              AND multi_source_n_env_events == 0
         env_caused        -- action == 4 (stay) AND
                              multi_source_n_env_events > 0
         agent_collateral  -- action != 4 AND transition_type == "none" AND
                              multi_source_n_env_events > 0 (agent moved but
                              env caused the change)
         env_correlated    -- shuffle: env_caused/collateral transitions paired
                              with random sampled actions (decouples action
                              label from env-driven transition)
    5. Train E2_harm_s on agent_caused only with SD-013 interventional loss for
       N_TRAIN_STEPS steps. HarmEncoder is frozen (z_harm_s is .detach()ed
       before forward).
    6. Measure per-condition counterfactual gap = mean ||E2(z, a_actual) -
       E2(z, a_cf)|| with random a_cf != a_actual (same metric as V3-EXQ-506).
    7. Per-seed C1/C2/C3/C4 (gap_agent / gap_env >= 1.5 etc; gap_agent >=
       interventional_margin).
  Aggregate per-arm: 2/3 seeds PASS each criterion -> arm-level PASS-on-criterion.

Acceptance criteria (per arm)
-----------------------------
  C1: gap_agent / gap_env_caused      >= 1.5
  C2: gap_agent / gap_agent_collateral >= 1.5
  C3: gap_agent / gap_env_correlated  >= 1.5
  C4: gap_agent                        >= INTERVENTIONAL_MARGIN (0.1)
arm_PASS_<criterion> = at least 2/3 seeds satisfy the criterion.
arm_PASS_all = PASS on C1 AND C2 AND C3 AND C4.

Interpretation grid (pre-registered, from SD-047 design doc)
-------------------------------------------------------------
| ARM_0       | ARM_2       | Sweep shape           | Reading                                       |
|-------------|-------------|-----------------------|-----------------------------------------------|
| C1-C3 FAIL  | all PASS    | any                   | OUTCOME_VALIDATED  (overall PASS)             |
| C1-C3 FAIL  | all PASS    | inverted U at ARM_2   | OUTCOME_INVERTED_U (overall PASS, calibration)|
| C1-C3 FAIL  | all FAIL    | flat all-FAIL         | OUTCOME_WOO_SPELKE (route MECH-095 -> V4)     |
| C1-C3 FAIL  | partial     | peak at ARM_1 / ARM_3 | OUTCOME_MISCALIBRATED (recalibrate per-source)|
| all PASS    | all PASS    | flat all-PASS         | OUTCOME_OPPOSITE_ARTEFACT (revisit EXQ-506)   |
| any other                                          | OUTCOME_INDETERMINATE                         |

PASS = OUTCOME in {VALIDATED, INVERTED_U}.

Top-level outcome and per-claim evidence direction
--------------------------------------------------
  PASS (VALIDATED, INVERTED_U):
    overall outcome=PASS, evidence_direction=supports for both MECH-095 + SD-047.
  WOO_SPELKE:
    overall outcome=FAIL, evidence_direction=mixed.
    MECH-095 evidence_direction=mixed (architectural reroute to V4, not lift in V3).
    SD-047 evidence_direction=weakens (env enrichment alone insufficient).
  MISCALIBRATED:
    overall outcome=FAIL, evidence_direction=inconclusive (calibration issue, not
    a substrate-architecture verdict).
  OPPOSITE_ARTEFACT:
    overall outcome=FAIL, evidence_direction=mixed (substrate-ceiling diagnosis
    was wrong in opposite direction; revisit EXQ-506 for non-substrate confound).
  INDETERMINATE:
    overall outcome=FAIL, evidence_direction=inconclusive.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_510_sd047_mech095_live_env_comparator_gap.py
  /opt/local/bin/python3 experiments/v3_exq_510_sd047_mech095_live_env_comparator_gap.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.latent.stack import HarmEncoder  # noqa: E402
from ree_core.predictors.e2_harm_s import E2HarmSConfig, E2HarmSForward  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_510_sd047_mech095_live_env_comparator_gap"
CLAIM_IDS = ["MECH-095", "SD-047"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = (42, 43, 44)
GRID_SIZE = 8
N_HAZARDS = 3
N_RESOURCES = 3

# Source defaults at ARM_2 (intensity_scale=1.0); same as V3-EXQ-509.
WEATHER_SUPER_CELLS = 4
WEATHER_ALPHA_AR1 = 0.95
WEATHER_SIGMA = 0.10
TRANSIENT_P_APPEAR = 5e-3
TRANSIENT_P_DISAPPEAR = 0.10
N_DRIFT_SOURCES = 2
DRIFT_POLICY = "random_walk"

# 4-arm sweep specification (matches V3-EXQ-509).
ARMS: List[Tuple[str, bool, float]] = [
    ("ARM_0_off",        False, 1.0),
    ("ARM_1_low_0p25",   True,  0.25),
    ("ARM_2_default",    True,  1.0),
    ("ARM_3_high_4p0",   True,  4.0),
]

# Encoder / comparator config (matches V3-EXQ-506 dims).
HARM_OBS_DIM = 51    # CausalGridWorldV2 emits harm_obs of dim 51 (SD-010 layout).
Z_HARM_DIM = 32
ACTION_DIM = 5       # CausalGridWorldV2 has 5 actions (up/down/left/right/stay).
HIDDEN_DIM = 128
LR = 5e-4
INTERVENTIONAL_MARGIN = 0.1
INTERVENTIONAL_FRACTION = 0.3

# Data collection / training defaults.
N_COLLECT_TICKS = 2000
N_TRAIN_STEPS = 800
BATCH = 64
N_EVAL_PER_CONDITION = 256

# Acceptance thresholds (per V3-EXQ-506 pre-registration).
C1_MIN_RATIO = 1.5
C2_MIN_RATIO = 1.5
C3_MIN_RATIO = 1.5
C4_MIN_AGENT_GAP = INTERVENTIONAL_MARGIN
PASS_FRACTION_REQUIRED = 2.0 / 3.0  # >= 2/3 seeds PASS per criterion.

# Transition-type set treated as agent-caused.
AGENT_CAUSED_TRANSITIONS = {
    "agent_caused_hazard",
    "resource",
    "hazard_approach",
    "benefit_approach",
}


def _action_onehot(idx_batch: torch.Tensor, n: int = ACTION_DIM) -> torch.Tensor:
    return F.one_hot(idx_batch, num_classes=n).float()


def collect_arm_data(
    seed: int, arm_name: str, enabled: bool, scale: float, n_ticks: int,
    encoder: HarmEncoder,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Roll out a uniform-random agent in CausalGridWorldV2 at this arm's intensity.

    Returns a dict mapping condition name -> {"z": [N, Z_HARM_DIM],
    "a_idx": [N], "z_next": [N, Z_HARM_DIM]}.
    """
    rng = np.random.default_rng(seed + hash(arm_name) % 2 ** 16)
    env = CausalGridWorldV2(
        size=GRID_SIZE, num_hazards=N_HAZARDS, num_resources=N_RESOURCES,
        seed=seed,
        multi_source_dynamics_enabled=enabled,
        multi_source_intensity_scale=scale,
        weather_field_enabled=enabled,
        weather_super_cells=WEATHER_SUPER_CELLS,
        weather_alpha_ar1=WEATHER_ALPHA_AR1,
        weather_sigma=WEATHER_SIGMA,
        transient_events_enabled=enabled,
        transient_p_appear=TRANSIENT_P_APPEAR,
        transient_p_disappear=TRANSIENT_P_DISAPPEAR,
        background_drift_enabled=enabled,
        n_drift_sources=N_DRIFT_SOURCES,
        drift_policy=DRIFT_POLICY,
    )
    flat, obs = env.reset()
    harm_obs_prev = obs["harm_obs"].clone()

    agent_caused: List[Tuple[torch.Tensor, int, torch.Tensor]] = []
    env_caused: List[Tuple[torch.Tensor, int, torch.Tensor]] = []
    agent_collateral: List[Tuple[torch.Tensor, int, torch.Tensor]] = []
    for _ in range(n_ticks):
        a_idx = int(rng.integers(0, ACTION_DIM))
        a = torch.tensor(a_idx, dtype=torch.long)
        flat, harm_signal, done, info, obs = env.step(a)
        harm_obs_curr = obs["harm_obs"].clone()
        ttype = info["transition_type"]
        # Aggregate env-event signal across all sources: legacy _drift_hazards
        # (env_drift_occurred), SD-029 scheduled injection (external_hazard_injected),
        # and SD-047 multi-source dynamics (multi_source_n_env_events > 0).
        # ARM_0 OFF still produces legacy env_drift events every env_drift_interval
        # ticks, so the comparator has env-side samples to learn the not-self
        # baseline (the substrate-ceiling diagnosis is precisely that this baseline
        # is too thin in ARM_0; ARM_2 thickens it via SD-047).
        env_event_fired = (
            int(info["multi_source_n_env_events"]) > 0
            or bool(info.get("external_hazard_injected", False))
            or bool(info.get("env_drift_occurred", False))
        )
        action_is_stay = (a_idx == 4)
        is_agent_caused_tt = ttype in AGENT_CAUSED_TRANSITIONS
        # Tag taxonomy:
        #   agent_caused     -- agent's action proximally produced a harm/benefit
        #                       transition AND no env event same tick.
        #   env_caused       -- agent stayed AND env event fired.
        #   agent_collateral -- agent moved, no agent-caused transition_type, env
        #                       event fired (action concurrent but not the cause).
        #   ambiguous ticks (e.g. agent-caused transition_type + env event same
        #   tick) are excluded from all conditions.
        if is_agent_caused_tt and not env_event_fired:
            agent_caused.append((harm_obs_prev, a_idx, harm_obs_curr))
        elif action_is_stay and env_event_fired:
            env_caused.append((harm_obs_prev, a_idx, harm_obs_curr))
        elif (not action_is_stay) and (not is_agent_caused_tt) and env_event_fired:
            agent_collateral.append((harm_obs_prev, a_idx, harm_obs_curr))
        harm_obs_prev = harm_obs_curr
        if done:
            flat, obs = env.reset()
            harm_obs_prev = obs["harm_obs"].clone()

    def _to_batch(rows: List[Tuple[torch.Tensor, int, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not rows:
            return {
                "z": torch.zeros(0, Z_HARM_DIM),
                "a_idx": torch.zeros(0, dtype=torch.long),
                "z_next": torch.zeros(0, Z_HARM_DIM),
                "n": 0,
            }
        h_prev = torch.stack([r[0] for r in rows])
        h_curr = torch.stack([r[2] for r in rows])
        a_idx = torch.tensor([r[1] for r in rows], dtype=torch.long)
        with torch.no_grad():
            z = encoder(h_prev).detach()
            z_next = encoder(h_curr).detach()
        return {"z": z, "a_idx": a_idx, "z_next": z_next, "n": len(rows)}

    out = {
        "agent_caused": _to_batch(agent_caused),
        "env_caused": _to_batch(env_caused),
        "agent_collateral": _to_batch(agent_collateral),
    }

    # env_correlated: take env_caused's z-pairs, pair with random resampled
    # actions. Decouples action label from env-driven transition; predicts
    # collapsed gap because action carries no informational signal.
    if out["env_caused"]["n"] > 0:
        n_corr = out["env_caused"]["n"]
        a_corr_idx = torch.tensor(rng.integers(0, ACTION_DIM, size=n_corr), dtype=torch.long)
        out["env_correlated"] = {
            "z": out["env_caused"]["z"].clone(),
            "a_idx": a_corr_idx,
            "z_next": out["env_caused"]["z_next"].clone(),
            "n": n_corr,
        }
    else:
        out["env_correlated"] = {
            "z": torch.zeros(0, Z_HARM_DIM),
            "a_idx": torch.zeros(0, dtype=torch.long),
            "z_next": torch.zeros(0, Z_HARM_DIM),
            "n": 0,
        }
    return out


def train_e2_on_agent_caused(harm_fwd: E2HarmSForward, batches: Dict[str, torch.Tensor],
                               n_steps: int, gen: torch.Generator) -> List[float]:
    """Train E2_harm_s on agent_caused (z, a, z_next) batches with SD-013 interventional."""
    n = batches["n"]
    if n == 0:
        return []
    opt = torch.optim.Adam(harm_fwd.parameters(), lr=LR)
    train_losses: List[float] = []
    z_all = batches["z"]
    a_all = batches["a_idx"]
    z_next_all = batches["z_next"]
    harm_fwd.train()
    for step in range(n_steps):
        # Sample a batch with replacement from agent_caused pool.
        idx = torch.randint(0, n, (BATCH,), generator=gen)
        z = z_all[idx]
        a = _action_onehot(a_all[idx])
        z_next = z_next_all[idx]
        z_pred = harm_fwd.forward(z, a)
        loss = harm_fwd.compute_loss(z_pred, z_next.detach())
        cf_idx = (a_all[idx] + 1 + torch.randint(0, ACTION_DIM - 1, (BATCH,), generator=gen)) % ACTION_DIM
        a_cf = _action_onehot(cf_idx)
        n_int = max(1, int(BATCH * INTERVENTIONAL_FRACTION))
        loss_int = harm_fwd.compute_interventional_loss(z[:n_int], a[:n_int], a_cf[:n_int])
        total = loss + loss_int
        opt.zero_grad()
        total.backward()
        opt.step()
        if step % max(1, n_steps // 4) == 0:
            train_losses.append(float(total.item()))
    return train_losses


def measure_gap(harm_fwd: E2HarmSForward, batch: Dict[str, torch.Tensor],
                 gen: torch.Generator):
    """Mean ||E2(z, a_actual) - E2(z, a_cf)|| with random a_cf != a_actual.

    Returns None when the condition pool is empty (criterion not evaluable),
    distinct from a measurable gap of 0.0.
    """
    n = batch["n"]
    if n == 0:
        return None
    eval_n = min(N_EVAL_PER_CONDITION, n)
    idx = torch.randperm(n, generator=gen)[:eval_n]
    z = batch["z"][idx]
    a_actual = _action_onehot(batch["a_idx"][idx])
    actual_idx = batch["a_idx"][idx]
    cf_idx = (actual_idx + 1 + torch.randint(0, ACTION_DIM - 1, (eval_n,), generator=gen)) % ACTION_DIM
    a_cf = _action_onehot(cf_idx)
    harm_fwd.eval()
    with torch.no_grad():
        z_actual = harm_fwd.forward(z, a_actual)
        z_cf = harm_fwd.counterfactual_forward(z, a_cf)
    gap = (z_actual - z_cf).norm(dim=-1).mean().item()
    return float(gap)


def run_arm_seed(seed: int, arm_name: str, enabled: bool, scale: float,
                  n_collect: int, n_train: int) -> Dict:
    """One (arm, seed) cell: collect, train, measure four gaps + per-seed C1-C4."""
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Frozen-random HarmEncoder: same RNG seed across arms within a seed-row, so
    # encoder weights are identical across arms; isolates the env contribution.
    encoder = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Per-arm fresh comparator.
    cfg = E2HarmSConfig(
        use_e2_harm_s_forward=True,
        z_harm_dim=Z_HARM_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        learning_rate=LR,
        use_interventional=True,
        interventional_fraction=INTERVENTIONAL_FRACTION,
        interventional_margin=INTERVENTIONAL_MARGIN,
    )
    harm_fwd = E2HarmSForward(cfg)

    batches = collect_arm_data(seed, arm_name, enabled, scale, n_collect, encoder)
    n_per_cond = {k: batches[k]["n"] for k in ("agent_caused", "env_caused", "agent_collateral", "env_correlated")}

    # Skip training when agent_caused pool is empty.
    if batches["agent_caused"]["n"] == 0:
        return {
            "seed": seed, "arm": arm_name, "enabled": enabled, "scale": scale,
            "n_per_condition": n_per_cond,
            "gap_agent_caused": None, "gap_env_caused": None,
            "gap_agent_collateral": None, "gap_env_correlated": None,
            "ratio_agent_over_env": None, "ratio_agent_over_collateral": None,
            "ratio_agent_over_correlated": None,
            "c1_pass": False, "c2_pass": False, "c3_pass": False, "c4_pass": False,
            "c1_evaluable": False, "c2_evaluable": False,
            "c3_evaluable": False, "c4_evaluable": False,
            "train_loss_first": None, "train_loss_last": None,
            "skip_reason": "empty_agent_caused_pool",
        }

    train_losses = train_e2_on_agent_caused(harm_fwd, batches["agent_caused"],
                                            n_train, gen)
    gap_agent = measure_gap(harm_fwd, batches["agent_caused"], gen)
    gap_env = measure_gap(harm_fwd, batches["env_caused"], gen)
    gap_collateral = measure_gap(harm_fwd, batches["agent_collateral"], gen)
    gap_correlated = measure_gap(harm_fwd, batches["env_correlated"], gen)

    # gap_X is None when the condition pool is empty (criterion not evaluable).
    # PASS only when gap_agent is measurable AND the comparison gap is measurable
    # AND the ratio meets the threshold. Not-evaluable -> False, distinguishable
    # in the manifest via the per-condition counts.
    def _ratio(num, den):
        if num is None or den is None:
            return None
        return num / max(1e-6, den)
    ratio_env = _ratio(gap_agent, gap_env)
    ratio_col = _ratio(gap_agent, gap_collateral)
    ratio_corr = _ratio(gap_agent, gap_correlated)

    c1_pass = ratio_env is not None and ratio_env >= C1_MIN_RATIO
    c2_pass = ratio_col is not None and ratio_col >= C2_MIN_RATIO
    c3_pass = ratio_corr is not None and ratio_corr >= C3_MIN_RATIO
    c4_pass = gap_agent is not None and gap_agent >= C4_MIN_AGENT_GAP

    return {
        "seed": seed,
        "arm": arm_name,
        "enabled": enabled,
        "scale": scale,
        "n_per_condition": n_per_cond,
        "gap_agent_caused": gap_agent,
        "gap_env_caused": gap_env,
        "gap_agent_collateral": gap_collateral,
        "gap_env_correlated": gap_correlated,
        "ratio_agent_over_env": ratio_env,
        "ratio_agent_over_collateral": ratio_col,
        "ratio_agent_over_correlated": ratio_corr,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "c4_pass": c4_pass,
        "c1_evaluable": ratio_env is not None,
        "c2_evaluable": ratio_col is not None,
        "c3_evaluable": ratio_corr is not None,
        "c4_evaluable": gap_agent is not None,
        "train_loss_first": train_losses[0] if train_losses else None,
        "train_loss_last": train_losses[-1] if train_losses else None,
        "skip_reason": None,
    }


def aggregate_arm(per_seed_rows: List[Dict]) -> Dict:
    """Per-arm aggregation: 2/3 seeds PASS each criterion -> arm PASS-on-criterion."""
    n = len(per_seed_rows)
    required = math.ceil(n * PASS_FRACTION_REQUIRED)
    c1 = sum(1 for r in per_seed_rows if r["c1_pass"])
    c2 = sum(1 for r in per_seed_rows if r["c2_pass"])
    c3 = sum(1 for r in per_seed_rows if r["c3_pass"])
    c4 = sum(1 for r in per_seed_rows if r["c4_pass"])
    arm_c1 = c1 >= required
    arm_c2 = c2 >= required
    arm_c3 = c3 >= required
    arm_c4 = c4 >= required
    arm_all = arm_c1 and arm_c2 and arm_c3 and arm_c4
    arm_c1_c3 = arm_c1 and arm_c2 and arm_c3
    def _mean_skip_none(key):
        vs = [r[key] for r in per_seed_rows if isinstance(r[key], (int, float))]
        return float(np.mean(vs)) if vs else None
    return {
        "n_seeds": n, "min_seeds_required": required,
        "n_c1": c1, "n_c2": c2, "n_c3": c3, "n_c4": c4,
        "arm_pass_c1": arm_c1, "arm_pass_c2": arm_c2,
        "arm_pass_c3": arm_c3, "arm_pass_c4": arm_c4,
        "arm_pass_all": arm_all, "arm_pass_c1_c2_c3": arm_c1_c3,
        "mean_gap_agent": _mean_skip_none("gap_agent_caused"),
        "mean_ratio_env": _mean_skip_none("ratio_agent_over_env"),
        "mean_ratio_col": _mean_skip_none("ratio_agent_over_collateral"),
        "mean_ratio_corr": _mean_skip_none("ratio_agent_over_correlated"),
    }


def classify_outcome(per_arm: Dict[str, Dict]) -> Dict:
    """Apply the SD-047 design-doc interpretation grid to arm aggregates.

    Returns dict with outcome (str), overall_pass (bool), evidence_direction
    (str), and per_claim direction dict.
    """
    a0 = per_arm["ARM_0_off"]
    a1 = per_arm["ARM_1_low_0p25"]
    a2 = per_arm["ARM_2_default"]
    a3 = per_arm["ARM_3_high_4p0"]

    arm_passes_all = {n: a["arm_pass_all"] for n, a in per_arm.items()}
    arm_c1_c3_fail = {n: not a["arm_pass_c1_c2_c3"] for n, a in per_arm.items()}

    # OUTCOME branches.
    if all(arm_passes_all.values()):
        outcome = "OPPOSITE_ARTEFACT"
    elif all(arm_c1_c3_fail.values()):
        outcome = "WOO_SPELKE"
    elif a0["arm_pass_c1_c2_c3"] is False and a2["arm_pass_all"]:
        # Standard validated case. Check inverted-U sub-shape.
        scores = [a1["n_c1"] + a1["n_c2"] + a1["n_c3"],
                  a2["n_c1"] + a2["n_c2"] + a2["n_c3"],
                  a3["n_c1"] + a3["n_c2"] + a3["n_c3"]]
        if scores[1] > scores[0] and scores[1] > scores[2]:
            outcome = "INVERTED_U"
        else:
            outcome = "VALIDATED"
    elif a0["arm_pass_c1_c2_c3"] is False and (a1["arm_pass_all"] or a3["arm_pass_all"]) and not a2["arm_pass_all"]:
        outcome = "MISCALIBRATED"
    else:
        outcome = "INDETERMINATE"

    if outcome in ("VALIDATED", "INVERTED_U"):
        overall_pass = True
        evidence_direction = "supports"
        per_claim = {"MECH-095": "supports", "SD-047": "supports"}
    elif outcome == "WOO_SPELKE":
        overall_pass = False
        evidence_direction = "mixed"
        per_claim = {"MECH-095": "mixed", "SD-047": "weakens"}
    elif outcome == "MISCALIBRATED":
        overall_pass = False
        evidence_direction = "inconclusive"
        per_claim = {"MECH-095": "inconclusive", "SD-047": "inconclusive"}
    elif outcome == "OPPOSITE_ARTEFACT":
        overall_pass = False
        evidence_direction = "mixed"
        per_claim = {"MECH-095": "mixed", "SD-047": "mixed"}
    else:
        overall_pass = False
        evidence_direction = "inconclusive"
        per_claim = {"MECH-095": "inconclusive", "SD-047": "inconclusive"}

    return {
        "outcome_branch": outcome,
        "overall_pass": overall_pass,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": per_claim,
    }


def main(dry_run: bool = False) -> int:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    if dry_run:
        n_collect = 300
        n_train = 60
        seeds = (SEEDS[0],)
    else:
        n_collect = N_COLLECT_TICKS
        n_train = N_TRAIN_STEPS
        seeds = SEEDS

    t0 = time.time()
    rows: List[Dict] = []
    for seed in seeds:
        for arm_name, enabled, scale in ARMS:
            r = run_arm_seed(seed, arm_name, enabled, scale, n_collect, n_train)
            rows.append(r)
            ncond = r["n_per_condition"]
            def _fmt(v, prec=3):
                return f"{v:.{prec}f}" if isinstance(v, (int, float)) else "n/a"
            print(
                f"  seed={seed} arm={arm_name:<18} "
                f"n[ag/env/col/corr]={ncond['agent_caused']}/{ncond['env_caused']}/"
                f"{ncond['agent_collateral']}/{ncond['env_correlated']} "
                f"gap_agent={_fmt(r['gap_agent_caused'])} "
                f"r_env/col/corr={_fmt(r['ratio_agent_over_env'], 2)}/"
                f"{_fmt(r['ratio_agent_over_collateral'], 2)}/"
                f"{_fmt(r['ratio_agent_over_correlated'], 2)} "
                f"C1/C2/C3/C4={int(r['c1_pass'])}{int(r['c2_pass'])}"
                f"{int(r['c3_pass'])}{int(r['c4_pass'])}",
                flush=True,
            )
    elapsed = time.time() - t0

    # Aggregate per arm.
    per_arm: Dict[str, Dict] = {}
    for arm_name, enabled, scale in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm_name]
        agg = aggregate_arm(arm_rows)
        agg.update({"arm": arm_name, "enabled": enabled, "scale": scale})
        per_arm[arm_name] = agg

    classification = classify_outcome(per_arm)
    outcome = "PASS" if classification["overall_pass"] else "FAIL"

    print(f"[{EXPERIMENT_TYPE}] per-arm aggregates:", flush=True)
    for arm_name in ("ARM_0_off", "ARM_1_low_0p25", "ARM_2_default", "ARM_3_high_4p0"):
        a = per_arm[arm_name]
        mga = a["mean_gap_agent"]
        mga_s = f"{mga:.3f}" if isinstance(mga, (int, float)) else "n/a"
        print(
            f"  {arm_name:<18} c1/c2/c3/c4_pass={a['arm_pass_c1']}/{a['arm_pass_c2']}/"
            f"{a['arm_pass_c3']}/{a['arm_pass_c4']} all={a['arm_pass_all']} "
            f"mean_gap_agent={mga_s}",
            flush=True,
        )
    print(f"[{EXPERIMENT_TYPE}] interpretation:", flush=True)
    print(f"  outcome_branch={classification['outcome_branch']}", flush=True)
    print(f"  evidence_direction={classification['evidence_direction']}", flush=True)
    print(f"  per_claim={classification['evidence_direction_per_claim']}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s", flush=True)

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.", flush=True)
        return 0

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "outcome": outcome,
        "evidence_direction": classification["evidence_direction"],
        "evidence_direction_per_claim": classification["evidence_direction_per_claim"],
        "outcome_branch": classification["outcome_branch"],
        "elapsed_seconds": elapsed,
        "n_seeds": len(seeds),
        "n_collect_ticks": n_collect,
        "n_train_steps": n_train,
        "per_arm": list(per_arm.values()),
        "per_seed_per_arm": rows,
        "thresholds": {
            "c1_min_ratio": C1_MIN_RATIO,
            "c2_min_ratio": C2_MIN_RATIO,
            "c3_min_ratio": C3_MIN_RATIO,
            "c4_min_agent_gap": C4_MIN_AGENT_GAP,
            "pass_fraction_required": PASS_FRACTION_REQUIRED,
        },
    }
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Result written to: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run))
