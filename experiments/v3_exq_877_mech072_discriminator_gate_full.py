#!/opt/local/bin/python3
"""
V3-EXQ-877 -- MECH-072: Full Discriminator-Gate Design (ARC-033 / E2_harm_s pathway)

experiment_purpose: evidence

Claim: MECH-072 (candidate, v3_pending)
  "Foreseeable-harm gating on residue accumulation reduces false attribution
  without degrading harm avoidance."

Background / gap this closes:
  V3-EXQ-213 PASS (2026-04-03) validated the CORE gating mechanism using a
  PROXY pathway: E3.harm_eval(E2.world_forward(z_world, a)) > GATE_THRESH.
  MECH-072's own evidence_quality_note is explicit that this proxy is valid
  but that "the full discriminator-gate design" additionally needs E2_harm_s
  (SD-011, ARC-033) -- a dedicated sensory-discriminative harm forward model
  operating on the harm stream itself, not on z_world. SD-011 is now `stable`
  and ARC-033 is `provisional` (V3-EXQ-525 PASS, 2026-05-06), so the substrate
  this claim was waiting on is built. No experiment has yet gated residue
  accumulation on the ARC-033 counterfactual discriminator -- V3-EXQ-431 and
  V3-EXQ-525 both validated the DISCRIMINATOR ITSELF (direction correctness,
  attribution gap) but neither one connects it to residue accumulation. This
  script is that connection: it is the "full discriminator-gate design" run.

Mechanism under test (the ARC-033 SD-003 counterfactual pipeline, per
  e2_harm_s.py / ARC-033 functional_restatement):
    z_harm_s        = HarmEncoder(harm_obs)                       [frozen after P0]
    z_harm_s_actual  = E2HarmSForward(z_harm_s, a_actual)
    z_harm_s_cf      = E2HarmSForward.counterfactual_forward(z_harm_s, a_stay)
    causal_sig       = E3.harm_eval_z_harm(z_harm_s_actual)
                        - E3.harm_eval_z_harm(z_harm_s_cf)
  causal_sig > 0 means the agent's actual action predicts MORE harm than the
  STAY counterfactual would have -- i.e. the harm was attributable to the
  agent's own action, not to the environment. This entirely replaces the
  V3-EXQ-213 proxy criterion (a magnitude threshold on E2.world_forward) with
  the validated ARC-033 counterfactual discriminator. Residue LOCATION still
  uses a frozen z_world embedding (as in EXQ-213) purely so false_attr_rate is
  computed the same way and is comparable across the two experiments; the
  z_world encoder is NOT trained anywhere in this script (nothing is trained
  on z_world, so the phased-training mandate does not apply to it -- only
  z_harm_s is trained, and IS phased below).

  Condition A: GATED_HARM_S
    Residue accumulates at a harm event ONLY IF causal_sig > GATE_THRESH_CF.
    Interpretation: "my action, not the environment, foreseeably caused this
    -- it goes on my record."

  Condition B: UNGATED
    Residue accumulates at ALL harm events regardless of causal_sig (old,
    unfiltered behavior -- identical semantics to V3-EXQ-213's UNGATED arm).

Phased training (MANDATORY -- z_harm_s is trained, unlike z_world here):
  P0 (HarmEncoder warmup): autoencoder + center-cell regression on harm_obs.
     No downstream loss -- matches V3-EXQ-525's proven P0.
  P1 (frozen HarmEncoder): train E2HarmSForward (forward-model MSE on
     .detach()ed z_harm_s) and E3.harm_eval_z_harm_head (balanced pos/neg
     harm classification) -- matches V3-EXQ-525's proven P1.
  P2 (evaluation): GATED vs UNGATED residue-gating comparison. No training.

SLEEP DRIVER: K=never (SleepLoopManager disabled during training; sleep called manually at eval)
  (No sleep loop is used in this script at all -- declared per the GAP-7
  convention because it is occasionally scanned for; N/A here.)

Pre-registered acceptance criteria:
  C1: false_attr_rate_GATED_HARM_S < false_attr_rate_UNGATED (each seed)
      Gating on the real discriminator must reduce false attribution -- the
      primary claim direction.
  C2: mean delta_false_attr (UNGATED - GATED_HARM_S) >= THRESH_C2_FAR_DELTA
      THRESH_C2_FAR_DELTA = 0.05 (matches V3-EXQ-213's own threshold, for a
      directly comparable bar between the proxy and the full-discriminator
      versions of this gate).
  C3: harm_rate_GATED_HARM_S <= harm_rate_UNGATED + THRESH_C3_HARM_TOLERANCE
      THRESH_C3_HARM_TOLERANCE = 0.03 (matches V3-EXQ-213).
  C4: n_env_caused_events >= THRESH_C4_MIN_ENV_EVENTS (each seed, data quality)
      THRESH_C4_MIN_ENV_EVENTS = 5 (matches V3-EXQ-213).
  C5: n_agent_caused_events >= THRESH_C5_MIN_AGENT_EVENTS (each seed, data quality)
      THRESH_C5_MIN_AGENT_EVENTS = 5 (matches V3-EXQ-213).
  C6: harm_forward_r2 >= THRESH_C6_HARM_QUALITY (E2_harm_s must be adequately
      trained for the discriminator to be meaningful)
      THRESH_C6_HARM_QUALITY = 0.05 (matches V3-EXQ-525's own C2 threshold for
      this exact pipeline/config family -- NOT the 0.914 figure from the
      EXQ-166e/195 harness, which uses different hyperparameters).
  C7: null_cf_max < THRESH_C7_NULL_CF (each seed, PIPELINE INTEGRITY)
      THRESH_C7_NULL_CF = 1e-5 (matches V3-EXQ-525's C5). When a_cf = a_actual,
      causal_sig must be exactly 0 (up to float noise). This is the first time
      causal_sig gates a real accumulation decision (rather than only being
      reported as a diagnostic, as in V3-EXQ-431/525), so a silently-broken
      discriminator could otherwise produce a result that looks like C1/C2
      passing or failing for the wrong reason.

Outcome:
  INCONCLUSIVE: NOT (C4 and C5 and C6 and C7)  [data/training/pipeline quality gate]
  PASS: C1 and C2 and C3 (given the quality gates pass)
  FAIL (weakens): NOT C1, but quality gates pass
  FAIL (mixed): otherwise

Evidence direction:
  "supports"     if PASS
  "weakens"      if quality gates pass but C1 fails in all seeds
  "mixed"        otherwise
  "inconclusive" if any quality gate fails

Decision scoring:
  PASS -> retain_ree (closes MECH-072's v3_pending -- the claim's own stated
          bar for full validation has been met)
  FAIL (weakens) -> hybridize (the real discriminator does not reduce false
          attribution at this scale, even though the proxy did -- MECH-072's
          full design needs revision)
  INCONCLUSIVE -> inconclusive

GOV-REUSE-1 check (Step 2.4): decisive readout is
  "mean_delta_false_attr computed via the ARC-033/E2_harm_s causal_sig gate".
  `reanalysis_query.py query --readout false_attr_rate --claim MECH-072` finds
  exactly ONE manifest carrying false_attr_rate at all (V3-EXQ-213,
  UNVERIFIABLE substrate_hash -- pre-standard, 2026-04-03) and it used the
  world_forward proxy pathway, not E2_harm_s -- not recoverable. No
  /failure-autopsy artifact braked MECH-072 (0 hits at RE_DERIVE_BRAKE
  threshold). New run required.

Estimated runtime: ~90 min on cloud (2 phases x 3 seeds x (P0+P1+eval) eps x steps)

Claims: MECH-072 (primary; the sole claim this experiment's implementation
  directly tests). ARC-033/SD-011/SD-003 are CONSUMED (already validated
  elsewhere -- V3-EXQ-525 PASS, V3-EXQ-329/431 PASS) and are not re-tested
  here, so they are intentionally NOT in claim_ids per the claim_ids accuracy
  rule.
"""

import copy
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder
from ree_core.predictors.e2_harm_s import E2HarmSConfig, E2HarmSForward
from ree_core.predictors.e3_selector import E3TrajectorySelector
from ree_core.residue.field import ResidueField
from ree_core.utils.config import E3Config, ResidueConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_877_mech072_discriminator_gate_full"
CLAIM_IDS          = ["MECH-072"]
EXPERIMENT_PURPOSE = "evidence"
SEEDS = [42, 7, 13]

# ---------------------------------------------------------------------------
# Environment (same family/params as V3-EXQ-213 for direct comparability)
# ---------------------------------------------------------------------------
ENV_SIZE      = 10
NUM_HAZARDS   = 3
NUM_RESOURCES = 3
HAZARD_HARM   = 0.5
ACTION_DIM    = 5       # up/down/left/right/stay
STAY_ACTION_IDX = 4     # CausalGridWorld ACTIONS[4] = (0, 0)
HARM_OBS_DIM  = 51      # hazard_field(25) + resource_field(25) + exposure(1)
WORLD_OBS_DIM = 250      # use_proxy_fields=True world_state flat dim

# Architecture dims
WORLD_DIM = 32   # residue LOCATION space only -- frozen, untrained encoder
Z_HARM_DIM = 32  # HarmEncoder / E2HarmSForward -- the trained discriminator stream

# Training
LR_HARM_ENC  = 1e-3
LR_HARM_FWD  = 1e-3
LR_HARM_EVAL = 1e-4
P0_EPISODES  = 100    # HarmEncoder warmup
P1_EPISODES  = 200    # E2HarmSForward + harm_eval_z_harm training (frozen encoder)
STEPS_PER_TRAIN_EP = 150

# Eval
N_EVAL_EPISODES  = 100
STEPS_PER_EVAL_EP = 150

# Gating threshold on causal_sig (ARC-033 discriminator). Matches V3-EXQ-525's
# own C1 attribution_gap threshold for this pipeline family -- NOT derived
# from this run's own statistics.
GATE_THRESH_CF = 0.002

# Pre-registered thresholds
THRESH_C2_FAR_DELTA        = 0.05
THRESH_C3_HARM_TOLERANCE   = 0.03
THRESH_C4_MIN_ENV_EVENTS   = 5
THRESH_C5_MIN_AGENT_EVENTS = 5
THRESH_C6_HARM_QUALITY     = 0.05
THRESH_C7_NULL_CF          = 1e-5

# Env-caused / agent-caused transition_type labels (CausalGridWorldV2)
ENV_CAUSED   = {"env_caused_hazard", "contaminated"}
AGENT_CAUSED = {"agent_caused_hazard"}


# ---------------------------------------------------------------------------
# Frozen world-location encoder (residue LOCATION only -- never trained)
# ---------------------------------------------------------------------------

class FrozenWorldLocationEncoder(nn.Module):
    """Untrained, frozen projection world_state -> residue-field location space.

    Nothing in this script trains a head on z_world, so the phased-training
    mandate (P0/P1/P2 for z_world/z_harm/encoder outputs) does not apply to
    it -- it is a fixed random projection used only so ResidueField has a
    location to accumulate at, matching V3-EXQ-213's residue bookkeeping.
    The mechanism under test here is exclusively the GATING CRITERION
    (causal_sig from the real ARC-033 discriminator vs the old world_forward
    proxy) -- z_world fidelity is deliberately not part of what's varied.
    """
    def __init__(self, in_dim: int = WORLD_OBS_DIM, out_dim: int = WORLD_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.net(x)


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=ENV_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=HAZARD_HARM,
        use_proxy_fields=True,
    )


def _onehot(idx: int, n: int) -> torch.Tensor:
    v = torch.zeros(1, n)
    v[0, idx] = 1.0
    return v


def build_modules(seed: int) -> dict:
    torch.manual_seed(seed)

    world_enc = FrozenWorldLocationEncoder()

    harm_enc = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)

    harm_fwd_cfg = E2HarmSConfig(
        use_e2_harm_s_forward=True,
        z_harm_dim=Z_HARM_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=128,
        learning_rate=LR_HARM_FWD,
    )
    harm_fwd = E2HarmSForward(harm_fwd_cfg)

    res_cfg = ResidueConfig(
        world_dim=WORLD_DIM,
        hidden_dim=32,
        accumulation_rate=0.2,
        num_basis_functions=32,
        kernel_bandwidth=1.0,
    )
    residue_field = ResidueField(res_cfg)

    # E3Config has no z_harm_dim field of its own; E3TrajectorySelector falls
    # back to world_dim for harm_eval_z_harm_head's input dim (see
    # e3_selector.py L283-291). Setting world_dim == Z_HARM_DIM here (both 32)
    # is what makes that fallback correct -- the same convention V3-EXQ-525/
    # 329/431 all rely on.
    e3_cfg = E3Config(
        world_dim=WORLD_DIM,
        hidden_dim=64,
        lambda_ethical=1.0,
        rho_residue=0.5,
    )
    e3 = E3TrajectorySelector(e3_cfg, residue_field=residue_field)

    return {
        "world_enc":     world_enc,
        "harm_enc":      harm_enc,
        "harm_fwd":      harm_fwd,
        "e3":            e3,
        "residue_field": residue_field,
    }


# ---------------------------------------------------------------------------
# P0: HarmEncoder warmup (autoencoder + center-cell regression)
# ---------------------------------------------------------------------------

def phase0_train_encoder(
    modules: dict, seed: int, n_episodes: int, steps_per_ep: int,
    total_train_eps: int, dry_run: bool = False,
) -> None:
    rng = random.Random(seed)
    harm_enc = modules["harm_enc"]
    device = next(harm_enc.parameters()).device

    harm_decoder = nn.Sequential(
        nn.Linear(harm_enc.z_harm_dim, 64),
        nn.ReLU(),
        nn.Linear(64, harm_enc.harm_obs_dim),
    ).to(device)
    optimizer = optim.Adam(
        list(harm_enc.parameters()) + list(harm_decoder.parameters()),
        lr=LR_HARM_ENC,
    )
    harm_enc.train()

    env = _make_env(seed)
    ae_loss_val = 0.0

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        for _ in range(steps_per_ep):
            harm_obs_t = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            if harm_obs_t.dim() == 1:
                harm_obs_t = harm_obs_t.unsqueeze(0)

            z_harm_s = harm_enc(harm_obs_t)
            recon = harm_decoder(z_harm_s)
            ae_loss = F.mse_loss(recon, harm_obs_t)
            # Center-cell regression (hazard_field_view center, idx 12) --
            # matches V3-EXQ-525's proven P0 shaping term.
            center_pred = z_harm_s[:, :1]
            ae_loss = ae_loss + 0.5 * F.mse_loss(center_pred, harm_obs_t[:, 12:13])

            optimizer.zero_grad()
            ae_loss.backward()
            optimizer.step()
            ae_loss_val = float(ae_loss.item())

            action_idx = rng.randint(0, ACTION_DIM - 1)
            _, _, done, _, obs_dict = env.step(action_idx)
            if done:
                break

        if (ep + 1) % 50 == 0 or ep == n_episodes - 1:
            print(
                f"  [train] seed={seed} ep {ep+1}/{total_train_eps}"
                f"  [P0 encoder] ae_loss={ae_loss_val:.5f}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# P1: freeze HarmEncoder, train E2HarmSForward + E3.harm_eval_z_harm_head
# ---------------------------------------------------------------------------

def phase1_train_heads(
    modules: dict, seed: int, n_episodes: int, steps_per_ep: int,
    p0_episodes: int, total_train_eps: int, dry_run: bool = False,
) -> Dict:
    harm_enc = modules["harm_enc"]
    harm_fwd = modules["harm_fwd"]
    e3       = modules["e3"]

    harm_enc.eval()
    for p in harm_enc.parameters():
        p.requires_grad_(False)

    harm_fwd.train()
    fwd_opt  = optim.Adam(harm_fwd.parameters(), lr=LR_HARM_FWD)
    eval_opt = optim.Adam(e3.harm_eval_z_harm_head.parameters(), lr=LR_HARM_EVAL)

    env = _make_env(seed + 1000)  # distinct stream from P0

    hf_data: List = []
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    rng = random.Random(seed + 1000)

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        z_harm_prev: Optional[torch.Tensor] = None
        a_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_ep):
            harm_obs_t = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            if harm_obs_t.dim() == 1:
                harm_obs_t = harm_obs_t.unsqueeze(0)

            with torch.no_grad():
                z_harm_s = harm_enc(harm_obs_t)

            action_idx = rng.randint(0, ACTION_DIM - 1)
            action = _onehot(action_idx, ACTION_DIM)
            _, harm_signal, done, info, obs_dict = env.step(action_idx)

            z_harm_detach = z_harm_s.detach()
            if harm_signal < 0:
                harm_buf_pos.append(z_harm_detach)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_harm_detach)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            if z_harm_prev is not None and a_prev is not None:
                hf_data.append((z_harm_prev, a_prev, z_harm_detach))
                if len(hf_data) > 5000:
                    hf_data = hf_data[-5000:]

            # Train E2HarmSForward (stop-grad target already detached above)
            if len(hf_data) >= 16:
                k = min(32, len(hf_data))
                idxs = torch.randperm(len(hf_data))[:k].tolist()
                zh_b  = torch.cat([hf_data[i][0] for i in idxs])
                a_b   = torch.cat([hf_data[i][1] for i in idxs])
                zh1_b = torch.cat([hf_data[i][2] for i in idxs])
                pred = harm_fwd(zh_b, a_b)
                loss = harm_fwd.compute_loss(pred, zh1_b)
                fwd_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(harm_fwd.parameters(), 0.5)
                fwd_opt.step()

            # Train E3.harm_eval_z_harm_head (balanced pos/neg classification,
            # over both directly-observed z_harm_s AND its one-step forward
            # prediction, matching V3-EXQ-525's proven P1)
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pi = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                ni = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zh_pos = torch.cat([harm_buf_pos[i] for i in pi])
                zh_neg = torch.cat([harm_buf_neg[i] for i in ni])

                with torch.no_grad():
                    a_rp = torch.zeros(k_pos, ACTION_DIM)
                    a_rp[torch.arange(k_pos), torch.randint(0, ACTION_DIM, (k_pos,))] = 1.0
                    a_rn = torch.zeros(k_neg, ACTION_DIM)
                    a_rn[torch.arange(k_neg), torch.randint(0, ACTION_DIM, (k_neg,))] = 1.0
                    zh_pos_pred = harm_fwd(zh_pos, a_rp)
                    zh_neg_pred = harm_fwd(zh_neg, a_rn)

                zh_all = torch.cat([zh_pos, zh_neg, zh_pos_pred, zh_neg_pred], dim=0)
                tgt = torch.cat([
                    torch.ones(k_pos, 1), torch.zeros(k_neg, 1),
                    torch.ones(k_pos, 1), torch.zeros(k_neg, 1),
                ], dim=0)
                pred_harm = e3.harm_eval_z_harm(zh_all)
                harm_loss = F.mse_loss(pred_harm, tgt)
                eval_opt.zero_grad()
                harm_loss.backward()
                nn.utils.clip_grad_norm_(e3.harm_eval_z_harm_head.parameters(), 0.5)
                eval_opt.step()

            z_harm_prev = z_harm_detach
            a_prev = action
            if done:
                break

        if (ep + 1) % 50 == 0 or ep == n_episodes - 1:
            print(
                f"  [train] seed={seed} ep {p0_episodes + ep + 1}/{total_train_eps}"
                f"  [P1 heads] replay_buf={len(hf_data)}",
                flush=True,
            )

    # harm_forward_r2 on a held-out tail of the replay buffer
    harm_forward_r2 = 0.0
    if len(hf_data) >= 20:
        n = len(hf_data)
        n_train = int(n * 0.8)
        with torch.no_grad():
            zh  = torch.cat([d[0] for d in hf_data])
            a   = torch.cat([d[1] for d in hf_data])
            zh1 = torch.cat([d[2] for d in hf_data])
            pred = harm_fwd(zh, a)
            ss_res = ((pred[n_train:] - zh1[n_train:]) ** 2).sum().item()
            ss_tot = ((zh1[n_train:] - zh1[n_train:].mean(0)) ** 2).sum().item()
            harm_forward_r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    n_pos = len(harm_buf_pos)
    n_neg = len(harm_buf_neg)
    label_balance = {
        "harm_eval_z_harm_train": {
            "train_pos_frac": n_pos / max(1, n_pos + n_neg),
        }
    }

    return {
        "harm_forward_r2": harm_forward_r2,
        "n_hf_data": len(hf_data),
        "label_balance": label_balance,
    }


# ---------------------------------------------------------------------------
# P2: eval -- GATED_HARM_S vs UNGATED residue-gating comparison
# ---------------------------------------------------------------------------

def eval_condition(condition: str, modules: dict, seed: int, dry_run: bool = False) -> dict:
    rng = random.Random(seed + 9999)
    n_eps   = 5  if dry_run else N_EVAL_EPISODES
    n_steps = 30 if dry_run else STEPS_PER_EVAL_EP

    env = _make_env(seed + 9999)

    eval_res_cfg = ResidueConfig(
        world_dim=WORLD_DIM, hidden_dim=32, accumulation_rate=0.2,
        num_basis_functions=32, kernel_bandwidth=1.0,
    )
    eval_residue = ResidueField(eval_res_cfg)

    world_enc = modules["world_enc"]
    harm_enc  = modules["harm_enc"]
    harm_fwd  = modules["harm_fwd"]
    e3        = modules["e3"]

    harm_enc.eval()
    harm_fwd.eval()
    e3.eval()

    harm_steps = 0
    total_steps = 0
    residue_env = 0.0
    residue_agent = 0.0
    n_env_events = 0
    n_agent_events = 0
    causal_sigs_env: List[float] = []
    causal_sigs_agent: List[float] = []
    null_cf_vals: List[float] = []

    a_stay = _onehot(STAY_ACTION_IDX, ACTION_DIM)

    for ep in range(n_eps):
        _, obs_dict = env.reset()

        for _ in range(n_steps):
            harm_obs_t = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            if harm_obs_t.dim() == 1:
                harm_obs_t = harm_obs_t.unsqueeze(0)
            world_state_t = obs_dict["world_state"].float()
            if world_state_t.dim() == 1:
                world_state_t = world_state_t.unsqueeze(0)

            action_idx = rng.randint(0, ACTION_DIM - 1)
            a_actual = _onehot(action_idx, ACTION_DIM)

            _, harm_signal, done, info, obs_dict_next = env.step(action_idx)
            harm_val = abs(float(harm_signal)) if harm_signal < 0 else 0.0
            transition_type = info.get("transition_type", "none")

            total_steps += 1
            if harm_signal < 0:
                harm_steps += 1

            if harm_val > 0.01:
                with torch.no_grad():
                    z_harm_s = harm_enc(harm_obs_t)
                    z_w = world_enc(world_state_t)

                    z_pred_actual = harm_fwd(z_harm_s, a_actual)
                    z_pred_cf     = harm_fwd.counterfactual_forward(z_harm_s, a_stay)
                    causal_sig = float(
                        (e3.harm_eval_z_harm(z_pred_actual)
                         - e3.harm_eval_z_harm(z_pred_cf)).item()
                    )

                    # C7 pipeline-integrity sanity: CF == actual -> causal_sig ~ 0.
                    z_pred_null = harm_fwd.counterfactual_forward(z_harm_s, a_actual)
                    null_sig = float(
                        (e3.harm_eval_z_harm(z_pred_actual)
                         - e3.harm_eval_z_harm(z_pred_null)).item()
                    )
                    null_cf_vals.append(abs(null_sig))

                is_agent_attributable = (causal_sig > GATE_THRESH_CF)

                should_accumulate = (
                    True if condition == "UNGATED" else is_agent_attributable
                )

                if should_accumulate:
                    eval_residue.accumulate(z_w, harm_magnitude=harm_val, hypothesis_tag=False)

                if transition_type in ENV_CAUSED:
                    n_env_events += 1
                    causal_sigs_env.append(causal_sig)
                    if should_accumulate:
                        residue_env += harm_val * eval_residue.config.accumulation_rate
                elif transition_type in AGENT_CAUSED:
                    n_agent_events += 1
                    causal_sigs_agent.append(causal_sig)
                    if should_accumulate:
                        residue_agent += harm_val * eval_residue.config.accumulation_rate

            obs_dict = obs_dict_next
            if done:
                break

    harm_rate = harm_steps / max(total_steps, 1)
    total_residue_accumulated = residue_env + residue_agent
    false_attr_rate = (
        residue_env / total_residue_accumulated
        if total_residue_accumulated > 1e-8 else 0.0
    )
    null_cf_max = float(max(null_cf_vals)) if null_cf_vals else 0.0

    return {
        "harm_rate": harm_rate,
        "harm_steps": harm_steps,
        "total_steps": total_steps,
        "n_env_events": n_env_events,
        "n_agent_events": n_agent_events,
        "residue_env": residue_env,
        "residue_agent": residue_agent,
        "false_attr_rate": false_attr_rate,
        "total_residue_eval": total_residue_accumulated,
        "mean_causal_sig_env": float(np.mean(causal_sigs_env)) if causal_sigs_env else 0.0,
        "mean_causal_sig_agent": float(np.mean(causal_sigs_agent)) if causal_sigs_agent else 0.0,
        "null_cf_max": null_cf_max,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> dict:
    t0 = time.perf_counter()
    per_seed: Dict[str, dict] = {}
    harm_r2_vals: List[float] = []
    gated_far: List[float] = []
    ungated_far: List[float] = []
    gated_harm: List[float] = []
    ungated_harm: List[float] = []
    null_cf_max_all: List[float] = []
    label_balances: Dict[str, dict] = {}

    n_p0 = 3 if dry_run else P0_EPISODES
    n_p1 = 3 if dry_run else P1_EPISODES
    total_train_eps = n_p0 + n_p1

    for seed in SEEDS:
        print(f"Seed {seed} Condition TRAIN", flush=True)
        modules = build_modules(seed)

        print(f"  [EXQ-877] seed={seed} P0 HarmEncoder warmup...", flush=True)
        phase0_train_encoder(modules, seed, n_p0, STEPS_PER_TRAIN_EP, total_train_eps, dry_run=dry_run)

        print(f"  [EXQ-877] seed={seed} P1 E2HarmSForward + harm_eval_z_harm...", flush=True)
        p1_out = phase1_train_heads(modules, seed, n_p1, STEPS_PER_TRAIN_EP, n_p0, total_train_eps, dry_run=dry_run)
        harm_r2_vals.append(p1_out["harm_forward_r2"])
        label_balances[str(seed)] = p1_out["label_balance"]
        print(
            f"  [EXQ-877] seed={seed} harm_forward_r2={p1_out['harm_forward_r2']:.4f} "
            f"n_hf_data={p1_out['n_hf_data']}",
            flush=True,
        )

        print(f"Seed {seed} Condition GATED_HARM_S", flush=True)
        gated_res = eval_condition("GATED_HARM_S", modules, seed, dry_run=dry_run)
        print(f"  verdict: {'PASS' if gated_res['false_attr_rate'] < 1.0 else 'FAIL'}", flush=True)

        modules_copy = copy.deepcopy(modules)
        print(f"Seed {seed} Condition UNGATED", flush=True)
        ungated_res = eval_condition("UNGATED", modules_copy, seed, dry_run=dry_run)
        print(f"  verdict: {'PASS' if ungated_res['false_attr_rate'] >= 0.0 else 'FAIL'}", flush=True)

        g_far = gated_res["false_attr_rate"]
        u_far = ungated_res["false_attr_rate"]
        gated_far.append(g_far)
        ungated_far.append(u_far)
        gated_harm.append(gated_res["harm_rate"])
        ungated_harm.append(ungated_res["harm_rate"])
        null_cf_max_all.append(max(gated_res["null_cf_max"], ungated_res["null_cf_max"]))

        print(
            f"  [EXQ-877] seed={seed} GATED_far={g_far:.4f} UNGATED_far={u_far:.4f} "
            f"delta={u_far - g_far:.4f} GATED_harm={gated_res['harm_rate']:.4f} "
            f"UNGATED_harm={ungated_res['harm_rate']:.4f} "
            f"null_cf_max={max(gated_res['null_cf_max'], ungated_res['null_cf_max']):.2e} "
            f"causal_sig[env={gated_res['mean_causal_sig_env']:.4f} "
            f"agent={gated_res['mean_causal_sig_agent']:.4f}]",
            flush=True,
        )

        per_seed[str(seed)] = {
            "harm_forward_r2": p1_out["harm_forward_r2"],
            "GATED_HARM_S": gated_res,
            "UNGATED":      ungated_res,
            "c1_direction_pass": g_far < u_far,
            "delta_false_attr":  u_far - g_far,
            "harm_difference":   gated_res["harm_rate"] - ungated_res["harm_rate"],
        }

    # -----------------------------------------------------------------------
    # Criteria evaluation
    # -----------------------------------------------------------------------
    harm_r2_mean = sum(harm_r2_vals) / len(harm_r2_vals)
    mean_delta_far = sum(ungated_far[i] - gated_far[i] for i in range(len(SEEDS))) / len(SEEDS)

    c1_pass = all(per_seed[str(s)]["c1_direction_pass"] for s in SEEDS)
    c2_pass = mean_delta_far >= THRESH_C2_FAR_DELTA
    c3_pass = all(
        (gated_harm[i] - ungated_harm[i]) <= THRESH_C3_HARM_TOLERANCE
        for i in range(len(SEEDS))
    )
    c4_pass = all(per_seed[str(s)]["GATED_HARM_S"]["n_env_events"] >= THRESH_C4_MIN_ENV_EVENTS for s in SEEDS)
    c5_pass = all(per_seed[str(s)]["GATED_HARM_S"]["n_agent_events"] >= THRESH_C5_MIN_AGENT_EVENTS for s in SEEDS)
    c6_pass = harm_r2_mean >= THRESH_C6_HARM_QUALITY
    c7_pass = all(v < THRESH_C7_NULL_CF for v in null_cf_max_all)

    criteria = {
        "C1_direction_all_seeds": c1_pass,
        "C2_mean_delta_far":      c2_pass,
        "C3_harm_not_degraded":   c3_pass,
        "C4_env_events":          c4_pass,
        "C5_agent_events":        c5_pass,
        "C6_harm_forward_quality": c6_pass,
        "C7_null_cf_sanity":      c7_pass,
    }
    print(f"[EXQ-877] Criteria: {criteria}", flush=True)
    print(f"[EXQ-877] mean_delta_far={mean_delta_far:.4f} harm_r2_mean={harm_r2_mean:.4f}", flush=True)

    quality_gates_pass = c4_pass and c5_pass and c6_pass and c7_pass

    if not quality_gates_pass:
        outcome = "FAIL"
        evidence_direction = "inconclusive"
        decision = "inconclusive"
    elif c1_pass and c2_pass and c3_pass:
        outcome = "PASS"
        evidence_direction = "supports"
        decision = "retain_ree"
    elif not c1_pass:
        outcome = "FAIL"
        evidence_direction = "weakens"
        decision = "hybridize"
    else:
        outcome = "FAIL"
        evidence_direction = "mixed"
        decision = "hybridize"

    print(f"[EXQ-877] outcome={outcome} evidence_direction={evidence_direction}", flush=True)

    # Non-degeneracy self-report (2026-06-11 rule -- applies to `evidence` runs
    # too, not just diagnostic/baseline). The load-bearing readouts are C1/C2's
    # per-seed false-attribution delta (must have real spread, not floor-pinned
    # at 0 across every seed) and the discriminator's own agent-vs-env causal_sig
    # separation (must not be identical across the two groups -- a pinned
    # discriminator would make C1/C2 pass or fail for a structural, not
    # scientific, reason).
    degeneracy = check_degeneracy({
        "false_attr_rate_delta": {
            "values": [ungated_far[i] - gated_far[i] for i in range(len(SEEDS))],
        },
        "causal_sig_agent_vs_env": {
            "groups": [
                [per_seed[str(s)]["GATED_HARM_S"]["mean_causal_sig_agent"] for s in SEEDS],
                [per_seed[str(s)]["GATED_HARM_S"]["mean_causal_sig_env"] for s in SEEDS],
            ],
        },
    })
    print(f"[EXQ-877] degeneracy: {degeneracy}", flush=True)

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts_str}_v3"

    pack = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_class": "discriminative_pair",
        "decision": decision,
        "criteria": criteria,
        "pre_registered_thresholds": {
            "GATE_THRESH_CF":            GATE_THRESH_CF,
            "THRESH_C2_FAR_DELTA":       THRESH_C2_FAR_DELTA,
            "THRESH_C3_HARM_TOLERANCE":  THRESH_C3_HARM_TOLERANCE,
            "THRESH_C4_MIN_ENV_EVENTS":  THRESH_C4_MIN_ENV_EVENTS,
            "THRESH_C5_MIN_AGENT_EVENTS": THRESH_C5_MIN_AGENT_EVENTS,
            "THRESH_C6_HARM_QUALITY":    THRESH_C6_HARM_QUALITY,
            "THRESH_C7_NULL_CF":         THRESH_C7_NULL_CF,
        },
        "summary_metrics": {
            "gated_false_attr_rate_mean":   sum(gated_far) / len(gated_far),
            "ungated_false_attr_rate_mean": sum(ungated_far) / len(ungated_far),
            "mean_delta_false_attr":        mean_delta_far,
            "gated_harm_rate_mean":         sum(gated_harm) / len(gated_harm),
            "ungated_harm_rate_mean":       sum(ungated_harm) / len(ungated_harm),
            "harm_forward_r2_mean":         harm_r2_mean,
            "null_cf_max_overall":          max(null_cf_max_all) if null_cf_max_all else 0.0,
        },
        "seeds": SEEDS,
        "config": {
            "env_size": ENV_SIZE,
            "num_hazards": NUM_HAZARDS,
            "hazard_harm": HAZARD_HARM,
            "world_dim": WORLD_DIM,
            "z_harm_dim": Z_HARM_DIM,
            "action_dim": ACTION_DIM,
            "p0_episodes": P0_EPISODES,
            "p1_episodes": P1_EPISODES,
            "n_eval_episodes": N_EVAL_EPISODES,
            "steps_per_train_ep": STEPS_PER_TRAIN_EP,
            "steps_per_eval_ep": STEPS_PER_EVAL_EP,
            "gate_thresh_cf": GATE_THRESH_CF,
        },
        "label_balance": label_balances,
        "scenario": (
            "discriminative_pair: GATED_HARM_S (residue accumulates only when "
            "the ARC-033/SD-003 counterfactual discriminator causal_sig = "
            "E3.harm_eval_z_harm(E2HarmSForward(z_harm_s,a_actual)) - "
            "E3.harm_eval_z_harm(E2HarmSForward(z_harm_s,a_stay)) exceeds "
            f"GATE_THRESH_CF={GATE_THRESH_CF}) vs UNGATED (residue accumulates "
            "at all harm events regardless of cause). Replaces V3-EXQ-213's "
            "world_forward-magnitude proxy gate with the real E2_harm_s "
            "counterfactual pathway (SD-011/ARC-033), which is the 'full "
            "discriminator-gate design' MECH-072's evidence_quality_note "
            f"calls for. CausalGridWorldV2 size={ENV_SIZE} hazards={NUM_HAZARDS}."
        ),
        "interpretation": (
            "PASS supports MECH-072 under the FULL discriminator-gate design: "
            "gating residue accumulation on the ARC-033 counterfactual "
            "causal_sig (not just the V3-EXQ-213 world_forward proxy) reduces "
            "false attribution while preserving harm avoidance -- MECH-072's "
            "own stated bar for full validation is met. FAIL (weakens): the "
            "real discriminator does not reduce false attribution at this "
            "scale even though quality gates pass, suggesting the proxy's "
            "V3-EXQ-213 PASS does not generalise to the dedicated harm-stream "
            "discriminator. INCONCLUSIVE: insufficient events, E2_harm_s "
            "quality too low, or the null-CF pipeline-integrity check failed."
        ),
        "per_seed_results": per_seed,
        "supersedes": None,
        "dry_run": dry_run,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    pack.update(degeneracy)

    out_path = write_flat_manifest(
        pack,
        out_dir=None,
        dry_run=dry_run,
        config=pack.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
    )
    if not dry_run:
        print(f"[EXQ-877] Result written to: {out_path}", flush=True)
    else:
        print(f"[EXQ-877] dry_run: result relocated to scratch: {out_path}", flush=True)

    pack["_out_path"] = str(out_path)
    return pack


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = run_experiment(dry_run=dry_run)
    print(f"[EXQ-877] Done. outcome={result['outcome']}", flush=True)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=result.get("_out_path"),
        dry_run=dry_run,
    )
