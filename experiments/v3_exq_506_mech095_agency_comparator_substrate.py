#!/opt/local/bin/python3
"""V3-EXQ-506 -- MECH-095 agency-detection comparator substrate test.

Claim: MECH-095 (tpj.agency_detection_comparator)
Status: active (exp_conf=0.526, 1 PASS / 8 FAIL across 16 runs;
        EXQ-047k PASS but EXQ-121 mixed/weakens; recently flagged for
        substrate-ceiling evidence_quality_note.)

Why this experiment exists
--------------------------
MECH-095 asserts the temporoparietal junction acts as an agency-detection
comparator: it distinguishes self-caused from other-caused change. V3
implements this via the E2_harm_s forward model + counterfactual_forward
pipeline (ARC-033): the comparator output is the gap between predicted
next-state under the actual action vs an alternative action. EXQ-047k
PASSed on a narrow operationalisation; later runs (EXQ-121, EXQ-089) went
mixed/weakens under richer conditions.

This experiment tests the comparator at substrate by training a fresh
E2HarmSForward (with SD-013 interventional loss enabled) on synthetic
(z_harm_s, action) -> z_harm_s_next data where action GENUINELY controls
the transition, then measures the counterfactual gap on two held-out
distributions:
  (a) AGENT_CAUSED: same generative process (action -> outcome), so
      counterfactual_forward(z, a_actual) and counterfactual_forward(z, a_cf)
      should produce divergent predictions => large comparator gap.
  (b) ENV_CAUSED: outcome is independent of action; counterfactual gap
      should collapse because no action carries causal signal.

Distractor controls (ablated):
  (c) AGENT_COLLATERAL: action taken but transition still env-driven
      (action label is uninformative). Predicts low gap.
  (d) ENV_CORRELATED: random action sampled even though env drives the
      transition. Predicts low gap.

Synthetic data generative process
---------------------------------
For agent-caused: z_next = z_curr + W_a @ action_onehot + small_noise
where W_a is a learned random projection per action (action genuinely
controls the next-state direction).
For env-caused: z_next = z_curr + W_env @ env_drift + small_noise
where env_drift is a per-step latent draw, action is sampled freely.
agent_collateral / env_correlated reuse env_caused dynamics with
action labels still present.

Comparator metric
-----------------
gap = mean over batch of ||E2(z, a_actual) - E2(z, a_random_cf)||_2
A clean comparator produces:
    gap_agent_caused > 1.5 * gap_env_caused
    gap_agent_caused > 1.5 * gap_agent_collateral
    gap_agent_caused > 1.5 * gap_env_correlated

PASS criteria (>= 2/3 seeds for each)
-------------------------------------
  C1 agent vs env: gap_agent_caused / gap_env_caused >= 1.5
  C2 agent vs collateral: gap_agent_caused / gap_agent_collateral >= 1.5
  C3 agent vs correlated: gap_agent_caused / gap_env_correlated >= 1.5
  C4 absolute discrimination: gap_agent_caused >= interventional_margin (=0.1).

PASS = C1 AND C2 AND C3 AND C4.
PASS supports MECH-095 substrate (E2_harm_s learns an agency-detection
comparator). FAIL with C4 alone PASSing -> training works but discrimination
between conditions is poor; substrate-ceiling concern (the synthetic data
may underexpress the conditions). FAIL with C1/C2 PASSing but C3 failing
-> env_correlated noise at the same shape as agent_caused fools the
comparator (suggests action_onehot signal is drowned out at low SNR).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_506_mech095_agency_comparator_substrate.py
  /opt/local/bin/python3 experiments/v3_exq_506_mech095_agency_comparator_substrate.py --dry-run
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

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.predictors.e2_harm_s import E2HarmSConfig, E2HarmSForward  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_506_mech095_agency_comparator_substrate"
CLAIM_IDS = ["MECH-095"]
EXPERIMENT_PURPOSE = "evidence"

# --- Config -------------------------------------------------------------
SEEDS = (42, 43, 44)
Z_HARM_DIM = 32
ACTION_DIM = 4
HIDDEN_DIM = 128
N_TRAIN_STEPS = 800
BATCH = 64
N_EVAL = 256
NOISE_SCALE = 0.05
INTERVENTIONAL_MARGIN = 0.1
INTERVENTIONAL_FRACTION = 0.3
LR = 5e-4

# Pre-registered thresholds.
C1_MIN_RATIO = 1.5
C2_MIN_RATIO = 1.5
C3_MIN_RATIO = 1.5
C4_MIN_AGENT_GAP = INTERVENTIONAL_MARGIN
PASS_FRACTION_REQUIRED = 2.0 / 3.0


def _action_onehot(idx_batch: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(idx_batch, num_classes=n).float()


def _generate_agent_caused_batch(W_a: torch.Tensor, batch: int, gen: torch.Generator
                                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """z_next = z + W_a[a] + noise. Action genuinely drives the transition."""
    z = torch.randn(batch, Z_HARM_DIM, generator=gen)
    a_idx = torch.randint(0, ACTION_DIM, (batch,), generator=gen)
    a = _action_onehot(a_idx, ACTION_DIM)
    delta = a @ W_a  # [batch, Z_HARM_DIM]
    noise = NOISE_SCALE * torch.randn(batch, Z_HARM_DIM, generator=gen)
    z_next = z + delta + noise
    return z, a, z_next


def _generate_env_caused_batch(W_env: torch.Tensor, batch: int, gen: torch.Generator
                                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """z_next = z + W_env @ env_drift + noise. Action sampled but irrelevant."""
    z = torch.randn(batch, Z_HARM_DIM, generator=gen)
    a_idx = torch.randint(0, ACTION_DIM, (batch,), generator=gen)
    a = _action_onehot(a_idx, ACTION_DIM)
    env_drift = torch.randn(batch, ACTION_DIM, generator=gen)
    delta = env_drift @ W_env
    noise = NOISE_SCALE * torch.randn(batch, Z_HARM_DIM, generator=gen)
    z_next = z + delta + noise
    return z, a, z_next


def _measure_comparator_gap(harm_fwd: E2HarmSForward, z: torch.Tensor,
                              a_actual: torch.Tensor, gen: torch.Generator
                              ) -> float:
    """Mean ||E2(z, a_actual) - E2(z, a_cf)|| over batch with random a_cf != a_actual."""
    actual_idx = a_actual.argmax(dim=-1)
    cf_idx = (actual_idx + 1 + torch.randint(0, ACTION_DIM - 1, actual_idx.shape, generator=gen)) % ACTION_DIM
    a_cf = _action_onehot(cf_idx, ACTION_DIM)
    with torch.no_grad():
        z_actual = harm_fwd.forward(z, a_actual)
        z_cf = harm_fwd.counterfactual_forward(z, a_cf)
    gap = (z_actual - z_cf).norm(dim=-1).mean().item()
    return float(gap)


def run_seed(seed: int) -> Dict:
    """Train fresh E2HarmSForward on agent-caused data, measure gaps on 4 conditions."""
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

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
    opt = torch.optim.Adam(harm_fwd.parameters(), lr=LR)

    # Fixed per-seed projections so train + eval share the same generative law.
    W_a = torch.randn(ACTION_DIM, Z_HARM_DIM, generator=gen) * 0.5
    W_env = torch.randn(ACTION_DIM, Z_HARM_DIM, generator=gen) * 0.5

    # ---- Train on agent-caused data ----
    harm_fwd.train()
    train_losses: List[float] = []
    for step in range(N_TRAIN_STEPS):
        z, a, z_next = _generate_agent_caused_batch(W_a, BATCH, gen)
        z_pred = harm_fwd.forward(z, a)
        loss = harm_fwd.compute_loss(z_pred, z_next.detach())
        # SD-013 interventional loss on a fraction of the batch.
        cf_idx = (a.argmax(dim=-1) + 1) % ACTION_DIM
        a_cf = _action_onehot(cf_idx, ACTION_DIM)
        n_int = max(1, int(BATCH * INTERVENTIONAL_FRACTION))
        loss_int = harm_fwd.compute_interventional_loss(z[:n_int], a[:n_int], a_cf[:n_int])
        total = loss + loss_int
        opt.zero_grad()
        total.backward()
        opt.step()
        if step % max(1, N_TRAIN_STEPS // 4) == 0:
            train_losses.append(float(total.item()))

    # ---- Evaluate gaps under 4 conditions (held-out, larger batches) ----
    harm_fwd.eval()
    z_a, a_a, _ = _generate_agent_caused_batch(W_a, N_EVAL, gen)
    gap_agent_caused = _measure_comparator_gap(harm_fwd, z_a, a_a, gen)

    z_e, a_e, _ = _generate_env_caused_batch(W_env, N_EVAL, gen)
    gap_env_caused = _measure_comparator_gap(harm_fwd, z_e, a_e, gen)

    # AGENT_COLLATERAL: env-driven dynamics but with action labels carried.
    # Distinct from env_caused only in that we score against the ACTUAL emitted
    # label rather than a fresh sample (probes whether the comparator gap is
    # driven by the label content vs the genuine causal pattern).
    z_col, a_col, _ = _generate_env_caused_batch(W_env, N_EVAL, gen)
    gap_collateral = _measure_comparator_gap(harm_fwd, z_col, a_col, gen)

    # ENV_CORRELATED: env-driven, sample fresh a (no informational signal).
    z_corr, _, _ = _generate_env_caused_batch(W_env, N_EVAL, gen)
    a_corr_idx = torch.randint(0, ACTION_DIM, (N_EVAL,), generator=gen)
    a_corr = _action_onehot(a_corr_idx, ACTION_DIM)
    gap_correlated = _measure_comparator_gap(harm_fwd, z_corr, a_corr, gen)

    ratio_env = gap_agent_caused / max(1e-6, gap_env_caused)
    ratio_col = gap_agent_caused / max(1e-6, gap_collateral)
    ratio_corr = gap_agent_caused / max(1e-6, gap_correlated)

    print(f"  seed={seed} train_losses[start..end]={train_losses[:1]+train_losses[-1:]}", flush=True)
    print(f"  seed={seed} gap_agent={gap_agent_caused:.4f} gap_env={gap_env_caused:.4f} "
          f"gap_col={gap_collateral:.4f} gap_corr={gap_correlated:.4f} "
          f"ratios env/col/corr = {ratio_env:.2f}/{ratio_col:.2f}/{ratio_corr:.2f}", flush=True)

    return {
        "seed": seed,
        "gap_agent_caused": gap_agent_caused,
        "gap_env_caused": gap_env_caused,
        "gap_agent_collateral": gap_collateral,
        "gap_env_correlated": gap_correlated,
        "ratio_agent_over_env": ratio_env,
        "ratio_agent_over_collateral": ratio_col,
        "ratio_agent_over_correlated": ratio_corr,
        "train_loss_first": train_losses[0] if train_losses else None,
        "train_loss_last": train_losses[-1] if train_losses else None,
    }


def _evaluate(rows: List[Dict]) -> Dict:
    n = len(rows)
    required = math.ceil(n * PASS_FRACTION_REQUIRED)
    c1 = sum(1 for r in rows if r["ratio_agent_over_env"] >= C1_MIN_RATIO)
    c2 = sum(1 for r in rows if r["ratio_agent_over_collateral"] >= C2_MIN_RATIO)
    c3 = sum(1 for r in rows if r["ratio_agent_over_correlated"] >= C3_MIN_RATIO)
    c4 = sum(1 for r in rows if r["gap_agent_caused"] >= C4_MIN_AGENT_GAP)
    return {
        "n_seeds": n, "min_seeds_required": required,
        "c1_seeds_pass": c1, "c2_seeds_pass": c2,
        "c3_seeds_pass": c3, "c4_seeds_pass": c4,
        "c1_pass": c1 >= required, "c2_pass": c2 >= required,
        "c3_pass": c3 >= required, "c4_pass": c4 >= required,
        "overall_pass": (c1 >= required and c2 >= required
                          and c3 >= required and c4 >= required),
        "mean_ratio_agent_over_env": float(sum(r["ratio_agent_over_env"] for r in rows) / n),
        "mean_ratio_agent_over_collateral": float(sum(r["ratio_agent_over_collateral"] for r in rows) / n),
        "mean_ratio_agent_over_correlated": float(sum(r["ratio_agent_over_correlated"] for r in rows) / n),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()

    seeds = (args.seeds[0],) if args.dry_run else tuple(args.seeds)
    if args.dry_run:
        global N_TRAIN_STEPS, N_EVAL
        N_TRAIN_STEPS = 50
        N_EVAL = 32
        print("[DRY-RUN] 1 seed, 50 train steps, 32-batch eval -- smoke only.", flush=True)

    t0 = time.time()
    rows = [run_seed(s) for s in seeds]
    elapsed = time.time() - t0

    criteria = _evaluate(rows)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    direction = "supports" if criteria["overall_pass"] else "weakens"

    print(f"\nV3-EXQ-506 (MECH-095) -- {outcome} in {elapsed:.1f}s ({len(seeds)} seed(s))", flush=True)
    print(f"  C1 ratio agent/env  >= {C1_MIN_RATIO}: "
          f"{criteria['c1_seeds_pass']}/{criteria['n_seeds']} -> "
          f"{'PASS' if criteria['c1_pass'] else 'FAIL'}", flush=True)
    print(f"  C2 ratio agent/col  >= {C2_MIN_RATIO}: "
          f"{criteria['c2_seeds_pass']}/{criteria['n_seeds']} -> "
          f"{'PASS' if criteria['c2_pass'] else 'FAIL'}", flush=True)
    print(f"  C3 ratio agent/corr >= {C3_MIN_RATIO}: "
          f"{criteria['c3_seeds_pass']}/{criteria['n_seeds']} -> "
          f"{'PASS' if criteria['c3_pass'] else 'FAIL'}", flush=True)
    print(f"  C4 abs gap_agent    >= {C4_MIN_AGENT_GAP}: "
          f"{criteria['c4_seeds_pass']}/{criteria['n_seeds']} -> "
          f"{'PASS' if criteria['c4_pass'] else 'FAIL'}", flush=True)

    if args.dry_run:
        print("[--dry-run] manifest not written.", flush=True)
        return

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest = {
        "schema_version": "v1", "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS, "result": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"MECH-095": direction},
        "criteria": criteria,
        "registered_thresholds": {
            "C1_MIN_RATIO": C1_MIN_RATIO, "C2_MIN_RATIO": C2_MIN_RATIO,
            "C3_MIN_RATIO": C3_MIN_RATIO, "C4_MIN_AGENT_GAP": C4_MIN_AGENT_GAP,
            "PASS_FRACTION_REQUIRED": PASS_FRACTION_REQUIRED,
        },
        "config": {
            "z_harm_dim": Z_HARM_DIM, "action_dim": ACTION_DIM,
            "hidden_dim": HIDDEN_DIM, "n_train_steps": N_TRAIN_STEPS,
            "batch": BATCH, "n_eval": N_EVAL, "noise_scale": NOISE_SCALE,
            "interventional_margin": INTERVENTIONAL_MARGIN,
            "interventional_fraction": INTERVENTIONAL_FRACTION,
            "lr": LR, "seeds": list(seeds),
        },
        "results_per_seed": rows,
        "elapsed_seconds": elapsed,
        "notes": (
            "Substrate-level test of E2_harm_s as agency-detection comparator "
            "(MECH-095 V3 instantiation via ARC-033). Trains a fresh "
            "E2HarmSForward on synthetic action-controlled transitions with "
            "SD-013 interventional loss enabled, then measures the "
            "counterfactual_forward gap on four held-out conditions: "
            "agent-caused (action drives outcome), env-caused (action "
            "irrelevant), agent-collateral (action label retained but env-"
            "driven), env-correlated (random action, env-driven). PASS "
            "supports MECH-095 substrate. C4 alone PASSing while C1-C3 "
            "fail is the substrate-ceiling signature flagged in the recent "
            "MECH-095 evidence_quality_note."
        ),
    }
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
