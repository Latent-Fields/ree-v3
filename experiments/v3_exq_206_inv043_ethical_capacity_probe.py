"""
V3-EXQ-206 -- INV-043: Ethical Capacity Probe (harm-rich vs harm-sparse training)

Claims: INV-043

INV-043 claim: REE architecture enables but does not guarantee ethical development.
Ethical selection machinery requires exposure to harm contexts ('caregiving experience').

Note: INV-043 strictly requires a multi-agent substrate with modelled caregiving for a
direct test. This experiment is a PROXY: it uses harm-rich vs harm-sparse training
regimes as a single-agent analog of developmental exposure. The test question is:
Does the architectural machinery (E3 harm evaluation) develop differently when given
harm exposure vs when trained on harm-absent data?

Design:
  Shared HarmEncoder trained on harm-rich data (RICH env, 4 hazards, 100 eps).
  Two harm_eval probe heads trained on different data:
    PROBE_RICH:   trained with harm-rich transition data (positive harm labels present)
    PROBE_SPARSE: trained with harm-absent transition data (no positive harm events)
  Evaluation: both probes tested on held-out RICH data.
  Metrics: r2 of harm prediction for RICH vs SPARSE probes.

EXPERIMENT_PURPOSE = "diagnostic"
  (INV-043 is not directly testable in single-agent V3 -- this is a substrate readiness
  proxy. Diagnostic experiments are excluded from governance confidence scoring.)

PRE-REGISTERED ACCEPTANCE CRITERIA:
  C1 (r2_rich > 0.1):
    RICH probe develops harm discrimination (architecture CAN develop ethical machinery
    when given harm experience).
  C2 (r2_sparse < 0.05):
    SPARSE probe fails to develop harm discrimination (without harm experience, the
    machinery does not develop despite architecture being capable).
  C3 (discrimination_delta > 0.05 in >= 2/3 seeds):
    Pairwise: RICH significantly outperforms SPARSE.
  C4 (n_harm_test >= 10):
    Sufficient harm test events for reliable r2 estimate.
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
import numpy as np

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder


EXPERIMENT_TYPE    = "v3_exq_206_inv043_ethical_capacity_probe"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS          = ["INV-043"]

HARM_OBS_DIM = 51  # hazard_field(25) + resource_field(25) + exposure(1)

# Pre-registered thresholds
THRESH_C1_R2_RICH   = 0.1
THRESH_C2_R2_SPARSE = 0.05
THRESH_C3_DELTA     = 0.05
THRESH_C4_N_HARM    = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_harm_eval_probe(z_harm_dim: int, hidden_dim: int = 64) -> nn.Sequential:
    """Simple linear probe: z_harm -> harm_score in [0, 1]."""
    return nn.Sequential(
        nn.Linear(z_harm_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


def _r2(preds: np.ndarray, targets: np.ndarray) -> float:
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    if ss_tot < 1e-8:
        return 0.0
    return float(1 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Phase 0: Train shared HarmEncoder on RICH environment
# ---------------------------------------------------------------------------

def _train_shared_encoder(
    harm_enc: HarmEncoder,
    seed: int,
    num_episodes: int,
    steps_per_episode: int,
    device,
    lr: float = 1e-3,
) -> None:
    """Train HarmEncoder with autoencoder + center-cell regression on RICH env."""
    env = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=0.05,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=0.1,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.2,
        hazard_field_decay=0.5,
    )
    harm_decoder = nn.Sequential(
        nn.Linear(harm_enc.z_harm_dim, 64),
        nn.ReLU(),
        nn.Linear(64, HARM_OBS_DIM),
    ).to(device)
    optimizer = optim.Adam(
        list(harm_enc.parameters()) + list(harm_decoder.parameters()), lr=lr
    )
    harm_enc.train()

    for ep in range(num_episodes):
        _, obs_dict = env.reset()
        for _ in range(steps_per_episode):
            harm_obs_t = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            if harm_obs_t.dim() == 1:
                harm_obs_t = harm_obs_t.unsqueeze(0)
            harm_obs_t = harm_obs_t.to(device)

            z = harm_enc(harm_obs_t)
            recon = harm_decoder(z)
            loss = F.mse_loss(recon, harm_obs_t)
            # Center-cell supervision (hazard_field_view idx 12)
            loss = loss + 0.5 * F.mse_loss(z[:, :1], harm_obs_t[:, 12:13])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            action = torch.zeros(1, env.action_dim, device=device)
            action[0, random.randint(0, env.action_dim - 1)] = 1.0
            _, _, done, _, obs_dict = env.step(action)
            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(f"  [P0 encoder] ep {ep+1}/{num_episodes}  loss={loss.item():.5f}", flush=True)

    harm_enc.eval()
    for p in harm_enc.parameters():
        p.requires_grad_(False)


# ---------------------------------------------------------------------------
# Phase 1: Collect transition data for probe training
# ---------------------------------------------------------------------------

def _collect_data(
    harm_enc: HarmEncoder,
    seed: int,
    num_episodes: int,
    steps_per_episode: int,
    device,
    num_hazards: int = 4,
) -> Tuple[List[torch.Tensor], List[float]]:
    """
    Collect (z_harm, harm_label) pairs from environment.
    harm_label: 1.0 if harm_signal < 0, else 0.0.
    Returns z_harm_list, labels_list.
    """
    env = CausalGridWorldV2(
        seed=seed + 10000, size=12, num_hazards=num_hazards, num_resources=5,
        hazard_harm=0.05,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=0.1,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.2,
        hazard_field_decay=0.5,
    )
    z_harms: List[torch.Tensor] = []
    labels:  List[float]        = []

    for _ in range(num_episodes):
        _, obs_dict = env.reset()
        for _ in range(steps_per_episode):
            harm_obs_t = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            if harm_obs_t.dim() == 1:
                harm_obs_t = harm_obs_t.unsqueeze(0)
            harm_obs_t = harm_obs_t.to(device)

            with torch.no_grad():
                z = harm_enc(harm_obs_t)
            z_harms.append(z.cpu())

            action = torch.zeros(1, env.action_dim, device=device)
            action[0, random.randint(0, env.action_dim - 1)] = 1.0
            _, harm_signal, done, _, obs_dict = env.step(action)
            labels.append(1.0 if harm_signal < 0 else 0.0)
            if done:
                break

    return z_harms, labels


# ---------------------------------------------------------------------------
# Phase 2: Train probe heads
# ---------------------------------------------------------------------------

def _train_probe(
    probe: nn.Module,
    z_harms: List[torch.Tensor],
    labels: List[float],
    device,
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> None:
    """Train a harm_eval probe head on collected (z_harm, label) pairs."""
    probe.train()
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    n = len(z_harms)
    if n < 4:
        return

    for epoch in range(num_epochs):
        idxs = list(range(n))
        random.shuffle(idxs)
        for start in range(0, n, batch_size):
            batch_idxs = idxs[start:start + batch_size]
            if not batch_idxs:
                continue
            zh_b = torch.cat([z_harms[i] for i in batch_idxs]).to(device)
            lb_b = torch.tensor([[labels[i]] for i in batch_idxs], device=device)
            pred = probe(zh_b)
            loss = F.mse_loss(pred, lb_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 0.5)
            optimizer.step()

    probe.eval()


# ---------------------------------------------------------------------------
# Phase 3: Evaluate probes on test data
# ---------------------------------------------------------------------------

def _eval_probe(
    probe: nn.Module,
    z_harms_test: List[torch.Tensor],
    labels_test: List[float],
    device,
) -> Dict:
    """Compute r2 on test data (must contain both harm and safe examples)."""
    probe.eval()
    n = len(z_harms_test)
    if n < 4:
        return {"r2": 0.0, "n_harm": 0, "n_safe": 0, "mean_pred_harm": 0.0, "mean_pred_safe": 0.0}

    with torch.no_grad():
        zh = torch.cat(z_harms_test).to(device)
        preds = probe(zh).cpu().numpy().flatten()

    labels_arr = np.array(labels_test)
    r2 = _r2(preds, labels_arr)

    harm_mask = labels_arr > 0.5
    n_harm = int(harm_mask.sum())
    n_safe = int((~harm_mask).sum())
    mean_pred_harm = float(preds[harm_mask].mean()) if n_harm > 0 else 0.0
    mean_pred_safe = float(preds[~harm_mask].mean()) if n_safe > 0 else 0.0

    return {
        "r2":            r2,
        "n_harm":        n_harm,
        "n_safe":        n_safe,
        "mean_pred_harm":mean_pred_harm,
        "mean_pred_safe":mean_pred_safe,
    }


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------

def run_seed(
    seed:              int,
    encoder_episodes:  int,
    probe_train_episodes: int,
    test_episodes:     int,
    steps_per_episode: int,
    z_harm_dim:        int,
    device,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Phase 0: shared encoder on RICH env
    print(f"\n  [seed {seed}] P0: train shared HarmEncoder ({encoder_episodes} eps)", flush=True)
    harm_enc = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=z_harm_dim).to(device)
    _train_shared_encoder(harm_enc, seed, encoder_episodes, steps_per_episode, device)

    # Phase 1: collect training data
    print(f"  [seed {seed}] Collecting RICH training data ({probe_train_episodes} eps)", flush=True)
    z_rich, lab_rich = _collect_data(
        harm_enc, seed, probe_train_episodes, steps_per_episode, device, num_hazards=4
    )
    print(
        f"    RICH: n={len(z_rich)}, harm={sum(1 for l in lab_rich if l > 0.5)}"
        f", safe={sum(1 for l in lab_rich if l <= 0.5)}",
        flush=True,
    )

    print(f"  [seed {seed}] Collecting SPARSE training data ({probe_train_episodes} eps)", flush=True)
    z_sparse, lab_sparse = _collect_data(
        harm_enc, seed + 1, probe_train_episodes, steps_per_episode, device, num_hazards=0
    )
    print(
        f"    SPARSE: n={len(z_sparse)}, harm={sum(1 for l in lab_sparse if l > 0.5)}"
        f", safe={sum(1 for l in lab_sparse if l <= 0.5)}",
        flush=True,
    )

    # Phase 2: train probes
    probe_rich   = _make_harm_eval_probe(z_harm_dim).to(device)
    probe_sparse = _make_harm_eval_probe(z_harm_dim).to(device)

    print(f"  [seed {seed}] Training PROBE_RICH", flush=True)
    _train_probe(probe_rich, z_rich, lab_rich, device, num_epochs=100)

    print(f"  [seed {seed}] Training PROBE_SPARSE", flush=True)
    _train_probe(probe_sparse, z_sparse, lab_sparse, device, num_epochs=100)

    # Phase 3: collect shared test data (from RICH env, shared eval set)
    print(f"  [seed {seed}] Collecting shared test data ({test_episodes} eps)", flush=True)
    z_test, lab_test = _collect_data(
        harm_enc, seed + 2, test_episodes, steps_per_episode, device, num_hazards=4
    )

    res_rich   = _eval_probe(probe_rich,   z_test, lab_test, device)
    res_sparse = _eval_probe(probe_sparse, z_test, lab_test, device)

    r2_rich   = res_rich["r2"]
    r2_sparse = res_sparse["r2"]
    delta     = r2_rich - r2_sparse
    n_harm    = res_rich["n_harm"]

    print(
        f"  [seed {seed}] PROBE_RICH   r2={r2_rich:.4f}  "
        f"harm_mean={res_rich['mean_pred_harm']:.4f}  safe_mean={res_rich['mean_pred_safe']:.4f}",
        flush=True,
    )
    print(
        f"  [seed {seed}] PROBE_SPARSE r2={r2_sparse:.4f}  "
        f"harm_mean={res_sparse['mean_pred_harm']:.4f}  safe_mean={res_sparse['mean_pred_safe']:.4f}",
        flush=True,
    )
    print(f"  [seed {seed}] discrimination_delta={delta:.4f}  n_harm_test={n_harm}", flush=True)

    c1_pass = r2_rich   > THRESH_C1_R2_RICH
    c2_pass = r2_sparse < THRESH_C2_R2_SPARSE
    c3_seed = delta     > THRESH_C3_DELTA
    c4_pass = n_harm    >= THRESH_C4_N_HARM

    return {
        "seed":          seed,
        "r2_rich":       r2_rich,
        "r2_sparse":     r2_sparse,
        "delta":         delta,
        "n_harm_test":   n_harm,
        "n_safe_test":   res_rich["n_safe"],
        "n_train_rich_harm":   sum(1 for l in lab_rich if l > 0.5),
        "n_train_sparse_harm": sum(1 for l in lab_sparse if l > 0.5),
        "rich_mean_pred_harm":   res_rich["mean_pred_harm"],
        "rich_mean_pred_safe":   res_rich["mean_pred_safe"],
        "sparse_mean_pred_harm": res_sparse["mean_pred_harm"],
        "sparse_mean_pred_safe": res_sparse["mean_pred_safe"],
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_seed": c3_seed,
        "c4_pass": c4_pass,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    seeds:                  Optional[List[int]] = None,
    encoder_episodes:       int = 100,
    probe_train_episodes:   int = 100,
    test_episodes:          int = 50,
    steps_per_episode:      int = 200,
    z_harm_dim:             int = 32,
    **kwargs,
) -> dict:
    if seeds is None:
        seeds = [0, 1, 2]

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(
        f"[V3-EXQ-206] INV-043: Ethical Capacity Proxy Probe\n"
        f"  PROBE_RICH: trained with 4-hazard harm data (has positive harm events)\n"
        f"  PROBE_SPARSE: trained with 0-hazard data (no positive harm events)\n"
        f"  Both evaluated on same RICH test set\n"
        f"  Seeds: {seeds}  Encoder: {encoder_episodes} eps"
        f"  ProbeTraining: {probe_train_episodes} eps  Eval: {test_episodes} eps\n"
        f"  Pre-registered thresholds:\n"
        f"    C1: r2_rich > {THRESH_C1_R2_RICH}\n"
        f"    C2: r2_sparse < {THRESH_C2_R2_SPARSE}\n"
        f"    C3: delta > {THRESH_C3_DELTA} in >= 2/3 seeds\n"
        f"    C4: n_harm_test >= {THRESH_C4_N_HARM} in ALL seeds",
        flush=True,
    )

    seed_results = []
    for seed in seeds:
        sr = run_seed(
            seed=seed,
            encoder_episodes=encoder_episodes,
            probe_train_episodes=probe_train_episodes,
            test_episodes=test_episodes,
            steps_per_episode=steps_per_episode,
            z_harm_dim=z_harm_dim,
            device=device,
        )
        seed_results.append(sr)

    n_seeds = len(seeds)

    def _smean(k): return float(np.mean([r[k] for r in seed_results]))
    def _sstd(k):  return float(np.std([r[k]  for r in seed_results]))

    c1_pass_agg = _smean("r2_rich")   > THRESH_C1_R2_RICH
    c2_pass_agg = _smean("r2_sparse") < THRESH_C2_R2_SPARSE
    c3_seeds    = sum(1 for r in seed_results if r["c3_seed"])
    c4_seeds    = sum(1 for r in seed_results if r["c4_pass"])

    majority = 2  # >= 2/3 for 3 seeds
    c3_pass = c3_seeds >= majority
    c4_pass = c4_seeds == n_seeds  # ALL seeds

    all_pass     = c1_pass_agg and c2_pass_agg and c3_pass and c4_pass
    criteria_met = sum([c1_pass_agg, c2_pass_agg, c3_pass, c4_pass])
    status       = "PASS" if all_pass else "FAIL"

    failure_notes = []
    if not c1_pass_agg:
        failure_notes.append(
            f"C1 FAIL: r2_rich_mean={_smean('r2_rich'):.4f} <= {THRESH_C1_R2_RICH}"
        )
    if not c2_pass_agg:
        failure_notes.append(
            f"C2 FAIL: r2_sparse_mean={_smean('r2_sparse'):.4f} >= {THRESH_C2_R2_SPARSE}"
        )
    if not c3_pass:
        deltas = [r["delta"] for r in seed_results]
        failure_notes.append(
            f"C3 FAIL: delta > {THRESH_C3_DELTA} in {c3_seeds}/{n_seeds} seeds"
            f" (need {majority}); values={[f'{d:.4f}' for d in deltas]}"
        )
    if not c4_pass:
        nharms = [r["n_harm_test"] for r in seed_results]
        failure_notes.append(
            f"C4 FAIL: n_harm_test < {THRESH_C4_N_HARM} in some seeds; values={nharms}"
        )

    print(f"\nV3-EXQ-206 verdict: {status}  ({criteria_met}/4)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)
    print(
        f"  r2_rich_mean={_smean('r2_rich'):.4f}"
        f"  r2_sparse_mean={_smean('r2_sparse'):.4f}"
        f"  delta_mean={_smean('delta'):.4f}",
        flush=True,
    )

    metrics = {
        "r2_rich_mean":   _smean("r2_rich"),
        "r2_rich_std":    _sstd("r2_rich"),
        "r2_sparse_mean": _smean("r2_sparse"),
        "r2_sparse_std":  _sstd("r2_sparse"),
        "delta_mean":     _smean("delta"),
        "delta_std":      _sstd("delta"),
        "n_harm_test_mean": _smean("n_harm_test"),
        "c3_seeds_pass":  float(c3_seeds),
        "criteria_met":   float(criteria_met),
        "n_seeds":        float(n_seeds),
    }
    for r in seed_results:
        s = r["seed"]
        for k in ("r2_rich", "r2_sparse", "delta", "n_harm_test",
                  "n_train_rich_harm", "n_train_sparse_harm"):
            metrics[f"seed{s}_{k}"] = float(r[k])

    rows = ""
    for r in seed_results:
        rows += (
            f"| {r['seed']} | {r['r2_rich']:.4f} | {r['r2_sparse']:.4f}"
            f" | {r['delta']:.4f} | {r['n_harm_test']} | {r['n_train_rich_harm']}"
            f" | {r['n_train_sparse_harm']} |\n"
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-206 -- INV-043: Ethical Capacity Proxy Probe

**Status:** {status}
**Claim:** INV-043 -- REE architecture enables but does not guarantee ethical development
**Experiment purpose:** diagnostic (proxy -- INV-043 requires multi-agent substrate)
**Seeds:** {seeds}

## Proxy Design

INV-043 requires multi-agent caregiving substrate not available in V3.
This proxy tests whether harm evaluation machinery develops differently under:
- PROBE_RICH: trained with harm-rich data (4 hazards, positive harm events)
- PROBE_SPARSE: trained with harm-absent data (0 hazards, no positive events)
Both probes use the same shared HarmEncoder. Evaluated on same RICH test set.

## Results by Seed

| Seed | r2_rich | r2_sparse | delta | n_harm_test | n_train_harm_rich | n_train_harm_sparse |
|------|---------|----------|-------|------------|------------------|---------------------|
{rows}

## Aggregate

| Metric | Mean | Std |
|--------|------|-----|
| r2_rich | {_smean("r2_rich"):.4f} | {_sstd("r2_rich"):.4f} |
| r2_sparse | {_smean("r2_sparse"):.4f} | {_sstd("r2_sparse"):.4f} |
| delta (rich - sparse) | {_smean("delta"):.4f} | {_sstd("delta"):.4f} |

## PASS Criteria

| Criterion | Value | Required | Result |
|-----------|-------|---------|--------|
| C1 r2_rich | {_smean("r2_rich"):.4f} | > {THRESH_C1_R2_RICH} | {"PASS" if c1_pass_agg else "FAIL"} |
| C2 r2_sparse | {_smean("r2_sparse"):.4f} | < {THRESH_C2_R2_SPARSE} | {"PASS" if c2_pass_agg else "FAIL"} |
| C3 delta | {c3_seeds}/{n_seeds} seeds | >= {majority} | {"PASS" if c3_pass else "FAIL"} |
| C4 n_harm_test | {c4_seeds}/{n_seeds} seeds | == {n_seeds} | {"PASS" if c4_pass else "FAIL"} |

Criteria met: {criteria_met}/4 -> **{status}**
{failure_section}
"""

    evidence_direction = (
        "supports" if all_pass
        else ("mixed" if criteria_met >= 2 else "weakens")
    )

    return {
        "status":             status,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": evidence_direction,
        "experiment_type":    EXPERIMENT_TYPE,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",        type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--enc-eps",      type=int, default=100)
    parser.add_argument("--probe-eps",    type=int, default=100)
    parser.add_argument("--test-eps",     type=int, default=50)
    parser.add_argument("--steps",        type=int, default=200)
    args = parser.parse_args()

    result = run(
        seeds=args.seeds,
        encoder_episodes=args.enc_eps,
        probe_train_episodes=args.probe_eps,
        test_episodes=args.test_eps,
        steps_per_episode=args.steps,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["experiment_purpose"] = EXPERIMENT_PURPOSE
    result["claim_ids"]          = CLAIM_IDS

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
            print(f"  {k}: {v:.5f}", flush=True)
