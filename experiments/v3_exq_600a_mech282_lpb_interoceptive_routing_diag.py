"""
V3-EXQ-600a -- MECH-282 LPB interoceptive harm routing substrate diagnostic.

Fix for V3-EXQ-600 ERROR: emit_outcome() used removed keyword args; no runner
sentinel written despite PASS/FAIL acceptance.

Three instrumented arms (substrate readiness only):
  LPB_OFF     -- use_broadcast_override=True, use_lpb_interoceptive_routing=False
  INTERO_HIGH -- LPB on; elevated drive + resource harm_obs_a slice
  EXT_HIGH    -- LPB on; elevated hazard harm_obs slice
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EVIDENCE_ROOT = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

EXPERIMENT_TYPE = "v3_exq_600a_mech282_lpb_interoceptive_routing_diag"
QUEUE_ID = "V3-EXQ-600a"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS = ["MECH-282", "SD-037"]
SUPERSEDES = "V3-EXQ-600"

ENV_KWARGS = dict(
    size=8,
    num_hazards=2,
    num_resources=4,
    use_proxy_fields=True,
    harm_history_len=0,
    limb_damage_enabled=False,
)

TICKS = 40
SELF_DIM = 16
WORLD_DIM = 16
HARM_DIM = 16
HARM_A_DIM = 16

ARMS = [
    {"id": "LPB_OFF", "use_lpb": False},
    {"id": "INTERO_HIGH", "use_lpb": True, "boost": "intero"},
    {"id": "EXT_HIGH", "use_lpb": True, "boost": "external"},
]


def _make_agent(env: CausalGridWorldV2, use_lpb: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        harm_dim=HARM_DIM,
        use_harm_stream=True,
        z_harm_dim=HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        use_broadcast_override=True,
        use_lpb_interoceptive_routing=use_lpb,
        use_pag_freeze_gate=True,
        override_recruitment_threshold=0.35,
        override_drive_weight=1.0,
        override_harm_weight=1.0,
        lpb_drive_weight=1.0,
        lpb_resource_weight=1.0,
    )
    return REEAgent(cfg)


def _boost_obs(obs_dict: Dict, mode: Optional[str]) -> None:
    if mode is None:
        return
    ho = obs_dict.get("harm_obs")
    ha = obs_dict.get("harm_obs_a")
    if mode == "intero" and ha is not None:
        ha = ha.clone()
        if ha.numel() >= 50:
            ha[25:50] = 0.85
        else:
            ha[:] = 0.85
        obs_dict["harm_obs_a"] = ha
        if ho is not None:
            ho = ho.clone()
            if ho.numel() >= 25:
                ho[0:25] = 0.05
            obs_dict["harm_obs"] = ho
    elif mode == "external" and ho is not None:
        ho = ho.clone()
        if ho.numel() >= 25:
            ho[0:25] = 0.85
        if ho.numel() >= 50:
            ho[25:50] = 0.05
        obs_dict["harm_obs"] = ho
        if ha is not None:
            ha = ha.clone()
            if ha.numel() >= 50:
                ha[25:50] = 0.05
            obs_dict["harm_obs_a"] = ha


def _run_arm(arm: Dict, seed: int, dry_run: bool) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    agent = _make_agent(env, use_lpb=arm["use_lpb"])
    agent.eval()

    flat, obs_dict = env.reset()
    agent.reset()
    if agent.goal_state is not None:
        agent.goal_state._last_drive_level = 0.85 if arm.get("boost") == "intero" else 0.2

    overrides: List[float] = []
    intero_norms: List[float] = []
    ext_norms: List[float] = []
    z_intero_norms: List[float] = []
    freeze_commits = 0

    n_ticks = 5 if dry_run else TICKS
    for _ in range(n_ticks):
        _boost_obs(obs_dict, arm.get("boost"))
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        latent = agent.sense(
            obs_body,
            obs_world,
            obs_harm=obs_dict.get("harm_obs"),
            obs_harm_a=obs_dict.get("harm_obs_a"),
        )
        if agent.broadcast_override is not None:
            overrides.append(float(agent.broadcast_override.override_signal))
        if agent._lpb_last_output is not None:
            intero_norms.append(float(agent._lpb_last_output.intero_magnitude))
            ext_norms.append(float(agent._lpb_last_output.external_magnitude))
        if latent.z_harm_intero is not None:
            z_intero_norms.append(float(latent.z_harm_intero.norm().item()))

        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent)
            if ticks.get("e1_tick", False)
            else torch.zeros(1, WORLD_DIM, device=agent.device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        action = agent.select_action(candidates, ticks, temperature=1.0)
        if action is None:
            action = torch.zeros(1, env.action_dim, device=agent.device)
            action[0, 0] = 1.0
        if agent._pag_last_output is not None and agent._pag_last_output.freeze_commit:
            freeze_commits += 1
        _, _, _, _, obs_dict = env.step(action)

    return {
        "arm": arm["id"],
        "override_mean": float(np.mean(overrides)) if overrides else 0.0,
        "intero_mean": float(np.mean(intero_norms)) if intero_norms else 0.0,
        "external_mean": float(np.mean(ext_norms)) if ext_norms else 0.0,
        "z_harm_intero_mean": float(np.mean(z_intero_norms)) if z_intero_norms else 0.0,
        "pag_freeze_commits": int(freeze_commits),
        "lpb_enabled": bool(arm["use_lpb"]),
    }


def _evaluate(results: List[Dict]) -> Dict:
    by_id = {r["arm"]: r for r in results}
    intero = by_id.get("INTERO_HIGH", {})
    ext = by_id.get("EXT_HIGH", {})
    c1 = intero.get("z_harm_intero_mean", 0.0) > 0.05
    c2 = intero.get("override_mean", 0.0) > ext.get("override_mean", 0.0)
    c3 = ext.get("pag_freeze_commits", 0) >= intero.get("pag_freeze_commits", 0)
    passed = c1 and c2 and c3
    return {
        "c1_z_harm_intero_populated": c1,
        "c2_intero_override_gt_external": c2,
        "c3_external_freeze_gte_intero": c3,
        "verdict": "PASS" if passed else "FAIL",
    }


def main(dry_run: bool = False) -> Dict:
    results = [_run_arm(a, seed=0, dry_run=dry_run) for a in ARMS]
    eval_out = _evaluate(results)
    outcome = eval_out["verdict"]
    print("MECH-282 substrate diagnostic")
    for r in results:
        print(
            "  {} override_mean={:.4f} intero={:.4f} external={:.4f} "
            "z_intero={:.4f} freeze_commits={}".format(
                r["arm"],
                r["override_mean"],
                r["intero_mean"],
                r["external_mean"],
                r["z_harm_intero_mean"],
                r["pag_freeze_commits"],
            )
        )
    print(f"verdict: {outcome}")
    print("  acceptance:", eval_out)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "evidence_direction_per_claim": {
            "MECH-282": "supports" if outcome == "PASS" else "weakens",
            "SD-037": "supports" if outcome == "PASS" else "weakens",
        },
        "evidence_direction_note": (
            "MECH-282 LPB interoceptive routing substrate diagnostic. "
            "Fix for V3-EXQ-600 emit_outcome API mismatch only; same arms."
        ),
        "outcome": outcome,
        "metrics": {"arms": results, "acceptance": eval_out},
        "dry_run": bool(dry_run),
    }

    out_path = None
    if not dry_run:
        out_dir = EVIDENCE_ROOT / EXPERIMENT_TYPE
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=None,
            script_path=Path(__file__),
        )
        print(f"Result written to: {out_path}", flush=True)

    return {
        "all_pass": outcome == "PASS",
        "outcome": outcome,
        "manifest_path": str(out_path) if out_path is not None else None,
        "run_id": run_id,
        "dry_run": bool(dry_run),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if not result["dry_run"]:
        emit_outcome(
            outcome=result["outcome"],
            manifest_path=result["manifest_path"],
            run_id=result["run_id"],
            queue_id=QUEUE_ID,
        )
    sys.exit(0 if result["all_pass"] else 1)
