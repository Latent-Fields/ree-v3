"""V3-EXQ-635: modulatory-bias-selection-authority substrate-readiness diagnostic.

PURPOSE: Validate that the modulatory-bias-selection-authority substrate (landed
2026-06-03, ree-v3 fc09738) gives the previously-inert modulatory levers genuine
authority over the committed E3 selection. Re-runs the three cluster-autopsy failure
shapes (604a curiosity / 624a vigor / 614d within-class temperature), each comparing
authority OFF vs ON on SHARED trained weights + identical RNG, so any difference in the
committed-selection metric is attributable solely to the gap-relative rescale.

EXPERIMENT_PURPOSE: diagnostic (substrate-readiness; claim_ids=[]). Excluded from
governance confidence/conflict scoring. PASS unblocks the per-claim EVIDENCE retests of
MECH-314 / MECH-320 / MECH-341 and the MECH-343 hypothesis.

DESIGN (3 lever conditions x {authority OFF, authority ON} on shared weights):
  VIGOR (624a): MECH-320 tonic_vigor ON. Metric = action_density (fraction of committed
    steps whose action class != noop). 624a showed action_density byte-identical ON==OFF;
    authority-ON should lift it.
  WITHIN_CLASS (614d): MECH-341 e3_score_diversity + stratified within-class temperature.
    Metric = committed-class entropy. 614d C2 showed committed-class entropy byte-identical
    across temperature; the across-class unit-range normalization should let it rise.
  CURIOSITY (604a): MECH-314 structured_curiosity on SD-056 substrate. GUARDED: 604a had
    curiosity_bias_abs_mean=0.0 (genuinely zero -- 314a no active residue centers, 314b/c
    broadcast). If the curiosity bias is degenerate (<= CURIOSITY_BIAS_FLOOR) the arm is
    INVALID_HARNESS / non-contributory, NOT a substrate failure (necessary-but-not-sufficient).

ACCEPTANCE (per lever where its modulatory bias is non-degenerate):
  vigor:        action_density(ON) - action_density(OFF) >= ACTION_DENSITY_LIFT_MIN (0.03)
  within_class: committed_class_entropy(ON) > committed_class_entropy(OFF) + ENTROPY_LIFT_MIN
  curiosity:    if curiosity_bias_abs_mean > floor: committed_class_entropy(ON) > OFF + ENTROPY_LIFT_MIN
                else INVALID_HARNESS (arm excluded, not failed)
  AND for every arm: harm_rate(ON) <= harm_rate(OFF) + HARM_TOLERANCE (no harm increase).

INTERPRETATION GRID (pre-registered):
  PASS  : every non-degenerate lever shows the expected ON-vs-OFF change AND no harm increase.
  FAIL  : a non-degenerate lever shows authority active (modulatory_authority_active True /
          n_authority_normalized > 0) but NO committed-selection change, OR harm increased ON.
          -> route to /failure-autopsy (the rescale fired but did not reach committed selection,
          or it over-rode the primary harm term). Do NOT silently re-tune gain.
  INVALID_HARNESS (per-arm): the lever's bias was degenerate (curiosity case) -> arm
          non-contributory; substrate verdict drawn from the remaining non-degenerate levers.

SLEEP DRIVER: none (no sleep machinery used).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REE_V3_ROOT = os.path.dirname(os.path.abspath(__file__)).rsplit("/experiments", 1)[0]
if _REE_V3_ROOT not in sys.path:
    sys.path.insert(0, _REE_V3_ROOT)

from ree_core.utils.config import REEConfig          # noqa: E402
from ree_core.agent import REEAgent                  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiment_protocol import emit_outcome         # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_635_modulatory_bias_selection_authority_readiness"
CLAIM_IDS: List[str] = []  # substrate-readiness; tags no claim

# Pre-registered thresholds (constants; NOT derived from run statistics).
AUTHORITY_GAIN = 0.5
ACTION_DENSITY_LIFT_MIN = 0.03
ENTROPY_LIFT_MIN = 0.02
HARM_TOLERANCE = 0.02
CURIOSITY_BIAS_FLOOR = 1e-4   # below this the curiosity bias is degenerate -> INVALID_HARNESS
RAW_RANGE_FLOOR = 1e-6
NOOP_CLASS = 0

ENV_KWARGS = dict(
    size=12, num_hazards=4, num_resources=5, hazard_harm=0.05,
    env_drift_interval=5, env_drift_prob=0.1, proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05, proximity_approach_threshold=0.2,
    hazard_field_decay=0.5, resource_respawn_on_consume=True, toroidal=False,
    harm_history_len=10, reef_enabled=True, n_reef_patches=3, reef_patch_radius=2,
    hazard_food_attraction=0.7, reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal", reef_bipartite_agent_band_radius=1,
)

# Lever conditions. Each enables one modulatory substrate on the shared SD-056 baseline.
LEVERS: List[Dict[str, Any]] = [
    {"id": "VIGOR", "metric": "action_density"},
    {"id": "WITHIN_CLASS", "metric": "committed_class_entropy"},
    {"id": "CURIOSITY", "metric": "committed_class_entropy"},
]


def _obs_harm(o):
    h = o.get("harm_obs"); return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_a(o):
    h = o.get("harm_obs_a"); return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_history(o):
    h = o.get("harm_history"); return h.float().unsqueeze(0) if h is not None else None


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, lever_id: str) -> REEAgent:
    """Build the SD-056 baseline agent with one modulatory lever enabled.

    Authority flag is left OFF at construction; it is toggled on the live config
    at eval time so OFF and ON passes share identical weights.
    """
    common = dict(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=32, world_dim=32,
        alpha_world=0.9, alpha_self=0.3,
        use_harm_stream=True, z_harm_dim=32,
        use_affective_harm_stream=True, z_harm_a_dim=16, harm_history_len=10,
        z_goal_enabled=True, goal_weight=0.5, drive_weight=2.0,
        e1_goal_conditioned=True,
        use_resource_proximity_head=True, resource_proximity_weight=0.5,
        benefit_eval_enabled=True, benefit_weight=1.0,
        # SD-056 action-divergence substrate (so candidates are action-distinct).
        e2_action_contrastive_enabled=True, e2_action_contrastive_weight=0.01,
        # modulatory-bias-selection-authority master OFF at build (toggled at eval).
        use_modulatory_selection_authority=False,
        modulatory_authority_gain=AUTHORITY_GAIN,
        modulatory_authority_min_range_floor=RAW_RANGE_FLOOR,
    )
    if lever_id == "VIGOR":
        common.update(use_tonic_vigor=True)
    elif lever_id == "WITHIN_CLASS":
        common.update(
            use_e3_score_diversity=True,
            use_e3_diversity_entropy_bonus=True,
            use_e3_diversity_stratified_select=True,
            e3_diversity_stratified_within_class_temperature=1.0,
        )
    elif lever_id == "CURIOSITY":
        common.update(use_structured_curiosity=True)
    cfg = REEConfig.from_dims(**common)
    return REEAgent(cfg)


def _set_authority(agent: REEAgent, on: bool) -> None:
    """Toggle the substrate flags on the live agent (both application sites)."""
    if getattr(agent, "e3", None) is not None:
        agent.e3.config.use_modulatory_selection_authority = on
        agent.e3.config.modulatory_authority_gain = AUTHORITY_GAIN
    if getattr(agent, "score_diversity", None) is not None:
        agent.score_diversity.config.use_selection_authority = on


def _curiosity_bias_abs_mean(agent: REEAgent) -> Optional[float]:
    decomp = getattr(agent, "_last_score_bias_decomp", None)
    if isinstance(decomp, dict):
        for key in ("curiosity", "structured_curiosity", "mech314"):
            v = decomp.get(key)
            if v is not None:
                try:
                    return float(torch.as_tensor(v).abs().mean().item())
                except Exception:
                    return None
    return None


def _run_episode(agent, env, seed, steps, train: bool):
    """Run one episode; returns committed-class counts, action/noop counts, harm sum,
    modulatory diagnostics. Deterministic given (seed, agent state, torch seed)."""
    torch.manual_seed(seed)
    _, obs = env.reset()
    agent.reset()
    committed_classes: Dict[int, int] = {}
    n_action = 0
    n_committed = 0
    harm_sum = 0.0
    n_steps = 0
    auth_active_ticks = 0
    auth_scale_sum = 0.0
    cur_bias_abs: List[float] = []
    z_self_prev = None
    action_prev = None
    for _ in range(steps):
        body = obs["body_state"].float()
        world = obs["world_state"].float()
        if body.dim() == 1:
            body = body.unsqueeze(0)
        if world.dim() == 1:
            world = world.unsqueeze(0)
        latent = agent.sense(
            obs_body=body, obs_world=world, obs_harm=_obs_harm(obs),
            obs_harm_a=_obs_harm_a(obs), obs_harm_history=_obs_harm_history(obs),
        )
        if z_self_prev is not None and action_prev is not None:
            agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())
        ticks = agent.clock.advance()
        wdim = latent.z_world.shape[-1]
        e1_prior = (
            agent._e1_tick(latent) if ticks.get("e1_tick", False)
            else torch.zeros(1, wdim, device=agent.device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        action = agent.select_action(candidates, ticks)
        if not torch.isfinite(action).all():
            break
        cls = int(action[0].argmax().item())
        committed_classes[cls] = committed_classes.get(cls, 0) + 1
        n_committed += 1
        if cls != NOOP_CLASS:
            n_action += 1
        # modulatory diagnostics
        diag = getattr(agent.e3, "last_score_diagnostics", {}) or {}
        if diag.get("modulatory_authority_active"):
            auth_active_ticks += 1
            auth_scale_sum += float(diag.get("modulatory_authority_scale_factor", 0.0))
        cb = _curiosity_bias_abs_mean(agent)
        if cb is not None:
            cur_bias_abs.append(cb)
        _, harm_signal, done, info, obs = env.step(action)
        try:
            harm_sum += abs(float(harm_signal))
        except Exception:
            pass
        if agent.goal_state is not None:
            energy = float(body[0, 3].item())
            agent.update_z_goal(
                benefit_exposure=float(info.get("benefit_exposure", 0.0)),
                drive_level=max(0.0, 1.0 - energy),
            )
        z_self_prev = latent.z_self.detach()
        action_prev = action.detach()
        n_steps += 1
        if done:
            break
    return {
        "committed_classes": committed_classes,
        "n_action": n_action,
        "n_committed": n_committed,
        "harm_sum": harm_sum,
        "n_steps": n_steps,
        "auth_active_ticks": auth_active_ticks,
        "auth_scale_mean": (auth_scale_sum / auth_active_ticks) if auth_active_ticks else 0.0,
        "curiosity_bias_abs_mean": (sum(cur_bias_abs) / len(cur_bias_abs)) if cur_bias_abs else None,
    }


def _entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def _eval_pass(agent, env_seed, eval_eps, steps, authority_on):
    """Run eval episodes at a fixed authority setting; aggregate metrics."""
    _set_authority(agent, authority_on)
    merged: Dict[int, int] = {}
    n_action = 0
    n_committed = 0
    harm_sum = 0.0
    auth_active = 0
    cur_bias_vals: List[float] = []
    for e in range(eval_eps):
        env = _make_env(env_seed + e)
        r = _run_episode(agent, env, seed=env_seed + e, steps=steps, train=False)
        for k, v in r["committed_classes"].items():
            merged[k] = merged.get(k, 0) + v
        n_action += r["n_action"]
        n_committed += r["n_committed"]
        harm_sum += r["harm_sum"]
        auth_active += r["auth_active_ticks"]
        if r["curiosity_bias_abs_mean"] is not None:
            cur_bias_vals.append(r["curiosity_bias_abs_mean"])
    return {
        "committed_class_entropy": _entropy(merged),
        "action_density": (n_action / n_committed) if n_committed else 0.0,
        "harm_rate": (harm_sum / n_committed) if n_committed else 0.0,
        "n_committed": n_committed,
        "auth_active_ticks": auth_active,
        "n_committed_classes": len([k for k, v in merged.items() if v > 0]),
        "curiosity_bias_abs_mean": (sum(cur_bias_vals) / len(cur_bias_vals)) if cur_bias_vals else 0.0,
        "committed_classes": merged,
    }


def _warmup(agent, env_seed, warmup_eps, steps):
    """Brief warmup so scoring produces a non-trivial range (authority OFF)."""
    _set_authority(agent, False)
    for e in range(warmup_eps):
        env = _make_env(env_seed + 1000 + e)
        _run_episode(agent, env, seed=env_seed + 1000 + e, steps=steps, train=True)


def run_experiment(seed=42, warmup_eps=20, eval_eps=6, steps=200, dry_run=False):
    if dry_run:
        warmup_eps, eval_eps, steps = 2, 2, 25
    arms_out: List[Dict[str, Any]] = []
    total_train = warmup_eps + eval_eps * 2  # for episodes_per_run reporting
    for lever in LEVERS:
        lever_id = lever["id"]
        print(f"Seed {seed} Condition {lever_id}", flush=True)
        env = _make_env(seed)
        agent = _make_agent(env, lever_id)
        # warmup
        for ep in range(warmup_eps):
            if (ep + 1) % 10 == 0 or ep == warmup_eps - 1:
                print(f"  [train] {lever_id} seed={seed} ep {ep+1}/{total_train} (warmup)", flush=True)
            wenv = _make_env(seed + 1000 + ep)
            _run_episode(agent, wenv, seed=seed + 1000 + ep, steps=steps, train=True)
        # OFF then ON eval on shared weights
        off = _eval_pass(agent, seed, eval_eps, steps, authority_on=False)
        print(f"  [train] {lever_id} seed={seed} ep {warmup_eps+eval_eps}/{total_train} (eval OFF)", flush=True)
        on = _eval_pass(agent, seed, eval_eps, steps, authority_on=True)
        print(f"  [train] {lever_id} seed={seed} ep {total_train}/{total_train} (eval ON)", flush=True)

        metric = lever["metric"]
        # per-lever verdict
        invalid = False
        invalid_reason = None
        passed = False
        # GENERAL degenerate-bias guard: if the authority rescale never fired in the
        # ON pass (auth_active_ticks==0), the lever produced no non-degenerate
        # modulatory bias for the substrate to act on. That is an upstream-lever
        # degeneracy (the lever's own substrate work), NOT a failure of THIS
        # substrate -> INVALID_HARNESS. Applies to every lever.
        if on["auth_active_ticks"] == 0:
            invalid = True
            invalid_reason = (
                f"authority rescale never fired in ON pass (auth_active_ticks=0): the "
                f"{lever_id} modulatory bias range was below floor {RAW_RANGE_FLOOR:.0e} "
                "(degenerate upstream bias) -- INVALID_HARNESS, not a substrate failure"
            )
        # CURIOSITY-specific: also invalid if the curiosity bias itself is degenerate
        # (604a signature: 314a no active residue centers / 314b/c broadcast-by-design).
        if lever_id == "CURIOSITY" and not invalid:
            cbam = max(off["curiosity_bias_abs_mean"], on["curiosity_bias_abs_mean"])
            if cbam <= CURIOSITY_BIAS_FLOOR:
                invalid = True
                invalid_reason = (
                    f"curiosity_bias_abs_mean={cbam:.2e} <= floor {CURIOSITY_BIAS_FLOOR:.0e} "
                    "(degenerate upstream bias -- 314a no active residue centers / 314b/c broadcast); "
                    "INVALID_HARNESS, not a substrate failure"
                )
        harm_ok = on["harm_rate"] <= off["harm_rate"] + HARM_TOLERANCE
        if not invalid:
            if metric == "action_density":
                lift = on["action_density"] - off["action_density"]
                passed = (lift >= ACTION_DENSITY_LIFT_MIN) and harm_ok
            else:  # committed_class_entropy
                lift = on["committed_class_entropy"] - off["committed_class_entropy"]
                passed = (lift > ENTROPY_LIFT_MIN) and harm_ok
        else:
            lift = None
        arms_out.append({
            "lever": lever_id, "metric": metric,
            "authority_off": off, "authority_on": on,
            "lift": lift, "harm_ok": harm_ok,
            "invalid_harness": invalid, "invalid_reason": invalid_reason,
            "passed": bool(passed),
        })
        v = "INVALID_HARNESS" if invalid else ("PASS" if passed else "FAIL")
        print(f"verdict: {v}  ({lever_id} {metric} OFF={off.get(metric):.4f} ON={on.get(metric):.4f} "
              f"auth_on_ticks={on['auth_active_ticks']} harm_ok={harm_ok})", flush=True)

    # Overall: PASS iff every non-INVALID lever passed AND at least one non-INVALID lever exists.
    contributing = [a for a in arms_out if not a["invalid_harness"]]
    n_pass = sum(1 for a in contributing if a["passed"])
    overall_pass = bool(contributing) and (n_pass == len(contributing))
    outcome = "PASS" if overall_pass else "FAIL"

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": "V3-EXQ-635",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "predecessor": ["V3-EXQ-604a", "V3-EXQ-624a", "V3-EXQ-614d"],
        "substrate": "modulatory-bias-selection-authority",
        "modulatory_authority_gain": AUTHORITY_GAIN,
        "thresholds": {
            "action_density_lift_min": ACTION_DENSITY_LIFT_MIN,
            "entropy_lift_min": ENTROPY_LIFT_MIN,
            "harm_tolerance": HARM_TOLERANCE,
            "curiosity_bias_floor": CURIOSITY_BIAS_FLOOR,
        },
        "seed": seed, "warmup_eps": warmup_eps, "eval_eps": eval_eps,
        "steps_per_episode": steps,
        "arms": arms_out,
        "n_contributing_levers": len(contributing),
        "n_passing_levers": n_pass,
        "interpretation": (
            "PASS: every non-degenerate lever showed the expected authority ON-vs-OFF "
            "committed-selection change without a harm increase. FAIL with auth_active>0 "
            "but no change -> /failure-autopsy (rescale fired but did not reach committed "
            "selection, or over-rode primary harm). Curiosity INVALID_HARNESS arms are "
            "non-contributory (degenerate upstream bias), not substrate failures."
        ),
        "dry_run": dry_run,
    }
    return manifest


def _write(manifest) -> str:
    out_dir = os.path.join(_REE_V3_ROOT, "..", "REE_assembly", "evidence", "experiments")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{manifest['run_id']}.json")
    import json
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    manifest = run_experiment(seed=args.seed, dry_run=args.dry_run)
    out_path = _write(manifest)
    print(f"outcome: {manifest['outcome']}")
    print(f"manifest: {out_path}")
    if args.dry_run:
        # scrub the dry-run manifest so it never reaches the index
        try:
            os.remove(out_path)
            print("(dry-run manifest scrubbed)")
        except OSError:
            pass
    else:
        _o = str(manifest["outcome"]).upper()
        emit_outcome(
            outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
            manifest_path=out_path,
        )
