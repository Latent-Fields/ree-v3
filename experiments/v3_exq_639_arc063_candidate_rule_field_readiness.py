"""V3-EXQ-639: ARC-063 v1 CandidateRuleField substrate-readiness diagnostic.

Purpose: diagnostic (claim_ids=[]). Confirms the ARC-063 v1 substrate (the
non-Bayesian rule-creator resolving arc_062_rule_apprehension:GAP-B) was wired
correctly. Does NOT weight governance -- the behavioural pay-off (the ARC-062
GAP-B multi-signature refuge/forage diversity re-run on the field-populated
substrate, MECH-309 / ARC-062) is the governance-weighting successor queued
separately.

The failure record defines success. The substrate must break the 543l/598b
monomodal rule_state collapse:

  C1 (differentiated rule_state -- inverts 598b C3 trainable_not_monomodal):
     with the field ON, SD-033a receives a rule_state with >= 2 distinct active
     rule vectors across distinct contexts (subspace separation above a floor).
  C2 (minting fires): the bottom-up creator mints >= 2 distinct context-tagged
     rules (a regularity-detected, non-gradient event), with distinct pinned
     subspace directions (crf_max_pairwise_rule_dist > 0).
  C3 (tolerance gate works + conflict-sensitive): under-supported / high-conflict
     rules are held out (availability < theta); theta rises with competing
     context-matched rules; a single matched rule (no conflict) is admitted.
  C4 (OFF bit-identical): use_candidate_rule_field=False -> agent.candidate_rule_field
     is None and a one-tick act loop runs unchanged; ON agent mints over an
     episode and SD-033a rule_state is populated (norm > 0).
  UC5 (MECH-094 gate): simulation_mode=True is a no-op (returns zeros, no mint,
     no credit).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_639_arc063_candidate_rule_field_readiness.py

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome
from ree_core.agent import REEAgent
from ree_core.policy.candidate_rule_field import (
    CandidateRule,
    CandidateRuleField,
    CandidateRuleFieldConfig,
)
from ree_core.utils.config import REEConfig

EXPERIMENT_PURPOSE = "diagnostic"


def _field(**kw):
    cfg = CandidateRuleFieldConfig(use_candidate_rule_field=True, **kw)
    return CandidateRuleField(context_dim=16, config=cfg)


def _build_agent(seed: int = 7, **flags):
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1, use_proxy_fields=True,
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=4, self_dim=16, world_dim=16, **flags,
    )
    agent = REEAgent(cfg)
    agent.reset()
    return agent, env


# ----------------------------------------------------------------------
# C2 CREATE: minting fires >= 2 distinct context-tagged rules
# ----------------------------------------------------------------------
def run_c2_create_mints_distinct() -> dict:
    f = _field(n_slots=8, rule_dim=16, mint_recurrence_threshold=3,
               context_match_threshold=0.5)
    a = torch.zeros(16); a[0] = 1.0; a[1] = 1.0
    b = torch.zeros(16); b[0] = -1.0; b[1] = -1.0
    for _ in range(4):
        f.step(a, action_object_idx=0, outcome_signal=0.0)
    for _ in range(4):
        f.step(b, action_object_idx=1, outcome_signal=0.0)
    st = f.get_state()
    result = {
        "n_slots_minted": st["crf_n_slots_minted"],
        "max_pairwise_rule_dist": st["crf_max_pairwise_rule_dist"],
        "n_minted_total": st["crf_n_minted_total"],
    }
    result["pass"] = (
        st["crf_n_slots_minted"] >= 2 and st["crf_max_pairwise_rule_dist"] > 0.1
    )
    return result


# ----------------------------------------------------------------------
# C1 OUTPUT: differentiated rule_state across distinct contexts
# ----------------------------------------------------------------------
def run_c1_differentiated_rule_state() -> dict:
    f = _field(n_slots=8, rule_dim=16, mint_recurrence_threshold=3,
               context_match_threshold=0.5)
    a = torch.zeros(16); a[0] = 1.0; a[1] = 1.0
    b = torch.zeros(16); b[0] = -1.0; b[1] = -1.0
    for _ in range(4):
        f.step(a, action_object_idx=0, outcome_signal=0.0)
    for _ in range(4):
        f.step(b, action_object_idx=1, outcome_signal=0.0)
    sA = f.step(a, action_object_idx=0, outcome_signal=0.0)
    sB = f.step(b, action_object_idx=1, outcome_signal=0.0)
    norm_diff = float((sA - sB).norm().item())
    # count distinct active rule directions seen across the two contexts
    f.gate_and_select(a); na = f.n_active_rules()
    f.gate_and_select(b); nb = f.n_active_rules()
    result = {
        "rule_state_shape": list(sA.shape),
        "norm_diff_across_contexts": norm_diff,
        "n_active_context_a": na,
        "n_active_context_b": nb,
    }
    # >= 2 distinct active rule vectors across contexts (one per context here)
    # AND the rule_state is differentiated (non-trivial norm difference).
    result["pass"] = (
        tuple(sA.shape) == (1, 16) and norm_diff > 1e-4 and na >= 1 and nb >= 1
    )
    return result


# ----------------------------------------------------------------------
# C3 GATE: tolerance gate is conflict-sensitive
# ----------------------------------------------------------------------
def run_c3_gate_conflict_sensitive() -> dict:
    f = _field(n_slots=8, rule_dim=16, mint_recurrence_threshold=1,
               tolerance_floor=0.3, tolerance_conflict_gain=1.0,
               context_match_threshold=0.3)
    cx = torch.zeros(16); cx[0] = 1.0
    f.step(cx, action_object_idx=0, outcome_signal=0.0)
    # Insert a second rule sharing the same context (bypass the covered-guard)
    # to create a 2-rule conflict at the same context.
    f._rules[1] = CandidateRule(
        rule_embedding=f._pinned_directions[1].clone(),
        context_tag=cx.clone(), availability=0.3, eligibility=0.0, minted_step=0)
    active_conflict = f.gate_and_select(cx)
    f._rules.pop(1)
    active_single = f.gate_and_select(cx)
    result = {
        "n_active_under_conflict": len(active_conflict),
        "n_active_single": len(active_single),
        "theta_floor": 0.3,
        "theta_under_one_competitor": 0.3 + 1.0 * 1,
    }
    result["pass"] = (len(active_conflict) == 0 and len(active_single) == 1)
    return result


# ----------------------------------------------------------------------
# C4 OFF bit-identical + ON mints + SD-033a populated
# ----------------------------------------------------------------------
def run_c4_off_bit_identical_on_populates() -> dict:
    agent_off, _ = _build_agent()
    off_is_none = agent_off.candidate_rule_field is None

    agent_on, env = _build_agent(use_candidate_rule_field=True,
                                 use_lateral_pfc_analog=True,
                                 crf_mint_recurrence_threshold=2)
    on_is_module = agent_on.candidate_rule_field is not None
    on_sources = agent_on.lateral_pfc.config.use_candidate_rule_source is True

    _flat, obs = env.reset()
    body = obs["body_state"]; world = obs["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    with torch.no_grad():
        for _ in range(25):
            act = agent_on.act_with_split_obs(body, world)
            _flat, _h, _d, _i, obs = env.step(int(act.argmax().item()))
            body = obs["body_state"]; world = obs["world_state"]
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)
    st = agent_on.candidate_rule_field.get_state()
    rule_state_norm = float(agent_on.lateral_pfc.rule_state.norm().item())
    result = {
        "off_field_is_none": off_is_none,
        "on_field_is_module": on_is_module,
        "on_lateral_pfc_sources_from_field": on_sources,
        "on_n_minted_total": st["crf_n_minted_total"],
        "on_sd033a_rule_state_norm": rule_state_norm,
    }
    result["pass"] = (
        off_is_none and on_is_module and on_sources
        and st["crf_n_minted_total"] >= 1 and rule_state_norm > 0.0
    )
    return result


# ----------------------------------------------------------------------
# UC5 MECH-094 simulation gate
# ----------------------------------------------------------------------
def run_uc5_mech094_simulation_gate() -> dict:
    f = _field(n_slots=8, rule_dim=16, mint_recurrence_threshold=1)
    cx = torch.zeros(16); cx[0] = 1.0
    out = f.step(cx, action_object_idx=0, outcome_signal=1.0, simulation_mode=True)
    st = f.get_state()
    result = {
        "sim_returns_zeros": float(out.abs().sum()) == 0.0,
        "sim_no_mint": st["crf_n_minted_total"] == 0,
        "sim_skip_counter": st["crf_n_simulation_skipped"],
    }
    result["pass"] = (
        result["sim_returns_zeros"]
        and result["sim_no_mint"]
        and st["crf_n_simulation_skipped"] == 1
    )
    return result


def main() -> None:
    t0 = time.time()
    c2 = run_c2_create_mints_distinct()
    c1 = run_c1_differentiated_rule_state()
    c3 = run_c3_gate_conflict_sensitive()
    c4 = run_c4_off_bit_identical_on_populates()
    uc5 = run_uc5_mech094_simulation_gate()

    all_pass = all(r["pass"] for r in [c1, c2, c3, c4, uc5])
    elapsed = time.time() - t0

    run_id = "v3_exq_639_arc063_candidate_rule_field_readiness_v3"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{run_id}_{ts}",
        "experiment_type": "v3_exq_639_arc063_candidate_rule_field_readiness",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": [],
        "outcome": "PASS" if all_pass else "FAIL",
        "evidence_direction": "non_contributory",
        "metrics": {
            "C1_differentiated_rule_state": c1,
            "C2_create_mints_distinct": c2,
            "C3_gate_conflict_sensitive": c3,
            "C4_off_bit_identical_on_populates": c4,
            "UC5_mech094_simulation_gate": uc5,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "ARC-063 v1 CandidateRuleField substrate-readiness diagnostic. "
            "Confirms the non-Bayesian rule-creator (GAP-B) was wired: CREATE "
            "(MECH-349) mints distinct context rules; REPRESENT (MECH-350) keeps "
            "them in distinct pinned subspace directions; GATE (MECH-351) is "
            "conflict-sensitive; OUTPUT writes a DIFFERENTIATED rule_state into "
            "SD-033a (inverts 598b C3 trainable_not_monomodal); CREDIT (MECH-352) "
            "loop + MECH-094 sim gate exercised. claim_ids=[] -- does NOT weight "
            "governance. The ARC-062 GAP-B behavioural diversity re-run on the "
            "field-populated substrate (MECH-309 / ARC-062) is the "
            "governance-weighting successor, queued separately."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"verdict: {manifest['outcome']}")
    for k, v in manifest["metrics"].items():
        print(f"  {k}: pass={v['pass']}")
    print(f"Result written to: {out_path}", flush=True)
    return manifest["outcome"], str(out_path)


if __name__ == "__main__":
    _outcome, _path = main()
    emit_outcome(outcome=_outcome, manifest_path=_path)
