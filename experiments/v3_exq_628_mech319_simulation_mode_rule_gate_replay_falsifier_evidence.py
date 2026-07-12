"""V3-EXQ-628: MECH-319 simulation-mode rule-write gate -- replay/caller_sim falsifier EVIDENCE.

EXPERIMENT_PURPOSE = evidence. claim_ids = [MECH-319].

WHY THIS EXPERIMENT EXISTS
--------------------------
MECH-319 (policy.arbitration.simulation_mode_write_gating_substrate_ree_novel_function;
arc_062 GAP-K) is a substrate-level instantiation of MECH-094 at the rule-
arbitration layer: a categorical write gate, keyed to a simulation-mode tag,
that SUPPRESSES arbitration-weight updates (LateralPFCAnalog.rule_state and the
GatedPolicy site) during ghost / replay / DMN passes, while admitting waking
writes. Master flag use_simulation_mode_rule_gate; the V3-EXQ-543c falsifier
flag simulation_mode_rule_gate_admit_writes flips the gate to ADMIT simulation
writes.

The only post-landing run, V3-EXQ-546, is experiment_purpose=diagnostic /
evidence_direction=non_contributory: it verified the truth-table semantics and
the waking-path wiring contract, but it never touched rule_state and it carries
ZERO governance weight. Governance needs an experiment_purpose=evidence,
MECH-319-tagged PASS to move MECH-319 off candidate_substrate_landed.

THE DESIGN FORK (resolved here)
-------------------------------
The MECH-319 gate is bit-identical on the waking path: REEAgent.select_action
ALWAYS consults the gate with simulation_mode=False at both wired arbitration-
write sites (agent.py GatedPolicy block + LateralPFCAnalog block), and for a
waking caller the gate returns False (admit) REGARDLESS of admit_writes. The
falsifier asymmetry (block vs admit a simulation-mode rule write) surfaces ONLY
when the gate is invoked with caller_sim=True -- a replay / ghost-goal / DMN
tick -- and NOTHING in the agent currently drives the arbitration-write sites
with caller_sim=True (that is the V3-EXQ-543c replay-driven invocation path the
substrate docstring explicitly reserves).

So this evidence experiment CONSTRUCTS that replay invocation path. It uses the
REAL agent gate object (agent.simulation_mode_rule_gate) and the REAL consumer
(agent.lateral_pfc.update) and replicates the agent.py:3356-3404 LateralPFC
wiring verbatim -- the only change being the literal simulation_mode=False
replaced by simulation_mode=True, exactly the seam the docstring names. It then
measures that LateralPFCAnalog.rule_state DIVERGES between two configs under
this replay invocation:

  ARM BLOCK  (master ON, admit_writes=False): gate returns True under
             caller_sim=True -> the simulation write is BLOCKED -> rule_state
             unchanged by replay (MECH-319 normal behaviour).
  ARM ADMIT  (master ON, admit_writes=True):  gate returns False under
             caller_sim=True -> the simulation write is ADMITTED -> rule_state
             updated by replay (V3-EXQ-543c falsifier control).

Both arms replay the SAME captured early-waking latent, so the rule_state
divergence is attributable to the gate alone.

PRE-REGISTERED ACCEPTANCE GRID (per seed)
-----------------------------------------
  C0 path-exercised / no-vacuity: both arms instantiate the gate + lateral_pfc;
     waking rule_state norm > MIN_WAKING_RULE_NORM (there is a real write to
     suppress); the replay path actually fired (BLOCK gate n_simulation_blocked
     == N_REPLAY AND ADMIT gate n_simulation_admitted == N_REPLAY).
  C1 BLOCK suppresses: BLOCK rule_state delta across the replay phase
     <= EPS_BLOCK (~0) AND its gate n_simulation_blocked == N_REPLAY AND
     n_simulation_admitted == 0.
  C2 ADMIT admits: ADMIT rule_state delta across the replay phase
     >= ADMIT_DELTA_THRESHOLD (> 0) AND its gate n_simulation_admitted ==
     N_REPLAY AND n_simulation_blocked == 0.
  C3 divergence: ||rule_state_ADMIT - rule_state_BLOCK|| after replay
     >= DIVERGENCE_THRESHOLD.
  C4 waking bit-identical falsifier (LOAD-BEARING): after the matched waking
     phase and BEFORE replay, ||rule_state_BLOCK - rule_state_ADMIT|| <=
     EPS_WAKING AND both gates show ZERO simulation firings on the waking path
     (admit_writes has no effect on waking -- the asymmetry is exclusively a
     caller_sim=True phenomenon).
  seed PASS = C0 AND C1 AND C2 AND C3 AND C4.
  overall PASS = majority of seeds pass (>= ceil(2/3 * n_seeds)).

DIAGNOSTIC INTERPRETATION GRID (one row per plausible outcome -> next action)
----------------------------------------------------------------------------
  PASS (all C): MECH-319 supported. The categorical simulation-mode tag gates
     the arbitration-weight update on a real rule-state-bearing consumer;
     admit_writes is the load-bearing falsifier control; the waking path is
     unaffected. -> evidence_direction=supports. Governance may move MECH-319
     off candidate_substrate_landed.
  C0 fail (path not exercised / waking norm ~0): harness/wiring defect, NOT a
     MECH-319 result. -> non_contributory; route /diagnose-errors.
  C4 fail (waking NOT bit-identical, or nonzero simulation firings on waking):
     admit_writes leaked into the waking path -> a substrate BUG violating the
     bit-identical-waking architectural commitment. -> route /diagnose-errors
     (this would falsify a different invariant than MECH-319's claim).
  C1 fail (BLOCK does not suppress): the gate fails to block a simulation write
     -> MECH-319 falsified. -> evidence_direction=weakens / does_not_support.
  C2 fail (ADMIT does not admit): the falsifier flag is inert -> MECH-319's
     falsifiable structure is vacuous. -> does_not_support / non_contributory.
  C3 fail (no divergence with C1/C2 ok): block and admit reach the same
     rule_state -> the gate has no functional consequence. -> does_not_support.

PREDECESSOR
-----------
  V3-EXQ-546 (v3_exq_546_mech319_simulation_mode_rule_gate_substrate_readiness):
  diagnostic / non_contributory substrate-readiness (truth-table + waking wiring
  contract; never touched rule_state). This experiment is the evidence-grade
  successor that demonstrates the rule_state-level write-suppression consequence
  on the real LateralPFCAnalog consumer. NOT a supersession -- 546 stays a valid
  substrate-readiness record.

PHASED TRAINING / MECH-094
--------------------------
  No phased training: MECH-319 is pure boolean/counter arithmetic and the
  LateralPFCAnalog rule_state is a non-trainable EMA buffer (no encoder head, no
  gradient flow). MECH-094: this experiment IS the simulation-mode (caller_sim=
  True) probe; the gate's normal-mode block is precisely MECH-094's hypothesis-
  tag write suppression realised at the arbitration layer.

Run:
  /opt/local/bin/python3 experiments/v3_exq_628_mech319_simulation_mode_rule_gate_replay_falsifier_evidence.py [--dry-run]

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.regulators import SITE_LATERAL_PFC
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_PURPOSE = "evidence"

# --- Pre-registered thresholds (constants; NOT derived from run statistics) ---
SEEDS = [0, 1, 2]
WAKING_STEPS = 60          # denominator M in the [train] ep N/M prints
N_REPLAY = 30              # replay/caller_sim=True invocations per arm
REPLAY_CAPTURE_STEP = 5    # waking step whose latent is replayed (a stored past state)
EPS_WAKING = 1e-6          # C4: max ||rs_BLOCK - rs_ADMIT|| on the waking path
EPS_BLOCK = 1e-7           # C1: max BLOCK-arm rule_state delta across replay (~0)
ADMIT_DELTA_THRESHOLD = 1e-4   # C2: min ADMIT-arm rule_state delta across replay
DIVERGENCE_THRESHOLD = 1e-4    # C3: min ||rs_ADMIT - rs_BLOCK|| after replay
MIN_WAKING_RULE_NORM = 1e-4    # C0: rule_state developed something to suppress


def _build_obs_kwargs(obs_dict, cfg):
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    kwargs = {"obs_body": body, "obs_world": world}
    obs_harm = obs_dict.get("harm_obs")
    if obs_harm is not None and getattr(cfg.latent, "use_harm_stream", False):
        if obs_harm.dim() == 1:
            obs_harm = obs_harm.unsqueeze(0)
        kwargs["obs_harm"] = obs_harm
    obs_harm_a = obs_dict.get("harm_obs_a")
    if obs_harm_a is not None and getattr(cfg.latent, "use_affective_harm_stream", False):
        if obs_harm_a.dim() == 1:
            obs_harm_a = obs_harm_a.unsqueeze(0)
        kwargs["obs_harm_a"] = obs_harm_a
    return kwargs


def _build_agent(env, seed, admit_writes):
    """Build a MECH-319-ON agent. Reseed before build so both arms get
    identical frozen-random lateral_pfc weights (precondition for C4)."""
    torch.manual_seed(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        use_simulation_mode_rule_gate=True,
        simulation_mode_rule_gate_admit_writes=admit_writes,
        use_gated_policy=True,
        use_lateral_pfc_analog=True,
    )
    agent = REEAgent(cfg)
    agent.reset()
    return agent, cfg


def run_seed(seed, waking_steps, n_replay, capture_step):
    """Run one seed: matched waking phase on BLOCK + ADMIT arms, then a
    replay/caller_sim=True phase, then the acceptance computation."""
    import random as _random

    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1, use_proxy_fields=True,
    )
    agent_block, cfg = _build_agent(env, seed, admit_writes=False)
    agent_admit, _ = _build_agent(env, seed, admit_writes=True)

    env_rng = _random.Random(seed * 1000 + 7)
    _flat, obs_dict = env.reset()
    captured = None

    # --- Waking phase: identical obs stream to both arms. The env trajectory
    #     is driven by a seeded RNG (independent of agent action sampling) so
    #     both arms see a bit-identical obs sequence -> bit-identical waking
    #     rule_state (rule_state updates from the latent, not the action). ---
    for step in range(waking_steps):
        kwargs = _build_obs_kwargs(obs_dict, cfg)
        ob, ow = kwargs["obs_body"], kwargs["obs_world"]
        with torch.no_grad():
            agent_block.act_with_split_obs(ob, ow, temperature=1.0)
            agent_admit.act_with_split_obs(ob, ow, temperature=1.0)
        if step == capture_step:
            lat = agent_block._current_latent
            captured = (
                lat.z_delta.detach().clone(),
                lat.z_world.detach().clone(),
            )
        a = env_rng.randrange(4)
        action_t = torch.zeros(4)
        action_t[a] = 1.0
        _flat, _h, done, _info, obs_dict = env.step(action_t)
        if done:
            _flat, obs_dict = env.reset()
        if (step + 1) % max(1, waking_steps // 4) == 0 or (step + 1) == waking_steps:
            print(f"  [train] mech319 seed={seed} ep {step + 1}/{waking_steps}", flush=True)

    if captured is None:
        # capture_step >= waking_steps (only at degenerate dry scale): fall back
        # to the final latent so the replay phase still has content.
        lat = agent_block._current_latent
        captured = (lat.z_delta.detach().clone(), lat.z_world.detach().clone())

    # --- Snapshot post-waking state (before any replay) ---
    rs_block_wak = agent_block.lateral_pfc.rule_state.detach().clone()
    rs_admit_wak = agent_admit.lateral_pfc.rule_state.detach().clone()
    waking_diff = float((rs_block_wak - rs_admit_wak).norm().item())
    waking_norm_block = float(rs_block_wak.norm().item())
    waking_norm_admit = float(rs_admit_wak.norm().item())
    gb_wak = agent_block.simulation_mode_rule_gate.get_state()
    ga_wak = agent_admit.simulation_mode_rule_gate.get_state()
    waking_sim_firings_block = gb_wak["n_simulation_blocked"] + gb_wak["n_simulation_admitted"]
    waking_sim_firings_admit = ga_wak["n_simulation_blocked"] + ga_wak["n_simulation_admitted"]

    # --- Replay / caller_sim=True phase. Verbatim agent.py LateralPFC wiring
    #     (gate consult -> if not eff_sim: lateral_pfc.update), the only change
    #     being simulation_mode=True. Identical captured replay content for
    #     both arms isolates the gate as the sole difference. ---
    zd, zw = captured
    for _ in range(n_replay):
        for agent in (agent_block, agent_admit):
            gate = agent.simulation_mode_rule_gate
            eff_sim = gate.effective_simulation_mode(
                simulation_mode=True, site=SITE_LATERAL_PFC
            )
            if not eff_sim:
                agent.lateral_pfc.update(z_delta=zd, z_world=zw, gate=1.0)

    rs_block_post = agent_block.lateral_pfc.rule_state.detach().clone()
    rs_admit_post = agent_admit.lateral_pfc.rule_state.detach().clone()
    block_delta = float((rs_block_post - rs_block_wak).norm().item())
    admit_delta = float((rs_admit_post - rs_admit_wak).norm().item())
    divergence = float((rs_admit_post - rs_block_post).norm().item())
    gb2 = agent_block.simulation_mode_rule_gate.get_state()
    ga2 = agent_admit.simulation_mode_rule_gate.get_state()
    block_n_blocked = gb2["n_simulation_blocked"]
    block_n_admitted = gb2["n_simulation_admitted"]
    admit_n_blocked = ga2["n_simulation_blocked"]
    admit_n_admitted = ga2["n_simulation_admitted"]

    # --- Acceptance criteria ---
    c0 = bool(
        agent_block.simulation_mode_rule_gate is not None
        and agent_admit.simulation_mode_rule_gate is not None
        and agent_block.lateral_pfc is not None
        and agent_admit.lateral_pfc is not None
        and waking_norm_block > MIN_WAKING_RULE_NORM
        and block_n_blocked == n_replay
        and admit_n_admitted == n_replay
    )
    c1 = bool(
        block_delta <= EPS_BLOCK
        and block_n_blocked == n_replay
        and block_n_admitted == 0
    )
    c2 = bool(
        admit_delta >= ADMIT_DELTA_THRESHOLD
        and admit_n_admitted == n_replay
        and admit_n_blocked == 0
    )
    c3 = bool(divergence >= DIVERGENCE_THRESHOLD)
    c4 = bool(
        waking_diff <= EPS_WAKING
        and waking_sim_firings_block == 0
        and waking_sim_firings_admit == 0
    )
    seed_pass = bool(c0 and c1 and c2 and c3 and c4)

    return {
        "seed": seed,
        "pass": seed_pass,
        "C0_path_exercised_no_vacuity": c0,
        "C1_block_suppresses": c1,
        "C2_admit_admits": c2,
        "C3_divergence": c3,
        "C4_waking_bit_identical_falsifier": c4,
        "waking_rule_state_diff": waking_diff,
        "waking_rule_norm_block": waking_norm_block,
        "waking_rule_norm_admit": waking_norm_admit,
        "waking_sim_firings_block": waking_sim_firings_block,
        "waking_sim_firings_admit": waking_sim_firings_admit,
        "replay_block_rule_state_delta": block_delta,
        "replay_admit_rule_state_delta": admit_delta,
        "post_replay_divergence": divergence,
        "block_gate_n_simulation_blocked": block_n_blocked,
        "block_gate_n_simulation_admitted": block_n_admitted,
        "admit_gate_n_simulation_blocked": admit_n_blocked,
        "admit_gate_n_simulation_admitted": admit_n_admitted,
        "n_replay": n_replay,
        "waking_steps": waking_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        seeds = [0]
        waking_steps = 8
        n_replay = 5
        capture_step = 2
    else:
        seeds = SEEDS
        waking_steps = WAKING_STEPS
        n_replay = N_REPLAY
        capture_step = REPLAY_CAPTURE_STEP

    t0 = time.time()
    per_seed = []
    for seed in seeds:
        print(f"Seed {seed} Condition base", flush=True)
        res = run_seed(seed, waking_steps, n_replay, capture_step)
        per_seed.append(res)
        print(f"verdict: {'PASS' if res['pass'] else 'FAIL'}", flush=True)

    n_pass = sum(1 for r in per_seed if r["pass"])
    required = math.ceil(2.0 / 3.0 * len(seeds))
    all_pass = n_pass >= required
    elapsed = time.time() - t0

    run_id_base = "v3_exq_628_mech319_simulation_mode_rule_gate_replay_falsifier_evidence_v3"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{run_id_base}_{ts}",
        "experiment_type": "v3_exq_628_mech319_simulation_mode_rule_gate_replay_falsifier_evidence",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": ["MECH-319"],
        "outcome": "PASS" if all_pass else "FAIL",
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "MECH-319": "supports" if all_pass else "weakens",
        },
        "predecessor": "V3-EXQ-546",
        "metrics": {
            "n_seeds": len(seeds),
            "n_pass": n_pass,
            "required_pass": required,
            "per_seed": per_seed,
            "thresholds": {
                "EPS_WAKING": EPS_WAKING,
                "EPS_BLOCK": EPS_BLOCK,
                "ADMIT_DELTA_THRESHOLD": ADMIT_DELTA_THRESHOLD,
                "DIVERGENCE_THRESHOLD": DIVERGENCE_THRESHOLD,
                "MIN_WAKING_RULE_NORM": MIN_WAKING_RULE_NORM,
                "N_REPLAY": n_replay,
                "WAKING_STEPS": waking_steps,
            },
        },
        "diagnostic_interpretation": {
            "PASS": "MECH-319 supported: categorical simulation-mode tag gates the arbitration rule_state write on the real LateralPFCAnalog consumer; admit_writes is the load-bearing falsifier control; waking path unaffected. evidence_direction=supports.",
            "C0_fail": "Path not exercised / waking rule_state ~0: harness defect, NOT a MECH-319 result -> non_contributory; /diagnose-errors.",
            "C4_fail": "Waking path not bit-identical (admit_writes leaked into waking): substrate BUG violating bit-identical-waking commitment -> /diagnose-errors.",
            "C1_fail": "BLOCK does not suppress the simulation write -> MECH-319 falsified -> weakens / does_not_support.",
            "C2_fail": "ADMIT does not admit (falsifier flag inert) -> MECH-319 falsifiable structure vacuous -> does_not_support / non_contributory.",
            "C3_fail": "No divergence with C1/C2 ok -> gate has no functional consequence -> does_not_support.",
        },
        "elapsed_seconds": elapsed,
        "dry_run": bool(args.dry_run),
        "notes": (
            "MECH-319 simulation-mode rule-write gate EVIDENCE (replay/caller_sim "
            "falsifier). Constructs the V3-EXQ-543c replay-driven invocation path "
            "(gate.effective_simulation_mode(simulation_mode=True, site=lateral_pfc) "
            "+ conditional lateral_pfc.update -- the agent.py:3356-3404 wiring with "
            "the literal False replaced by True) that select_action never exercises, "
            "and measures rule_state divergence between BLOCK (admit_writes=False) "
            "and ADMIT (admit_writes=True) arms replaying the SAME captured early-"
            "waking latent. PASS = BLOCK suppresses (delta~0) AND ADMIT admits "
            "(delta>0) AND arms diverge AND the waking path is bit-identical across "
            "arms with zero simulation firings (the load-bearing C4 falsifier). "
            "Predecessor V3-EXQ-546 was diagnostic/non_contributory (truth-table + "
            "waking wiring; never touched rule_state); this is the evidence-grade "
            "successor demonstrating the rule_state-level write-suppression "
            "consequence MECH-319 asserts. See ree-v3/CLAUDE.md 'MECH-319 (arc_062 "
            "GAP-K)' and docs/architecture/mech_319_simulation_mode_rule_gate.md."
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

    print(f"OVERALL: {manifest['result']}", flush=True)
    print(f"  n_pass={n_pass}/{len(seeds)} (required {required})", flush=True)
    for r in per_seed:
        print(
            f"  seed {r['seed']}: pass={r['pass']} "
            f"C0={r['C0_path_exercised_no_vacuity']} C1={r['C1_block_suppresses']} "
            f"C2={r['C2_admit_admits']} C3={r['C3_divergence']} "
            f"C4={r['C4_waking_bit_identical_falsifier']} "
            f"block_delta={r['replay_block_rule_state_delta']:.3e} "
            f"admit_delta={r['replay_admit_rule_state_delta']:.3e} "
            f"divergence={r['post_replay_divergence']:.3e}",
            flush=True,
        )
    print(f"Result written to: {out_path}", flush=True)

    return manifest["result"], str(out_path)


if __name__ == "__main__":
    _result, _manifest_path = main()
    emit_outcome(outcome=_result, manifest_path=_manifest_path)
