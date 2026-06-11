"""V3-EXQ-668: MECH-319 simulation-mode rule-write gate -- ACCUMULATED ghost-write
arbitration-DRIFT / overcommitment falsifier EVIDENCE.

EXPERIMENT_PURPOSE = evidence. claim_ids = [MECH-319].

WHY THIS EXPERIMENT EXISTS (distinct from V3-EXQ-628)
----------------------------------------------------
MECH-319 (policy.arbitration.simulation_mode_write_gating_substrate_ree_novel_
function; arc_062 GAP-K) is a categorical write gate, keyed to a simulation-mode
tag, that SUPPRESSES arbitration-weight (LateralPFCAnalog.rule_state) updates
during ghost / replay / DMN passes while admitting waking writes.

V3-EXQ-628 (predecessor; PASS/supports 2026-06-02) tested ONE pre-registered V3
falsification path: SINGLE-replay rule_state divergence -- BLOCK (admit_writes=
False) vs ADMIT (admit_writes=True) on ONE captured latent, measuring the
immediate rule_state delta of a single gated write.

This experiment tests MECH-319's OTHER pre-registered V3 falsification path,
quoted verbatim from claims.yaml MECH-319 functional_restatement:

  "a config flag that artificially routes simulation-mode passes through the
   write channel anyway should produce OVERCOMMITMENT to ghost-derived
   arbitration weights and observable rule-arbitration DRIFT; with MECH-319
   enabled, no drift."

That is the ACCUMULATED / behavioural consequence over a SUSTAINED replay-heavy
regime -- not a single step. 668 measures (a) the accumulated rule_state drift
trajectory across many interleaved replay passes and (b) whether that drift
reaches the arbitration OUTPUT (LateralPFCAnalog.compute_bias) -- the observable
rule-arbitration drift the claim asserts. 668 does NOT supersede 628; both are
distinct contributory evidence entries (the min_experimental_entries=2 promotion
gate on MECH-319 needs a SECOND contributory entry, and a re-run of the 628
replay slice would not be one).

THE DESIGN
----------
Two arms, master ON, identical seed (identical frozen-random lateral_pfc weights
AND -- because lateral_pfc_train_rule_bias_head=True -- identical non-zero bias
head, so compute_bias is a live readout of rule_state rather than the default
zeroed-head 0):

  ARM BLOCK  (master ON, admit_writes=False): gate returns True under
             caller_sim=True -> every simulation write BLOCKED -> rule_state
             stays at its waking baseline (MECH-319 normal: no drift).
  ARM ADMIT  (master ON, admit_writes=True):  gate returns False under
             caller_sim=True -> every simulation write ADMITTED -> rule_state
             drifts toward the ghost-derived (replayed) content (the V3-EXQ-543c
             falsifier: overcommitment to non-realized data).

Phase 1 (matched waking warmup): identical seeded obs stream to both arms ->
  bit-identical waking rule_state. Capture several distinct early-waking latents.
Phase 2 (sustained replay regime): N_REPLAY_ROUNDS rounds; each round replays the
  captured latents round-robin through the gate -> conditional lateral_pfc.update
  (caller_sim=True). Per-round divergence ||rs_ADMIT - rs_BLOCK|| tracked across
  the whole regime. BLOCK stays ~0; ADMIT accumulates (drift trajectory).
Phase 3 (behavioural overcommitment probe): a FIXED candidate set is scored by
  each arm's (now-drifted) lateral_pfc.compute_bias WITHOUT any further update.
  The ADMIT arm's bias diverges from BLOCK by an amount carried by the
  accumulated ghost drift -- the rule-arbitration drift reaching the output E3
  consumes.

PRE-REGISTERED ACCEPTANCE GRID (per seed)
-----------------------------------------
  C0 path-exercised / no-vacuity: both arms instantiate gate + lateral_pfc;
     waking rule_state norm > MIN_WAKING_RULE_NORM (a real arbitration state that
     CAN drift); the replay path fired N_REPLAY times in each arm (BLOCK
     n_simulation_blocked == N_REPLAY AND ADMIT n_simulation_admitted == N_REPLAY).
  C4 waking bit-identical falsifier (LOAD-BEARING, retained from 628): BEFORE
     replay, ||rs_BLOCK - rs_ADMIT|| <= EPS_WAKING AND both gates show ZERO
     simulation firings on the waking path (admit_writes has NO effect on waking;
     the asymmetry is exclusively a caller_sim=True phenomenon).
  C_ADMIT_LIVE readiness positive control (same-statistic non-vacuity gate, the
     V3-EXQ-643 lesson): the FIRST admitted replay produces a non-zero ADMIT
     rule_state delta >= ADMIT_DELTA_THRESHOLD. Confirms the admit write channel
     is live before the accumulation criterion is judged. If this fails the
     substrate is not ready (admit path inert) -> substrate_not_ready_requeue,
     NEVER a MECH-319 weakening.
  C1 BLOCK no drift: ADMIT-vs-BLOCK aside, the BLOCK arm's own drift from its
     waking snapshot stays <= EPS_BLOCK across the ENTIRE sustained regime AND
     its gate admitted ZERO simulation writes.
  C2 ADMIT accumulates (the distinct-from-628 criterion): accumulated ADMIT drift
     from its waking baseline >= ACCUM_THRESHOLD AND the final cross-arm
     divergence exceeds the single-replay divergence by >= ACCUM_GROWTH_FACTOR
     (drift GROWS over the sustained regime, it is not a single step).
  C3 drift reaches the arbitration output: ||compute_bias_ADMIT -
     compute_bias_BLOCK|| on the fixed candidate set >= BIAS_DRIFT_THRESHOLD AND
     the bias readout is non-degenerate (||compute_bias_ADMIT|| > BIAS_MIN_NORM,
     i.e. the trained-on head is live, so C3 is not vacuously satisfied by a
     zeroed head).
  seed PASS = C0 AND C4 AND C_ADMIT_LIVE AND C1 AND C2 AND C3.
  overall PASS = majority of seeds pass (>= ceil(2/3 * n_seeds)).

VERDICT / evidence_direction routing
------------------------------------
  non_vacuous(seed) = C0 AND C4 AND C_ADMIT_LIVE AND bias-readout-non-degenerate.
  If a MAJORITY of seeds are NOT non_vacuous: the substrate could not be exercised
     (replay inert / waking state degenerate / arms not matched / dead bias head)
     -> outcome FAIL, evidence_direction=non_contributory, substrate_not_ready_
     requeue. NEVER weakens MECH-319.
  Else if majority seed PASS: outcome PASS, evidence_direction=supports. MECH-319
     supported on the accumulated-drift / overcommitment path; with the gate
     normal (BLOCK) there is no drift, with it artificially opened (ADMIT) ghost
     writes accumulate into the arbitration weights and reach the output.
  Else (non-vacuous but C1/C2/C3 fail): outcome FAIL, evidence_direction=weakens.
     The substrate is live yet the gate's predicted no-drift / drift dissociation
     does not hold -> MECH-319 weakened on this path.

PREDECESSOR (NOT superseded)
----------------------------
  V3-EXQ-628 (v3_exq_628_mech319_simulation_mode_rule_gate_replay_falsifier_
  evidence): single-replay rule_state divergence; PASS/supports 2026-06-02. 668
  is the DISTINCT second evidence entry on the accumulated-drift path; 628 stays
  a valid, standing single-step record.

PHASED TRAINING / MECH-094
--------------------------
  No phased training: MECH-319 is pure boolean/counter arithmetic; the
  LateralPFCAnalog rule_state is a non-trainable EMA buffer and the bias head is
  used only as a fixed (random-init, never optimized) readout -- no gradient flow,
  no encoder head, no latent-target collapse risk. lateral_pfc_train_rule_bias_
  head=True here only un-zeroes the head so compute_bias is a non-degenerate
  function of rule_state; this experiment runs NO optimizer step. MECH-094: this
  experiment IS the simulation-mode (caller_sim=True) probe; the gate's normal-
  mode block is precisely MECH-094's hypothesis-tag write suppression realised at
  the arbitration layer.

Run:
  /opt/local/bin/python3 experiments/v3_exq_668_mech319_accumulated_ghost_write_arbitration_drift_evidence.py [--dry-run]

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

EXPERIMENT_PURPOSE = "evidence"

# --- Pre-registered thresholds (constants; NOT derived from run statistics) ---
SEEDS = [0, 1, 2]
WAKING_STEPS = 60              # denominator M in the [train] ep N/M prints
CAPTURE_STEPS = [5, 15, 25, 35]   # distinct early-waking latents replayed round-robin
N_REPLAY_ROUNDS = 12          # rounds; each round replays all captured latents
N_PROBE_CANDIDATES = 4        # fixed candidate set for the compute_bias readout
WORLD_DIM = 16                # matches the from_dims world_dim below
EPS_WAKING = 1e-6             # C4: max ||rs_BLOCK - rs_ADMIT|| on the waking path
EPS_BLOCK = 1e-6             # C1: max BLOCK-arm drift from waking baseline (~0)
ADMIT_DELTA_THRESHOLD = 1e-4  # C_ADMIT_LIVE: min ADMIT delta on the FIRST replay
ACCUM_THRESHOLD = 1e-3        # C2: min accumulated ADMIT drift from waking baseline
ACCUM_GROWTH_FACTOR = 1.5     # C2: final divergence >= factor * single-replay divergence
BIAS_DRIFT_THRESHOLD = 1e-4   # C3: min ||bias_ADMIT - bias_BLOCK|| on the probe set
BIAS_MIN_NORM = 1e-6          # C3: bias readout non-degenerate (trained-on head live)
MIN_WAKING_RULE_NORM = 1e-4   # C0: rule_state developed something to drift


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
    """Build a MECH-319-ON agent. Reseed before build so both arms get identical
    frozen-random lateral_pfc weights AND identical (random, un-zeroed) bias head
    -- the precondition for C4 bit-identical waking AND a live compute_bias readout."""
    torch.manual_seed(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16,
        world_dim=WORLD_DIM,
        use_simulation_mode_rule_gate=True,
        simulation_mode_rule_gate_admit_writes=admit_writes,
        use_gated_policy=True,
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,  # un-zero the head so compute_bias reads rule_state
    )
    agent = REEAgent(cfg)
    agent.reset()
    return agent, cfg


def _fixed_probe_summaries(seed):
    """A deterministic, seed-fixed [K, world_dim] candidate-summary set, identical
    across both arms -- so any compute_bias divergence is attributable to the
    drifted rule_state alone, not to different candidate inputs."""
    g = torch.Generator()
    g.manual_seed(seed * 7919 + 13)
    return torch.randn(N_PROBE_CANDIDATES, WORLD_DIM, generator=g)


def run_seed(seed, waking_steps, capture_steps, n_rounds):
    """Matched waking warmup on BLOCK + ADMIT arms, a sustained replay regime, a
    behavioural compute_bias probe, then the acceptance computation."""
    import random as _random

    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1, use_proxy_fields=True,
    )
    agent_block, cfg = _build_agent(env, seed, admit_writes=False)
    agent_admit, _ = _build_agent(env, seed, admit_writes=True)

    env_rng = _random.Random(seed * 1000 + 7)
    _flat, obs_dict = env.reset()
    captured = []

    # --- Phase 1: matched waking warmup. Env trajectory driven by a seeded RNG
    #     (independent of agent action sampling) so both arms see a bit-identical
    #     obs sequence -> bit-identical waking rule_state. Capture several distinct
    #     early-waking latents for round-robin replay. ---
    for step in range(waking_steps):
        kwargs = _build_obs_kwargs(obs_dict, cfg)
        ob, ow = kwargs["obs_body"], kwargs["obs_world"]
        with torch.no_grad():
            agent_block.act_with_split_obs(ob, ow, temperature=1.0)
            agent_admit.act_with_split_obs(ob, ow, temperature=1.0)
        if step in capture_steps:
            lat = agent_block._current_latent
            captured.append(
                (lat.z_delta.detach().clone(), lat.z_world.detach().clone())
            )
        a = env_rng.randrange(4)
        action_t = torch.zeros(4)
        action_t[a] = 1.0
        _flat, _h, done, _info, obs_dict = env.step(action_t)
        if done:
            _flat, obs_dict = env.reset()
        if (step + 1) % max(1, waking_steps // 4) == 0 or (step + 1) == waking_steps:
            print(f"  [train] mech319drift seed={seed} ep {step + 1}/{waking_steps}", flush=True)

    if not captured:
        # capture_steps all >= waking_steps (only at degenerate dry scale): fall
        # back to the final latent so the replay regime still has content.
        lat = agent_block._current_latent
        captured.append((lat.z_delta.detach().clone(), lat.z_world.detach().clone()))

    # --- Snapshot post-waking baseline (before any replay) ---
    rs_block_wak = agent_block.lateral_pfc.rule_state.detach().clone()
    rs_admit_wak = agent_admit.lateral_pfc.rule_state.detach().clone()
    waking_diff = float((rs_block_wak - rs_admit_wak).norm().item())
    waking_norm_block = float(rs_block_wak.norm().item())
    waking_norm_admit = float(rs_admit_wak.norm().item())
    gb_wak = agent_block.simulation_mode_rule_gate.get_state()
    ga_wak = agent_admit.simulation_mode_rule_gate.get_state()
    waking_sim_firings_block = gb_wak["n_simulation_blocked"] + gb_wak["n_simulation_admitted"]
    waking_sim_firings_admit = ga_wak["n_simulation_blocked"] + ga_wak["n_simulation_admitted"]

    # --- Phase 2: sustained replay / caller_sim=True regime. Verbatim agent.py
    #     LateralPFC wiring (gate consult -> if not eff_sim: lateral_pfc.update),
    #     the only change being simulation_mode=True. Each round replays every
    #     captured latent round-robin; per-round cross-arm divergence tracked. ---
    per_round_divergence = []
    single_replay_divergence = None      # cross-arm divergence after the FIRST replay
    admit_single_write_delta = None      # ADMIT rule_state delta on the FIRST replay
    n_replay = 0
    for r in range(n_rounds):
        for (zd, zw) in captured:
            rs_admit_before = agent_admit.lateral_pfc.rule_state.detach().clone()
            for agent in (agent_block, agent_admit):
                gate = agent.simulation_mode_rule_gate
                eff_sim = gate.effective_simulation_mode(
                    simulation_mode=True, site=SITE_LATERAL_PFC
                )
                if not eff_sim:
                    agent.lateral_pfc.update(z_delta=zd, z_world=zw, gate=1.0)
            n_replay += 1
            if admit_single_write_delta is None:
                rs_admit_after = agent_admit.lateral_pfc.rule_state.detach().clone()
                admit_single_write_delta = float((rs_admit_after - rs_admit_before).norm().item())
                single_replay_divergence = float(
                    (agent_admit.lateral_pfc.rule_state - agent_block.lateral_pfc.rule_state).norm().item()
                )
        div = float(
            (agent_admit.lateral_pfc.rule_state - agent_block.lateral_pfc.rule_state).norm().item()
        )
        per_round_divergence.append(div)

    rs_block_post = agent_block.lateral_pfc.rule_state.detach().clone()
    rs_admit_post = agent_admit.lateral_pfc.rule_state.detach().clone()
    accumulated_drift_block = float((rs_block_post - rs_block_wak).norm().item())
    accumulated_drift_admit = float((rs_admit_post - rs_admit_wak).norm().item())
    final_divergence = per_round_divergence[-1] if per_round_divergence else 0.0
    first_divergence = per_round_divergence[0] if per_round_divergence else 0.0
    gb2 = agent_block.simulation_mode_rule_gate.get_state()
    ga2 = agent_admit.simulation_mode_rule_gate.get_state()
    block_n_blocked = gb2["n_simulation_blocked"]
    block_n_admitted = gb2["n_simulation_admitted"]
    admit_n_blocked = ga2["n_simulation_blocked"]
    admit_n_admitted = ga2["n_simulation_admitted"]

    # --- Phase 3: behavioural overcommitment probe. Score a fixed candidate set
    #     by each (drifted) arm's compute_bias WITHOUT updating. ADMIT's bias
    #     diverges from BLOCK by the accumulated ghost drift carried into the
    #     arbitration output. ---
    summaries = _fixed_probe_summaries(seed)
    with torch.no_grad():
        bias_block = agent_block.lateral_pfc.compute_bias(summaries).detach().clone()
        bias_admit = agent_admit.lateral_pfc.compute_bias(summaries).detach().clone()
    bias_drift = float((bias_admit - bias_block).norm().item())
    bias_admit_norm = float(bias_admit.norm().item())
    bias_block_norm = float(bias_block.norm().item())

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
    c4 = bool(
        waking_diff <= EPS_WAKING
        and waking_sim_firings_block == 0
        and waking_sim_firings_admit == 0
    )
    c_admit_live = bool(
        admit_single_write_delta is not None
        and admit_single_write_delta >= ADMIT_DELTA_THRESHOLD
    )
    bias_non_degenerate = bool(bias_admit_norm > BIAS_MIN_NORM)
    c1 = bool(
        accumulated_drift_block <= EPS_BLOCK
        and block_n_admitted == 0
    )
    c2 = bool(
        accumulated_drift_admit >= ACCUM_THRESHOLD
        and single_replay_divergence is not None
        and final_divergence >= ACCUM_GROWTH_FACTOR * single_replay_divergence
    )
    c3 = bool(bias_drift >= BIAS_DRIFT_THRESHOLD and bias_non_degenerate)

    non_vacuous = bool(c0 and c4 and c_admit_live and bias_non_degenerate)
    seed_pass = bool(c0 and c4 and c_admit_live and c1 and c2 and c3)

    return {
        "seed": seed,
        "pass": seed_pass,
        "non_vacuous": non_vacuous,
        "C0_path_exercised_no_vacuity": c0,
        "C4_waking_bit_identical_falsifier": c4,
        "C_admit_live_readiness": c_admit_live,
        "C1_block_no_drift": c1,
        "C2_admit_accumulates": c2,
        "C3_drift_reaches_arbitration_output": c3,
        "bias_readout_non_degenerate": bias_non_degenerate,
        "waking_rule_state_diff": waking_diff,
        "waking_rule_norm_block": waking_norm_block,
        "waking_rule_norm_admit": waking_norm_admit,
        "waking_sim_firings_block": waking_sim_firings_block,
        "waking_sim_firings_admit": waking_sim_firings_admit,
        "admit_single_write_delta": admit_single_write_delta,
        "single_replay_divergence": single_replay_divergence,
        "accumulated_drift_block": accumulated_drift_block,
        "accumulated_drift_admit": accumulated_drift_admit,
        "first_round_divergence": first_divergence,
        "final_round_divergence": final_divergence,
        "per_round_divergence": per_round_divergence,
        "bias_drift": bias_drift,
        "bias_admit_norm": bias_admit_norm,
        "bias_block_norm": bias_block_norm,
        "block_gate_n_simulation_blocked": block_n_blocked,
        "block_gate_n_simulation_admitted": block_n_admitted,
        "admit_gate_n_simulation_blocked": admit_n_blocked,
        "admit_gate_n_simulation_admitted": admit_n_admitted,
        "n_replay": n_replay,
        "n_rounds": n_rounds,
        "n_captured_latents": len(captured),
        "waking_steps": waking_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        seeds = [0]
        waking_steps = 8
        capture_steps = [2, 4]
        n_rounds = 3
    else:
        seeds = SEEDS
        waking_steps = WAKING_STEPS
        capture_steps = CAPTURE_STEPS
        n_rounds = N_REPLAY_ROUNDS

    t0 = time.time()
    per_seed = []
    for seed in seeds:
        print(f"Seed {seed} Condition base", flush=True)
        res = run_seed(seed, waking_steps, capture_steps, n_rounds)
        per_seed.append(res)
        print(f"verdict: {'PASS' if res['pass'] else 'FAIL'}", flush=True)

    n_pass = sum(1 for r in per_seed if r["pass"])
    n_non_vacuous = sum(1 for r in per_seed if r["non_vacuous"])
    required = math.ceil(2.0 / 3.0 * len(seeds))
    all_pass = n_pass >= required
    majority_non_vacuous = n_non_vacuous >= required
    elapsed = time.time() - t0

    # --- Verdict / evidence_direction routing ---
    if not majority_non_vacuous:
        evidence_direction = "non_contributory"
        interpretation_label = "substrate_not_ready_requeue"
    elif all_pass:
        evidence_direction = "supports"
        interpretation_label = "mech319_accumulated_drift_supported"
    else:
        evidence_direction = "weakens"
        interpretation_label = "mech319_accumulated_drift_weakened"

    run_id_base = "v3_exq_668_mech319_accumulated_ghost_write_arbitration_drift_evidence_v3"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{run_id_base}_{ts}",
        "experiment_type": "v3_exq_668_mech319_accumulated_ghost_write_arbitration_drift_evidence",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": ["MECH-319"],
        "outcome": "PASS" if all_pass else "FAIL",
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {"MECH-319": evidence_direction},
        "predecessor": "V3-EXQ-628",
        "metrics": {
            "n_seeds": len(seeds),
            "n_pass": n_pass,
            "n_non_vacuous": n_non_vacuous,
            "required_pass": required,
            "interpretation_label": interpretation_label,
            "per_seed": per_seed,
            "thresholds": {
                "EPS_WAKING": EPS_WAKING,
                "EPS_BLOCK": EPS_BLOCK,
                "ADMIT_DELTA_THRESHOLD": ADMIT_DELTA_THRESHOLD,
                "ACCUM_THRESHOLD": ACCUM_THRESHOLD,
                "ACCUM_GROWTH_FACTOR": ACCUM_GROWTH_FACTOR,
                "BIAS_DRIFT_THRESHOLD": BIAS_DRIFT_THRESHOLD,
                "BIAS_MIN_NORM": BIAS_MIN_NORM,
                "MIN_WAKING_RULE_NORM": MIN_WAKING_RULE_NORM,
                "N_REPLAY_ROUNDS": n_rounds,
                "WAKING_STEPS": waking_steps,
                "CAPTURE_STEPS": capture_steps,
                "N_PROBE_CANDIDATES": N_PROBE_CANDIDATES,
            },
        },
        "diagnostic_interpretation": {
            "PASS": "MECH-319 supported on the accumulated-drift / overcommitment path: with the gate normal (BLOCK) the arbitration rule_state shows no drift across a sustained replay regime; with it artificially opened (ADMIT) ghost writes accumulate into rule_state and reach the compute_bias output. evidence_direction=supports.",
            "non_contributory": "Majority of seeds NOT non_vacuous (C0/C4/C_admit_live/bias-readout): replay inert OR waking state degenerate OR arms not matched OR dead bias head -> substrate not ready, NOT a MECH-319 result -> substrate_not_ready_requeue.",
            "C2_fail": "ADMIT does not accumulate beyond a single step (final divergence ~ single-replay divergence) -> the sustained-regime drift the claim predicts is absent -> weakens.",
            "C1_fail": "BLOCK arm drifts despite the gate blocking simulation writes -> MECH-319 falsified on the no-drift prediction -> weakens.",
            "C3_fail": "Accumulated rule_state drift does not reach the arbitration compute_bias output -> drift not behaviourally observable -> weakens.",
            "C4_fail": "Waking path not bit-identical (admit_writes leaked into waking): substrate BUG violating the bit-identical-waking commitment -> /diagnose-errors (falsifies a different invariant than MECH-319).",
        },
        "elapsed_seconds": elapsed,
        "dry_run": bool(args.dry_run),
        "notes": (
            "MECH-319 ACCUMULATED ghost-write arbitration-DRIFT / overcommitment "
            "falsifier EVIDENCE -- the SECOND, distinct contributory entry for "
            "MECH-319 (V3-EXQ-628 tested single-replay rule_state divergence; this "
            "tests the claim's OTHER pre-registered V3 path: overcommitment to "
            "ghost-derived arbitration weights + observable rule-arbitration drift "
            "over a sustained replay regime). Does NOT supersede 628 -- both are "
            "distinct evidence entries (MECH-319 promotion needs "
            "min_experimental_entries=2). BLOCK (admit_writes=False) shows no drift; "
            "ADMIT (admit_writes=True) accumulates ghost drift into rule_state (C2) "
            "that reaches the compute_bias arbitration output (C3); waking path "
            "bit-identical (C4, load-bearing); admit-channel-live readiness "
            "positive control (C_admit_live) self-routes substrate_not_ready_requeue "
            "rather than a false weakening. See ree-v3/CLAUDE.md 'MECH-319 (arc_062 "
            "GAP-K)', claims.yaml MECH-319, and docs/architecture/mech_319_"
            "simulation_mode_rule_gate.md."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"OVERALL: {manifest['result']}", flush=True)
    print(
        f"  n_pass={n_pass}/{len(seeds)} (required {required}) "
        f"non_vacuous={n_non_vacuous}/{len(seeds)} direction={evidence_direction}",
        flush=True,
    )
    for r in per_seed:
        print(
            f"  seed {r['seed']}: pass={r['pass']} non_vacuous={r['non_vacuous']} "
            f"C0={r['C0_path_exercised_no_vacuity']} C4={r['C4_waking_bit_identical_falsifier']} "
            f"live={r['C_admit_live_readiness']} C1={r['C1_block_no_drift']} "
            f"C2={r['C2_admit_accumulates']} C3={r['C3_drift_reaches_arbitration_output']} "
            f"admit_drift={r['accumulated_drift_admit']:.3e} "
            f"single_div={r['single_replay_divergence']!r} "
            f"final_div={r['final_round_divergence']:.3e} "
            f"bias_drift={r['bias_drift']:.3e}",
            flush=True,
        )
    print(f"Result written to: {out_path}", flush=True)

    return manifest["result"], str(out_path)


if __name__ == "__main__":
    _result, _manifest_path = main()
    emit_outcome(outcome=_result, manifest_path=_manifest_path)
