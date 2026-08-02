"""V3-EXQ-880: ARC-014 commitment-suppressed internal simulation mode -- EVIDENCE.

EXPERIMENT_PURPOSE = evidence. claim_ids = [ARC-014].

WHY THIS EXPERIMENT EXISTS
--------------------------
ARC-014 (default_mode.internal_generative;
docs/architecture/default_mode.md#arc-014) asserts Default Mode is a
non-commitment operating regime: internal simulation / replay / ghost / DMN
passes must leave NO durable commitment or world-state effect, distinct from
waking behaviour. The doc's own architectural-necessity list states
"trajectory commitment occurs only via E3" and "ethical residue phi(z) is
path dependent" -- i.e. ARC-014's prediction spans (a) the E3-arbitration
commitment channel and (b) the residue/world-state channel. As of 2026-08-02,
claim_evidence.v1.json shows genuine_exp_count=0 for ARC-014 (3 entries, all
lit:theoretical_review / lit:empirical_review -- Bratman 1987, Elster 2000,
Rangel/Camerer/Montague 2008; exp_conf=0.0, lit_conf=0.766); no run has ever
tagged ARC-014 in claim_ids, and 0 failure_autopsy_*.json targets tag it
either (re-derive brake Step 2.5b not applicable). This is the first
experimental test.

SUBSTRATE -- two REAL, already-landed mechanisms instantiate the prediction
---------------------------------------------------------------------------
  MECH-319 (ree_core/regulators/simulation_mode_rule_gate.py; status
    provisional, supports via V3-EXQ-628/668): a categorical write gate
    keyed to a simulation-mode tag that suppresses
    LateralPFCAnalog.rule_state updates (the arbitration-weight
    "commitment" channel -- MECH-312 sub-mechanism) during ghost / replay /
    DMN passes, while admitting waking writes. Master switch
    use_simulation_mode_rule_gate; production default
    simulation_mode_rule_gate_admit_writes=False (the V3-EXQ-543c falsifier
    flag is NOT used here -- this experiment tests the production/default
    behaviour ARC-014 predicts, not the mechanism's own switchability,
    which 628 already evidenced under MECH-319's own claim_id).
  MECH-094 (ree_core/residue/field.py, ResidueField.accumulate; status
    stable): a hypothesis_tag gate -- simulated/replay content
    (hypothesis_tag=True) cannot accumulate ethical residue at a z_world
    location (the module's own docstring: "Accumulate residue at a
    z_world location" -- this IS the "world-state effect" channel);
    only waking-tagged content (hypothesis_tag=False) accumulates.

Both mechanisms are wired into the real REEAgent (agent.py
self.simulation_mode_rule_gate + self.residue_field). Read agent.py in full
(2026-08-02) to confirm: agent._do_replay() (the current production
ghost/replay call site, fired on E3-quiescent ticks) calls
hippocampal.replay()/diverse_replay() and DISCARDS the resulting
trajectories without ever calling lateral_pfc.update() or
residue_field.accumulate() -- "these trajectories cannot update residue
(MECH-094 -- enforced in ResidueField.accumulate)". So there is currently no
production call site that drives a real replay tick through either
consumer with a simulation tag; MECH-319's own docstring names this the
"reserved seam" for the V3-EXQ-543c replay-driven invocation path. This
experiment constructs that seam directly (identical technique to
V3-EXQ-628's replay phase) for the rule_state channel, and uses the
residue channel's real, always-reachable API
(ResidueField.accumulate(hypothesis_tag=...)) directly -- both are
legitimate instrument access for a targeted_probe (EVB-0097 dispatch_mode),
not synthetic tensors: all latent content is genuine forward-pass output
from a real seeded env rollout.

GOV-REUSE-1 (Step 2.4) CHECK
----------------------------
Decisive readout: does simulation-tagged content leave rule_state /
total_residue durably unchanged while waking-tagged content changes them,
on the current substrate? Checked V3-EXQ-628
(v3_exq_628_..._20260602T191625Z, 2026-06-02): its BLOCK arm (production
MECH-319 config, admit_writes=False) already shows C0 waking_rule_norm >
floor and C1 replay_block_rule_state_delta == 0.0 -- i.e. the closest prior
evidence for the rule_state half of this claim. But that manifest predates
the 2026-07-12 Experimental Recording Standard and carries no
`substrate_hash` (confirmed: 'substrate_hash' not in the manifest keys) --
per this skill's Step 2.4 pre-standard caveat, an unverifiable-substrate
match is treated as NOT RECOVERABLE, not as a reuse hit. Additionally 628
tagged claim_ids=[MECH-319] only (never ARC-014) and never touched the
MECH-094/residue "world-state" channel at all. -> Not recoverable -> run
(documented here per Step 2.4 item 4).

Re-derive brake (Step 2.5b): grep of
REE_assembly/evidence/planning/failure_autopsy_*.json for ARC-014 in any
target's claim_ids returns 0 hits (2026-08-02) -> brake not applicable,
consistent with genuine_exp_count=0.

DESIGN -- matched-content, tag-only manipulation, two independent channels
---------------------------------------------------------------------------
Per seed:
  1. GENERATION PHASE: one throwaway agent steps a real seeded
     CausalGridWorldV2 for N_CAPTURE ticks via act_with_split_obs (genuine
     forward-pass latent content; untrained weights are fine here -- only
     the gate's boolean routing and the resulting deltas are under test,
     not representation quality). Captures (z_delta_t, z_world_t) at every
     tick -> N_CAPTURE content tuples. This loop also supplies the
     progress-instrumentation [train] ep N/M denominator.
  2. Build TWO FRESH agents per seed from the SAME seed (use_simulation_
     mode_rule_gate=True, simulation_mode_rule_gate_admit_writes=False --
     MECH-319 PRODUCTION default; use_lateral_pfc_analog=True;
     use_gated_policy=True). Reseeding torch identically before each build
     makes the untrained lateral_pfc projection weights bit-identical
     across both arms (precondition for a clean divergence read --
     mirrors V3-EXQ-628's agent_block/agent_admit construction).
       ARM_WAKE: content tagged waking  (hypothesis_tag=False, simulation_mode=False)
       ARM_SIM:  content tagged simulation (hypothesis_tag=True, simulation_mode=True)
  3. Apply the SAME N_CAPTURE captured (z_delta, z_world) tuples to BOTH
     arms, the tag being the ONLY difference:
       residue channel:    agent.residue_field.accumulate(z_world,
                            harm_magnitude=HARM_MAG, hypothesis_tag=<tag>)
       rule_state channel: eff_sim = gate.effective_simulation_mode(
                            simulation_mode=<tag>, site=SITE_LATERAL_PFC);
                            if not eff_sim: agent.lateral_pfc.update(
                            z_delta=zd, z_world=zw, gate=1.0) -- the
                            verbatim agent.py MECH-319 LateralPFC wiring
                            (gate consult -> conditional update), the only
                            change being which tag is passed.
  4. Measure pre/post total_residue and rule_state norm on both arms;
     compute within-arm deltas and cross-arm (post) divergence.

PRE-REGISTERED ACCEPTANCE GRID (per seed)
------------------------------------------
  C0 path-exercised/no-vacuity: both arms instantiate
     simulation_mode_rule_gate + lateral_pfc + residue_field; gate firing
     counts match expectation EXACTLY (WAKE: n_waking_admitted==N_CAPTURE;
     SIM: n_simulation_blocked==N_CAPTURE, n_simulation_admitted==0);
     residue skip counts match exactly (WAKE: 0 "skipped_hypothesis"
     returns; SIM: N_CAPTURE "skipped_hypothesis" returns).
  C1 SIM suppresses commitment (rule_state): sim_rule_state_delta <= EPS_RULE.
  C2 WAKE commits (rule_state): wake_rule_state_delta >= RULE_DELTA_THRESHOLD.
  C3 rule_state divergence: ||rule_state_WAKE_post - rule_state_SIM_post||
     >= RULE_DIVERGENCE_THRESHOLD.
  C4 SIM suppresses commitment (residue/world-state):
     sim_residue_delta <= EPS_RESIDUE.
  C5 WAKE commits (residue/world-state):
     wake_residue_delta >= RESIDUE_DELTA_THRESHOLD.
  C6 residue divergence: |total_residue_WAKE_post - total_residue_SIM_post|
     >= RESIDUE_DIVERGENCE_THRESHOLD.
  combination_rule: seed PASS = C0 AND C1 AND C2 AND C3 AND C4 AND C5 AND C6
     (plain AND across two independent channels). Load-bearing = C1, C2,
     C4, C5 (the actual commit/no-commit discriminations); C0/C3/C6 are
     non-degeneracy / sanity checks.
  overall PASS = majority of seeds pass (>= ceil(2/3 * n_seeds)).

  Expected magnitudes (deterministic, not seed-sensitive): residue
  accumulation_rate defaults to 0.1 (ree_core/utils/config.py), so a WAKE
  arm calling accumulate(harm_magnitude=HARM_MAG=1.0) N_CAPTURE=20 times
  should reach total_residue ~ 20 * 1.0 * 0.1 = 2.0, well above
  RESIDUE_DELTA_THRESHOLD=1.0. rule_state's expected WAKE delta is
  comparable in order of magnitude to V3-EXQ-628's measured ADMIT-arm
  delta (~0.37 over 30 replay ticks at eta default), well above
  RULE_DELTA_THRESHOLD=1e-4.

DV-SYMMETRY INVARIANCE DECLARATION (Step 3.5 mandatory, every arm)
---------------------------------------------------------------------
Both channels' manipulation is a CATEGORICAL BRANCH (whether
accumulate()/update() executes at all, gated by an if-statement keyed on
the tag), not a magnitude fed into an argmax / rank / set-aggregate DV. Not
subject to the broadcast-constant (cancels under argmax / softmax-sample
selection), monotone-rescaling (cancels under rank / order DVs), or
permutation-of-interchangeable-units (cancels under set-aggregate DVs)
traps documented in the skill's DV-SYMMETRY INVARIANCE section -- the
manipulation directly controls whether a deterministic write happens at
all, not a value fed into a symmetry-invariant reduction.

DIAGNOSTIC-STYLE INTERPRETATION (recorded even though EXPERIMENT_PURPOSE=evidence)
------------------------------------------------------------------------------
  PASS (all C): ARC-014 supported on both instantiating mechanisms --
     simulation-tagged content leaves neither commitment (rule_state) nor
     world-state (residue) durably changed, while waking-tagged content
     changes both, from IDENTICAL captured latent content.
     -> evidence_direction=supports.
  C0 fail: harness/wiring defect (gate/residue firing counts off
     expectation) -- NOT an ARC-014 result -> non_contributory;
     route /diagnose-errors.
  C1 or C4 fail (SIM does NOT suppress): the categorical tag leaked into
     commitment (C1) or world-state (C4) -> ARC-014 falsified on that
     channel -> weakens / does_not_support.
  C2 or C5 fail (WAKE does NOT commit): the channel is inert even for
     waking content -> falsifier structure vacuous for that channel ->
     non_contributory (nothing to distinguish simulation from).
  C3 or C6 fail (no divergence despite C1/C2 or C4/C5 passing): an
     internally-inconsistent state not expected to occur given the other
     criteria; flag for /diagnose-errors if observed.

PHASED TRAINING / GRADIENT FLOW
--------------------------------
None. rule_state is a non-trainable EMA buffer (registered buffer, no
gradient flow -- matches MECH-319's own "Non-trainable: pure boolean /
counter arithmetic" note); total_residue is a registered buffer updated by
plain arithmetic. Both updates run under no_grad or are buffer-only by
construction. Untrained lateral_pfc / gated_policy projection weights are
acceptable here -- only forward-pass latent CONTENT and the gates'
BOOLEAN ROUTING are under test, never learned representation quality.

SLEEP DRIVER: none (no SleepLoopManager / sws_enabled / rem_enabled /
use_sleep_aggregation_cluster in this experiment).

PREDECESSOR / RELATED (NOT a supersession of any of these)
-------------------------------------------------------------
V3-EXQ-628 (MECH-319 rule_state replay/caller_sim falsifier EVIDENCE,
claim_ids=[MECH-319]); V3-EXQ-546 (MECH-319 substrate-readiness,
diagnostic); V3-EXQ-668 (MECH-319 accumulated ghost-write arbitration
drift evidence). This experiment reuses the same rule_state instrument as
628 but (a) tests the ARCHITECTURAL claim ARC-014 rather than the
substrate mechanism MECH-319 itself, (b) uses the PRODUCTION MECH-319
config throughout rather than the ON/OFF falsifier arms, and (c) adds the
MECH-094/residue "world-state" channel that no prior run has tested. First
ARC-014-tagged claim probe.

Run:
  /opt/local/bin/python3 experiments/v3_exq_880_arc014_simulation_mode_commitment.py [--dry-run]

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.regulators import SITE_LATERAL_PFC  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_880_arc014_simulation_mode_commitment"
QUEUE_ID = "V3-EXQ-880"
CLAIM_IDS = ["ARC-014"]

# --- Pre-registered thresholds (constants; NOT derived from run statistics) ---
SEEDS = [0, 1, 2]
N_CAPTURE = 20          # content ticks captured + applied to both arms (also [train] ep N/M denominator)
HARM_MAG = 1.0          # fixed magnitude passed to residue_field.accumulate on every call
EPS_RULE = 1e-7                     # C1: max SIM-arm rule_state delta (~0 expected)
RULE_DELTA_THRESHOLD = 1e-4         # C2: min WAKE-arm rule_state delta
RULE_DIVERGENCE_THRESHOLD = 1e-4    # C3: min ||rule_state_WAKE - rule_state_SIM|| post
EPS_RESIDUE = 1e-6                  # C4: max SIM-arm total_residue delta (~0 expected)
RESIDUE_DELTA_THRESHOLD = 1.0       # C5: min WAKE-arm total_residue delta (expect ~N_CAPTURE*HARM_MAG*0.1=2.0)
RESIDUE_DIVERGENCE_THRESHOLD = 1.0  # C6: min |total_residue_WAKE - total_residue_SIM| post


def _build_obs_kwargs(obs_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return {"obs_body": body, "obs_world": world}


def _build_agent(env: CausalGridWorldV2, seed: int) -> Tuple[REEAgent, REEConfig]:
    """Build a MECH-319-PRODUCTION agent (admit_writes=False -- NOT the
    V3-EXQ-543c falsifier config; this experiment tests the default/
    production behaviour ARC-014 predicts). Reseed before build so
    lateral_pfc's randomly-initialised projection weights are bit-identical
    across agents built from the same seed (precondition for a clean
    divergence read, mirroring V3-EXQ-628's agent_block/agent_admit pair)."""
    torch.manual_seed(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        use_simulation_mode_rule_gate=True,
        simulation_mode_rule_gate_admit_writes=False,
        use_gated_policy=True,
        use_lateral_pfc_analog=True,
    )
    agent = REEAgent(cfg)
    agent.reset()
    return agent, cfg


def _capture_content(
    seed: int, n_capture: int, zg: ZGoalStreamAccumulator,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """GENERATION PHASE: step one throwaway agent through a real seeded env
    for n_capture ticks, capturing (z_delta, z_world) at every tick. Genuine
    forward-pass latent content from a real rollout -- untrained weights are
    fine here since only the gates' boolean routing and resulting deltas are
    under test, not representation quality."""
    import random as _random

    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1, use_proxy_fields=True,
    )
    gen_agent, _cfg = _build_agent(env, seed)
    env_rng = _random.Random(seed * 1000 + 13)
    _flat, obs_dict = env.reset()
    captured: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for step in range(n_capture):
        kwargs = _build_obs_kwargs(obs_dict)
        with torch.no_grad():
            gen_agent.act_with_split_obs(kwargs["obs_body"], kwargs["obs_world"], temperature=1.0)
        lat = gen_agent._current_latent
        captured.append((lat.z_delta.detach().clone(), lat.z_world.detach().clone()))
        a = env_rng.randrange(4)
        action_t = torch.zeros(4)
        action_t[a] = 1.0
        _flat, _h, done, _info, obs_dict = env.step(action_t)
        if done:
            _flat, obs_dict = env.reset()
        if (step + 1) % max(1, n_capture // 4) == 0 or (step + 1) == n_capture:
            print(f"  [train] arc014 seed={seed} ep {step + 1}/{n_capture}", flush=True)

    zg.observe(gen_agent)
    return captured


def run_seed(seed: int, n_capture: int, zg: ZGoalStreamAccumulator) -> Dict[str, Any]:
    """One seed: capture N content tuples, build matched WAKE/SIM agents
    from the same seed, apply the SAME content with only the tag differing,
    measure deltas on both the rule_state and residue channels."""
    captured = _capture_content(seed, n_capture, zg)

    dummy_env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1, use_proxy_fields=True,
    )
    agent_wake, _ = _build_agent(dummy_env, seed)
    agent_sim, _ = _build_agent(dummy_env, seed)

    rs_wake_pre = agent_wake.lateral_pfc.rule_state.detach().clone()
    rs_sim_pre = agent_sim.lateral_pfc.rule_state.detach().clone()
    tr_wake_pre = float(agent_wake.residue_field.total_residue.item())
    tr_sim_pre = float(agent_sim.residue_field.total_residue.item())

    residue_wake_skips = 0
    residue_sim_skips = 0
    for zd, zw in captured:
        # --- WAKE arm: waking tag (hypothesis_tag=False / simulation_mode=False) ---
        rm_wake = agent_wake.residue_field.accumulate(
            zw, harm_magnitude=HARM_MAG, hypothesis_tag=False,
        )
        if "skipped_hypothesis" in rm_wake:
            residue_wake_skips += 1
        gate_w = agent_wake.simulation_mode_rule_gate
        eff_sim_w = gate_w.effective_simulation_mode(simulation_mode=False, site=SITE_LATERAL_PFC)
        if not eff_sim_w:
            agent_wake.lateral_pfc.update(z_delta=zd, z_world=zw, gate=1.0)

        # --- SIM arm: simulation tag (hypothesis_tag=True / simulation_mode=True) ---
        rm_sim = agent_sim.residue_field.accumulate(
            zw, harm_magnitude=HARM_MAG, hypothesis_tag=True,
        )
        if "skipped_hypothesis" in rm_sim:
            residue_sim_skips += 1
        gate_s = agent_sim.simulation_mode_rule_gate
        eff_sim_s = gate_s.effective_simulation_mode(simulation_mode=True, site=SITE_LATERAL_PFC)
        if not eff_sim_s:
            agent_sim.lateral_pfc.update(z_delta=zd, z_world=zw, gate=1.0)

    zg.observe(agent_wake)
    zg.observe(agent_sim)

    rs_wake_post = agent_wake.lateral_pfc.rule_state.detach().clone()
    rs_sim_post = agent_sim.lateral_pfc.rule_state.detach().clone()
    tr_wake_post = float(agent_wake.residue_field.total_residue.item())
    tr_sim_post = float(agent_sim.residue_field.total_residue.item())

    wake_rule_delta = float((rs_wake_post - rs_wake_pre).norm().item())
    sim_rule_delta = float((rs_sim_post - rs_sim_pre).norm().item())
    rule_divergence = float((rs_wake_post - rs_sim_post).norm().item())
    wake_residue_delta = tr_wake_post - tr_wake_pre
    sim_residue_delta = tr_sim_post - tr_sim_pre
    residue_divergence = abs(tr_wake_post - tr_sim_post)

    gw = agent_wake.simulation_mode_rule_gate.get_state()
    gs = agent_sim.simulation_mode_rule_gate.get_state()
    n = len(captured)

    c0 = bool(
        agent_wake.simulation_mode_rule_gate is not None
        and agent_sim.simulation_mode_rule_gate is not None
        and agent_wake.lateral_pfc is not None
        and agent_sim.lateral_pfc is not None
        and gw["n_waking_admitted"] == n
        and gs["n_simulation_blocked"] == n
        and gs["n_simulation_admitted"] == 0
        and residue_wake_skips == 0
        and residue_sim_skips == n
    )
    c1 = bool(sim_rule_delta <= EPS_RULE)
    c2 = bool(wake_rule_delta >= RULE_DELTA_THRESHOLD)
    c3 = bool(rule_divergence >= RULE_DIVERGENCE_THRESHOLD)
    c4 = bool(sim_residue_delta <= EPS_RESIDUE)
    c5 = bool(wake_residue_delta >= RESIDUE_DELTA_THRESHOLD)
    c6 = bool(residue_divergence >= RESIDUE_DIVERGENCE_THRESHOLD)
    seed_pass = bool(c0 and c1 and c2 and c3 and c4 and c5 and c6)

    return {
        "seed": seed,
        "pass": seed_pass,
        "n_captured": n,
        "C0_path_exercised_no_vacuity": c0,
        "C1_sim_suppresses_commitment_rule_state": c1,
        "C2_wake_commits_rule_state": c2,
        "C3_rule_state_divergence": c3,
        "C4_sim_suppresses_worldstate_residue": c4,
        "C5_wake_commits_worldstate_residue": c5,
        "C6_residue_divergence": c6,
        "wake_rule_state_delta": wake_rule_delta,
        "sim_rule_state_delta": sim_rule_delta,
        "rule_state_divergence": rule_divergence,
        "wake_residue_delta": wake_residue_delta,
        "sim_residue_delta": sim_residue_delta,
        "residue_divergence": residue_divergence,
        "total_residue_wake_pre": tr_wake_pre,
        "total_residue_wake_post": tr_wake_post,
        "total_residue_sim_pre": tr_sim_pre,
        "total_residue_sim_post": tr_sim_post,
        "gate_wake_state": gw,
        "gate_sim_state": gs,
        "residue_wake_skips": residue_wake_skips,
        "residue_sim_skips": residue_sim_skips,
    }


def main(args: argparse.Namespace) -> Tuple[str, str]:
    if args.dry_run:
        seeds = [0]
        n_capture = 4
    else:
        seeds = SEEDS
        n_capture = N_CAPTURE

    t0 = time.time()
    zg = ZGoalStreamAccumulator()
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"Seed {seed} Condition base", flush=True)
        res = run_seed(seed, n_capture, zg)
        per_seed.append(res)
        print(f"verdict: {'PASS' if res['pass'] else 'FAIL'}", flush=True)

    n_pass = sum(1 for r in per_seed if r["pass"])
    required = math.ceil(2.0 / 3.0 * len(seeds))
    all_pass = n_pass >= required
    elapsed = time.time() - t0

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    evidence_direction = "supports" if all_pass else "weakens"

    # Non-degeneracy self-report (Step 3 "Non-degeneracy scoring net -- applies
    # to evidence runs too" / validate_experiments.py degeneracy-self-report
    # check). The WAKE-arm deltas are the metrics expected to show genuine
    # cross-seed movement (the load-bearing C2/C5 criteria); a run where every
    # seed's wake delta sits at/below its own pass-floor would mean C2/C5 could
    # never fire regardless of behaviour (the V3-EXQ-514m / V3-EXQ-642 vacuous-
    # criterion pattern). The SIM-arm deltas are *expected* to be near-zero
    # (that is the predicted result, not degeneracy) and are instead guarded
    # structurally by C0's exact gate/skip firing-count checks above.
    degeneracy = check_degeneracy({
        "wake_rule_state_delta": {
            "values": [r["wake_rule_state_delta"] for r in per_seed],
            "floor": RULE_DELTA_THRESHOLD,
        },
        "wake_residue_delta": {
            "values": [r["wake_residue_delta"] for r in per_seed],
            "floor": RESIDUE_DELTA_THRESHOLD,
        },
    })

    config = {
        "seeds": seeds,
        "n_capture": n_capture,
        "harm_mag": HARM_MAG,
        "env": {"size": 5, "num_hazards": 1, "num_resources": 1, "use_proxy_fields": True},
        "agent_config_shared": {
            "use_simulation_mode_rule_gate": True,
            "simulation_mode_rule_gate_admit_writes": False,
            "use_gated_policy": True,
            "use_lateral_pfc_analog": True,
            "action_dim": 4,
            "self_dim": 16,
            "world_dim": 16,
        },
        "thresholds": {
            "EPS_RULE": EPS_RULE,
            "RULE_DELTA_THRESHOLD": RULE_DELTA_THRESHOLD,
            "RULE_DIVERGENCE_THRESHOLD": RULE_DIVERGENCE_THRESHOLD,
            "EPS_RESIDUE": EPS_RESIDUE,
            "RESIDUE_DELTA_THRESHOLD": RESIDUE_DELTA_THRESHOLD,
            "RESIDUE_DIVERGENCE_THRESHOLD": RESIDUE_DIVERGENCE_THRESHOLD,
        },
    }

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": "PASS" if all_pass else "FAIL",
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": evidence_direction,
        "related_claims": ["MECH-319", "MECH-094"],
        "related_runs": ["V3-EXQ-628", "V3-EXQ-546", "V3-EXQ-668"],
        "supersedes": None,
        "metrics": {
            "n_seeds": len(seeds),
            "n_pass": n_pass,
            "required_pass": required,
            "per_seed": per_seed,
        },
        "interpretation": {
            "label": (
                "arc014_commitment_worldstate_suppressed"
                if all_pass else "arc014_channel_inconclusive"
            ),
            "combination_rule": (
                "seed_pass = C0 AND C1 AND C2 AND C3 AND C4 AND C5 AND C6 "
                "(plain AND, two independent channels: rule_state / MECH-319, "
                "residue / MECH-094)"
            ),
            "criteria": [
                {"name": "C1_sim_suppresses_commitment_rule_state", "load_bearing": True},
                {"name": "C2_wake_commits_rule_state", "load_bearing": True},
                {"name": "C4_sim_suppresses_worldstate_residue", "load_bearing": True},
                {"name": "C5_wake_commits_worldstate_residue", "load_bearing": True},
            ],
        },
        "diagnostic_interpretation": {
            "PASS": (
                "ARC-014 supported on both instantiating mechanisms (MECH-319 "
                "rule_state / MECH-094 residue): simulation-tagged content "
                "leaves neither commitment nor world-state durably changed, "
                "while waking-tagged content changes both, from IDENTICAL "
                "captured latent content."
            ),
            "C0_fail": (
                "Path not exercised as expected (gate/residue firing counts "
                "off expectation) -- harness defect, not an ARC-014 result -> "
                "non_contributory; /diagnose-errors."
            ),
            "C1_or_C4_fail": (
                "Simulation tag leaked into commitment (C1) or world-state "
                "(C4) -> ARC-014 falsified on that channel."
            ),
            "C2_or_C5_fail": (
                "Waking content failed to commit on that channel -> channel "
                "inert / falsifier structure vacuous for that channel -> "
                "non_contributory for that channel."
            ),
            "C3_or_C6_fail": (
                "No divergence despite C1/C2 (or C4/C5) passing -- unexpected "
                "internally-inconsistent state; flag for /diagnose-errors."
            ),
        },
        "gov_reuse_1_note": (
            "Checked V3-EXQ-628 (2026-06-02) rule_state BLOCK-arm data (its "
            "C0+C1) as the closest prior evidence for the rule_state channel; "
            "that manifest predates the 2026-07-12 Experimental Recording "
            "Standard and carries no substrate_hash -> unverifiable substrate "
            "compatibility -> treated as NOT RECOVERABLE per skill Step 2.4 "
            "pre-standard caveat. Also: 628 tested only MECH-319/rule_state "
            "(never the MECH-094/residue world-state channel) and never "
            "tagged ARC-014. 0 failure_autopsy_*.json targets tag ARC-014 "
            "(re-derive brake Step 2.5b not applicable, consistent with "
            "genuine_exp_count=0 in claim_evidence.v1.json as of 2026-08-02)."
        ),
        "dv_symmetry_note": (
            "Both channels' manipulation is a categorical branch (whether "
            "accumulate()/update() executes, gated by an if on the tag), not "
            "a magnitude fed into an argmax/rank/aggregate DV -- not subject "
            "to the broadcast-constant, monotone-rescaling, or "
            "permutation-invariance traps documented in the skill's "
            "DV-SYMMETRY INVARIANCE section."
        ),
        "non_degenerate": degeneracy["non_degenerate"],
        "degeneracy_reason": degeneracy["degeneracy_reason"],
        "degenerate_metrics": degeneracy["degenerate_metrics"],
        "elapsed_seconds": elapsed,
        "dry_run": bool(args.dry_run),
        "config": config,
        "notes": (
            "ARC-014 (Default Mode: non-commitment operating regime) "
            "EVIDENCE. Tests the architectural claim directly (not the "
            "substrate mechanisms in isolation) using the two REAL "
            "already-landed mechanisms that instantiate it: MECH-319 "
            "(simulation_mode_rule_gate, gates LateralPFCAnalog.rule_state = "
            "the arbitration-weight commitment channel) and MECH-094 "
            "(ResidueField.accumulate hypothesis_tag gate = the residue / "
            "world-state channel). Matched-content design: the SAME captured "
            "(z_delta, z_world) content from a real seeded env rollout is "
            "applied to two agents built from the same seed, differing ONLY "
            "in whether the content is tagged waking (hypothesis_tag=False, "
            "simulation_mode=False) or simulation (hypothesis_tag=True, "
            "simulation_mode=True), isolating the tag as the sole causal "
            "variable. PASS = waking tag commits on BOTH channels (rule_state "
            "moves, residue accumulates) AND simulation tag commits on "
            "NEITHER (both ~0 delta) AND the two conditions diverge, for a "
            "majority of seeds. See "
            "REE_assembly/docs/architecture/default_mode.md#arc-014 and "
            "docs/architecture/mech_319_simulation_mode_rule_gate.md."
        ),
    }

    script_dir = Path(__file__).resolve().parents[1]
    out_dir = script_dir.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=bool(args.dry_run),
        config=config,
        seeds=seeds,
        script_path=Path(__file__),
        elapsed_seconds=elapsed,
        z_goal_stream_stats=zg.stats(),
    )

    print(f"OVERALL: {manifest['result']}", flush=True)
    print(f"  n_pass={n_pass}/{len(seeds)} (required {required})", flush=True)
    for r in per_seed:
        print(
            f"  seed {r['seed']}: pass={r['pass']} "
            f"C0={r['C0_path_exercised_no_vacuity']} "
            f"C1={r['C1_sim_suppresses_commitment_rule_state']} "
            f"C2={r['C2_wake_commits_rule_state']} "
            f"C3={r['C3_rule_state_divergence']} "
            f"C4={r['C4_sim_suppresses_worldstate_residue']} "
            f"C5={r['C5_wake_commits_worldstate_residue']} "
            f"C6={r['C6_residue_divergence']} "
            f"wake_rule_delta={r['wake_rule_state_delta']:.3e} "
            f"sim_rule_delta={r['sim_rule_state_delta']:.3e} "
            f"wake_residue_delta={r['wake_residue_delta']:.3e} "
            f"sim_residue_delta={r['sim_residue_delta']:.3e}",
            flush=True,
        )
    print(f"Result written to: {out_path}", flush=True)

    return manifest["result"], str(out_path)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--dry-run", action="store_true")
    _args = _parser.parse_args()
    _result, _manifest_path = main(_args)
    emit_outcome(
        outcome=_result if _result in ("PASS", "FAIL") else "FAIL",
        manifest_path=_manifest_path,
        dry_run=bool(_args.dry_run),
    )
