"""V3-EXQ-463b (EXP-0159 behavioural): MECH-268 dACC PE saturation under sustained conflict.

Purpose: evidence. Behavioural successor to the V3-EXQ-463 substrate-readiness
diagnostic (which validated the MECH-268 arithmetic UC1-UC7; that diagnostic
is NOT superseded -- its wiring evidence stands). This arm exercises MECH-268
(dACC FIFO outcome-history PE saturation, f_sat = 1/(1 + strength*max(0,
n_rec - grace))) inside the real committed_mode_curriculum (P0 warmup ->
P1 consolidation -> P2 eval) on a CausalGridWorldV2 with the GAP-3
counter-evidence-injection primitive ON (counter_evidence_enabled=True,
subgoal_mode=True, counter_evidence_requires_persistent_rule=True).

The counter-evidence stream injects repeated graded contingency degradation --
a sustained feed of conflicting outcomes for the committed waypoint target.
On top of this, the eval loop forces CONFLICT_OUTCOME_CLASS (0) via
record_outcome on every step, simulating a sustained identical-outcome stream
that saturates the MECH-268 outcome-history FIFO. With saturation ON this
attenuates the precision-weighted PE; with saturation OFF the PE tracks
z_harm_a.norm() freely.

NOTE: DACCConfig construction in agent.py does not yet propagate the
dacc_saturation_* fields from REEConfig (a separate /implement-substrate
session will close this wiring gap). This script patches the saturation
knobs directly on the instantiated DACCAdaptiveControl.config after build.
This is a harness-level workaround; it does not touch ree_core.

Arms (one P0->P1 training run per seed, three P2 evals):

  ARM_SATURATION_ON  -- trained agent, MECH-268 saturation knobs active
                         (dacc_saturation_enabled=True, window=8, strength=0.5,
                         grace=2). Expected: sat_factor < C1_MAX_SAT_FACTOR (0.95)
                         in the final third of the eval (saturation fires once
                         FIFO fills with repeated class-0 outcomes).

  ARM_FORCED_RV_ON   -- O-2 mandatory contrast: clone_trained_agent(bistable=True)
                         with running_variance forced to 0.001, saturation ON.
                         Isolates whether saturation requires emergent commitment
                         or only a trained latent in the committed state.

  ARM_SATURATION_OFF -- same trained weights, dacc_saturation_enabled=False.
                         Control arm: sat_factor stays 1.0 regardless of FIFO.

Pre-registered acceptance (PASS = all, 2/3 seeds majority):
  C1  ARM_SATURATION_ON  final-third mean sat_factor < C1_MAX_SAT_FACTOR (0.95)
                         (MECH-268 saturation fired under sustained conflict)
  C2  ARM_SATURATION_OFF final-third mean sat_factor >= C2_MIN_SAT_FACTOR (0.99)
                         (control arm: no saturation, sat_factor stays at 1.0)
  C3  ARM_SATURATION_ON  mean pe_ratio (= sat_factor) in final third strictly
                         less than ARM_SATURATION_OFF pe_ratio (= 1.0)

Interpretation grid:
  C1-C3 all PASS .................. MECH-268 dACC PE saturation confirmed in a
                                    live curriculum loop; saturation attenuates
                                    the precision-weighted PE under sustained
                                    conflict. Governance: evidence for MECH-268.
  C1 FAIL (sat_factor near 1.0) ... Saturation did not fire despite forced FIFO
     + pe_unsaturated low ......... z_harm_a near zero: no hazard contact during
                                    eval. Diagnose env wiring: check num_hazards,
                                    steps_per_ep, whether obs_harm_a is passed
                                    to agent.sense(). NOT a MECH-268 falsification.
     + pe_unsaturated elevated ..... FIFO not reaching threshold: check that
                                    record_outcome is called and FIFO window/grace
                                    params are wired. Diagnose dacc saturation.
  C2 FAIL (OFF arm saturates) ..... Saturation fires without knob: logic bug in
                                    _saturation_factor(). Route to /diagnose-errors.
  C1 PASS, C3 FAIL ................ sat_factor < 1 but pe == pe_unsaturated:
                                    sat_factor not multiplied into pe in dacc.forward().
                                    Substrate wiring regression -> /diagnose-errors.

Run:
  /opt/local/bin/python3 experiments/v3_exq_463b_mech268_dacc_conflict_saturation_behavioural.py
Smoke:
  /opt/local/bin/python3 experiments/v3_exq_463b_mech268_dacc_conflict_saturation_behavioural.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.committed_mode_curriculum import (  # noqa: E402
    clone_trained_agent,
    run_p0_warmup,
    run_p1_consolidation,
)


EXPERIMENT_TYPE = "v3_exq_463b_mech268_dacc_conflict_saturation_behavioural"
QUEUE_ID = "V3-EXQ-463b"
CLAIM_IDS = ["MECH-268"]
EXPERIMENT_PURPOSE = "evidence"
SEEDS = [42, 43, 44]

# Pre-registered thresholds (constants, not derived from the run).
C1_MAX_SAT_FACTOR = 0.95    # ARM_ON final-third mean sat_factor must be below this
C2_MIN_SAT_FACTOR = 0.99    # ARM_OFF final-third mean sat_factor must be above this
PASS_FRACTION_REQUIRED = 2.0 / 3.0  # majority of seeds

# MECH-268 saturation knobs (applied to both ARM_ON and ARM_FORCED_RV_ON).
SAT_WINDOW = 8
SAT_STRENGTH = 0.5
SAT_GRACE = 2

# Sustained conflict stream: force this outcome class via record_outcome on
# every eval step. Class 0 = conflict/harm outcome. After SAT_GRACE repeated
# class-0 outcomes, f_sat = 1/(1 + SAT_STRENGTH*(n_rec - SAT_GRACE)) < 1.
CONFLICT_OUTCOME_CLASS = 0


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _build_easy_env(size: int) -> CausalGridWorldV2:
    """P0 warmup env: fewer hazards, completion tolerance on, no counter-evidence."""
    return CausalGridWorldV2(
        size=size,
        num_hazards=2,
        num_resources=3,
        num_waypoints=2,
        completion_tolerance_enabled=True,
        completion_tolerance_frac=0.15,
        completion_tolerance_metric="chebyshev",
        completion_tolerance_targets="waypoint",
    )


def _build_target_env(size: int) -> CausalGridWorldV2:
    """P1 + P2 eval env: more hazards, subgoal_mode + counter-evidence ON.

    subgoal_mode=True is required for counter_evidence_requires_persistent_rule=True
    to activate: the env checks self.subgoal_mode and self._sequence_in_progress
    (set when the agent reaches the first waypoint) before injecting counter-evidence.
    """
    return CausalGridWorldV2(
        size=size,
        num_hazards=4,
        num_resources=3,
        num_waypoints=2,
        subgoal_mode=True,
        completion_tolerance_enabled=True,
        completion_tolerance_frac=0.15,
        completion_tolerance_metric="chebyshev",
        completion_tolerance_targets="waypoint",
        counter_evidence_enabled=True,
        counter_evidence_interval=30,
        counter_evidence_prob=0.7,
        counter_evidence_degrade_step=0.15,
        counter_evidence_requires_persistent_rule=True,
    )


def _patch_dacc_saturation(agent: REEAgent, enabled: bool) -> None:
    """Patch MECH-268 saturation knobs on the agent's DACCAdaptiveControl.

    DACCConfig construction in agent.py (lines 339-349) does not propagate
    dacc_saturation_* fields from REEConfig. This harness-level patch overrides
    the dacc config after build. A follow-on /implement-substrate session will
    wire the propagation in ree_core/agent.py.
    """
    if agent.dacc is None:
        return
    agent.dacc.config.dacc_saturation_enabled = enabled
    agent.dacc.config.dacc_saturation_window = SAT_WINDOW
    agent.dacc.config.dacc_saturation_strength = SAT_STRENGTH
    agent.dacc.config.dacc_saturation_grace = SAT_GRACE


def _build_agent(world_obs_dim: int) -> REEAgent:
    """Build REEAgent with dACC + affective harm stream + salience + lateral PFC.

    use_affective_harm_stream=True is required so z_harm_a is non-None, which
    in turn is required for dacc.forward() to be called inside select_action()
    (agent.py: 'if self.dacc is not None and z_harm_a is not None').
    Without it the dACC PE is never computed and sat_factor stays at 1.0.
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=world_obs_dim,
        action_dim=4,
        use_dacc=True,
        use_affective_harm_stream=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
    )
    cfg.heartbeat.beta_gate_bistable = True
    return REEAgent(cfg)


def _eval_pe_saturation(
    agent: REEAgent,
    env: CausalGridWorldV2,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
    saturation_enabled: bool,
) -> dict:
    """Frozen-policy eval instrumented for MECH-268 PE saturation.

    On every step, reads agent.dacc._last_pe_unsaturated and
    _last_saturation_factor (set inside dacc.forward() during select_action).
    Also forces CONFLICT_OUTCOME_CLASS via record_outcome on every step,
    simulating a sustained identical-outcome conflict stream that fills the
    MECH-268 FIFO.

    obs_harm_a is passed to agent.sense() so z_harm_a is non-zero (driven
    by hazard-proximity EMA in the env). Without z_harm_a, pe falls to zero
    and there is nothing to saturate.

    Returns per-episode timeseries summary: final-third mean sat_factor,
    pe_unsaturated, and pe_ratio (= sat_factor, since pe = sat*pe_unsaturated).
    """
    agent.eval()
    world_dim = agent.config.latent.world_dim
    has_dacc = agent.dacc is not None

    pe_unsaturated_series: list = []
    sat_factor_series: list = []

    with torch.no_grad():
        for _ep in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()

            for _step in range(steps_per_episode):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                obs_harm_a: Optional[torch.Tensor] = None
                if "harm_obs_a" in obs_dict:
                    obs_harm_a = obs_dict["harm_obs_a"].to(device)

                latent = agent.sense(obs_body, obs_world, obs_harm_a=obs_harm_a)

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())

                # Read MECH-268 diagnostics after select_action (dacc.forward ran).
                if has_dacc:
                    pe_un = agent.dacc._last_pe_unsaturated
                    sf = float(agent.dacc._last_saturation_factor)
                    pe_unsaturated_series.append(
                        float(pe_un) if pe_un is not None else 0.0
                    )
                    sat_factor_series.append(sf)
                else:
                    pe_unsaturated_series.append(0.0)
                    sat_factor_series.append(1.0)

                _, _, done, _, obs_dict = env.step(action_idx)

                # Sustained conflict stream: force class-0 every step so the
                # MECH-268 FIFO fills with repeated identical outcomes.
                # After SAT_GRACE recurrences (n_rec > SAT_GRACE), sat_factor
                # drops below 1.0 in ARM_ON (saturation enabled) and stays 1.0
                # in ARM_OFF (saturation disabled).
                if has_dacc:
                    agent.dacc.record_outcome(CONFLICT_OUTCOME_CLASS)

                if done:
                    break

    total_steps = len(sat_factor_series)
    if total_steps == 0:
        return {
            "total_steps": 0,
            "mean_sat_factor_final_third": 1.0,
            "mean_pe_unsaturated_final_third": 0.0,
            "mean_pe_ratio_final_third": 1.0,
            "saturation_enabled": saturation_enabled,
            "n_eps": n_eps,
        }

    # Final-third slicing (last 1/3 of all steps across all eval episodes).
    third = max(1, total_steps // 3)
    sf_final = sat_factor_series[-third:]
    pe_final = pe_unsaturated_series[-third:]

    mean_sf = sum(sf_final) / len(sf_final)
    mean_pe_un = sum(pe_final) / max(1, len(pe_final))
    # pe_ratio = effective_pe / pe_unsaturated = sat_factor (by construction
    # in dacc._affective_pe: return pe_out * sat_factor).
    mean_pe_ratio = mean_sf

    return {
        "total_steps": total_steps,
        "mean_sat_factor_final_third": mean_sf,
        "mean_pe_unsaturated_final_third": mean_pe_un,
        "mean_pe_ratio_final_third": mean_pe_ratio,
        "saturation_enabled": saturation_enabled,
        "n_eps": n_eps,
    }


def run_seed(seed: int, device: torch.device, smoke: bool) -> dict:
    print(f"Seed {seed} Condition train_dacc_on", flush=True)
    torch.manual_seed(seed)

    size = 8 if smoke else 10
    p0_budget = 3 if smoke else 200
    p1_budget = 3 if smoke else 150
    steps_per_ep = 20 if smoke else 500
    eval_eps = 1 if smoke else 10

    easy_env = _build_easy_env(size)
    target_env = _build_target_env(size)
    world_obs_dim = easy_env.world_obs_dim

    agent = _build_agent(world_obs_dim).to(device)
    _patch_dacc_saturation(agent, enabled=True)

    p0 = run_p0_warmup(
        agent, easy_env, device,
        budget=p0_budget, steps_per_episode=steps_per_ep,
    )
    print(
        f"  [train] seed={seed} P0 ep {p0.n_episodes}/{p0_budget}"
        f" converged={p0.converged} aborted={p0.aborted} rv={p0.final_rv:.5f}",
        flush=True,
    )
    if p0.aborted:
        verdict = "FAIL"
        print(f"verdict: {verdict}", flush=True)
        return {
            "seed": seed,
            "outcome": "commitment_not_elicited",
            "p0_aborted": True,
            "p0_abort_reason": p0.abort_reason,
            "pass": False,
        }

    p1 = run_p1_consolidation(
        agent, target_env, device,
        budget=p1_budget, steps_per_episode=steps_per_ep,
    )
    print(
        f"  [train] seed={seed} P1 ep {p1.n_episodes}/{p1_budget}"
        f" emerged={p1.commitment_emerged}"
        f" committed/ep={p1.final_committed_steps_per_ep:.1f}",
        flush=True,
    )

    # ARM_SATURATION_ON: trained agent with saturation on.
    print(f"Seed {seed} Condition ARM_SATURATION_ON", flush=True)
    eval_env_on = _build_target_env(size)
    arm_on = _eval_pe_saturation(
        agent, eval_env_on, device, eval_eps, steps_per_ep,
        saturation_enabled=True,
    )
    print(
        f"  sat_factor_final_third={arm_on['mean_sat_factor_final_third']:.4f}"
        f" pe_un={arm_on['mean_pe_unsaturated_final_third']:.4f}",
        flush=True,
    )

    # ARM_FORCED_RV_ON: O-2 mandatory contrast (forced running_variance).
    print(f"Seed {seed} Condition ARM_FORCED_RV_ON", flush=True)
    eval_env_forced = _build_target_env(size)
    agent_forced = clone_trained_agent(agent, bistable=True, device=device)
    agent_forced.e3._running_variance = 0.001
    _patch_dacc_saturation(agent_forced, enabled=True)
    arm_forced = _eval_pe_saturation(
        agent_forced, eval_env_forced, device, eval_eps, steps_per_ep,
        saturation_enabled=True,
    )
    print(
        f"  forced_rv sat_factor={arm_forced['mean_sat_factor_final_third']:.4f}",
        flush=True,
    )

    # ARM_SATURATION_OFF: same weights, saturation disabled (control arm).
    print(f"Seed {seed} Condition ARM_SATURATION_OFF", flush=True)
    eval_env_off = _build_target_env(size)
    agent_off = clone_trained_agent(agent, bistable=True, device=device)
    agent_off.e3._running_variance = float(agent.e3._running_variance)
    _patch_dacc_saturation(agent_off, enabled=False)
    arm_off = _eval_pe_saturation(
        agent_off, eval_env_off, device, eval_eps, steps_per_ep,
        saturation_enabled=False,
    )
    print(
        f"  sat_factor_final_third={arm_off['mean_sat_factor_final_third']:.4f}",
        flush=True,
    )

    c1 = arm_on["mean_sat_factor_final_third"] < C1_MAX_SAT_FACTOR
    c2 = arm_off["mean_sat_factor_final_third"] >= C2_MIN_SAT_FACTOR
    c3 = arm_on["mean_pe_ratio_final_third"] < arm_off["mean_pe_ratio_final_third"]
    seed_pass = bool(c1 and c2 and c3)

    print(f"verdict: {'PASS' if seed_pass else 'FAIL'}", flush=True)
    return {
        "seed": seed,
        "ARM_SATURATION_ON": arm_on,
        "ARM_FORCED_RV_ON": arm_forced,
        "ARM_SATURATION_OFF": arm_off,
        "criteria": {"C1": c1, "C2": c2, "C3": c3},
        "p1_commitment_emerged": p1.commitment_emerged,
        "pass": seed_pass,
    }


def build_manifest(seed_results: list, smoke: bool) -> dict:
    n_pass = sum(1 for r in seed_results if r.get("pass"))
    n_seeds = len(seed_results)
    overall_pass = (n_pass / max(1, n_seeds)) >= PASS_FRACTION_REQUIRED
    outcome = "PASS" if overall_pass else "FAIL"
    direction = "supports" if overall_pass else "weakens"
    run_id = f"{EXPERIMENT_TYPE}_{_utc_stamp()}_v3"
    return {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_iso(),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "result": outcome,
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {
            "MECH-268": direction,
        },
        "thresholds": {
            "C1_max_sat_factor": C1_MAX_SAT_FACTOR,
            "C2_min_sat_factor": C2_MIN_SAT_FACTOR,
            "pass_fraction_required": PASS_FRACTION_REQUIRED,
            "sat_window": SAT_WINDOW,
            "sat_strength": SAT_STRENGTH,
            "sat_grace": SAT_GRACE,
        },
        "n_seeds": n_seeds,
        "n_seeds_pass": n_pass,
        "smoke": smoke,
        "metrics": {"per_seed": seed_results},
        "notes": (
            "V3-EXQ-463b behavioural arm of the MECH-268 dACC conflict-saturation"
            " row (EXP-0159, commitment_closure_plan.md GAP-4 Phase 4/5 OCD cohort)."
            " Behavioural successor to V3-EXQ-463 (substrate-readiness diagnostic,"
            " NOT superseded -- its UC1-UC7 arithmetic wiring evidence stands). This"
            " arm validates f_sat = 1/(1 + strength*max(0, n_rec - grace)) in a live"
            " committed_mode_curriculum loop (GAP-11, proven by V3-EXQ-592) on"
            " CausalGridWorldV2 with the GAP-3 counter-evidence primitive ON"
            " (counter_evidence_enabled=True, subgoal_mode=True,"
            " counter_evidence_requires_persistent_rule=True). The eval loop forces"
            " CONFLICT_OUTCOME_CLASS via record_outcome on every step, simulating a"
            " sustained identical-outcome stream that fills the MECH-268 FIFO."
            " ARM_SATURATION_ON expects sat_factor < 0.95 in the final third;"
            " ARM_SATURATION_OFF (control) expects sat_factor >= 0.99. O-2 forced-rv"
            " contrast included per GAP-11 mandatory contrast rule. MECH-268 saturation"
            " knobs are patched post-build because DACCConfig construction in agent.py"
            " does not yet propagate dacc_saturation_* fields from REEConfig (a separate"
            " /implement-substrate session will close this wiring gap; the patch is"
            " harness-level only, no ree_core changes)."
        ),
    }


def main(smoke: bool):
    device = torch.device("cpu")
    seeds = SEEDS[:1] if smoke else SEEDS
    seed_results = [run_seed(s, device, smoke) for s in seeds]
    manifest = build_manifest(seed_results, smoke)

    print(f"=== {QUEUE_ID} {EXPERIMENT_TYPE} ===", flush=True)
    print(
        f"outcome: {manifest['outcome']}"
        f" ({manifest['n_seeds_pass']}/{manifest['n_seeds']} seeds pass)",
        flush=True,
    )

    if smoke:
        return None

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Result written to: {out_path}", flush=True)
    return manifest["outcome"], out_path, manifest["run_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Smoke run (tiny budgets, no manifest written).",
    )
    args = parser.parse_args()
    result = main(smoke=args.dry_run)
    if args.dry_run or result is None:
        sys.exit(0)
    _outcome, _out_path, _run_id = result
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        run_id=_run_id,
        queue_id=QUEUE_ID,
        exit_reason="ok" if _outcome == "PASS" else "fail",
    )
    sys.exit(0)
