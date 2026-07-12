"""V3-EXQ-729: MECH-268 dACC conflict-saturation -- LIVE-LOOP (ecological) falsifier.

Purpose: evidence. Ecological analogue of V3-EXQ-463b. 463b validated the
MECH-268 arithmetic (f_sat = 1/(1 + strength*max(0, n_rec - grace))) but ONLY by
manually injecting record_outcome() on every eval step -- a synthetic harness
stream, not the agent's own loop. The 2026-07-09 design/implementation audit
(design_implementation_audit_2026-07-09.md sec 3A F-C3 + sec 6) found
DACC.record_outcome had ZERO live callers, so dacc_saturation_enabled=True was
silently inert ecologically. The F-C3 fix (ree-v3 main c7fc045) wired
record_outcome into the agent.py select_action tail (gated on the REEConfig
dacc_saturation_enabled flag; outcome class = harm-vs-no-harm binary on z_harm_a
thresholded by contextual_safety_harm_threshold) AND propagated the
dacc_saturation_* knobs from REEConfig into DACCConfig.

This experiment closes the "validated only via synthetic injection" gap: it runs
the real agent loop (sense -> generate_trajectories -> select_action) in an env
that produces SUSTAINED harm (chronic hazard proximity), makes NO manual
record_outcome call anywhere, and asserts (a) the DACC _outcome_history FIFO is
genuinely populated by the LIVE select_action path (a counting spy on
record_outcome proves the calls originate in the substrate, not the harness), and
(b) the final-third mean sat_factor drops below 1.0 in the ON arm while staying
~1.0 in the OFF arm. This is the ecological analogue of 463b's C1/C2/C3.

Design (clean 1-variable contrast + reusable OFF baseline):
  Each (seed, arm) is an independent arm_cell (RNG fully reset on entry). BOTH
  arms build the agent with saturation OFF and run the SAME light P0 warmup, so
  their trained weights are bit-identical by construction (deterministic training
  from the reset RNG). The ONLY difference is at eval:

  ARM_ON  -- flip dacc_saturation_enabled=True (REEConfig flag drives the live
             record_outcome call; DACCConfig flag + window/strength/grace drive
             f_sat) for the eval loop. Expected: the live path fills the FIFO with
             the recurring harm class and final-third mean sat_factor < 0.95.
  ARM_OFF -- leave saturation disabled. The live path makes NO record_outcome call
             at all (byte-identical gating), the FIFO stays empty, sat_factor
             stays 1.0. This is the control AND the reusable baseline (minted
             reuse-eligible: a successor iteration that only tweaks the ON knobs
             reuses this OFF cell unchanged).

  NOTE on training: P0 warmup trains E1+E2 (world model) only; it does NOT train
  the affective-harm encoder, so z_harm_a is a fixed projection of harm_obs_a and
  its norm tracks hazard proximity regardless of training depth. Training here
  exists to make the agent a real (non-degenerate) actor, not to shape the harm
  stream. No downstream head is trained on detached latents, so the phased-training
  moving-target hazard (EXQ-166b/c/d) does not apply -- there is no probe head.

Arms x seeds: 2 arms x 3 seeds. One verdict per seed (ON+OFF criteria combined).

Pre-registered acceptance (PASS = all criteria, 2/3 seeds majority):
  C1  ARM_ON  final-third mean sat_factor < C1_MAX_SAT_FACTOR (0.95)
              (MECH-268 saturation fired ecologically under sustained harm)
  C2  ARM_OFF final-third mean sat_factor >= C2_MIN_SAT_FACTOR (0.99)
              (control: saturation disabled, no attenuation)
  C3  ARM_ON  final-third mean sat_factor strictly < ARM_OFF final-third mean
  C4  ARM_ON  LIVE population: live_record_calls > 0 AND fifo_len_end > 0 AND
              harm_class_fraction >= HARM_CLASS_FRAC_MIN (0.5) AND
              live_record_harm_fraction >= HARM_CLASS_FRAC_MIN (0.5)
              (the FIFO is filled by the live select_action path with the recurring
              harm class -- NO manual injection anywhere in this script)
  C5  ARM_OFF gating: live_record_calls == 0 AND fifo_len_end == 0
              (saturation disabled -> the live path makes no record_outcome call)

Interpretation grid:
  C1-C5 all PASS ............ MECH-268 dACC PE saturation confirmed in the LIVE
                             agent loop; the ecological gap 463b left open is
                             closed. Governance: evidence for MECH-268.
  C4 FAIL, live_calls==0 .... the F-C3 live wiring did not fire: check
                             agent.config.dacc_saturation_enabled is set for the
                             ON eval and c7fc045 is present. NOT a MECH-268
                             falsification -- a wiring regression -> /diagnose-errors.
  C4 FAIL, harm_frac low .... env not producing chronic harm: z_harm_a.norm() did
                             not clear contextual_safety_harm_threshold. Raise
                             EVAL_HAZARDS / shrink ENV_SIZE. NOT a falsification.
  C1 FAIL, C4 PASS .......... FIFO populated with harm class but sat did not drop:
                             check window/strength/grace propagation into DACCConfig
                             (c7fc045 second gap) -> /diagnose-errors.
  C5 FAIL (OFF records) ..... gating bug: live path recorded with the flag off
                             -> /diagnose-errors.

Run:
  /opt/local/bin/python3 experiments/v3_exq_729_mech268_dacc_saturation_liveloop.py
Smoke:
  /opt/local/bin/python3 experiments/v3_exq_729_mech268_dacc_saturation_liveloop.py --dry-run
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
from experiments.committed_mode_curriculum import run_p0_warmup  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_729_mech268_dacc_saturation_liveloop"
QUEUE_ID = "V3-EXQ-729"
CLAIM_IDS = ["MECH-268"]
EXPERIMENT_PURPOSE = "evidence"
SEEDS = [42, 43, 44]
ARMS = ["OFF", "ON"]

# Pre-registered thresholds (constants, not derived from the run).
C1_MAX_SAT_FACTOR = 0.95     # ARM_ON final-third mean sat_factor must be below this
C2_MIN_SAT_FACTOR = 0.99     # ARM_OFF final-third mean sat_factor must be above this
HARM_CLASS_FRAC_MIN = 0.5    # ARM_ON chronic-harm confirmation (both z_harm_a and live-recorded)
PASS_FRACTION_REQUIRED = 2.0 / 3.0  # majority of seeds

# MECH-268 saturation knobs (ARM_ON eval only).
SAT_WINDOW = 8
SAT_STRENGTH = 0.5
SAT_GRACE = 2

# Substrate constant the live path thresholds z_harm_a against (for reporting).
CONTEXTUAL_SAFETY_HARM_THRESHOLD = 0.05

# Env geometry. Chronic-harm eval env is deliberately hazard-dense so z_harm_a
# stays above the harm threshold every step regardless of policy. Train env is
# milder (a functioning world model is enough; the harm stream is not trained).
ENV_SIZE = 7
ENV_RESOURCES = 2
ENV_WAYPOINTS = 1
TRAIN_HAZARDS = 3
EVAL_HAZARDS = 6

# Schedule (real / smoke). Eval is ONE continuous chronic-harm exposure of
# EVAL_TOTAL_STEPS steps: the agent is respawned via env.reset() on death but the
# DACC outcome FIFO (and the env's own harm_obs_a_ema) persist across respawns, so
# habituation accrues over the agent's lifetime rather than resetting per episode.
P0_BUDGET = 120
STEPS_PER_EP = 300
EVAL_TOTAL_STEPS = 450
P0_BUDGET_SMOKE = 3
STEPS_PER_EP_SMOKE = 20
EVAL_TOTAL_STEPS_SMOKE = 40


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _build_train_env() -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=ENV_SIZE,
        num_hazards=TRAIN_HAZARDS,
        num_resources=ENV_RESOURCES,
        num_waypoints=ENV_WAYPOINTS,
    )


def _build_eval_env() -> CausalGridWorldV2:
    """Chronic-harm eval env: hazard-dense so z_harm_a stays supra-threshold."""
    return CausalGridWorldV2(
        size=ENV_SIZE,
        num_hazards=EVAL_HAZARDS,
        num_resources=ENV_RESOURCES,
        num_waypoints=ENV_WAYPOINTS,
    )


def _build_agent(world_obs_dim: int) -> REEAgent:
    """REEAgent with dACC + affective harm stream. Saturation OFF at build.

    use_affective_harm_stream=True is required so z_harm_a is non-None, which is
    the precondition for dacc.forward() to run inside select_action (and hence
    for _last_saturation_factor to update). Built with saturation OFF so BOTH
    arms train identically; the ON arm flips it on for eval only.
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=world_obs_dim,
        action_dim=4,
        use_dacc=True,
        use_affective_harm_stream=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        dacc_saturation_enabled=False,
    )
    cfg.heartbeat.beta_gate_bistable = True
    return REEAgent(cfg)


def _enable_saturation(agent: REEAgent) -> None:
    """Flip MECH-268 saturation ON for the ON-arm eval.

    Sets BOTH the REEConfig flag (drives the live select_action record_outcome
    call -- F-C3 wiring) and the DACCConfig flag + knobs (drive f_sat). c7fc045
    propagates these at build; here we flip them post-build for the eval arm so
    training stays identical to the OFF arm.
    """
    agent.config.dacc_saturation_enabled = True
    agent.config.dacc_saturation_window = SAT_WINDOW
    agent.config.dacc_saturation_strength = SAT_STRENGTH
    agent.config.dacc_saturation_grace = SAT_GRACE
    if agent.dacc is not None:
        agent.dacc.config.dacc_saturation_enabled = True
        agent.dacc.config.dacc_saturation_window = SAT_WINDOW
        agent.dacc.config.dacc_saturation_strength = SAT_STRENGTH
        agent.dacc.config.dacc_saturation_grace = SAT_GRACE


def _install_record_spy(agent: REEAgent) -> dict:
    """Wrap dacc.record_outcome with a counter to PROVE live population.

    The live select_action tail (c7fc045) is the only caller of record_outcome in
    the substrate (ClosureOperator uses reset_outcome_history, not record_outcome).
    This script itself NEVER calls record_outcome. So a non-zero counter proves the
    FIFO was filled by the live agent path, not by any harness injection.
    """
    counter = {"n": 0, "classes": []}
    if agent.dacc is None:
        return counter
    orig = agent.dacc.record_outcome

    def _spy(outcome_class: int) -> None:
        counter["n"] += 1
        counter["classes"].append(int(outcome_class))
        return orig(outcome_class)

    agent.dacc.record_outcome = _spy  # instance-level shadow; live path resolves to it
    return counter


def _eval_live_loop(
    agent: REEAgent,
    env: CausalGridWorldV2,
    device: torch.device,
    total_steps_target: int,
    saturation_enabled: bool,
) -> dict:
    """Continuous chronic-harm eval on the REAL agent loop. NO manual record_outcome.

    Runs one continuous exposure of total_steps_target steps. agent.reset() is
    called ONCE at the start; on env `done` (agent death or step cap) the env is
    respawned via env.reset() but the agent is NOT reset, so the DACC outcome FIFO
    (which agent.reset would clear via dacc.reset) and the env's persistent
    harm_obs_a_ema accrue over the whole lifetime -- the chronic-harm regime the
    saturation mechanism is meant to model. Reads _last_saturation_factor /
    _last_pe_unsaturated (set inside dacc.forward during select_action) and
    z_harm_a.norm() each step. The ON arm's live select_action tail fills the FIFO;
    the OFF arm's does not (gated off), so its FIFO stays empty and sat stays 1.0.
    """
    agent.eval()
    world_dim = agent.config.latent.world_dim
    has_dacc = agent.dacc is not None

    sat_factor_series: list = []
    pe_unsaturated_series: list = []
    z_harm_a_norm_series: list = []
    n_respawns = 0

    with torch.no_grad():
        agent.reset()                       # single clean init for the lifetime
        spy = _install_record_spy(agent)    # after reset so the count is eval-only
        _, obs_dict = env.reset()

        for _step in range(total_steps_target):
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
            # LIVE path: select_action's tail records the outcome (ON arm only).
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())

            if has_dacc:
                sf = float(agent.dacc._last_saturation_factor)
                pe_un = agent.dacc._last_pe_unsaturated
                sat_factor_series.append(sf)
                pe_unsaturated_series.append(
                    float(pe_un) if pe_un is not None else 0.0
                )
                zha = (
                    agent._current_latent.z_harm_a
                    if agent._current_latent is not None
                    else None
                )
                z_harm_a_norm_series.append(
                    float(zha.detach().norm().item()) if zha is not None else 0.0
                )
            else:
                sat_factor_series.append(1.0)
                pe_unsaturated_series.append(0.0)
                z_harm_a_norm_series.append(0.0)

            # NO manual record_outcome -- the whole point of the ecological test.
            _, _, done, _, obs_dict = env.step(action_idx)
            if done:
                _, obs_dict = env.reset()   # respawn; FIFO + harm_obs_a_ema persist
                n_respawns += 1

    total_steps = len(sat_factor_series)
    fifo_len_end = len(agent.dacc._outcome_history) if has_dacc else 0
    live_calls = spy["n"]
    live_harm = sum(1 for c in spy["classes"] if c == 1)
    live_record_harm_fraction = (live_harm / live_calls) if live_calls else 0.0
    harm_class_fraction = (
        sum(1 for z in z_harm_a_norm_series if z > CONTEXTUAL_SAFETY_HARM_THRESHOLD)
        / max(1, total_steps)
    )

    if total_steps == 0:
        return {
            "total_steps": 0,
            "mean_sat_factor_final_third": 1.0,
            "mean_pe_unsaturated_final_third": 0.0,
            "mean_z_harm_a_norm": 0.0,
            "harm_class_fraction": 0.0,
            "live_record_calls": live_calls,
            "live_record_harm_fraction": live_record_harm_fraction,
            "fifo_len_end": fifo_len_end,
            "saturation_enabled": saturation_enabled,
            "n_respawns": n_respawns,
        }

    third = max(1, total_steps // 3)
    sf_final = sat_factor_series[-third:]
    pe_final = pe_unsaturated_series[-third:]

    return {
        "total_steps": total_steps,
        "mean_sat_factor_final_third": sum(sf_final) / len(sf_final),
        "mean_pe_unsaturated_final_third": sum(pe_final) / len(pe_final),
        "mean_z_harm_a_norm": sum(z_harm_a_norm_series) / max(1, total_steps),
        "harm_class_fraction": harm_class_fraction,
        "live_record_calls": live_calls,
        "live_record_harm_fraction": live_record_harm_fraction,
        "fifo_len_end": fifo_len_end,
        "saturation_enabled": saturation_enabled,
        "n_respawns": n_respawns,
    }


def _arm_config_slice(arm: str, p0_budget: int, steps_per_ep: int, eval_total_steps: int) -> dict:
    """Declared config slice for the arm cell fingerprint.

    Declares ONLY what the cell computation reads. The OFF arm's slice omits the
    saturation knobs (inert when disabled) so a successor that only tweaks ON
    knobs leaves the OFF fingerprint unchanged -> OFF baseline reuse HITs.
    """
    slice_ = {
        "env_train": {
            "size": ENV_SIZE, "num_hazards": TRAIN_HAZARDS,
            "num_resources": ENV_RESOURCES, "num_waypoints": ENV_WAYPOINTS,
        },
        "env_eval": {
            "size": ENV_SIZE, "num_hazards": EVAL_HAZARDS,
            "num_resources": ENV_RESOURCES, "num_waypoints": ENV_WAYPOINTS,
        },
        "schedule": {
            "p0_budget": p0_budget, "steps_per_ep": steps_per_ep, "eval_total_steps": eval_total_steps,
        },
        "agent": {
            "body_obs_dim": 12, "action_dim": 4, "use_dacc": True,
            "use_affective_harm_stream": True, "use_salience_coordinator": True,
            "use_lateral_pfc_analog": True, "beta_gate_bistable": True,
        },
        "harm_threshold": CONTEXTUAL_SAFETY_HARM_THRESHOLD,
    }
    if arm == "ON":
        slice_["saturation"] = {
            "enabled": True, "window": SAT_WINDOW,
            "strength": SAT_STRENGTH, "grace": SAT_GRACE,
        }
    else:
        slice_["saturation"] = {"enabled": False}
    return slice_


def run_cell(seed: int, arm: str, device: torch.device, smoke: bool) -> dict:
    """One (seed, arm) cell: reset RNG, train (sat OFF), eval (arm's sat state)."""
    p0_budget = P0_BUDGET_SMOKE if smoke else P0_BUDGET
    steps_per_ep = STEPS_PER_EP_SMOKE if smoke else STEPS_PER_EP
    eval_total_steps = EVAL_TOTAL_STEPS_SMOKE if smoke else EVAL_TOTAL_STEPS

    slice_ = _arm_config_slice(arm, p0_budget, steps_per_ep, eval_total_steps)
    # OFF arm is the reusable baseline: include_driver_script_in_hash=False so a
    # future consumer's different driver can match this mint by construction.
    with arm_cell(
        seed,
        config_slice=slice_,
        script_path=Path(__file__),
        config_slice_declared=True,
        include_driver_script_in_hash=False,
    ) as cell:
        # cell entry already did a complete RNG reset.
        train_env = _build_train_env()
        eval_env = _build_eval_env()
        assert train_env.world_obs_dim == eval_env.world_obs_dim, (
            "train/eval world_obs_dim mismatch -- agent cannot sense both envs"
        )
        agent = _build_agent(train_env.world_obs_dim).to(device)

        p0 = run_p0_warmup(
            agent, train_env, device,
            budget=p0_budget, steps_per_episode=steps_per_ep,
        )
        print(
            f"  [arm {arm}] seed={seed} P0 ep {p0.n_episodes}/{p0_budget}"
            f" converged={p0.converged} aborted={p0.aborted} rv={p0.final_rv:.5f}",
            flush=True,
        )
        # P0 abort is a commitment-gate finding, orthogonal to MECH-268; the agent
        # is still a functioning actor. Proceed to eval regardless (log only).

        if arm == "ON":
            _enable_saturation(agent)

        res = _eval_live_loop(
            agent, eval_env, device, eval_total_steps,
            saturation_enabled=(arm == "ON"),
        )
        res["p0_converged"] = bool(p0.converged)
        res["p0_aborted"] = bool(p0.aborted)
        cell.stamp(res)

    print(
        f"  [arm {arm}] seed={seed} sat_final={res['mean_sat_factor_final_third']:.4f}"
        f" live_calls={res['live_record_calls']} fifo_end={res['fifo_len_end']}"
        f" harm_frac={res['harm_class_fraction']:.3f}"
        f" zha_norm={res['mean_z_harm_a_norm']:.4f}",
        flush=True,
    )
    return res


def run_seed(seed: int, device: torch.device, smoke: bool) -> dict:
    print(f"Seed {seed} Condition liveloop", flush=True)
    arm_off = run_cell(seed, "OFF", device, smoke)
    arm_on = run_cell(seed, "ON", device, smoke)

    c1 = arm_on["mean_sat_factor_final_third"] < C1_MAX_SAT_FACTOR
    c2 = arm_off["mean_sat_factor_final_third"] >= C2_MIN_SAT_FACTOR
    c3 = arm_on["mean_sat_factor_final_third"] < arm_off["mean_sat_factor_final_third"]
    c4 = bool(
        arm_on["live_record_calls"] > 0
        and arm_on["fifo_len_end"] > 0
        and arm_on["harm_class_fraction"] >= HARM_CLASS_FRAC_MIN
        and arm_on["live_record_harm_fraction"] >= HARM_CLASS_FRAC_MIN
    )
    c5 = bool(arm_off["live_record_calls"] == 0 and arm_off["fifo_len_end"] == 0)
    seed_pass = bool(c1 and c2 and c3 and c4 and c5)

    print(f"verdict: {'PASS' if seed_pass else 'FAIL'}", flush=True)
    return {
        "seed": seed,
        "ARM_ON": arm_on,
        "ARM_OFF": arm_off,
        "criteria": {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5},
        "pass": seed_pass,
        "arm_fingerprints": {
            "ARM_ON": arm_on.get("arm_fingerprint"),
            "ARM_OFF": arm_off.get("arm_fingerprint"),
        },
    }


def build_manifest(seed_results: list, smoke: bool) -> dict:
    n_pass = sum(1 for r in seed_results if r.get("pass"))
    n_seeds = len(seed_results)
    overall_pass = (n_pass / max(1, n_seeds)) >= PASS_FRACTION_REQUIRED
    outcome = "PASS" if overall_pass else "FAIL"
    direction = "supports" if overall_pass else "weakens"
    run_id = f"{EXPERIMENT_TYPE}_{_utc_stamp()}_v3"

    # Non-degeneracy self-report (scoring net for evidence runs, V3-EXQ-514m class).
    # Load-bearing signal = the per-seed ON-vs-OFF sat_factor contrast (degenerate
    # if the arms never separate -> mechanism not exercised / wiring dead) and the
    # ON harm-class fraction (degenerate if no chronic harm was produced). Either
    # makes the "supports"/"weakens" read vacuous, so the indexer scoring-excludes
    # it rather than deferring to a manual failure-autopsy.
    sat_contrast_groups = [
        [
            r.get("ARM_ON", {}).get("mean_sat_factor_final_third", 1.0),
            r.get("ARM_OFF", {}).get("mean_sat_factor_final_third", 1.0),
        ]
        for r in seed_results
    ]
    on_harm_fracs = [
        r.get("ARM_ON", {}).get("harm_class_fraction", 0.0) for r in seed_results
    ]
    degeneracy = check_degeneracy({
        "sat_factor_arm_contrast": {"groups": sat_contrast_groups},
        "harm_class_fraction_on": {"values": on_harm_fracs, "floor": 0.01},
    })

    manifest = {
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
        "evidence_direction_per_claim": {"MECH-268": direction},
        "thresholds": {
            "C1_max_sat_factor": C1_MAX_SAT_FACTOR,
            "C2_min_sat_factor": C2_MIN_SAT_FACTOR,
            "harm_class_frac_min": HARM_CLASS_FRAC_MIN,
            "pass_fraction_required": PASS_FRACTION_REQUIRED,
            "sat_window": SAT_WINDOW,
            "sat_strength": SAT_STRENGTH,
            "sat_grace": SAT_GRACE,
            "contextual_safety_harm_threshold": CONTEXTUAL_SAFETY_HARM_THRESHOLD,
        },
        "n_seeds": n_seeds,
        "n_seeds_pass": n_pass,
        "smoke": smoke,
        "metrics": {"per_seed": seed_results},
        "notes": (
            "V3-EXQ-729 LIVE-LOOP ecological analogue of V3-EXQ-463b (MECH-268 dACC"
            " conflict saturation). 463b validated f_sat only via manual per-step"
            " record_outcome injection; this run makes NO manual record_outcome call"
            " and proves the DACC _outcome_history FIFO is filled by the LIVE"
            " agent.py select_action tail (F-C3 wiring, ree-v3 main c7fc045) via a"
            " counting spy on record_outcome (live_record_calls). Chronic-harm env"
            " (num_hazards=%d, size=%d) keeps z_harm_a.norm() above"
            " contextual_safety_harm_threshold so the recurring harm class saturates"
            " the FIFO. ARM_ON flips dacc_saturation_enabled=True for eval only"
            " (training identical to ARM_OFF -> bit-identical weights). ARM_OFF is"
            " the control AND the reuse-eligible minted baseline (OFF cell fingerprint"
            " omits the ON saturation knobs). PROMOTES NOTHING -- closes the"
            " 'validated only via synthetic injection' gap for MECH-268; the F-C3"
            " governance follow-up #2 (design_implementation_audit_2026-07-09.md)."
            % (EVAL_HAZARDS, ENV_SIZE)
        ),
    }
    manifest.update(degeneracy)
    return manifest


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
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
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
        dry_run=args.dry_run,
    )
    sys.exit(0)
