"""V3-EXQ-964 -- SD-102 (MECH-482) epistemic_deficit_accumulator substrate
readiness validation: 314c source ablation (broadcast vs epistemic_deficit),
yoked pair.

EXPERIMENT_PURPOSE = "diagnostic" -- this is SUBSTRATE READINESS validation
(does the new persistent-target accumulator populate with non-trivial values
and can it change candidate selection), not governance evidence for MECH-482's
own claim hypothesis. Per /queue-experiment / /implement-substrate step 8: "the
implementation doesn't change these numbers, it hasn't solved the problem" --
this driver measures exactly that, nothing stronger.

WHAT THIS TESTS. Two arms, isolated to the 314c sub-flavour alone (314a
novelty and 314b uncertainty both OFF on both arms, so any yoked divergence is
attributable to the learning-progress source alone, not a confound):
  ARM_OFF (reference, drives the env): curiosity_learning_progress_source=
    "broadcast" -- today's status-quo Phase-1 uniform lp_ema.
  ARM_ON: curiosity_learning_progress_source="epistemic_deficit" -- the SD-102
    accumulator. Stepped on ARM_OFF's identical observation sequence (yoked),
    so every comparison is a paired argmin flip, non-compounding.

Both arms instantiate + train the SD-063 E2WorldUncertaintyHead identically
(use_e2_world_uncertainty=True, online_training=True) -- the head is a shared
DEPENDENCY of the manipulation, not the manipulation itself; training it
identically on both arms isolates the learning_progress_source factor alone
(mirrors the 947/949 drivers' own "head trains on every arm but is CONSUMED
only on the ON arm(s)" discipline).

OWNS ITS OWN RNG STREAM (mandatory idiom this codebase is scarred by getting
wrong -- see the 947 driver's own discovery, cited in the SD-102 doc's ML/AI
engineering notes). Each _Runner snapshots and restores its own generator
state around every tick, episode reset, and residue update; the self-yoked
`paired_control_divergence` instrument control runs for both arm identities
before any scored compute.

READINESS (diagnostic-scope, per SD-102's own binding gate): the ON arm's
per-candidate learning-progress read is expected to be REFUSED (None,
self-reported vacuous) on a meaningful fraction of ticks early in a rollout --
the accumulator has no targets yet, and the gate correctly refuses until
e2_world_uncertainty_last_pvar_relative_spread > 0 is observed. This driver
measures, not assumes, the vacuous-readout rate and requires it below a
generous ceiling (0.5) rather than zero.

PRE-REGISTERED CRITERIA (diagnostic-scope; NOT a claim-evidence PASS/FAIL on
MECH-482's own hypothesis -- see the SD doc's "Validation experiment" note):
  C1 (load-bearing): the accumulator becomes LIVE -- at least one persistent
     target exists and at least one UPDATE fired by the end of the rollout.
  C2 (load-bearing): the downstream consumer (StructuredCuriosity.
     compute_score_bias via the per_candidate_learning_progress slot) can
     actually change the committed action -- yoked divergence (ON vs OFF
     reference) is > 0 on at least one seed.
  C3 (supporting, NOT load-bearing): the self-yoked instrument control is
     bit-identical (== 0) for both arm identities -- confirms the DV itself is
     trustworthy, not a fact about MECH-482.
Combination: overall_pass = C1 AND C2 AND (ARM_ON readiness gate green).

SLEEP: none. No sleep flag is set; no SLEEP DRIVER line required.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EXPERIMENT_PURPOSE = "diagnostic"

EXPERIMENT_TYPE = "v3_exq_964_mech482_epistemic_deficit_validation"
QUEUE_ID = "V3-EXQ-964"
CLAIM_IDS = ["MECH-482"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

SEEDS = [71, 101, 202]
EPISODES = 3
STEPS_PER_EPISODE = 60
WORLD_DIM = 16
SELF_DIM = 32

E2U_WARMUP_STEPS = 60
E2U_BATCH_SIZE = 16

# Set ONLY by --dry-run, so the smoke can train the head inside its tiny tick
# budget and therefore actually demonstrate the accumulator can go live.
_ACTIVE_WARMUP = E2U_WARMUP_STEPS

VACUOUS_READOUT_RATE_CEILING = 0.5

ARM_OFF = "ARM_LP_BROADCAST"
ARM_ON = "ARM_LP_EPISTEMIC_DEFICIT"


def build_config(lp_source: str) -> REEConfig:
    env = CausalGridWorldV2()
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        reafference_action_dim=env.action_dim,
    )
    cfg.use_structured_curiosity = True
    # Isolate the learning-progress (314c) sub-flavour alone -- 314a/314b OFF
    # on both arms so a yoked divergence is attributable to the
    # learning_progress_source factor, not a confound.
    cfg.use_curiosity_novelty = False
    cfg.use_curiosity_uncertainty = False
    cfg.use_curiosity_learning_progress = True
    cfg.curiosity_learning_progress_source = lp_source
    # THE SHARED DEPENDENCY, trained identically on both arms (see module
    # docstring) so it is not itself a confound.
    cfg.latent.use_e2_world_uncertainty = True
    cfg.latent.use_e2_world_uncertainty_online_training = True
    cfg.latent.e2_world_uncertainty_warmup_steps = _ACTIVE_WARMUP
    cfg.latent.e2_world_uncertainty_batch_size = E2U_BATCH_SIZE
    return cfg


def config_slice(lp_source: str) -> Dict[str, Any]:
    """Exactly what this cell's computation reads -- no acceptance thresholds."""
    return {
        "env": "CausalGridWorldV2",
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "episodes": EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "use_structured_curiosity": True,
        "use_curiosity_novelty": False,
        "use_curiosity_uncertainty": False,
        "use_curiosity_learning_progress": True,
        "curiosity_learning_progress_source": lp_source,
        "use_e2_world_uncertainty": True,
        "use_e2_world_uncertainty_online_training": True,
        "e2_world_uncertainty_warmup_steps": _ACTIVE_WARMUP,
        "e2_world_uncertainty_batch_size": E2U_BATCH_SIZE,
    }


class _Runner:
    """One agent, driven tick by tick. OWNS ITS OWN RNG STREAM -- see module
    docstring."""

    def __init__(self, cfg: REEConfig) -> None:
        self.agent = REEAgent(cfg)
        self._rng_state = torch.get_rng_state()
        self.world_dim = cfg.latent.world_dim
        self.n_ticks = 0

    def reset_episode(self) -> None:
        ambient = torch.get_rng_state()
        torch.set_rng_state(self._rng_state)
        try:
            self.agent.reset()
        finally:
            self._rng_state = torch.get_rng_state()
            torch.set_rng_state(ambient)

    def choose(self, obs: Dict[str, Any]) -> int:
        ambient = torch.get_rng_state()
        torch.set_rng_state(self._rng_state)
        try:
            return self._choose_inner(obs)
        finally:
            self._rng_state = torch.get_rng_state()
            torch.set_rng_state(ambient)

    def _choose_inner(self, obs: Dict[str, Any]) -> int:
        agent = self.agent
        latent = agent.sense(obs["body_state"], obs["world_state"])
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent)
            if ticks.get("e1_tick")
            else torch.zeros(1, self.world_dim, device=agent.device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        if candidates:
            self.n_ticks += 1
        agent.update_z_goal(
            benefit_exposure=0.0,
            drive_level=REEAgent.compute_drive_level(obs["body_state"]),
        )
        action = agent.select_action(candidates, ticks)
        return int(action.argmax(dim=-1).item())

    def observe(self, harm: Any) -> None:
        ambient = torch.get_rng_state()
        torch.set_rng_state(self._rng_state)
        try:
            self.agent.update_residue(harm)
        finally:
            self._rng_state = torch.get_rng_state()
            torch.set_rng_state(ambient)

    def accumulator_summary(self) -> Dict[str, Any]:
        acc = getattr(self.agent, "epistemic_deficit", None)
        if acc is None:
            return {}
        state = acc.get_state()
        vacuous_rate = (
            state["n_vacuous_readouts"] / state["n_readouts"]
            if state["n_readouts"] > 0
            else 0.0
        )
        return {**state, "n_candidate_ticks": self.n_ticks, "vacuous_readout_rate": vacuous_rate}


def run_yoked_pair(seed: int, episodes: int, steps: int,
                   zg: ZGoalStreamAccumulator) -> Dict[str, Any]:
    reset_all_rng(seed)
    ref = _Runner(build_config("broadcast"))
    reset_all_rng(seed)
    on = _Runner(build_config("epistemic_deficit"))
    reset_all_rng(seed)

    env = CausalGridWorldV2()
    n_cmp = 0
    n_diff = 0
    per_episode: List[Dict[str, Any]] = []
    for ep in range(episodes):
        _, obs = env.reset()
        ref.reset_episode()
        on.reset_episode()
        ep_cmp = ep_diff = 0
        for _ in range(steps):
            a_ref = ref.choose(obs)
            a_on = on.choose(obs)
            n_cmp += 1
            ep_cmp += 1
            if a_on != a_ref:
                n_diff += 1
                ep_diff += 1
            _f, harm, _d, _i, obs = env.step(a_ref)
            ref.observe(harm)
            on.observe(harm)
        per_episode.append({
            "episode": ep, "n_compared": ep_cmp, "n_diverged": ep_diff,
            "divergence_frac": ep_diff / ep_cmp if ep_cmp else 0.0,
        })
        print(f"  [train] yoked seed={seed} ep {ep + 1}/{episodes} "
              f"diverged={ep_diff}/{ep_cmp}", flush=True)
    zg.observe(ref.agent)
    zg.observe(on.agent)

    return {
        "ref_n_candidate_ticks": ref.n_ticks,
        ARM_ON: {
            "accumulator": on.accumulator_summary(),
            "yoked_n_compared": n_cmp,
            "yoked_n_diverged": n_diff,
            "yoked_divergence_frac": n_diff / n_cmp if n_cmp else 0.0,
            "yoked_per_episode": per_episode,
        },
    }


def paired_control_divergence(seed: int, lp_source: str, episodes: int = 1,
                             steps: int = 15) -> float:
    """INSTRUMENT CONTROL: yoke an arm against ITSELF. MUST be 0.0."""
    reset_all_rng(seed)
    a = _Runner(build_config(lp_source))
    reset_all_rng(seed)
    b = _Runner(build_config(lp_source))
    reset_all_rng(seed)
    env = CausalGridWorldV2()
    n = d = 0
    for _ in range(episodes):
        _, obs = env.reset()
        a.reset_episode()
        b.reset_episode()
        for _ in range(steps):
            aa = a.choose(obs)
            bb = b.choose(obs)
            n += 1
            if aa != bb:
                d += 1
            _f, harm, _dn, _i, obs = env.step(aa)
            a.observe(harm)
            b.observe(harm)
    return (d / n) if n else 0.0


PRECONDITIONS = [
    PreconditionSpec(
        name="paired_control_is_bit_identical",
        description=(
            "INSTRUMENT CONTROL: an arm yoked against ITSELF diverges on 0 "
            "ticks. Non-zero means the two yoked agents differ in something "
            "other than the manipulation and the whole DV is void"
        ),
        control="same arm vs same arm, identical seed and config",
        threshold=1e-9,
        direction="upper",
        kind="readiness",
    ),
    PreconditionSpec(
        name="vacuous_readout_rate_bounded",
        description=(
            "the readiness-gate refusal rate (self-reported vacuous READOUTs, "
            "e.g. before any target exists yet or a tick has < 2 candidates) "
            "stays below a generous ceiling across the rollout -- a near-100% "
            "rate would mean the accumulator never got a chance to matter"
        ),
        control="live rollout with the SD-063 head trained identically to the reference",
        threshold=VACUOUS_READOUT_RATE_CEILING,
        direction="upper",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["lp_on"]),
        applies_note="the reference arm never instantiates the accumulator at all",
    ),
    PreconditionSpec(
        name="accumulator_becomes_live",
        description=(
            "at least one persistent target exists and at least one UPDATE "
            "fired by the end of the rollout -- the substrate is genuinely "
            "accumulating, not sitting empty"
        ),
        control="live rollout, epistemic_deficit source enabled",
        threshold=0.5,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["lp_on"]),
        applies_note="the reference arm never instantiates the accumulator at all",
    ),
]


def _arm_ctx(lp_on: bool) -> Dict[str, Any]:
    return {"arm_id": (ARM_ON if lp_on else ARM_OFF), "lp_on": lp_on}


def run_experiment(episodes: int, steps: int, seeds: List[int],
                   dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    zg = ZGoalStreamAccumulator()

    all_ctxs = [_arm_ctx(False), _arm_ctx(True)]
    assert_no_structurally_unsatisfiable_gate(PRECONDITIONS, all_ctxs)

    control_div: Dict[str, float] = {}
    for lp_on, aid in ((False, ARM_OFF), (True, ARM_ON)):
        control_div[aid] = paired_control_divergence(
            seeds[0], "epistemic_deficit" if lp_on else "broadcast",
            episodes=1, steps=(6 if dry_run else 15))
        print(f"[control] {aid}: self-yoked divergence = "
              f"{control_div[aid]:.6f} (must be 0)", flush=True)

    on_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"Seed {seed} Condition yoked_pair", flush=True)
        slice_on = config_slice("epistemic_deficit")
        with arm_cell(seed, config_slice=slice_on, script_path=Path(__file__),
                      config_slice_declared=True,
                      include_driver_script_in_hash=False) as cell:
            pair = run_yoked_pair(seed, episodes, steps, zg)
            row = {
                "arm_id": ARM_ON, "seed": seed,
                "ref_n_candidate_ticks": pair["ref_n_candidate_ticks"],
                **pair[ARM_ON],
            }
            cell.stamp(row)
        on_rows.append(row)
        print(f"verdict: {'PASS' if row['ref_n_candidate_ticks'] > 0 else 'FAIL'}",
              flush=True)

    # ---- per-arm readiness gates -------------------------------------------
    arm_gates = []
    for lp_on, aid in ((False, ARM_OFF), (True, ARM_ON)):
        ctx = _arm_ctx(lp_on)
        if not lp_on:
            measured = {"paired_control_is_bit_identical": control_div.get(aid, 1.0)}
        else:
            n_targets_min = min(r["accumulator"].get("n_targets", 0) for r in on_rows)
            n_updates_min = min(r["accumulator"].get("n_updates", 0) for r in on_rows)
            measured = {
                "paired_control_is_bit_identical": control_div.get(aid, 1.0),
                "vacuous_readout_rate_bounded": max(
                    r["accumulator"].get("vacuous_readout_rate", 1.0) for r in on_rows
                ),
                "accumulator_becomes_live": float(
                    min(n_targets_min, n_updates_min) > 0
                ),
            }
        arm_gates.append(evaluate_arm_gate(aid, ctx, PRECONDITIONS, measured))
    aggregate = aggregate_arm_gates(arm_gates)

    # ---- pre-registered criteria (diagnostic-scope) ------------------------
    n_targets_min = min(r["accumulator"].get("n_targets", 0) for r in on_rows)
    n_updates_min = min(r["accumulator"].get("n_updates", 0) for r in on_rows)
    max_div = max(r["yoked_divergence_frac"] for r in on_rows)
    mean_div = sum(r["yoked_divergence_frac"] for r in on_rows) / len(on_rows)

    c1 = bool(n_targets_min > 0 and n_updates_min > 0)
    c2 = bool(max_div > 0.0)
    c3 = bool(max(control_div.values()) < 1e-9) if control_div else False

    green = set(aggregate["green_arms"])
    overall_pass = bool(c1 and c2 and (ARM_ON in green))

    criteria = [
        {"name": "C1_accumulator_becomes_live", "load_bearing": True,
         "passed": c1, "measured": float(min(n_targets_min, n_updates_min)),
         "threshold": 0.5},
        {"name": "C2_downstream_consumer_can_diverge", "load_bearing": True,
         "passed": c2, "measured": max_div, "threshold": 0.0},
        {"name": "C3_instrument_control_bit_identical", "load_bearing": False,
         "passed": c3, "measured": max(control_div.values()) if control_div else 1.0,
         "threshold": 1e-9},
    ]
    combination_rule = (
        "overall_pass = C1 AND C2 AND (ARM_LP_EPISTEMIC_DEFICIT readiness gate "
        "green). This is a DIAGNOSTIC substrate-readiness check, NOT a "
        "governance-evidence verdict on MECH-482's own claim hypothesis (that "
        "requires a separate, later evidence-purpose experiment per the SD-102 "
        "doc). C3 is an instrument-quality control and does not gate the "
        "outcome."
    )

    if not overall_pass and ARM_ON not in green:
        direction = "unknown"
        label = "substrate_not_ready_requeue"
    elif overall_pass:
        direction = "supports"
        label = "epistemic_deficit_accumulator_live_and_selection_relevant"
    elif c1 and not c2:
        direction = "mixed"
        label = "accumulator_live_but_never_changes_committed_action"
    else:
        direction = "weakens"
        label = "accumulator_did_not_become_live"

    criteria_nd = arm_criteria_non_degenerate(
        {
            ARM_ON: [
                "C1_accumulator_becomes_live",
                "C2_downstream_consumer_can_diverge",
            ],
        },
        aggregate,
        extra={},
    )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": "PASS" if overall_pass else "FAIL",
        "timestamp_utc": ts,
        "evidence_direction": direction,
        "dry_run": dry_run,
        "metrics": {
            "n_targets_min": n_targets_min,
            "n_updates_min": n_updates_min,
            "yoked_divergence_frac_max": max_div,
            "yoked_divergence_frac_mean": mean_div,
            "vacuous_readout_rate_max": max(
                r["accumulator"].get("vacuous_readout_rate", 1.0) for r in on_rows
            ),
            "paired_control_divergence_max": max(control_div.values())
            if control_div else 1.0,
        },
        "criteria": criteria,
        "combination_rule": combination_rule,
        "arm_results": on_rows,
        "per_arm_gate": aggregate["per_arm_gate"],
        "non_degenerate": aggregate["non_degenerate"],
        "degeneracy_reason": aggregate["degeneracy_reason"],
        "interpretation": {
            "label": label,
            "preconditions": aggregate["adjudication_preconditions"],
            "preconditions_scope_note": aggregate.get(
                "preconditions_scope_note", ""),
            "criteria_non_degenerate": criteria_nd,
        },
        "custom_information": {
            "diagnostic_scope_note": (
                "EXPERIMENT_PURPOSE=diagnostic: this validates the SUBSTRATE "
                "(does the accumulator populate, can it move the argmin), not "
                "MECH-482's own claim hypothesis (importance x uncertainty x "
                "resolvability x persistence; satiation on resolution). See "
                "REE_assembly/docs/architecture/sd_102_epistemic_deficit_"
                "accumulator.md 'Validation experiment'."
            ),
            "isolation_note": (
                "314a novelty and 314b uncertainty are OFF on both arms so "
                "the yoked divergence is attributable to the "
                "learning_progress_source factor (broadcast vs "
                "epistemic_deficit) alone, not a confound."
            ),
        },
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
    }

    full_config = {
        "seeds": seeds,
        "episodes": episodes,
        "steps_per_episode": steps,
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "e2_world_uncertainty_warmup_steps": E2U_WARMUP_STEPS,
        "e2_world_uncertainty_batch_size": E2U_BATCH_SIZE,
        "thresholds": {
            "VACUOUS_READOUT_RATE_CEILING": VACUOUS_READOUT_RATE_CEILING,
        },
    }

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=zg.stats(),
    )
    return {"outcome": manifest["outcome"], "manifest": manifest,
            "out_path": out_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--steps", type=int, default=STEPS_PER_EPISODE)
    args = ap.parse_args()

    global _ACTIVE_WARMUP
    if args.dry_run:
        episodes, steps, seeds = 2, 30, [71]
        _ACTIVE_WARMUP = 12
    else:
        episodes, steps, seeds = args.episodes, args.steps, SEEDS

    result = run_experiment(episodes, steps, seeds, args.dry_run)
    out_path = result["out_path"]
    print(f"manifest: {out_path}", flush=True)
    print(json.dumps(result["manifest"]["metrics"], indent=2), flush=True)

    if args.dry_run:
        m = result["manifest"]["metrics"]
        checks = {
            "accumulator becomes live (n_targets_min>0)": m["n_targets_min"] > 0,
            "accumulator updates fire (n_updates_min>0)": m["n_updates_min"] > 0,
            "INSTRUMENT CONTROL: self-yoked arms are bit-identical (==0)":
                m["paired_control_divergence_max"] == 0.0,
        }
        print("[smoke] substrate-readiness checks:", flush=True)
        for label, ok in checks.items():
            print(f"  [{'OK' if ok else 'XX'}] {label}", flush=True)

    raw = str(result["outcome"]).upper()
    return (raw if raw in ("PASS", "FAIL") else "FAIL"), out_path, args.dry_run


if __name__ == "__main__":
    _outcome_raw, _out_path, _dry = main()
    emit_outcome(
        outcome=_outcome_raw,
        manifest_path=_out_path,
        dry_run=_dry,
    )
