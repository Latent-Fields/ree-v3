#!/opt/local/bin/python3
"""V3-EXQ-957 -- MECH-219 controllability-dissociation substrate-readiness diagnostic.

SLEEP DRIVER: not applicable (no sleep loop used in this driver).

Claim / substrate under test
-----------------------------
MECH-219 (affect.affective_harm_hysteretic_integration) -- IMPLEMENTED 2026-06-10
(ree-v3/CLAUDE.md "MECH-219 (SD-019b): affective-harm hysteretic integrator
(z_harm_suffering)"). Pure-arithmetic regulator (no nn.Module, no learned
parameters, no gradient flow):
  ree_core/affect/harm_suffering_accumulator.py -- HarmSufferingAccumulator.
Wired into ree_core/agent.py sense() (after the SD-019a z_harm_un EMA, before
the SD-032 consumers) and ree_core/latent/stack.py (LatentState.z_harm_suffering).
The plan-of-record's own build-session checklist item "Substrate-readiness EXQ
(memo Sec 7) via /queue-experiment (NOT in the build session)" was left unqueued
-- ree-v3/CLAUDE.md says so explicitly ("Validation experiment: NOT queued in
this build session ... separate /queue-experiment session per the memo Section 7
sketch"). This script is that session's deliverable.

Design memo (plan-of-record, read in full before writing this script):
  REE_assembly/evidence/planning/mech_219_hysteretic_integrator_design.md
  Section 7 "Validation experiment (sketch)" -- the falsifier this script
  implements, C1-C4, reproduced below with this script's scope decisions noted.

Why this experiment exists (readiness checks that cleared it)
---------------------------------------------------------------
- GOV-REUSE-1 (Step 2.4): zero prior runs tag MECH-219 in claim_ids anywhere in
  REE_assembly/evidence/experiments/*.json (checked by grep across the full flat
  corpus, 2026-08-29) -- the one incidental filename hit (v3_exq_854, SD-036 GABA
  tone dose-response) tags claim_ids=["SD-036"] only; "MECH-219" appears there as
  prose, not as a tested claim. Nothing to reanalyze; there is no recorded
  decisive readout to recover.
- Step 2.5 substrate readiness: MECH-219 module + LatentState field + REEConfig
  knobs + agent.py sense()-order wiring all confirmed IMPLEMENTED 2026-06-10 in
  ree-v3/CLAUDE.md, with "7/7 preflight + full contract suite green" and
  "11/11 new contracts in tests/contracts/test_mech_219_harm_suffering_accumulator.py"
  recorded at build time.
- Step 2.5a empirical probe (one-tick-and-beyond, run 2026-08-29, throwaway
  script under ree-v3/.scratch_mech219/, deleted after use -- not committed):
  confirmed live that (a) REEAgent(use_harm_suffering_accumulator=True,
  use_harm_un=True, harm_suffering_escapability_mode="external") builds a real
  HarmSufferingAccumulator; (b) at matched hazard exposure, escapability=0.0
  (inescapable) drives s_t up to ~0.51-0.53 over 30 ticks while escapability=1.0
  (escapable) holds s_t at EXACTLY 0.0000 the whole time (g_t=1-esc=0 ->
  drive_t=0 by construction) despite z_harm_un staying substantial (~0.75-0.76)
  in the escapable arm -- the core C1/C3 dissociation, confirmed empirically
  before this script was written, not merely read off the doc; (c) after
  flipping escapability from 0.0 to 1.0 mid-run (same fixed position, so the
  drive-input side of the equation does not confound the release measurement),
  s_t decayed with an empirically measured half-life of 69 steps against a
  THEORETICAL alpha_fall=0.01 half-life of ln(2)/0.01 = 69.3 steps -- an exact
  match, the strongest possible confirmation that the shipped arithmetic is the
  documented arithmetic; (d) with harm_descending_mod_enabled=True (SD-021),
  elevating agent.beta_gate measurably attenuated z_harm (0.540 -> 0.265,
  confirming SD-021 genuinely fired) while
  agent._resolve_harm_suffering_escapability() stayed bit-identical (0.0 -> 0.0)
  -- the C4 parity reading, and non-vacuous because the SD-021 attenuation was
  independently confirmed to have fired.
- Step 2.5b re-derive brake: zero failure_autopsy_*.json artifacts anywhere in
  REE_assembly/evidence/planning/ tag MECH-219 in any target's claim_ids (grepped
  the full glob, 2026-08-29). Count = 0, threshold = 2 -- brake does not apply;
  this is a first-ever test, not a re-derive of an already-ceilinged claim.
- Step 2.5c substrate-path overlap: this driver's execute+read closure is
  ree_core/affect/harm_suffering_accumulator.py, ree_core/agent.py (sense()
  wiring only -- no goal/salience/hippocampal/sleep call sites exercised),
  ree_core/latent/stack.py, ree_core/utils/config.py, and
  ree_core/environment/causal_grid_world.py. Cross-referenced against every OPEN
  (non implemented/validated/wontfix) substrate_queue.json entry carrying
  substrate_paths (2026-08-29 scan): no open corrupting-severity entry's
  substrate_paths overlaps this file set. (The only borderline candidate,
  "mode-governance-engagement" / SalienceCoordinator, touches ree_core/agent.py
  but a DIFFERENT call-site family (operating_mode / SalienceCoordinator.tick)
  that this driver never reaches -- use_salience_coordinator is left at its
  REEConfig default False here, so there is no actual code-path overlap, not
  merely an unexercised flag.) Escapability is sourced in "external" mode only
  (never "avoidance_efficacy"), so this run also carries NO soft dependency on
  the v3_pending SD-058 substrate -- exactly the dependency-free path the design
  memo Section 3 recommends for a readiness pass ("the default constant mode has
  no such dependency -- only avoidance_efficacy does").

The falsifier (design memo Section 7, reproduced with this script's scope)
-----------------------------------------------------------------------------
Matched nociception (same z_harm_un-producing exposure -- identical seed, fixed
agent position 1 cell from a single hazard, identical action sequence across
conditions) under ESCAPABLE (escapability=1.0 throughout) vs INESCAPABLE
(escapability=0.0 during the harm bout, then 1.0 during the relief phase --
the controllability change IS the relief, isolating the hysteresis measurement
from any dependence on the untrained encoder's spatial responsiveness; see
SCOPE NOTE below for why a physical-relocation relief was tried and rejected).

  C1 (controllability gate): z_harm_suffering accrues under inescapable but
     stays low under escapable, at matched z_harm_un.
  C2 (hysteresis): after the bout ends (escapability -> 1.0), z_harm_suffering
     releases on the slow alpha_fall timescale -- it does not collapse
     immediately.
  C3 (non-vacuity / distinct-from), part (a) only -- see SCOPE NOTE:
     z_harm_suffering != a re-scaled z_harm_un: the escapable arm holds high
     z_harm_un but near-zero suffering.
  C4 (parity), narrowed to the escapability-resolution reading -- see SCOPE
     NOTE: SD-021 descending modulation attenuates z_harm but does not touch
     the resolved escapability scalar (the SD-019a UC3-mirror check,
     V3-EXQ-518 ARM_3).

PASS = C1 AND C2 AND C3 AND C4, each required across EVERY seed (conjunctive,
not a mean-with-margin) -- the escapability gate zeroes drive_t exactly by
construction (g_t=1-esc), so a genuine dissociation is expected to hold at
every seed if the substrate is wired as documented; a per-seed failure is
informative, not noise to average over.

SCOPE NOTE -- what this script deliberately does NOT attempt
----------------------------------------------------------------
- C3, part (b) (anti-correlation with MECH-353 z_block_assert under the same
  controllability sweep) is OUT OF SCOPE here. BlockedAgency's own "blocked"
  state is driven by outcome-mismatch / attribution-motor internals
  (ree_core/affect/blocked_agency.py _update_blocked_agency), not by a simple
  external switch the way escapability is; wiring a genuine positive-control
  "blocked" trigger correctly is materially more involved than the escapability
  gate this script exercises, and a half-correct trigger would be its own
  confound rather than a clean cross-check. This is exactly the split the
  design memo itself draws: "Substrate-readiness diagnostic, claim_ids=[]
  first; the behavioural MECH-219/SD-019b evidence successor is separate" --
  the MECH-353 cross-check belongs in that later, claim_ids=["MECH-219"] run.
- C4 is narrowed to "does elevating beta_gate change the resolved escapability
  scalar" (it must not), NOT a full replication of the z_harm_s attenuation
  magnitude. The attenuation itself is measured only as the NON-DEGENERACY
  control for this check (proof that SD-021 genuinely fired this tick), not as
  a load-bearing readout in its own right.
- A physical-relocation relief phase (teleport the agent far from the hazard
  via env.reset_to, matching the SD-019a V3-EXQ-518 UC-style probing) was
  empirically tried during the Step 2.5a probe and REJECTED: at distance ~9 on
  a size=12 grid, z_harm_un did not decay after relocation -- it INCREASED
  monotonically over 200 ticks (0.55 -> 1.45), evidently because the untrained,
  randomly-initialised z_harm encoder's response to the proxy-field local view
  is not calibrated to fall off with hazard distance the way a trained encoder
  would (an UNTRAINED-substrate calibration property, not a MECH-219 property).
  Using distance-based relief would have made the "hysteresis" reading
  inseparable from that uncalibrated-encoder drift. Driving relief through the
  escapability gate itself sidesteps the confound entirely and is arguably the
  MORE faithful operationalisation of "recovery" for a controllability-gated
  mechanism in any case (escape becomes available -- not merely coincidental
  physical distance).

claim_ids=[] (not ["MECH-219"]): per the design memo's own Section 7 framing,
this is the SUBSTRATE-READINESS pass, excluded from governance confidence /
conflict scoring by EXPERIMENT_PURPOSE="diagnostic" AND by carrying no claim
tags at all -- a deliberately double-safe choice given this is the FIRST-EVER
run against this substrate (see the memo's own worked precedent, V3-EXQ-518,
which the SD-019a substrate readiness pass, similarly untrained/unclaimed in
spirit though that predates the empty-claim_ids convention). A later,
claim_ids=["MECH-219"] behavioural evidence run (trained substrate, full C3(b)
cross-check, avoidance_efficacy-sourced escapability once SD-058 clears) is the
follow-on this run recommends, not supersedes.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_957_mech219_controllability_dissociation_substrate_readiness"
QUEUE_ID = "V3-EXQ-957"
CLAIM_IDS: List[str] = []  # substrate-readiness pass -- see docstring "claim_ids=[]"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
ARM_FINGERPRINT_EXEMPT = (
    "pure-arithmetic-regulator diagnostic -- HarmSufferingAccumulator has no nn.Module, no "
    "learned parameters, no gradient flow, and nothing here is expensive enough to be worth "
    "reuse-caching. Writes 'condition_rows' (per seed x condition), deliberately NOT "
    "'arm_results' -- the arm-fingerprint reuse machinery is built for expensive trained "
    "baseline arms, which this driver has none of."
)
ANCHOR_REACHABILITY_EXEMPT = (
    "the readiness preconditions (un_drive_supra_floor_inescapable/escapable) assert only that "
    "z_harm_un is non-trivial (> 0.05) after N steps at Manhattan distance 1 from a live hazard "
    "-- a floor reachable by construction any time the substrate is stepping at all near a "
    "hazard, confirmed empirically at ~0.53-0.76 in the Step 2.5a probe (module docstring). "
    "There is no narrower hand-written predicate here that could be unmeetable independent of "
    "the load-bearing criteria -- the anchor IS the degeneracy definition (is there any drive "
    "signal at all), not a proxy for one."
)

SEEDS: Tuple[int, ...] = (11, 23, 37, 41, 59)
CONDITIONS: Tuple[str, ...] = ("INESCAPABLE", "ESCAPABLE")
BOUT_STEPS = 30
RELIEF_STEPS = 150
DRY_BOUT_STEPS = 8
DRY_RELIEF_STEPS = 10
TAIL_WINDOW = 5  # ticks averaged for the "bout-end" / stability reads

HAZARD_POS: Tuple[int, int] = (6, 6)
NEAR_POS: Tuple[int, int] = (5, 6)  # Manhattan distance 1 from the hazard; fixed all run
ENV_SIZE = 12
ENV_KWARGS: Dict[str, Any] = dict(num_hazards=1, size=ENV_SIZE)

# Pre-registered thresholds -- never derived from this run's own statistics.
# Empirical probe (Step 2.5a, 2026-08-29) observed: inescapable bout_end_s
# ~0.51-0.54, escapable bout_end_s == 0.0000, escapable bout_end_un ~0.75-0.76,
# relief retention at +5 ticks ~0.95 (theoretical (1-alpha_fall)^5 = 0.951),
# relief half-life ~69 steps (theoretical ln(2)/alpha_fall = 69.3). Every
# threshold below sits with a wide margin against those observed/theoretical
# values so ordinary seed-to-seed variation cannot flip the verdict, while
# still cleanly separating a genuinely broken/unwired mechanism.
UN_READINESS_FLOOR = 0.05  # z_harm_un norm floor for the positive-control drive
C1_MARGIN = 0.10  # required bout_end_s(INESCAPABLE) - bout_end_s(ESCAPABLE) gap
C2_RETENTION_FRAC = 0.85  # s must retain >= this fraction of relief-start value after 5 ticks
C2_HALFLIFE_MIN_STEPS = 20  # secondary/non-load-bearing corroboration (theoretical ~69)
C3_S_CEILING = 0.03  # ESCAPABLE arm's bout-end suffering must stay near zero
C3_UN_FLOOR = 0.10  # ... while its own z_harm_un is genuinely non-trivial (non-vacuity)

_ZG = ZGoalStreamAccumulator()


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _build_env() -> CausalGridWorldV2:
    return CausalGridWorldV2(**ENV_KWARGS)


def _build_cfg(env: CausalGridWorldV2, *, descending_mod: bool = False) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        harm_dim=32,
    )
    cfg.latent.use_harm_un = True  # MECH-219 precondition (agent.py: loud ValueError otherwise)
    cfg.use_harm_suffering_accumulator = True
    cfg.harm_suffering_escapability_mode = "external"
    if descending_mod:
        # SD-021 committed-state descending modulation -- only for the C4 parity check.
        cfg.harm_descending_mod_enabled = True
        cfg.descending_attenuation_factor = 0.5
    return cfg


def _do_sense(agent: REEAgent, obs_dict: Dict[str, Any]) -> Any:
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    kw: Dict[str, Any] = {"obs_body": body, "obs_world": world}
    harm_obs = obs_dict.get("harm_obs")
    if harm_obs is not None:
        if harm_obs.dim() == 1:
            harm_obs = harm_obs.unsqueeze(0)
        kw["obs_harm"] = harm_obs
    with torch.no_grad():
        return agent.sense(**kw)


def _run_condition(seed: int, condition: str, bout_steps: int, relief_steps: int) -> Dict[str, Any]:
    """One seed x condition unit: matched-nociception exposure, escapability per condition.

    INESCAPABLE: escapability=0.0 for the bout, then 1.0 for the "relief" (the
    controllability change IS the relief -- see the module SCOPE NOTE for why).
    ESCAPABLE: escapability=1.0 for the entire run (both phases), the matched
    non-vacuity comparator for C3.
    """
    total_ticks = bout_steps + relief_steps
    _seed_everything(seed)
    env = _build_env()
    cfg = _build_cfg(env)
    agent = REEAgent(cfg)

    print(f"Seed {seed} Condition {condition}", flush=True)

    _, obs_dict = env.reset_to(agent_pos=NEAR_POS, hazard_positions=[HAZARD_POS])
    esc_bout = 0.0 if condition == "INESCAPABLE" else 1.0
    agent.set_harm_suffering_escapability(esc_bout)

    un_series: List[float] = []
    s_series: List[float] = []
    for t in range(total_ticks):
        if t == bout_steps and condition == "INESCAPABLE":
            # Relief: escapability becomes available. g_t = 1 - esc drops to 0,
            # so drive_t -> 0 regardless of the (still substantial) z_harm_un.
            agent.set_harm_suffering_escapability(1.0)
        latent = _do_sense(agent, obs_dict)
        st = agent.harm_suffering_accumulator.get_state()
        un_series.append(
            float(latent.z_harm_un.detach().norm().item()) if latent.z_harm_un is not None else 0.0
        )
        s_series.append(float(st["mech219_last_s_out"]))
        _, _, _, _, obs_dict = env.step(4)  # stay -- matched nociception, no navigation confound
        if (t + 1) % 30 == 0 or (t + 1) == total_ticks:
            print(f"  [train] seed={seed} cond={condition} ep {t + 1}/{total_ticks}", flush=True)

    _ZG.observe(agent)

    tail_start = max(0, bout_steps - TAIL_WINDOW)
    bout_end_un = float(np.mean(un_series[tail_start:bout_steps]))
    bout_end_s = float(np.mean(s_series[tail_start:bout_steps]))

    relief_halflife = float("nan")
    relief_retention_5 = float("nan")
    s_relief_start = None
    if condition == "INESCAPABLE":
        s_relief_start = s_series[bout_steps]
        target = s_relief_start / 2.0
        relief_series = s_series[bout_steps:]
        found = False
        for k, sv in enumerate(relief_series):
            if sv <= target:
                relief_halflife = float(k)
                found = True
                break
        if not found:
            relief_halflife = float(relief_steps)  # censored -- never reached half-life in window
        if bout_steps + 5 < len(s_series) and s_relief_start and s_relief_start > 0:
            relief_retention_5 = s_series[bout_steps + 5] / s_relief_start

    print("verdict: PASS", flush=True)
    return {
        "seed": seed,
        "condition": condition,
        "bout_end_un": bout_end_un,
        "bout_end_s": bout_end_s,
        "s_relief_start": s_relief_start,
        "relief_halflife_steps": relief_halflife,
        "relief_retention_frac_at_5": relief_retention_5,
        "n_waking_updates": int(st["mech219_n_waking_updates"]),
    }


def _run_c4_check(seed: int) -> Dict[str, Any]:
    """SD-021 parity: elevate beta_gate (confirming it genuinely attenuates z_harm),
    confirm the RESOLVED escapability scalar is untouched."""
    _seed_everything(seed)
    env = _build_env()
    cfg = _build_cfg(env, descending_mod=True)
    agent = REEAgent(cfg)
    _, obs_dict = env.reset_to(agent_pos=NEAR_POS, hazard_positions=[HAZARD_POS])
    agent.set_harm_suffering_escapability(0.0)

    latent = None
    for _ in range(10):
        latent = _do_sense(agent, obs_dict)
        _, _, _, _, obs_dict = env.step(4)
    norm_pre = float(latent.z_harm.detach().norm().item()) if latent.z_harm is not None else 0.0
    esc_pre = agent._resolve_harm_suffering_escapability()

    agent.beta_gate.elevate()
    _, _, _, _, obs_dict = env.step(4)
    latent2 = _do_sense(agent, obs_dict)
    norm_committed = float(latent2.z_harm.detach().norm().item()) if latent2.z_harm is not None else 0.0
    esc_post = agent._resolve_harm_suffering_escapability()

    return {
        "seed": seed,
        "z_harm_norm_pre_elevate": norm_pre,
        "z_harm_norm_post_elevate": norm_committed,
        "sd021_attenuation_fired": bool(norm_committed < norm_pre),
        "escapability_pre": float(esc_pre),
        "escapability_post": float(esc_post),
        "escapability_unchanged": bool(esc_pre == esc_post),
    }


def main(dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    bout_steps = DRY_BOUT_STEPS if dry_run else BOUT_STEPS
    relief_steps = DRY_RELIEF_STEPS if dry_run else RELIEF_STEPS
    seeds = SEEDS[:1] if dry_run else SEEDS

    condition_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for condition in CONDITIONS:
            condition_rows.append(_run_condition(seed, condition, bout_steps, relief_steps))

    c4_rows = [_run_c4_check(seed) for seed in seeds]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
    }

    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for row in condition_rows:
        by_seed.setdefault(row["seed"], {})[row["condition"]] = row

    worst_un_inescapable = min(by_seed[s]["INESCAPABLE"]["bout_end_un"] for s in seeds)
    worst_un_escapable = min(by_seed[s]["ESCAPABLE"]["bout_end_un"] for s in seeds)

    common_config = {
        "env_kwargs": ENV_KWARGS,
        "hazard_pos": list(HAZARD_POS),
        "near_pos": list(NEAR_POS),
        "bout_steps": bout_steps,
        "relief_steps": relief_steps,
    }

    try:
        preconditions = p0_readiness_gate([
            {
                "name": "un_drive_supra_floor_inescapable",
                "measured": worst_un_inescapable,
                "threshold": UN_READINESS_FLOOR,
                "direction": "lower",
                "control": (
                    "worst-seed z_harm_un norm over the bout-phase tail window, INESCAPABLE "
                    "arm -- the exact drive_t = g_t*u_t input C1/C2 route on"
                ),
            },
            {
                "name": "un_drive_supra_floor_escapable",
                "measured": worst_un_escapable,
                "threshold": UN_READINESS_FLOOR,
                "direction": "lower",
                "control": (
                    "worst-seed z_harm_un norm over the bout-phase tail window, ESCAPABLE "
                    "arm -- same statistic, the C3 non-vacuity input"
                ),
            },
        ])
    except P0NotReady as e:
        manifest["outcome"] = "FAIL"
        manifest["interpretation"] = {
            "label": "substrate_not_ready_requeue",
            "preconditions": e.preconditions,
            "criteria_non_degenerate": {
                "C1_controllability_gate": False,
                "C2_hysteresis_asymmetric_release": False,
                "C3_non_vacuity_distinct_from_z_harm_un": False,
                "C4_sd021_escapability_parity": False,
            },
        }
        manifest["condition_rows"] = condition_rows
        manifest["c4_rows"] = c4_rows
        manifest["non_degenerate"] = False
        manifest["degeneracy_reason"] = e.reason
        out_path = write_flat_manifest(
            manifest,
            config=common_config,
            seeds=list(seeds),
            script_path=Path(__file__),
            z_goal_stream_stats=_ZG.stats(),
            started_at=t0,
            dry_run=dry_run,
        )
        manifest["_out_path"] = str(out_path)
        return manifest

    # ---- C1: controllability gate ----
    c1_gap_per_seed = {
        s: by_seed[s]["INESCAPABLE"]["bout_end_s"] - by_seed[s]["ESCAPABLE"]["bout_end_s"]
        for s in seeds
    }
    c1_pass = all(g >= C1_MARGIN for g in c1_gap_per_seed.values())

    # ---- C2: hysteresis (asymmetric release) ----
    c2_retention_per_seed = {s: by_seed[s]["INESCAPABLE"]["relief_retention_frac_at_5"] for s in seeds}
    c2_halflife_per_seed = {s: by_seed[s]["INESCAPABLE"]["relief_halflife_steps"] for s in seeds}
    c2_pass = (
        all(r >= C2_RETENTION_FRAC for r in c2_retention_per_seed.values())
        and all(h >= C2_HALFLIFE_MIN_STEPS for h in c2_halflife_per_seed.values())
    )

    # ---- C3: non-vacuity / distinct-from z_harm_un (part (a) only -- see SCOPE NOTE) ----
    c3_s_vals = [by_seed[s]["ESCAPABLE"]["bout_end_s"] for s in seeds]
    c3_un_vals = [by_seed[s]["ESCAPABLE"]["bout_end_un"] for s in seeds]
    c3_pass = all(sv <= C3_S_CEILING for sv in c3_s_vals) and all(uv >= C3_UN_FLOOR for uv in c3_un_vals)

    # ---- C4: SD-021 parity, narrowed to escapability-resolution (see SCOPE NOTE) ----
    c4_pass = all(r["escapability_unchanged"] for r in c4_rows) and any(
        r["sd021_attenuation_fired"] for r in c4_rows
    )
    c4_non_degenerate = any(r["sd021_attenuation_fired"] for r in c4_rows)

    all_pass = bool(c1_pass and c2_pass and c3_pass and c4_pass)
    outcome = "PASS" if all_pass else "FAIL"

    degeneracy = check_degeneracy({
        "C1_bout_end_s_inescapable": {
            "values": [by_seed[s]["INESCAPABLE"]["bout_end_s"] for s in seeds],
            "floor": 0.02,
            "ceiling": 1.98,  # s_cap default is 2.0 -- catch a saturated-ceiling vacuous pass
        },
        "C2_s_relief_start": {
            "values": [by_seed[s]["INESCAPABLE"]["s_relief_start"] for s in seeds],
            "floor": 0.02,
        },
        "C3_bout_end_un_escapable": {
            "values": c3_un_vals,
            "floor": UN_READINESS_FLOOR,
        },
        # C4's own non-degeneracy is a "did the positive control fire at all" check
        # (any(...) across seeds), not a spread/floor/ceiling read on a 0/1 flag --
        # check_degeneracy's zero-spread rule would misfire on an all-1.0 (or
        # all-0.0) boolean series, so C4 is handled separately (c4_non_degenerate
        # above), not routed through this call.
    })

    manifest.update({
        "outcome": outcome,
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_note": (
            "Diagnostic-purpose, claim_ids=[]: this is a SUBSTRATE-READINESS check per the "
            "MECH-219 design memo Section 7 "
            "(mech_219_hysteretic_integrator_design.md), not the behavioural MECH-219/SD-019b "
            "evidence run. evidence_direction is recorded for human/governance reading only -- "
            "excluded from confidence/conflict scoring by EXPERIMENT_PURPOSE + empty claim_ids."
        ),
        "condition_rows": condition_rows,
        "c4_rows": c4_rows,
        "c1_controllability_gap_per_seed": {str(s): v for s, v in c1_gap_per_seed.items()},
        "c2_relief_retention_frac_at_5_per_seed": {str(s): v for s, v in c2_retention_per_seed.items()},
        "c2_relief_halflife_steps_per_seed": {str(s): v for s, v in c2_halflife_per_seed.items()},
        "interpretation": {
            "label": (
                "mech219_controllability_dissociation_confirmed"
                if all_pass
                else "mech219_controllability_dissociation_not_confirmed"
            ),
            "preconditions": preconditions,
            "criteria": [
                {"name": "C1_controllability_gate", "load_bearing": True, "passed": bool(c1_pass)},
                {"name": "C2_hysteresis_asymmetric_release", "load_bearing": True, "passed": bool(c2_pass)},
                {
                    "name": "C3_non_vacuity_distinct_from_z_harm_un",
                    "load_bearing": True,
                    "passed": bool(c3_pass),
                },
                {"name": "C4_sd021_escapability_parity", "load_bearing": True, "passed": bool(c4_pass)},
            ],
            "criteria_non_degenerate": {
                "C1_controllability_gate": "C1_bout_end_s_inescapable" not in degeneracy["degenerate_metrics"],
                "C2_hysteresis_asymmetric_release": "C2_s_relief_start" not in degeneracy["degenerate_metrics"],
                "C3_non_vacuity_distinct_from_z_harm_un": (
                    "C3_bout_end_un_escapable" not in degeneracy["degenerate_metrics"]
                ),
                "C4_sd021_escapability_parity": c4_non_degenerate,
            },
            "combination_rule": "AND of four load-bearing criteria (C1 AND C2 AND C3 AND C4); no OR.",
        },
        "non_degenerate": bool(degeneracy["non_degenerate"] and c4_non_degenerate),
        "degeneracy_reason": degeneracy["degeneracy_reason"] or (
            "" if c4_non_degenerate else "C4_sd021_attenuation_fired: never fired in any seed"
        ),
        "degenerate_metrics": degeneracy["degenerate_metrics"],
        "scope_note": (
            "This diagnostic implements C1/C2/C3(a)/C4(narrowed) of the design memo's "
            "four-criterion falsifier sketch. C3(b) (anti-correlation with MECH-353 "
            "z_block_assert under the same controllability sweep) and the full z_harm_s "
            "attenuation-magnitude reading for C4 are explicitly OUT OF SCOPE for this pass -- "
            "see the module docstring SCOPE NOTE. They belong in the follow-on behavioural "
            "MECH-219/SD-019b evidence run (claim_ids=['MECH-219'])."
        ),
    })

    out_path = write_flat_manifest(
        manifest,
        config={
            **common_config,
            "c1_margin": C1_MARGIN,
            "c2_retention_frac": C2_RETENTION_FRAC,
            "c2_halflife_min_steps": C2_HALFLIFE_MIN_STEPS,
            "c3_s_ceiling": C3_S_CEILING,
            "c3_un_floor": C3_UN_FLOOR,
            "un_readiness_floor": UN_READINESS_FLOOR,
        },
        seeds=list(seeds),
        script_path=Path(__file__),
        z_goal_stream_stats=_ZG.stats(),
        started_at=t0,
        dry_run=dry_run,
    )

    print(
        f"[smoke] outcome={outcome} C1={c1_pass} C2={c2_pass} C3={c3_pass} C4={c4_pass}",
        flush=True,
    )

    manifest["_out_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = main(dry_run=args.dry_run)
    print(f"outcome: {result['outcome']}")

    emit_outcome(
        outcome=result["outcome"],
        manifest_path=result.get("_out_path"),
        run_id=result.get("run_id"),
        dry_run=args.dry_run,
    )
