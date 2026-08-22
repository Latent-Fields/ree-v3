"""V3-EXQ-945: CEM elite-stage modulatory authority + throughput readiness (DIAGNOSTIC).

EXPERIMENT_PURPOSE = "diagnostic"
claim_ids: [] -- see "WHY NO CLAIM TAG" below.

QUESTION
--------
The 2026-08-19 build (chip-20260819-modulatory-authority-cem-throughput-build,
ree_core/hippocampal/module.py + ree_core/agent.py + ree_core/predictors/e3_selector.py
+ ree_core/utils/config.py) shipped a bounded authority rescale at the CEM
elite-selection stage plus an optional throughput route into E3's committed
selection. V3-EXQ-931's own "addressed_by" note on the substrate_queue entry
`modulatory-bias-selection-authority` says explicitly: the build cannot
discharge V3-EXQ-931's empirical failure_record target -- only a run can.
This run IS that validation. It measures, at the SHIPPED DEFAULT operating
point (cem_modulatory_authority_gain=0.5):
  (a) AUTHORITY -- does the rescale actually flip the CEM elite argmin, with a
      competitive (not merely nonzero) cross-candidate spread ratio?
  (b) THROUGHPUT -- when routed into E3 via modulatory_channel_route_source=
      "cem_elite", does a flipped elite argmin produce a nonzero behavioural
      DV gap vs the ablated arm?
  (c) NO-HARM -- does enabling the routed throughput regress harm exposure
      relative to the ablated arm?

Design doc: REE_assembly/evidence/planning/cem_elite_authority_throughput_design_2026-08-19.md
Substrate queue entry: modulatory-bias-selection-authority (failure_record
v3_exq_931_cem_wanting_weight_selection_authority, resolved:"open").

WHAT THIS IS NOT
-----------------
Section 8 of the design doc is explicit: "No behavioural falsifier is queued.
... A readiness/validation diagnostic for the build itself is the correct
next step; a behavioural falsifier queued now would reproduce V3-EXQ-931's
structural null." This run is exactly that readiness/validation diagnostic --
NOT a claim-tagged behavioural falsifier. It tags no claim_ids and its PASS
would support no registered claim; it exists to flip the substrate_queue
failure_record's `resolved` field from "open", per that record's own
`addressed_by` instruction ("Flip to resolved only on a validation run
meeting BOTH halves"). Same framing V3-EXQ-931 itself used for the same
reason.

WHY THE WANTING TERM IS NOT THE LEVER HERE
-------------------------------------------
V3-EXQ-931 manipulated `wanting_weight`. This run does not: on a FRESH
ResidueField the valence head is identically zero (V3-EXQ-869/923 wash-out
regime; confirmed live by a Step-2.5a probe run from this session, see
below), so a wanting-keyed lever is flat across candidates and the min-
spread-floor guard correctly refuses to amplify it ("scaling zero is still
zero", V3-EXQ-648) -- it would never test the rescale at all. Instead this
run uses the `mode_value` term (SD-MECH267-CEM-SELECTION-FIX H2,
`HippocampalConfig.mode_value_weight` + `mode_conditioning_enabled`), which
keys on the trajectory's mean z_world and is therefore always non-trivial
regardless of residue-field accumulation state (module.py:1650-1683
docstring: "Keys on the trajectory's mean z_world (always non-trivial...)").
`mode_value_weight` is held IDENTICAL across every arm (never manipulated) --
only the authority/throughput flags vary between arms, exactly mirroring how
931 held z_goal formation constant across its wanting_weight ladder.

STEP 2.5a EMPIRICAL PROBE (this session, before writing this script)
----------------------------------------------------------------------
Ran a throwaway probe (not committed) instantiating a real REEAgent with
use_cem_modulatory_authority=True, use_cem_modulatory_throughput=True, and a
mode_value-keyed lever, calling hip.propose_trajectories() directly and
reading get_last_propose_diagnostics(). Confirmed live and reachable:
cem_modulatory_authority_enabled=True, fired_iterations=3,
cem_modulatory_authority_ratio=725.36 (well above the 0.1 competitive floor),
cem_modulatory_authority_competitive=True, cem_modulatory_throughput_available
=True with a [K]-shaped last_candidate_modulatory_bias, and "cem_elite" is a
member of agent._MODULATORY_ROUTE_BACKSTOP_SOURCES (confirming the inert-
route backstop watches this source) while correctly ABSENT from
_MODULATORY_ROUTE_CHANNEL_SOURCES (the cache is not a bias-head channel, per
the amend's own C4b contract). The doc claim is empirically reachable, not
merely recorded -- Step 2.5a's own purpose.

WHY THE DRIVER CALLS hip.propose_trajectories() DIRECTLY, NOT
agent.generate_trajectories()/agent._e3_tick()
----------------------------------------------------------------------------
Read directly from ree_core/agent.py (origin/main): `_e3_tick`'s own call to
`self.hippocampal.propose_trajectories(...)` (agent.py:~5578-5586) does NOT
pass `operating_mode` at all -- there is currently NO live production call
site that threads operating_mode through to the CEM proposer (the ONE
`operating_mode=` grep hit elsewhere in agent.py, ~line 4696, is a different
function entirely, unrelated to trajectory proposal). So `mode_value` cannot
be engaged today via the ordinary agent pipeline; the only way to exercise it
is the same direct call the mechanism's own contract tests use
(tests/contracts/test_cem_modulatory_authority.py::_pool). This driver
replicates `_e3_tick`'s OTHER call arguments (z_world=theta_buffer.summary(),
z_self, e1_prior, action_bias, current_z_goal, persistence_appraisal,
z_harm_a) exactly, adding only `operating_mode`, and then caches the result
into `agent._committed_candidates` and calls `agent.select_action(candidates,
ticks)` normally -- the SAME committed-selection path production uses.
DELIBERATELY SKIPPED, and verified from source to be no-ops under this run's
config (none of the following flags are set, so each guarded block in
`_e3_tick` is unreached regardless): `self.difficulty_gated_proposal_entropy`
(None unless a DGPE config is set), `self.coalition` (None unless a coalition
controller is constructed), `self.multi_content_theta_packet` (None unless
opted in), `config.heartbeat.beta_gate_bistable` (default False). Skipping
these leaves goal-proximity recording, DGPE candidate widening, coalition
gain, theta-packet sealing and the beta-gate hippocampal-completion release
unexercised -- all orthogonal to the CEM-authority/throughput mechanism under
test and all genuinely inert on this run's config, so this replication is
bit-identical to `_e3_tick` plus the one addition (`operating_mode`).

STEP 2.5c SUBSTRATE-PATH OVERLAP GATE -- REASONED, NOT SILENTLY SKIPPED
--------------------------------------------------------------------------
One open `severity: corrupting` substrate_queue entry, `mode-governance-
engagement`, lists `ree_core/agent.py` and `ree_core/utils/config.py` as
bare-file (unqualified) substrate_paths, which this driver's imports
technically overlap at the whole-file level. Read in full: that entry's
corrupting finding is specifically about SalienceCoordinator's external_task/
internal_planning mode ARBITRATION (the `affinity_input_cap` calibration and
the boolean beta_gate.is_elevated commitment-latch discreteness source) and
about `experiments/_lib/regime_occupancy_gate.py`'s min/max-band summary
pattern for reading that arbitration's occupancy statistics -- NEITHER of
which this driver touches. Confirmed structurally, not asserted: this driver
never calls `agent.select_action`'s SalienceCoordinator.tick() path (that is
reached only via a different call chain this driver does not invoke), never
reads `self.salience.operating_mode` (see the paragraph above -- production
does not thread it into proposal at all, and this driver supplies its own
fixed `operating_mode` dict directly, bypassing SalienceCoordinator entirely),
never sets `use_external_task_drive`, and never imports
`regime_occupancy_gate.py`. The `mode_value` H2 term this driver DOES use is
a separate, pre-existing SD-MECH267-CEM-SELECTION-FIX mechanism inside
HippocampalModule, unrelated to cingulate mode governance. Per the gate's own
worked example (an unrelated claim's autopsy "gates nothing for a brand-new
experiment that happens to import the same module" once the actual exercised
code paths are disjoint), this is recorded as reasoned-clear, not silently
passed.

MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION (corrupting,
`ree_core/agent.py::run_sws_schema_pass`) is function-qualified to a sleep
consolidation path this driver never reaches (no sleep flags are set) --
clear no-match.

STEP 2.5b RE-DERIVE BRAKE -- NOT APPLICABLE
---------------------------------------------
claim_ids=[] (see "WHY NO CLAIM TAG"), so no claim this run tags carries a
brake count to check. Per the skill's own "Not braked" carve-out: "a
diagnostic whose purpose is to discriminate WHY the ceiling holds" is
exempted outright, independent of any claim's autopsy history.

DESIGN -- 4 ARMS x 5 SEEDS
-----------------------------------
  ARM_OFF             use_cem_modulatory_authority=False,
                       use_cem_modulatory_throughput=False.
                       THE ABLATION -- matches the shipped production default
                       posture (every new flag off; CEM elite stage advisory-
                       only). mode_value is STILL subtracted into the live
                       score here (mode_conditioning_enabled is held on in
                       every arm, unmanipulated) -- what is ablated is
                       specifically the AUTHORITY RESCALE and the THROUGHPUT
                       ROUTE, not mode_value's raw presence. This makes
                       ARM_OFF's own counterfactual flip-rate a genuinely
                       informative context reading (does the raw, unbounded
                       mode_value term already dominate terrain on this
                       substrate, independent of any rescale?) rather than a
                       trivial P0-style zero.
  ARM_AUTH_RANGE       use_cem_modulatory_authority=True, gain=0.5 (the
                       documented default), normalize_basis="range" (the
                       shipped default basis). THE OPERATING ARM -- the
                       load-bearing C_AUTH criterion reads this arm.
  ARM_AUTH_STD         Same as ARM_AUTH_RANGE but normalize_basis="std".
                       Reported context (non-load-bearing): the chip prompt
                       asked "ideally the range vs std normalize_basis" be
                       compared; this arm answers that at negligible marginal
                       cost (config-only variation, shared harness).
  ARM_AUTH_THROUGHPUT  Same authority config as ARM_AUTH_RANGE PLUS
                       use_cem_modulatory_throughput=True,
                       use_modulatory_channel_routing=True (top-level
                       REEConfig, per the two-flag coupling design.md part 2
                       names as the V3-EXQ-863 silent-inert shape),
                       modulatory_channel_route_source="cem_elite". THE
                       THROUGHPUT ARM -- the load-bearing C_THROUGHPUT
                       criterion reads this arm, GATED on this arm's own
                       selection_flip_rate>0 (per the failure_record's target
                       (b): "an arm whose CEM argmin demonstrably flips").

Everything else -- env, agent dims, mode_value_weight, seeds, episode/step
counts -- is IDENTICAL across arms.

COUNTERFACTUAL AUTHORITY INSTRUMENT (mirrors the substrate's OWN internal
computation, module.py:2140-2196, applied externally to the FINAL returned
candidate pool -- the same pool scope the design doc itself defines for the
throughput cache: "the modulatory contribution over the final returned pool")
--------------------------------------------------------------------------
At every GENUINE E3 tick (gated on ticks["e3_tick"], per the E3 cadence
convention below), for the returned candidate pool:
  1. Recompute (terrain_i, modulatory_i) per candidate via
     hip._score_trajectory(t, operating_mode=OPERATING_MODE,
     return_components=True) -- the exact API this build added.
  2. terrain_spread / modulatory_spread via the substrate's own
     authority_spread_ratio() helper (same definition at both layers, per
     design doc part (c)).
  3. argmin_terrain_only = argmin(terrain).
  4. If this arm's use_cem_modulatory_authority is True AND modulatory_spread
     > cem_modulatory_authority_min_spread_floor: scale =
     gain*terrain_spread/modulatory_spread (the substrate's exact formula);
     argmin_live = argmin(terrain + scale*modulatory).
     Else: argmin_live = argmin(terrain + modulatory) -- the UNSCALED
     combination, matching what _score_trajectory always returns as `score`
     when authority is off (mode_value is subtracted unconditionally whenever
     mode_conditioning is on; only the ELITE-STAGE RESCALE is gated by the
     authority flag).
  5. flip = (argmin_live != argmin_terrain_only).
This is deliberately NOT read off the substrate's own internal per-iteration
elite_indices (those are not externally observable per-iteration); it is
computed on the externally-visible final pool, exactly as designed for the
throughput cache's own scope, so both DVs are measured over the identical
object.

E3 CADENCE / LATCHING (load-bearing for both the propose gate and the read
of agent.e3.last_score_diagnostics)
----------------------------------------------------------------------------
`REEAgent.generate_trajectories` returns CACHED candidates whenever not
ticks["e3_tick"] (e3_steps_per_tick default 10, ree_core/heartbeat/clock.py).
This driver gates its own manual propose call on `ticks["e3_tick"] or
agent._committed_candidates is None` (the `is None` disjunct covers only the
very first tick of a cell, before any real tick has fired) and reports
n_scored_ticks vs n_latched_ticks so the counterfactual's true denominator is
auditable -- the V3-EXQ-785a / V3-EXQ-931 convention. Separately,
`agent.e3.last_score_diagnostics` is populated ONLY inside E3.select() and
LATCHES on ticks where select() did not run -- cleared with
`agent.e3.last_score_diagnostics = None` immediately before every
`select_action(...)` call, per the e3_diagnostics_staleness convention, so a
held commitment tick is never mis-recorded as a fresh selection.

DVs
---
PRIMARY (a) AUTHORITY, C_AUTH (load-bearing): ARM_AUTH_RANGE's
selection_flip_rate > 0 AND cem_modulatory_authority_competitive (read live
from get_last_propose_diagnostics()) True, in >= 3/5 seeds. Reported
alongside: cem_modulatory_authority_ratio (the substrate's own PRE-rescale
readiness statistic -- the same one V3-EXQ-931 read as 0.0037 for wanting).

PRIMARY (b) THROUGHPUT, C_THROUGHPUT (load-bearing): in ARM_AUTH_THROUGHPUT,
selection_flip_rate > 0 AND |mean_resource_proximity(ARM_AUTH_THROUGHPUT) -
mean_resource_proximity(ARM_OFF)| > BEHAV_GAP_EPSILON, in >= 3/5 seeds --
GATED on this arm's own flip rate per the failure_record's target (b)
wording ("an arm whose CEM argmin demonstrably flips"), never interpreted
from a non-flipping arm. Also reads modulatory_channel_route_range and
score_bias_to_raw_range_ratio from agent.e3.last_score_diagnostics (the E3-
layer readiness diagnostics this build's part (b) inherits).

SECONDARY (c) NO-HARM (reported, non-gating): harm_events / steps_alive
(contact_rate for the negative direction) in ARM_AUTH_THROUGHPUT vs ARM_OFF.

CONTEXT (reported, non-gating): ARM_OFF's own raw (unscaled) counterfactual
flip rate and modulatory_authority_ratio -- tells us whether an ON arm's
competitive reading reflects the bounded rescale actually doing work, or a
term that was already naturally dominant regardless (the scratch probe's
toy-scale reading of ratio=725 suggests raw dominance is at least possible on
this substrate; this run measures it under realistic env-driven terrain
magnitudes rather than random init).

DV-SYMMETRY (Step 3.5 mandatory declaration)
---------------------------------------------
`selection_flip_rate` is a RANK/argmin DV -- invariant under any monotone
rescaling that preserves candidate ordering, which is exactly why it detects
a uniform-broadcast-style hazard instead of being fooled by one. It is NOT
invariant under this run's manipulation: the authority gain/basis flags
rescale a per-candidate (not broadcast) modulatory vector non-uniformly
across candidates whenever that vector is non-constant, which the P1
readiness precondition below certifies per cell. `mean_resource_proximity` is
a per-step continuous MAGNITUDE, not derived from any CEM-internal rank
score, so monotone-rescaling symmetry does not apply to it.

PRE-REGISTERED ACCEPTANCE CRITERIA (absolute constants, not run-derived)
--------------------------------------------------------------------------
Verbatim from the substrate_queue failure_record target:
"(a) selection_flip_rate > 0 at the documented operating weight with a
competitive cross-candidate spread ratio, AND (b) a nonzero behavioural gap
vs ablation in an arm whose CEM argmin demonstrably flips."
  P1 (gating, readiness) MODULATORY_TERM_LIVE: worst arm/seed's mean raw
     modulatory cross-candidate spread over genuine-tick final pools clears
     P1_MOD_SPREAD_FLOOR. mode_value is documented always-non-trivial; this
     verifies that empirically rather than trusting the doc (Step 2.5a's
     purpose, restated as an in-run precondition per the always-record
     convention).
  C_AUTH (load-bearing): see PRIMARY (a) above.
  C_THROUGHPUT (load-bearing): see PRIMARY (b) above.
Overall PASS = P1 AND C_AUTH AND C_THROUGHPUT (plain AND).

NO GRADIENT TRAINING
---------------------
E1/E2 stay at random init throughout; no head is trained on z_world / z_harm
/ any encoder output. The mechanism under test operates purely structurally
on the CEM elite-selection scoring/argsort step, so the phased-training
protocol does not apply.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_945_cem_elite_authority_throughput_readiness.py [--dry-run]

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.predictors.e3_selector import (  # noqa: E402
    AUTHORITY_COMPETITIVE_RATIO_FLOOR_DEFAULT,
    authority_ratio_is_competitive,
    authority_spread_ratio,
)
from ree_core.utils.config import REEConfig  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_945_cem_elite_authority_throughput_readiness"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []

# Readiness-anchor reachability guard: none of this script's criteria hand-
# writes a narrowed signature. C_AUTH / C_THROUGHPUT read the substrate's own
# get_last_propose_diagnostics() / e3.last_score_diagnostics fields directly,
# and the counterfactual flip instrument replicates the substrate's own
# published rescale formula rather than an independently-invented one.
ANCHOR_REACHABILITY_EXEMPT = (
    "No hand-written narrowed anchor exists in this script; C_AUTH/"
    "C_THROUGHPUT read the substrate's own diagnostics fields and the "
    "counterfactual instrument replicates the substrate's own published "
    "rescale formula (module.py:2140-2196) rather than an independent one."
)

GAIN = 0.5  # HippocampalConfig.cem_modulatory_authority_gain documented default
MIN_SPREAD_FLOOR = 1e-6  # cem_modulatory_authority_min_spread_floor default

ARM_FLAGS: Dict[str, Dict[str, Any]] = {
    "ARM_OFF": {
        "use_cem_modulatory_authority": False,
        "cem_modulatory_authority_gain": GAIN,
        "cem_modulatory_authority_normalize_basis": "range",
        "use_cem_modulatory_throughput": False,
        "use_modulatory_channel_routing": False,
        "modulatory_channel_route_source": "none",
    },
    "ARM_AUTH_RANGE": {
        "use_cem_modulatory_authority": True,
        "cem_modulatory_authority_gain": GAIN,
        "cem_modulatory_authority_normalize_basis": "range",
        "use_cem_modulatory_throughput": False,
        "use_modulatory_channel_routing": False,
        "modulatory_channel_route_source": "none",
    },
    "ARM_AUTH_STD": {
        "use_cem_modulatory_authority": True,
        "cem_modulatory_authority_gain": GAIN,
        "cem_modulatory_authority_normalize_basis": "std",
        "use_cem_modulatory_throughput": False,
        "use_modulatory_channel_routing": False,
        "modulatory_channel_route_source": "none",
    },
    "ARM_AUTH_THROUGHPUT": {
        "use_cem_modulatory_authority": True,
        "cem_modulatory_authority_gain": GAIN,
        "cem_modulatory_authority_normalize_basis": "range",
        "use_cem_modulatory_throughput": True,
        "use_modulatory_channel_routing": True,
        "modulatory_channel_route_source": "cem_elite",
    },
}
ARMS: List[str] = list(ARM_FLAGS)
BASELINE_ARM = "ARM_OFF"
OPERATING_ARM = "ARM_AUTH_RANGE"
THROUGHPUT_ARM = "ARM_AUTH_THROUGHPUT"

SEEDS: List[int] = [42, 43, 45, 46, 47]  # seed 44: recurring per-seed early-death
                                          # instability on this env config family
                                          # (V3-EXQ-868/881/914/931)

N_EPISODES = 10
STEPS_PER_EPISODE = 40

# --- env (reuses V3-EXQ-914/931's calibration for a realistic foraging env;
# not load-bearing for THIS run's manipulation, which does not depend on
# residue-field harm/benefit accumulation) ---------------------------------
GRID_SIZE = 16
NUM_RESOURCES = 3
NUM_HAZARDS = 1
HAZARD_HARM = 0.1

# --- the manipulation-independent lever: mode_value keyed on z_world ------
WORLD_DIM = 32
SELF_DIM = 32
OPERATING_MODE: Dict[str, float] = {"explore": 1.0}
MODE_VALUE_WEIGHT: Dict[str, List[float]] = {"explore": [0.7] * WORLD_DIM}

# --- pre-registered thresholds (absolute constants, NOT run-derived) ------
P1_MOD_SPREAD_FLOOR = 1e-4
P1_SEED_PASS_FRACTION = 4.0 / 5.0
C_AUTH_SEED_PASS_FRACTION = 3.0 / 5.0
C_THROUGHPUT_SEED_PASS_FRACTION = 3.0 / 5.0
BEHAV_GAP_EPSILON = 1e-4  # "nonzero" behavioural gap, above float/env noise

# One representative agent per ARM, retained for write_flat_manifest's
# enabled_default_off_flags recording (flags differ by arm). Not one per
# cell -- see V3-EXQ-931's own precedent for why (avoid holding every
# arm x seed agent alive at once).
_ARM_AGENTS: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=NUM_RESOURCES,
        num_hazards=NUM_HAZARDS,
        hazard_harm=HAZARD_HARM,
        resource_benefit=0.5,
        resource_respawn_on_consume=True,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.05,
        proximity_approach_threshold=0.15,
    )


def _config_slice(arm: str) -> Dict[str, Any]:
    return {
        "env": {"size": GRID_SIZE, "num_resources": NUM_RESOURCES,
                "num_hazards": NUM_HAZARDS, "hazard_harm": HAZARD_HARM},
        "schedule": {"n_episodes": N_EPISODES, "steps": STEPS_PER_EPISODE},
        "alpha_world": 0.9,
        "mode_conditioning_enabled": True,
        "mode_value_weight": MODE_VALUE_WEIGHT,
        "operating_mode": OPERATING_MODE,
        **ARM_FLAGS[arm],
    }


def _make_agent(env: CausalGridWorldV2, arm: str) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        **ARM_FLAGS[arm],
    )
    # mode_conditioning_enabled / mode_value_weight are NOT REEConfig.from_dims
    # params (reference-reeconfig-from-dims-silent-kwargs would silently drop
    # them there) -- set directly on the hippocampal sub-config, matching
    # tests/contracts/test_cem_modulatory_authority.py::_mode_cfg exactly.
    cfg.hippocampal.mode_conditioning_enabled = True
    cfg.hippocampal.mode_value_weight = dict(MODE_VALUE_WEIGHT)
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Counterfactual authority instrument (mirrors module.py:2140-2196)
# ---------------------------------------------------------------------------
def _components(hip, traj) -> Optional[Any]:
    if traj.world_states is None or traj.get_world_state_sequence() is None:
        return None
    with torch.no_grad():
        _score, comp = hip._score_trajectory(
            traj, operating_mode=OPERATING_MODE, return_components=True
        )
    return comp


def _argmin(values: List[float]) -> int:
    """CEM MINIMISES the score; ties resolve to the lowest index, matching
    module.py's own torch.argsort(scores_tensor)[:num_elite] tie behaviour."""
    return min(range(len(values)), key=lambda i: (values[i], i))


def _counterfactual_flip(hip, candidates, authority_on: bool) -> Dict[str, Any]:
    """Returns dict with terrain_spread, modulatory_spread, raw_ratio, flip,
    or None fields if the pool could not be scored (too few candidates /
    missing world_states)."""
    if not candidates or len(candidates) < 2:
        return {"scorable": False}
    comps = [_components(hip, t) for t in candidates]
    if any(c is None for c in comps):
        return {"scorable": False}
    terr = torch.stack([c["terrain"] for c in comps]).detach()
    mod = torch.stack([c["modulatory"] for c in comps]).detach()
    terr_spread = float((terr.max() - terr.min()).item())
    mod_spread = float((mod.max() - mod.min()).item())
    raw_ratio = authority_spread_ratio(mod, terr, basis="range")

    argmin_terrain = _argmin(terr.tolist())
    if authority_on and mod_spread > MIN_SPREAD_FLOOR and terr_spread >= 0.0:
        scale = (GAIN * terr_spread) / mod_spread
        combined = terr + scale * mod
    else:
        combined = terr + mod  # unscaled -- matches live _score_trajectory
    argmin_live = _argmin(combined.tolist())
    return {
        "scorable": True,
        "terrain_spread": terr_spread,
        "modulatory_spread": mod_spread,
        "raw_ratio": raw_ratio,
        "flip": bool(argmin_live != argmin_terrain),
    }


# ---------------------------------------------------------------------------
# Single arm x seed cell
# ---------------------------------------------------------------------------
def _run_cell(arm: str, seed: int, dry_run: bool = False) -> Dict[str, Any]:
    n_episodes = 2 if dry_run else N_EPISODES
    steps_per = 8 if dry_run else STEPS_PER_EPISODE
    authority_on = bool(ARM_FLAGS[arm]["use_cem_modulatory_authority"])

    print(f"Seed {seed} Condition {arm}", flush=True)

    with arm_cell(seed, config_slice=_config_slice(arm),
                  script_path=Path(__file__)) as cell:
        env = _make_env(seed)
        agent = _make_agent(env, arm)
        device = agent.device
        wd = agent.config.latent.world_dim
        hip = agent.hippocampal

        n_scored_ticks = 0
        n_latched_ticks = 0
        n_flips = 0
        n_scorable = 0
        n_authority_fired = 0
        n_authority_competitive = 0
        raw_ratios: List[float] = []
        mod_spreads: List[float] = []
        authority_ratios: List[float] = []  # substrate's own diagnostics field

        n_e3_diag_reads = 0
        route_ranges: List[float] = []
        throughput_available_count = 0

        steps_alive = 0
        resource_contacts = 0
        harm_events = 0
        resource_proximity: List[float] = []

        for ep in range(n_episodes):
            _, obs = env.reset()
            agent.reset()
            for t in range(steps_per):
                rfv = obs.get("resource_field_view")
                if rfv is not None:
                    resource_proximity.append(float(rfv.max().item()))

                obs_body = obs["body_state"].to(device)
                obs_world = obs["world_state"].to(device)
                latent = agent.sense(obs_body, obs_world)
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", True)
                    else torch.zeros(1, wd, device=device)
                )

                if ticks.get("e3_tick", False) or agent._committed_candidates is None:
                    z_world_for_e3 = agent.theta_buffer.summary()
                    _current_z_goal = (
                        agent.goal_state.z_goal
                        if (agent.goal_state is not None and agent.goal_state.is_active())
                        else None
                    )
                    _persistence = agent._compute_persistence_appraisal(z_world_for_e3)
                    candidates = hip.propose_trajectories(
                        z_world=z_world_for_e3,
                        z_self=latent.z_self,
                        num_candidates=None,
                        e1_prior=e1_prior,
                        action_bias=getattr(agent, "_cue_action_bias", None),
                        current_z_goal=_current_z_goal,
                        persistence_appraisal=_persistence,
                        z_harm_a=latent.z_harm_a,
                        operating_mode=OPERATING_MODE,
                    )
                    agent._committed_candidates = candidates
                    n_scored_ticks += 1

                    cf = _counterfactual_flip(hip, candidates, authority_on)
                    if cf.get("scorable"):
                        n_scorable += 1
                        mod_spreads.append(cf["modulatory_spread"])
                        raw_ratios.append(cf["raw_ratio"])
                        if cf["flip"]:
                            n_flips += 1

                    pd = hip.get_last_propose_diagnostics()
                    if pd.get("cem_modulatory_authority_enabled"):
                        if int(pd.get("cem_modulatory_authority_fired_iterations", 0)) > 0:
                            n_authority_fired += 1
                        ratio = pd.get("cem_modulatory_authority_ratio")
                        if ratio is not None:
                            authority_ratios.append(float(ratio))
                        if pd.get("cem_modulatory_authority_competitive"):
                            n_authority_competitive += 1
                else:
                    candidates = agent._committed_candidates
                    n_latched_ticks += 1

                # E3-diagnostics-staleness convention: clear immediately
                # before select(), record only if select() genuinely ran.
                agent.e3.last_score_diagnostics = None
                action = agent.select_action(candidates, ticks)
                e3_diag = agent.e3.last_score_diagnostics
                if e3_diag is not None:
                    n_e3_diag_reads += 1
                    rr = e3_diag.get("modulatory_channel_route_range")
                    if rr is not None:
                        route_ranges.append(float(rr))
                    if e3_diag.get("cem_modulatory_throughput_available"):
                        throughput_available_count += 1

                _flat, harm_signal, done, info, obs = env.step(action)
                if float(harm_signal) < 0:
                    harm_events += 1
                elif float(harm_signal) > 0:
                    resource_contacts += 1
                steps_alive += 1
                if done:
                    break

            print(
                f"  [train] {arm} seed={seed} ep {ep + 1}/{n_episodes} "
                f"scored={n_scored_ticks} flips={n_flips}",
                flush=True,
            )

        _ARM_AGENTS.setdefault(arm, agent)

        def _mean(vals: List[float]) -> float:
            return float(statistics.fmean(vals)) if vals else 0.0

        mod_spread_mean = _mean(mod_spreads)
        row: Dict[str, Any] = {
            "arm": arm,
            "seed": seed,
            "authority_on": authority_on,
            "gain": ARM_FLAGS[arm]["cem_modulatory_authority_gain"],
            "basis": ARM_FLAGS[arm]["cem_modulatory_authority_normalize_basis"],
            "throughput_on": bool(ARM_FLAGS[arm]["use_cem_modulatory_throughput"]),
            # --- primary DV (a) ---
            "selection_flip_rate": (n_flips / n_scorable if n_scorable else 0.0),
            "n_flips": int(n_flips),
            "n_scorable": int(n_scorable),
            "n_scored_ticks": int(n_scored_ticks),
            "n_latched_ticks": int(n_latched_ticks),
            "modulatory_spread_mean": mod_spread_mean,
            "counterfactual_raw_ratio_mean": _mean(raw_ratios),
            "cem_modulatory_authority_ratio_mean": _mean(authority_ratios),
            "cem_modulatory_authority_fired_frac": (
                n_authority_fired / n_scored_ticks if n_scored_ticks else 0.0
            ),
            "cem_modulatory_authority_competitive_frac": (
                n_authority_competitive / n_scored_ticks if n_scored_ticks else 0.0
            ),
            # --- primary DV (b) ---
            "n_e3_diag_reads": int(n_e3_diag_reads),
            "modulatory_channel_route_range_mean": _mean(route_ranges),
            "cem_modulatory_throughput_available_frac": (
                throughput_available_count / n_e3_diag_reads if n_e3_diag_reads else 0.0
            ),
            # --- behavioural / no-harm, secondary ---
            "mean_resource_proximity": _mean(resource_proximity),
            "contact_rate": (resource_contacts / steps_alive if steps_alive else 0.0),
            "resource_contacts": int(resource_contacts),
            "harm_events": int(harm_events),
            "harm_rate": (harm_events / steps_alive if steps_alive else 0.0),
            "steps_alive": int(steps_alive),
        }
        cell.stamp(row)

    cell_ok = mod_spread_mean > P1_MOD_SPREAD_FLOOR
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row


# ---------------------------------------------------------------------------
# Aggregation + pre-registered criteria
# ---------------------------------------------------------------------------
def _by_arm(rows: List[Dict[str, Any]], arm: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r["arm"] == arm]


def _analyse(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_seeds = len(SEEDS)

    # --- P1: modulatory term genuinely live in every arm ---
    p1_per_arm: Dict[str, float] = {}
    for arm in ARMS:
        ok = sum(
            1 for r in _by_arm(rows, arm)
            if r["modulatory_spread_mean"] > P1_MOD_SPREAD_FLOOR
        )
        p1_per_arm[arm] = ok / n_seeds if n_seeds else 0.0
    p1_worst_arm = min(p1_per_arm, key=lambda a: p1_per_arm[a])
    p1_measured = p1_per_arm[p1_worst_arm]
    p1_met = p1_measured >= P1_SEED_PASS_FRACTION

    # --- C_AUTH: operating arm demonstrably flips, competitively ---
    auth_rows = _by_arm(rows, OPERATING_ARM)
    c_auth_ok = sum(
        1 for r in auth_rows
        if r["selection_flip_rate"] > 0.0
        and r["cem_modulatory_authority_competitive_frac"] >= 0.5
    )
    c_auth_measured = c_auth_ok / n_seeds if n_seeds else 0.0
    c_auth_pass = c_auth_measured >= C_AUTH_SEED_PASS_FRACTION

    # --- C_THROUGHPUT: throughput arm flips AND shows a real behavioural gap ---
    thr_rows = _by_arm(rows, THROUGHPUT_ARM)
    base_rows = {r["seed"]: r for r in _by_arm(rows, BASELINE_ARM)}
    c_thr_ok = 0
    behav_gaps: Dict[int, float] = {}
    for r in thr_rows:
        base = base_rows.get(r["seed"])
        gap = (
            r["mean_resource_proximity"] - base["mean_resource_proximity"]
            if base is not None else 0.0
        )
        behav_gaps[r["seed"]] = gap
        if r["selection_flip_rate"] > 0.0 and abs(gap) > BEHAV_GAP_EPSILON:
            c_thr_ok += 1
    c_thr_measured = c_thr_ok / n_seeds if n_seeds else 0.0
    c_thr_pass = c_thr_measured >= C_THROUGHPUT_SEED_PASS_FRACTION

    def _arm_mean(arm: str, key: str) -> float:
        vals = [r[key] for r in _by_arm(rows, arm)]
        return float(statistics.fmean(vals)) if vals else 0.0

    per_arm = {
        arm: {
            "selection_flip_rate_mean": _arm_mean(arm, "selection_flip_rate"),
            "n_seeds_with_any_flip": sum(
                1 for r in _by_arm(rows, arm) if r["selection_flip_rate"] > 0.0
            ),
            "modulatory_spread_mean": _arm_mean(arm, "modulatory_spread_mean"),
            "counterfactual_raw_ratio_mean": _arm_mean(arm, "counterfactual_raw_ratio_mean"),
            "cem_modulatory_authority_ratio_mean": _arm_mean(
                arm, "cem_modulatory_authority_ratio_mean"
            ),
            "cem_modulatory_authority_competitive_frac_mean": _arm_mean(
                arm, "cem_modulatory_authority_competitive_frac"
            ),
            "modulatory_channel_route_range_mean": _arm_mean(
                arm, "modulatory_channel_route_range_mean"
            ),
            "cem_modulatory_throughput_available_frac_mean": _arm_mean(
                arm, "cem_modulatory_throughput_available_frac"
            ),
            "mean_resource_proximity_mean": _arm_mean(arm, "mean_resource_proximity"),
            "contact_rate_mean": _arm_mean(arm, "contact_rate"),
            "harm_rate_mean": _arm_mean(arm, "harm_rate"),
            "n_scored_ticks_total": sum(r["n_scored_ticks"] for r in _by_arm(rows, arm)),
            "n_latched_ticks_total": sum(r["n_latched_ticks"] for r in _by_arm(rows, arm)),
        }
        for arm in ARMS
    }

    base_harm = per_arm[BASELINE_ARM]["harm_rate_mean"]
    thr_harm = per_arm[THROUGHPUT_ARM]["harm_rate_mean"]
    c_noharm_gap = thr_harm - base_harm  # positive = MORE harm under throughput

    gates_met = p1_met
    outcome = "PASS" if (gates_met and c_auth_pass and c_thr_pass) else "FAIL"

    criteria_non_degenerate = {
        "P1": bool(any(r["n_scorable"] > 0 for r in rows)),
        "C_AUTH": bool(p1_per_arm[OPERATING_ARM] >= P1_SEED_PASS_FRACTION),
        "C_THROUGHPUT": bool(p1_per_arm[THROUGHPUT_ARM] >= P1_SEED_PASS_FRACTION),
    }

    if not p1_met:
        label = "substrate_not_ready_requeue"
    elif c_auth_pass and c_thr_pass:
        label = "cem_authority_and_throughput_validated_at_operating_gain"
    elif c_auth_pass and not c_thr_pass:
        label = "cem_authority_established_but_throughput_not_demonstrated"
    elif not c_auth_pass:
        label = "cem_authority_not_established_at_operating_gain"
    else:
        label = "cem_authority_and_throughput_indeterminate"

    preconditions = [
        {
            "name": "modulatory_term_reaches_scoring_live",
            "kind": "readiness",
            "description": (
                "worst arm's fraction of seeds with mean raw (pre-rescale) "
                "modulatory cross-candidate spread over genuine-tick final "
                f"pools > {P1_MOD_SPREAD_FLOOR}"
            ),
            "control": (
                "mode_value keys on trajectory mean z_world, documented "
                "always-non-trivial independent of residue-field state -- "
                "verified live by a Step-2.5a probe before this run"
            ),
            "measured": p1_measured,
            "threshold": P1_SEED_PASS_FRACTION,
            "direction": "lower",
            "offending_cell": p1_worst_arm,
            "certifies_arms": ARMS,
            "met": bool(p1_met),
        },
    ]

    return {
        "outcome": outcome,
        "per_arm": per_arm,
        "c_auth_seed_fraction": c_auth_measured,
        "c_auth_pass": bool(c_auth_pass),
        "c_throughput_seed_fraction": c_thr_measured,
        "c_throughput_pass": bool(c_thr_pass),
        "c_behav_proximity_gap_throughput_vs_ablated": behav_gaps,
        "c_noharm_harm_rate_gap_throughput_vs_ablated": c_noharm_gap,
        "gates_met": bool(gates_met),
        "interpretation": {
            "label": label,
            "combination_rule": (
                "PASS = P1 AND C_AUTH AND C_THROUGHPUT (plain AND), matching "
                "the failure_record's own '(a) ... AND (b) ...' acceptance "
                "wording verbatim."
            ),
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": [
                {"name": "P1_modulatory_term_live", "load_bearing": False, "passed": bool(p1_met)},
                {"name": "C_AUTH_operating_gain_flips_competitively",
                 "load_bearing": True, "passed": bool(c_auth_pass)},
                {"name": "C_THROUGHPUT_behavioural_gap_in_flipping_arm",
                 "load_bearing": True, "passed": bool(c_thr_pass)},
            ],
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = SEEDS[:2] if dry_run else SEEDS
    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            rows.append(_run_cell(arm, seed, dry_run=dry_run))

    analysis = _analyse(rows)

    manifest: Dict[str, Any] = {
        "run_id": (
            f"{EXPERIMENT_TYPE}_"
            f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "outcome": analysis["outcome"],
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": bool(dry_run),
        "arm_results": rows,
        "per_arm": analysis["per_arm"],
        "c_auth_seed_fraction": analysis["c_auth_seed_fraction"],
        "c_auth_pass": analysis["c_auth_pass"],
        "c_throughput_seed_fraction": analysis["c_throughput_seed_fraction"],
        "c_throughput_pass": analysis["c_throughput_pass"],
        "c_behav_proximity_gap_throughput_vs_ablated": (
            analysis["c_behav_proximity_gap_throughput_vs_ablated"]
        ),
        "c_noharm_harm_rate_gap_throughput_vs_ablated": (
            analysis["c_noharm_harm_rate_gap_throughput_vs_ablated"]
        ),
        "interpretation": analysis["interpretation"],
        "pre_registered_thresholds": {
            "P1_MOD_SPREAD_FLOOR": P1_MOD_SPREAD_FLOOR,
            "P1_SEED_PASS_FRACTION": P1_SEED_PASS_FRACTION,
            "C_AUTH_SEED_PASS_FRACTION": C_AUTH_SEED_PASS_FRACTION,
            "C_THROUGHPUT_SEED_PASS_FRACTION": C_THROUGHPUT_SEED_PASS_FRACTION,
            "BEHAV_GAP_EPSILON": BEHAV_GAP_EPSILON,
            "GAIN": GAIN,
        },
        "custom_information": {
            "manipulated_fields": [
                "HippocampalConfig.use_cem_modulatory_authority",
                "HippocampalConfig.cem_modulatory_authority_normalize_basis",
                "HippocampalConfig.use_cem_modulatory_throughput",
                "REEConfig.use_modulatory_channel_routing",
                "REEConfig.modulatory_channel_route_source",
            ],
            "held_constant_lever": "HippocampalConfig.mode_value_weight (mode_conditioning_enabled=True, operating_mode={'explore': 1.0}) -- identical across every arm",
            "consumer_call_site": (
                "ree_core/hippocampal/module.py::HippocampalModule."
                "propose_trajectories (elite argsort block) + "
                "ree_core/agent.py::REEAgent.select_action (cem_elite route dispatch)"
            ),
            "e3_cadence_note": (
                "manual propose call gated on ticks['e3_tick'] or "
                "agent._committed_candidates is None; n_latched_ticks reported "
                "so the counterfactual's true denominator is auditable"
            ),
            "substrate_queue_target": "modulatory-bias-selection-authority (failure_record v3_exq_931_cem_wanting_weight_selection_authority)",
            "step_2_5c_overlap_gate_note": (
                "open corrupting entry mode-governance-engagement lists "
                "ree_core/agent.py + ree_core/utils/config.py (whole-file) in "
                "substrate_paths; reasoned clear (not skipped) -- this driver "
                "never exercises SalienceCoordinator.tick()/external_task_drive/"
                "regime_occupancy_gate.py, and supplies operating_mode directly "
                "rather than via self.salience.operating_mode. See module "
                "docstring section 'STEP 2.5c SUBSTRATE-PATH OVERLAP GATE' for "
                "full reasoning."
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

    if not analysis["interpretation"]["criteria_non_degenerate"]["P1"]:
        manifest["non_degenerate"] = False
        manifest["degeneracy_reason"] = (
            "the modulatory term was never scorable (n_scorable=0) in every "
            "arm -- the counterfactual instrument measured nothing"
        )

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config={
            "env": {"size": GRID_SIZE, "num_resources": NUM_RESOURCES,
                    "num_hazards": NUM_HAZARDS, "hazard_harm": HAZARD_HARM},
            "schedule": {"n_episodes": N_EPISODES, "steps": STEPS_PER_EPISODE},
            "arms": ARM_FLAGS,
            "mode_conditioning_enabled": True,
            "mode_value_weight": MODE_VALUE_WEIGHT,
            "operating_mode": OPERATING_MODE,
            "alpha_world": 0.9,
        },
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        agent=list(_ARM_AGENTS.values()),
    )
    return {"outcome": analysis["outcome"], "manifest_path": out_path,
            "analysis": analysis}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    result = run_experiment(dry_run=args.dry_run)
    out_path = result["manifest_path"]
    analysis = result["analysis"]

    print("", flush=True)
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label:   {analysis['interpretation']['label']}", flush=True)
    for arm in ARMS:
        pa = analysis["per_arm"][arm]
        print(
            f"  {arm:20s} flip_rate={pa['selection_flip_rate_mean']:.4f} "
            f"seeds_with_flip={pa['n_seeds_with_any_flip']}/{len(SEEDS)} "
            f"auth_ratio={pa['cem_modulatory_authority_ratio_mean']:.3g} "
            f"competitive_frac={pa['cem_modulatory_authority_competitive_frac_mean']:.2f} "
            f"route_range={pa['modulatory_channel_route_range_mean']:.3g} "
            f"prox={pa['mean_resource_proximity_mean']:.4f} "
            f"harm_rate={pa['harm_rate_mean']:.4f}",
            flush=True,
        )
    print(f"manifest: {out_path}", flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
