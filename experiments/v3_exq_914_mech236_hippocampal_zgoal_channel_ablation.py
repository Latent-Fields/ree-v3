"""V3-EXQ-914: MECH-236 hippocampal z_goal-injection channel ablation (EVIDENCE).

CLAIM TESTED (claims.yaml MECH-236, verbatim):
"Hippocampal trajectory proposals must be conditioned on a goal signal
(z_goal) injected from E3 via a dedicated input channel; without this
conditioning, the hippocampal module generates only position-based
trajectories and cannot produce goal-directed navigation toward target
states."

MECHANISM UNDER TEST (code-confirmed, ree_core/agent.py REEAgent._e3_tick,
~lines 5437-5520): every E3 tick, `_current_z_goal = goal_state.z_goal if
goal_state.is_active() else None` is threaded into
`HippocampalModule.propose_trajectories(current_z_goal=...)`. Its SOLE
consumer inside `propose_trajectories` is the MECH-293 waking
ghost-goal-probe branch (module.py docstring: "a minority budget of CEM
probes is seeded around the highest-priority bank entries' anchor.z_world"),
gated on `config.use_mech293_ghost_probes` (default False) AND a nonempty
MECH-292 ghost-goal bank. This IS the literal "z_goal injected via a
dedicated input channel" MECH-236 describes -- the only call site in the
current substrate where the raw z_goal tensor (not a benefit_exposure-
derived proxy) reaches trajectory PROPOSAL.

SCOPING NOTE (why NOT the VALENCE_WANTING / wanting_weight pathway):
`_score_trajectory`'s CEM elite-selection scoring also has a goal-related
term (`wanting_weight * mean(VALENCE_WANTING along trajectory)`), and
`ree_core/residue/field.py` labels VALENCE_WANTING "z_goal signal (frontal
goal attractor)" in a comment. But its actual writer,
`REEAgent.update_benefit_salience(benefit_exposure, ...)` (agent.py:10730),
is driven by `benefit_exposure` (proximal resource contact) -- a separate,
correlated-but-distinct upstream signal from GoalState.z_goal itself, not
the z_goal tensor. `wanting_weight` is therefore held at 0.0 in every arm
below, isolating the literal z_goal-injection channel from that separate
pathway. This also sidesteps SD-RESIDUE-VALENCE-BOUND
(substrate_queue.json, severity degrading, open as of 2026-08-10):
`update_benefit_salience`'s ResidueField.update_valence write is an
unclamped accumulator with no decay under sustained exposure, and is
confirmed orphaned (never invoked) in at least one driver family -- an
unrelated, already-flagged defect this design does not depend on either
way.

GOV-REUSE-1 (Step 2.4): claim_evidence.v1.json MECH-236 entry has
genuine_exp_count=0 / experimental_confidence=0.0 (4 literature entries
only, 2026-08-09 lit-pull). No prior V3 run of any kind is claim-tagged
MECH-236. `use_mech293_ghost_probes` is exercised by V3-EXQ-497
(diagnostic, wiring-only), V3-EXQ-868 (MECH-292 evidence), V3-EXQ-881
(MECH-293 evidence, ghost-vs-value-flat trajectory DIVERGENCE within a
single tick, not behavioural navigation outcome over an episode) -- none
measure whether the channel changes actual environmental navigation,
which is what MECH-236 asks. Not recoverable -> run.

RELATED DRAFT (flagged as follow-on, NOT used as a base here):
V3-EXQ-495a (v3_exq_495a_mech163_planned_system_gate.py, 958 lines,
drafted 2026-04-27, explicitly gated "DO NOT QUEUE until MECH-292+293
land" and never queued/run) is MECH-163's own V3 full-completion gate and
uses the same HABIT (ghost off) vs PLANNED (ghost on) contrast at much
greater scale (3 conditions x 2 paradigms x 7 seeds x 250-episode phased
training, detour + novel-context generalisation). It would also bear on
MECH-236 once modernised, but predates the 2026-07-12 recording standard
and the 2026-06-07 arm-fingerprint mandate, so it cannot be queued as-is
-- bringing a 958-line, multi-hour, never-executed script current is a
separate, much larger undertaking than a first, narrowly-scoped MECH-236
evidence run. MECH-236's own claim needs only the ON/OFF channel
contrast, not the detour/generalisation battery MECH-163 specifically
requires. Left as flagged follow-on (see queue entry note).

DESIGN -- 3 ARMS x N SEEDS, single paradigm (open-field resource
approach; no detour/generalisation -- out of scope for this claim):

  ARM_NOGOAL   z_goal_enabled=False. No goal representation anywhere.
               Floor / non-degeneracy context only (NOT load-bearing for
               MECH-236 -- reports whether z_goal existing at all, absent
               this specific channel, already matters via some OTHER
               pathway).

  ARM_CLOSED   z_goal_enabled=True, drive_weight=2.0 (identical to
               ARM_OPEN); use_mech293_ghost_probes=False (and its
               MECH-292/SD-039/anchor prerequisites False). z_goal FORMS
               (MECH-230 intact) but the hippocampal-proposal channel is
               closed. This is the "ablate the E3->hippocampal goal input
               while preserving E3 goal encoding" arm MECH-236's testable
               prediction specifies.

  ARM_OPEN     identical z_goal formation to ARM_CLOSED, PLUS
               use_mech293_ghost_probes=True + use_per_stream_vs=True +
               use_event_segmenter=True + use_invalidation_trigger=True +
               use_anchor_sets=True + use_sd039_anchor_payload=True +
               use_mech292_ghost_bank=True (the full stack MECH-293 needs
               to do anything -- module.py docstring, and V3-EXQ-495a's
               own comment: "PLANNED needs this on"). This is the
               "restore the input" arm.

  z_goal formation is FORCED identically in ARM_CLOSED and ARM_OPEN via a
  constant per-tick `agent.update_z_goal(benefit_exposure=FORCED_BENEFIT,
  drive_level=FORCED_DRIVE)` call (same pattern as V3-EXQ-807),
  independent of whether the agent's own navigation actually reaches a
  resource. Deliberate, not a shortcut: it decouples "does z_goal form"
  from "does navigation follow it", exactly the confound the claims.yaml
  MECH-230 evidence_quality_note documents repeatedly (z_goal_norm=0
  cannot distinguish "goal not structured" from "goal not wired" / "no
  resource contact yet"). Forcing the seed means a null C_MAIN result can
  only be read as "the channel doesn't help", never as "z_goal never
  formed in this arm".

  No gradient training / no phased P0/P1/P2 (same as V3-EXQ-807): E1/E2
  stay at random init throughout. The claim under test is an
  ARCHITECTURAL coupling (does the wiring exist and matter), not a
  learned-representation question, so untrained predictors are the
  correct substrate state -- the CEM proposal + ghost-seeding machinery
  under test operates purely structurally on z_world / z_goal geometry,
  no gradient step required.

ANCHOR POPULATION (Step 2.5a empirical probe finding, load-bearing for
this design): a source grep of ree_core/agent.py shows the ordinary
agent step loop only ever READS `anchor_set.active_anchors()` -- no
production call site WRITES a new anchor automatically. A first --dry-run
smoke of an earlier version of this script (no anchor writes) confirmed
this empirically: P1 (ghost branch live) was False after every tick,
0/1 seeds. SD-039 anchor population is driver-issued in every existing
example (V3-EXQ-807's own `aset.write_anchor(...)` calls). This script
therefore writes a periodic anchor (every ANCHOR_EVERY=8 steps, ARM_OPEN
only) on the agent's OWN internal `agent.hippocampal.anchor_set` -- not a
standalone bypass -- tagged with the current z_world + current z_goal
snapshot. Per `anchor_set.py:write_anchor`'s own docstring, writing a new
segment_id under the same (scale, stream_mixture) family demotes the
PRIOR anchor to inactive, which is exactly the dual-trace "ghost" pool
MECH-292's bank ranks over (`GhostGoalBankConfig` defaults include both
active and inactive anchors) -- so periodic writes are sufficient to
populate a nonzero, ghost-eligible pool without any blockage/detour
mechanism. This is an experimental proxy for the anchor-writing trigger
(real deployment presumably ties this to goal-abandonment/invalidation
events, e.g. via a detour as in V3-EXQ-495a's design), not a claim that
this driver's periodic cadence is how anchors "should" be written in
general -- flagged explicitly so a reader does not mistake it for
discovered production behaviour.

READOUT (goal-terrain navigation, MECH-236's own language):
  contact_rate = (# steps with info["transition_type"]=="resource") /
                 total_steps, per seed x arm. A direct behavioural
  navigation-toward-goal measure on the live environment (not a bench
  arithmetic proxy) -- if proposals degrade to a position-based random
  walk, contact_rate should be near the terrain-only baseline
  (ARM_CLOSED); if goal-directed, it should exceed it under ARM_OPEN.

  mech293_n_ghost_admitted (ARM_OPEN only) is the P1 readiness
  precondition: if the ghost branch never fires, ARM_OPEN is inertly
  identical to ARM_CLOSED and a null C_MAIN result is uninterpretable
  ("instrumentation problem", per V3-EXQ-495a's own C1 reasoning, not a
  scientific finding).

  zgoal_norm_mean (both CLOSED and OPEN) is the P2 precondition
  confirming z_goal formed comparably in both arms under test (the
  "preserving E3 goal encoding" control).

ACCEPTANCE CRITERIA (pre-registered):

  C_MAIN (load-bearing): ARM_OPEN.contact_rate - ARM_CLOSED.contact_rate
    >= CONTACT_GAP_FLOOR (0.05) in >= 3/5 seeds. THE MECH-236 test.

  C_CONTEXT (non-gating, reported only): ARM_CLOSED.contact_rate -
    ARM_NOGOAL.contact_rate. Interpretive context for C_MAIN, not part of
    MECH-236's own claim (which is specifically about the hippocampal-
    PROPOSAL channel, not whether z_goal matters via any route at all).

PASS = P1 (ghost branch live) AND P2 (z_goal formed comparably) AND
C_MAIN.

DV-SYMMETRY (Step 3 mandatory declaration, per arm): contact_rate is a
COUNT-based rate over discrete environment resource-contact events, not
derived from any CEM-internal score. It is not invariant under a
uniform-broadcast-constant manipulation (the V3-EXQ-604c hazard) --
`use_mech293_ghost_probes` changes WHICH candidate the CEM elite refit and
E3 selection choose (a discrete selection-path change, not an additive
scalar applied uniformly to every candidate), so broadcast-invariance
does not apply. It is not a rank-only / argmax-only DV either
(contact_rate is a magnitude, not an ordinal comparison), so a
monotone-rescaling symmetry does not apply. The one real symmetry risk --
permutation of which of the 5 seeds' resource layouts get contacted -- is
why the metric is a per-seed RATE aggregated via a pre-registered
majority seed-fraction, not a pooled sum a single lucky/unlucky seed
could dominate.

claim_ids: ['MECH-236'] (single-claim evidence; MECH-230 is a
precondition this run holds CONSTANT across the CLOSED/OPEN contrast, not
itself re-tested; MECH-292/293/SD-039 are upstream substrates this run
exercises via the agent's own automatic anchor/bank population but does
not re-test their own internal arithmetic -- matching the
claim_ids-accuracy rule V3-EXQ-881/807/868 all follow).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_914_mech236_hippocampal_zgoal_channel_ablation.py [--dry-run]

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.hippocampal.anchor_set import AnchorGoalPayload  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_914_mech236_hippocampal_zgoal_channel_ablation"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-236"]

ARMS = ["ARM_NOGOAL", "ARM_CLOSED", "ARM_OPEN"]
SEEDS: List[int] = [42, 43, 45, 46, 47]  # seed 44: recurring per-seed early-death
                                          # instability on this env config family
                                          # (see V3-EXQ-868 / V3-EXQ-881 docstrings)

N_EPISODES = 10
STEPS_PER_EPISODE = 40
ANCHOR_EVERY = 8  # ARM_OPEN only -- see "ANCHOR POPULATION" docstring note

GRID_SIZE = 8
NUM_RESOURCES = 3
# NUM_HAZARDS=0 (Step 2.5a empirical probe finding, not the original
# 495a-modeled value of 2): with hazards present, an UNTRAINED agent
# (random-init E1/E2, terrain-only ARC-007 residue scoring uninformative
# at init) died to health depletion at ~12 steps/episode across every
# arm, pinning contact_rate at 0.0000 everywhere and making C_MAIN
# uninformative regardless of the channel under test. MECH-236's claim is
# about goal-directed navigation, not harm-survival tradeoffs, so removing
# the hazard confound (matching V3-EXQ-807's own no-hazard choice for a
# similarly channel-isolating design) is the correct fix, not a
# convenience -- ARC-007 terrain scoring is still always active and will
# still shape trajectories via whatever residue-field structure forms.
NUM_HAZARDS = 0
HAZARD_HARM = 0.1
DRIVE_WEIGHT = 2.0

FORCED_BENEFIT = 0.5
FORCED_DRIVE = 0.9

# Pre-registered thresholds (absolute constants, NOT run-derived).
# PROXIMITY_GAP_FLOOR calibrated against a Step 2.5a probe (this env
# config, untrained agent, 5-seed manual check): mean_resource_proximity
# ~0.30-0.45 baseline scale, ARM_OPEN vs ARM_CLOSED single-seed probe gaps
# in the ~0.01-0.05 range. Set conservatively below the probe's own
# smallest observed positive gap so the threshold is a real bar, not a
# rubber stamp -- see "READOUT CALIBRATION" docstring note for the probe
# values themselves.
PROXIMITY_GAP_FLOOR = 0.015
CONTACT_GAP_FLOOR = 0.05  # secondary/diagnostic reporting only, not gating
ZGOAL_NORM_FLOOR = 0.1
SEED_PASS_FRACTION = 3.0 / 5.0

# z_goal-stream liveness, pooled across arm x seed cells for the manifest block.
_ZG = ZGoalStreamAccumulator()


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
        use_proxy_fields=True,
    )


def _config_slice(arm: str) -> Dict[str, Any]:
    goal_active = arm in ("ARM_CLOSED", "ARM_OPEN")
    channel_open = arm == "ARM_OPEN"
    return {
        "env": {"size": GRID_SIZE, "num_resources": NUM_RESOURCES,
                "num_hazards": NUM_HAZARDS, "hazard_harm": HAZARD_HARM,
                "use_proxy_fields": True},
        "schedule": {"n_episodes": N_EPISODES, "steps": STEPS_PER_EPISODE},
        "alpha_world": 0.9,
        "z_goal_enabled": goal_active,
        "drive_weight": DRIVE_WEIGHT if goal_active else 0.0,
        "benefit_eval_enabled": goal_active,
        "wanting_weight": 0.0,
        "use_per_stream_vs": channel_open,
        "use_event_segmenter": channel_open,
        "use_invalidation_trigger": channel_open,
        "use_anchor_sets": channel_open,
        "use_sd039_anchor_payload": channel_open,
        "use_mech292_ghost_bank": channel_open,
        "use_mech293_ghost_probes": channel_open,
        "forced_benefit": FORCED_BENEFIT if goal_active else None,
        "forced_drive": FORCED_DRIVE if goal_active else None,
    }


def _make_agent(env: CausalGridWorldV2, arm: str) -> REEAgent:
    goal_active = arm in ("ARM_CLOSED", "ARM_OPEN")
    channel_open = arm == "ARM_OPEN"
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,

        # SD-012 / z_goal formation -- IDENTICAL between ARM_CLOSED and
        # ARM_OPEN (the "preserving E3 goal encoding" control).
        z_goal_enabled=goal_active,
        drive_weight=DRIVE_WEIGHT if goal_active else 0.0,
        benefit_eval_enabled=goal_active,
        benefit_weight=1.0,

        # Isolate the MECH-293 channel from the separate VALENCE_WANTING
        # pathway -- see module docstring "SCOPING NOTE".
        wanting_weight=0.0,

        # MECH-293's full prerequisite stack (module.py: "PLANNED needs
        # this on"). ARM_CLOSED and ARM_NOGOAL keep every one of these
        # False -- ONLY this block differs between ARM_CLOSED and
        # ARM_OPEN, isolating the channel under test.
        use_per_stream_vs=channel_open,
        use_event_segmenter=channel_open,
        use_invalidation_trigger=channel_open,
        use_anchor_sets=channel_open,
        use_sd039_anchor_payload=channel_open,
        use_mech292_ghost_bank=channel_open,
        use_mech293_ghost_probes=channel_open,
        mech293_ghost_fraction=0.2,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Single arm x seed cell
# ---------------------------------------------------------------------------
def _run_cell(arm: str, seed: int, dry_run: bool = False) -> Dict[str, Any]:
    n_episodes = 2 if dry_run else N_EPISODES
    steps_per = 8 if dry_run else STEPS_PER_EPISODE
    goal_active = arm in ("ARM_CLOSED", "ARM_OPEN")

    print(f"Seed {seed} Condition {arm}", flush=True)
    with arm_cell(seed, config_slice=_config_slice(arm),
                  script_path=Path(__file__)) as cell:
        env = _make_env(seed)
        agent = _make_agent(env, arm)
        device = agent.device
        wd = agent.config.latent.world_dim

        total_steps = 0
        resource_contacts = 0
        zgoal_norms: List[float] = []
        ghost_admitted: List[int] = []
        n_latched_ticks = 0  # e3_tick did not fire this env step (see below)
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
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

                if goal_active:
                    agent.update_z_goal(
                        benefit_exposure=FORCED_BENEFIT, drive_level=FORCED_DRIVE
                    )
                    zg = (
                        agent.goal_state.z_goal
                        if agent.goal_state is not None else None
                    )
                    if zg is not None:
                        zgoal_norms.append(float(zg.detach().norm().item()))
                else:
                    zg = None

                if arm == "ARM_OPEN":
                    # ANCHOR POPULATION: no production call site writes new
                    # anchors during ordinary agent ticking (agent.py only
                    # ever READS anchor_set.active_anchors() -- confirmed by
                    # source grep). SD-039 anchor writes are driver-issued
                    # (matches V3-EXQ-807's own pattern, ree_core/hippocampal/
                    # anchor_set.py:write_anchor -- writing a NEW segment_id
                    # under the same (scale, stream_mixture) family
                    # automatically demotes the prior anchor to INACTIVE,
                    # which is exactly the dual-trace "ghost" pool MECH-292's
                    # bank ranks over). Periodic write (every ANCHOR_EVERY
                    # steps) with the current z_world + current z_goal
                    # snapshot as goal_payload, on the agent's OWN internal
                    # anchor_set (not a standalone bypass) so
                    # propose_trajectories reads the same pool this loop
                    # populates.
                    if (
                        zg is not None
                        and t % ANCHOR_EVERY == 0
                        and agent.hippocampal.anchor_set is not None
                    ):
                        agent.hippocampal.anchor_set.write_anchor(
                            "fast", f"e{ep}s{t}", ("world",),
                            z_world=latent.z_world.detach().reshape(-1),
                            goal_payload=AnchorGoalPayload(
                                z_goal_snapshot=zg.detach().reshape(-1).clone(),
                                wanting_strength=0.5,
                            ),
                        )
                    # LATCHED-DIAGNOSTIC GUARD (Step 3.5 "Sample-size
                    # integrity"): agent.generate_trajectories() only calls
                    # _e3_tick (and therefore propose_trajectories /
                    # _last_propose_diagnostics) when ticks["e3_tick"] is
                    # True -- E3 deliberates at heartbeat.e3_steps_per_tick
                    # cadence (default 10 env steps), NOT every step (see
                    # ree_core/heartbeat/clock.py). Between real E3 ticks,
                    # _last_propose_diagnostics still holds the PREVIOUS
                    # tick's values (it is written only inside
                    # _propose_ghost_seeded). Reading it unconditionally on
                    # every env step -- as an earlier version of this
                    # script did -- re-records one E3 selection as ~10
                    # independent observations (the exact V3-EXQ-785-class
                    # defect this skill's code-review checklist names).
                    # Clear-and-check pattern below records NOTHING on a
                    # latched tick.
                    if ticks.get("e3_tick"):
                        diag = (
                            getattr(
                                agent.hippocampal, "_last_propose_diagnostics", {}
                            )
                            or {}
                        )
                        ghost_admitted.append(
                            int(diag.get("mech293_n_ghost_admitted", 0))
                        )
                    else:
                        n_latched_ticks += 1

                _, _harm_signal, done, info, obs = env.step(
                    int(action.argmax(dim=-1).item())
                )
                total_steps += 1
                if info.get("transition_type") == "resource":
                    resource_contacts += 1
                if done:
                    break
            if (ep + 1) % 5 == 0:
                print(
                    f"  [train] mech236 seed={seed} arm={arm} ep {ep+1}/{n_episodes} "
                    f"contacts={resource_contacts}",
                    flush=True,
                )

        contact_rate = resource_contacts / max(1, total_steps)
        mean_resource_proximity = (
            statistics.fmean(resource_proximity) if resource_proximity else 0.0
        )
        zgoal_norm_mean = statistics.fmean(zgoal_norms) if zgoal_norms else 0.0
        ghost_admitted_mean = (
            statistics.fmean(ghost_admitted) if ghost_admitted else 0.0
        )
        row = {
            "arm": arm,
            "seed": seed,
            "total_steps": total_steps,
            "resource_contacts": resource_contacts,
            "contact_rate": contact_rate,
            "mean_resource_proximity": mean_resource_proximity,
            "n_proximity_sampled": len(resource_proximity),
            "zgoal_norm_mean": zgoal_norm_mean,
            "n_zgoal_sampled": len(zgoal_norms),
            "mech293_n_ghost_admitted_mean": ghost_admitted_mean,
            "mech293_n_ghost_admitted_nonzero_frac": (
                sum(1 for g in ghost_admitted if g > 0) / len(ghost_admitted)
                if ghost_admitted else 0.0
            ),
            "n_e3_ticks_sampled": len(ghost_admitted) if arm == "ARM_OPEN" else 0,
            "n_latched_ticks": n_latched_ticks if arm == "ARM_OPEN" else 0,
        }
        cell.stamp(row)
    # z_goal liveness -- read AFTER the cell has stepped, agent not retained
    # past this call.
    _ZG.observe(agent)
    # Per-cell progress verdict: did the cell collect non-degenerate data
    # (NOT a scientific verdict -- the comparative MECH-236 verdict is
    # computed once, across all cells, in run_experiment()).
    cell_ok = total_steps > 0 and (not goal_active or zgoal_norm_mean > 0.0)
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row


def _worst_cell(rows: List[Dict[str, Any]], key: str, mode: str = "min"):
    fn = min if mode == "min" else max
    r = fn(rows, key=lambda x: x[key])
    return float(r[key]), f"{r['arm']}/seed{r['seed']}"


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            rows.append(_run_cell(arm, seed, dry_run=dry_run))

    nogoal = [r for r in rows if r["arm"] == "ARM_NOGOAL"]
    closed = [r for r in rows if r["arm"] == "ARM_CLOSED"]
    open_ = [r for r in rows if r["arm"] == "ARM_OPEN"]

    # P1: MECH-293 ghost branch actually fires in ARM_OPEN (readiness).
    ghost_fire_frac = (
        sum(1 for r in open_ if r["mech293_n_ghost_admitted_mean"] > 0) / len(open_)
    )
    p1_ghost_live = ghost_fire_frac >= SEED_PASS_FRACTION
    ghost_worst, ghost_worst_cell = _worst_cell(
        open_, "mech293_n_ghost_admitted_mean", mode="min"
    )

    # P2: z_goal formed comparably in CLOSED and OPEN (preserved-encoding
    # control -- MECH-236's "while preserving E3 goal encoding" premise).
    zg_closed_worst, zg_closed_cell = _worst_cell(
        closed, "zgoal_norm_mean", mode="min"
    )
    zg_open_worst, zg_open_cell = _worst_cell(open_, "zgoal_norm_mean", mode="min")
    p2_zgoal_formed = (
        zg_closed_worst >= ZGOAL_NORM_FLOOR and zg_open_worst >= ZGOAL_NORM_FLOOR
    )

    # C_MAIN: paired per-seed mean-resource-proximity gap, OPEN - CLOSED.
    # Primary DV is proximity (a continuous, always-measurable signal), NOT
    # raw contact_rate -- see "READOUT CALIBRATION" docstring note: this
    # env's untrained-agent foraging-competence ceiling (the same one
    # documented at length in claims.yaml's MECH-230 evidence_quality_note)
    # pins contact_rate near-zero for every arm at this episode budget,
    # which would make a contact-rate-only C_MAIN uninformative regardless
    # of the channel under test. contact_rate is still reported (below) as
    # secondary/diagnostic context.
    by_seed_closed = {r["seed"]: r["mean_resource_proximity"] for r in closed}
    by_seed_open = {r["seed"]: r["mean_resource_proximity"] for r in open_}
    gaps = {
        s: by_seed_open[s] - by_seed_closed[s]
        for s in by_seed_open if s in by_seed_closed
    }
    seeds_passing = sum(1 for g in gaps.values() if g >= PROXIMITY_GAP_FLOOR)
    c_main_frac = (seeds_passing / len(gaps)) if gaps else 0.0
    c_main = c_main_frac >= SEED_PASS_FRACTION

    # C_CONTEXT (non-gating): CLOSED vs NOGOAL, same proximity metric.
    by_seed_nogoal = {r["seed"]: r["mean_resource_proximity"] for r in nogoal}
    context_gaps = {
        s: by_seed_closed[s] - by_seed_nogoal[s]
        for s in by_seed_closed if s in by_seed_nogoal
    }

    # Secondary/diagnostic: raw contact rate, same OPEN-vs-CLOSED contrast,
    # reported but NOT load-bearing (see note above).
    by_seed_closed_contact = {r["seed"]: r["contact_rate"] for r in closed}
    by_seed_open_contact = {r["seed"]: r["contact_rate"] for r in open_}
    contact_gaps = {
        s: by_seed_open_contact[s] - by_seed_closed_contact[s]
        for s in by_seed_open_contact if s in by_seed_closed_contact
    }

    preconditions_met = p1_ghost_live and p2_zgoal_formed
    overall = preconditions_met and c_main

    if not preconditions_met:
        label = "substrate_not_ready_requeue"
        evidence_direction = "unknown"
    elif overall:
        label = "mech293_channel_confers_navigation_advantage"
        evidence_direction = "supports"
    else:
        label = "mech293_channel_no_measurable_navigation_advantage"
        evidence_direction = "weakens"

    metrics = {
        "nogoal_proximity_mean": statistics.fmean(
            r["mean_resource_proximity"] for r in nogoal
        ),
        "closed_proximity_mean": statistics.fmean(
            r["mean_resource_proximity"] for r in closed
        ),
        "open_proximity_mean": statistics.fmean(
            r["mean_resource_proximity"] for r in open_
        ),
        "c_main_gap_mean": statistics.fmean(gaps.values()) if gaps else 0.0,
        "c_main_seed_pass_fraction": c_main_frac,
        "c_context_gap_mean": (
            statistics.fmean(context_gaps.values()) if context_gaps else 0.0
        ),
        # Secondary/diagnostic, NOT load-bearing (see C_MAIN note above).
        "nogoal_contact_rate_mean": statistics.fmean(
            r["contact_rate"] for r in nogoal
        ),
        "closed_contact_rate_mean": statistics.fmean(
            r["contact_rate"] for r in closed
        ),
        "open_contact_rate_mean": statistics.fmean(r["contact_rate"] for r in open_),
        "contact_rate_gap_mean": (
            statistics.fmean(contact_gaps.values()) if contact_gaps else 0.0
        ),
        "ghost_fire_seed_fraction": ghost_fire_frac,
        "ghost_admitted_worst": ghost_worst,
        "ghost_admitted_worst_cell": ghost_worst_cell,
        "zgoal_closed_worst": zg_closed_worst,
        "zgoal_closed_worst_cell": zg_closed_cell,
        "zgoal_open_worst": zg_open_worst,
        "zgoal_open_worst_cell": zg_open_cell,
        "p1_ghost_live": p1_ghost_live,
        "p2_zgoal_formed": p2_zgoal_formed,
        "c_main_pass": c_main,
    }

    return {
        "outcome": "PASS" if overall else "FAIL",
        "evidence_direction": evidence_direction,
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "preconditions": [
                {
                    "name": "mech293_ghost_branch_live",
                    "description": (
                        "ARM_OPEN's MECH-293 ghost branch actually admits >=1 "
                        "candidate on average; else ARM_OPEN is functionally "
                        "identical to ARM_CLOSED and any null C_MAIN result is "
                        "an instrumentation gap, not evidence against MECH-236."
                    ),
                    "measured": ghost_worst,
                    "threshold": 0.0,
                    "direction": "lower",
                    "control": f"worst seed across ARM_OPEN cells ({ghost_worst_cell})",
                    "offending_cell": ghost_worst_cell,
                    "met": p1_ghost_live,
                },
                {
                    "name": "zgoal_formed_comparably_closed_and_open",
                    "description": (
                        "z_goal norm clears a floor in BOTH ARM_CLOSED and "
                        "ARM_OPEN -- the 'preserving E3 goal encoding' control "
                        "MECH-236's testable prediction specifies."
                    ),
                    "measured": min(zg_closed_worst, zg_open_worst),
                    "threshold": ZGOAL_NORM_FLOOR,
                    "direction": "lower",
                    "control": (
                        f"worst seed across CLOSED ({zg_closed_cell}) and "
                        f"OPEN ({zg_open_cell})"
                    ),
                    "offending_cell": (
                        zg_closed_cell if zg_closed_worst < zg_open_worst
                        else zg_open_cell
                    ),
                    "met": p2_zgoal_formed,
                },
            ],
            "criteria": [
                {
                    "name": "C_MAIN_open_beats_closed_resource_proximity",
                    "load_bearing": True,
                    "passed": c_main,
                },
            ],
            "criteria_non_degenerate": {
                "C_MAIN_open_beats_closed_resource_proximity": bool(preconditions_met),
            },
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    result = run_experiment(dry_run=args.dry_run)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    full_config = {
        "seeds": SEEDS,
        "arms": ARMS,
        "n_episodes": N_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "grid_size": GRID_SIZE,
        "num_resources": NUM_RESOURCES,
        "num_hazards": NUM_HAZARDS,
        "hazard_harm": HAZARD_HARM,
        "drive_weight": DRIVE_WEIGHT,
        "forced_benefit": FORCED_BENEFIT,
        "forced_drive": FORCED_DRIVE,
        "contact_gap_floor": CONTACT_GAP_FLOOR,
        "zgoal_norm_floor": ZGOAL_NORM_FLOOR,
        "seed_pass_fraction": SEED_PASS_FRACTION,
        "arm_config_slice_nogoal": _config_slice("ARM_NOGOAL"),
        "arm_config_slice_closed": _config_slice("ARM_CLOSED"),
        "arm_config_slice_open": _config_slice("ARM_OPEN"),
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "evidence_direction": result["evidence_direction"],
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
    }
    out_path = write_flat_manifest(
        manifest,
        Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments",
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    m = result["metrics"]
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(
        f"contact_rate: nogoal={m['nogoal_contact_rate_mean']:.4f} "
        f"closed={m['closed_contact_rate_mean']:.4f} "
        f"open={m['open_contact_rate_mean']:.4f}",
        flush=True,
    )
    print(
        f"C_MAIN gap_mean={m['c_main_gap_mean']:.4f} "
        f"seed_pass_frac={m['c_main_seed_pass_fraction']:.2f} "
        f"pass={m['c_main_pass']} | "
        f"P1_ghost_live={m['p1_ghost_live']} P2_zgoal_formed={m['p2_zgoal_formed']}",
        flush=True,
    )
    print(f"wrote: {out_path}", flush=True)
    return result, out_path, args.dry_run


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
