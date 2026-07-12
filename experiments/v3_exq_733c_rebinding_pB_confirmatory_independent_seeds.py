#!/opt/local/bin/python3
"""
V3-EXQ-733c -- MECH-456 functional-rebinding, P-B directed-traversal CONFIRMATORY
RETEST on an INDEPENDENT disjoint seed-set (does NOT supersede V3-EXQ-733b).

WHAT THIS IS. A verbatim re-run of the V3-EXQ-733b Leg P-B directed-traversal
design (the coverage-decoupled harness, directed_tour_p0_coverage=True) with ONE
scientific change: an INDEPENDENT, disjoint seed-set. 733b established MECH-456's
FIRST functional rebinding demonstration on seeds [42..47]; this run re-runs the
identical mechanism + criteria on seeds [101..106] so the support is not resting
on a single seed-set / single pathway. The binder mechanism, the test-bed, the
coverage-decoupling harness, and every pre-registered acceptance criterion are
bit-identical to 733b -- ONLY the seeds (and the run identity) change.

WHY THIS IS A CONFIRMATORY RETEST (the provisional -> active progression). MECH-456
was PROMOTED candidate -> provisional (v3_pending lifted) on 2026-07-12 via
/governance IGW-20260712-001 on the strength of the SINGLE V3-EXQ-733b PASS. The
MECH-456 evidence_quality_note explicitly records: "Positive base is a single PASS
on the directed-traversal design; a confirmatory retest on an independent design /
seed-set would strengthen but is not gating for provisional." THIS run is that
confirmatory retest -- a 2nd clean PASS on an independent seed-set strengthens the
provisional reading toward active; a FAIL routes /failure-autopsy (it does NOT
auto-demote a provisional claim). It is NOT urgent / NOT gating: MECH-456 is
already provisional.

RE-DERIVE BRAKE (released, for the audit trail). The brake FIRED on MECH-456 (2
non_contributory readiness autopsies: failure_autopsy_V3-EXQ-733_2026-07-10,
failure_autopsy_V3-EXQ-733a_2026-07-11). It is RELEASED here for two independent
reasons: (1) per /queue-experiment Step 2.5b rule 1, the named upstream substrate
(rebinding-harness-p0-coverage-decoupling, directed_tour_p0_coverage; ree-v3 main
commit 9cd896c) is now BUILT; and (2) the ceiling those autopsies circled was
already LIFTED -- V3-EXQ-733b PASSed on this exact substrate and promoted the
claim. The brake exists to refuse re-tests of a persistently-failing ceiling; this
is the confirmatory strengthening run of a PASSing result, not a same-ceiling
re-queue. This run does NOT supersede V3-EXQ-733b: 733b's PASS remains valid
evidence and this run ADDS an independent replication.

THE P-B DESIGN (verbatim from V3-EXQ-733b / 733a Leg P-B -- unchanged):
  The COLD V3-EXQ-733 agent (no onboarding -- isolates the test-bed axis), but the
  TEST-BED makes overtakes guaranteed:
  (1) SOFTENED SURVIVAL PRESSURE -- hazard_harm 0.05 -> 0.0, proximity_harm_scale
      0.1 -> 0.0, hazard_food_attraction 0.7 -> 0.0. Hazards stay present (region
      z_world structure) but non-lethal.
  (2) FINER LATTICE -- G_PARTITION 2 -> 4 (K=16 regions): more boundary crossings
      per unit movement, and finer prototypes.
  (3) DIRECTED TRAVERSAL -- a scripted region tour: the agent is directed-respawned
      (teleported, obs refreshed) to the next lattice region every TRAVERSE_PERIOD
      steps. Every teleport is a GUARANTEED ground-truth overtake and a guaranteed
      region visit. Between teleports the agent still senses/acts naturally, so
      z_self/z_world are the agent's genuine latents; the teleport only sets WHICH
      region is currently true (the competing-configuration overtake).
  (4) COVERAGE DECOUPLING -- directed_tour_p0_coverage=True suppresses the P0
      health-death `done` truncation so the scripted tour runs its full step budget
      and per-region P0 coverage is guaranteed BY CONSTRUCTION (the 733b substrate
      fix; the SINGLE lever that let 733a's P-B near-pass be counted).

INDEPENDENCE (the sole scientific change vs 733b). SEEDS = [101..106], disjoint
from 733b's [42..47]. Each seed independently seeds torch (agent init + P0
binder-training stochasticity) and the env (layout / drift), so each is a genuine
independent binder-convergence + latent-trajectory replication -- NOT a re-run of
identical computation. A PASS on 6 fresh seeds shows the 733b support is not a
seed-set artefact.

NULL (declared): with overtakes GUARANTEED and P0 coverage guaranteed by
construction, if DV1/DV2 fail to clear on >= MIN_SEEDS_FOR_PASS seeds on this
INDEPENDENT seed-set -> the binder does NOT robustly track the true competitor /
confers no graded advantage across seed-sets (the 733b PASS did not replicate) =>
WEAKENS MECH-456 (routes /failure-autopsy; does NOT auto-demote provisional). If
the binder does not converge / readiness is somehow short -> substrate_not_ready_
requeue (non_contributory), NOT a verdict.

WHAT THIS TESTS (MECH-456 what_would_answer, via the shared 733 harness):
  DV1: rebinding tracks the TRUE current region above a region-label-shuffle control.
  DV2: rebinding-ON re-acquires the correct binding FASTER than a rebinding-FROZEN
       arm (graded, non-saturating censored latency) over ONE shared trajectory/seed.
RUN PASS = DV1 AND DV2 on >= MIN_SEEDS_FOR_PASS seeds with readiness met. Expected:
functional_rebinding_supported -> counted SUPPORT for MECH-456 (2nd independent
PASS). MECH-456 is already provisional; a confirmatory PASS strengthens toward
active but PROMOTES NOTHING by itself.

NO ree_core change -- harness-level measurement over the trained-then-frozen learned
cross_stream_binder. The teleport is a harness-level directed-respawn using the
env's public agent_x/agent_y + _get_observation_dict() (a pure read); it does not
mutate env step/health state. Env / agent config / measurement primitives shared
with 733/733a/733b via experiments/_lib/rebinding_functional_harness.py.

Claims: [MECH-456] (experiment_purpose=evidence; claim-tagged functional test).
Bears on (cited, NOT tagged): MECH-269, MECH-270, ARC-006, MECH-045, INV-002.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiment_protocol import emit_outcome
from experiments._metrics import check_degeneracy
from experiments._lib import rebinding_functional_harness as H
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_733c_rebinding_pB_confirmatory_independent_seeds"
QUEUE_ID = "V3-EXQ-733c"
CLAIM_IDS: List[str] = ["MECH-456"]
EXPERIMENT_PURPOSE = "evidence"
# Confirmatory retest: does NOT supersede V3-EXQ-733b (733b's PASS stays valid;
# this ADDS an independent-seed replication).
SUPERSEDES: Optional[str] = None

# Converged learned cross_stream_binder activation (identical to V3-EXQ-733/733a/733b).
CROSS_STREAM_BINDING_ENABLED = True
CROSS_STREAM_BINDING_LEARNED = True
CROSS_STREAM_BINDING_STRENGTH = 0.5
CROSS_STREAM_BINDING_TEMPERATURE = 0.2
CROSS_STREAM_BINDING_CONV_FRAC = 0.85

# FINER lattice (test-bed axis): G 2 -> 4 (K=16 regions). Identical to 733b.
SIZE = 12
G_PARTITION = 4
K_REGIONS = G_PARTITION * G_PARTITION

# Directed traversal cadence: teleport to the next region every TRAVERSE_PERIOD
# steps. Identical to 733b.
TRAVERSE_PERIOD = 8

# INDEPENDENT disjoint seed-set (the sole scientific change vs 733b's [42..47]).
SEEDS = [101, 102, 103, 104, 105, 106]
P0_WARMUP_EPISODES = 40
P1_MEASUREMENT_EPISODES = 25
STEPS_PER_EPISODE = 120

DRY_RUN_SEEDS = [101]
DRY_RUN_P0 = 3
DRY_RUN_P1 = 3
DRY_RUN_STEPS = 40

# 733 reef-bipartite SD-054 env, SOFTENED so survival never truncates traversal.
# Bit-identical to 733b (only survival-pressure kwargs differ from V3-EXQ-733:
# hazard_harm 0.05->0.0, proximity_harm_scale 0.1->0.0, hazard_food_attraction
# 0.7->0.0); the structural / dim-affecting kwargs are UNCHANGED, so the agent
# config is bit-identical to 733/733b.
ENV_KWARGS = dict(
    size=SIZE,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.0,             # SOFTENED (733: 0.05) -- hazards present but non-lethal
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.0,    # SOFTENED (733: 0.1) -- no proximity harm
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.0,  # SOFTENED (733: 0.7) -- hazards do not chase food
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """The COLD V3-EXQ-733 agent, VERBATIM (no onboarding stack). Isolating the
    test-bed axis means the agent is bit-identical to 733/733b -- only the seeds
    change vs 733b."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        z_harm_dim=32,
        use_affective_harm_stream=True,
        z_harm_a_dim=16,
        harm_history_len=10,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        use_noise_floor=True,
        noise_floor_alpha=0.1,
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        cross_stream_binding_enabled=CROSS_STREAM_BINDING_ENABLED,
        cross_stream_binding_learned=CROSS_STREAM_BINDING_LEARNED,
        cross_stream_binding_strength=CROSS_STREAM_BINDING_STRENGTH,
        cross_stream_binding_temperature=CROSS_STREAM_BINDING_TEMPERATURE,
        cross_stream_binding_conv_frac=CROSS_STREAM_BINDING_CONV_FRAC,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Directed traversal: teleport the agent across the G x G lattice on a fixed
# cadence to GENERATE ground-truth overtakes independent of foraging survival.
# (Verbatim from V3-EXQ-733b.)
# ---------------------------------------------------------------------------


def _region_center_cell(region_idx: int, size: int, g: int) -> Tuple[int, int]:
    """Interior cell at the centre of lattice region_idx. Clamped to [1, size-2]
    so the target is never a border wall (non-toroidal env). region_of() maps the
    returned cell back to region_idx by construction."""
    rx, ry = divmod(int(region_idx), int(g))
    cw = size / g
    x = int(min(size - 2, max(1, int((rx + 0.5) * cw))))
    y = int(min(size - 2, max(1, int((ry + 0.5) * cw))))
    return x, y


def _teleport(env: CausalGridWorldV2, x: int, y: int) -> Dict[str, Any]:
    """Directed-respawn: set the agent position and return a FRESH obs dict
    (_get_observation_dict is a pure read that rebuilds obs from agent_x/agent_y;
    it does NOT advance steps/health). Keeps the grid agent-marker consistent so
    the local_view agent channel is not polluted by a stale marker."""
    empty = env.ENTITY_TYPES["empty"]
    agent_t = env.ENTITY_TYPES["agent"]
    ox, oy = int(env.agent_x), int(env.agent_y)
    if 0 <= ox < env.size and 0 <= oy < env.size and env.grid[ox, oy] == agent_t:
        env.grid[ox, oy] = empty
    env.agent_x, env.agent_y = int(x), int(y)
    if env.grid[x, y] == empty:  # do not clobber a resource/hazard cell
        env.grid[x, y] = agent_t
    return env._get_observation_dict()


def _make_directors() -> Tuple[Callable, Callable]:
    """Per-seed scripted region tour. The tour index PERSISTS across episodes (P0
    and P1) so region coverage accumulates deterministically over the whole seed.
    spawn_director places the agent in the current tour region at episode entry;
    step_director advances the tour + teleports every TRAVERSE_PERIOD steps."""
    state = {"tour": 0}

    def spawn_director(env, obs, ep, is_p1, rng):
        x, y = _region_center_cell(state["tour"], SIZE, G_PARTITION)
        return _teleport(env, x, y)

    def step_director(env, obs, ep, step_idx, is_p1, rng):
        if (step_idx + 1) % TRAVERSE_PERIOD == 0:
            state["tour"] = (state["tour"] + 1) % K_REGIONS
            x, y = _region_center_cell(state["tour"], SIZE, G_PARTITION)
            return _teleport(env, x, y)
        return obs

    return spawn_director, step_director


def _run_seed(seed: int, p0_episodes: int, p1_episodes: int,
              steps_per_episode: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    probe_env = _make_env(seed)
    agent = _make_agent(probe_env)

    spawn_director, step_director = _make_directors()

    def env_factory(s: int) -> CausalGridWorldV2:
        env = _make_env(s)
        if (int(env.body_obs_dim) != int(probe_env.body_obs_dim)
                or int(env.world_obs_dim) != int(probe_env.world_obs_dim)):
            raise RuntimeError(f"env dim parity FAILED seed={s}")
        return env

    return H.run_functional_test_seed(
        seed=seed,
        agent=agent,
        env_factory=env_factory,
        g_partition=G_PARTITION,
        p0_episodes=p0_episodes,
        p1_episodes=p1_episodes,
        steps_per_episode=steps_per_episode,
        spawn_director=spawn_director,
        step_director=step_director,
        # Coverage-decoupling harness (V3-EXQ-733b substrate fix): decouple P0
        # prototype-coverage from foraging survival. Identical to 733b -- the
        # confirmatory retest holds the substrate fixed and only varies the seeds.
        directed_tour_p0_coverage=True,
    )


def run_experiment(seeds: List[int], p0_episodes: int, p1_episodes: int,
                   steps_per_episode: int, dry_run: bool) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    first = True
    for s in seeds:
        print(f"Seed {s} Condition rebinding_pB_confirmatory", flush=True)
        if first:
            print(
                f"  (func-test P0={p0_episodes} ep, P1={p1_episodes} ep, "
                f"steps={steps_per_episode}, K_regions={K_REGIONS}, "
                f"traverse_period={TRAVERSE_PERIOD}, align_margin={H.ALIGN_MARGIN}, "
                f"latency_margin={H.LATENCY_MARGIN}, "
                f"min_region_visits_floor={H.min_region_visits_floor(K_REGIONS)}, "
                f"directed_tour_p0_coverage=True, independent_seeds={seeds}, "
                f"dry_run={dry_run})",
                flush=True,
            )
            first = False
        row = _run_seed(s, p0_episodes, p1_episodes, steps_per_episode)
        rows.append(row)
        verdict = "PASS" if row["error_note"] is None else "FAIL"
        print(f"verdict: {verdict}", flush=True)

    interp = H.interpret(rows, conv_frac=CROSS_STREAM_BINDING_CONV_FRAC)
    ok = [r for r in rows if r["error_note"] is None]
    n_completed = len(ok)

    passed = bool(
        n_completed >= H.MIN_SEEDS_COMPLETED
        and interp["label"] == "functional_rebinding_supported"
    )

    degen = check_degeneracy({
        "alignment_margin": [r["alignment_margin_obs"] for r in ok] or [0.0, 0.0],
        "latency_gap": [r["latency_gap_obs"] for r in ok] or [0.0, 0.0],
    })

    return {
        "outcome": "PASS" if passed else "FAIL",
        "seeds": list(seeds),
        "n_completed": int(n_completed),
        "n_total_runs": int(len(seeds)),
        "min_seeds_completed": int(H.MIN_SEEDS_COMPLETED),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "steps_per_episode": int(steps_per_episode),
        "k_regions": int(K_REGIONS),
        "g_partition": int(G_PARTITION),
        "traverse_period": int(TRAVERSE_PERIOD),
        "directed_tour_p0_coverage": True,
        "confirmatory_retest_of": "V3-EXQ-733b",
        "independent_seed_set": list(seeds),
        "acceptance_thresholds": {
            "align_margin": float(H.ALIGN_MARGIN),
            "latency_margin": float(H.LATENCY_MARGIN),
            "reacq_window": int(H.REACQ_WINDOW),
            "n_shuffle_perms": int(H.N_SHUFFLE_PERMS),
            "min_seeds_for_pass": int(H.MIN_SEEDS_FOR_PASS),
            # K-SCALED readiness floor actually applied by the harness (identical
            # to 733b): min_region_visits_floor(K), NOT the K=4 constant.
            "min_region_visits_p0": int(H.min_region_visits_floor(K_REGIONS)),
            "min_region_visits_p0_floor": int(H.min_region_visits_floor(K_REGIONS)),
            "min_region_visits_p0_base_k4": int(H.MIN_REGION_VISITS_P0),
            "min_overtakes_p1": int(H.MIN_OVERTAKES_P1),
        },
        "binder_config": {
            "cross_stream_binding_enabled": bool(CROSS_STREAM_BINDING_ENABLED),
            "cross_stream_binding_learned": bool(CROSS_STREAM_BINDING_LEARNED),
            "cross_stream_binding_strength": float(CROSS_STREAM_BINDING_STRENGTH),
            "cross_stream_binding_temperature": float(CROSS_STREAM_BINDING_TEMPERATURE),
            "cross_stream_binding_conv_frac": float(CROSS_STREAM_BINDING_CONV_FRAC),
        },
        "env_kwargs": dict(ENV_KWARGS),
        "per_seed_results": rows,
        "interpretation": interp,
        "non_degenerate": degen["non_degenerate"],
        "degeneracy_reason": degen["degeneracy_reason"],
        "degenerate_metrics": degen["degenerate_metrics"],
    }


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    interp = result.get("interpretation", {})
    label = interp.get("label", "")
    direction, direction_note = H.evidence_direction(label)
    region_floor = int(H.min_region_visits_floor(K_REGIONS))  # K-scaled (identical to 733b)

    review_caveats = [
        "V3-EXQ-733c: CONFIRMATORY retest of the V3-EXQ-733b MECH-456 functional-"
        "rebinding P-B directed-traversal design on an INDEPENDENT disjoint seed-set "
        "([101..106] vs 733b's [42..47]). This DOES NOT supersede V3-EXQ-733b -- "
        "733b's PASS remains valid evidence and this run ADDS an independent "
        "replication. MECH-456 was promoted candidate -> provisional (v3_pending "
        "lifted) via /governance IGW-20260712-001 on the single 733b PASS; the "
        "evidence_quality_note flagged a confirmatory retest on an independent "
        "seed-set as strengthening-but-not-gating. A 2nd clean PASS strengthens "
        "provisional -> active; a FAIL routes /failure-autopsy (does NOT auto-demote "
        "a provisional claim).",
        "The mechanism, test-bed, coverage-decoupling harness and every pre-"
        "registered acceptance criterion are BIT-IDENTICAL to 733b: cold V3-EXQ-733 "
        "agent (no onboarding), softened survival (hazard_harm/proximity_harm/"
        "hazard_food_attraction -> 0), finer lattice (G 2 -> 4, K=16), scripted "
        "region tour teleporting every traverse_period steps, and "
        "directed_tour_p0_coverage=True (P0 coverage guaranteed by construction). "
        "The ONLY change is the seed-set; each seed independently seeds torch (agent "
        "init + P0 binder-training) and the env (layout/drift), so each is a genuine "
        "independent binder-convergence + latent-trajectory replication, NOT a re-run "
        "of identical computation. Readiness coverage floor is the granularity-scaled "
        f"min_region_visits_floor(K={K_REGIONS})={region_floor}.",
        "RE-DERIVE BRAKE (released, audit trail): the brake FIRED on MECH-456 (2 "
        "non_contributory readiness autopsies: failure_autopsy_V3-EXQ-733_2026-07-10, "
        "failure_autopsy_V3-EXQ-733a_2026-07-11) and is RELEASED for two independent "
        "reasons -- (1) Step 2.5b rule 1: the named upstream substrate (rebinding-"
        "harness-p0-coverage-decoupling, directed_tour_p0_coverage; ree-v3 main "
        "9cd896c) is BUILT; and (2) the ceiling those autopsies circled was already "
        "LIFTED by the V3-EXQ-733b PASS on this exact substrate. This is the "
        "confirmatory strengthening run of a PASSing result, not a same-ceiling "
        "re-queue.",
        "The teleport is a harness-level directed-respawn (env.agent_x/agent_y + "
        "_get_observation_dict(), a pure read); it does NOT advance env step/health "
        "state and does NOT trigger contact harm/consumption, so it generates a "
        "clean ground-truth overtake without foraging.",
        "GROUND TRUTH is the agent's 4x4 lattice region read off env.agent_x/"
        "agent_y -- binder-independent. DV1 (alignment_real vs alignment_shuffle) "
        "tests true-region tracking above a region-label-shuffle baseline; DV2 is a "
        "graded, non-saturating censored re-acquisition latency (REBIND-ON vs "
        "REBIND-FROZEN over ONE shared trajectory per seed).",
        "NULL declared: with overtakes AND P0 coverage guaranteed by construction, "
        f"DV1/DV2 failing on >= {H.MIN_SEEDS_FOR_PASS} of 6 INDEPENDENT seeds -> the "
        "binder does not ROBUSTLY track the true competitor / confers no graded "
        "advantage across seed-sets (the 733b PASS did not replicate) => WEAKENS "
        "MECH-456. rebinding_inert_off_equals_on (DV1 holds, DV2 fails) and "
        "rebinding_not_tracking_truth (DV1 fails) both WEAKEN; "
        "substrate_not_ready_requeue is non_contributory (not a verdict).",
    ]
    if not interp.get("all_ready", False):
        review_caveats.insert(
            0,
            "WARNING readiness gate UNMET on >=1 completed seed "
            f"(min_overtakes={interp.get('min_overtakes')} vs {H.MIN_OVERTAKES_P1}; "
            f"min_region_visits_p0={interp.get('min_region_visits_p0')} vs "
            f"K-scaled floor {region_floor}). For V3-EXQ-733c this should NOT happen "
            "(directed_tour_p0_coverage guarantees P0 coverage and the teleport "
            "guarantees overtakes) -- investigate the binder-convergence leg. Self-"
            "routes substrate_not_ready_requeue (non_contributory); NOT a verdict.",
        )

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": direction,
        "evidence_direction_note": direction_note,
        "evidence_direction_per_claim": {"MECH-456": direction},
        "non_degenerate": result.get("non_degenerate", True),
        "degeneracy_reason": result.get("degeneracy_reason", ""),
        "confirmatory_retest_of": "V3-EXQ-733b",
        "independent_seed_set": list(result.get("seeds", [])),
        "bears_on_not_tagged": [
            "MECH-269", "MECH-270", "ARC-006", "MECH-045", "INV-002",
        ],
        "review_caveats": review_caveats,
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "leg": "P-B_directed_traversal",
            "confirmatory_retest": True,
            "confirmatory_retest_of": "V3-EXQ-733b",
            "supersedes": None,
            "retest_after_substrate": True,
            "brake_permitted_retest": True,
            "brake_released_reasons": [
                "step_2.5b_rule1_upstream_substrate_built",
                "ceiling_already_lifted_by_V3-EXQ-733b_PASS",
            ],
            "substrate_fix_commit": "9cd896c",
            "braked_autopsies": [
                "failure_autopsy_V3-EXQ-733_2026-07-10",
                "failure_autopsy_V3-EXQ-733a_2026-07-11",
            ],
            "directed_tour_p0_coverage": True,
            "min_region_visits_p0_floor": region_floor,
            "g_partition": int(G_PARTITION),
            "traverse_period": int(TRAVERSE_PERIOD),
            "harness_level_measurement": True,
            "ree_core_modified": False,
            "cross_stream_binding_enabled": bool(CROSS_STREAM_BINDING_ENABLED),
            "cross_stream_binding_learned": bool(CROSS_STREAM_BINDING_LEARNED),
        },
        "result": result,
    }
    if SUPERSEDES is not None:
        manifest["supersedes"] = SUPERSEDES
    return manifest


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Short smoke-test (1 seed, 3+3 ep, 40 steps).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override output dir (default: REE_assembly evidence/experiments).")
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0, p1, steps = DRY_RUN_P0, DRY_RUN_P1, DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0, p1, steps = P0_WARMUP_EPISODES, P1_MEASUREMENT_EPISODES, STEPS_PER_EPISODE

    result = run_experiment(seeds=seeds, p0_episodes=p0, p1_episodes=p1,
                            steps_per_episode=steps, dry_run=bool(args.dry_run))

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly" / "evidence" / "experiments"
        )
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config") or manifest.get("config_summary"),
        seeds=SEEDS,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
    )

    print(f"manifest: {out_path}", flush=True)
    print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} label={result['interpretation']['label']} "
        f"DV1={result['interpretation']['n_DV1']}/{result['interpretation']['n_completed']} "
        f"DV2={result['interpretation']['n_DV2']}/{result['interpretation']['n_completed']} "
        f"(need {H.MIN_SEEDS_FOR_PASS}) min_overtakes={result['interpretation'].get('min_overtakes')} "
        f"min_region_visits_p0={result['interpretation'].get('min_region_visits_p0')} "
        f"(floor {H.min_region_visits_floor(K_REGIONS)}) "
        f"direction={manifest['evidence_direction']}",
        flush=True,
    )

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    return outcome_emit, str(out_path), bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry)
    sys.exit(0)
