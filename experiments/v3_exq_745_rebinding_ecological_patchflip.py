#!/opt/local/bin/python3
"""
V3-EXQ-745 -- MECH-456 ecological rebinding, world-driven PATCH-SET-FLIP test-bed
(DESIGN + DV diversity for provisional -> active).

WHAT THIS IS. An INDEPENDENT MECH-456 test-bed + a new DV class. The two prior
PASSes (V3-EXQ-733b / 733c) both rest on ONE teleport-based directed-traversal
harness measuring a POLICY-INERT affinity readout (binder.binding_score). Resting
an active-promotion on a single design + a single (inert) DV is a monoculture. This
experiment adds diversity along two orthogonal axes:

  (a) DESIGN diversity -- a WORLD-DRIVEN productive-patch-set flip (no teleport).
      The world exposes one of N distinct foragable resource clusters; every
      flip_period steps a competing config OVERTAKES the current one (old cluster
      cleared, new cluster placed). Ground-truth world-config g(t) is the flip
      schedule -- binder-independent. Between flips the current cluster is restocked
      so it stays continuously foragable. Contrast with the 733 harness, where a
      SCRIPTED teleport moved the AGENT; here the WORLD moves.

  (b) DV diversity -- an ECOLOGICAL behavioural DV on the POLICY-COUPLED binder
      path. binder.couple() (the only policy-coupled binder path -- fires inside
      e2.rollout_with_world during CEM planning; e2_fast.py:718/730/739) injects the
      binding vector into the rollout world-states the planner scores, steering the
      committed action. DV2 compares FORAGING competence of live rebinding (ree_on)
      vs a factor-FROZEN arm (ree_frozen; binds once at episode entry, never
      re-evaluates). DV1 is the (policy-inert) affinity readout kept for continuity:
      does argmax binding_score track the true productive config above a
      config-label-shuffle control.

WHY THIS IS NOT A RE-QUEUE OF 733. Different test-bed (world-driven flip vs agent
teleport), different DV (ecological foraging on the couple() path vs inert affinity
latency). It does NOT supersede 733b/733c (no supersedes field): it broadens the
evidence base, it does not correct those runs.

NULL (declared). A small / null DV2 with an unready or ceiling-bound substrate is
substrate_ceiling (non_contributory), NOT a clean refutation of MECH-456 -- the
couple() effect can be real but below the foraging-resolution of this test-bed. A
DV1 failure (affinity not above the config-label shuffle) DOES weaken (the readout
is not tracking the true competitor). Readiness-unmet self-routes
substrate_not_ready_requeue (non_contributory), never a verdict.

RUN PASS = >= MIN_SEEDS_COMPLETED seeds complete AND interpret label ==
"ecological_rebinding_supported" (all seeds ready, DV1 and DV2 each on >=
MIN_SEEDS_FOR_PASS seeds). MECH-456 stays candidate / v3_pending regardless
(v3_pending gate; PROMOTES NOTHING by itself).

NO ree_core change -- harness-level measurement + a harness-level factor-freeze
monkeypatch over the trained-then-frozen learned cross_stream_binder. Env / agent
config / measurement primitives shared with 733 via
experiments/_lib/rebinding_functional_harness.py and the new
experiments/_lib/rebinding_ecological_harness.py.

Claims: [MECH-456] (experiment_purpose=evidence).
Bears on (cited, NOT tagged): MECH-269, MECH-270, ARC-006, MECH-045, INV-002.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiment_protocol import emit_outcome
from experiments._metrics import check_degeneracy
from experiments._lib import rebinding_ecological_harness as E
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.capability_eval import OraclePolicy, RandomPolicy
from experiments.pack_writer import write_flat_manifest
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_745_rebinding_ecological_patchflip"
QUEUE_ID = "V3-EXQ-745"
CLAIM_IDS: List[str] = ["MECH-456"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = None

# Converged learned cross_stream_binder activation (identical to V3-EXQ-733/733b).
CROSS_STREAM_BINDING_ENABLED = True
CROSS_STREAM_BINDING_LEARNED = True
CROSS_STREAM_BINDING_STRENGTH = 0.5
CROSS_STREAM_BINDING_TEMPERATURE = 0.2
CROSS_STREAM_BINDING_CONV_FRAC = 0.85

# Patch-flip test-bed geometry.
SIZE = 12
N_CONFIGS = 4
RESOURCES_PER_CONFIG = 4
FLIP_PERIOD = 15

SEEDS = [42, 43, 44, 45, 46, 47]
P0_EPISODES = 40
M_EPISODES = 20
STEPS_PER_EPISODE = 120

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 3
DRY_RUN_M = 3
DRY_RUN_STEPS = 40

# Softened-survival env (mirror 733b) so the flip clusters stay foragable and the
# ONLY resource manager is the flip controller. reef DISABLED (independent test-bed).
ENV_KWARGS = dict(
    size=SIZE,
    num_hazards=4,
    num_resources=5,               # ignored in practice -- flip controller owns resources
    hazard_harm=0.0,               # SOFTENED -- hazards present but non-lethal
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.0,      # SOFTENED -- no proximity harm
    hazard_food_attraction=0.0,    # SOFTENED -- hazards do not chase food
    resource_respawn_on_consume=False,  # flip controller is the sole resource manager
    reef_enabled=False,
    toroidal=False,
    harm_history_len=10,
)

FLIP_KWARGS = dict(
    size=SIZE,
    n_configs=N_CONFIGS,
    resources_per_config=RESOURCES_PER_CONFIG,
    flip_period=FLIP_PERIOD,
)


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """The COLD V3-EXQ-733 agent config, VERBATIM (REEConfig.from_dims block copied
    from v3_exq_733b_rebinding_pB_directed_traversal.py:175-212)."""
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


def _full_config() -> Dict[str, Any]:
    """The full config snapshot for arm_cell fingerprinting + stamp_recording_core."""
    return {
        "env_kwargs": dict(ENV_KWARGS),
        "flip_kwargs": dict(FLIP_KWARGS),
        "size": int(SIZE),
        "n_configs": int(N_CONFIGS),
        "resources_per_config": int(RESOURCES_PER_CONFIG),
        "flip_period": int(FLIP_PERIOD),
        "p0_episodes": int(P0_EPISODES),
        "m_episodes": int(M_EPISODES),
        "steps_per_episode": int(STEPS_PER_EPISODE),
        "align_margin": float(E.ALIGN_MARGIN),
        "min_overtakes": int(E.MIN_OVERTAKES),
        "dv2_abs_margin": float(E.DV2_ABS_MARGIN),
        "min_seeds_for_pass": int(E.MIN_SEEDS_FOR_PASS),
        "min_seeds_completed": int(E.MIN_SEEDS_COMPLETED),
        "binder_config": {
            "cross_stream_binding_enabled": bool(CROSS_STREAM_BINDING_ENABLED),
            "cross_stream_binding_learned": bool(CROSS_STREAM_BINDING_LEARNED),
            "cross_stream_binding_strength": float(CROSS_STREAM_BINDING_STRENGTH),
            "cross_stream_binding_temperature": float(CROSS_STREAM_BINDING_TEMPERATURE),
            "cross_stream_binding_conv_frac": float(CROSS_STREAM_BINDING_CONV_FRAC),
        },
    }


def _run_seed(seed: int, p0_episodes: int, m_episodes: int, steps_per_episode: int,
              full_config: Dict[str, Any], arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Train the binder P0 on the flip test-bed, then measure the four arms
    (ree_on, ree_frozen, greedy_oracle, random_walk) and build the per-seed row.

    NOTE (confound, flagged for review): ree_on and ree_frozen share the SAME
    post-P0 agent; ree_on runs first, so its online E2 self-model adaptation (via
    record_transition) precedes ree_frozen. The binder itself (the thing under test)
    is frozen identically in both arms; the residual E2 drift, if anything, helps the
    later FROZEN arm -> conservative for the supports direction.
    """
    torch.manual_seed(seed)
    probe_env = _make_env(seed)
    agent = _make_agent(probe_env)

    def env_factory(s: int) -> CausalGridWorldV2:
        env = _make_env(s)
        if (int(env.body_obs_dim) != int(probe_env.body_obs_dim)
                or int(env.world_obs_dim) != int(probe_env.world_obs_dim)):
            raise RuntimeError(f"env dim parity FAILED seed={s}")
        return env

    error_note: Optional[str] = None
    try:
        protos, proto_cnt, config_visits, agent = E.train_binder_p0(
            agent=agent, env_factory=env_factory, flip_kwargs=FLIP_KWARGS,
            seed=seed, p0_episodes=p0_episodes, steps_per_episode=steps_per_episode)
        all_configs_visited = bool(all(c > 0 for c in config_visits))
        n_configs_covered = int(sum(1 for c in config_visits if c > 0))

        ineligible = ["shared_frozen_binder_measurement_arm"]

        # -- ree_on (live rebinding; DV1 streams + foraging_on) ----------------
        with arm_cell(seed, config_slice={**full_config, "arm_id": "ree_on", "frozen": False},
                      script_path=Path(__file__),
                      include_driver_script_in_hash=False,
                      extra_ineligible_reasons=ineligible) as cell:
            on = E.measure_arm(
                agent=agent, env_factory=env_factory, flip_kwargs=FLIP_KWARGS,
                seed=seed, m_episodes=m_episodes, steps_per_episode=steps_per_episode,
                protos=protos, arm_name="ree_on", frozen=False, record_dv1=True)
            on_row = {
                "arm_id": "ree_on", "seed": int(seed),
                "foraging_competence": float(on["foraging_competence"]),
                "survival_horizon": float(on["survival_horizon"]),
                "n_flips": int(on["n_flips"]),
                "error_note": on["error_note"],
            }
            cell.stamp(on_row)
            arm_results.append(on_row)
        if on["error_note"] is not None:
            error_note = on["error_note"]

        # -- ree_frozen (factor frozen at episode entry; foraging_frozen) ------
        with arm_cell(seed, config_slice={**full_config, "arm_id": "ree_frozen", "frozen": True},
                      script_path=Path(__file__),
                      include_driver_script_in_hash=False,
                      extra_ineligible_reasons=ineligible) as cell:
            fr = E.measure_arm(
                agent=agent, env_factory=env_factory, flip_kwargs=FLIP_KWARGS,
                seed=seed, m_episodes=m_episodes, steps_per_episode=steps_per_episode,
                protos=protos, arm_name="ree_frozen", frozen=True, record_dv1=False)
            fr_row = {
                "arm_id": "ree_frozen", "seed": int(seed),
                "foraging_competence": float(fr["foraging_competence"]),
                "survival_horizon": float(fr["survival_horizon"]),
                "error_note": fr["error_note"],
            }
            cell.stamp(fr_row)
            arm_results.append(fr_row)
        if error_note is None and fr["error_note"] is not None:
            error_note = fr["error_note"]

        # -- ree_nocouple (POSITIVE CONTROL: binding coupling disabled) --------
        # couple() authority = |foraging_on - foraging_nocouple|. If ~0 the binding is
        # behaviourally inert and a null ON-vs-FROZEN DV2 is UNinterpretable (routes to
        # non_contributory, NOT a false inert-rebinding weakens). Shares the frozen binder.
        with arm_cell(seed, config_slice={**full_config, "arm_id": "ree_nocouple", "couple_off": True},
                      script_path=Path(__file__),
                      include_driver_script_in_hash=False,
                      extra_ineligible_reasons=ineligible) as cell:
            nc = E.measure_arm(
                agent=agent, env_factory=env_factory, flip_kwargs=FLIP_KWARGS,
                seed=seed, m_episodes=m_episodes, steps_per_episode=steps_per_episode,
                protos=protos, arm_name="ree_nocouple", frozen=False, record_dv1=False,
                couple_off=True)
            nc_row = {
                "arm_id": "ree_nocouple", "seed": int(seed),
                "foraging_competence": float(nc["foraging_competence"]),
                "survival_horizon": float(nc["survival_horizon"]),
                "error_note": nc["error_note"],
            }
            cell.stamp(nc_row)
            arm_results.append(nc_row)
        if error_note is None and nc["error_note"] is not None:
            error_note = nc["error_note"]

        # -- greedy_oracle (achievability ceiling) -----------------------------
        with arm_cell(seed, config_slice={**full_config, "arm_id": "greedy_oracle"},
                      script_path=Path(__file__),
                      include_driver_script_in_hash=False) as cell:
            orc = E.calib_arm(
                policy_factory=lambda: OraclePolicy(), env_factory=env_factory,
                flip_kwargs=FLIP_KWARGS, seed=seed, m_episodes=m_episodes,
                steps_per_episode=steps_per_episode)
            orc_row = {
                "arm_id": "greedy_oracle", "seed": int(seed),
                "foraging_competence": float(orc["foraging_competence"]),
                "survival_horizon": float(orc["survival_horizon"]),
                "error_note": None,
            }
            cell.stamp(orc_row)
            arm_results.append(orc_row)

        # -- random_walk (floor) ----------------------------------------------
        with arm_cell(seed, config_slice={**full_config, "arm_id": "random_walk"},
                      script_path=Path(__file__),
                      include_driver_script_in_hash=False) as cell:
            rnd = E.calib_arm(
                policy_factory=lambda: RandomPolicy(seed), env_factory=env_factory,
                flip_kwargs=FLIP_KWARGS, seed=seed, m_episodes=m_episodes,
                steps_per_episode=steps_per_episode)
            rnd_row = {
                "arm_id": "random_walk", "seed": int(seed),
                "foraging_competence": float(rnd["foraging_competence"]),
                "survival_horizon": float(rnd["survival_horizon"]),
                "error_note": None,
            }
            cell.stamp(rnd_row)
            arm_results.append(rnd_row)

        # -- DV1 over the ON affinity stream -----------------------------------
        gen_align = torch.Generator()
        gen_align.manual_seed(seed * 104729 + 7)
        dv1 = E.compute_dv1(on["b_on"], on["gts"], on["ep_ids"],
                            k_regions=N_CONFIGS, gen=gen_align)

        binder = agent.cross_stream_binder
        binder_converged = bool(getattr(binder, "binder_converged", False)) if binder else False
        binder_loss_ema = getattr(binder, "loss_ema", None) if binder else None
        binder_chance_floor = getattr(binder, "chance_floor", None) if binder else None

        row = {
            "seed": int(seed),
            "error_note": error_note,
            "binder_converged": binder_converged,
            "binder_loss_ema": binder_loss_ema,
            "binder_chance_floor": binder_chance_floor,
            "all_configs_visited": all_configs_visited,
            "n_configs": int(N_CONFIGS),
            "n_configs_covered": n_configs_covered,
            "config_visits_p0": [int(c) for c in config_visits],
            "n_overtakes": int(dv1["n_overtakes"]),
            "n_flips_on": int(on["n_flips"]),
            "foraging_on": float(on["foraging_competence"]),
            "foraging_frozen": float(fr["foraging_competence"]),
            "foraging_nocouple": float(nc["foraging_competence"]),
            "foraging_oracle": float(orc["foraging_competence"]),
            "foraging_random": float(rnd["foraging_competence"]),
            "foraging_dv2_gap": float(on["foraging_competence"] - fr["foraging_competence"]),
            "couple_authority_obs": float(abs(on["foraging_competence"] - nc["foraging_competence"])),
            "couple_authority": bool(
                abs(on["foraging_competence"] - nc["foraging_competence"]) >= E.COUPLE_AUTHORITY_FLOOR),
            "survival_on": float(on["survival_horizon"]),
            "survival_frozen": float(fr["survival_horizon"]),
            "alignment_real": float(dv1["alignment_real"]),
            "alignment_shuffle": float(dv1["alignment_shuffle"]),
            "alignment_margin_obs": float(dv1["alignment_margin_obs"]),
            "n_dv1_steps": int(dv1["n_dv1_steps"]),
            "dv1": bool(dv1["dv1"]),
        }
    except Exception as exc:  # noqa: BLE001 -- record + continue so other seeds run
        row = {
            "seed": int(seed),
            "error_note": f"exception seed={seed}: {type(exc).__name__}: {exc}",
            "binder_converged": False,
            "binder_loss_ema": None,
            "binder_chance_floor": None,
            "all_configs_visited": False,
            "n_configs": int(N_CONFIGS),
            "n_configs_covered": 0,
            "config_visits_p0": [],
            "n_overtakes": 0,
            "n_flips_on": 0,
            "foraging_on": 0.0,
            "foraging_frozen": 0.0,
            "foraging_nocouple": 0.0,
            "foraging_oracle": 0.0,
            "foraging_random": 0.0,
            "foraging_dv2_gap": 0.0,
            "couple_authority_obs": 0.0,
            "couple_authority": False,
            "survival_on": 0.0,
            "survival_frozen": 0.0,
            "alignment_real": 0.0,
            "alignment_shuffle": 0.0,
            "alignment_margin_obs": 0.0,
            "n_dv1_steps": 0,
            "dv1": False,
        }
    return row


def run_experiment(seeds: List[int], p0_episodes: int, m_episodes: int,
                   steps_per_episode: int, dry_run: bool) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    full_config = _full_config()
    rows: List[Dict[str, Any]] = []
    arm_results: List[Dict[str, Any]] = []
    first = True
    for s in seeds:
        print(f"Seed {s} Condition ecological_patchflip", flush=True)
        if first:
            print(
                f"  (P0={p0_episodes} ep, M={m_episodes} ep, steps={steps_per_episode}, "
                f"n_configs={N_CONFIGS}, flip_period={FLIP_PERIOD}, "
                f"align_margin={E.ALIGN_MARGIN}, min_overtakes={E.MIN_OVERTAKES}, "
                f"dv2_abs_margin={E.DV2_ABS_MARGIN}, dry_run={dry_run})",
                flush=True,
            )
            first = False
        row = _run_seed(s, p0_episodes, m_episodes, steps_per_episode,
                        full_config, arm_results)
        rows.append(row)

        ready = E._seed_ready(row)
        dv2 = E._dv2_seed(row, E.DV2_ABS_MARGIN)
        seed_pass = bool(row["error_note"] is None and ready and row["dv1"] and dv2)
        print(
            f"verdict: {'PASS' if seed_pass else 'FAIL'} "
            f"(ready={ready} dv1={row['dv1']} dv2={dv2} "
            f"foraging_on={row['foraging_on']:.3f} foraging_frozen={row['foraging_frozen']:.3f} "
            f"nocouple={row['foraging_nocouple']:.3f} gap={row['foraging_dv2_gap']:.3f} "
            f"couple_authority={row['couple_authority']}({row['couple_authority_obs']:.3f}) "
            f"oracle={row['foraging_oracle']:.3f} "
            f"align_margin={row['alignment_margin_obs']:.3f} overtakes={row['n_overtakes']})",
            flush=True,
        )

    interp = E.interpret_ecological(rows, dv2_abs_margin=E.DV2_ABS_MARGIN,
                                    conv_frac=CROSS_STREAM_BINDING_CONV_FRAC)
    ok = [r for r in rows if r["error_note"] is None]
    n_completed = len(ok)

    passed = bool(
        n_completed >= E.MIN_SEEDS_COMPLETED
        and interp["label"] == "ecological_rebinding_supported"
    )

    degen = check_degeneracy({
        "dv2_foraging_gap": [r["foraging_dv2_gap"] for r in ok] or [0.0, 0.0],
        "dv1_alignment_margin": [r["alignment_margin_obs"] for r in ok] or [0.0, 0.0],
    })

    result = {
        "outcome": "PASS" if passed else "FAIL",
        "seeds": list(seeds),
        "n_completed": int(n_completed),
        "n_total_runs": int(len(seeds)),
        "min_seeds_completed": int(E.MIN_SEEDS_COMPLETED),
        "p0_episodes": int(p0_episodes),
        "m_episodes": int(m_episodes),
        "steps_per_episode": int(steps_per_episode),
        "n_configs": int(N_CONFIGS),
        "resources_per_config": int(RESOURCES_PER_CONFIG),
        "flip_period": int(FLIP_PERIOD),
        "acceptance_thresholds": {
            "align_margin": float(E.ALIGN_MARGIN),
            "min_overtakes": int(E.MIN_OVERTAKES),
            "dv2_abs_margin": float(E.DV2_ABS_MARGIN),
            "min_seeds_for_pass": int(E.MIN_SEEDS_FOR_PASS),
            "min_seeds_completed": int(E.MIN_SEEDS_COMPLETED),
            "competence_resource_floor": float(E.COMPETENCE_RESOURCE_FLOOR),
        },
        "binder_config": {
            "cross_stream_binding_enabled": bool(CROSS_STREAM_BINDING_ENABLED),
            "cross_stream_binding_learned": bool(CROSS_STREAM_BINDING_LEARNED),
            "cross_stream_binding_strength": float(CROSS_STREAM_BINDING_STRENGTH),
            "cross_stream_binding_temperature": float(CROSS_STREAM_BINDING_TEMPERATURE),
            "cross_stream_binding_conv_frac": float(CROSS_STREAM_BINDING_CONV_FRAC),
        },
        "env_kwargs": dict(ENV_KWARGS),
        "flip_kwargs": dict(FLIP_KWARGS),
        "per_seed_results": rows,
        "arm_results": arm_results,
        "interpretation": interp,
        "non_degenerate": degen["non_degenerate"],
        "degeneracy_reason": degen["degeneracy_reason"],
        "degenerate_metrics": degen["degenerate_metrics"],
    }
    return result, arm_results


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    interp = result.get("interpretation", {})
    label = interp.get("label", "")
    direction, direction_note = E.evidence_direction_ecological(label)

    review_caveats = [
        "V3-EXQ-745 adds DESIGN + DV diversity to MECH-456 beyond the 733 harness. "
        "The two prior PASSes (V3-EXQ-733b/733c) both used ONE teleport-based "
        "directed-traversal test-bed and ONE policy-INERT affinity readout "
        "(binder.binding_score). This run is an INDEPENDENT world-driven patch-set-flip "
        "test-bed (the WORLD moves, not the agent) with an ECOLOGICAL behavioural DV on "
        "the POLICY-COUPLED couple() path. It does NOT supersede 733b/733c -- it broadens "
        "the evidence base rather than correcting those runs (no supersedes field).",
        "DV2 is the load-bearing new signal: foraging competence of live rebinding "
        "(ree_on) vs a factor-FROZEN arm (ree_frozen) that binds once at episode entry "
        "and never re-evaluates. The freeze is a harness-level monkeypatch of "
        "binder.factor (episode-entry g reused for every CEM candidate); "
        "binder.binding_score (the DV1 readout) is left untouched, so freezing changes "
        "BEHAVIOUR (couple path) not the readout. Pre-registered effect-size floor "
        f"DV2_ABS_MARGIN={E.DV2_ABS_MARGIN} resources/episode plus a non-saturation "
        "guard (foraging_on < foraging_oracle on DV2-passing seeds).",
        "CONFOUND (flagged): ree_on and ree_frozen share the SAME post-P0 agent and "
        "ree_on runs first, so its online E2 self-model adaptation (record_transition) "
        "precedes ree_frozen. The binder under test is frozen identically in both arms; "
        "the residual E2 drift, if anything, helps the later FROZEN arm -> conservative "
        "for the supports direction. A per-arm agent snapshot would remove even this "
        "conservative asymmetry (not done here to stay within the single-P0-train design).",
        "P0 config coverage depends on foraging survival: the flip controller resets to "
        "config 0 at every episode entry (install), so configs 1..N-1 are only visited "
        "in P0 episodes that survive past the flip boundaries. Softened survival "
        "(hazard_harm / proximity_harm / hazard_food_attraction -> 0) makes this likely; "
        "if a config is never visited the readiness gate self-routes "
        "substrate_not_ready_requeue (non_contributory), NOT a verdict.",
        "GROUND TRUTH g(t) is the flip-schedule config index, read off the controller -- "
        "binder-independent. DV1 (alignment_real vs config-label-shuffle) tests "
        "true-config tracking above chance; the affinity readout is policy-INERT "
        "(used only for DV1 measurement, not for control).",
        "POSITIVE CONTROL (couple authority): a ree_nocouple arm disables the binding "
        "coupling (binder.strength=0 -> couple() identity). couple_authority = "
        "|foraging_on - foraging_nocouple| must clear COUPLE_AUTHORITY_FLOOR="
        f"{E.COUPLE_AUTHORITY_FLOOR} resources/episode for a seed to be READY. This is the "
        "load-bearing readiness gate the smoke test motivated: the dry-run showed "
        "foraging_on == foraging_frozen EXACTLY (gap 0.000), i.e. the couple() perturbation "
        "may be too weak to move foraging AT ALL on this substrate. Without this control a "
        "null ON-vs-FROZEN DV2 would mis-route to an inert-rebinding WEAKENS; WITH it, a "
        "couple-inert substrate self-routes substrate_not_ready_requeue (non_contributory), "
        "and an inert-rebinding WEAKENS is emitted ONLY when the binding demonstrably CAN "
        "move foraging (authority passed) yet re-evaluating it does not (DV2 null). This "
        "honours the claim's own V3-EXQ-478 precedent (inert == inconclusive, not weakens).",
        "NULL declared: a small / null DV2 on an unready or couple-INERT substrate is "
        "substrate_ceiling (non_contributory), NOT a clean refutation of MECH-456 -- the "
        "couple() effect can be real but below this test-bed's foraging resolution. A DV1 "
        "failure (affinity not above shuffle) DOES weaken. rebinding_inert_off_equals_on "
        "(DV1 holds, couple authority holds, DV2 fails) is the MECH-269(b)/V3-EXQ-478 inert "
        "signature and weakens.",
        "Brake note: MECH-456 previously drew 2 non_contributory readiness autopsies on "
        "the 733/733a survival-onboarding axis (failure_autopsy_V3-EXQ-733_2026-07-10, "
        "failure_autopsy_V3-EXQ-733a_2026-07-11). This is NOT a re-run of that braked "
        "axis: it is a NEW measurement class (world-driven flip design + ecological "
        "couple()-path DV), whose declared null routes to non_contributory/weakens rather "
        "than re-litigating the survival-onboarding refusal scope.",
        "NO ree_core change -- harness-level measurement + factor-freeze only, over the "
        "trained-then-frozen learned cross_stream_binder.",
    ]
    if not interp.get("all_ready", False):
        review_caveats.insert(
            0,
            "WARNING readiness gate UNMET on >=1 completed seed "
            f"(min_overtakes={interp.get('min_overtakes_obs')} vs {E.MIN_OVERTAKES}). "
            "Self-routes substrate_not_ready_requeue (non_contributory); NOT a verdict.",
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
        "bears_on_not_tagged": [
            "MECH-269", "MECH-270", "ARC-006", "MECH-045", "INV-002",
        ],
        "review_caveats": review_caveats,
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "flip_kwargs": dict(FLIP_KWARGS),
        "config_summary": {
            "test_bed": "world_driven_patch_set_flip",
            "dv_class": "ecological_foraging_on_couple_path",
            "supersedes_733": False,
            "n_configs": int(N_CONFIGS),
            "flip_period": int(FLIP_PERIOD),
            "resources_per_config": int(RESOURCES_PER_CONFIG),
            "harness_level_measurement": True,
            "ree_core_modified": False,
            "cross_stream_binding_enabled": bool(CROSS_STREAM_BINDING_ENABLED),
            "cross_stream_binding_learned": bool(CROSS_STREAM_BINDING_LEARNED),
        },
        "arm_results": result.get("arm_results", []),
        "result": result,
    }
    return manifest


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(description="V3-EXQ-745 ecological rebinding patch-flip")
    parser.add_argument("--dry-run", action="store_true",
                        help="Short smoke-test (1 seed, 3+3 ep, 40 steps).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override output dir (default: REE_assembly evidence/experiments).")
    args = parser.parse_args()

    t0 = time.perf_counter()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0, m, steps = DRY_RUN_P0, DRY_RUN_M, DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0, m, steps = P0_EPISODES, M_EPISODES, STEPS_PER_EPISODE

    result, _arm_results = run_experiment(
        seeds=seeds, p0_episodes=p0, m_episodes=m,
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
    # Single sanctioned flat-manifest writer (ERS sec 4): stamps the always-record
    # core (substrate_hash hoisted from the per-arm fingerprints) + enforces the
    # run_id/_v3 + status invariants. Handles the dry-run _dry_ filename prefix.
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=bool(args.dry_run),
        config=_full_config(), seeds=SEEDS, script_path=Path(__file__),
        started_at=t0)

    interp = result["interpretation"]
    print(f"manifest: {out_path}", flush=True)
    print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} label={interp['label']} "
        f"DV1={interp['n_DV1']}/{interp['n_completed']} "
        f"DV2={interp['n_DV2']}/{interp['n_completed']} (need {E.MIN_SEEDS_FOR_PASS}) "
        f"all_ready={interp['all_ready']} min_overtakes={interp.get('min_overtakes_obs')} "
        f"direction={manifest['evidence_direction']}",
        flush=True,
    )

    outcome_norm = str(result["outcome"]).upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    return outcome_emit, str(out_path), bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry)
    sys.exit(0)
