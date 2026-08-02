"""V3-EXQ-868: MECH-292 ghost-goal ranking, goal-relevance confirmer (EVIDENCE).

experiment_purpose: evidence
GOV-CONFIRM-1: MECH-292 is candidate/v3_pending with lit_conf 0.871 and
genuine_exp_count=0 (claim_evidence.v1.json) despite a landed diagnostic PASS
(V3-EXQ-496, 2026-04-27) -- that run is experiment_purpose="diagnostic" (an
instrument/wiring check: module exposure, master-off no-op, floor exclusion
of payload-less anchors, component-sum consistency) and is correctly
excluded from claim confidence scoring. This script is the first
experiment_purpose="evidence" test of MECH-292's own scientific content.

CLAIM TESTED (claims.yaml MECH-292 functional_restatement, verbatim):
"Falsifiable: in a reward-relocation or blocked-corridor task, anchors from
the now-obstructed but still-valued path should rank above equally stale
but goal-irrelevant anchors." ghost_goal_bank.py's own module docstring
sharpens this into the exact pre-registered design this script implements:
"in a paired comparison of a goal-relevant inactive anchor vs a
goal-irrelevant inactive anchor with matched staleness, the goal-relevant
anchor's ghost_priority should be strictly higher in >=4/7 seeds, with the
gap dominated by the goal_match component."

DV TYPE: representational / wall-independent (per the workset why_now,
which asks for exactly this and to self-route substrate_not_ready_requeue
if only a behavioral DV were available). This script never touches action
selection or env reward -- it reads GhostGoalBankEntry.ghost_priority and
.components directly off HippocampalModule.rank_ghost_goals(), the same
representational surface V3-EXQ-496's diagnostic already reads. No
self-route needed: V3-EXQ-496 already proved this DV is reachable.

DESIGN (matched-staleness paired comparison, single mixed-relevance probe
per seed -- NOT the two-way payload-vs-no-payload split V3-EXQ-496's UC4
used, which tested only "has a snapshot" vs "has none". This tests genuine
DIRECTIONAL discrimination between two real, populated snapshots):

  Per seed, build TWO agents from the IDENTICAL torch.manual_seed(seed)
  (so identical random-init encoder weights -> their z_goal outputs live
  in the SAME embedding space and are directly cosine-comparable -- the
  exact trick V3-EXQ-496's own UC4 already validated: bank_a =
  agent_a.rank_ghost_goals(z_goal_b) cross-agent-compares for a PASSING
  diagnostic), but step them through DIFFERENT env layouts (env seed vs
  env seed + 1000) under the IDENTICAL goal-active schedule (same
  force_drive, force_benefit, same n_ticks, same force_boundary_every).
  Identical schedules => identical AnchorSet._tick progression => an
  anchor created at the i-th forced boundary in agent_B has the SAME
  staleness as the i-th-boundary anchor in agent_A once both are read at
  the same tick count. This is what makes it a genuine matched-staleness
  design without hand-crafted key manipulation.

    agent_B ("relevant" source): env seed = seed. Its OWN inactive
      anchors, ranked against its OWN final z_goal (still pointed the
      same direction it was pursuing -- "still-valued", not abandoned),
      are the goal-relevant candidates.
    agent_A ("irrelevant" source): env seed = seed + 1000 (different
      resource layout -> a genuinely different, but equally goal-active
      and equally real, pursued direction). Its inactive anchors, ranked
      against agent_B's z_goal probe (cross-agent, same trick as UC4),
      are the goal-irrelevant-but-real candidates.

  goal_cue_centering=True (deliberate, SD-079-motivated choice -- see
  note below), so both banks use the AnchorSet's SD-079 common-mode
  baseline. Per seed: compare bank_relevant[0] (best-preserved
  goal-relevant trace) vs bank_irrelevant[0] (best-preserved
  goal-irrelevant trace). Comparable only when BOTH banks are non-empty
  (both cleared goal_match_floor); when bank_irrelevant is empty but
  bank_relevant is not, that is ALSO evidence for the claim (the floor
  itself excluded the irrelevant trace) and is recorded as a distinct
  win sub-type rather than folded into the direct-comparison count.

SD-079 / goal_cue_centering NOTE (deliberate substrate-config choice, not
an arbitrary knob change): AnchorSetConfig.goal_cue_centering defaults
False in production. Without it, z_goal lives in an extremely tight raw
cone (min pairwise cosine measured 0.9767-0.9878 elsewhere in this
substrate family -- ghost_goal_bank.py's own rank() docstring cites
0.9878 with goal_match spanning only 0.0111 across a 24-anchor pool
UNcentered), which would make a two-genuinely-different-direction
discrimination test degenerate by construction (both "relevant" and
"irrelevant" directions would score goal_match ~1.0, and the floor/
ranking could not separate them). Centering is what SD-079 built
specifically to open up that spread. The centering pathway is already
fully wired end-to-end (AnchorSet.observe_goal_cue -> goal_cue_baseline
-> GhostGoalBank.rank() -> Anchor.goal_match(baseline=...)) and reachable
via REEConfig.from_dims(goal_cue_centering=True) (verified against
ree_core/utils/config.py and ree_core/hippocampal/anchor_set.py before
writing this script) -- this is NOT a silently-swallowed kwarg. The
OFF-centering regime is a KNOWN non-discriminating regime for this design,
not a bug; this script does not run an OFF-centering arm because doing so
would just reproduce that documented near-zero-spread degeneracy, not
inform the claim.

NON-DEGENERACY: a seed is "comparable" only when bank_relevant is
non-empty (the substrate engaged at all for the relevant direction). A
seed where bank_relevant is empty (the "still-valued" direction failed to
clear its own floor against itself) contributes nothing and does not
count toward the verdict. If fewer than a majority (4/7) of seeds are
comparable, the whole run self-routes non_contributory (substrate not
engaged enough on this design) -- it never weakens the claim on a design
failure.

PASS CRITERION (load-bearing, matches the ghost_goal_bank.py pre-registered
signature): among comparable seeds, the relevant trace "wins" (higher
top-1 ghost_priority, OR the irrelevant bank was floor-excluded entirely)
in >= the same proportional majority as ">=4/7". Secondary (reported,
non-gating) criterion: on direct-comparison wins, the goal_match component
delta is the largest-magnitude of the four component deltas (wanting,
goal_match, staleness, recoverability) -- the "gap dominated by goal_match"
half of the pre-registered signature.

claim_ids: ['MECH-292'] (single claim; MECH-293/ARC-060/MECH-339/MECH-340
are downstream consumers tagged to their own claim ids and are out of
scope here, matching the claim_ids-accuracy rule).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_868_mech292_ghost_priority_relevance_confirmer.py [--dry-run]

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.hippocampal.ghost_goal_bank import GhostGoalBankEntry
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

EVIDENCE_ROOT = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

EXPERIMENT_PURPOSE = "evidence"

SEEDS: List[int] = [42, 43, 45, 46, 47, 48, 49]  # 7 seeds; seed 44 excluded
                                                  # (recurring per-seed early-
                                                  # death instability on this
                                                  # env config family).
ENV_SEED_OFFSET_IRRELEVANT = 1000  # different resource layout for agent_A
N_TICKS = 60
FORCE_BOUNDARY_EVERY = 8
FORCE_DRIVE = 0.8
FORCE_BENEFIT = 0.4

# Majority bar, proportional to the pre-registered ">=4/7" (matches exactly
# when all 7 seeds are comparable; scales down gracefully if fewer are).
_MIN_COMPARABLE_SEEDS = 4
_PREREG_WIN_FRACTION = 4.0 / 7.0

COMPONENT_KEYS = ("wanting", "goal_match", "staleness", "recoverability")

_ZG = ZGoalStreamAccumulator()


# ---------------------------------------------------------------------------
# Helpers (mirrors V3-EXQ-496's _make_env / _step_episode pattern)
# ---------------------------------------------------------------------------
def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=8,
        num_resources=3,
        num_hazards=2,
        hazard_harm=0.1,
        resource_benefit=0.5,
        resource_respawn_on_consume=True,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.05,
        proximity_approach_threshold=0.15,
        use_proxy_fields=True,
    )


def _build_config(env: CausalGridWorldV2) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        use_per_stream_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        use_sd039_anchor_payload=True,
        use_mech292_ghost_bank=True,
        # SD-079 (see module docstring): centering is REQUIRED for this
        # design to be non-degenerate -- the raw z_goal cone is too tight
        # to discriminate two genuinely different pursued directions.
        goal_cue_centering=True,
        drive_weight=2.0,
        use_resource_proximity_head=True,
    )


def _make_agent(weight_seed: int, env: CausalGridWorldV2) -> REEAgent:
    """Build an agent whose init weights are a pure function of weight_seed.

    Calling this twice with the same weight_seed (regardless of what
    happened to the global RNG in between) yields identical initial
    weights, because torch.manual_seed resets the generator immediately
    before construction -- the same pattern V3-EXQ-496's UC4 uses to make
    two agents' z_goal outputs cross-comparable.
    """
    cfg = _build_config(env)
    cfg.goal.z_goal_enabled = True  # MUST be set before construction --
    # REEAgent.__init__ builds goal_state from this flag at construction
    # time; setting it post-hoc (as agent.config.goal.z_goal_enabled = ...)
    # is too late and leaves goal_state absent/inert. Matches V3-EXQ-496's
    # proven _make_agent pattern.
    torch.manual_seed(weight_seed)
    agent = REEAgent(cfg)
    return agent


def _step_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    n_ticks: int,
    *,
    force_drive: float,
    force_benefit: float,
    force_boundary_every: int,
) -> None:
    """Standard env loop, forcing drive/benefit and periodic boundaries.

    Identical shape to V3-EXQ-496's _step_episode -- kept byte-similar
    deliberately so both agents in a seed pair run the SAME schedule,
    which is what makes their AnchorSet._tick progression (and hence
    per-anchor staleness) directly comparable.
    """
    flat_obs, obs_dict = env.reset()
    agent.reset()
    for tick_idx in range(n_ticks):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        latent = agent.sense(obs_body, obs_world)
        ticks = agent.clock.advance()
        if ticks.get("e1_tick", False):
            e1_prior = agent._e1_tick(latent)
        else:
            e1_prior = torch.zeros(1, agent.config.latent.world_dim)
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        agent.update_z_goal(
            benefit_exposure=force_benefit,
            drive_level=force_drive,
        )
        if (
            force_boundary_every > 0
            and tick_idx > 0
            and (tick_idx % force_boundary_every) == 0
            and agent.hippocampal is not None
            and agent.hippocampal.event_segmenter is not None
            and agent.hippocampal.anchor_set is not None
        ):
            ev = agent.hippocampal.event_segmenter.force_boundary(
                "fast", reason=f"v3_exq_868_t{tick_idx}",
            )
            payload = agent.hippocampal.build_goal_payload(
                latent_state=latent,
                goal_state=agent.goal_state,
                residue_field=agent.residue_field,
                bla_output=agent._bla_last_output,
                current_step=int(agent._step_count),
                simulation_mode=False,
            )
            agent.hippocampal.tick_anchor_set(latent, [ev], goal_payload=payload)
        action = agent.select_action(candidates, ticks, temperature=1.0)
        if action is None:
            action = torch.zeros(1, env.action_dim)
            action[0, 0] = 1.0
            agent._last_action = action
        flat_obs, harm_signal, done, info, obs_dict = env.step(action)
        if done:
            flat_obs, obs_dict = env.reset()


def _current_z_goal(agent: REEAgent) -> Optional[torch.Tensor]:
    if agent.goal_state is None:
        return None
    z_goal = getattr(agent.goal_state, "z_goal", None)
    if z_goal is None:
        return None
    return z_goal.detach()


def _n_inactive_with_payload(agent: REEAgent) -> int:
    if agent.hippocampal is None or agent.hippocampal.anchor_set is None:
        return 0
    return sum(
        1
        for a in agent.hippocampal.anchor_set.all_anchors()
        if not a.active
        and a.goal_payload is not None
        and a.goal_payload.z_goal_snapshot is not None
    )


def _component_deltas(
    winner: GhostGoalBankEntry, loser: GhostGoalBankEntry
) -> Dict[str, float]:
    return {
        k: float(winner.components.get(k, 0.0)) - float(loser.components.get(k, 0.0))
        for k in COMPONENT_KEYS
    }


# ---------------------------------------------------------------------------
# Per-seed cell
# ---------------------------------------------------------------------------
def _run_seed(seed: int, dry_run: bool = False) -> Dict[str, Any]:
    # Dry-run still needs >= 2 force_boundary_every intervals to get any
    # inactive-with-payload anchor at all (the first boundary only installs
    # the first active anchor per family; a SECOND boundary is what remaps
    # it to inactive). 20 ticks / force_boundary_every=8 fires at tick 8 and
    # 16 -- enough to exercise the real code path cheaply.
    n_ticks = 20 if dry_run else N_TICKS

    env_b = _make_env(seed=seed)
    agent_b = _make_agent(weight_seed=seed, env=env_b)
    _step_episode(
        agent_b, env_b, n_ticks=n_ticks,
        force_drive=FORCE_DRIVE, force_benefit=FORCE_BENEFIT,
        force_boundary_every=FORCE_BOUNDARY_EVERY,
    )
    z_goal_probe = _current_z_goal(agent_b)

    env_a = _make_env(seed=seed + ENV_SEED_OFFSET_IRRELEVANT)
    agent_a = _make_agent(weight_seed=seed, env=env_a)  # SAME weight_seed
    _step_episode(
        agent_a, env_a, n_ticks=n_ticks,
        force_drive=FORCE_DRIVE, force_benefit=FORCE_BENEFIT,
        force_boundary_every=FORCE_BOUNDARY_EVERY,
    )

    _ZG.observe(agent_b)
    _ZG.observe(agent_a)

    bank_relevant = agent_b.hippocampal.rank_ghost_goals(z_goal_probe)
    bank_irrelevant = agent_a.hippocampal.rank_ghost_goals(z_goal_probe)
    diag_relevant = agent_b.hippocampal.ghost_goal_bank.get_diagnostics()
    diag_irrelevant = agent_a.hippocampal.ghost_goal_bank.get_diagnostics()

    n_relevant_candidates = _n_inactive_with_payload(agent_b)
    n_irrelevant_candidates = _n_inactive_with_payload(agent_a)

    result: Dict[str, Any] = {
        "seed": seed,
        "n_relevant_candidates": n_relevant_candidates,
        "n_irrelevant_candidates": n_irrelevant_candidates,
        "n_relevant_admitted": len(bank_relevant),
        "n_irrelevant_admitted": len(bank_irrelevant),
        "relevant_diagnostics": diag_relevant,
        "irrelevant_diagnostics": diag_irrelevant,
        "comparable": False,
        "win_type": None,       # "direct" | "floor_exclusion" | None
        "relevant_wins": None,
        "goal_match_dominant": None,
        "top_relevant_components": None,
        "top_irrelevant_components": None,
        "component_deltas": None,
    }

    if not bank_relevant:
        # Substrate did not even engage for the "still-valued" direction
        # against itself -- not comparable, never counted against the claim.
        return result

    result["comparable"] = True
    top_relevant = bank_relevant[0]
    result["top_relevant_components"] = dict(top_relevant.components)
    result["top_relevant_priority"] = float(top_relevant.ghost_priority)

    if not bank_irrelevant:
        # The irrelevant direction was excluded by the floor entirely --
        # a legitimate win for the claim via a different mechanism.
        result["win_type"] = "floor_exclusion"
        result["relevant_wins"] = True
        return result

    top_irrelevant = bank_irrelevant[0]
    result["top_irrelevant_components"] = dict(top_irrelevant.components)
    result["top_irrelevant_priority"] = float(top_irrelevant.ghost_priority)
    result["win_type"] = "direct"
    relevant_wins = float(top_relevant.ghost_priority) > float(top_irrelevant.ghost_priority)
    result["relevant_wins"] = bool(relevant_wins)

    if relevant_wins:
        deltas = _component_deltas(top_relevant, top_irrelevant)
        result["component_deltas"] = deltas
        dominant_key = max(deltas, key=lambda k: abs(deltas[k]))
        result["goal_match_dominant"] = bool(dominant_key == "goal_match")

    return result


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(dry_run: bool = False) -> int:
    t0 = time.perf_counter()
    seeds = SEEDS[:2] if dry_run else SEEDS
    print(
        f"[v3_exq_868] MECH-292 goal-relevance ranking confirmer, "
        f"{len(seeds)} seed(s) ({'dry-run' if dry_run else 'full'})...",
        flush=True,
    )

    per_seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"Seed {seed} Condition mixed_relevance_probe", flush=True)
        r = _run_seed(seed, dry_run=dry_run)
        per_seed_results.append(r)
        verdict = "PASS" if r["relevant_wins"] else ("FAIL" if r["comparable"] else "SKIP")
        print(
            f"  seed={seed} comparable={r['comparable']} win_type={r['win_type']} "
            f"relevant_wins={r['relevant_wins']}",
            flush=True,
        )
        print(f"verdict: {verdict}", flush=True)

    comparable = [r for r in per_seed_results if r["comparable"]]
    n_comparable = len(comparable)
    n_wins = sum(1 for r in comparable if r["relevant_wins"])
    direct_wins = [r for r in comparable if r["win_type"] == "direct" and r["relevant_wins"]]
    n_direct_wins = len(direct_wins)
    n_goal_match_dominant = sum(1 for r in direct_wins if r["goal_match_dominant"])

    n_attempted = len(seeds)
    min_comparable = _MIN_COMPARABLE_SEEDS if not dry_run else 1
    non_degenerate = n_comparable >= min_comparable

    win_threshold = max(1, round(n_comparable * _PREREG_WIN_FRACTION)) if n_comparable else 0
    c1_majority_relevant_wins = non_degenerate and (n_wins >= win_threshold)
    c2_goal_match_dominant_majority = (
        n_direct_wins > 0 and n_goal_match_dominant >= max(1, round(n_direct_wins / 2.0))
    )

    if not non_degenerate:
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        interpretation_label = "substrate_not_engaged_insufficient_comparable_seeds"
    elif c1_majority_relevant_wins:
        outcome = "PASS"
        evidence_direction = "supports"
        interpretation_label = "goal_relevant_trace_outranks_goal_irrelevant"
    else:
        outcome = "FAIL"
        evidence_direction = "weakens"
        interpretation_label = "goal_relevant_trace_does_not_outrank_goal_irrelevant"

    elapsed = time.perf_counter() - t0
    print(
        f"[v3_exq_868] overall: {outcome} "
        f"(comparable={n_comparable}/{n_attempted}, wins={n_wins}/{max(n_comparable,1)}, "
        f"threshold={win_threshold}, goal_match_dominant={n_goal_match_dominant}/{max(n_direct_wins,1)}) "
        f"({elapsed:.1f}s)",
        flush=True,
    )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_868_mech292_ghost_priority_relevance_confirmer_{ts}_v3"

    full_config = {
        "seeds": seeds,
        "n_ticks": N_TICKS if not dry_run else 20,
        "force_boundary_every": FORCE_BOUNDARY_EVERY,
        "force_drive": FORCE_DRIVE,
        "force_benefit": FORCE_BENEFIT,
        "env_seed_offset_irrelevant": ENV_SEED_OFFSET_IRRELEVANT,
        "goal_cue_centering": True,
        "use_mech292_ghost_bank": True,
        "use_sd039_anchor_payload": True,
        "min_comparable_seeds": min_comparable,
        "prereg_win_fraction": _PREREG_WIN_FRACTION,
    }

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": "v3_exq_868_mech292_ghost_priority_relevance_confirmer",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": ["MECH-292"],
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {"MECH-292": evidence_direction},
        "evidence_direction_note": (
            "Representational/wall-independent confirmer of MECH-292's own "
            "falsifiable signature (ghost_goal_bank.py module docstring): a "
            "goal-relevant inactive anchor (still-valued direction) should "
            "outrank a goal-irrelevant inactive anchor of matched staleness, "
            "gap dominated by goal_match. Two same-weight-seed agents "
            "(cross-comparable z_goal embedding, per V3-EXQ-496 UC4's already-"
            "validated trick) run the identical goal-active schedule under "
            "different env layouts; goal_cue_centering=True per SD-079 so the "
            "test is non-degenerate. "
            f"{n_comparable}/{n_attempted} seeds comparable (substrate "
            "engaged), relevant trace won "
            f"{n_wins}/{max(n_comparable,1)} (threshold {win_threshold}, "
            "matching the pre-registered >=4/7 bar proportionally); of "
            f"{n_direct_wins} direct (both-admitted) wins, "
            f"{n_goal_match_dominant} were goal_match-dominated "
            "(secondary, non-gating criterion)."
        ),
        "outcome": outcome,
        "interpretation": {
            "label": interpretation_label,
            "criteria": [
                {
                    "name": "C1_relevant_majority_wins",
                    "load_bearing": True,
                    "passed": bool(c1_majority_relevant_wins),
                    "n_comparable": n_comparable,
                    "n_attempted": n_attempted,
                    "n_wins": n_wins,
                    "win_threshold": win_threshold,
                },
                {
                    "name": "C2_goal_match_dominant_on_direct_wins",
                    "load_bearing": False,
                    "passed": bool(c2_goal_match_dominant_majority),
                    "n_direct_wins": n_direct_wins,
                    "n_goal_match_dominant": n_goal_match_dominant,
                },
            ],
            "criteria_non_degenerate": {
                "C1_relevant_majority_wins": bool(non_degenerate),
                "C2_goal_match_dominant_on_direct_wins": bool(n_direct_wins > 0),
            },
            "combination_rule": (
                "outcome = FAIL/non_contributory if n_comparable < "
                f"{min_comparable} (substrate_not_engaged, never weakens); "
                "else PASS/supports iff C1 (relevant wins >= proportional "
                ">=4/7 majority of comparable seeds); else FAIL/weakens. "
                "C2 is reported alongside as a secondary, non-gating "
                "corroboration of the docstring's goal_match-dominance half."
            ),
        },
        "non_degenerate": bool(non_degenerate),
        "degeneracy_reason": (
            None if non_degenerate else
            f"only {n_comparable}/{n_attempted} seeds produced a non-empty "
            "relevant bank (substrate did not engage the 'still-valued' "
            "direction against itself); below the min_comparable_seeds="
            f"{min_comparable} floor"
        ),
        "per_seed_results": per_seed_results,
        "n_comparable_seeds": n_comparable,
        "n_attempted_seeds": n_attempted,
        "n_relevant_wins": n_wins,
        "win_threshold": win_threshold,
        "n_direct_comparison_wins": n_direct_wins,
        "n_goal_match_dominant_wins": n_goal_match_dominant,
        "elapsed_sec": elapsed,
        "dry_run": bool(dry_run),
    }

    out_file = None
    if not dry_run:
        out_dir = EVIDENCE_ROOT / "v3_exq_868_mech292_ghost_priority_relevance_confirmer"
        out_file = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=full_config,
            seeds=seeds,
            script_path=Path(__file__),
            started_at=t0,
            z_goal_stream_stats=_ZG.stats(),
        )
        print(f"Result written to: {out_file}", flush=True)
    else:
        # Smoke: still exercise the stamper's dry-run reporting path (the
        # [smoke] z_goal_stream: line) without landing anything under
        # evidence/experiments/.
        out_file = write_flat_manifest(
            manifest,
            EVIDENCE_ROOT / "v3_exq_868_mech292_ghost_priority_relevance_confirmer",
            dry_run=True,
            config=full_config,
            seeds=seeds,
            script_path=Path(__file__),
            started_at=t0,
            z_goal_stream_stats=_ZG.stats(),
        )

    return {
        "outcome": outcome,
        "manifest_path": out_file,
        "run_id": run_id,
        "dry_run": dry_run,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="2 seeds, 20-tick episodes; relocates the smoke manifest, no evidence/ write.",
    )
    args = parser.parse_args()
    _result = main(dry_run=args.dry_run)
    emit_outcome(
        outcome=_result["outcome"],
        manifest_path=_result["manifest_path"],
        run_id=_result["run_id"],
        dry_run=_result["dry_run"],
    )
    sys.exit(0 if _result["outcome"] == "PASS" else 1)
