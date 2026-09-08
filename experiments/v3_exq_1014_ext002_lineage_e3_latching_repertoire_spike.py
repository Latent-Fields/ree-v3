#!/opt/local/bin/python3
"""V3-EXQ-1014 -- STAGE-1 DIAGNOSTIC SPIKE for the V3-EXQ-983 lineage (EXT-002 rider / ARC-013):
does the canonical E3 policy LATCH a single action class per seed on the 8 pinned boards?

NEW EXQ number, not 983b: the CONFIRMED autopsy failure_autopsy_V3-EXQ-983a_2026-09-06
(governance-20260906-1604, user gate 2026-09-06T16:51:46Z; Fable red-team CONTESTED on routing,
all four verdict-moving findings folded) REFUSES any 983-lineage letter that keeps the
repeat-rate DV or fresh-tick-only key registration, and routes a two-stage successor: FIRST this
cheap diagnostic spike, THEN (only if a repertoire exists) the redesigned residue-frozen ablation
with a hold-aware revisit-AVOIDANCE DV. This driver is stage 1 ONLY. It queues no stage 2; the
session that reads this run's manifest decides stage 2 per the declared null below.

THE MISSING FACT (work-graph: complex (probe-gated) / puzzle (known rules)). 983a excluded 6/8
seeds at its training-completion gate: casualties died to agent_health <= 0 within 6-12% of the
planned steps, survivors trained cleanly, and no board screen separates them -- the red team's
control-policy probe showed a CONSTANT mover survives 200/200 on all 8 pinned boards, a RANDOM
walker dies on all 8 in 13-16 steps, and STAY dies on all 8. The autopsy's best explanation is
that E3's argmin latches ONE action class per seed (stay-like -> death; constant-move -> survival),
inferred by profile-matching and a 25-step post-training probe. The training-phase executed-action-
class histogram existed at run time and was never written (autopsy learning 5). This run WRITES IT.

DESIGN (autopsy routing_note, verbatim in substance):
  * The SAME 8 pinned 6x6 boards: seeds (42, 123, 456, 7, 11, 17, 23, 31), `_make_env` kwargs
    verbatim from 983a, env rebuilt from the same seed at every episode (layout AND start pinned),
    env_drift_prob 0.0, contamination_spread 0.0.
  * The CANONICAL E3 policy exactly as 983a's `_select_action` (agent.select_action over the
    hippocampal CEM candidates, residue read through E3's J(zeta)), residue INTACT (983a's A0
    semantics: accumulate on every harm), the same encoder/E1/E2 training step per env step.
  * A SHORT budget: TRAIN_EPISODES x STEPS_PER_EPISODE = 40 x 200 per seed (hours, not 18h; a
    casualty seed that dies in ~10 steps costs ~400 steps in total).
  * RECORDED per seed: (a) the executed-action-class histogram over FRESH E3 decisions AND over
    every executed step (held steps included); (b) the survival curve -- realised steps per
    episode until agent_health <= 0 / done; (c) harm attribution in which EVERY harm event,
    held-tick harms included, is keyed to the GOVERNING fresh decision (cell, action) -- the
    registration 983a lacked (its fresh-tick-only gate discarded ~90% of harm events and zeroed
    seed 31: 330 harms, 0 keys); (d) the number of distinct erred keys under that registration;
    (e) n_fresh_select / n_latched, harm rate, steps_realized_frac.
  * The red team's CONTROL-POLICY YARDSTICK in the SAME run, on each board: random walk, each
    constant action, and stay -- CONTROL_EPISODES episodes each -- recording survival and harm
    rate, so the E3 profile is matched against them IN THE MANIFEST rather than by hand.

PRE-REGISTERED READOUT (the rule the code executes; revised at the Step 4.5 red-team, see the
RED-TEAM paragraph). A FRESH decision is an e3_tick OR the first step of an episode (after
agent.reset() `_last_action` is None and E3 deliberates even though the clock reports
e3_tick False); only actions that came from E3 (not the no-candidates fallback) enter the
repertoire histogram. A seed is ADJUDICABLE if it made >= FLOOR_FRESH_DECISIONS = 30 fresh
decisions (per-seed; never a whole-run veto); a seed is a CASUALTY if >= 50 percent of its
episodes ended before the step budget. Over ADJUDICABLE seeds:
  latched_fraction = fraction whose fresh-decision repertoire is a single action class (max
    class share >= SINGLE_CLASS_SHARE = 0.95); casualty_latched_fraction = the same among the
    adjudicable CASUALTY seeds.
  ALTERNATIVE (label `latching_dominant_actor_adequacy_gated`): latched_fraction >= 0.5, OR
    (>= 3 adjudicable casualty seeds AND casualty_latched_fraction >= 0.5) -- the rider test is
    gated on the actor-adequacy / monostrategy line (registry zworld_actor_adequacy_locus;
    MECH-269 / MECH-341 / ARC-065 substrate entries); report to governance; do NOT queue stage 2.
  DECLARED NULL (label `repertoire_exists_stage2_proceeds`): neither clause -- latching is NOT
    the survival lever; stage 2 (the avoidance-DV re-pose) proceeds.
  The latched x died 2x2 over adjudicable seeds is recorded so the survival split's relation to
  latching is read from the manifest, not inferred.
  Secondary readouts (recorded, not load-bearing): survival-latching association (do the seeds
    that die match the STAY control profile and the survivors the constant-move profile), the
    executed-step histogram (hold-weighted), harm attribution coverage (fraction of harm events
    that received a governing key, must be 1.0 by construction).

DIAGNOSTIC ADJUDICATION (Step 3): preconditions `adjudicable_seeds_sufficient` (count of
adjudicable seeds >= MIN_ADJUDICABLE_SEEDS = 4 -- seeds below the per-seed floor are recorded as
unadjudicated, never a whole-run veto), `controls_ran_on_every_board`,
`harm_attribution_complete` (every harm event, held-tick harms included, keyed to the governing
decision; 1.0 by construction, recorded so a regression is visible), `e3_selection_used`
(fraction of steps whose action came from E3 rather than the no-candidates fallback >=
FLOOR_E3_USED). criteria_non_degenerate = adjudicability (>= 4 adjudicable seeds) -- NOT spread:
eight seeds all at share 1.0 is the hypothesis's own predicted outcome, not degeneracy. A
below-floor precondition self-routes `substrate_not_ready_requeue`, never a verdict. experiment_purpose = diagnostic; claim_ids = []
(bears_on: actor_adequacy_monostrategy, residue_error_persistence_readout); evidence_direction
`diagnostic_no_direction`.

DV-SYMMETRY: the readout is a per-seed class share -- a set-aggregate over decisions -- and
there is NO manipulation here (single arm, diagnostic); the yardstick policies are recorded, not
contrasted. Nothing is scored against a criterion that a symmetry could annihilate.

ARM FINGERPRINT: one cell per seed, emitted with include_driver_script_in_hash=False (first
experiment of a lineage mints its baseline in-line; CLAUDE.md Experiment Scripts) so a stage-2
driver may cite this run's trained cells via reuse_baseline_from.

GOV-REUSE-1 (Step 2.4): decisive readout = the training-phase fresh-decision action-class
histogram per seed on the pinned boards. Recorded in NO manifest: 983a records only a 25-step
post-training probe's class COUNT (executed_action_classes 1.0-2.0), 983 nothing. NOT
RECOVERABLE -> run. Substrate readiness (2.5/2.5a): identical construction to 983a (ran 18h on
this exact config on 2026-09-05; residue live 32/32, P2 candidate score range positive). Re-derive
brake (2.5b): EXT-002 literal count 2 / ARC-013 1, released by the confirmed autopsy as
instrument/test-design defects (action none); this is a NEW number and a diagnostic and tags no
claim, so the brake does not apply. Substrate-path overlap (2.5c): same footprint as 983a
(agent.py, hippocampal/, residue/, e3_selector.py, latent/stack.py); open corrupting entries
mode-governance-engagement (salience_affinity_input_cap unset here) and SD-082
(use_lateral_pfc_analog False) gated off on this REEConfig.from_dims construction.

red-team (Step 4.5, DIFFERENT model, model=fable, 2026-09-08). PASS 1 BLOCKING, 8 findings, all
verified against the source: F1 (a min-over-seeds fresh-decision floor of 50 was unreachable for
the 5 casualty seeds and vetoed the whole run) -> per-seed adjudicability floor 30, run gate = >= 4
adjudicable seeds, TRAIN_EPISODES 40; F2 (the first executed action of every episode IS an E3
deliberation but was counted as held) -> step-1 counted fresh; F3 (pooled fraction survival-blind)
-> casualty-conditioned clause + 2x2 recorded; F4 (identical shares tripped non-degeneracy) ->
non-degeneracy = adjudicability; F5 (profile match on steps alone) -> (survival, harm rate)
jointly; F6 (fallback actions in the histogram) -> gated on `used`; F7 (machine class) -> recorded;
F8 clear. PASS 2 (fresh session, same model, re-spawned once because F1/F2 changed the readout's
causal chain) CONTESTED: F1/F2/F3/F4/F5/F6 CLOSED, F7 open (queue entry stays machine_affinity
any; machine_class recorded and the readout declared not cross-class comparable), F8 clear; N1
(the casualty clause could fire on one latched seed of two) -> minimum three adjudicable
casualties; N2 (docstring registered the old rule) -> this docstring; N3 (casualty keyed to mean
steps) -> keyed to the fraction of early-ended episodes; N4 (fallback-path harm keying) dormant at
0 fallbacks, recorded; N5 (floor cannot bite once step-1 decisions count) note. No third pass.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.optim as optim  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1014_ext002_lineage_e3_latching_repertoire_spike"
QUEUE_ID = "V3-EXQ-1014"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []
BEARS_ON = ["actor_adequacy_monostrategy", "residue_error_persistence_readout"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
LINEAGE = "V3-EXQ-983 / V3-EXQ-983a (stage-1 spike; NOT a letter, NOT a supersession)"

SEEDS: Tuple[int, ...] = (42, 123, 456, 7, 11, 17, 23, 31)   # 983a's 8 pinned boards, verbatim
TRAIN_EPISODES = 40      # casualty seeds die in ~10 steps/episode (983a), so 40 episodes costs them little and yields ~40-80 fresh decisions
STEPS_PER_EPISODE = 200
NUM_CANDIDATES = 32
CONTROL_EPISODES = 3
CONTROL_POLICIES = ["RANDOM_WALK", "STAY"] + [f"CONSTANT_{a}" for a in range(4)]

# --- pre-registered readout and readiness floors --------------------------------------
SINGLE_CLASS_SHARE = 0.95        # a seed is LATCHED if one class holds >= this share of fresh decisions
LATCHED_FRACTION_SPLIT = 0.5     # < split -> repertoire exists (null); >= split -> latching dominant
FLOOR_FRESH_DECISIONS = 30       # PER-SEED adjudicability floor (a seed below it is UNADJUDICATED, never a whole-run veto -- red-team F1)
MIN_ADJUDICABLE_SEEDS = 4        # run-level readiness: at least this many seeds adjudicable
CASUALTY_EARLY_END_FRACTION = 0.5   # a seed is a CASUALTY if >= this fraction of its episodes ended before the step budget (pass-2 N3)
MIN_CASUALTY_SEEDS_FOR_CLAUSE = 3   # the casualty-conditioned clause needs >= 3 adjudicable casualties (pass-2 N1: 1-of-2 must not flip the label)
FLOOR_E3_USED = 0.90             # fraction of executed steps whose action came from E3 (not the fallback)
MIN_SEEDS_NON_DEGENERATE = 4
# The four readiness entries are COUNT floors and construction identities (fresh decisions,
# E3-used fraction, attribution coverage, control completeness), not a control's signature above
# a scoring gate -- reachable by construction; nothing is anchored to a recorded control.
ANCHOR_REACHABILITY_EXEMPT = (
    "readiness entries are count floors / construction identities (fresh_decisions_sufficient, "
    "e3_selection_used, harm_attribution_complete, controls_ran_on_every_board); no predicate is "
    "anchored to a recorded control signature")


# =========================================================================
# substrate helpers -- COPIED VERBATIM from v3_exq_983a (same boards, same policy)
# =========================================================================
def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _make_env(seed: int, full_config: Dict[str, Any]) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=4,
        num_resources=3,
        hazard_harm=full_config["hazard_harm"],
        env_drift_interval=5,
        env_drift_prob=0.0,
        proximity_harm_scale=full_config["proximity_harm_scale"],
        proximity_benefit_scale=full_config["proximity_harm_scale"] * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
        contamination_spread=0.0,
    )


def _make_agent(env: CausalGridWorldV2, full_config: Dict[str, Any]) -> REEAgent:
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=full_config["self_dim"],
        world_dim=full_config["world_dim"],
        alpha_world=full_config["alpha_world"],
        alpha_self=full_config["alpha_self"],
        reafference_action_dim=0,
    )
    config.latent.unified_latent_mode = False
    return REEAgent(config)


def _candidate_scores(agent, candidates) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for i, traj in enumerate(candidates):
        world_seq = traj.get_world_state_sequence()
        if world_seq is None:
            continue
        val = float(agent.residue_field.evaluate_trajectory(world_seq).sum().item())
        if math.isfinite(val):
            out.append((i, val))
    return out


def _select_action(agent, latent, ticks, n_actions: int, rng: random.Random,
                   ) -> Tuple[torch.Tensor, Optional[float], bool]:
    with torch.no_grad():
        candidates = agent.hippocampal.propose_trajectories(
            latent.z_world.detach(), z_self=latent.z_self.detach(), num_candidates=NUM_CANDIDATES,
        )
        if not candidates:
            return (_action_to_onehot(rng.randint(0, n_actions - 1), n_actions, agent.device), None, False)
        scored = _candidate_scores(agent, candidates)
        vals = [v for _i, v in scored]
        score_range = float(max(vals) - min(vals)) if len(vals) >= 2 else None
        action = agent.select_action(candidates, ticks)
        return action, score_range, True


# =========================================================================
# the agent cell: canonical E3 policy, residue intact, everything recorded
# =========================================================================
def run_agent_cell(seed: int, full_config: Dict[str, Any], n_episodes: int, n_steps: int,
                   ) -> Tuple[Dict[str, Any], REEAgent]:
    print(f"Seed {seed} Condition E3_CANONICAL_RESIDUE_INTACT", flush=True)
    with arm_cell(seed, config_slice=full_config, script_path=Path(__file__),
                  config_slice_declared=True, include_driver_script_in_hash=False) as cell:
        rng = random.Random(seed)
        env = _make_env(seed, full_config)
        n_actions = env.action_dim
        agent = _make_agent(env, full_config)
        optimizer = optim.Adam(list(agent.parameters()), lr=full_config["lr"])
        agent.train()

        fresh_hist: Counter = Counter()
        exec_hist: Counter = Counter()
        per_episode_fresh: List[Dict[str, int]] = []
        survival: List[int] = []
        died: List[bool] = []
        harm_by_key: Counter = Counter()
        harm_events = 0
        harm_on_held = 0
        harm_unattributed = 0
        n_fresh = 0
        n_latched = 0
        n_e3_used = 0
        steps_total = 0
        fallback_steps = 0
        governing_key: Optional[Tuple[int, int, int]] = None
        n_pre_fresh_governing = 0
        n_fresh_fallback = 0

        for ep in range(n_episodes):
            env = _make_env(seed, full_config)          # pinned layout AND start, every episode
            _, obs_dict = env.reset()
            agent.reset()
            governing_key = None
            ep_fresh: Counter = Counter()
            steps_this = 0
            ended_by_death = False
            for step_idx in range(n_steps):
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                ticks = agent.clock.advance()
                z_world_curr = latent.z_world.detach()
                pre_x, pre_y = int(env.agent_x), int(env.agent_y)
                action, _span, used = _select_action(agent, latent, ticks, n_actions, rng)
                if used:
                    n_e3_used += 1
                else:
                    fallback_steps += 1
                a_idx = int(torch.argmax(action, dim=-1).item())
                # A FRESH decision is an e3_tick OR the first step of an episode: after agent.reset()
                # `_last_action` is None, so select_action's held branch (agent.py:7192,
                # `if not ticks["e3_tick"] and self._last_action is not None`) is skipped and E3
                # deliberates even though the clock reports e3_tick False (red-team F2). Only
                # actions that came FROM E3 (`used`) enter the repertoire histogram (F6).
                is_fresh = bool(ticks.get("e3_tick", True)) or (step_idx == 0)
                if is_fresh and used:
                    n_fresh += 1
                    fresh_hist[a_idx] += 1
                    ep_fresh[a_idx] += 1
                    governing_key = (pre_x, pre_y, a_idx)
                    if step_idx == 0:
                        n_pre_fresh_governing += 1
                elif is_fresh and not used:
                    n_fresh_fallback += 1
                    governing_key = (pre_x, pre_y, a_idx)
                else:
                    n_latched += 1
                exec_hist[a_idx] += 1

                _, harm_signal, done, info, obs_dict = env.step(action)
                steps_total += 1
                steps_this += 1
                if float(harm_signal) < 0:
                    harm_events += 1
                    if governing_key is not None:
                        harm_by_key[governing_key] += 1
                        if not is_fresh:
                            harm_on_held += 1
                    else:
                        harm_unattributed += 1
                    agent.residue_field.accumulate(z_world_curr, harm_magnitude=abs(float(harm_signal)))

                loss = agent.compute_prediction_loss() + agent.compute_e2_loss()
                if loss.requires_grad:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    optimizer.step()
                if done:
                    ended_by_death = True   # `done` before the step budget = the env terminated the episode (agent_health <= 0)
                    break
            survival.append(steps_this)
            died.append(bool(ended_by_death and steps_this < n_steps))
            per_episode_fresh.append({str(k): int(v) for k, v in ep_fresh.items()})
            if (ep + 1) % 5 == 0 or ep == n_episodes - 1:
                print(f"  [train] seed={seed} ep {ep+1}/{n_episodes} steps_this={steps_this} "
                      f"harm={harm_events} fresh={n_fresh} fresh_hist={dict(fresh_hist)}", flush=True)

        planned = n_episodes * n_steps
        top_share = (max(fresh_hist.values()) / sum(fresh_hist.values())) if fresh_hist else float("nan")
        top_class = (max(fresh_hist, key=fresh_hist.get) if fresh_hist else None)
        exec_top_share = (max(exec_hist.values()) / sum(exec_hist.values())) if exec_hist else float("nan")
        row: Dict[str, Any] = {
            "arm_id": "E3_CANONICAL_RESIDUE_INTACT", "seed": int(seed),
            "fresh_action_class_hist": {str(k): int(v) for k, v in sorted(fresh_hist.items())},
            "executed_step_action_class_hist": {str(k): int(v) for k, v in sorted(exec_hist.items())},
            "per_episode_fresh_hist": per_episode_fresh,
            "fresh_top_class": (int(top_class) if top_class is not None else None),
            "fresh_top_class_share": float(top_share),
            "fresh_n_classes": int(len(fresh_hist)),
            "latched": bool(top_share == top_share and top_share >= SINGLE_CLASS_SHARE),
            "executed_top_class_share": float(exec_top_share),
            "survival_curve_steps_per_episode": survival,
            "episodes_ended_early": int(sum(died)),
            "mean_steps_per_episode": float(statistics.fmean(survival)) if survival else 0.0,
            "steps_realized_frac": float(steps_total / max(1, planned)),
            "harm_events": int(harm_events), "harm_rate": float(harm_events / max(1, steps_total)),
            "harm_on_held_ticks": int(harm_on_held), "harm_unattributed": int(harm_unattributed),
            "n_episode_initial_deliberations_counted_fresh": int(n_pre_fresh_governing),
            "n_fresh_ticks_with_hippocampal_fallback_excluded": int(n_fresh_fallback),
            "adjudicable": bool(n_fresh >= FLOOR_FRESH_DECISIONS),
            "casualty": bool(survival and (sum(died) / len(survival)) >= CASUALTY_EARLY_END_FRACTION),
            "harm_attribution_coverage": float((harm_events - harm_unattributed) / harm_events) if harm_events else 1.0,
            "n_distinct_erred_keys_governing_registration": int(len(harm_by_key)),
            "harm_by_governing_key_top20": [[list(k), int(v)] for k, v in harm_by_key.most_common(20)],
            "n_fresh_select": int(n_fresh), "n_latched": int(n_latched),
            "fresh_select_yield": float(n_fresh / max(1, n_fresh + n_latched)),
            "e3_used_fraction": float(n_e3_used / max(1, steps_total)),
            "hippo_fallback_steps": int(fallback_steps),
            "residue_total": float(agent.residue_field.total_residue.item()),
            "residue_harm_events": int(agent.residue_field.num_harm_events.item()),
            "planned_total_steps": int(planned), "realized_total_steps": int(steps_total),
        }
        cell.stamp(row)
    print(f"verdict: {'PASS' if row['n_fresh_select'] >= 1 else 'FAIL'}", flush=True)
    return row, agent


def run_control_policies(seed: int, full_config: Dict[str, Any], n_episodes: int, n_steps: int,
                         ) -> Dict[str, Any]:
    """The red team's yardstick, on the same pinned board: random walk, stay, each constant
    action; CONTROL_EPISODES each; survival + harm rate. Policy-free, own RNG."""
    out: Dict[str, Any] = {}
    for policy in CONTROL_POLICIES:
        rng = random.Random(seed * 100 + CONTROL_POLICIES.index(policy))
        surv: List[int] = []
        harms = 0
        steps = 0
        for _ep in range(n_episodes):
            env = _make_env(seed, full_config)
            _, obs_dict = env.reset()
            n_actions = env.action_dim
            stay_idx = n_actions - 1   # CausalGridWorldV2: the last action is stay (0,0)
            steps_this = 0
            for _ in range(n_steps):
                if policy == "RANDOM_WALK":
                    a = rng.randrange(n_actions)
                elif policy == "STAY":
                    a = stay_idx
                else:
                    a = int(policy.split("_")[1])
                action = torch.zeros(1, n_actions)
                action[0, a] = 1.0
                _, harm_signal, done, _info, obs_dict = env.step(action)
                steps += 1
                steps_this += 1
                if float(harm_signal) < 0:
                    harms += 1
                if done:
                    break
            surv.append(steps_this)
        out[policy] = {"survival_steps_per_episode": surv, "mean_steps": float(statistics.fmean(surv)),
                       "survives_full_budget_all_episodes": bool(all(s >= n_steps for s in surv)),
                       "harm_rate": float(harms / max(1, steps)), "n_episodes": n_episodes}
    return out


def run(seeds: Tuple[int, ...], dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    n_episodes = 3 if dry_run else TRAIN_EPISODES
    n_steps = 20 if dry_run else STEPS_PER_EPISODE
    c_eps = 1 if dry_run else CONTROL_EPISODES
    full_config: Dict[str, Any] = {
        "env": "CausalGridWorldV2", "size": 6, "num_hazards": 4, "num_resources": 3,
        "hazard_harm": 0.02, "proximity_harm_scale": 0.05, "use_proxy_fields": True,
        "self_dim": 32, "world_dim": 32, "alpha_world": 0.9, "alpha_self": 0.3, "lr": 1e-3,
        "num_candidates": NUM_CANDIDATES, "train_episodes": n_episodes, "steps_per_episode": n_steps,
        "unified_latent_mode": False, "reafference_action_dim": 0,
        "selection_rule": "argmin_over_candidate_residue_scores (983a verbatim)",
        "residue": "intact (983a A0 semantics)", "lineage": LINEAGE,
        "control_policies": CONTROL_POLICIES, "control_episodes": c_eps,
        # readout-affecting constants the cell's call graph reads (config_slice declaration lint)
        "single_class_share": SINGLE_CLASS_SHARE, "latched_fraction_split": LATCHED_FRACTION_SPLIT,
        "floor_fresh_decisions": FLOOR_FRESH_DECISIONS, "floor_e3_used": FLOOR_E3_USED,
        "casualty_early_end_fraction": CASUALTY_EARLY_END_FRACTION, "min_adjudicable_seeds": MIN_ADJUDICABLE_SEEDS,
        "min_casualty_seeds_for_clause": MIN_CASUALTY_SEEDS_FOR_CLAUSE, "min_seeds_non_degenerate": MIN_SEEDS_NON_DEGENERATE,
    }
    rows: List[Dict[str, Any]] = []
    agents: List[REEAgent] = []
    controls: Dict[str, Any] = {}
    for seed in seeds:
        row, agent = run_agent_cell(seed, full_config, n_episodes, n_steps)
        rows.append(row)
        agents.append(agent)
        controls[str(seed)] = run_control_policies(seed, full_config, c_eps, n_steps)

    # ---- readiness (diagnostic adjudication gate) --------------------------------------
    # PER-SEED adjudicability, never a whole-run min-veto (red-team F1: a min-over-seeds floor
    # on fresh decisions excludes exactly the casualty seeds the spike exists to measure).
    adjudicable = [r for r in rows if r["adjudicable"]]
    n_adjudicable = len(adjudicable)
    e3_worst = min(r["e3_used_fraction"] for r in rows)
    attrib_worst = min(r["harm_attribution_coverage"] for r in rows)
    controls_ok = all(all(controls[str(r["seed"])][p]["n_episodes"] == c_eps for p in CONTROL_POLICIES) for r in rows)
    preconditions = [
        {"name": "adjudicable_seeds_sufficient", "kind": "readiness",
         "description": f"count of seeds with >= {FLOOR_FRESH_DECISIONS} fresh E3 decisions (episode-initial deliberations counted; fallback ticks excluded)",
         "control": "a repertoire read off a handful of decisions is not a repertoire; seeds below the floor are recorded as unadjudicated",
         "measured": float(n_adjudicable), "threshold": float(MIN_ADJUDICABLE_SEEDS), "comparator": ">=", "direction": "lower",
         "offending_cell": "seeds " + ",".join(str(r["seed"]) for r in rows if not r["adjudicable"]),
         "met": bool(n_adjudicable >= MIN_ADJUDICABLE_SEEDS)},
        {"name": "e3_selection_used", "kind": "readiness",
         "description": "min over seeds of the fraction of executed steps whose action came from E3 (not the no-candidates fallback)",
         "control": "hippocampal proposals present", "measured": float(e3_worst), "threshold": FLOOR_E3_USED, "comparator": ">=", "direction": "lower",
         "met": bool(e3_worst >= FLOOR_E3_USED)},
        {"name": "harm_attribution_complete", "kind": "readiness",
         "description": "min over seeds of the fraction of harm events attributed to a governing fresh decision (held-tick harms included) -- 1.0 by construction; recorded so a regression is visible",
         "control": "governing-key registration", "measured": float(attrib_worst), "threshold": 1.0, "comparator": ">=", "direction": "lower",
         "met": bool(attrib_worst >= 1.0)},
        {"name": "controls_ran_on_every_board", "kind": "readiness",
         "description": "every control policy ran its episodes on every board", "control": "yardstick completeness",
         "measured": 1.0 if controls_ok else 0.0, "threshold": 1.0, "comparator": ">=", "direction": "lower", "met": bool(controls_ok)},
    ]
    ready = all(p["met"] for p in preconditions)

    # ---- pre-registered readout (over ADJUDICABLE seeds) -----------------------------------
    shares = [r["fresh_top_class_share"] for r in adjudicable]
    n_latched_seeds = sum(1 for r in adjudicable if r["latched"])
    latched_fraction = n_latched_seeds / max(1, n_adjudicable)
    # red-team F3: the pooled fraction is survival-blind. The autopsy's hypothesis is that the
    # CASUALTY seeds are the latched ones, so the routing also reads latching AMONG CASUALTIES.
    casualties = [r for r in adjudicable if r["casualty"]]
    n_casualty_latched = sum(1 for r in casualties if r["latched"])
    casualty_latched_fraction = (n_casualty_latched / len(casualties)) if casualties else float("nan")
    latching_dominant = bool(latched_fraction >= LATCHED_FRACTION_SPLIT
                             or (len(casualties) >= MIN_CASUALTY_SEEDS_FOR_CLAUSE and casualty_latched_fraction >= LATCHED_FRACTION_SPLIT))
    # red-team F4: identical shares (e.g. every seed at 1.0) is the hypothesis's own predicted
    # outcome, not degeneracy; non-degeneracy is adjudicability, not spread.
    non_degenerate = bool(ready and n_adjudicable >= MIN_SEEDS_NON_DEGENERATE)
    # secondary: survival x latching association against the control profiles
    surv_assoc = []
    for r in rows:
        c = controls[str(r["seed"])]
        surv_assoc.append({"seed": r["seed"], "latched": r["latched"], "top_class": r["fresh_top_class"],
                           "agent_mean_steps": r["mean_steps_per_episode"], "agent_harm_rate": r["harm_rate"],
                           "stay_mean_steps": c["STAY"]["mean_steps"], "random_mean_steps": c["RANDOM_WALK"]["mean_steps"],
                           "constant_mean_steps": {p: c[p]["mean_steps"] for p in CONTROL_POLICIES if p.startswith("CONSTANT")},
                           "control_harm_rates": {p: c[p]["harm_rate"] for p in CONTROL_POLICIES},
                           "adjudicable": r["adjudicable"], "casualty": r["casualty"],
                           # red-team F5: match on (survival, harm rate) jointly -- harm rate is what the autopsy matched on
                           "closest_control_profile": min(CONTROL_POLICIES, key=lambda p: math.hypot(
                               (c[p]["mean_steps"] - r["mean_steps_per_episode"]) / max(1, n_steps),
                               c[p]["harm_rate"] - r["harm_rate"]))})
    if not ready:
        label = "substrate_not_ready_requeue"
    elif latching_dominant:
        label = "latching_dominant_actor_adequacy_gated"
    else:
        label = "repertoire_exists_stage2_proceeds"

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_v3",
        "queue_id": QUEUE_ID, "experiment_type": EXPERIMENT_TYPE, "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS, "bears_on": BEARS_ON, "experiment_purpose": EXPERIMENT_PURPOSE,
        "lineage": LINEAGE,
        "outcome": "PASS" if (ready and non_degenerate) else "FAIL",
        "evidence_direction": "diagnostic_no_direction",
        "non_degenerate": non_degenerate,
        "degeneracy_reason": ("" if non_degenerate else ("readiness unmet" if not ready else "too few adjudicable seeds")),
        "machine_class_note": ("E3's uncommitted path draws torch.multinomial, which is not bit-identical across machine classes; "
                               "983a ran on linux-x86_64. The readout (per-seed repertoire class share) is reported with machine_class "
                               "so a cross-class replication is not read as the same trajectory (red-team F7)."),
        "interpretation": {"label": label, "preconditions": preconditions,
                           "criteria_non_degenerate": {"latched_fraction_readout": non_degenerate}, "gate_green": ready},
        "criteria": [{"name": "latched_fraction_readout", "load_bearing": True,
                      "passed": bool(ready and not latching_dominant),
                      "description": (f"over ADJUDICABLE seeds: fraction whose fresh-decision repertoire is a single class (share >= {SINGLE_CLASS_SHARE}) "
                                      f">= {LATCHED_FRACTION_SPLIT}, OR (>= {MIN_CASUALTY_SEEDS_FOR_CLAUSE} adjudicable casualty seeds and their latched fraction >= {LATCHED_FRACTION_SPLIT}) "
                                      f"-> latching dominant; passed = DECLARED NULL holds (repertoire exists, stage 2 proceeds)"),
                      "measured": latched_fraction, "casualty_latched_fraction": casualty_latched_fraction,
                      "n_latched_seeds": n_latched_seeds, "n_adjudicable": n_adjudicable, "n_casualties_adjudicable": len(casualties),
                      "n_seeds": len(rows)}],
        "readout": {"latched_fraction": latched_fraction, "casualty_latched_fraction": casualty_latched_fraction,
                    "latching_dominant": latching_dominant,
                    "latched_x_died_2x2": {"latched_casualty": sum(1 for r in adjudicable if r["latched"] and r["casualty"]),
                                           "latched_survivor": sum(1 for r in adjudicable if r["latched"] and not r["casualty"]),
                                           "diverse_casualty": sum(1 for r in adjudicable if (not r["latched"]) and r["casualty"]),
                                           "diverse_survivor": sum(1 for r in adjudicable if (not r["latched"]) and not r["casualty"]),
                                           "unadjudicated": len(rows) - n_adjudicable},
                    "per_seed_top_class_share": {str(r["seed"]): r["fresh_top_class_share"] for r in rows},
                    "per_seed_top_class": {str(r["seed"]): r["fresh_top_class"] for r in rows},
                    "per_seed_latched": {str(r["seed"]): r["latched"] for r in rows},
                    "survival_latching_association": surv_assoc},
        "stage2_routing": ("STAGE 2 (residue-frozen ablation with a hold-aware revisit-AVOIDANCE DV) is queued by the session "
                           "that reads this manifest ONLY under the declared null (repertoire exists); under the alternative "
                           "it is gated on zworld_actor_adequacy_locus / MECH-269 / MECH-341 / ARC-065 and reported to governance."),
        "control_policies": controls,
        "arm_results": rows,
        "pre_registered_thresholds": {"single_class_share": SINGLE_CLASS_SHARE, "latched_fraction_split": LATCHED_FRACTION_SPLIT,
                                      "floor_fresh_decisions": FLOOR_FRESH_DECISIONS, "floor_e3_used": FLOOR_E3_USED,
                                      "min_seeds_non_degenerate": MIN_SEEDS_NON_DEGENERATE},
        "ethics_preflight": {"involves_negative_valence": False, "involves_suffering_like_state": False, "involves_self_model": False,
                             "involves_inescapability_or_helplessness": False, "involves_offline_replay_over_harm": False,
                             "involves_social_mind_or_language": False, "involves_human_data_or_clinical_context": False, "decision": "allow"},
        "dry_run": bool(dry_run),
    }
    stamp_recording_core(manifest, config=full_config, seeds=list(seeds), script_path=Path(__file__), started_at=t0, agent=agents)
    print(f"\n[V3-EXQ-1014] label={label} latched_fraction={latched_fraction:.3f} ({n_latched_seeds}/{n_adjudicable} adjudicable of {len(rows)}) "
          f"casualty_latched={casualty_latched_fraction} ready={ready} non_degenerate={non_degenerate}", flush=True)
    for a in surv_assoc:
        print(f"  seed={a['seed']} latched={a['latched']} top={a['top_class']} agent_steps={a['agent_mean_steps']:.1f} "
              f"closest_control={a['closest_control_profile']}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    seeds = tuple(args.seeds[:2]) if (args.dry_run and args.seeds == list(SEEDS)) else tuple(args.seeds)
    _t = time.perf_counter()
    manifest = run(seeds, args.dry_run)
    out_path = write_flat_manifest(manifest, None, dry_run=args.dry_run, config=manifest.get("config"),
                                   seeds=list(seeds), script_path=Path(__file__), started_at=_t)
    print(f"\nResult written to: {out_path}", flush=True)
    for p in manifest["interpretation"]["preconditions"]:
        print(f"  precondition {p['name']:<34} met={p['met']} measured={p['measured']}", flush=True)
    _outcome = str(manifest.get("outcome", "FAIL")).upper()
    emit_outcome(outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL", manifest_path=out_path,
                 queue_id=QUEUE_ID, dry_run=args.dry_run)
