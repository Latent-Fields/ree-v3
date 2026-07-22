"""V3-EXQ-808 -- Return-decomposition diagnostic: is the training objective the scored objective?

GOV-FANOUT-1 discriminating probe for the leg `H-objective-misspecification`
(axis `reward`) of the frozen-ledger question `conversion_ceiling_root`, pre-registered
2026-07-22 from the confirmed cluster autopsy
`failure_autopsy_competence-objective-cluster-734-737b-742a_2026-07-22`.

THE FINDING THIS PROBES. At the hazard-free D3 rung every learner scores BELOW its own
declared random-walk anchor of 0.933 res/ep -- ppo_ree_latent 0.233, ppo_raw_obs 0.567,
the four actor-critic arms 0.200-0.267, the REE bias head 0.533 -- while a local-view
greedy reading the SAME 5x5 field scores 48.05 and the global-info greedy oracle 57.2.
At D0, vanilla PPO survives 175.0 steps against the greedy oracle's 20.4 while foraging
17x less. A trained learner below its own random floor has not been shown to lack a
mechanism; it has been shown to be maximising something else. The autopsy's reading:
the training return is dominated by episode survival, survival is maximised by NOT
foraging, and every learner is therefore correctly optimising an objective that is not
the one the DV scores.

THE MEASUREMENT GAP THIS CLOSES. No run in the 724/734/737/742 driver family records
PER-TERM RETURN ATTRIBUTION, so "which term is the learner actually maximising" is
unanswerable from any existing manifest (autopsy learning #5; GOV-REUSE-1 check below).
This is a MEASUREMENT gap, not a recording gap -- the metric was never computed.

=== HYPOTHESES UNDER TEST ===

H-objective-misspecification (this probe's leg, axis `reward`):
    The realised training return at D3 is dominated by survival-linked per-step accrual
    (proximity-to-resource benefit, novelty, ambient) rather than by consumption, so the
    optimal policy under the given reward is not to forage. Re-weighting consumption
    against the survival-linked terms should move D3 foraging competence.

H-policy-learning (INCUMBENT; stays alive -- see below):
    The policy-learning stage genuinely cannot convert representation into competent
    action, independent of the objective.

DECLARED NULL (pre-registered; the result that REFUTES this leg):
    The consumption term ALREADY dominates the realised return at the training horizon
    (consumption_share >= CONSUMPTION_DOMINANCE_THRESHOLD at the baseline weighting) AND
    re-weighting the consumption:survival ratio does NOT move D3 foraging competence
    (cross-level range < COMPETENCE_MOVE_DELTA). Under that reading the objective is NOT
    the binding constraint, H-objective-misspecification is refuted, and the
    observability / policy-learning route re-opens. A null here is INFORMATIVE, not
    wasted: it is the only result that licenses eliminating this leg.

    NOTE the autopsy's explicit instruction, which this script honours: a null on THIS
    probe eliminates THIS leg only. It does not eliminate H-policy-learning, whose own
    discriminating probe (737b's PPO-on-latent arm re-run under a corrected objective)
    is a separate leg on a different design axis.

=== INTERPRETATION GRID ===

Pre-registered self-route (a HYPOTHESIS, not a verdict -- adjudicate via /failure-autopsy
before any governance use). This experiment PROMOTES AND DEMOTES NOTHING.

  READINESS gate red (any applicable precondition unmet on every level)
      -> `substrate_not_ready_requeue`. NEVER a substrate/objective verdict label.

  C1 False (survival-linked terms dominate) AND C2 True (re-weighting moves competence)
      -> `objective_misspecification_confirmed`  [PASS]
         The scored task and the trained task differ, and correcting the ratio recovers
         foraging. Route: re-specify the training objective before any further
         representation or policy-learning build on this family.

  C1 False AND C2 False
      -> `objective_dominant_but_reweighting_insufficient`  [FAIL]
         The return IS survival-dominated, but the sweep's dynamic range did not recover
         competence. The leg stays ALIVE; a wider sweep or a structural reward change is
         indicated, and H-policy-learning is NOT thereby confirmed.

  C1 True AND C2 False
      -> `objective_misspecification_refuted`  [FAIL]  <-- THE DECLARED NULL
         Consumption already dominates and the ratio is not load-bearing. Eliminate this
         leg; re-open the observability / policy-learning route.

  C1 True AND C2 True
      -> `reweighting_moves_competence_without_survival_dominance`  [FAIL]
         Surprising: the ratio matters although consumption already dominated. The
         dominance statistic and the manipulation disagree; do not read either as a
         verdict without autopsy.

=== DV-SYMMETRY DECLARATION (mandatory; failure_autopsy_V3-EXQ-604c section 3) ===

DV: `foraging_competence` (mean resources consumed per episode) under a GREEDY (argmax)
evaluation of the trained PPO actor. Its symmetry group, as a function of the TRAINING
reward, contains: (i) any positive uniform rescaling of the whole per-step reward -- PPO
here normalises rewards by a running standard deviation (x734._RunningStd), so a uniform
scale is annihilated EXACTLY; and (ii) transforms that leave the per-state ordering of
returns unchanged.

Per arm, the manipulation and its invariance:

  W0_baseline        (w_consume 1.0, w_survival 1.0) -- the reference weighting; it IS
      the 737b/742a reward verbatim. Not a manipulation; no invariance claim needed.
  W1_consume_x5      (5.0, 1.0)  -- reweights consumption RELATIVE to the survival-linked
      terms. Not invariant under (i): it is a ratio change, not a uniform scale, so it
      survives the running-std normalisation. Not invariant under (ii): consumption
      events occur at some states and not others, so the per-state ordering changes.
  W2_consume_x25     (25.0, 1.0) -- same argument, larger dose.
  W3_survival_zeroed (1.0, 0.0)  -- DELETES the survival-linked terms. Not expressible as
      any rescaling of the baseline reward, so invariance under (i) is impossible.

  THE TRAP THIS DESIGN MUST NOT FALL INTO, stated explicitly. If at D3 the survival-linked
  terms happen to be identically zero (no live proximity benefit, no novelty, no ambient),
  then scaling `w_consume` scales the WHOLE reward uniformly -- and PPO's reward
  normalisation makes every level bit-equivalent to the baseline. The measured cross-level
  delta would then be an arithmetic identity fixed before the run, exactly the 604c class:
  the arms fire, the gate would read green, and nothing would have been measured.
  Precondition P6 (`weighting_sweep_changes_realised_composition`) exists solely to catch
  this: it asserts a non-trivial cross-level RANGE of `consumption_share` -- the SAME
  statistic C1 routes on and the statistic in which a uniform rescale provably cancels.
  If P6 is unmet the run self-routes `substrate_not_ready_requeue`, NOT a verdict.

=== RETURN DECOMPOSITION ===

The per-step training reward in this family (x734 / 737 / 742) is

    shaped = harm_signal + FORAGE_BONUS*[transition_type == "resource"] + novelty_bonus(pos)

`harm_signal` is a single env scalar whose meaning is fixed by `info["transition_type"]`,
so the attribution is exact rather than inferred:

  consumption : env harm_signal on transition_type "resource" (contact_benefit)
                + the driver's FORAGE_BONUS shaping
  proximity   : "benefit_approach" (per-step reward for being NEAR a resource without
                consuming it -- the hovering incentive), "harm_gradient", "zone_c_ambient"
  harm        : "env_caused_hazard", "agent_caused_hazard", "hazard_approach",
                "env_caused_multisource" (negative; ~0 at D3 by construction, num_hazards=0)
  novelty     : the driver's count-based exploration bonus
  other       : everything else ("none", "action_blocked", "waypoint", "sequence_complete")

SURVIVAL-LINKED := proximity + novelty + other. These accrue per step for merely staying
alive and are what `w_survival` scales. `harm` is never re-weighted (it is the penalty
side, not the survival incentive) and `survival_horizon` is recorded separately as the
behavioural counterpart.

The reweighted reward actually optimised at each level:

    shaped = w_consume * consumption + w_survival * (proximity + novelty + other) + harm

=== LEARNER HELD FIXED ===

One all-ON REE stack is warmed PER SEED with the 724-A0 recipe (x734._train_all_on_agent)
ON THE SD-070 z_world ENCODER-WARMUP PATH (`zworld_p0_episodes > 0`), and that single
frozen agent supplies z_world to every weighting level. Only the reward changes across
levels; the representation, the actor architecture, the optimiser and the budget are
identical. Sharing the warmed agent is legitimate here precisely because it is frozen --
z_world is read under `torch.no_grad()` and no gradient reaches the stack during PPO.

ZWORLD ENCODER GUARD: GREEN-GATING (not detection-only). 734 and 742a both ran with
`encoder_moved_in_p0 false` -- the `sd_zworld_warmup_optimizer_group` adoption had not
reached their driver copies -- and their z_world arms were therefore PPO on a frozen
random projection. This driver refuses to draw a conclusion in that state: the guard is a
GATING precondition (P1), so an unmoved encoder self-routes `substrate_not_ready_requeue`
rather than an objective verdict. It is a gating PRECONDITION rather than a raise so the
run lands an interpretable manifest and routes to /failure-autopsy instead of ERRORing out
of the queue.

PURPOSE: diagnostic. `claim_ids = []` -- this probe discriminates WHY the family fails; it
tests no claim's mechanism, promotes and demotes nothing, and is excluded from governance
confidence/conflict scoring. MECH-457 is deliberately NOT tagged: the autopsy records that
MECH-457 already carries 19 prior confirmed autopsy targets and is under a
`/claim-synthesis` granularity-debt recommendation, and it explicitly asks that no further
ceiling reading be attributed to it from this cluster.

GOV-REUSE-1 (Step 2.4): `reanalysis_query.py query --readout return_decomposition /
reward_term_attribution / consumption_share / survival_return_share` returns NO MATCH on
any compatible substrate_hash across the corpus (only UNVERIFIABLE pre-standard rows,
including 742's own manifest). Consistent with autopsy learning #5: the decomposition was
never computed by any run in this family. Not recoverable -> run.

This module is ASCII-only in all runtime strings.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments._lib.capability_eval import (  # noqa: E402
    COMPETENCE_RESOURCE_FLOOR,
    LocalViewGreedyPolicy,
    OraclePolicy,
    Policy,
    RandomPolicy,
    evaluate_seed,
)
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    assert_world_encoder_trained,
    latent_stack_snapshot,
    latent_stack_weight_delta,
)
import experiments.v3_exq_724_competence_localization_diagnostic as x724  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_737_ree_latent_policy_head_competence_probe as x737  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_808_return_decomposition_objective_misspecification"
QUEUE_ID = "V3-EXQ-808"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# Multi-arm (weighting levels x seeds) but every cell trains a PPO actor from scratch on a
# per-seed re-warmed REE stack. Nothing is a pure function of a shared baseline config that
# a later run could reuse, so there is no baseline arm to bank.
ARM_FINGERPRINT_EXEMPT = (
    "per-seed REE warmup + per-level PPO training from scratch; no reusable baseline arm to bank"
)

DEVICE = torch.device("cpu")

SEEDS: List[int] = [42, 43, 44]

# Budget sourced from x734 so this driver cannot drift from the family it compares against.
ZWORLD_P0_EPISODES = x734.ZWORLD_P0_EPISODES        # 60; SD-070 z_world encoder warmup (P0a)
P0_WARMUP_EPISODES = x734.P0_WARMUP_EPISODES        # 200
P1_REINFORCE_EPISODES = x734.P1_REINFORCE_EPISODES  # 90
P1_PPO_EPISODES = x734.P1_PPO_EPISODES              # 1000
EVAL_EPISODES = x734.EVAL_EPISODES                  # 20
STEPS_PER_EPISODE = x734.STEPS_PER_EPISODE          # 200
PPO_ROLLOUT_EPISODES = x734.PPO_ROLLOUT_EPISODES    # 8

DRY_RUN_SEEDS = [42]
DRY_RUN_ZWORLD_P0 = x734.DRY_RUN_ZWORLD_P0
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_PPO = 6
DRY_RUN_EVAL = 2
DRY_RUN_STEPS = 20
DRY_RUN_ROLLOUT = 3

# The decisive rung: hazard-free, oracle-achievable, hazard confound removed. Imported from
# 734 so it is byte-identical to the sibling sweep this probe is reading against.
RUNG = x734.DIFFICULTY_RUNGS[-1]
RUNG_ID = RUNG["rung_id"]

# --------------------------------------------------------------------------------------
# PRE-REGISTERED THRESHOLDS (constants; never derived from this run's own statistics)
# --------------------------------------------------------------------------------------
# C1: consumption "dominates" the realised return at a level.
CONSUMPTION_DOMINANCE_THRESHOLD = 0.50
# C2: a cross-level movement in D3 foraging competence that counts as the ratio biting.
# Half a resource per episode, against a competence floor of 1.0 and an observed family
# spread of 0.200-0.567 res/ep -- i.e. larger than the entire spread the family has shown.
COMPETENCE_MOVE_DELTA = 0.50
# P4: the share denominator must be non-trivial or every share is undefined.
ATTRIBUTED_RETURN_MAGNITUDE_FLOOR = 1e-3
# P5: a term family counts as "carrying" the return at |share| >= this.
TERM_FAMILY_NONTRIVIAL_SHARE = 0.01
N_TERM_FAMILIES_FLOOR = 1.5          # i.e. strictly more than one family
# P6: the sweep must actually move the realised composition (the DV-symmetry guard).
COMPOSITION_RANGE_FLOOR = 0.02
# P1: the z_world encoder must have moved in P0.
ZWORLD_DELTA_FLOOR = 1e-6

# --------------------------------------------------------------------------------------
# Weighting sweep. w_consume scales the consumption terms; w_survival scales the
# survival-linked per-step accrual (proximity + novelty + other). harm is never reweighted.
# Four levels: the baseline reference, two graded doses (dose-response readable), and one
# structural endpoint that deletes the survival-linked terms entirely.
# --------------------------------------------------------------------------------------
WEIGHTINGS: List[Dict[str, Any]] = [
    {"id": "W0_baseline", "w_consume": 1.0, "w_survival": 1.0,
     "role": "reference -- the 737b/742a reward verbatim"},
    {"id": "W1_consume_x5", "w_consume": 5.0, "w_survival": 1.0, "role": "dose"},
    {"id": "W2_consume_x25", "w_consume": 25.0, "w_survival": 1.0, "role": "dose"},
    {"id": "W3_survival_zeroed", "w_consume": 1.0, "w_survival": 0.0,
     "role": "structural endpoint -- survival-linked terms deleted"},
]
BASELINE_LEVEL_ID = WEIGHTINGS[0]["id"]
LEVEL_IDS = [w["id"] for w in WEIGHTINGS]

ANCHOR_IDS = ["random_walk", "local_view_greedy", "greedy_oracle"]

# --------------------------------------------------------------------------------------
# Term-family attribution. transition_type is the env's own label for what the single
# harm_signal scalar meant on that tick, so this mapping is exact, not inferred.
# --------------------------------------------------------------------------------------
CONSUMPTION_TYPES = frozenset({"resource"})
HARM_TYPES = frozenset({
    "env_caused_hazard", "agent_caused_hazard", "hazard_approach", "env_caused_multisource",
})
PROXIMITY_TYPES = frozenset({"benefit_approach", "harm_gradient", "zone_c_ambient"})
TERM_FAMILIES = ("consumption", "proximity", "harm", "novelty", "other")
SURVIVAL_LINKED_FAMILIES = ("proximity", "novelty", "other")


def _family_for(ttype: str) -> str:
    if ttype in CONSUMPTION_TYPES:
        return "consumption"
    if ttype in HARM_TYPES:
        return "harm"
    if ttype in PROXIMITY_TYPES:
        return "proximity"
    return "other"


def _new_terms() -> Dict[str, float]:
    return {f: 0.0 for f in TERM_FAMILIES}


def _weighted(terms: Dict[str, float], w_consume: float, w_survival: float) -> Dict[str, float]:
    """Apply the level's weights. harm is never reweighted (penalty, not survival incentive)."""
    out = dict(terms)
    out["consumption"] = terms["consumption"] * float(w_consume)
    for fam in SURVIVAL_LINKED_FAMILIES:
        out[fam] = terms[fam] * float(w_survival)
    return out


def _shares(weighted_terms: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """Return (share-by-family, total attributed magnitude).

    Shares are computed on ABSOLUTE contributions so a negative harm term cannot cancel a
    positive consumption term into a meaningless denominator.
    """
    mags = {f: abs(float(weighted_terms[f])) for f in TERM_FAMILIES}
    total = float(sum(mags.values()))
    if total <= 0.0:
        return {f: 0.0 for f in TERM_FAMILIES}, 0.0
    return {f: mags[f] / total for f in TERM_FAMILIES}, total


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _rng(vals: List[float]) -> float:
    return float(max(vals) - min(vals)) if vals else 0.0


def _worst_cell(rows: List[Dict[str, Any]], key: str, mode: str = "min") -> Tuple[float, str]:
    """Extremum plus the offending cell id, so a precondition `measured` is recomputable.

    A mean would not be: `met` below quantifies over cells, and an in-band mean can mask a
    single out-of-band cell -- the exact shape validate_experiments flags.
    """
    if not rows:
        return 0.0, ""
    pick = min(rows, key=lambda r: float(r[key])) if mode == "min" \
        else max(rows, key=lambda r: float(r[key]))
    return float(pick[key]), str(pick.get("cell_id", ""))


# --------------------------------------------------------------------------------------
# PPO trainer with per-term return attribution + level reweighting.
#
# Structurally identical to x737._train_ppo_generic (same _RunningStd normalisation, same
# GAE, same x734._ppo_update) so the learner is held fixed against the sibling it reads
# against. The ONLY differences: the per-step reward is assembled from weighted term
# families instead of a flat sum, and the unweighted/weighted terms are accumulated.
# --------------------------------------------------------------------------------------
def _train_ppo_decomposed(
    env: CausalGridWorldV2,
    policy: Any,
    optimiser: torch.optim.Optimizer,
    state_fn,
    on_reset,
    n_episodes: int,
    rollout_episodes: int,
    steps_per_episode: int,
    level_id: str,
    w_consume: float,
    w_survival: float,
    seed: int,
    total_denominator: int,
) -> Dict[str, Any]:
    reward_std = x734._RunningStd()
    novelty_counter: Dict[Tuple[int, int], int] = {}
    episodes_done = 0

    raw_totals = _new_terms()
    weighted_totals = _new_terms()
    ticks_total = 0

    while episodes_done < n_episodes:
        batch_states: List[torch.Tensor] = []
        batch_actions: List[int] = []
        batch_old_logp: List[float] = []
        batch_returns: List[float] = []
        batch_advantages: List[float] = []

        eps_this_batch = min(rollout_episodes, n_episodes - episodes_done)
        for _b in range(eps_this_batch):
            _flat, obs_dict = env.reset()
            if on_reset is not None:
                on_reset()
            ep_states: List[torch.Tensor] = []
            ep_actions: List[int] = []
            ep_logp: List[float] = []
            ep_values: List[float] = []
            ep_rewards: List[float] = []
            terminal = False
            bootstrap_value = 0.0
            state = state_fn(obs_dict)
            for _step in range(steps_per_episode):
                with torch.no_grad():
                    logits, value = policy(state)
                    dist = torch.distributions.Categorical(logits=logits.reshape(1, -1))
                    a = dist.sample()
                    logp = dist.log_prob(a)
                a_idx = int(a.item())
                _flat, harm_signal, done, info, obs_dict = env.step(a_idx)
                ttype = str(info.get("transition_type", "none"))
                pos = (int(env.agent_x), int(env.agent_y))

                terms = _new_terms()
                terms[_family_for(ttype)] += float(harm_signal)
                if ttype in CONSUMPTION_TYPES:
                    terms["consumption"] += x734.FORAGE_BONUS
                terms["novelty"] += x734._novelty_bonus(novelty_counter, pos)

                wterms = _weighted(terms, w_consume, w_survival)
                shaped = float(sum(wterms.values()))

                for fam in TERM_FAMILIES:
                    raw_totals[fam] += terms[fam]
                    weighted_totals[fam] += wterms[fam]
                ticks_total += 1

                reward_std.update(shaped)
                ep_states.append(state.reshape(-1).detach())
                ep_actions.append(a_idx)
                ep_logp.append(float(logp.item()))
                ep_values.append(float(value.item()))
                ep_rewards.append(shaped)
                if done:
                    terminal = True
                    break
                state = state_fn(obs_dict)
            if not terminal:
                with torch.no_grad():
                    _logits, bv = policy(state_fn(obs_dict))
                bootstrap_value = float(bv.item())
            scale = reward_std.std + x734.REWARD_STD_EPS
            scaled_rewards = [r / scale for r in ep_rewards]
            advs, rets = x734._compute_gae(scaled_rewards, ep_values, bootstrap_value, terminal)
            batch_states.extend(ep_states)
            batch_actions.extend(ep_actions)
            batch_old_logp.extend(ep_logp)
            batch_returns.extend(rets)
            batch_advantages.extend(advs)
            episodes_done += 1
            if episodes_done % 200 == 0 or episodes_done == n_episodes:
                print(
                    f"  [train] ppo_{level_id} rung={RUNG_ID} seed={seed} phase=P1 "
                    f"ep {episodes_done}/{total_denominator}", flush=True,
                )

        if not batch_states:
            continue
        states_t = torch.stack(batch_states).to(DEVICE)
        actions_t = torch.tensor(batch_actions, dtype=torch.long, device=DEVICE)
        old_logp_t = torch.tensor(batch_old_logp, dtype=torch.float32, device=DEVICE)
        returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=DEVICE)
        adv_t = torch.tensor(batch_advantages, dtype=torch.float32, device=DEVICE)
        x734._ppo_update(
            policy, optimiser, states_t, actions_t, old_logp_t, returns_t, adv_t, DEVICE,
        )

    n_eps = max(1, episodes_done)
    shares, total_mag = _shares(weighted_totals)
    return {
        "phase": "train",
        "n_episodes": int(episodes_done),
        "n_ticks": int(ticks_total),
        "terms_raw_per_episode": {f: round(raw_totals[f] / n_eps, 6) for f in TERM_FAMILIES},
        "terms_weighted_per_episode": {
            f: round(weighted_totals[f] / n_eps, 6) for f in TERM_FAMILIES
        },
        "shares": {f: round(shares[f], 6) for f in TERM_FAMILIES},
        "attributed_magnitude_per_episode": round(total_mag / n_eps, 6),
        "consumption_share": round(shares["consumption"], 6),
        "survival_linked_share": round(
            sum(shares[f] for f in SURVIVAL_LINKED_FAMILIES), 6),
        "n_term_families_nontrivial": int(
            sum(1 for f in TERM_FAMILIES if shares[f] >= TERM_FAMILY_NONTRIVIAL_SHARE)),
    }


# --------------------------------------------------------------------------------------
# Eval-side decomposition: run the GREEDY eval policy and attribute the return it would
# have received under this level's weighting. Recorded so train-time and eval-time
# composition can be compared (a policy that changes behaviour changes its own return mix).
# --------------------------------------------------------------------------------------
def _decompose_eval(
    policy: Policy,
    env: CausalGridWorldV2,
    n_episodes: int,
    steps_per_episode: int,
    w_consume: float,
    w_survival: float,
) -> Dict[str, Any]:
    novelty_counter: Dict[Tuple[int, int], int] = {}
    raw_totals = _new_terms()
    weighted_totals = _new_terms()
    ticks_total = 0

    for _ep in range(int(n_episodes)):
        _flat, obs_dict = env.reset()
        policy.reset(env)
        for _step in range(steps_per_episode):
            action = policy.act(env, obs_dict)
            _flat, harm_signal, done, info, obs_dict = env.step(action)
            if not isinstance(info, dict):
                info = {}
            ttype = str(info.get("transition_type", "none"))
            pos = (int(env.agent_x), int(env.agent_y))

            terms = _new_terms()
            terms[_family_for(ttype)] += float(harm_signal)
            if ttype in CONSUMPTION_TYPES:
                terms["consumption"] += x734.FORAGE_BONUS
            terms["novelty"] += x734._novelty_bonus(novelty_counter, pos)
            wterms = _weighted(terms, w_consume, w_survival)
            for fam in TERM_FAMILIES:
                raw_totals[fam] += terms[fam]
                weighted_totals[fam] += wterms[fam]
            ticks_total += 1
            policy.post_step(env, info, obs_dict)
            if done:
                break

    n_eps = max(1, int(n_episodes))
    shares, total_mag = _shares(weighted_totals)
    return {
        "phase": "eval",
        "n_episodes": int(n_episodes),
        "n_ticks": int(ticks_total),
        "terms_raw_per_episode": {f: round(raw_totals[f] / n_eps, 6) for f in TERM_FAMILIES},
        "terms_weighted_per_episode": {
            f: round(weighted_totals[f] / n_eps, 6) for f in TERM_FAMILIES
        },
        "shares": {f: round(shares[f], 6) for f in TERM_FAMILIES},
        "attributed_magnitude_per_episode": round(total_mag / n_eps, 6),
        "consumption_share": round(shares["consumption"], 6),
        "survival_linked_share": round(
            sum(shares[f] for f in SURVIVAL_LINKED_FAMILIES), 6),
        "n_term_families_nontrivial": int(
            sum(1 for f in TERM_FAMILIES if shares[f] >= TERM_FAMILY_NONTRIVIAL_SHARE)),
    }


# --------------------------------------------------------------------------------------
# Preconditions. Regime-conditioned via `applies_to` (the 785 rule): a precondition that is
# structurally unsatisfiable for a level is SCOPED OUT of it, never failed by it, and a red
# level never vacates a green one.
# --------------------------------------------------------------------------------------
PRECONDITIONS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="zworld_encoder_trained_in_p0",
        description=(
            "The z_world world_encoder must have moved during P0 (SD-070 warmup path). "
            "734 and 742a both ran with it frozen, making their z_world arms PPO on a "
            "random projection; this driver refuses to draw a conclusion in that state."
        ),
        control="worst-cell world_encoder_max_abs_delta over all (seed) warmups vs 1e-6",
        threshold=ZWORLD_DELTA_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="d3_oracle_clears_floor",
        description="The hazard-free D3 env must be floor-achievable with global information.",
        control="greedy_oracle worst-seed mean resources/ep vs the 1.0 competence floor",
        threshold=float(COMPETENCE_RESOURCE_FLOOR),
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="d3_local_view_greedy_clears_floor",
        description=(
            "The env must be floor-achievable under the LEARNER'S OWN observability (the "
            "same 5x5 local field), not only under a privileged global oracle -- the 732a "
            "confound. 738 measured 48.05 here."
        ),
        control="local_view_greedy worst-seed mean resources/ep vs the 1.0 competence floor",
        threshold=float(COMPETENCE_RESOURCE_FLOOR),
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="attributed_return_magnitude_nondegenerate",
        description=(
            "The share denominator must be non-trivial: with a near-zero attributed return "
            "every share is undefined and C1 cannot discriminate."
        ),
        control="worst-cell train-phase attributed magnitude per episode at this level",
        threshold=ATTRIBUTED_RETURN_MAGNITUDE_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="n_term_families_nontrivial",
        description=(
            "At least two term families must carry a non-trivial share, or there is no "
            "composition to re-weight at this level."
        ),
        control="worst-cell count of families with share >= 0.01 at this level",
        threshold=N_TERM_FAMILIES_FLOOR,
        direction="lower",
        kind="readiness",
        # DISPOSITION (a): not MEANINGFUL for the survival-zeroed endpoint. W3 deletes the
        # survival-linked families by construction, so it can carry at most `consumption`
        # (plus a harm term that is ~0 at hazard-free D3). Asserting it there would make
        # that level structurally un-passable and collapse the 4-level sweep to 3 -- exactly
        # the V3-EXQ-785 defect. The level stays fully scorable; only this gate is scoped out.
        applies_to=lambda ctx: float(ctx.get("w_survival", 1.0)) > 0.0,
        applies_note=(
            "W3_survival_zeroed deletes the survival-linked families by design, so a "
            ">=2-families gate is structurally unsatisfiable there and is not the right "
            "question for that level"
        ),
        structural_max=lambda ctx: (1.0 if float(ctx.get("w_survival", 1.0)) == 0.0 else None),
    ),
    PreconditionSpec(
        name="weighting_sweep_changes_realised_composition",
        description=(
            "THE DV-SYMMETRY GUARD. The sweep must move the realised return composition. If "
            "the survival-linked terms are ~0 at D3 then scaling w_consume is a UNIFORM "
            "rescale of the whole reward, which PPO's running-std normalisation annihilates "
            "exactly -- every level would then be bit-equivalent to the baseline and the "
            "measured cross-level delta would be an arithmetic identity fixed before the "
            "run. Asserted on consumption_share, the SAME statistic C1 routes on and the "
            "one in which a uniform rescale provably cancels."
        ),
        control=(
            "cross-level RANGE of train-phase consumption_share; positive control is "
            "W3_survival_zeroed vs W0_baseline, which must differ by construction"
        ),
        threshold=COMPOSITION_RANGE_FLOOR,
        direction="lower",
        kind="readiness",
    ),
]


def _arm_contexts(dry_run: bool = False) -> List[Dict[str, Any]]:
    return [
        {"id": w["id"], "w_consume": float(w["w_consume"]),
         "w_survival": float(w["w_survival"]), "dry_run": bool(dry_run)}
        for w in WEIGHTINGS
    ]


# --------------------------------------------------------------------------------------
def _run_seed(
    seed: int,
    zworld_p0: int,
    p0: int,
    p1: int,
    ppo_eps: int,
    eval_eps: int,
    steps: int,
    rollout: int,
    dry_run: bool,
) -> Dict[str, Any]:
    """One seed: warm ONE all-ON stack, then train + eval a PPO actor per weighting level."""
    env_kwargs = x734._env_kwargs_for_rung(RUNG)
    total_denom = p0 + p1

    torch.manual_seed(seed)
    np.random.seed(seed)
    warm_env = x734._make_env(seed, env_kwargs)
    agent = x734._make_all_on_agent(warm_env)
    print(f"Seed {seed} Condition {RUNG_ID}:warmup_all_on", flush=True)
    before = latent_stack_snapshot(agent)
    x734._train_all_on_agent(
        agent, warm_env, seed=seed, p0_episodes=p0, p1_episodes=p1,
        steps_per_episode=steps, rung_id=RUNG_ID, total_denominator=total_denom,
        zworld_p0_episodes=zworld_p0,
        zworld_p0_env=(x734._make_env(seed, env_kwargs) if zworld_p0 > 0 else None),
        zworld_p0_dry_run=dry_run,
    )
    guard = latent_stack_weight_delta(agent, before)
    guard["seed"] = int(seed)
    guard["rung_id"] = str(RUNG_ID)
    guard["p0_episodes"] = int(p0)
    guard["zworld_p0_episodes"] = int(zworld_p0)
    # Loud, unmissable warning. NOT strict: the run must land an interpretable manifest and
    # route to /failure-autopsy, so the guard gates via precondition P1 instead of raising.
    assert_world_encoder_trained(
        agent, before, p0=p0, strict=False,
        context=f"{QUEUE_ID} rung={RUNG_ID} seed={seed}",
        escape_hint=(
            "this driver GATES on the guard via precondition zworld_encoder_trained_in_p0: "
            "an unmoved encoder self-routes substrate_not_ready_requeue, never an "
            "objective verdict"
        ),
    )

    _flat, probe_obs = x734._make_env(seed, env_kwargs).reset()
    z_dim = int(x737._agent_zworld(agent, probe_obs).shape[-1])
    action_dim = int(warm_env.action_dim)

    level_rows: List[Dict[str, Any]] = []
    for offset, wcfg in enumerate(WEIGHTINGS):
        lid = str(wcfg["id"])
        w_c = float(wcfg["w_consume"])
        w_s = float(wcfg["w_survival"])
        print(f"Seed {seed} Condition {RUNG_ID}:{lid}", flush=True)
        torch.manual_seed(seed + 1000 * (offset + 1))
        np.random.seed(seed + 1000 * (offset + 1))
        net = x734.PPOPolicyNet(in_dim=z_dim, action_dim=action_dim).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=x734.PPO_LR)
        train_env = x734._make_env(seed, env_kwargs)
        train_decomp = _train_ppo_decomposed(
            train_env, net, opt,
            state_fn=lambda od: x737._agent_zworld(agent, od),
            on_reset=agent.reset,
            n_episodes=ppo_eps, rollout_episodes=rollout, steps_per_episode=steps,
            level_id=lid, w_consume=w_c, w_survival=w_s, seed=seed,
            total_denominator=ppo_eps,
        )

        eval_policy = x737.LatentPPOEvalPolicy(net, agent)
        row = evaluate_seed(eval_policy, x734._make_env(seed, env_kwargs), eval_eps, steps)
        eval_decomp = _decompose_eval(
            x737.LatentPPOEvalPolicy(net, agent), x734._make_env(seed, env_kwargs),
            eval_eps, steps, w_c, w_s,
        )
        level_rows.append({
            "cell_id": f"{lid}|seed{seed}",
            "level_id": lid,
            "seed": int(seed),
            "w_consume": w_c,
            "w_survival": w_s,
            "foraging_competence": float(row["foraging_competence"]),
            "survival_horizon": float(row["survival_horizon"]),
            "death_rate": float(row["death_rate"]),
            "mean_episode_reward": float(row["mean_episode_reward"]),
            "competence_supra_floor": bool(row["competence_supra_floor"]),
            "per_episode_resources": row["per_episode_resources"],
            "train_decomposition": train_decomp,
            "eval_decomposition": eval_decomp,
            # promoted for readable per-cell precondition recompute
            "train_consumption_share": float(train_decomp["consumption_share"]),
            "train_attributed_magnitude": float(train_decomp["attributed_magnitude_per_episode"]),
            "train_n_term_families_nontrivial": float(
                train_decomp["n_term_families_nontrivial"]),
        })
        print(f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'}", flush=True)

    anchor_rows: List[Dict[str, Any]] = []
    anchors = {
        "random_walk": RandomPolicy(seed),
        "local_view_greedy": LocalViewGreedyPolicy(seed),
        "greedy_oracle": OraclePolicy(),
    }
    for aid in ANCHOR_IDS:
        print(f"Seed {seed} Condition {RUNG_ID}:{aid}", flush=True)
        row = evaluate_seed(anchors[aid], x734._make_env(seed, env_kwargs), eval_eps, steps)
        anchor_rows.append({
            "cell_id": f"{aid}|seed{seed}",
            "anchor_id": aid,
            "seed": int(seed),
            "foraging_competence": float(row["foraging_competence"]),
            "survival_horizon": float(row["survival_horizon"]),
            "mean_episode_reward": float(row["mean_episode_reward"]),
            "competence_supra_floor": bool(row["competence_supra_floor"]),
        })
        print(f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'}", flush=True)

    return {"seed": int(seed), "guard": guard, "levels": level_rows, "anchors": anchor_rows}


# --------------------------------------------------------------------------------------
def run_experiment(
    seeds: List[int],
    zworld_p0: int,
    p0: int,
    p1: int,
    ppo_eps: int,
    eval_eps: int,
    steps: int,
    rollout: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    contexts = _arm_contexts(dry_run=dry_run)
    # Design-time refusal BEFORE compute: catches a gate no level could pass from its own
    # pre-registered config (the V3-EXQ-785 arithmetic, for free at queue time).
    audited = assert_no_structurally_unsatisfiable_gate(PRECONDITIONS, contexts, arm_id_key="id")
    print(
        f"structural-vacuity audit: {len(audited)} (spec, level) pairs checked, "
        f"no unsatisfiable gate", flush=True,
    )

    seed_rows = [
        _run_seed(s, zworld_p0, p0, p1, ppo_eps, eval_eps, steps, rollout, dry_run)
        for s in seeds
    ]

    guards = [r["guard"] for r in seed_rows]
    for g in guards:
        g["cell_id"] = f"warmup|seed{g['seed']}"
    all_levels = [row for r in seed_rows for row in r["levels"]]
    all_anchors = [row for r in seed_rows for row in r["anchors"]]

    # ---- per-level aggregates ------------------------------------------------------
    per_level: Dict[str, Dict[str, Any]] = {}
    for lid in LEVEL_IDS:
        rows = [r for r in all_levels if r["level_id"] == lid]
        per_level[lid] = {
            "level_id": lid,
            "w_consume": rows[0]["w_consume"] if rows else 0.0,
            "w_survival": rows[0]["w_survival"] if rows else 0.0,
            "n_seeds": len(rows),
            "foraging_competence_mean": round(_mean([r["foraging_competence"] for r in rows]), 6),
            "foraging_competence_per_seed": [round(r["foraging_competence"], 6) for r in rows],
            "survival_horizon_mean": round(_mean([r["survival_horizon"] for r in rows]), 6),
            "survival_horizon_per_seed": [round(r["survival_horizon"], 6) for r in rows],
            "n_seeds_supra_floor": int(sum(1 for r in rows if r["competence_supra_floor"])),
            "train_consumption_share_mean": round(
                _mean([r["train_consumption_share"] for r in rows]), 6),
            "train_consumption_share_per_seed": [
                round(r["train_consumption_share"], 6) for r in rows],
            "train_survival_linked_share_mean": round(
                _mean([r["train_decomposition"]["survival_linked_share"] for r in rows]), 6),
            "eval_consumption_share_mean": round(
                _mean([r["eval_decomposition"]["consumption_share"] for r in rows]), 6),
            "train_terms_weighted_per_episode_mean": {
                f: round(_mean([r["train_decomposition"]["terms_weighted_per_episode"][f]
                                for r in rows]), 6)
                for f in TERM_FAMILIES
            },
            "train_terms_raw_per_episode_mean": {
                f: round(_mean([r["train_decomposition"]["terms_raw_per_episode"][f]
                                for r in rows]), 6)
                for f in TERM_FAMILIES
            },
            "per_seed_cells": rows,
        }

    per_anchor: Dict[str, Dict[str, Any]] = {}
    for aid in ANCHOR_IDS:
        rows = [r for r in all_anchors if r["anchor_id"] == aid]
        per_anchor[aid] = {
            "anchor_id": aid,
            "n_seeds": len(rows),
            "foraging_competence_mean": round(_mean([r["foraging_competence"] for r in rows]), 6),
            "foraging_competence_per_seed": [round(r["foraging_competence"], 6) for r in rows],
            "survival_horizon_mean": round(_mean([r["survival_horizon"] for r in rows]), 6),
            "per_seed_cells": rows,
        }

    # ---- run-level statistics the preconditions and criteria read -------------------
    level_share_means = [per_level[lid]["train_consumption_share_mean"] for lid in LEVEL_IDS]
    composition_range = _rng(level_share_means)
    level_competence_means = [per_level[lid]["foraging_competence_mean"] for lid in LEVEL_IDS]
    competence_range = _rng(level_competence_means)

    guard_measured, guard_cell = _worst_cell(guards, "world_encoder_max_abs_delta", "min")
    oracle_measured, oracle_cell = _worst_cell(
        [r for r in all_anchors if r["anchor_id"] == "greedy_oracle"], "foraging_competence", "min")
    lvg_measured, lvg_cell = _worst_cell(
        [r for r in all_anchors if r["anchor_id"] == "local_view_greedy"],
        "foraging_competence", "min")

    # ---- per-level gates (regime-conditioned; a red level never vacates a green one) --
    arm_gates: List[Dict[str, Any]] = []
    for ctx in contexts:
        lid = str(ctx["id"])
        rows = [r for r in all_levels if r["level_id"] == lid]
        mag_measured, mag_cell = _worst_cell(rows, "train_attributed_magnitude", "min")
        fam_measured, fam_cell = _worst_cell(rows, "train_n_term_families_nontrivial", "min")
        measured = {
            "zworld_encoder_trained_in_p0": guard_measured,
            "d3_oracle_clears_floor": oracle_measured,
            "d3_local_view_greedy_clears_floor": lvg_measured,
            "attributed_return_magnitude_nondegenerate": mag_measured,
            "n_term_families_nontrivial": fam_measured,
            "weighting_sweep_changes_realised_composition": composition_range,
        }
        gate = evaluate_arm_gate(lid, ctx, PRECONDITIONS, measured)
        gate["offending_cells"] = {
            "zworld_encoder_trained_in_p0": guard_cell,
            "d3_oracle_clears_floor": oracle_cell,
            "d3_local_view_greedy_clears_floor": lvg_cell,
            "attributed_return_magnitude_nondegenerate": mag_cell,
            "n_term_families_nontrivial": fam_cell,
        }
        arm_gates.append(gate)

    agg = aggregate_arm_gates(arm_gates)

    # ---- criteria -------------------------------------------------------------------
    baseline_share = per_level[BASELINE_LEVEL_ID]["train_consumption_share_mean"]
    c1_consumption_dominates = bool(baseline_share >= CONSUMPTION_DOMINANCE_THRESHOLD)
    c2_reweighting_moves = bool(
        competence_range >= COMPETENCE_MOVE_DELTA
        or any(per_level[lid]["n_seeds_supra_floor"] > len(seeds) // 2 for lid in LEVEL_IDS)
    )

    readiness_met = bool(agg["any_green"])
    if not readiness_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif (not c1_consumption_dominates) and c2_reweighting_moves:
        outcome, label = "PASS", "objective_misspecification_confirmed"
    elif (not c1_consumption_dominates) and (not c2_reweighting_moves):
        outcome, label = "FAIL", "objective_dominant_but_reweighting_insufficient"
    elif c1_consumption_dominates and (not c2_reweighting_moves):
        outcome, label = "FAIL", "objective_misspecification_refuted"
    else:
        outcome, label = "FAIL", "reweighting_moves_competence_without_survival_dominance"

    # Both criteria are read off the BASELINE level (C1 directly; C2 is a cross-level range
    # whose reference point is the baseline), so they are owned by that level's gate. C2
    # additionally requires EVERY level to be scored -- a cross-level range computed over a
    # partially-scored sweep is not a measurement -- which is what `extra` encodes.
    criteria_nd = arm_criteria_non_degenerate(
        {BASELINE_LEVEL_ID: ["C1_consumption_dominates_return_at_baseline",
                             "C2_reweighting_moves_d3_competence"]},
        agg,
        extra={"C2_reweighting_moves_d3_competence": bool(agg["all_green"])},
    )

    interpretation = {
        "label": label,
        "declared_null": (
            "consumption_share >= 0.50 at W0_baseline AND cross-level foraging-competence "
            "range < 0.50 res/ep -> H-objective-misspecification REFUTED; the observability "
            "/ policy-learning route re-opens. Eliminates THIS leg only; H-policy-learning "
            "is explicitly NOT eliminated by this probe (autopsy section 8)."
        ),
        "dv_symmetry_note": (
            "DV = foraging_competence under greedy argmax eval. The manipulation is a RATIO "
            "change between term families, not a uniform rescale, so it is not annihilated "
            "by PPO's running-std reward normalisation -- EXCEPT in the degenerate case "
            "where the survival-linked terms are ~0, in which case w_consume becomes a "
            "uniform scale and the sweep measures nothing. Precondition "
            "weighting_sweep_changes_realised_composition guards exactly that case on the "
            "same statistic C1 routes on. W3_survival_zeroed deletes a term and is not "
            "expressible as any rescale of the baseline."
        ),
        "preconditions_scope_note": agg["per_arm_gate"]["preconditions_scope_note"],
        "preconditions": agg["adjudication_preconditions"],
        "criteria_non_degenerate": criteria_nd,
        "criteria": [
            {
                "name": "C1_consumption_dominates_return_at_baseline",
                "load_bearing": True,
                "passed": bool(c1_consumption_dominates),
                "measured": round(baseline_share, 6),
                "threshold": float(CONSUMPTION_DOMINANCE_THRESHOLD),
            },
            {
                "name": "C2_reweighting_moves_d3_competence",
                "load_bearing": True,
                "passed": bool(c2_reweighting_moves),
                "measured": round(competence_range, 6),
                "threshold": float(COMPETENCE_MOVE_DELTA),
            },
        ],
    }

    return {
        "outcome": outcome,
        "interpretation": interpretation,
        "non_degenerate": bool(agg["non_degenerate"]),
        "degeneracy_reason": agg["degeneracy_reason"],
        "per_arm_gate": agg["per_arm_gate"],
        "per_level": per_level,
        "per_anchor": per_anchor,
        "per_seed_levels": all_levels,
        "per_seed_anchors": all_anchors,
        "diagnostics": {
            "zworld_encoder_guard": {
                "policy": "green_gating",
                "policy_reason": (
                    "734 and 742a both ran with the encoder frozen and their z_world arms "
                    "were PPO on a random projection. This driver gates on the guard so an "
                    "unmoved encoder self-routes substrate_not_ready_requeue rather than "
                    "producing an objective verdict on a random projection."
                ),
                "n_cells": len(guards),
                "worst_cell_max_abs_delta": round(guard_measured, 9),
                "worst_cell": guard_cell,
                "all_trained": bool(all(
                    float(g.get("world_encoder_max_abs_delta", 0.0)) > ZWORLD_DELTA_FLOOR
                    for g in guards)),
                "per_cell": guards,
            },
            "composition_range_across_levels": round(composition_range, 6),
            "competence_range_across_levels": round(competence_range, 6),
            "consumption_share_by_level": {
                lid: per_level[lid]["train_consumption_share_mean"] for lid in LEVEL_IDS
            },
        },
        "headline": {
            "baseline_consumption_share": round(baseline_share, 6),
            "baseline_survival_linked_share": round(
                per_level[BASELINE_LEVEL_ID]["train_survival_linked_share_mean"], 6),
            "c1_consumption_dominates_at_baseline": c1_consumption_dominates,
            "c2_reweighting_moves_competence": c2_reweighting_moves,
            "competence_by_level": {
                lid: per_level[lid]["foraging_competence_mean"] for lid in LEVEL_IDS},
            "survival_horizon_by_level": {
                lid: per_level[lid]["survival_horizon_mean"] for lid in LEVEL_IDS},
            "anchor_competence": {
                aid: per_anchor[aid]["foraging_competence_mean"] for aid in ANCHOR_IDS},
            "readiness_met": readiness_met,
        },
    }


# --------------------------------------------------------------------------------------
def _build_manifest(result: Dict[str, Any], timestamp_utc: str, cfg: Dict[str, Any],
                    dry_run: bool) -> Dict[str, Any]:
    return {
        "run_id": f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": [],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "brake_exempt": True,
        "brake_exempt_reason": (
            "GOV-FANOUT-1 discrimination probe; claim_ids=[]; promotes/demotes nothing and "
            "adds no ceiling reading to MECH-457 (autopsy section 3 records this explicitly)"
        ),
        "timestamp_utc": timestamp_utc,
        "dry_run": bool(dry_run),
        "outcome": result["outcome"],
        "interpretation": result["interpretation"],
        "interpretation_label": result["interpretation"]["label"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "per_arm_gate": result["per_arm_gate"],
        "diagnostics": result["diagnostics"],
        "headline": result["headline"],
        "per_level": result["per_level"],
        "per_anchor": result["per_anchor"],
        "per_seed_levels": result["per_seed_levels"],
        "per_seed_anchors": result["per_seed_anchors"],
        "config": cfg,
        "hypothesis_space": {
            "question_id": "conversion_ceiling_root",
            "leg": "H-objective-misspecification",
            "axis": "reward",
            "role": "discriminating probe (GOV-FANOUT-1)",
            "source_autopsy": (
                "failure_autopsy_competence-objective-cluster-734-737b-742a_2026-07-22"
            ),
            "does_not_eliminate": ["H-policy-learning"],
        },
        "load_bearing_dv": (
            "C1: train-phase consumption_share at W0_baseline vs 0.50. C2: cross-level range "
            "of D3 foraging_competence vs 0.50 res/ep. Both load-bearing; the 2x2 grid in the "
            "module docstring routes on their conjunction."
        ),
        "notes": (
            "Return-decomposition diagnostic for the pre-registered leg "
            "H-objective-misspecification of conversion_ceiling_root. Holds the learner fixed "
            "(one all-ON REE stack per seed, SD-070 encoder-warmup path, guard GREEN-GATING) "
            "and sweeps the consumption:survival weighting over 4 levels while recording "
            "per-term return attribution (consumption / proximity / harm / novelty / other) "
            "in BOTH the training and eval phases, alongside foraging_competence and "
            "survival_horizon. This readout exists nowhere in the 724/734/737/742 family -- "
            "the measurement gap the cluster autopsy named. DIAGNOSTIC: promotes and demotes "
            "nothing; route to /failure-autopsy before any governance use. A null REFUTES "
            "this leg only and re-opens the observability/policy-learning route; it does NOT "
            "eliminate H-policy-learning."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-808 return-decomposition diagnostic for H-objective-misspecification "
            "(conversion_ceiling_root; diagnostic; claim_ids=[])"
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    _started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        zworld_p0, p0, p1, ppo = DRY_RUN_ZWORLD_P0, DRY_RUN_P0, DRY_RUN_P1, DRY_RUN_PPO
        eval_eps, steps, rollout = DRY_RUN_EVAL, DRY_RUN_STEPS, DRY_RUN_ROLLOUT
    else:
        seeds = list(SEEDS)
        zworld_p0, p0, p1, ppo = (
            ZWORLD_P0_EPISODES, P0_WARMUP_EPISODES, P1_REINFORCE_EPISODES, P1_PPO_EPISODES)
        eval_eps, steps, rollout = EVAL_EPISODES, STEPS_PER_EPISODE, PPO_ROLLOUT_EPISODES

    cfg: Dict[str, Any] = {
        "seeds": seeds,
        "rung": RUNG_ID,
        "rung_overrides": RUNG["overrides"],
        "env_kwargs": {k: v for k, v in x734._env_kwargs_for_rung(RUNG).items()
                       if isinstance(v, (int, float, bool, str)) or v is None},
        "weighting_levels": WEIGHTINGS,
        "anchors": ANCHOR_IDS,
        "zworld_p0_episodes": zworld_p0,
        "p0_warmup_episodes": p0,
        "p1_reinforce_episodes": p1,
        "p1_ppo_episodes": ppo,
        "eval_episodes": eval_eps,
        "steps_per_episode": steps,
        "ppo_rollout_episodes": rollout,
        "ppo_lr": float(x734.PPO_LR),
        "forage_bonus": float(x734.FORAGE_BONUS),
        "novelty_coef": float(x734.NOVELTY_COEF),
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "consumption_dominance_threshold": float(CONSUMPTION_DOMINANCE_THRESHOLD),
        "competence_move_delta": float(COMPETENCE_MOVE_DELTA),
        "composition_range_floor": float(COMPOSITION_RANGE_FLOOR),
        "attributed_return_magnitude_floor": float(ATTRIBUTED_RETURN_MAGNITUDE_FLOOR),
        "term_family_nontrivial_share": float(TERM_FAMILY_NONTRIVIAL_SHARE),
        "n_term_families_floor": float(N_TERM_FAMILIES_FLOOR),
        "zworld_delta_floor": float(ZWORLD_DELTA_FLOOR),
        "term_families": list(TERM_FAMILIES),
        "survival_linked_families": list(SURVIVAL_LINKED_FAMILIES),
    }

    result = run_experiment(
        seeds=seeds, zworld_p0=zworld_p0, p0=p0, p1=p1, ppo_eps=ppo,
        eval_eps=eval_eps, steps=steps, rollout=rollout, dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, cfg, dry_run=bool(args.dry_run))

    out_dir = (Path(args.out_dir) if args.out_dir is not None
               else REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments")
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=bool(args.dry_run),
        config=cfg, seeds=seeds, script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _started).total_seconds(),
    )

    hl = result["headline"]
    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} label={result['interpretation']['label']} "
        f"readiness_met={hl['readiness_met']} "
        f"baseline_consumption_share={hl['baseline_consumption_share']} "
        f"(C1_dominates={hl['c1_consumption_dominates_at_baseline']}) "
        f"competence_range={result['diagnostics']['competence_range_across_levels']} "
        f"(C2_moves={hl['c2_reweighting_moves_competence']}) "
        f"composition_range={result['diagnostics']['composition_range_across_levels']}",
        flush=True,
    )
    for lid in LEVEL_IDS:
        pl = result["per_level"][lid]
        print(
            f"  LEVEL {lid} (w_c={pl['w_consume']} w_s={pl['w_survival']}): "
            f"forage/ep={pl['foraging_competence_mean']} "
            f"(supra {pl['n_seeds_supra_floor']}/{pl['n_seeds']}) "
            f"survival={pl['survival_horizon_mean']} "
            f"consumption_share={pl['train_consumption_share_mean']} "
            f"survival_linked_share={pl['train_survival_linked_share_mean']}",
            flush=True,
        )
    for aid in ANCHOR_IDS:
        pa = result["per_anchor"][aid]
        print(
            f"  ANCHOR {aid}: forage/ep={pa['foraging_competence_mean']} "
            f"survival={pa['survival_horizon_mean']}", flush=True,
        )
    pag = result["per_arm_gate"]
    print(
        f"  gate: green={pag['green_arms']} red={pag['red_arms']} "
        f"non_degenerate={result['non_degenerate']}", flush=True,
    )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = str(result["outcome"]).upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    return outcome_emit, (str(out_path) if not args.dry_run else None), bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
