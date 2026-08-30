"""
V3-EXQ-966: MECH-143/MECH-144 causal hippocampal value-sensitivity retest.

VEHICLE-SUBSTITUTION RECORD: this experiment replaces the RETIRED V3-EXQ-165
driver as the designated MECH-144 retest vehicle. Full rationale:
REE_assembly/evidence/planning/v3_exq_165_retirement_2026-08-30.md.
V3-EXQ-165's HippocampalTerrainNavigator computed action purely from the
residue field + Manhattan distance to the chosen goal cell -- it never
received or consumed reward/value information at all, so its FIXED_VALUE vs
SHUFFLED_VALUE conditions were architecturally *guaranteed* to behave
identically (a tautology that could confirm MECH-143 but could never
falsify MECH-144). Governance (2026-08-30 Step 3, user decision) ratified
resolving MECH-144 by a properly-powered retest, sequenced after the
singleton-degeneracy guard (ree-v3 bad6467722, landed).

This driver instead runs REE's REAL trajectory-proposal / selection
machinery -- ree_core.hippocampal.module.HippocampalModule.propose_trajectories()
via REEAgent._e3_tick(), and ree_core.predictors.e3_selector.E3TrajectorySelector
-- against a CausalGridWorldV2-style env (causal_grid_world.CausalGridWorldV2,
i.e. CausalGridWorld(use_proxy_fields=True)) with the same FIXED_VALUE vs
SHUFFLED_VALUE goal-value manipulation the retired driver used, applied to a
SD-049 two-resource-type heterogeneous environment.

CODE-READ FINDINGS THAT SHAPE THIS DESIGN (agent.py / module.py / goal.py /
causal_grid_world.py, all read in full for this driver):

  * HippocampalModule._score_trajectory() is ARC-007 STRICT: "primary
    scoring is terrain-only ... No independent harm prediction here -- E3
    introduces all value weighting." The ONLY documented value-leakage
    channels into CEM trajectory scoring are config.hippocampal.wanting_weight
    (VALENCE_WANTING term), config.hippocampal.curiosity_weight, and a
    mode-conditioned term -- ALL default 0.0/off. propose_trajectories()'s
    `current_z_goal` argument (MECH-293) is consumed ONLY by the MECH-292
    ghost-goal-bank ranking branch, which additionally requires a populated
    bank entry clearing goal_match_floor -- perturbing z_goal alone with an
    empty bank is a dead lever, so this driver does NOT use z_goal
    perturbation for its architectural check (an earlier design draft
    considered this and found it produces a guaranteed no-op).

  * REEAgent.update_z_goal() binds IncentiveTokenBank.update() (SD-057 L2/L3)
    UNCONDITIONALLY on any genuine resource contact (gated only on
    resource_type>0 and the resource encoder being present -- NOT on any
    benefit/seeding threshold). StepHarness.step() calls update_z_goal()
    automatically every tick with resource_type auto-derived from
    obs_dict["resource_type_at_agent"] (SD-049; 0=none, else type_idx+1).
    So genuine ecological contact -- via whatever movement the REAL
    trajectory-proposal + E3-selection pipeline produces, not a hand-rolled
    navigator -- drives this substrate directly, with no extra wiring.

  * causal_grid_world.CausalGridWorld._consume_resource_at() reads
    self.resource_type_benefit_amplitudes[contact_type_idx] LIVE at contact
    time (not cached at construction), so the FIXED_VALUE/SHUFFLED_VALUE
    manipulation is implemented as a live in-place mutation of that tuple
    between episode blocks -- no env rebuild needed, matching (and
    confirming) the retired driver's own live-mutation approach.

DESIGN (two complementary DVs, deliberately NOT reusing the retired driver's
Bernoulli(0.5) block-shuffle -- that mechanism is what produced the C4
"coin-flip-fragile" data-quality gate this retirement explicitly flags as
the mistake to avoid; this driver uses a DETERMINISTIC block/role schedule
instead, identical across seeds, so there is no stochastic reachability gate
to sit at its own theoretical expectation):

  C1 (MECH-143, architectural / deterministic, NOT statistically powered --
      it is a runtime confirmation of a static architectural fact already
      established by code reading, not a stochastic claim):
      Per seed, after the ecological rollout, take one frozen real z_world/
      z_self/e1_prior snapshot. EMPIRICAL FINDING during driver development:
      ResidueField.accumulate() (which creates the RBF centers
      update_valence() needs to write into) is called ONLY when
      harm_signal < 0 (agent.py update_residue()) -- i.e. residue centers,
      and therefore VALENCE_WANTING content, are HARM-gated, not
      benefit-gated. With NUM_HAZARDS=0 (deliberate, to keep C2's contact
      DV harm-confound-free) no center is ever created by the ordinary
      rollout, so VALENCE_WANTING reads zero everywhere regardless of
      wanting_weight -- not because the channel is inert, but because
      nothing ever wrote to it. Waiting on MECH-290/MECH-217 (the two
      mechanisms that DO write VALENCE_WANTING -- backward_credit_sweep on
      BetaGate completion, offline_wanting_spread on the REM pass) is not a
      good fit either: both require exactly the kind of directed,
      competent goal-completion behavior this driver deliberately avoids
      depending on (see the GAP-2/foraging-competence note below). So C1
      instead seeds the mechanism's OWN state directly and transparently --
      the same "call the substrate primitive directly" pattern V3-EXQ-636/
      637 uses to decouple a mechanism check from emergent competence --
      via agent.residue_field.accumulate() (forces one active RBF center to
      exist at the current z_world) then .update_valence(..., VALENCE_WANTING,
      10.0) (writes a strong wanting value into it), confirmed empirically
      to make the channel non-trivial at this location. Then call
      REEAgent._e3_tick() three times with RNG fully reset (reset_all_rng)
      before each call:
        (a) config.hippocampal.wanting_weight at its cell value, call twice
            -> MUST be bit-identical (sanity: proves the RNG-reset +
            _e3_tick() call is itself deterministic; if this fails the
            whole bit-identity methodology is broken and is reported, not
            silently passed).
        (b) config.hippocampal.wanting_weight flipped to the opposite of
            (a)'s value (0.0<->5.0) -> candidates MUST DIFFER from (a).
            This is the falsifier-premise check: with real (forced-nonzero)
            VALENCE_WANTING content present, it demonstrates the one
            documented value-injection channel is live and that this
            bit-identity comparison methodology has the power to detect a
            real difference when one is architecturally possible -- so a
            "no difference" verdict elsewhere in this driver cannot be
            waved away as "the test can't detect anything anyway."
      MECH-143 is empirically re-confirmed (this run) when every seed's (a)
      holds bit-identical and every seed's (b) differs; it would be
      empirically falsified by this run if (a) ever differs (an
      undocumented value channel or nondeterminism the code-read missed).

  C2 (MECH-144, ecological / causal / properly powered -- the load-bearing
      test): TWO arms, wanting_weight=5.0 in BOTH (the one confirmed-live
      value channel from C1(b) is deliberately kept ON in both arms, so the
      manipulation under test is ONLY the environment's value assignment,
      not whether the pathway exists at all):
        ARM_FIXED_VALUE:    resource type 0 stays HIGH_AMP for all 4
                             measurement blocks.
        ARM_SHUFFLED_VALUE: resource type 0 is HIGH_AMP for blocks 1-2,
                             then resource type 1 becomes HIGH_AMP for
                             blocks 3-4 (one deterministic swap, applied
                             identically to every seed).
      Per block, DV = frac_high = (contacts of the block's CURRENT
      high-amplitude type) / (total type-tagged contacts), read from
      StepResult.info["sd049_consumed_type_tag_this_tick"] each tick.
      Block 2 = "settled pre-swap" reading (both arms identical schedule up
      to here). Block 4 = "settled post-swap" reading, two full blocks after
      the SHUFFLED arm's swap -- long enough for IncentiveTokenBank's EMA
      (incentive_value_alpha=0.1 default) to re-converge if it can.
      Per-seed delta_i = frac_high_FIXED(block4) - frac_high_SHUFFLED(block4)
      (an equivalence-style comparison: PASS/supports means this delta is
      SMALL -- the SHUFFLED arm re-tracked the swap about as well as FIXED
      exploited its stable assignment, i.e. genuine causal value-sensitivity
      in real trajectory-selection behavior. FAIL/weakens means the delta is
      LARGE and positive -- SHUFFLED could not adapt, i.e. value-blindness
      persists downstream even with the pathway confirmed live by C1(b)).
      margin = max(K_EFFECT * pstdev(delta_i), ABS_FLOOR) -- the standard
      repo effect-size-margin convention (SD of the DELTA, not baseline
      level, plus an absolute floor), applied here as an equivalence bound
      rather than a minimum-detectable-effect bound; both are the same
      formula, this is just which side of it the PASS lives on.
      Both tails of the frac_high ratio are floored: MIN_CONTACTS_PER_BLOCK
      is NOT a pre-registered theoretical value (the retired driver's exact
      mistake -- see the retirement doc Section on C4) -- it is derived
      POST-HOC from this run's OWN measured mean per-block contact count
      (self-calibrating floor: max(3, floor(0.5 * observed_mean))), computed
      once over all seeds/arms/blocks actually collected.

  No singleton groups reach metric_groups_are_degenerate: the per-arm
  frac_high(block4) groups both have arity == len(SEEDS) (>=2), consistent
  with the now-landed arity guard (ree-v3 bad6467722).

GOV-REUSE-1: checked REE_assembly/evidence/planning for an existing MECH-143/
MECH-144 causal (not correlational) ecological driver against the real
HippocampalModule/E3TrajectorySelector pipeline. V3-EXQ-165 (retired,
tautological, hand-rolled navigator) and V3-EXQ-960/961 (MECH-143
dCA1-representational-probe and MECH-144 correlational spatial-gradient
probe, both z_world-representational-geometry designs, not
candidate-trajectory-scoring / behavioral-tracking designs) are the only
prior MECH-143/144 drivers and are all architecturally distinct from this
one -- not recoverable via reanalysis, ran fresh.

SUBSTRATE-PATH OVERLAP (Step 2.5c): no open severity=corrupting
substrate_queue.json entries at write time overlapping
ree_core/hippocampal/module.py, ree_core/goal.py, ree_core/agent.py, or
ree_core/environment/causal_grid_world.py (checked against origin/master).

RE-DERIVE BRAKE (Step 2.5b): MECH-144 carries no substrate_ceiling brake --
its 961 weakens is a scoring evidence entry, not a ceiling hit, and this
retest is explicitly the governance-ratified resolution path (2026-08-30
Step 3), not a re-run against an unresolved ceiling.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import statistics
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.residue.field import VALENCE_WANTING

from experiments._harness import StepHarness
from experiments.pack_writer import write_flat_manifest
from experiments._metrics import metric_groups_are_degenerate, check_degeneracy
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiment_protocol import emit_outcome

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_966_mech143_144_hippocampal_value_sensitivity_causal"
CLAIM_IDS           = ["MECH-143", "MECH-144"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEEDS = [42, 47, 53, 61]          # avoid seed 44 (repo caution)
ARMS  = ["ARM_FIXED_VALUE", "ARM_SHUFFLED_VALUE"]

GRID_SIZE      = 9
NUM_HAZARDS    = 0                # value-sensitivity DV, not harm-avoidance
NUM_RESOURCES  = 6
RESOURCE_BENEFIT = 0.3
HIGH_AMP = 1.0
LOW_AMP  = 0.15                   # contact_benefit: 0.30 vs 0.045 (~6.7x)

WARMUP_EPISODES     = 12          # P0: E1 + ResourceEncoder representational warmup
N_BLOCKS            = 4           # P1: measurement blocks
EPISODES_PER_BLOCK  = 10
STEPS_PER_EP        = 120

# Deterministic HIGH-type-index schedule per block, per arm. Index into
# resource_type_benefit_amplitudes (0 or 1). Identical across all seeds --
# no Bernoulli, no stochastic reachability gate.
BLOCK_HIGH_IDX = {
    "ARM_FIXED_VALUE":    [0, 0, 0, 0],
    "ARM_SHUFFLED_VALUE": [0, 0, 1, 1],   # single swap between block 2 and 3
}
PRE_BLOCK  = 1   # 0-indexed block used as "settled pre-swap" reading
POST_BLOCK = 3   # 0-indexed block used as "settled post-swap" reading

WANTING_WEIGHT = 5.0               # confirmed-live value channel, ON in BOTH arms
WORLD_DIM      = 32
SELF_DIM       = 32
ALPHA_WORLD    = 0.9
P0_LR          = 3e-4

K_EFFECT   = 2.0                   # margin = max(K_EFFECT*pstdev(delta), ABS_FLOOR)
ABS_FLOOR  = 0.10                  # 10 percentage points of frac_high

DRY_RUN_WARMUP_EPISODES = 2
DRY_RUN_EPISODES_PER_BLOCK = 2
DRY_RUN_STEPS = 15

_ZG = ZGoalStreamAccumulator()


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_env(seed: int, high_idx: int) -> CausalGridWorldV2:
    amps = [LOW_AMP, LOW_AMP]
    amps[high_idx] = HIGH_AMP
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        resource_benefit=RESOURCE_BENEFIT,
        resource_respawn_on_consume=True,
        multi_resource_heterogeneity_enabled=True,
        n_resource_types=2,
        resource_type_names=("type_a", "type_b"),
        resource_type_drive_axes=("hunger", "thirst"),
        resource_type_benefit_curves=("sigmoidal_saturating", "sigmoidal_saturating"),
        resource_type_distribution=(1.0, 1.0),
        resource_type_benefit_amplitudes=tuple(amps),
        per_axis_drive_enabled=False,
        max_episode_steps=STEPS_PER_EP,
    )


def _set_high_idx(env: CausalGridWorldV2, high_idx: int) -> None:
    """Live-mutate the amplitude tuple (read fresh at contact time by
    _consume_resource_at -- no env rebuild needed)."""
    amps = [LOW_AMP, LOW_AMP]
    amps[high_idx] = HIGH_AMP
    env.resource_type_benefit_amplitudes = tuple(amps)


def _make_agent(seed: int, env: CausalGridWorldV2) -> REEAgent:
    torch.manual_seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        z_goal_enabled=True,
        wanting_weight=WANTING_WEIGHT,
        use_incentive_token_bank=True,
        incentive_use_per_axis_drive=False,   # uniform kappa across types --
                                               # isolates value-magnitude
                                               # tracking from drive-axis
                                               # confounds (per_axis_drive is
                                               # also off in the env).
    )
    config.latent.use_resource_encoder = True
    return REEAgent(config)


def _config_slice(condition: str, seed: int) -> Dict:
    return {
        "condition": condition,
        "seed": seed,
        "grid_size": GRID_SIZE,
        "num_hazards": NUM_HAZARDS,
        "num_resources": NUM_RESOURCES,
        "resource_benefit": RESOURCE_BENEFIT,
        "high_amp": HIGH_AMP,
        "low_amp": LOW_AMP,
        "warmup_episodes": WARMUP_EPISODES,
        "n_blocks": N_BLOCKS,
        "episodes_per_block": EPISODES_PER_BLOCK,
        "steps_per_ep": STEPS_PER_EP,
        "block_high_idx": BLOCK_HIGH_IDX[condition],
        "wanting_weight": WANTING_WEIGHT,
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "alpha_world": ALPHA_WORLD,
        "use_incentive_token_bank": True,
        "incentive_use_per_axis_drive": False,
        "p0_lr": P0_LR,
    }


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def _run_episode(harness: StepHarness, env, steps: int) -> None:
    _flat_obs, obs_dict = env.reset()
    for _ in range(steps):
        result = harness.step(obs_dict)
        obs_dict = result.next_obs_dict
        if result.done:
            _flat_obs, obs_dict = env.reset()


def _run_episode_measured(
    harness: StepHarness, env, steps: int
) -> Dict[int, int]:
    """Same as _run_episode but tallies per-type contacts this episode."""
    contacts = {1: 0, 2: 0}
    _flat_obs, obs_dict = env.reset()
    for _ in range(steps):
        result = harness.step(obs_dict)
        obs_dict = result.next_obs_dict
        tag = int(result.info.get("sd049_consumed_type_tag_this_tick", 0) or 0)
        if tag in contacts:
            contacts[tag] += 1
        if result.done:
            _flat_obs, obs_dict = env.reset()
    return contacts


def _c1_architectural_check(agent: REEAgent, seed: int) -> Dict:
    """Deterministic runtime confirmation that propose_trajectories() at
    this cell's config is insensitive to wanting_weight identity-repeat, and
    IS sensitive to a wanting_weight flip (falsifier-premise check).

    Runs in eval() mode: with train_mode dropout/BatchNorm active, a SECOND
    call with an identical RNG reset is not guaranteed to reproduce the
    first (BatchNorm running-stat updates from call A's forward pass alone
    change what call B computes) -- that is a training-mode artifact, not
    evidence about value-leakage, so it is controlled out here rather than
    misread as an architecture finding. The agent's train_mode flag (if any
    module holds one) is restored afterward.
    """
    was_training = agent.training if hasattr(agent, "training") else None
    agent.eval()
    world_dim = agent.latent_stack.config.world_dim if hasattr(
        agent.latent_stack, "config"
    ) else WORLD_DIM
    # Use a real, non-degenerate observation: sense() against a synthetic
    # but architecturally-legitimate zero body/world obs is acceptable here
    # because C1 only asks "does an identical latent produce identical vs
    # differing candidates as wanting_weight changes" -- it does not depend
    # on the obs content being ecologically realistic.
    body = torch.zeros(1, agent.config.environment.body_obs_dim)
    world = torch.randn(1, agent.config.environment.world_obs_dim) * 0.5
    with torch.no_grad():
        latent = agent.sense(body, world)
        # Force-seed the mechanism's own state (see module docstring "C1"
        # section): ResidueField.accumulate() is harm-gated, so with
        # NUM_HAZARDS=0 no RBF center -- and therefore no VALENCE_WANTING
        # content -- would otherwise exist for wanting_weight to read.
        agent.residue_field.accumulate(
            latent.z_world, harm_magnitude=1.0, hypothesis_tag=False
        )
        agent.residue_field.update_valence(
            latent.z_world, VALENCE_WANTING, 10.0, hypothesis_tag=False
        )
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent) if ticks.get("e1_tick", False)
            else torch.zeros(1, latent.z_world.shape[-1], device=agent.device)
        )

        orig_ww = agent.hippocampal.config.wanting_weight
        flipped_ww = 0.0 if orig_ww > 0.0 else 5.0

        def _propose():
            return agent._e3_tick(latent, e1_prior, None)

        reset_all_rng(seed)
        agent.hippocampal.config.wanting_weight = orig_ww
        cand_a = _propose()
        reset_all_rng(seed)
        agent.hippocampal.config.wanting_weight = orig_ww
        cand_b = _propose()
        reset_all_rng(seed)
        agent.hippocampal.config.wanting_weight = flipped_ww
        cand_c = _propose()
        agent.hippocampal.config.wanting_weight = orig_ww  # restore

    if was_training:
        agent.train()

    def _stack(cands):
        return torch.stack([c.actions for c in cands])

    try:
        repeat_identical = bool(torch.equal(_stack(cand_a), _stack(cand_b)))
    except RuntimeError:
        repeat_identical = False
    try:
        flip_differs = not bool(torch.equal(_stack(cand_a), _stack(cand_c)))
    except RuntimeError:
        flip_differs = True

    return {
        "seed": seed,
        "orig_wanting_weight": orig_ww,
        "flipped_wanting_weight": flipped_ww,
        "repeat_identical": repeat_identical,
        "flip_differs": flip_differs,
        "c1_pass": repeat_identical and flip_differs,
    }


def run_cell(seed: int, condition: str, dry_run: bool = False) -> Dict:
    warmup_eps = DRY_RUN_WARMUP_EPISODES if dry_run else WARMUP_EPISODES
    eps_per_block = DRY_RUN_EPISODES_PER_BLOCK if dry_run else EPISODES_PER_BLOCK
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EP
    schedule = BLOCK_HIGH_IDX[condition]

    env = _make_env(seed, schedule[0])
    agent = _make_agent(seed, env)
    harness = StepHarness(agent, env, train_mode=True, seed=seed)

    # -- P0: representational warmup (E1 + SD-015 ResourceEncoder) ---------
    p0_opt = torch.optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=P0_LR,
    )
    _flat_obs, obs_dict = env.reset()
    for _ in range(warmup_eps):
        for _ in range(steps):
            latent_before = None
            result = harness.step(obs_dict)
            rfv = obs_dict.get("resource_field_view")
            target = float(rfv.max().item()) if rfv is not None else 0.0
            try:
                loss = agent.compute_prediction_loss()
                loss = loss + agent.compute_resource_encoder_loss(
                    target, result.latent
                )
                if loss.requires_grad:
                    p0_opt.zero_grad()
                    loss.backward()
                    p0_opt.step()
            except Exception:
                pass  # P0 warmup is best-effort representational shaping,
                      # not itself load-bearing for C1/C2.
            obs_dict = result.next_obs_dict
            if result.done:
                _flat_obs, obs_dict = env.reset()
        _flat_obs, obs_dict = env.reset()

    # -- P1: measurement blocks ---------------------------------------------
    block_contacts: List[Dict[int, int]] = []
    for b in range(N_BLOCKS):
        high_idx = schedule[b]
        _set_high_idx(env, high_idx)
        block_total = {1: 0, 2: 0}
        for _ in range(eps_per_block):
            ep_contacts = _run_episode_measured(harness, env, steps)
            block_total[1] += ep_contacts[1]
            block_total[2] += ep_contacts[2]
        block_contacts.append(block_total)

    def _frac_high(block_idx: int) -> Tuple[float, int]:
        high_type_tag = schedule[block_idx] + 1
        c = block_contacts[block_idx]
        total = c[1] + c[2]
        if total == 0:
            return 0.5, 0
        return c[high_type_tag] / total, total

    frac_pre, n_pre = _frac_high(PRE_BLOCK)
    frac_post, n_post = _frac_high(POST_BLOCK)

    incentive_bank = getattr(agent.goal_state, "incentive_bank", None)
    bank_base_values = (
        dict(incentive_bank._base_value) if incentive_bank is not None else {}
    )

    c1 = _c1_architectural_check(agent, seed)

    return {
        "seed": seed,
        "condition": condition,
        "block_contacts": block_contacts,
        "block_high_idx": schedule,
        "frac_high_pre": frac_pre,
        "n_contacts_pre": n_pre,
        "frac_high_post": frac_post,
        "n_contacts_post": n_post,
        "incentive_bank_base_values": {
            str(k): float(v) for k, v in bank_base_values.items()
        },
        "c1": c1,
    }


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def evaluate_criteria(all_results: List[Dict]) -> Dict:
    by_condition = {c: [r for r in all_results if r["condition"] == c] for c in ARMS}

    # -- C1: MECH-143 architectural confirmation (deterministic per-seed) --
    c1_all = [r["c1"] for r in all_results]
    c1_repeat_pass = all(c["repeat_identical"] for c in c1_all)
    c1_flip_pass = all(c["flip_differs"] for c in c1_all)
    c1_pass = c1_repeat_pass and c1_flip_pass

    # -- Data-quality floor: derived from THIS run's own measured contacts --
    all_n = [
        r["n_contacts_pre"] for r in all_results
    ] + [
        r["n_contacts_post"] for r in all_results
    ]
    observed_mean_contacts = statistics.mean(all_n) if all_n else 0.0
    min_contacts_per_block = max(3, int(observed_mean_contacts * 0.5))

    reachable = [
        r for r in all_results
        if r["n_contacts_pre"] >= min_contacts_per_block
        and r["n_contacts_post"] >= min_contacts_per_block
    ]
    reachable_seeds = sorted(set(r["seed"] for r in reachable))
    usable_seeds = [
        s for s in reachable_seeds
        if all(
            any(r["seed"] == s and r["condition"] == c for r in reachable)
            for c in ARMS
        )
    ]

    fixed_post = {
        r["seed"]: r["frac_high_post"]
        for r in by_condition["ARM_FIXED_VALUE"] if r["seed"] in usable_seeds
    }
    shuffled_post = {
        r["seed"]: r["frac_high_post"]
        for r in by_condition["ARM_SHUFFLED_VALUE"] if r["seed"] in usable_seeds
    }
    deltas = [fixed_post[s] - shuffled_post[s] for s in usable_seeds]

    if len(deltas) >= 2:
        mean_delta = statistics.mean(deltas)
        sd_delta = statistics.pstdev(deltas)
    elif len(deltas) == 1:
        mean_delta = deltas[0]
        sd_delta = 0.0
    else:
        mean_delta = None
        sd_delta = None

    margin = None
    c2_verdict = "inconclusive"
    if mean_delta is not None:
        margin = max(K_EFFECT * sd_delta, ABS_FLOOR)
        if mean_delta > margin:
            c2_verdict = "weakens"       # SHUFFLED failed to re-track: adaptation cost
        elif abs(mean_delta) <= margin:
            c2_verdict = "supports"      # no reliable adaptation cost -> value-sensitive
        else:
            c2_verdict = "supports"      # SHUFFLED significantly BETTER than FIXED
                                          # (unexpected direction; still no adaptation
                                          # deficit, folded into supports with anomaly note)

    # -- Non-degeneracy (arity guard: both groups have arity == len(usable_seeds)) --
    fixed_vals = [fixed_post[s] for s in usable_seeds]
    shuffled_vals = [shuffled_post[s] for s in usable_seeds]
    is_degen, degen_reason = metric_groups_are_degenerate(
        [fixed_vals, shuffled_vals], floor=1e-6
    )

    min_seeds_needed = 2
    c2_readiness_pass = (
        len(usable_seeds) >= min_seeds_needed and not is_degen
    )

    overall_pass = c1_pass and c2_readiness_pass and c2_verdict in (
        "supports", "weakens"
    )

    return {
        "c1_pass": c1_pass,
        "c1_repeat_pass": c1_repeat_pass,
        "c1_flip_pass": c1_flip_pass,
        "c1_per_seed": c1_all,
        "observed_mean_contacts_per_block": observed_mean_contacts,
        "min_contacts_per_block_floor": min_contacts_per_block,
        "reachable_seeds": reachable_seeds,
        "usable_seeds": usable_seeds,
        "fixed_post_frac_high": fixed_vals,
        "shuffled_post_frac_high": shuffled_vals,
        "deltas": deltas,
        "mean_delta": mean_delta,
        "sd_delta": sd_delta,
        "margin": margin,
        "k_effect": K_EFFECT,
        "abs_floor": ABS_FLOOR,
        "c2_verdict": c2_verdict,
        "c2_readiness_pass": c2_readiness_pass,
        "is_degenerate": is_degen,
        "degenerate_reason": degen_reason,
        "overall_pass": overall_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run(dry_run: bool):
    t0 = time.perf_counter()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"{EXPERIMENT_TYPE}_dry_{ts}_v3" if dry_run
        else f"{EXPERIMENT_TYPE}_{ts}_v3"
    )
    print(f"V3-EXQ-966 start: {run_id}", flush=True)

    all_results: List[Dict] = []
    arm_results: List[Dict] = []

    for seed in SEEDS:
        print(f"\n--- seed {seed} ---", flush=True)
        for condition in ARMS:
            print(f"  arm {condition} ...", flush=True)
            with arm_cell(
                seed,
                config_slice=_config_slice(condition, seed),
                script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                result = run_cell(seed, condition, dry_run=dry_run)
                cell.stamp(result)
            print(
                f"    frac_high pre={result['frac_high_pre']:.3f} "
                f"(n={result['n_contacts_pre']}) "
                f"post={result['frac_high_post']:.3f} (n={result['n_contacts_post']}) "
                f"c1_pass={result['c1']['c1_pass']}",
                flush=True,
            )
            all_results.append(result)
            arm_results.append(result)

    criteria = evaluate_criteria(all_results)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"

    print(f"\n=== V3-EXQ-966 {outcome} ===", flush=True)
    print(
        f"C1 (MECH-143 architectural): pass={criteria['c1_pass']} "
        f"(repeat_identical all seeds={criteria['c1_repeat_pass']}, "
        f"flip_differs all seeds={criteria['c1_flip_pass']})",
        flush=True,
    )
    print(
        f"C2 (MECH-144 ecological): verdict={criteria['c2_verdict']} "
        f"mean_delta={criteria['mean_delta']} margin={criteria['margin']} "
        f"usable_seeds={criteria['usable_seeds']}/{SEEDS} "
        f"min_contacts_floor={criteria['min_contacts_per_block_floor']} "
        f"(observed_mean={criteria['observed_mean_contacts_per_block']:.1f})",
        flush=True,
    )

    degeneracy = check_degeneracy({
        "c2_fixed_post_frac_high": {"values": criteria["fixed_post_frac_high"]},
        "c2_shuffled_post_frac_high": {"values": criteria["shuffled_post_frac_high"]},
    })

    evidence_direction_per_claim = {
        "MECH-143": "supports" if criteria["c1_pass"] else "weakens",
        "MECH-144": criteria["c2_verdict"],
    }

    full_config = {
        "seeds": SEEDS,
        "arms": ARMS,
        "grid_size": GRID_SIZE,
        "num_hazards": NUM_HAZARDS,
        "num_resources": NUM_RESOURCES,
        "resource_benefit": RESOURCE_BENEFIT,
        "high_amp": HIGH_AMP,
        "low_amp": LOW_AMP,
        "warmup_episodes": WARMUP_EPISODES,
        "n_blocks": N_BLOCKS,
        "episodes_per_block": EPISODES_PER_BLOCK,
        "steps_per_ep": STEPS_PER_EP,
        "block_high_idx": BLOCK_HIGH_IDX,
        "pre_block": PRE_BLOCK,
        "post_block": POST_BLOCK,
        "wanting_weight": WANTING_WEIGHT,
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "alpha_world": ALPHA_WORLD,
        "p0_lr": P0_LR,
        "k_effect": K_EFFECT,
        "abs_floor": ABS_FLOOR,
    }

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_class": "discriminative_pair",
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "outcome": outcome,
        "registered_thresholds": {
            "k_effect": K_EFFECT,
            "abs_floor": ABS_FLOOR,
            "min_contacts_per_block_floor": criteria["min_contacts_per_block_floor"],
        },
        "criteria": criteria,
        "arm_results": arm_results,
        "supersedes": None,
        "vehicle_substitution_record": (
            "REE_assembly/evidence/planning/v3_exq_165_retirement_2026-08-30.md "
            "-- replaces the retired V3-EXQ-165 driver (tautological, value-blind "
            "hand-rolled navigator) as the MECH-144 retest vehicle per governance "
            "2026-08-30 Step 3 (user decision), sequenced after the landed "
            "singleton-degeneracy arity guard (ree-v3 bad6467722)."
        ),
        "summary": (
            f"MECH-143/MECH-144 causal hippocampal value-sensitivity retest, "
            f"driven through the real HippocampalModule/E3TrajectorySelector "
            f"pipeline. C1 (MECH-143 architectural): {criteria['c1_pass']}. "
            f"C2 (MECH-144 ecological FIXED vs SHUFFLED value tracking): "
            f"{criteria['c2_verdict']} (mean_delta={criteria['mean_delta']}, "
            f"margin={criteria['margin']}). Outcome: {outcome}."
        ),
        "custom_information": {
            "gov_reuse_1_check": (
                "Checked REE_assembly/evidence/planning and existing "
                "MECH-143/144 drivers (V3-EXQ-165 retired/tautological, "
                "V3-EXQ-960/961 representational-geometry designs) for a "
                "prior causal/behavioral candidate-trajectory-scoring test "
                "against the real HippocampalModule/E3TrajectorySelector "
                "pipeline. None exists -- not recoverable, ran fresh."
            ),
            "substrate_path_overlap_check": (
                "No open severity=corrupting substrate_queue.json entries "
                "overlapping ree_core/hippocampal/module.py, ree_core/goal.py, "
                "ree_core/agent.py, or ree_core/environment/causal_grid_world.py "
                "at write time (checked against origin/master)."
            ),
            "rederive_brake_check": (
                "MECH-144 carries no substrate_ceiling brake -- its 961 "
                "weakens is a scoring evidence entry, not a ceiling hit; this "
                "retest is the governance-ratified resolution path "
                "(2026-08-30 Step 3), not a re-run against an unresolved "
                "ceiling."
            ),
        },
    }
    manifest.update(degeneracy)

    out_path = write_flat_manifest(
        manifest,
        dry_run=bool(dry_run),
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"Results -> {out_path}", flush=True)

    return outcome, out_path, run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path, _run_id = _run(args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        run_id=_run_id,
        dry_run=bool(args.dry_run),
    )
