#!/opt/local/bin/python3
"""V3-EXQ-1039 -- waypoint_field_consumer_reach H1 (axis: drive / training signal).

Registry: REE_assembly/evidence/planning/hypothesis_space_registry.v1.json,
qid `waypoint_field_consumer_reach`, hid `H-wpfield-objective-sparsity`. Pre-registered by
the confirmed `failure_autopsy_V3-EXQ-1004_2026-09-05` fan-out (fanout_recommendation
suggested_probes[0]; H2, the sibling `representation`-axis leg, was queued separately as
V3-EXQ-1030 on 2026-09-14 and has already returned FAIL/CONTESTED -- unrelated to this leg's
own verdict, since GOV-FANOUT-1 legs are independent by design).

QUESTION. V3-EXQ-1004 established that the SD-WAYPOINT-FIELD observable makes the pending
waypoint's direction decodable and BEHAVIOURALLY SUFFICIENT FOR A SUPERVISED READER cloned
from an oracle (0.573 -> 0.839 BC-imitation accuracy; 0.35 -> 58.6 visits/ep). That reader is
NOT a REE consumer: it is a bespoke 2-layer net over a hand-sliced 25/50-dim vector, trained
purely by behaviour-cloning, with no z_world, no E1/E2/E3, no z_goal. This run asks the
orthogonal drive-axis question the 1004 autopsy fanned out: for an ACTUAL REEAgent consumer
(z_world via the standard latent stack; action selection via the real E1-candidate-generation
-> E3-selection loop, `agent.generate_trajectories` + `agent.select_action`, exactly the
`_train_all_on_agent` "ree_trained_allon" recipe already used by V3-EXQ-724/734/737/742),
holding the field ON and the consumer fixed, does the TRAINING SIGNAL determine whether
perceivability converts into navigation?

DECLARED NULL (H1, from the registry). visits/ep is FLAT across training signals -- i.e. a
denser (shaped) or warm-started signal does NOT lift the consumer off the sparse-reward floor.
Meeting this null supports "sparsity is not the residual blocker" (the null the fanout named).
A CLEARED lift under either dense arm, with the sparse arm still blocked, supports H1 (reward
sparsity IS the residual blocker on this consumer).

CAVEAT CARRIED FORWARD (per chip instruction). V3-EXQ-1004's docstring reports an A2C reader
at 0.00 (field OFF) / 0.10 (field ON) visits/ep after 400 episodes on seed 42 only -- that
figure is DOCSTRING-ONLY (never committed, no manifest, no per-episode trace) and was measured
BEFORE the SD-094 contamination gate, on a self-contaminating env where every arm ended
health_depleted at 44-134 steps against a declared 400 and per-episode contamination (~-1.2)
outweighed the waypoint reward (~+0.04) by 30:1 -- not moving was reward-optimal. That figure
MOTIVATES this leg's existence; it is not cited below as a baseline, a bar, or a comparison
point, and no threshold in this script is derived from it.

=== WHY THIS IS A DIFFERENT CONSUMER THAN V3-EXQ-1004's CLONE, AND WHAT IT IS NOT ===

This run builds a REEAgent via the same "all-ON" config builders V3-EXQ-724/734/737/742 use
(`x724._base_config_kwargs` + `x724._all_on_extra_kwargs` -> `REEConfig.from_dims` ->
`REEAgent`), so E1 (deep predictor), E2 (fast forward model), E3 (`E3TrajectorySelector`,
a trainable `nn.Module` with several channel-scoring heads), dACC, OFC, go/no-go and the
modulatory-authority superstructure are all LIVE. Every env tick in every treatment arm calls
`agent.sense(...)` (populating z_world/z_self through the real SplitEncoder latent stack),
`agent.generate_trajectories(...)` (E1-conditioned candidate generation) and
`agent.select_action(candidates, ticks)` (the real E3 competitive selection over those
candidates) -- this IS "a REEAgent, z_world -> E1/E2/E3", not a bolt-on policy head reading
z_world from the side. This is the SAME action-selection call `_train_all_on_agent` already
uses for its P0/P1 loop (imported unmodified from `experiments/_lib/allon_training.py`), so
this driver adds no new candidate-generation or selection code -- only a reward-shaping env
wrapper and a demonstration-warm-start auxiliary loss (both described below) sit on top of it.

WHAT THIS IS **NOT**, STATED PLAINLY (so the claim-tagging call below is checkable). The
per-tick REWARD-DRIVEN learning in `_train_all_on_agent`'s P1 phase does NOT train E3's own
channel-scoring heads: it trains the LATERAL-PFC BIAS head and the OFC DEVALUATION-BIAS head
(both of which modulate E3's competitive scoring downstream, via `score_bias`/devaluation
bias terms E3 consumes) through a two-head REINFORCE update keyed to episode return. E3's own
scoring-head parameters are not in either optimiser's parameter list and receive no gradient
from this training signal. So: this run exercises the REAL E1-generate -> E3-select decision
loop on every tick (the loop that actually DRIVES the agent's behaviour), and REAL
reward-contingent learning through the modulatory heads that bias that loop, but does not
itself train E3's core scoring network. This is a materially different (and, for the training
-signal question this leg asks, sufficient) consumer than 1004's clone; it does not, however,
license "E3's selection mechanism was trained on this reward" as a description of what
happened. Stated so a reader does not need to re-derive it from the source.

=== THE THREE TRAINING-SIGNAL ARMS (field ON, consumer construction IDENTICAL across arms) ===

  sparse_rl       stock env reward only (waypoint_visit_reward / waypoint_completion_reward,
                  hazards+resources zeroed -- see NAVIGATION ISOLATION below). REINFORCE via
                  `_train_all_on_agent(p0_episodes=0, p1_episodes=RL_EPISODES)`, UNMODIFIED.
  shaped_rl       IDENTICAL call, on the SAME env wrapped in `_ShapedWaypointEnv`: every
                  `step()` reward is augmented with a MONOTONE, NON-NEGATIVE per-step
                  progress bonus `shaping_coef * max(0, manhattan_prev - manhattan_next)`
                  toward the pending waypoint. Only the reward channel is touched;
                  observations, transitions and env dynamics are byte-identical to
                  sparse_rl's env. THIS IS NOT Ng-Harada-Russell potential-DIFFERENCE
                  shaping, and deliberately so -- see `_ShapedWaypointEnv`'s own docstring
                  for the full reasoning (a Step 4.5 red-team pass, fable-5.1 cross-model,
                  found the potential-difference formulation BLOCKING): `_train_all_on_agent`
                  credits every tick in an episode with the SAME whole-episode return
                  (`ep_reward`, summed once and broadcast to every REINFORCE outcome tuple),
                  under which a telescoping potential-difference term collapses to a single
                  boundary quantity dominated by env-respawn/timeout noise rather than by
                  anything resembling "reward density". A monotone, additive (non-cancelling)
                  progress bonus does not have this defect and genuinely increases what the
                  learner is credited with in proportion to within-episode approach behaviour.
                  Clamped non-negative so it cannot introduce a harm_signal<0 tick that the
                  unshaped env could not already produce (relevant because `harm_signal` also
                  drives `agent.update_residue`'s E3 harm-triggered commitment-abort branch;
                  see `_ShapedWaypointEnv` docstring for the full disclosure of what this
                  wrapper does and does not decouple from the substrate's affective pathway).
  demo_warmstart  N_DEMO_EPISODES of AUXILIARY supervised warm-start on the SAME
                  lateral-PFC bias head P1 REINFORCE later trains, THEN
                  `_train_all_on_agent(p1_episodes=RL_EPISODES)` UNMODIFIED (identical budget
                  and env to sparse_rl). The warm-start never overrides the agent's own action
                  or touches env.step's action argument -- the agent always acts on its own
                  real E3-selected choice throughout (avoiding any risk of corrupting E3's
                  internal commitment/beta-gate state machinery via an externally forced
                  action, which nothing in this corpus does and which could not be validated
                  against the substrate's own invariants at this session's disposal). Instead,
                  at every tick, alongside the agent's own action, an ORACLE LABEL (greedy
                  direction to the pending waypoint, from env ground truth, identical formula
                  to V3-EXQ-1004/1030's `_oracle_action`) is computed and used as a supervised
                  cross-entropy target for `agent.lateral_pfc.compute_bias(candidate_summaries)`
                  -- the SAME per-candidate bias vector `_lpfc_reinforce_loss` trains via
                  REINFORCE in phase B, and the same `candidate_summaries` input
                  (`_consumed_summaries`, detached) that call already uses, so warm-start and
                  RL fine-tune train literally the same head through the same input channel.
                  This is "learning from demonstration" applied to the one component this
                  consumer's reward-contingent learning also updates -- not a parallel,
                  disconnected imitation network the way 1004's clone is.

NAVIGATION ISOLATION (verbatim rationale from V3-EXQ-1004, reapplied here for the SAME reason:
without it, a reward-maximising learner in ANY arm can rationally ignore waypoints in favour of
hazard avoidance / foraging, and a training-signal null would be a property of the OBJECTIVE
COMPOSITION rather than of density/warm-start). The TRAINING/EVAL env has num_hazards=0,
num_resources=0, energy_decay=0.0, hazard_free_contamination_gate=True (SD-094; a hazard-free
probe self-contaminates to death without it). The waypoint field stays ON in every arm (this
leg does not re-vary field ON/OFF -- that leg is resolved by V3-EXQ-1004).

WHY A SEPARATE, RESOURCE-BEARING WARM ENV. The SD-070 P0a encoder-warmup recipe
(`run_zworld_p0`, invoked via `zworld_p0_episodes=` inside `_train_all_on_agent`) trains the
world encoder against a resource-proximity target (SD-018) and needs a live resource signal to
warm up against -- exactly V3-EXQ-1030's own rationale for leaving hazards/resources ON during
ITS warmup phase. A dedicated warm env (num_hazards=2, num_resources=3, matching 1030) supplies
this; `world_obs_dim` is 275 regardless of entity counts (fixed by which optional channels are
enabled: use_proxy_fields + waypoint_proximity_field_enabled), so the SAME agent trained on the
warm env transfers its (body_obs_dim, world_obs_dim, action_dim) construction to the
navigation-isolated train/eval env without re-initialisation. This warmup is GENERIC (unrelated
to waypoints), so a positive result cannot be an artifact of having been trained on the very
quantity under test -- same guard V3-EXQ-1030 already establishes for this exact recipe.

DV-SYMMETRY INVARIANCE (queue-experiment Step 3 mandatory declaration). The DV is
waypoints-visited-per-episode, a behavioural ROLLOUT COUNT from an independently, fully
retrained policy per (seed, arm) cell -- not a selection over a shared externally-scored
candidate set, so a broadcast-additive-constant symmetry has no purchase (there is no shared
scored-candidate axis across arms for a constant to cancel on). Not a rank/order statistic
derived from one shared scalar under a monotone transform (each arm's count comes from its own
distinct trained policy's rollout, not a rescaling of one shared score). Not a permutation-
symmetric aggregate over interchangeable seeds (each seed differs by construction: distinct
RNG stream, distinct training trajectory, distinct final policy). None of the three
manipulation-invisibility classes in the Step 3 table can mask an sparse/shaped/demo_warmstart
contrast on this DV.

CLAIMS. INV-086 and MECH-428 are carried as READ-ACROSS ONLY -- NEITHER claim's own
`what_would_answer` regime is exercised: INV-086 needs a WM-decay regime + feedback-channel
ablation; MECH-428 needs a seeding-sparse z_goal_norm + forced-seed control. This run
instantiates neither. `evidence_direction_per_claim` is `non_contributory` for both,
unconditionally, matching the sibling H2 leg (V3-EXQ-1030)'s own disposition.

experiment_purpose: diagnostic -- a GOV-FANOUT-1 discrimination leg; it does not test either
claim's own hypothesis and PROMOTES/DEMOTES NOTHING. Routes to /failure-autopsy for
adjudication before any governance action.

GOV-REUSE-1 (Step 2.4): checked `reanalysis_query.py --readout visits_per_episode --claim
MECH-428` against v3_exq_884 (substrate_hash 0091fba4d567a1ae), v3_exq_1030 (substrate_hash
85adf63e99c09c71) and v3_exq_1004 (substrate_hash 9a9fbe795140370f) -- 0/3 carry the readout
-> not recoverable, run.

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow

SLEEP DRIVER: none (no sleep flag set by this driver; use_sleep_loop / sws_enabled /
rem_enabled / use_sleep_aggregation_cluster all OFF/default).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fnn  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._metrics import p0_readiness_gate, P0NotReady, check_degeneracy  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.allon_training import (  # noqa: E402
    _train_all_on_agent,
    _consumed_summaries,
    _obs_harm,
    _obs_harm_a,
    _obs_harm_history,
    REINFORCE_BATCH_SIZE,
    POLICY_TEMPERATURE,
)
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot,
    assert_world_encoder_trained,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
import experiments.v3_exq_724_competence_localization_diagnostic as x724  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1039_mech428_inv086_waypoint_field_consumer_drive_signal"
QUEUE_ID = "V3-EXQ-1039"
CLAIM_IDS = ["INV-086", "MECH-428"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
DEVICE = torch.device("cpu")

# --- env: navigation-isolated train/eval config (V3-EXQ-1004 rationale, reapplied) -------
GRID_SIZE = 12
N_WAYPOINTS = 3
STEPS_PER_EPISODE = 80
WAYPOINT_VISIT_REWARD = 0.2
WAYPOINT_FIELD_DECAY = 0.25
WAYPOINT_COMPLETION_REWARD = 0.8
SEQUENCE_COMMITMENT_TIMEOUT = 20
HAZARD_FREE_CONTAMINATION_GATE = True

# --- env: dedicated P0a warmup config (generic; resources present for SD-018 target) -----
WARM_NUM_HAZARDS = 2
WARM_NUM_RESOURCES = 3
# P0a warmup rolls out on the SAME waypoint/field geometry (subgoal_mode + field ON) so the
# env's action space and layout family match the train env; only hazard/resource presence
# differs. STEPS_PER_EPISODE shared with the train env (see docstring: world_obs_dim is
# geometry-fixed, not entity-count-fixed, so this does not perturb agent construction).

# --- agent / encoder (SD-008: alpha_world >= 0.9 for z_world fidelity) -------------------
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3

ZWORLD_P0_EPISODES = 20      # SD-070 P0a encoder warmup (generic; run_zworld_p0 recipe)
P0_WARMUP_EPISODES = 25      # ambient e2-forward-model training on the warm env
RL_EPISODES = 90             # self-directed REINFORCE budget (sparse_rl, shaped_rl, and
                              # demo_warmstart's phase-B fine-tune -- IDENTICAL across arms).
                              # Deliberately ~13x SMALLER per cell than the MECH-457 fanout
                              # precedent (V3-EXQ-747/748/749: p0=200 + rl=1000 episodes x
                              # 200 steps) -- this is a bounded FIRST probe of a fresh design
                              # (queue-experiment Step 3 "measure the DV range at probe scale
                              # before pre-registering"; a power-bump of an unproven design is
                              # not appropriate any more than a power-bump of an already-
                              # answered one). If this under-trains rather than genuinely
                              # nulling, that is itself informative and motivates a
                              # larger-budget lettered successor -- see queue note.
N_DEMO_EPISODES = 25         # demo_warmstart phase-A auxiliary CE warm-start episodes
EVAL_EPISODES = 10

SHAPING_COEF = 1.0            # per-step progress-bonus weight -- see _ShapedWaypointEnv
                              # docstring for why this is a monotone bonus, not Ng-et-al
                              # potential-difference shaping (Step 4.5 red-team F1 fix)

SEEDS: List[int] = [42, 43, 44]
MIN_SEEDS = 2                  # strict majority of 3

TREATMENT_ARMS: Tuple[str, ...] = ("sparse_rl", "shaped_rl", "demo_warmstart")
ANCHOR_ARMS: Tuple[str, ...] = ("greedy_oracle", "random_walk")
ARM_ORDER: Tuple[str, ...] = TREATMENT_ARMS + ANCHOR_ARMS

# --- pre-registered bars (REACHABILITY-CHECKED at --probe scale; see queue note) ---------
# C0 readiness: greedy_oracle must clear this floor (visits/ep) at this budget/geometry --
# the positive control that the env/geometry is solvable, same statistic C1 routes on.
ORACLE_FLOOR = 4.0
# C0c readiness: the (oracle - random) span must clear this floor before LIFT_MARGIN_FRAC
# (a fraction OF that span) is a meaningful bar -- a near-zero span would make ANY positive
# lift clear the fraction trivially, which is exactly the degenerate case a span-relative
# margin is otherwise supposed to avoid.
MIN_ACHIEVABLE_SPAN = 2.0
# R1: sparse_rl must stay within this fraction of the (oracle - random) span, i.e. the
# premise "the consumer stays at the floor under stock sparse reward" must hold for THIS
# run's own arms before a null/lift verdict is meaningful. Mirrors V3-EXQ-1004's
# BASELINE_BLOCKED_FRAC, self-calibrated to this run's own achievable span rather than an
# imported literal.
BASELINE_BLOCKED_FRAC = 0.35
# C1 (load-bearing): a dense/warm-started arm must beat sparse_rl's OWN per-seed value by at
# least this FRACTION of the (oracle - random) achievable span to count as a lift. A
# span-relative fraction, not an absolute visits/ep literal: an attempted --probe-scale
# reachability run (see queue note) did not complete within session time -- REE all-on
# agent steps are expensive enough that even the reduced probe budget did not finish a
# single seed in ~10 minutes of wall time -- so no empirical effect size was available to
# tune an absolute bar against. Self-calibrating to whatever span THIS run's own oracle/
# random anchors realise avoids importing an absolute literal from an unrelated budget/
# geometry (the mistake CLAUDE.md's caveat about the docstring-only 1004 A2C figure warns
# against), at the cost of being a genuinely un-piloted bar -- stated plainly rather than
# presented as pilot-verified.
LIFT_MARGIN_FRAC = 0.20
# R2: a policy that argmaxes into a wall is stationary, not "did not navigate" -- distinct
# cells visited separates the two so a degenerate eval protocol self-routes rather than
# being read as a channel/training-signal verdict. Step 4.5 red-team F12: a 4.0 floor over
# STEPS_PER_EPISODE=80 is weak (the dry run's own random-walk anchor reached 6 distinct
# cells in just 10 steps) -- 8.0 is still well below a genuine random walk's expected
# reach at this budget, but no longer trivially cleared by a near-stationary policy.
MIN_DISTINCT_CELLS = 8.0

# --- dry-run / probe budgets --------------------------------------------------------------
DRY_SEEDS = [42]
DRY_ZWORLD_P0 = 2
DRY_P0_WARMUP = 2
DRY_RL = 3
DRY_DEMO = 2
DRY_STEPS = 10
DRY_EVAL = 2


def _mean(vals: List[float]) -> float:
    return float(statistics.fmean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------------------
# Env builders
# ---------------------------------------------------------------------------------------
def _make_warm_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed, size=GRID_SIZE, use_proxy_fields=True, subgoal_mode=True,
        num_waypoints=N_WAYPOINTS, waypoint_visit_reward=WAYPOINT_VISIT_REWARD,
        subgoal_arrival_position_check=True,
        num_hazards=WARM_NUM_HAZARDS, num_resources=WARM_NUM_RESOURCES,
        waypoint_completion_reward=WAYPOINT_COMPLETION_REWARD,
        sequence_commitment_timeout=SEQUENCE_COMMITMENT_TIMEOUT,
        waypoint_proximity_field_enabled=True,
        waypoint_field_decay=WAYPOINT_FIELD_DECAY,
    )


def _make_train_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed, size=GRID_SIZE, use_proxy_fields=True, subgoal_mode=True,
        num_waypoints=N_WAYPOINTS, waypoint_visit_reward=WAYPOINT_VISIT_REWARD,
        subgoal_arrival_position_check=True,
        num_hazards=0, num_resources=0, energy_decay=0.0,
        hazard_free_contamination_gate=HAZARD_FREE_CONTAMINATION_GATE,
        waypoint_completion_reward=WAYPOINT_COMPLETION_REWARD,
        sequence_commitment_timeout=SEQUENCE_COMMITMENT_TIMEOUT,
        waypoint_proximity_field_enabled=True,
        waypoint_field_decay=WAYPOINT_FIELD_DECAY,
    )


class _ShapedWaypointEnv:
    """Proxy around a navigation-isolated CausalGridWorldV2 that adds a MONOTONE, NON-
    NEGATIVE progress bonus toward the pending waypoint to the reward channel `step()`
    returns: `bonus = max(0, manhattan_prev - manhattan_next)`. Observations, transitions
    and env dynamics are untouched -- only the scalar reward is augmented. shaping_coef=0.0
    would be byte-identical to the unwrapped env; sparse_rl never uses this wrapper at all,
    so there is no shared-code path that could silently leak shaping into the sparse arm.

    NOT Ng-Harada-Russell potential-difference shaping, and deliberately so -- a Step 4.5
    red-team pass (fable-5.1, cross-model) found that design BLOCKING. `_train_all_on_agent`'s
    P1 REINFORCE (`experiments/_lib/allon_training.py`) credits every tick in an episode with
    the SAME whole-episode return (`ep_reward`, summed once, broadcast to every
    `(cand_features, sel)` tuple -- verified at source: `ep_reward += harm_signal` per step,
    then `outcome_buf.append((cand_features, sel, ep_reward))` for every tick at episode end).
    A telescoping potential-difference term `gamma*Phi(s')-Phi(s)` collapses, under exactly
    this whole-episode-sum objective, to the single boundary quantity `Phi(s_T)-Phi(s_0)` --
    it adds ZERO per-step reward density to what the learner is actually credited with, and
    that boundary term is dominated by where the episode happens to truncate (respawn-timeout
    noise: `sequence_commitment_timeout` fires a fresh randomised waypoint every 20 uncredited
    steps, and completing a sequence respawns the whole set) rather than by anything the
    training-signal manipulation is supposed to test. So a plain potential-difference formula
    would make `shaped_rl` a evaluate-a-boundary-noise-term arm, not a denser-training-signal
    arm, and the registered hypothesis would be untestable under ANY outcome. A monotone
    per-step progress bonus does not have this problem: it is additive (not a cancelling
    difference) across the whole-episode sum, so it genuinely increases `ep_reward`'s
    magnitude and its correlation with within-episode approach behaviour -- which is what
    "a denser training signal" is supposed to mean here. Clamped to `max(0, ...)` (never
    negative) so it cannot introduce a NEW negative harm_signal into `agent.update_residue`
    (see F2 in the red-team's findings, `/tmp/redteam_1039_findings.md`) beyond what the
    unshaped env can already produce -- and in this navigation-isolated config (hazards=0,
    resources=0) the unshaped env's `harm_signal` is provably non-negative in every step
    (`causal_grid_world.py::step` initialises `harm_signal = 0.0` and every negative-valued
    branch is gated on hazard/contamination presence, both zeroed here), so `shaped_rl`'s
    harm_signal stays non-negative just like every other arm's -- the E3
    post_action_update harm-triggered commitment-abort branch (gated on `harm_signal < 0`)
    therefore never fires differently across arms on this account. The progress bonus DOES
    add a larger positive value into `update_residue`'s benefit-accumulation path than the
    other arms see on progress steps -- disclosed, not eliminated: REE's reward-contingent
    residue/benefit machinery is wired to read the same reward channel this experiment's
    entire question is about, and de-coupling "the number driving the REINFORCE credit" from
    "the number driving the substrate's affective bookkeeping" would require editing the
    shared, heavily-audited `_train_all_on_agent`, which this driver deliberately does not
    do (module docstring)."""

    def __init__(self, env: CausalGridWorldV2, shaping_coef: float) -> None:
        self._env = env
        self.shaping_coef = float(shaping_coef)
        self._dist_prev: Optional[int] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self):
        flat, obs = self._env.reset()
        self._dist_prev = _pending_waypoint_manhattan(self._env)
        return flat, obs

    def step(self, action):
        flat, r, done, info, obs = self._env.step(action)
        dist_next = _pending_waypoint_manhattan(self._env)
        bonus = 0.0
        if self._dist_prev is not None and dist_next is not None:
            bonus = max(0.0, float(self._dist_prev - dist_next))
        self._dist_prev = dist_next
        shaped = float(r) + self.shaping_coef * bonus
        return flat, shaped, done, info, obs


# ---------------------------------------------------------------------------------------
# Agent builder (V3-EXQ-724/734/737/742 "all-ON" recipe -- E1/E2/E3, dACC, OFC, lateral-PFC,
# modulatory authority all live; action selection is the REAL generate_trajectories ->
# select_action loop, reused unmodified via _train_all_on_agent).
# ---------------------------------------------------------------------------------------
def _make_agent(env: CausalGridWorldV2, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    kwargs = x724._base_config_kwargs(env)
    kwargs.update(x724._all_on_extra_kwargs())
    kwargs["alpha_world"] = ALPHA_WORLD
    kwargs["alpha_self"] = ALPHA_SELF
    kwargs["self_dim"] = 32
    kwargs["world_dim"] = 32
    cfg = x724.REEConfig.from_dims(**kwargs)
    return x724.REEAgent(cfg)


# ---------------------------------------------------------------------------------------
# Oracle / anchors (env ground truth; never reads the field -- cannot launder the
# manipulation into its own reference). Direction formula verbatim from V3-EXQ-1004/1030.
# ---------------------------------------------------------------------------------------
def _oracle_action(env: CausalGridWorldV2) -> int:
    idx = int(getattr(env, "_next_waypoint_idx", 0))
    wps = getattr(env, "waypoints", []) or []
    if not wps or idx >= len(wps):
        return 4
    wx, wy = int(wps[idx][0]), int(wps[idx][1])
    ax, ay = int(env.agent_x), int(env.agent_y)
    dx, dy = wx - ax, wy - ay
    if abs(dx) >= abs(dy) and dx != 0:
        return 1 if dx > 0 else 0
    if dy != 0:
        return 3 if dy > 0 else 2
    return 4


def _pending_waypoint_manhattan(env: CausalGridWorldV2) -> Optional[int]:
    idx = int(getattr(env, "_next_waypoint_idx", 0))
    wps = getattr(env, "waypoints", []) or []
    if not wps or idx >= len(wps):
        return None
    wx, wy = int(wps[idx][0]), int(wps[idx][1])
    return abs(wx - int(env.agent_x)) + abs(wy - int(env.agent_y))


def _random_action(env: CausalGridWorldV2, rng: np.random.RandomState) -> int:
    return int(rng.randint(0, int(env.action_dim)))


# ---------------------------------------------------------------------------------------
# Rollout / eval counting. NO scripted walk for treatment arms -- act_fn is the arm's OWN
# (trained) policy, driven through agent.act() (the complete V3 REE action loop: sense ->
# e1_tick -> generate_trajectories -> select_action), exactly as the training loop drives
# it, just deterministic-favouring via a low temperature.
# ---------------------------------------------------------------------------------------
EVAL_TEMPERATURE = 0.2


def _rollout_counts(env: CausalGridWorldV2, act_fn, n_episodes: int,
                    steps_per_episode: int) -> Dict[str, Any]:
    visits, seqs, cells, steps = [], [], [], []
    for _ep in range(n_episodes):
        flat, obs = env.reset()
        v = s = 0
        seen = {(int(env.agent_x), int(env.agent_y))}
        n_steps = 0
        for _t in range(steps_per_episode):
            n_steps += 1
            a = act_fn(env, obs, flat)
            flat, _r, done, info, obs = env.step(a)
            tt = str(info.get("transition_type", "") or "")
            if tt == "waypoint":
                v += 1
            elif tt == "sequence_complete":
                v += 1
                s += 1
            seen.add((int(env.agent_x), int(env.agent_y)))
            if done:
                break
        visits.append(float(v))
        seqs.append(float(s))
        cells.append(float(len(seen)))
        steps.append(float(n_steps))
    return {
        "waypoints_visited_per_ep": _mean(visits),
        "sequences_completed_per_ep": _mean(seqs),
        "n_eval_episodes": int(n_episodes),
        "distinct_cells_per_ep": _mean(cells),
        "steps_per_ep": _mean(steps),
    }


def _agent_eval_act(agent) -> Any:
    def _fn(env: CausalGridWorldV2, obs: Dict[str, Any], flat: torch.Tensor) -> int:
        with torch.no_grad():
            action = agent.act(flat, temperature=EVAL_TEMPERATURE)
        if not torch.isfinite(action).all():
            return int(np.random.randint(0, int(env.action_dim)))
        return int(action[0].argmax().item())
    return _fn


def _oracle_eval_act(env: CausalGridWorldV2, obs: Dict[str, Any], flat: torch.Tensor) -> int:
    return _oracle_action(env)


def _make_random_eval_act(seed: int):
    rng = np.random.RandomState(seed + 777)

    def _fn(env: CausalGridWorldV2, obs: Dict[str, Any], flat: torch.Tensor) -> int:
        return _random_action(env, rng)
    return _fn


# ---------------------------------------------------------------------------------------
# demo_warmstart phase A -- auxiliary supervised warm-start on the lateral-PFC bias head.
# The agent ALWAYS acts on its own real E3-selected action (agent.select_action is called
# normally and env.step receives that real action, unmodified) -- this loop never forces an
# action. The oracle direction is used ONLY as a cross-entropy TARGET for
# agent.lateral_pfc.compute_bias(candidate_summaries), the exact per-candidate bias vector
# and the exact detached candidate_summaries input _lpfc_reinforce_loss trains in phase B.
# ---------------------------------------------------------------------------------------
def _demo_warmstart(agent, env: CausalGridWorldV2, seed: int, n_episodes: int,
                    steps_per_episode: int, denom: int) -> Dict[str, Any]:
    has_lpfc = getattr(agent, "lateral_pfc", None) is not None
    if not has_lpfc or n_episodes <= 0:
        return {"demo_warmstart_ran": False, "demo_ce_final_loss": float("nan")}
    bias_opt = torch.optim.Adam(list(agent.lateral_pfc.bias_head_parameters()), lr=1e-3)
    last_loss = float("nan")
    n_labelled_ticks = 0
    n_target_matched = 0
    n_target_defaulted = 0
    for ep in range(n_episodes):
        _flat, obs_dict = env.reset()
        agent.reset()
        for _step in range(steps_per_episode):
            body = obs_dict["body_state"].float()
            world = obs_dict["world_state"].float()
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)
            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=_obs_harm(obs_dict), obs_harm_a=_obs_harm_a(obs_dict),
                obs_harm_history=_obs_harm_history(obs_dict),
            )
            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            oracle_a = _oracle_action(env)
            # Gate the auxiliary CE update on a FRESH E3 tick (validate_experiments.py
            # HOLD-WEIGHTED-E3-READOUT finding): on a held tick, agent.py returns the
            # cached action and generate_trajectories returns cached candidates BEFORE
            # e3.select() is reached, so an ungated update would re-train on the same
            # cached candidate_summaries once per held step -- weighting the CE loss by
            # commitment-hold duration rather than by the number of genuinely distinct
            # selections observed.
            if ticks.get("e3_tick", False) and candidates and len(candidates) >= 2:
                cand_features = _consumed_summaries(agent, candidates)
                if cand_features is not None and torch.isfinite(cand_features).all():
                    # Step 4.5 red-team F4 fix: a miss (no candidate's first action equals
                    # the oracle direction) must SKIP the update, not silently default to
                    # candidate 0 -- training "prefer whatever the proposer ranks first" on
                    # a miss is not learning from demonstration and would systematically
                    # bias the head toward the proposer's own ranking rather than the
                    # oracle's, unattributably to either.
                    target_idx = None
                    for ci, c in enumerate(candidates):
                        if (
                            getattr(c, "actions", None) is not None
                            and c.actions.shape[1] >= 1
                            and int(c.actions[:, 0, :].argmax(-1).reshape(-1)[0].item())
                            == oracle_a
                        ):
                            target_idx = min(ci, cand_features.shape[0] - 1)
                            break
                    if target_idx is None:
                        n_target_defaulted += 1
                    else:
                        n_target_matched += 1
                        bias = agent.lateral_pfc.compute_bias(cand_features)
                        log_p = torch.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
                        loss = -log_p[min(target_idx, log_p.shape[0] - 1)]
                        if torch.isfinite(loss):
                            bias_opt.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                agent.lateral_pfc.bias_head_parameters(), 1.0
                            )
                            bias_opt.step()
                            last_loss = float(loss.detach())
                            n_labelled_ticks += 1

            action = agent.select_action(candidates, ticks)
            if action is None or not torch.isfinite(action).all():
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
            _flat, _harm, done, _info, obs_dict = env.step(action)
            if done:
                break
        cur = ep + 1
        if cur % 20 == 0 or cur == n_episodes:
            print(f"  [train] demo_warmstart seed={seed} phase=DEMO ep {cur}/{denom}",
                  flush=True)
    return {
        "demo_warmstart_ran": True,
        "demo_ce_final_loss": last_loss,
        "demo_n_labelled_ticks": int(n_labelled_ticks),
        "demo_n_target_matched": int(n_target_matched),
        "demo_n_target_defaulted": int(n_target_defaulted),
    }


# ---------------------------------------------------------------------------------------
# Per-cell runner
# ---------------------------------------------------------------------------------------
_ZG = ZGoalStreamAccumulator()


def _run_treatment_cell(arm_id: str, seed: int, zworld_p0: int, p0_warm: int, rl_eps: int,
                        n_demo: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    warm_env = _make_warm_env(seed)
    agent = _make_agent(warm_env, seed)

    before = latent_stack_snapshot(agent)
    zworld_env = _make_warm_env(seed + 500_000_003)  # F6-style offset: distinct layout stream
    _train_all_on_agent(
        agent, warm_env, seed=seed, p0_episodes=p0_warm, p1_episodes=0,
        steps_per_episode=steps, rung_id=f"h1_{arm_id}", total_denominator=max(1, p0_warm),
        zworld_p0_episodes=zworld_p0, zworld_p0_env=zworld_env,
    )
    zworld_report = assert_world_encoder_trained(
        agent, before, p0=zworld_p0, strict=False,
        context="v3_exq_1039.warmup", escape_hint="strict=False deliberate",
    )

    # Step 4.5 red-team F3: nothing upstream certifies the ONE component every arm's
    # training signal actually reaches (the lateral-PFC bias head) ever moves off its
    # zero-initialised last layer -- a degenerate candidate-summary spread makes its
    # gradient exactly zero regardless of REINFORCE/CE target, in EVERY arm at once,
    # which the dry-run's own `demo_ce_final_loss == ln(K)` signature is consistent with.
    # Snapshot the last-linear weight norm before the arm's own training phase so the
    # delta (and the last-observed candidate-summary-degeneracy flag) can be checked as
    # a readiness precondition in `_score`, alongside C0b's world-encoder check.
    bias_norm_before = None
    if getattr(agent, "lateral_pfc", None) is not None:
        bias_norm_before = agent.lateral_pfc.get_state()["rule_bias_head_last_linear_weight_norm"]

    extra: Dict[str, Any] = {}
    if arm_id == "sparse_rl":
        train_env = _make_train_env(seed)
        _train_all_on_agent(
            agent, train_env, seed=seed, p0_episodes=0, p1_episodes=rl_eps,
            steps_per_episode=steps, rung_id="h1_sparse_rl", total_denominator=max(1, rl_eps),
        )
    elif arm_id == "shaped_rl":
        train_env = _ShapedWaypointEnv(_make_train_env(seed), SHAPING_COEF)
        _train_all_on_agent(
            agent, train_env, seed=seed, p0_episodes=0, p1_episodes=rl_eps,
            steps_per_episode=steps, rung_id="h1_shaped_rl", total_denominator=max(1, rl_eps),
        )
        train_env = train_env._env  # unwrap for eval below (unshaped rollout counts)
    elif arm_id == "demo_warmstart":
        train_env = _make_train_env(seed)
        demo_stats = _demo_warmstart(agent, train_env, seed, n_demo, steps, max(1, n_demo))
        extra.update(demo_stats)
        _train_all_on_agent(
            agent, train_env, seed=seed, p0_episodes=0, p1_episodes=rl_eps,
            steps_per_episode=steps, rung_id="h1_demo_warmstart",
            total_denominator=max(1, rl_eps),
        )
    else:
        raise ValueError(f"unknown treatment arm {arm_id!r}")

    eval_env = _make_train_env(seed)
    row = _rollout_counts(eval_env, _agent_eval_act(agent), eval_eps, steps)
    row["zworld_encoder_trained"] = bool(zworld_report.get("zworld_encoder_trained", False))
    if getattr(agent, "lateral_pfc", None) is not None:
        lpfc_state = agent.lateral_pfc.get_state()
        bias_norm_after = lpfc_state["rule_bias_head_last_linear_weight_norm"]
        row["lpfc_bias_head_moved"] = bool(
            bias_norm_before is not None
            and abs(bias_norm_after - bias_norm_before) > 1e-9
        )
        row["lpfc_candidate_summary_degenerate"] = bool(
            lpfc_state["candidate_summary_degenerate"]
        )
    else:
        row["lpfc_bias_head_moved"] = False
        row["lpfc_candidate_summary_degenerate"] = True
    row.update(extra)
    _ZG.observe(agent)
    return row


def _run_anchor_cell(arm_id: str, seed: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    env = _make_train_env(seed)
    if arm_id == "greedy_oracle":
        return _rollout_counts(env, _oracle_eval_act, eval_eps, steps)
    if arm_id == "random_walk":
        return _rollout_counts(env, _make_random_eval_act(seed), eval_eps, steps)
    raise ValueError(f"unknown anchor {arm_id!r}")


def _arm_config_slice(arm_id: str, zworld_p0: int, p0_warm: int, rl_eps: int, n_demo: int,
                      eval_eps: int, steps: int) -> Dict[str, Any]:
    base = {
        "arm_id": arm_id, "steps_per_episode": int(steps), "eval_episodes": int(eval_eps),
        "grid_size": GRID_SIZE, "n_waypoints": N_WAYPOINTS,
        "waypoint_visit_reward": WAYPOINT_VISIT_REWARD,
        "waypoint_completion_reward": WAYPOINT_COMPLETION_REWARD,
        "waypoint_field_decay": WAYPOINT_FIELD_DECAY, "field_on": True,
        "alpha_world": ALPHA_WORLD, "alpha_self": ALPHA_SELF,
    }
    if arm_id in TREATMENT_ARMS:
        base.update({
            "kind": "reeagent_e1e3_allon", "zworld_p0_episodes": int(zworld_p0),
            "p0_warmup_episodes": int(p0_warm), "rl_episodes": int(rl_eps),
        })
        if arm_id == "shaped_rl":
            base.update({"shaping_coef": SHAPING_COEF})
        if arm_id == "demo_warmstart":
            base.update({"n_demo_episodes": int(n_demo)})
    else:
        base.update({"kind": "anchor"})
    return base


# ---------------------------------------------------------------------------------------
# Score / verdict
# ---------------------------------------------------------------------------------------
def _score(per_seed: Dict[str, List[Dict[str, Any]]], seeds: List[int]) -> Dict[str, Any]:
    def col(arm: str, key: str) -> List[float]:
        return [float(row[key]) for row in per_seed[arm]]

    oracle_v = col("greedy_oracle", "waypoints_visited_per_ep")
    random_v = col("random_walk", "waypoints_visited_per_ep")
    sparse_v = col("sparse_rl", "waypoints_visited_per_ep")
    shaped_v = col("shaped_rl", "waypoints_visited_per_ep")
    demo_v = col("demo_warmstart", "waypoints_visited_per_ep")

    n = len(seeds)
    oracle_mean = _mean(oracle_v)
    random_mean = _mean(random_v)

    # C0 readiness: same statistic (visits/ep) on the KNOWN-POSITIVE oracle control.
    c0_measured = min(oracle_v) if oracle_v else 0.0
    c0 = {
        "name": "greedy_oracle_clears_floor", "kind": "readiness",
        "description": (
            "greedy_oracle (env ground truth, never reads the field) visits/ep clears "
            "ORACLE_FLOOR on every seed at this budget/geometry -- the positive control "
            "that the env/geometry is navigable, same statistic C1 routes on."
        ),
        "control": "greedy_oracle waypoints_visited_per_ep, worst seed",
        "measured": round(float(c0_measured), 6), "threshold": float(ORACLE_FLOOR),
        "direction": "lower",
        "met": bool(c0_measured >= ORACLE_FLOOR),
    }

    zworld_frac = _mean([
        1.0 if row.get("zworld_encoder_trained") else 0.0
        for arm in TREATMENT_ARMS for row in per_seed[arm]
    ])
    c0b = {
        "name": "zworld_encoder_trained_majority_cells", "kind": "readiness",
        "description": (
            "Fraction of (seed, treatment-arm) cells whose SD-070 P0a warmup measurably "
            "moved the world encoder (assert_world_encoder_trained, strict=False)."
        ),
        "control": "latent_stack pre/post-warmup weight-delta audit",
        "measured": round(float(zworld_frac), 6), "threshold": 0.5,
        "direction": "lower",
        "met": bool(zworld_frac >= 0.5),
    }

    # Step 4.5 red-team F3: the ONE component every arm's training signal actually
    # trains (the lateral-PFC bias head; see module docstring) must be confirmed to have
    # moved, and its candidate-summary input confirmed non-degenerate, on a majority of
    # cells -- otherwise a flat null across all three arms is indistinguishable from
    # "the trained component never received a usable gradient in any arm".
    lpfc_moved_frac = _mean([
        1.0 if row.get("lpfc_bias_head_moved") else 0.0
        for arm in TREATMENT_ARMS for row in per_seed[arm]
    ])
    lpfc_nondegenerate_frac = _mean([
        0.0 if row.get("lpfc_candidate_summary_degenerate") else 1.0
        for arm in TREATMENT_ARMS for row in per_seed[arm]
    ])
    c0d = {
        "name": "lpfc_bias_head_trainable_majority_cells", "kind": "readiness",
        "description": (
            "Fraction of (seed, treatment-arm) cells whose lateral-PFC bias-head last "
            "linear layer measurably moved from its zero-init AND whose final "
            "candidate-summary input was non-degenerate (LateralPFCAnalog.get_state()) "
            "-- the component every arm's training signal is confirmed (module docstring) "
            "to actually train."
        ),
        "control": "rule_bias_head last-linear weight-norm delta + candidate_summary_degenerate flag",
        "measured": round(float(min(lpfc_moved_frac, lpfc_nondegenerate_frac)), 6),
        "threshold": 0.5, "direction": "lower",
        "met": bool(lpfc_moved_frac >= 0.5 and lpfc_nondegenerate_frac >= 0.5),
    }

    span = max(0.0, oracle_mean - random_mean)
    c0c = {
        "name": "achievable_span_clears_floor", "kind": "readiness",
        "description": (
            "(oracle - random) visits/ep span clears MIN_ACHIEVABLE_SPAN -- below this, "
            "LIFT_MARGIN_FRAC (a fraction OF the span) would be a near-zero, trivially-"
            "cleared bar rather than a meaningful one."
        ),
        "control": "greedy_oracle_mean - random_walk_mean, this run's own anchors",
        "measured": round(float(span), 6), "threshold": float(MIN_ACHIEVABLE_SPAN),
        "direction": "lower",
        "met": bool(span >= MIN_ACHIEVABLE_SPAN),
    }

    if not (c0["met"] and c0b["met"] and c0c["met"] and c0d["met"]):
        return {
            "label": "substrate_not_ready_requeue",
            "preconditions": [c0, c0b, c0c, c0d],
            "criteria_non_degenerate": {},
            "combination_rule": "C0 AND C0b AND C0c AND C0d (all readiness) gate before any verdict criterion",
            "oracle_mean": round(oracle_mean, 6), "random_mean": round(random_mean, 6),
        }

    sparse_mean = _mean(sparse_v)
    r1_measured = sparse_mean
    r1_threshold = random_mean + BASELINE_BLOCKED_FRAC * span
    r1 = {
        "name": "sparse_rl_stays_blocked", "kind": "premise",
        "description": (
            "sparse_rl (stock waypoint reward only) stays within BASELINE_BLOCKED_FRAC of "
            "the (oracle - random) span -- the premise this run's null/lift verdict "
            "depends on. Self-calibrated to THIS run's own achievable span."
        ),
        "measured": round(float(r1_measured), 6), "threshold": round(float(r1_threshold), 6),
        "direction": "upper",
        "met": bool(r1_measured <= r1_threshold),
    }
    if not r1["met"]:
        return {
            "label": "sparse_baseline_not_blocked",
            "preconditions": [c0, c0b, c0c, c0d, r1],
            "criteria_non_degenerate": {"C1": False},
            "combination_rule": "R1 unmet -- the blocked-baseline premise is false",
            "oracle_mean": round(oracle_mean, 6), "random_mean": round(random_mean, 6),
            "sparse_mean": round(sparse_mean, 6),
        }

    def per_seed_lift(dense: List[float]) -> List[float]:
        return [d - s for d, s in zip(dense, sparse_v)]

    lift_margin = LIFT_MARGIN_FRAC * span
    shaped_lift = per_seed_lift(shaped_v)
    demo_lift = per_seed_lift(demo_v)
    shaped_clears = sum(1 for x in shaped_lift if x >= lift_margin)
    demo_clears = sum(1 for x in demo_lift if x >= lift_margin)
    shaped_negative = sum(1 for x in shaped_lift if x <= -lift_margin)
    demo_negative = sum(1 for x in demo_lift if x <= -lift_margin)

    r2_cells = [
        row["distinct_cells_per_ep"]
        for arm in TREATMENT_ARMS for row in per_seed[arm]
    ]
    r2 = {
        "name": "treatment_arms_move", "kind": "eval_protocol",
        "description": "Every treatment-arm cell visits >= MIN_DISTINCT_CELLS (not a "
                       "stationary argmax-into-wall eval policy).",
        "measured": round(float(min(r2_cells)) if r2_cells else 0.0, 6),
        "threshold": float(MIN_DISTINCT_CELLS),
        "direction": "lower",
        "met": bool(r2_cells and min(r2_cells) >= MIN_DISTINCT_CELLS),
    }

    c1_clear = (shaped_clears >= MIN_SEEDS) or (demo_clears >= MIN_SEEDS)
    c1_anomalous_negative = (shaped_negative >= MIN_SEEDS) or (demo_negative >= MIN_SEEDS)

    criteria_non_degenerate = {
        "C1": bool(r2["met"]),
    }
    preconditions = [c0, c0b, c0c, c0d, r1, r2]

    if not r2["met"]:
        label = "degenerate_eval_protocol"
    elif c1_clear and not c1_anomalous_negative:
        label = "dense_or_warmstart_signal_converts_h1_supported"
    elif c1_anomalous_negative and not c1_clear:
        label = "dense_or_warmstart_signal_anomalous_negative_lift"
    elif c1_clear and c1_anomalous_negative:
        label = "dense_or_warmstart_signal_inconsistent_across_arms"
    else:
        label = "training_signal_does_not_convert_h1_not_supported"

    return {
        "label": label,
        "preconditions": preconditions,
        "criteria_non_degenerate": criteria_non_degenerate,
        "combination_rule": (
            "C1 = (>=MIN_SEEDS of N seeds clear LIFT_MARGIN_FRAC*span under shaped_rl) OR "
            "(>=MIN_SEEDS under demo_warmstart), evaluated only after C0/C0b/C0c/C0d/R1/R2 hold"
        ),
        "note": (
            "Step 4.5 red-team F9: the composite label is an OR/AND over BOTH arms and can "
            "read 'inconsistent_across_arms' even when ONE arm (shaped_rl or demo_warmstart) "
            "gave a clean, clearly-directional result -- read shaped_clears_n_seeds / "
            "demo_clears_n_seeds and shaped_lift_per_seed / demo_lift_per_seed individually "
            "before trusting the composite label as the interpretable quantity."
        ) if label == "dense_or_warmstart_signal_inconsistent_across_arms" else None,
        "load_bearing": True,
        "oracle_mean": round(oracle_mean, 6), "random_mean": round(random_mean, 6),
        "sparse_mean": round(sparse_mean, 6),
        "shaped_mean": round(_mean(shaped_v), 6), "demo_mean": round(_mean(demo_v), 6),
        "shaped_lift_per_seed": [round(x, 6) for x in shaped_lift],
        "demo_lift_per_seed": [round(x, 6) for x in demo_lift],
        "shaped_clears_n_seeds": int(shaped_clears), "demo_clears_n_seeds": int(demo_clears),
        "min_seeds_required": int(MIN_SEEDS), "n_seeds": int(n),
    }


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------
def main(dry_run: bool = False, probe: bool = False) -> Dict[str, Any]:
    if dry_run:
        seeds = DRY_SEEDS
        zworld_p0, p0_warm, rl_eps = DRY_ZWORLD_P0, DRY_P0_WARMUP, DRY_RL
        n_demo, eval_eps, steps = DRY_DEMO, DRY_EVAL, DRY_STEPS
    elif probe:
        seeds = [42]
        zworld_p0, p0_warm, rl_eps = 8, 10, 15
        n_demo, eval_eps, steps = 8, 5, 30
    else:
        seeds = SEEDS
        zworld_p0, p0_warm, rl_eps = ZWORLD_P0_EPISODES, P0_WARMUP_EPISODES, RL_EPISODES
        n_demo, eval_eps, steps = N_DEMO_EPISODES, EVAL_EPISODES, STEPS_PER_EPISODE

    t0 = __import__("time").perf_counter()
    per_seed: Dict[str, List[Dict[str, Any]]] = {arm: [] for arm in ARM_ORDER}
    arm_results: List[Dict[str, Any]] = []

    for seed in seeds:
        for arm_id in ARM_ORDER:
            cfg_slice = _arm_config_slice(arm_id, zworld_p0, p0_warm, rl_eps, n_demo,
                                          eval_eps, steps)
            with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__)) as cell:
                if arm_id in TREATMENT_ARMS:
                    row = _run_treatment_cell(arm_id, seed, zworld_p0, p0_warm, rl_eps,
                                              n_demo, eval_eps, steps)
                else:
                    row = _run_anchor_cell(arm_id, seed, eval_eps, steps)
                cell.stamp(row)
            row["arm_id"] = arm_id
            row["seed"] = int(seed)
            per_seed[arm_id].append(row)
            arm_results.append(row)
            # Signals "this (seed, arm) cell's run completed", per the runner's progress
            # convention (CLAUDE.md Step 3 "Progress instrumentation") -- NOT a scientific
            # per-cell verdict (there is no natural per-cell pass/fail here; the actual
            # verdict is the cross-arm comparison in _score, printed once below). Matches
            # V3-EXQ-1030's identical convention for the same reason.
            print("verdict: PASS", flush=True)

    verdict = _score(per_seed, seeds)
    elapsed = __import__("time").perf_counter() - t0

    non_degen = check_degeneracy({
        "waypoints_visited_per_ep_spread": {
            "values": [row["waypoints_visited_per_ep"] for row in arm_results],
        },
    })

    outcome = "PASS" if verdict["label"] not in (
        "substrate_not_ready_requeue", "sparse_baseline_not_blocked", "degenerate_eval_protocol",
    ) else "FAIL"

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "unknown",
        "evidence_direction_per_claim": {"INV-086": "non_contributory",
                                         "MECH-428": "non_contributory"},
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "timestamp_utc": ts,
        "sleep_driver_pattern": "none",
        "interpretation": {
            "label": verdict["label"],
            "preconditions": verdict.get("preconditions", []),
            "criteria_non_degenerate": verdict.get("criteria_non_degenerate", {}),
        },
        "combination_rule": verdict.get("combination_rule"),
        "readout": {
            "oracle_mean_visits_per_ep": verdict.get("oracle_mean"),
            "random_mean_visits_per_ep": verdict.get("random_mean"),
            "sparse_mean_visits_per_ep": verdict.get("sparse_mean"),
            "shaped_mean_visits_per_ep": verdict.get("shaped_mean"),
            "demo_mean_visits_per_ep": verdict.get("demo_mean"),
            "shaped_clears_n_seeds": verdict.get("shaped_clears_n_seeds"),
            "demo_clears_n_seeds": verdict.get("demo_clears_n_seeds"),
            "min_seeds_required": verdict.get("min_seeds_required"),
        },
        "arm_results": arm_results,
        "per_seed": per_seed,
        "n_seeds": len(seeds),
        "budget": {
            "zworld_p0_episodes": zworld_p0, "p0_warmup_episodes": p0_warm,
            "rl_episodes": rl_eps, "n_demo_episodes": n_demo,
            "eval_episodes": eval_eps, "steps_per_episode": steps,
        },
    }
    manifest.update(non_degen)

    out_path = write_flat_manifest(
        manifest, dry_run=dry_run,
        config={
            "seeds": seeds, "budget": manifest["budget"],
            "shaping_coef": SHAPING_COEF,
            "lift_margin_frac": LIFT_MARGIN_FRAC, "min_achievable_span": MIN_ACHIEVABLE_SPAN, "oracle_floor": ORACLE_FLOOR,
            "baseline_blocked_frac": BASELINE_BLOCKED_FRAC,
        },
        seeds=seeds, script_path=Path(__file__), started_at=t0, elapsed_seconds=elapsed,
        z_goal_stream_stats=_ZG.stats(),
    )

    print(f"label: {verdict['label']}", flush=True)
    print(f"outcome: {outcome}", flush=True)

    return {"outcome": outcome, "manifest_path": str(out_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe", action="store_true",
                        help="reachability check only -- NOT evidence, see queue note")
    args = parser.parse_args()
    _res = main(dry_run=args.dry_run, probe=args.probe)
    _o = str(_res["outcome"]).upper()
    emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                 manifest_path=_res.get("manifest_path"), dry_run=args.dry_run)
