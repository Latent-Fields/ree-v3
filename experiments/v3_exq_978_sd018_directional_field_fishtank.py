"""V3-EXQ-978 -- SD-018 AMEND validation: does supervising z_world on the DIRECTIONAL
resource field lift a z_world-only reader's foraging competence? (+ fishtank observable)

Claims: INV-088 (primary), MECH-457 (read-across)

EXPERIMENT_PURPOSE = "evidence"

SLEEP DRIVER: not applicable -- no sleep flag is set by this driver (the x734 all-ON stack
does not enable use_sleep_loop / sws_enabled / rem_enabled / use_sleep_aggregation_cluster
at this rung). Recorded as sleep_driver_pattern="none" rather than omitted, so a reader can
tell "no sleep" from "nobody declared".

red-team (fable): see the queue entry note for V3-EXQ-978.

=== WHY THIS RUN ===

This is the owed post-implementation validation of the SD-018 AMEND (directional
resource-field head on z_world), landed on ree-v3 main as 028a625 on 2026-09-02, whose own
commit message records "Validation experiment NOT queued". Routing chip:
chip-20260902-sd018-fieldhead-validation-run.

The confirmed autopsy failure_autopsy_V3-EXQ-948_2026-08-25 (H-observation-interface
CONFIRMED, user-confirmed) established the gap this amend is meant to close. With SD-018's
SCALAR proximity head already active (the x734/737/808/948 family base config), a PPO reader
of z_world ALONE forages 0.5 res/ep against the 1.0 D3 competence floor, while the same
reader handed the full 25-dim `resource_field_view` clears it 3/3 (2.23 res/ep). That field
is `world_obs[225:250]` -- already INSIDE z_world's own input. z_world discards it. A scalar
max(field) target supervises MAGNITUDE only ("how close is food"); foraging needs DIRECTION.

The amend's shape (a) -- chosen over (b) routing the raw field past z_world as a side-channel
-- generalises the scalar head from 1 to 25 dims, so the directional gradient has to survive
INTO the latent rather than around it. This run is the test of whether that works.

  If it PASSES: the observation-interface leg converts, and the amend is the right lever.
  If it NULLS:  shape (b) is the next build. That is a real, pre-registered outcome, not a
                failure of this run -- see "WHAT A NULL MEANS" below.

=== WHAT THIS RUN CONTRIBUTES THAT 948 COULD NOT ===

948 is a POSITIVE DEMONSTRATION with a side-channel: it concatenated the raw field onto the
reader's input. That proves the CONTENT is sufficient for competence; it says nothing about
whether the content can be carried THROUGH z_world. This run removes the side-channel
entirely -- both arms' readers see z_world and nothing else (32 dims) -- and moves the
manipulation upstream into how z_world is trained. The reader is identical across arms; only
the encoder's P0 objective differs.

=== ARMS (2) -- learner, objective, env and budget held FIXED ===

  field_loss_off   z_world (32)   directional head BUILT but its P0a loss weight is 0.0, so
                                  it is never trained and nothing reads it. Replication
                                  anchor for 948's 0.5 res/ep latent arm.
  field_loss_on    z_world (32)   same architecture; directional head TRAINED in P0a at
                                  resource_field_weight=P0A_FIELD_WEIGHT.

ARCHITECTURE IS IDENTICAL ACROSS ARMS; ONLY THE SUPERVISION SIGNAL DIFFERS. This is a
correction made before queueing, not a stylistic choice -- see `_make_agent`. Constructing
`resource_field_head` consumes torch RNG draws and SplitEncoder builds it BEFORE
world_topdown / self_topdown, so an arm pair differing in whether the head EXISTS also
differs in the random init of every module built after it. Those modules are never trained
(P0a optimises world_encoder + world_precision_logit only; allon_training touches no topdown
path) yet they feed sense()-time z_world directly. Measured at seed 42 pre-training:
world_encoder identical across arms, world_topdown and self_topdown different. Building the
head in both arms removes that confound entirely and sharpens the question from "head plus
its training" to "the training signal alone".

Both arms get their OWN warmup. This is the one structural difference from 948, and it is
forced: 948 shared a single warmed stack across its arms precisely BECAUSE its manipulation
was downstream of the encoder (what the reader was handed). Here the manipulation IS the
encoder's training objective, so a shared warmup would erase it. Everything else -- env
kwargs, rung, seeds, weighting, PPO hyperparameters, eval protocol and the P0/P1 budgets --
is imported from x734/x808 rather than redefined, so this driver cannot drift from the family
it reads against.

=== THE SEAM THIS RUN DEPENDS ON (read before re-running) ===

Between the amend landing and 2026-09-02, the directional head was UNTRAINABLE from any
driver in this family: `ZWorldP0Config.resource_field_weight` defaults to 0.0 (its scalar
sibling `proximity_weight` defaults to 0.5), `_lib/zworld_p0_warmup.run_zworld_p0` -- the
only P0a path `_train_all_on_agent` uses -- built its config with no override, and
`_train_all_on_agent` calls neither `compute_resource_proximity_loss` nor
`compute_resource_field_loss`. So the head would have received ZERO gradient steps and the ON
arm would have differed from OFF only by an untrained randomly-initialised head that nothing
reads -- a manipulation that cannot reach the DV. The seam
(`zworld_p0_resource_field_weight`, default 0.0, contract C6 in
tests/contracts/test_sd018_resource_field_head.py) was added for this run;
chip-20260902-sd018-p0a-field-weight-seam.

Consequence worth stating plainly: the ONLINE P1 loss weight
(`LatentStackConfig.resource_field_weight`, set by `from_dims`) is INERT in this family
because nothing calls `compute_resource_field_loss`. P0a is the operative training path here.
Precondition `field_leg_ran` reports whether the trainer's own three-way gate opened (head
present AND weight > 0 AND world_obs wide enough) -- it is a CONFIGURATION predicate, so it
catches a regression in the seam but cannot certify that the head learned anything. The
measurement that does that is the trainer's held-out `resource_field_holdout.r2`, gated
separately as `field_head_decodes_on_arm`. Both are ON-arm-scoped.

=== DV, AND THE MECHANISM CHECK THAT MAKES IT ATTRIBUTABLE ===

DV: `foraging_competence` (mean resources consumed per eval episode) against the family's
1.0 `COMPETENCE_RESOURCE_FLOOR`, on a strict majority of seeds. Computed by the shared
`_lib.capability_eval.evaluate_seed` -- NOT reimplemented here, so it is the same number
948/813/734 report.

A competence lift alone would be ambiguous: the ON arm's P0 carries an extra loss term, so
its encoder could differ for reasons unrelated to directional content (more gradient signal,
incidental regularisation). So both arms also get a HELD-OUT LINEAR DECODE of
`resource_field_view` from FROZEN z_world, fit on one split and scored on another
(`_field_decode_r2`). That is arm-comparable by construction -- the P0 trainer's own
`resource_field_holdout` exists only on the ON arm and cannot serve as a contrast -- and it
is the direct measurement of the thing the amend claims to change: whether z_world EXPOSES
the gradient. C_decode_lift is that check.

=== THE COLLAPSE HAZARD IS PRE-REGISTERED, NOT DISCOVERED ===

The amend's own ML note names it: a 25-dim target can dominate P0 and collapse a 32-dim
z_world onto the field. This is not hypothetical in this substrate --
`ree_core/latent/zworld_p0.py`'s header records participation_ratio falling from 9.21
(untrained) to 1.06 under SD-009 + scalar SD-018 at lr 1e-4, i.e. onto a single effective
dimension. So `zworld_participation_ratio` is measured per arm and gated
(`zworld_not_collapsed`, floor PARTICIPATION_RATIO_FLOOR). A collapsed ON arm self-routes
substrate_not_ready_requeue at a different operating point -- it must NOT read as "the
directional head does not help".

=== THE MANIPULATION IS A REWEIGHTING, NOT PRESENCE-VS-ABSENCE ===

State this plainly because it bounds what a null can mean. `ZWorldP0Config` also carries
`reconstruction_weight = 10.0` (default, and this driver does not override it), whose head
reconstructs the FULL 250-wide `world_obs` -- which CONTAINS the field at [225:250]. So the
OFF arm's z_world is already supervised on the field, as part of whole-observation
reconstruction, at a per-element weight of 10/250 = 0.04. The ON arm adds a dedicated leg at
0.5/25 = 0.02 per element. The contrast is therefore roughly a 1.5x reweighting of the field
cells through a dedicated sigmoid head, NOT "field supervision vs none".

Consequence for reading the result: the OFF arm's ABSOLUTE decode r2 is as informative as
the lift, and is reported (`off_mean_field_decode_r2`). If OFF already decodes the field well
and neither arm converts, the honest reading is that linear availability of the gradient in
z_world is not the binding constraint -- which is shape (b) territory -- rather than "the
recipe needs a bigger weight".

=== WHAT A NULL MEANS (declared up front, per leg) ===

- ON does not clear the floor, decode r2 DID lift, PR healthy -> the directional content is
  in z_world and still does not convert. That is a real result: it displaces the
  observation-interface leg toward the CONSUMER side and shape (b) is next.
- ON does not clear the floor and decode r2 did NOT lift -> read the OFF arm's ABSOLUTE r2
  first (see the reweighting section above). If OFF already decodes the field well, the
  gradient is linearly present in z_world in both arms and still does not convert: that
  points at the consumer, i.e. shape (b). Only if BOTH arms decode poorly is this a statement
  about the training recipe (weight, lr, schedule) warranting a re-run at another weight.
  Either way it is NOT evidence against the hypothesis, and this run records INV-088 as
  `unknown` on that branch rather than `weakens`.
- OFF CLEARS the floor -> the 948/813 anchor did not replicate and the whole contrast is
  uninterpretable. C_off_subfloor_replication catches exactly this, which is why it is a
  criterion and not an assumption.

=== DV-SYMMETRY INVARIANCE (declared per arm, per the skill's mandatory check) ===

DV = mean resources/episode over eval episodes; its symmetry group is permutation of
episodes and of within-episode step order that leaves consumption counts fixed.

- field_head_off: the manipulation (adding an MSE leg to the P0a objective) is absent by
  construction, so this arm is the control; nothing to be invariant to.
- field_head_on: the manipulation changes `world_encoder` WEIGHTS, hence z_world, hence the
  reader's input distribution, hence which cells the policy visits and consumes. It is NOT a
  broadcast constant added to candidate scores (which an argmax would annihilate), NOT a
  monotone rescaling of a ranked quantity, and NOT a permutation of interchangeable units.
  The DV is a COUNT of consumption events, and the manipulation moves the trajectory that
  generates them. Path open; not invariant.

Corollary, per the same rule: `zworld_encoder_trained_in_p0` certifies that the WORLD path
moved -- it speaks for both arms because both run the same P0a recipe. `field_leg_ran`
certifies only the ON arm's directional leg and is scoped out of OFF accordingly.

=== FISHTANK OBSERVABLE (why this run writes an episode log) ===

Queued at user request so the behaviour is watchable rather than only tabulated. The
companion `<run_id>_episode_log.json` is rendered by REE_assembly's `/fishtank_viz.html`
(discovered via `/api/fishtank/logs`), in the schema that viewer reads: top-level
`experiment_type` / `phase` / `toroidal` / `env_config` / `seeds` / `run_id`, per-seed
`episodes[]`, per-episode `initial_resources` / `initial_hazards` / `steps[]`, and per-step
`t` / `pos` / `action` / `resources` / `hazards` / `health` / `energy`.

Three SD-018-specific channels are added per step so the amend's own subject matter is
directly visible rather than inferred:

  resource_field_true_argmax  which of the 25 local cells the REAL gradient peaks at
  resource_field_pred_argmax  which cell the head PREDICTS it peaks at, read on the
                              SENSE-time z_world the policy actually consumes
  resource_field_pred_argmax_encoder_path
                              the same head read on the ENCODER-path z_world that P0a
                              actually trained (red-team F7)

so a viewer can watch, step by step, whether the agent's internal directional estimate tracks
the actual gradient. All three are recorded on BOTH arms -- since the F1 fix both arms BUILD
the head and only its P0a loss weight differs, so the OFF arm's series is the untrained
within-run chance baseline rather than a null column.

The two predicted series are not redundant. Sense-time z_world is
`(world_encoder(w) + world_topdown(...)) * prec` while P0a trains `world_encoder(w) * prec`
alone, so a head that tracks the gradient on the encoder path but not at sense time localises
the failure to the top-down term the reader sees -- which is precisely the observation that
would route a null to shape (b) rather than back to the P0a recipe.

THE LOG IS A SEPARATE, ADDITIONAL PASS and is NOT the scored data. The DV comes from
`evaluate_seed` on its own fresh env, untouched; the log pass re-runs the SAME trained policy
on ANOTHER fresh env at the same seed purely to record traces. This is deliberate:
reimplementing the eval loop to bolt logging onto it would risk the DV drifting from the
family definition, which is a far worse trade than one extra rollout pass. Logged episode
counts and scored episode counts are therefore different numbers on purpose.

Size is bounded by thinning: full per-step records for LOG_SEEDS only (the first seed) and
the first LOG_EPISODES episodes per arm; every other seed contributes a per-seed summary.
Every statistic in the manifest is computed from the full scored data, so thinning the log
biases nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.capability_eval import (  # noqa: E402
    COMPETENCE_RESOURCE_FLOOR,
    LocalViewGreedyPolicy,
    RandomPolicy,
    evaluate_seed,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot,
    latent_stack_weight_delta,
)
from ree_core.latent.stack import SplitEncoder  # noqa: E402
import experiments.v3_exq_724_competence_localization_diagnostic as x724  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_737_ree_latent_policy_head_competence_probe as x737  # noqa: E402
import experiments.v3_exq_808_return_decomposition_objective_misspecification as x808  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_978_sd018_directional_field_fishtank"
QUEUE_ID = "V3-EXQ-978"
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["INV-088", "MECH-457"]

# Every cell re-warms its own REE stack and trains a PPO actor from scratch; the two arms
# differ in their P0a objective, so neither is a reusable pure-function-of-config baseline a
# later run could bank. Same structural reason 737b / 808 / 813 / 948 exempt themselves.
ARM_FINGERPRINT_EXEMPT = (
    "per-arm P0a objective differs, so each cell re-warms its own encoder; PPO trained from "
    "scratch per cell. No arm is a reusable baseline"
)

DEVICE = torch.device("cpu")

# Same seeds as 948 / 813 so field_head_off is a per-seed replication of their latent anchor.
# Not a reef config at this rung, so the seed-44 reef-instability rule does not apply.
SEEDS: List[int] = [42, 43, 44]

# Budgets sourced from x734 -- never redefined -- so this driver cannot drift from the family.
ZWORLD_P0_EPISODES = x734.ZWORLD_P0_EPISODES        # 60
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

# The decisive rung: hazard-free, oracle-achievable. Imported from 734 so it is byte-identical
# to the sibling family (737b, 808, 813, 948) this run reads against.
RUNG = x734.DIFFICULTY_RUNGS[-1]
RUNG_ID = RUNG["rung_id"]

# Fixed weighting, imported verbatim from 808 -- the level 813/948 trained under.
_W3 = next(w for w in x808.WEIGHTINGS if w["id"] == "W3_survival_zeroed")
LEVEL_ID = str(_W3["id"])
W_CONSUME = float(_W3["w_consume"])
W_SURVIVAL = float(_W3["w_survival"])

ARM_OFF = "field_loss_off"
ARM_ON = "field_loss_on"
ARM_IDS = [ARM_OFF, ARM_ON]

ANCHOR_IDS = ["random_walk", "local_view_greedy"]

LOCALFIELD_KEY = "resource_field_view"
RESOURCE_FIELD_DIM = 25

# --------------------------------------------------------------------------------------
# PRE-REGISTERED THRESHOLDS (constants; never derived from this run's own statistics)
# --------------------------------------------------------------------------------------
# P0a directional-field loss weight for the ON arm. 0.5 mirrors the scalar leg's own default
# (ZWorldP0Config.proximity_weight = 0.5) rather than inventing a new operating point, and
# matches LatentStackConfig.resource_field_weight's default. A null at this weight is a
# statement about THIS operating point -- said so in the docstring's null table.
P0A_FIELD_WEIGHT = 0.5

# Encoder-moved floor: same guard floor the family uses.
ZWORLD_DELTA_FLOOR = x808.ZWORLD_DELTA_FLOOR         # 1e-6

# Anti-collapse floor on z_world's participation ratio. The substrate's own measured
# collapse reads 1.06 (onto a single effective dimension) against 9.21 untrained
# (zworld_p0.py header). 2.0 is a deliberately weak "more than one effective dimension"
# bound: strong enough to catch the named hazard, weak enough not to fail a healthy encoder
# that legitimately concentrates. Direction: FLOOR.
PARTICIPATION_RATIO_FLOOR = 2.0

# Held-out decode lift the ON arm must show over OFF for C_decode_lift. In r2 units on the
# same held-out split. A margin, not >0, so ordinary seed noise cannot carry the criterion.
DECODE_R2_LIFT_MIN = 0.05

# Pairs collected for the frozen-z_world decode probe, and the train fraction.
DECODE_PROBE_STEPS = 1200
DECODE_TRAIN_FRAC = 0.7
DECODE_MIN_TEST = 64

# Fishtank episode-log thinning (see the docstring's size note).
LOG_EPISODES = 6
DRY_RUN_LOG_EPISODES = 2


# Each cell builds a fresh agent inside _run_seed, so a run-level list would keep every
# arm x seed agent alive until the last cell finished. The accumulator reads the counters at
# observe() time instead (skill Step 4, "wiring it").
_ZG = ZGoalStreamAccumulator()


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _holdout_r2(row: Dict[str, Any]) -> float:
    """P0a's own held-out field r2 for one seed row, or -1.0 when it cannot be established.

    -1.0 rather than None so the worst-seed min stays orderable and an absent/undefined
    readout FAILS the `field_head_decodes_on_arm` floor instead of silently passing it
    (red-team F5). r2 is None when the constant-mean baseline MSE is 0, i.e. the target was
    constant over the held-out split -- which certifies nothing either.
    """
    hold = ((row.get("p0a") or {}).get("resource_field_holdout") or {})
    r2 = hold.get("r2")
    return float(r2) if r2 is not None else -1.0


def _arm_contexts() -> List[Dict[str, Any]]:
    """Arm context consumed by the precondition specs' `applies_to`."""
    return [{"id": aid, "field_head_on": (aid == ARM_ON)} for aid in ARM_IDS]


def _localfield_vector(obs_dict: Dict[str, Any]) -> torch.Tensor:
    """The 25-dim agent-centred resource gradient, shaped [1, 25].

    Fails LOUDLY: a missing channel (use_proxy_fields=False) would silently turn the decode
    probe into a zero-information control whose null would read as a scientific result.
    """
    v = obs_dict.get(LOCALFIELD_KEY)
    if v is None:
        raise KeyError(
            "obs_dict has no %r -- this driver requires use_proxy_fields=True. Without it "
            "the SD-018 target does not exist and every field readout is vacuous."
            % (LOCALFIELD_KEY,)
        )
    t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
    return t.reshape(1, -1).float().to(DEVICE)


def _encoder_path_zworld(agent, obs_dict: Dict[str, Any]) -> Optional[torch.Tensor]:
    """z_world along the ENCODER path ALONE: world_encoder(w) * sigmoid(precision).

    This is EXACTLY what P0a optimises (`zworld_p0.ZWorldP0Trainer._encode`), whereas
    sense()-time z_world is `(world_encoder(w) + world_topdown(beta_to_split(z_beta))) * prec`
    -- it carries an additive top-down term that P0a never trains and never sees.

    Reporting both paths is what separates two failure modes that otherwise look identical in
    the DV (red-team F6/F7): "the directional head did not learn" versus "the head learned
    fine on the path P0a trains, and the top-down term the READER consumes swamps it". Only
    the second is an argument for shape (b); the first is an argument about the P0a recipe.
    Returns None if the stack has no split encoder, so callers degrade rather than raise.
    """
    se = getattr(getattr(agent, "latent_stack", None), "split_encoder", None)
    if se is None or not hasattr(se, "world_encoder"):
        return None
    world = obs_dict["world_state"].float()
    if world.dim() == 1:
        world = world.unsqueeze(0)
    with torch.no_grad():
        z = se.world_encoder(world.to(DEVICE))
        z = z * torch.sigmoid(se.world_precision_logit).unsqueeze(0)
    return z.detach().reshape(1, -1).to(DEVICE)


def _participation_ratio(z: torch.Tensor) -> float:
    """PR = (sum eig)^2 / sum(eig^2) over the covariance spectrum of the z samples.

    The standard 'how many dimensions are effectively in use' readout, and the exact
    statistic zworld_p0.py's header reports (9.21 untrained -> 1.06 collapsed). Returns 0.0
    on a degenerate input rather than raising: a PR of 0 fails the floor, which is the
    correct reading for an unusable sample.
    """
    if z.ndim != 2 or z.shape[0] < 2:
        return 0.0
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / float(zc.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    s1 = float(eig.sum())
    s2 = float((eig ** 2).sum())
    if s2 <= 0.0 or s1 <= 0.0:
        return 0.0
    return float((s1 * s1) / s2)


def _collect_zworld_field_pairs(agent, env, seed: int, n_steps: int
                                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(z_world, resource_field_view) pairs under a fixed random policy, encoder FROZEN.

    A random policy (not either arm's trained reader) on purpose: the decode probe must
    measure what the LATENT exposes, not what a particular policy happens to visit. Using an
    arm's own policy would confound the mechanism check with that arm's behaviour -- the
    circularity 948's autopsy flagged in 813's per-arm consumption-share gate.
    """
    pol = RandomPolicy(seed)
    zs: List[torch.Tensor] = []
    fs: List[torch.Tensor] = []
    zes: List[torch.Tensor] = []   # encoder-path z (red-team F6)
    steps_left = int(n_steps)
    while steps_left > 0:
        _flat, obs = env.reset()
        pol.reset(env)
        for _ in range(int(STEPS_PER_EPISODE)):
            if steps_left <= 0:
                break
            with torch.no_grad():
                zs.append(x737._agent_zworld(agent, obs).reshape(-1).cpu())
                fs.append(_localfield_vector(obs).reshape(-1).cpu())
                ze = _encoder_path_zworld(agent, obs)
                if ze is not None:
                    zes.append(ze.reshape(-1).cpu())
            action = pol.act(env, obs)
            with torch.no_grad():
                _f, _h, done, _i, obs = env.step(action)
            steps_left -= 1
            if done:
                break
    if not zs:
        return torch.zeros(0, 1), torch.zeros(0, 1), torch.zeros(0, 1)
    z_enc = torch.stack(zes) if len(zes) == len(zs) else torch.zeros(0, 1)
    return torch.stack(zs), torch.stack(fs), z_enc


def _field_decode_r2(z: torch.Tensor, f: torch.Tensor) -> Dict[str, Any]:
    """Held-out linear decode of the 25-dim field from frozen z_world.

    Least-squares fit on a train split, scored on a disjoint test split against the
    CONSTANT-MEAN predictor -- the same baseline the P0 trainer's own
    `resource_field_holdout` uses, so the two numbers are read on the same scale. r2 <= 0
    means z_world carries no more linear information about the field than its mean does.

    Arm-comparable by construction: identical probe, identical policy, identical split, on
    each arm's own frozen encoder. That comparability is the whole point -- the trainer's
    built-in holdout exists only on the ON arm and cannot form a contrast.
    """
    n = int(z.shape[0])
    n_train = int(n * DECODE_TRAIN_FRAC)
    n_test = n - n_train
    if n < 8 or n_test < DECODE_MIN_TEST:
        return {"r2": None, "n_train": n_train, "n_test": n_test,
                "reason": "insufficient pairs for a held-out decode"}
    ztr, zte = z[:n_train], z[n_train:]
    ftr, fte = f[:n_train], f[n_train:]
    # bias column, so the fit can express the constant-mean predictor as a special case
    one_tr = torch.ones(ztr.shape[0], 1)
    one_te = torch.ones(zte.shape[0], 1)
    xtr = torch.cat([ztr, one_tr], dim=1)
    xte = torch.cat([zte, one_te], dim=1)
    sol = torch.linalg.lstsq(xtr, ftr).solution
    pred = xte @ sol
    sse = float(((pred - fte) ** 2).mean())
    base = float(((ftr.mean(dim=0, keepdim=True).expand_as(fte) - fte) ** 2).mean())
    r2 = (1.0 - sse / base) if base > 0.0 else None
    return {"r2": (float(r2) if r2 is not None else None),
            "mse": sse, "mean_predictor_mse": base,
            "n_train": int(n_train), "n_test": int(n_test)}


def _make_agent(env):
    """All-ON x734 stack. BOTH arms construct the directional head -- identically.

    ARCHITECTURE IS HELD FIXED ACROSS ARMS ON PURPOSE, and this is not cosmetic. Building
    `resource_field_head` consumes torch RNG draws, and SplitEncoder constructs it BEFORE
    `world_topdown` / `self_topdown` (stack.py), with LatentStack building `beta_encoder` ...
    `beta_to_split` afterwards. So an arm pair that differed in whether the head EXISTS would
    also differ in the random initialisation of every module built after it. That matters
    because those modules are never trained: the P0a recipe optimises
    `world_path_parameters()` = world_encoder + world_precision_logit ONLY
    (`zworld_p0.py`), and `_lib/allon_training.py` contains no topdown reference at all --
    while sense()-time z_world is `(world_encoder(w) + world_topdown(beta_to_split(z_beta)))
    * prec`. Measured at seed 42 before any training: world_encoder identical across arms,
    world_topdown and self_topdown DIFFERENT. An arm-correlated untrained random term in the
    very quantity the reader consumes would have made any competence delta unattributable.

    So the head is built in both arms and the ONLY difference is its P0a LOSS WEIGHT
    (`zworld_p0_resource_field_weight`: 0.0 vs P0A_FIELD_WEIGHT). On the OFF arm the head
    exists, is never trained, and is read by nothing -- inert. The manipulation is therefore
    the SUPERVISION SIGNAL alone, which is exactly the hypothesis under test.

    The head's ONLINE loss weight (LatentStackConfig.resource_field_weight) is set for
    completeness but is INERT in this family -- nothing calls compute_resource_field_loss.
    P0a is what trains it; see the module docstring's seam section.
    """
    kwargs = x724._base_config_kwargs(env)
    kwargs.update(x724._all_on_extra_kwargs())
    kwargs["use_resource_field_head"] = True
    kwargs["resource_field_dim"] = RESOURCE_FIELD_DIM
    kwargs["resource_field_weight"] = P0A_FIELD_WEIGHT
    cfg = x724.REEConfig.from_dims(**kwargs)
    return x724.REEAgent(cfg)


# --------------------------------------------------------------------------------------
# PRE-REGISTERED PRECONDITIONS
# --------------------------------------------------------------------------------------
PRECONDITION_SPECS = [
    PreconditionSpec(
        name="zworld_encoder_trained_in_p0",
        description=(
            "The P0a SD-070 recipe must actually move split_encoder.world_encoder. Both arms "
            "run the same recipe, so this speaks for both."),
        control="latent_stack weight delta over the warmup, vs the family's guard floor",
        threshold=float(ZWORLD_DELTA_FLOOR),
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="field_leg_ran",
        description=(
            "The ON arm's directional P0a leg must have ACTUALLY run (weight > 0 AND the head "
            "present AND world_obs wide enough), not merely been configured. Measures the "
            "trainer's own used_resource_field_head, so a regression in the P0a seam reads as "
            "substrate-not-ready rather than as a scientific null."),
        control="ZWorldP0Trainer.used_resource_field_head on the ON arm's own warmup",
        threshold=0.5,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["field_head_on"]),
        applies_note=(
            "field_head_off runs the same architecture with P0a weight 0.0, so its leg not "
            "running is what that arm IS, not a readiness failure"),
    ),
    PreconditionSpec(
        name="field_head_decodes_on_arm",
        description=(
            "The ON arm's trained head must beat a constant-mean predictor on P0a's OWN "
            "held-out field split. `field_leg_ran` certifies only that the leg was configured "
            "and stepped; this certifies it LEARNED something. Without it a head that ran and "
            "learned nothing reads as a scientific null about SD-018 rather than as a failed "
            "operating point for the P0a recipe."),
        control="ZWorldP0Trainer resource_field_holdout.r2 on the ON arm's own warmup",
        threshold=0.0,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["field_head_on"]),
        applies_note=(
            "field_head_off trains no directional leg (P0a weight 0.0), so it has no "
            "held-out field r2 to certify"),
    ),
    PreconditionSpec(
        name="zworld_not_collapsed",
        description=(
            "z_world must retain more than one effective dimension. The amend's own ML note "
            "names 25-dim-target-dominates-P0 collapse as its hazard, and the substrate has "
            "measured PR 9.21 -> 1.06 under a related recipe. A collapsed arm is an operating- "
            "point failure, NOT evidence the directional head does not help."),
        control="participation ratio of frozen z_world over the decode-probe samples",
        threshold=float(PARTICIPATION_RATIO_FLOOR),
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="d3_local_view_greedy_clears_floor",
        description=(
            "The env must be floor-achievable FROM THE SAME 5x5 resource field the amend "
            "supervises -- closes the V3-EXQ-732a privileged-oracle confound, so an arm that "
            "fails is genuinely under-powered rather than obs-starved."),
        control="local_view_greedy worst-seed mean resources/ep vs the 1.0 competence floor",
        threshold=float(COMPETENCE_RESOURCE_FLOOR),
        direction="lower",
        kind="readiness",
    ),
]


def _worst_cell(rows: List[Dict[str, Any]], key: str, mode: str = "min"
                ) -> Tuple[float, str]:
    """Worst cell for a quantifier-shaped precondition, plus its id.

    Reports the EXTREMUM, never the mean: the indexer recomputes `met` from the reported
    number, so an in-band mean would mask an out-of-band cell and recompute MET against our
    own met=False (the V3-EXQ-779b tonic_axis_live shape).
    """
    vals = [(float(r[key]), str(r.get("cell_id", r.get("seed", "?")))) for r in rows
            if r.get(key) is not None]
    if not vals:
        return 0.0, "(none)"
    return min(vals) if mode == "min" else max(vals)


# --------------------------------------------------------------------------------------
# Fishtank episode log
# --------------------------------------------------------------------------------------
def _log_pass(agent, policy, env, arm_id: str, seed: int, n_episodes: int,
              steps: int) -> List[Dict[str, Any]]:
    """Instrumented OBSERVATIONAL pass -- not the scored data (see the docstring).

    Re-runs the already-trained policy on a fresh env purely to record traces in the schema
    REE_assembly/fishtank_viz.html renders.
    """
    episodes: List[Dict[str, Any]] = []
    for ep in range(int(n_episodes)):
        _flat, obs = env.reset()
        policy.reset(env)
        ep_steps: List[Dict[str, Any]] = []
        initial_resources = [list(r) for r in env.resources]
        initial_hazards = [list(h) for h in env.hazards]
        done_cause = "step_limit"
        for t in range(int(steps)):
            field = _localfield_vector(obs).reshape(-1)
            true_argmax = int(torch.argmax(field).item())
            pred_argmax: Optional[int] = None
            pred_argmax_enc: Optional[int] = None
            z_norm = 0.0
            with torch.no_grad():
                z = x737._agent_zworld(agent, obs)
                z_norm = float(z.norm().item())
                # red-team F7: read the head on BOTH paths. `pred_argmax` is the head applied
                # to the SENSE-time z the reader actually consumes; `pred_argmax_enc` applies
                # it to the ENCODER-path z that P0a trained. When the head is right on the
                # encoder path and wrong at sense time, the top-down term is the culprit and
                # the null routes to shape (b) -- a distinction the sense-time trace alone
                # cannot make. Read on BOTH arms: the OFF arm's head is untrained, so its
                # traces are the within-run chance baseline for these two series.
                head = getattr(
                    agent.latent_stack.split_encoder, "resource_field_head", None)
                if head is not None:
                    pred_argmax = int(torch.argmax(head(z).reshape(-1)).item())
                    z_enc = _encoder_path_zworld(agent, obs)
                    if z_enc is not None:
                        pred_argmax_enc = int(
                            torch.argmax(head(z_enc).reshape(-1)).item())
            action = policy.act(env, obs)
            _f, harm_signal, done, info, obs = env.step(action)
            if not isinstance(info, dict):
                info = {}
            ep_steps.append({
                "t": int(t),
                "pos": [int(env.agent_x), int(env.agent_y)],
                "action": int(action),
                "harm_signal": float(harm_signal),
                "transition_type": str(info.get("transition_type", "none")),
                "health": float(getattr(env, "agent_health", 1.0)),
                "energy": float(getattr(env, "agent_energy", 1.0)),
                "z_world_norm": z_norm,
                # SD-018's own subject matter, made watchable:
                "resource_field_true_argmax": true_argmax,
                "resource_field_pred_argmax": pred_argmax,
                "resource_field_pred_argmax_encoder_path": pred_argmax_enc,
                "resource_field_max": float(field.max()),
                "hazards": [list(h) for h in env.hazards],
                "resources": [list(r) for r in env.resources],
            })
            if done:
                done_cause = ("health_depleted"
                              if float(getattr(env, "agent_health", 1.0)) <= 0.0
                              else "step_limit")
                break
        episodes.append({
            "ep": int(ep),
            "arm": arm_id,
            "seed": int(seed),
            "initial_resources": initial_resources,
            "initial_hazards": initial_hazards,
            "steps": ep_steps,
            "realized_steps": len(ep_steps),
            "done_cause": done_cause,
        })
    return episodes


# --------------------------------------------------------------------------------------
def _run_seed(seed: int, zworld_p0: int, p0: int, p1: int, ppo_eps: int, eval_eps: int,
              steps: int, rollout: int, log_eps: int, want_log: bool, dry_run: bool
              ) -> Dict[str, Any]:
    """One seed: warm ONE stack PER ARM (the manipulation is in P0a), then train + eval."""
    env_kwargs = x734._env_kwargs_for_rung(RUNG)
    total_denom = p0 + p1
    arm_rows: Dict[str, Dict[str, Any]] = {}
    log_episodes: List[Dict[str, Any]] = []

    for arm_id in ARM_IDS:
        field_on = (arm_id == ARM_ON)
        print(f"Seed {seed} Condition {RUNG_ID}:{arm_id}:warmup", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        warm_env = x734._make_env(seed, env_kwargs)
        agent = _make_agent(warm_env)
        before = latent_stack_snapshot(agent)
        stats = x734._train_all_on_agent(
            agent, warm_env, seed=seed, p0_episodes=p0, p1_episodes=p1,
            steps_per_episode=steps, rung_id=RUNG_ID, total_denominator=total_denom,
            zworld_p0_episodes=zworld_p0,
            zworld_p0_env=(x734._make_env(seed, env_kwargs) if zworld_p0 > 0 else None),
            zworld_p0_dry_run=dry_run,
            zworld_p0_resource_field_weight=(P0A_FIELD_WEIGHT if field_on else 0.0),
        )
        guard = latent_stack_weight_delta(agent, before)
        p0a = (stats or {}).get("zworld_p0", {}) or {}

        # --- frozen-encoder mechanism probe (identical protocol on both arms) ------------
        z_s, f_s, ze_s = _collect_zworld_field_pairs(
            agent, x734._make_env(seed, env_kwargs), seed,
            (60 if dry_run else DECODE_PROBE_STEPS))
        decode = _field_decode_r2(z_s, f_s)
        pr = _participation_ratio(z_s)
        # red-team F6: the gate below reads SENSE-time PR because that is what the reader
        # consumes, but P0a trains the encoder path -- so an encoder-path collapse and a
        # top-down-driven one are different diagnoses. Reported, not gated.
        pr_encoder = _participation_ratio(ze_s)
        decode_encoder = _field_decode_r2(ze_s, f_s) if ze_s.shape[0] == f_s.shape[0] else {}

        # --- the reader: PPO on z_world ALONE, identical across arms ---------------------
        print(f"Seed {seed} Condition {RUNG_ID}:{arm_id}", flush=True)
        torch.manual_seed(seed + 1000)
        np.random.seed(seed + 1000)
        _flat, probe_obs = x734._make_env(seed, env_kwargs).reset()
        z_dim = int(x737._agent_zworld(agent, probe_obs).shape[-1])
        action_dim = int(warm_env.action_dim)
        net = x734.PPOPolicyNet(in_dim=z_dim, action_dim=action_dim).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=x734.PPO_LR)
        train_decomp = x808._train_ppo_decomposed(
            x734._make_env(seed, env_kwargs), net, opt,
            state_fn=(lambda od: x737._agent_zworld(agent, od)),
            on_reset=agent.reset,
            n_episodes=ppo_eps, rollout_episodes=rollout, steps_per_episode=steps,
            level_id=LEVEL_ID, w_consume=W_CONSUME, w_survival=W_SURVIVAL, seed=seed,
            total_denominator=ppo_eps,
        )
        eval_policy = x737.LatentPPOEvalPolicy(net, agent)
        eval_row = evaluate_seed(
            eval_policy, x734._make_env(seed, env_kwargs), eval_eps, steps)

        arm_rows[arm_id] = {
            "cell_id": f"{arm_id}|seed{seed}",
            "arm_id": arm_id,
            "seed": int(seed),
            "obs_dim": int(z_dim),
            "field_head_configured": bool(field_on),
            "foraging_competence": float(eval_row["foraging_competence"]),
            "competence_supra_floor": bool(eval_row["competence_supra_floor"]),
            "survival_horizon": float(eval_row["survival_horizon"]),
            "death_rate": float(eval_row["death_rate"]),
            "mean_episode_reward": float(eval_row["mean_episode_reward"]),
            "per_episode_resources": list(eval_row["per_episode_resources"]),
            "zworld_weight_delta": guard,
            "zworld_participation_ratio": float(pr),
            "zworld_encoder_path_participation_ratio": float(pr_encoder),
            "field_decode_holdout": decode,
            "field_decode_holdout_encoder_path": decode_encoder,
            "p0a": {
                "ran": bool(p0a.get("p0a_ran")),
                "resource_field_weight": p0a.get("p0a_resource_field_weight"),
                "used_resource_field_head": bool(p0a.get("p0a_used_resource_field_head")),
                "used_proximity_head": p0a.get("p0a_used_proximity_head"),
                "resource_field_holdout": p0a.get("p0a_resource_field_holdout"),
                "holdout_mean_lift": p0a.get("p0a_holdout_mean_lift"),
            },
            "ppo_train_decomposition": train_decomp,
        }
        print(f"verdict: {'PASS' if eval_row['competence_supra_floor'] else 'FAIL'}",
              flush=True)

        _ZG.observe(agent)   # AFTER stepping -- reads the counters at call time

        if want_log:
            log_episodes.extend(_log_pass(
                agent, x737.LatentPPOEvalPolicy(net, agent),
                x734._make_env(seed + 7777, env_kwargs), arm_id, seed, log_eps, steps))

    # --- anchors -----------------------------------------------------------------------
    anchor_rows: List[Dict[str, Any]] = []
    anchors = {"random_walk": RandomPolicy(seed),
               "local_view_greedy": LocalViewGreedyPolicy(seed)}
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

    return {"seed": int(seed), "arms": arm_rows, "anchors": anchor_rows,
            "log_episodes": log_episodes,
            "env_config": x734._env_kwargs_for_rung(RUNG)}


# --------------------------------------------------------------------------------------
def run_experiment(seeds: List[int], dry_run: bool = False) -> Dict[str, Any]:
    zworld_p0 = DRY_RUN_ZWORLD_P0 if dry_run else ZWORLD_P0_EPISODES
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1 = DRY_RUN_P1 if dry_run else P1_REINFORCE_EPISODES
    ppo_eps = DRY_RUN_PPO if dry_run else P1_PPO_EPISODES
    eval_eps = DRY_RUN_EVAL if dry_run else EVAL_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    rollout = DRY_RUN_ROLLOUT if dry_run else PPO_ROLLOUT_EPISODES
    log_eps = DRY_RUN_LOG_EPISODES if dry_run else LOG_EPISODES

    # Design-time refusal BEFORE compute: a precondition no arm could satisfy is a design
    # bug, and the remedy is scoping the gate, never lowering a pre-registered threshold.
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, _arm_contexts())

    log_seeds = {seeds[0]} if seeds else set()
    seed_results = [
        _run_seed(s, zworld_p0, p0, p1, ppo_eps, eval_eps, steps, rollout,
                  log_eps, (s in log_seeds), dry_run)
        for s in seeds
    ]

    all_arm_rows = [r["arms"][aid] for r in seed_results for aid in ARM_IDS]
    anchor_rows = [a for r in seed_results for a in r["anchors"]]

    per_arm: Dict[str, Dict[str, Any]] = {}
    for aid in ARM_IDS:
        rows = [r["arms"][aid] for r in seed_results]
        n_supra = int(sum(1 for x in rows if x["competence_supra_floor"]))
        r2s = [x["field_decode_holdout"].get("r2") for x in rows
               if x["field_decode_holdout"].get("r2") is not None]
        per_arm[aid] = {
            "arm_id": aid,
            "n_seeds": len(rows),
            "n_seeds_supra_floor": n_supra,
            "majority_supra_floor": bool(n_supra >= (len(rows) + 1) // 2) if rows else False,
            "mean_foraging_competence": _mean([x["foraging_competence"] for x in rows]),
            "per_seed_foraging_competence": [x["foraging_competence"] for x in rows],
            "mean_field_decode_r2": (_mean(r2s) if r2s else None),
            "per_seed_field_decode_r2": [x["field_decode_holdout"].get("r2") for x in rows],
            "mean_participation_ratio": _mean([x["zworld_participation_ratio"] for x in rows]),
            "per_seed_participation_ratio": [x["zworld_participation_ratio"] for x in rows],
            # red-team F6: encoder-path PR alongside the sense-time PR the gate reads.
            "mean_encoder_path_participation_ratio": _mean(
                [x["zworld_encoder_path_participation_ratio"] for x in rows]),
            "per_seed_encoder_path_participation_ratio": [
                x["zworld_encoder_path_participation_ratio"] for x in rows],
            "mean_encoder_path_field_decode_r2": _mean(
                [x["field_decode_holdout_encoder_path"].get("r2") for x in rows
                 if (x["field_decode_holdout_encoder_path"] or {}).get("r2") is not None]),
            "field_leg_ran_all_seeds": bool(all(x["p0a"]["used_resource_field_head"]
                                                for x in rows)),
            # red-team F5 (code half): worst-seed held-out field r2 from P0a's OWN split.
            # None (head absent, or base MSE 0 so r2 is undefined) reads as -1.0 = fails the
            # >0 floor, which is the correct reading for "cannot certify it learned".
            "min_p0a_field_holdout_r2": min(
                [_holdout_r2(x) for x in rows] or [-1.0]),
            "per_seed_p0a_field_holdout_r2": [_holdout_r2(x) for x in rows],
        }

    lvg_rows = [a for a in anchor_rows if a["anchor_id"] == "local_view_greedy"]
    lvg_measured, lvg_cell = _worst_cell(lvg_rows, "foraging_competence", "min")

    # --- per-arm gates (never AND'd whole-run: one arm's failure must not vacate the other)
    arm_gates = []
    for aid in ARM_IDS:
        rows = [r["arms"][aid] for r in seed_results]
        delta_measured, _dc = _worst_cell(
            [{"cell_id": x["cell_id"],
              "d": float((x["zworld_weight_delta"] or {}).get(
                  "world_encoder_max_abs_delta", 0.0) or 0.0)}
             for x in rows], "d", "min")
        pr_measured, _pc = _worst_cell(rows, "zworld_participation_ratio", "min")
        measured = {
            "zworld_encoder_trained_in_p0": delta_measured,
            "zworld_not_collapsed": pr_measured,
            "d3_local_view_greedy_clears_floor": lvg_measured,
        }
        ctx = {"id": aid, "field_head_on": (aid == ARM_ON)}
        if aid == ARM_ON:
            measured["field_leg_ran"] = 1.0 if per_arm[aid]["field_leg_ran_all_seeds"] else 0.0
            measured["field_head_decodes_on_arm"] = float(
                per_arm[aid]["min_p0a_field_holdout_r2"])
        arm_gates.append(evaluate_arm_gate(aid, ctx, PRECONDITION_SPECS, measured))
    gate = aggregate_arm_gates(arm_gates)

    # --- pre-registered criteria --------------------------------------------------------
    on_clears = bool(per_arm[ARM_ON]["majority_supra_floor"])
    off_clears = bool(per_arm[ARM_OFF]["majority_supra_floor"])
    r_on = per_arm[ARM_ON]["mean_field_decode_r2"]
    r_off = per_arm[ARM_OFF]["mean_field_decode_r2"]
    decode_lift = ((float(r_on) - float(r_off))
                   if (r_on is not None and r_off is not None) else None)
    # red-team F4: the MARGIN alone is satisfiable between two useless decodes -- r2 -0.90
    # -> -0.85 is a 0.05 "lift" of nothing, since a negative r2 means the decode is worse
    # than predicting the constant mean. Require the ON arm to beat that constant-mean
    # predictor in ABSOLUTE terms too, so C_decode_lift can only fire on a decode that
    # actually carries directional content.
    decode_lifted = bool(decode_lift is not None
                         and decode_lift >= DECODE_R2_LIFT_MIN
                         and r_on is not None and float(r_on) > 0.0)

    criteria = [
        {"name": "C_on_clears_floor", "load_bearing": True, "passed": on_clears,
         "description": ("field_head_on clears the 1.0 competence floor on a strict majority "
                         "of seeds, reading z_world alone")},
        {"name": "C_off_subfloor_replication", "load_bearing": False, "passed": (not off_clears),
         "description": ("field_head_off does NOT clear the floor -- replicates the 948/813 "
                         "latent anchor. If this fails the contrast is uninterpretable, which "
                         "is why it is a criterion and not an assumption")},
        {"name": "C_decode_lift", "load_bearing": False, "passed": decode_lifted,
         "description": ("held-out linear decode of resource_field_view from frozen z_world "
                         "lifts by >= %.2f r2 on the ON arm AND clears r2 > 0 absolutely -- "
                         "the mechanism check that makes a competence lift attributable to "
                         "directional content. The absolute floor is load-bearing: a margin "
                         "between two sub-mean decodes is a lift of nothing (red-team F4)"
                         % DECODE_R2_LIFT_MIN)},
    ]
    combination_rule = (
        "PASS iff C_on_clears_floor (load-bearing) AND C_off_subfloor_replication. "
        "C_decode_lift is REPORTED, not gating: it attributes a lift rather than producing "
        "one, and a lift with a flat decode is a real and interesting result (it would mean "
        "the auxiliary loss helped by some route other than exposing the gradient) that must "
        "not be suppressed into a FAIL."
    )
    overall_pass = bool(on_clears and (not off_clears))

    if not gate["non_degenerate"]:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    elif off_clears:
        outcome = "FAIL"
        label = "anchor_replication_failed_off_arm_supra_floor"
    elif on_clears:
        outcome = "PASS"
        label = "directional_field_head_lifts_zworld_only_competence"
    elif decode_lifted:
        outcome = "FAIL"
        label = "directional_content_present_but_does_not_convert"
    elif decode_lift is None:
        # red-team F3: an UNMEASURABLE decode is not the same claim as a measured flat one.
        # Collapsing the two let a probe failure be reported as "the head did not change
        # z_world" -- an assertion about the substrate that the run has no evidence for.
        outcome = "FAIL"
        label = "decode_unmeasurable_direction_unattributable"
    else:
        outcome = "FAIL"
        label = "directional_head_did_not_change_zworld_at_this_operating_point"

    # Per-claim direction. INV-088 is the claim under test (z_world differentiation bounds
    # its downstream readers); MECH-457 is READ-ACROSS ONLY -- this run holds the learner
    # fixed and varies the representation, so it cannot test MECH-457's own mechanism (a
    # dedicated RPE actor-critic substrate). A lift here weakens the necessity of that
    # mechanism for THIS competence floor; a null says nothing about it either way.
    on_arm_red = bool(ARM_ON in (gate.get("red_arms") or []))
    if outcome == "PASS":
        dir_inv088, dir_mech457 = "supports", "weakens"
    elif on_arm_red:
        # Extension of red-team F3 to the PARTIAL-non-vacuity case: under partial
        # non-vacuity the run is still scored (a red arm must not vacate a green one), so
        # control reaches the direction block with the ON arm UNSCORED. Its criteria are
        # already flagged via arm_criteria_non_degenerate, but the claim direction was
        # computed from `label` alone and could still emit "weakens" off an arm whose own
        # readiness gate failed. An unscored arm supports no direction, either way.
        dir_inv088, dir_mech457 = "unknown", "unknown"
    elif label == "substrate_not_ready_requeue" or off_clears:
        dir_inv088, dir_mech457 = "unknown", "unknown"
    elif label == "directional_content_present_but_does_not_convert":
        # The ONLY null branch that bears on INV-088 itself: the directional content is
        # demonstrably IN z_world (the decode lifted and cleared its absolute floor) and a
        # z_world-only reader still cannot convert it into competence.
        dir_inv088, dir_mech457 = "weakens", "unknown"
    else:
        # red-team F3: the remaining branches -- the P0a recipe never moved z_world's field
        # content at this operating point, or the decode was unmeasurable -- are statements
        # about THIS recipe and instrumentation, NOT about the hypothesis. INV-088 is
        # UNTESTED on them, not weakened. The module docstring's null-reading section
        # already says so; this is the code half of that (it previously read "weakens").
        dir_inv088, dir_mech457 = "unknown", "unknown"

    # Keyed arm_id -> the criteria that arm owns: a criterion belonging to a RED arm reads
    # non_degenerate=False, so one arm's readiness failure marks only its own criteria and
    # never vacates the other arm's (the V3-EXQ-785 whole-run AND defect).
    non_degenerate_flags = arm_criteria_non_degenerate(
        {ARM_ON: ["C_on_clears_floor", "C_decode_lift"],
         ARM_OFF: ["C_off_subfloor_replication"]},
        gate,
    )

    metrics = {
        "competence_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "p0a_field_weight": float(P0A_FIELD_WEIGHT),
        "off_mean_foraging_competence": per_arm[ARM_OFF]["mean_foraging_competence"],
        "on_mean_foraging_competence": per_arm[ARM_ON]["mean_foraging_competence"],
        "off_n_seeds_supra_floor": per_arm[ARM_OFF]["n_seeds_supra_floor"],
        "on_n_seeds_supra_floor": per_arm[ARM_ON]["n_seeds_supra_floor"],
        "off_mean_field_decode_r2": per_arm[ARM_OFF]["mean_field_decode_r2"],
        "on_mean_field_decode_r2": per_arm[ARM_ON]["mean_field_decode_r2"],
        "field_decode_r2_lift": decode_lift,
        "off_mean_participation_ratio": per_arm[ARM_OFF]["mean_participation_ratio"],
        "on_mean_participation_ratio": per_arm[ARM_ON]["mean_participation_ratio"],
        "local_view_greedy_worst_seed_competence": lvg_measured,
        "n_seeds": len(seed_results),
    }

    interpretation = {
        "label": label,
        "combination_rule": combination_rule,
        "preconditions": gate["adjudication_preconditions"],
        "per_arm_gate": gate,
        "criteria_non_degenerate": non_degenerate_flags,
        "criteria": criteria,
        "null_reading": {
            "directional_content_present_but_does_not_convert": (
                "decode lifted, competence did not -- displaces the observation-interface leg "
                "toward the CONSUMER side; shape (b) is the next build"),
            "directional_head_did_not_change_zworld_at_this_operating_point": (
                "neither lifted -- a statement about the P0a recipe at "
                "resource_field_weight=%.2f, NOT about the hypothesis. Re-run at another "
                "weight before concluding" % P0A_FIELD_WEIGHT),
        },
    }

    summary_markdown = f"""# {QUEUE_ID} -- SD-018 directional resource-field head validation

Outcome: **{outcome}** ({label})

| arm | mean res/ep | seeds supra 1.0 floor | held-out field decode r2 | z_world PR |
|---|---|---|---|---|
| {ARM_OFF} | {per_arm[ARM_OFF]['mean_foraging_competence']:.4f} | {per_arm[ARM_OFF]['n_seeds_supra_floor']}/{per_arm[ARM_OFF]['n_seeds']} | {per_arm[ARM_OFF]['mean_field_decode_r2']} | {per_arm[ARM_OFF]['mean_participation_ratio']:.3f} |
| {ARM_ON} | {per_arm[ARM_ON]['mean_foraging_competence']:.4f} | {per_arm[ARM_ON]['n_seeds_supra_floor']}/{per_arm[ARM_ON]['n_seeds']} | {per_arm[ARM_ON]['mean_field_decode_r2']} | {per_arm[ARM_ON]['mean_participation_ratio']:.3f} |

Anchor: local_view_greedy worst seed = {lvg_measured:.4f} against the 1.0 floor (cell {lvg_cell}).

{combination_rule}

The reader is PPO on z_world alone (32 dims) in BOTH arms -- no side-channel, unlike 948.
The manipulation is upstream, in the P0a objective, at resource_field_weight={P0A_FIELD_WEIGHT}.

A fishtank episode log companion is written alongside this manifest and is rendered by
REE_assembly's /fishtank_viz.html. It is a SEPARATE observational pass, not the scored data:
per step it carries the agent's position, the true resource-gradient argmax, and (ON arm) the
argmax z_world's own head predicts -- so the amend's subject matter is directly watchable.
"""

    first_env = seed_results[0]["env_config"] if seed_results else {}
    episode_log = {
        "experiment_type": EXPERIMENT_TYPE,
        "phase": "sd018_directional_field_off_vs_on",
        "toroidal": bool(first_env.get("toroidal", False)),
        "env_config": first_env,
        "seeds": [{"seed": r["seed"], "episodes": r["log_episodes"]}
                  for r in seed_results if r["log_episodes"]],
    }

    return {
        "status": outcome,
        "outcome": outcome,
        "overall_pass": overall_pass,
        "metrics": metrics,
        "interpretation": interpretation,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": ("supports" if outcome == "PASS" else "mixed"),
        "evidence_direction_per_claim": {"INV-088": dir_inv088, "MECH-457": dir_mech457},
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "experiment_type": EXPERIMENT_TYPE,
        "sleep_driver_pattern": "none",
        "arm_results": all_arm_rows,
        "per_arm": per_arm,
        "anchor_results": anchor_rows,
        "rung_id": RUNG_ID,
        "level_id": LEVEL_ID,
        "episode_log": episode_log,
        "supersedes": None,
    }


if __name__ == "__main__":
    import argparse
    import json
    import time
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    seeds = args.seeds if args.seeds else (DRY_RUN_SEEDS if args.dry_run else SEEDS)
    result = run_experiment(seeds=seeds, dry_run=args.dry_run)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["timestamp_utc"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = ARCHITECTURE_EPOCH
    result["queue_id"] = QUEUE_ID

    episode_log = result.pop("episode_log", None)

    full_config = {
        "rung": RUNG,
        "level_id": LEVEL_ID,
        "w_consume": W_CONSUME,
        "w_survival": W_SURVIVAL,
        "p0a_field_weight": P0A_FIELD_WEIGHT,
        "zworld_p0_episodes": (DRY_RUN_ZWORLD_P0 if args.dry_run else ZWORLD_P0_EPISODES),
        "p0_warmup_episodes": (DRY_RUN_P0 if args.dry_run else P0_WARMUP_EPISODES),
        "p1_ppo_episodes": (DRY_RUN_PPO if args.dry_run else P1_PPO_EPISODES),
        "eval_episodes": (DRY_RUN_EVAL if args.dry_run else EVAL_EPISODES),
        "steps_per_episode": (DRY_RUN_STEPS if args.dry_run else STEPS_PER_EPISODE),
        "participation_ratio_floor": PARTICIPATION_RATIO_FLOOR,
        "decode_r2_lift_min": DECODE_R2_LIFT_MIN,
        "env_kwargs": x734._env_kwargs_for_rung(RUNG),
        "dry_run": bool(args.dry_run),
    }
    # write_flat_manifest stamps the always-core (recording_schema / substrate_hash /
    # machine / machine_class / elapsed_seconds / config / seeds) itself; z_goal_stream_stats
    # carries the liveness block the per-cell agents would otherwise not reach.
    out_path = write_flat_manifest(
        result, None, dry_run=args.dry_run,
        config=full_config, seeds=seeds, script_path=Path(__file__),
        started_at=t0, z_goal_stream_stats=_ZG.stats(),
    )

    if episode_log is not None and not args.dry_run:
        episode_log["run_id"] = result["run_id"]
        log_path = Path(out_path).parent / f"{EXPERIMENT_TYPE}_{ts}_episode_log.json"
        log_path.write_text(json.dumps(episode_log, indent=2) + "\n", encoding="utf-8")
        print(f"[fishtank] episode log -> {log_path}", flush=True)

    print(f"outcome: {result['outcome']} ({result['interpretation']['label']})", flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
