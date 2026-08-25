"""V3-EXQ-948 -- Does a foraging-adequate re-representation lift D3 competence?
(GOV-FANOUT-1 representation-axis probe for H-observation-interface; conversion_ceiling_root
DIAGNOSTIC.)

POSITIVE-DEMONSTRATION probe for the leg `H-observation-interface` (axis `representation`)
of the frozen-ledger question `conversion_ceiling_root`. That leg was pre-registered and
ELEVATED on 2026-07-24 by the confirmed autopsy `failure_autopsy_backlog_2026-07-24` off the
V3-EXQ-813 override, and its registry entry states its own gate verbatim:

    "Needs a positive demonstration (a foraging-adequate re-representation lifts
     competence) to confirm."

As of this script's authoring that gate had no adjudicating run: only V3-EXQ-808 (reward
axis) and V3-EXQ-813 (policy axis) carry the `conversion_ceiling_root` tag, both dated
2026-07-24. This script is that positive demonstration.

=== WHAT 813 ESTABLISHED, AND THE ONE THING IT COULD NOT ===

Under the IDENTICAL W3_survival_zeroed (consumption-only) objective at hazard-free D3, a PPO
actor on RAW OBSERVATIONS cleared the 1.0 competence floor (9.033 res/ep) while a PPO actor
on the frozen REE z_world latent did NOT (0.5 res/ep). Same learner, same reward, same env,
same seeds -- so neither the learner (H-policy-learning, ELIMINATED by that contrast) nor the
reward (H-objective-misspecification, DISPLACED by V3-EXQ-808's inert 25:1 reweighting sweep)
can account for the gap. What remains is what the latent EXPOSES to a downstream reader.

But 813's contrast is a NEGATIVE result about the latent. It shows the latent is not
sufficient; it does NOT show that the missing content is recoverable, nor WHICH content is
missing, nor whether the latent's presence is merely uninformative or actively obstructive.
A leg confirmed only by an absence is not confirmed. This script supplies the presence.

=== THE RE-REPRESENTATION IS NOT INVENTED -- IT IS ALREADY PROVEN ADEQUATE ===

The added channel is `obs_dict["resource_field_view"]`: the agent-centred 5x5 resource
gradient, 25 dims (measured, not assumed -- see the module's own smoke output). Three facts
make it the principled choice rather than an arbitrary augmentation:

  1. It is PROVEN SUFFICIENT FOR COMPETENCE UNDER THE LEARNER'S OWN OBSERVABILITY. The
     `local_view_greedy` anchor reads ONLY this channel and scores ~45.75 res/ep against a
     1.0 floor. That anchor exists precisely to close the V3-EXQ-732a confound (a privileged
     global oracle is an unfair yardstick); it certifies the floor is reachable FROM THIS
     VIEW. So an arm given this channel and still failing is genuinely under-powered or
     obstructed, never obs-starved.
  2. It is ALREADY INSIDE THE 9.033 ARM. `resource_field_view` is a subset of `world_state`
     (`causal_grid_world.py`: world_state[225:250]), and `_RAW_OBS_CANDIDATE_KEYS` includes
     `world_state`. So the arm that clears the floor already sees this exact structure.
  3. It is ALREADY INSIDE THE ENCODER'S INPUT. z_world is the REE encoder's compression of
     that same world_state. So the leg becomes sharp and falsifiable: does z_world fail to
     expose resource-gradient structure that is demonstrably present in its own input?

Dimensionality is near-matched by construction (z_world 32 dims vs resource_field_view 25),
so a lift on the field arms cannot be attributed to simply handing the learner a much wider
input.

=== ARMS (learner and objective held FIXED across all four) ===

  ppo_ree_latent             z_world (32)                  FLOOR anchor; replicates 813's 0.5
  ppo_raw_obs                body+world+harm (373)         CEILING anchor; replicates 813's 9.033
  ppo_latent_plus_localfield z_world + resource_field (57)  THE POSITIVE DEMONSTRATION
  ppo_localfield_only        resource_field (25)            INTERFERENCE DISCRIMINATOR

The decisive contrast is `ppo_latent_plus_localfield` MINUS `ppo_ree_latent`: identical REE
stack, identical warmup, identical objective, identical seeds, differing ONLY in whether the
25-dim resource gradient is concatenated onto the observation vector.

`ppo_localfield_only` exists to close a verdict ALIASING gap the three-arm design leaves
open. If arm 3 fails, two very different causes produce the same reading: (a) the field is
not learnable at this budget by this learner, or (b) the latent ACTIVELY OBSTRUCTS -- its
presence degrades a policy that the field alone would support. Arm 4 separates them: it
gives the field WITHOUT the latent. Field-only clears while latent+field does not => (b),
active interference, which is a sharper and different claim from "the latent under-exposes".

=== HYPOTHESES UNDER TEST ===

H-observation-interface (this probe's leg, axis `representation`):
    The REE latent does not expose foraging-adequate structure to a downstream reader.
    If true, restoring that structure alongside the latent should lift competence over the
    latent-alone arm, holding learner and objective fixed.

H-substrate-ceiling (co-alive, axis `substrate`; NOT adjudicated here):
    The substrate caps competence independent of driver. This probe can SHARPEN it (via the
    interference reading) but cannot eliminate it: a null on both field arms is consistent
    with a substrate ceiling and this design does not discriminate that.

DECLARED NULL (pre-registered): `ppo_latent_plus_localfield` does NOT clear the 1.0
competence floor on a strict majority of seeds AND `ppo_localfield_only` does not either.
Reading: restoring provably-adequate foraging structure to the observation does not lift
competence, so the ceiling is NOT the observation interface -- H-observation-interface is
REFUTED as a passive-under-exposure account and live discrimination passes back to
H-substrate-ceiling / H-f-dominance. This does NOT eliminate H-substrate-ceiling (nothing
here tests it positively) and does NOT re-open H-policy-learning (813's raw-obs arm already
eliminated it, and this run re-measures that arm as a standing replication check).

ALTERNATIVE OUTCOMES:
  - arm 3 clears  -> H-observation-interface CONFIRMED, and NAMED: the missing content is the
    resource gradient. Actionable at the substrate (the encoder must preserve it).
  - arm 3 flat, arm 4 clears -> the latent ACTIVELY OBSTRUCTS. Re-route: this is not passive
    under-exposure, and the substrate work is interference removal, not content addition.
  - arm 2 sub-floor, or arm 1 supra-floor -> the 813 anchor pair did not replicate; this run
    is NOT comparable to 813 and every reading above is withheld pending autopsy.

=== WHY 813's `consumption_share` READINESS GATE IS NOT REPRODUCED HERE ===

813 declared `consumption_share_dominates_under_correction` (worst-cell train-phase
consumption share vs a 0.90 floor). It FAILED on both arms (0.5717 latent / 0.8097 raw obs)
and forced a whole-run `substrate_not_ready_requeue` self-route that the confirmed autopsy
had to OVERRIDE by user judgment to use the run at all.

That gate is CIRCULAR, and the manifest proves it. At w_survival=0.0 every survival-linked
family is deleted BY CONSTRUCTION -- `survival_linked_share` reads exactly 0.0 on all six of
813's cells -- so the entire non-consumption residual is `harm`, which `_weighted` never
reweights. Harm accrues per-tick from the environment while consumption accrues only when the
arm actually forages. So the share is LOW PRECISELY WHEN THE ARM FORAGES LITTLE: the latent
arm read 0.5717 at 0.5 res/ep, the competent raw-obs arm 0.8097 train / 0.997 eval at 9.033
res/ep. The gate penalised each arm for the very failure under test, and would do so again
here -- more severely, since this design deliberately includes an arm expected to sit at the
floor.

Replaced by two NON-CIRCULAR checks that assert what the gate was actually for:

  survival_linked_share_zeroed      STRUCTURAL. Worst-cell survival_linked_share across all
                                    arms must be <= 1e-9 (direction `upper`). This is the
                                    real "did the objective correction land" question and it
                                    cannot be moved by an arm's competence.
  harm_share_negligible_on_control  POSITIVE CONTROL. Eval-phase harm share measured on the
                                    `local_view_greedy` anchor -- a competent forager that is
                                    NOT under test -- so env-level harm accrual is certified
                                    without reading it off an arm whose competence is the DV.

=== DV-SYMMETRY INVARIANCE (per-arm declaration; Step 3.5 requirement) ===

DV: D3 `foraging_competence` -- mean resources consumed per eval episode, a count over the
eval trajectory. Its symmetry group is the set of transforms leaving that count fixed. Each
arm's manipulation changes the OBSERVATION VECTOR the PPO policy conditions on, hence the
action distribution, hence the trajectory, hence the count. No arm's manipulation is a
broadcast additive constant over candidate scores (there is no candidate-score layer here --
PPO emits logits over 5 primitive actions directly from the observation), none is a monotone
rescaling of a rank-valued DV, and none is a permutation of interchangeable units. So no
arm's delta is an arithmetic identity fixed before the run; every arm can, in principle,
move the DV. Arms 1 and 2 are exact replications of 813 cells that DID move it (0.5 vs
9.033), which is direct empirical confirmation that this DV is live under this design.

=== KNOWN SUBSTRATE LIMITATIONS UNDER WHICH THIS RUNS (Step 2.5c, released as degrading) ===

Two open `corrupting` substrate_queue entries overlap this driver's import closure, both
`implemented_pending_validation` with their repairs landed behind DEFAULT-OFF flags (ree-v3
`9bcde4c` salience_affinity_input_cap no-op-default; `692f852` default-off ContextMemory
'refractory' write-selection mode), so the default config path this driver uses still
executes the un-repaired code:

  mode-governance-engagement                        agent.py, utils/config.py, salience_coordinator.py
  contextmemory-write-path-addressing-degeneracy    predictors/e1_deep.py

Released to `degrading` treatment (user-adjudicated) because this design is INTERNALLY
CONTROLLED against both: arms 1 and 3 carry the identical REE stack and therefore identical
exposure to both defects, and they are the pair carrying the decisive contrast; arms 2 and 4
instantiate no REE agent at all. Neither defect can manufacture or mask the arm-3-minus-arm-1
lift. Recorded here and in the queue entry so any later autopsy sees the run happened under a
known limitation. (`MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION` is scoped OUT: its
`::function` entries are `agent.py::run_sws_schema_pass` and `mel_consumer::relative_novelty`,
and this driver runs no sleep.)

DIAGNOSTIC: promotes and demotes nothing; claim_ids=[]. Route to /failure-autopsy before any
governance use.
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
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_737_ree_latent_policy_head_competence_probe as x737  # noqa: E402
import experiments.v3_exq_808_return_decomposition_objective_misspecification as x808  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_948_observation_interface_re_representation_probe"
QUEUE_ID = "V3-EXQ-948"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# Four-arm grid (arm x seed) but every cell trains a PPO actor from scratch on a per-seed
# re-warmed REE stack -- nothing is a pure function of a shared baseline config a later run
# could reuse. Matches 737b / 808 / 813's own exemption for the same structural reason.
ARM_FINGERPRINT_EXEMPT = (
    "per-seed REE warmup + per-arm PPO training from scratch under a fixed reward level; "
    "no reusable baseline arm to bank"
)

DEVICE = torch.device("cpu")

# Same seeds as V3-EXQ-813 so arms 1 and 2 are exact per-seed replications of its anchor
# pair. This env is NOT a reef config (reef_enabled=False at this rung), so the seed-44
# reef instability rule does not apply -- and 813 ran 44 cleanly.
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
# 734 so it is byte-identical to the sibling family (737b, 808, 813) this probe reads against.
RUNG = x734.DIFFICULTY_RUNGS[-1]
RUNG_ID = RUNG["rung_id"]

# The fixed weighting -- imported verbatim from V3-EXQ-808, never redefined here, so this
# probe cannot drift from the family's definition of the corrected objective. Identical to
# the level V3-EXQ-813 trained under, which is what makes arms 1/2 true replications.
_W3 = next(w for w in x808.WEIGHTINGS if w["id"] == "W3_survival_zeroed")
LEVEL_ID = str(_W3["id"])
W_CONSUME = float(_W3["w_consume"])
W_SURVIVAL = float(_W3["w_survival"])

ARM_LATENT = "ppo_ree_latent"
ARM_RAW = "ppo_raw_obs"
ARM_LATENT_FIELD = "ppo_latent_plus_localfield"
ARM_FIELD = "ppo_localfield_only"
ARM_IDS = [ARM_LATENT, ARM_RAW, ARM_LATENT_FIELD, ARM_FIELD]

# The arms that read z_world, and therefore the ones the encoder guard is meaningful for.
LATENT_READING_ARMS = frozenset({ARM_LATENT, ARM_LATENT_FIELD})

ANCHOR_IDS = ["random_walk", "local_view_greedy", "greedy_oracle"]

# The re-representation channel. Agent-centred 5x5 resource gradient; a subset of
# world_state (world_state[225:250]) and the sole input of the local_view_greedy anchor.
LOCALFIELD_KEY = "resource_field_view"

# --------------------------------------------------------------------------------------
# PRE-REGISTERED THRESHOLDS (constants; never derived from this run's own statistics)
# --------------------------------------------------------------------------------------
ZWORLD_DELTA_FLOOR = x808.ZWORLD_DELTA_FLOOR         # 1e-6; same guard floor as the family
# Structural check that the objective correction landed. At w_survival=0.0 the survival-
# linked families are multiplied by zero, so this is 0.0 by construction; a non-zero reading
# means a leaked term, i.e. an implementation defect. Direction UPPER: measured must be <=.
SURVIVAL_LINKED_SHARE_CEIL = 1e-9
# Env-level harm accrual, measured on the local_view_greedy POSITIVE CONTROL (not on an arm
# under test -- see the module docstring on why 813's per-arm consumption-share gate was
# circular). Direction UPPER. A competent forager at hazard-free D3 should sit far below
# this; 813's competent raw-obs arm read harm shares of 0.002-0.027 at eval.
CONTROL_HARM_SHARE_CEIL = 0.50


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _arm_contexts() -> List[Dict[str, Any]]:
    return [{"id": aid} for aid in ARM_IDS]


def _localfield_vector(obs_dict: Dict[str, Any]) -> torch.Tensor:
    """The 25-dim agent-centred resource gradient, shaped [1, 25] to match state_fn callers.

    Fails LOUDLY rather than degrading: a missing channel (use_proxy_fields=False) would
    silently turn both field arms into zero-information controls and their nulls would read
    as evidence against H-observation-interface when nothing had been exposed at all.
    """
    rfv = obs_dict.get(LOCALFIELD_KEY)
    if rfv is None:
        raise KeyError(
            f"{LOCALFIELD_KEY} absent from obs_dict -- the re-representation arms cannot be "
            "constructed; this rung must expose the local resource field"
        )
    return rfv.float().reshape(1, -1).to(DEVICE)


class _LocalFieldPPOEvalPolicy(Policy):
    """Greedy (argmax) eval of a PPO actor trained on the re-representation observation.

    Subclasses Policy so it inherits the exact eval contract capability_eval.evaluate_seed
    drives -- notably `reset(self, env)` (called at the start of every eval episode, with the
    env argument) and the no-op `post_step`. Mirrors x737.LatentPPOEvalPolicy.

    `agent` is None for ppo_localfield_only (no REE stack in the loop at all) and the warmed
    all-ON stack for ppo_latent_plus_localfield, whose state is z_world concatenated with the
    local field -- built by the SAME _state_for_field_arm used as that arm's training
    state_fn, so train and eval observation construction cannot drift apart.
    """

    def __init__(self, policy_net, agent=None) -> None:
        self.policy_net = policy_net
        self.agent = agent
        self.name = ARM_FIELD if agent is None else ARM_LATENT_FIELD

    def reset(self, env: Any) -> None:
        if self.agent is not None:
            self.agent.reset()

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        state = _state_for_field_arm(self.agent, obs_dict)
        with torch.no_grad():
            logits, _v = self.policy_net(state)
        if not torch.isfinite(logits).all():
            return int(np.random.randint(0, int(env.action_dim)))
        return int(torch.argmax(logits.reshape(-1), dim=-1).item())


def _state_for_field_arm(agent, obs_dict: Dict[str, Any]) -> torch.Tensor:
    """z_world (+) local field when an agent is supplied; local field alone when not.

    Single definition, used by BOTH the training state_fn and the eval policy, so train and
    eval observation construction cannot drift apart.
    """
    field = _localfield_vector(obs_dict)
    if agent is None:
        return field
    z = x737._agent_zworld(agent, obs_dict).reshape(1, -1).to(DEVICE)
    return torch.cat([z, field], dim=-1)


# --------------------------------------------------------------------------------------
# Preconditions. Regime-conditioned via `applies_to` (the V3-EXQ-785 rule): the z_world guard
# is scoped OUT of the two arms that never read z_world, so an unmoved encoder red-flags only
# the latent-reading arms and can never vacate ppo_raw_obs / ppo_localfield_only.
# --------------------------------------------------------------------------------------
PRECONDITIONS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="zworld_encoder_trained_in_p0",
        description=(
            "The z_world world_encoder must have moved during P0 (SD-070 warmup path). An "
            "unmoved encoder means the latent-reading arms are PPO on a frozen random "
            "projection, under which neither a lift nor a null says anything about what the "
            "TRAINED latent exposes."
        ),
        control="worst-cell world_encoder_max_abs_delta over all seed warmups vs 1e-6",
        threshold=ZWORLD_DELTA_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: ctx["id"] in LATENT_READING_ARMS,
        applies_note=(
            "ppo_raw_obs and ppo_localfield_only never read z_world, so an untrained encoder "
            "is not meaningful for those arms' readiness"
        ),
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
            "The env must be floor-achievable from the SAME 5x5 resource field this probe "
            "hands the re-representation arms. This is the readiness check that makes the "
            "field arms interpretable: it certifies the added channel carries enough signal "
            "for competence, so a null on those arms is a fact about the LEARNER or the "
            "LATENT, never about the channel being empty. 738 measured 48.05 here."
        ),
        control="local_view_greedy worst-seed mean resources/ep vs the 1.0 competence floor",
        threshold=float(COMPETENCE_RESOURCE_FLOOR),
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="survival_linked_share_zeroed",
        description=(
            "STRUCTURAL check that the objective correction landed: at W3_survival_zeroed "
            "the proximity/novelty/other families are multiplied by zero, so the realised "
            "survival-linked share must be exactly 0. A non-zero reading means a leaked "
            "term (implementation defect), not a finding. Deliberately replaces V3-EXQ-813's "
            "per-arm consumption_share floor, which was CIRCULAR -- consumption share falls "
            "when an arm forages less, so it penalised each arm for the failure under test."
        ),
        control="worst-cell (max) train-phase survival_linked_share for this arm",
        threshold=SURVIVAL_LINKED_SHARE_CEIL,
        direction="upper",
        kind="readiness",
    ),
    PreconditionSpec(
        name="harm_share_negligible_on_control",
        description=(
            "Env-level harm accrual must not dominate the corrected return. Measured on the "
            "local_view_greedy POSITIVE CONTROL -- a competent forager NOT under test -- so "
            "this certifies an environment property without reading it off an arm whose "
            "competence is the dependent variable."
        ),
        control="worst-seed (max) eval-phase harm share of the local_view_greedy anchor",
        threshold=CONTROL_HARM_SHARE_CEIL,
        direction="upper",
        kind="readiness",
    ),
]


def _worst_cell(rows: List[Dict[str, Any]], key: str, mode: str = "min") -> Tuple[float, str]:
    return x808._worst_cell(rows, key, mode)


def _arm_row(arm_id: str, seed: int, eval_row: Dict[str, Any],
             train_decomp: Dict[str, Any], eval_decomp: Dict[str, Any],
             obs_dim: int) -> Dict[str, Any]:
    return {
        "cell_id": f"{arm_id}|seed{seed}",
        "arm_id": arm_id,
        "seed": int(seed),
        "obs_dim": int(obs_dim),
        "foraging_competence": float(eval_row["foraging_competence"]),
        "survival_horizon": float(eval_row["survival_horizon"]),
        "death_rate": float(eval_row["death_rate"]),
        "mean_episode_reward": float(eval_row["mean_episode_reward"]),
        "competence_supra_floor": bool(eval_row["competence_supra_floor"]),
        "train_decomposition": train_decomp,
        "eval_decomposition": eval_decomp,
        "train_consumption_share": float(train_decomp["consumption_share"]),
        "train_survival_linked_share": float(train_decomp["survival_linked_share"]),
        "eval_harm_share": float(eval_decomp["shares"]["harm"]),
    }


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
    """One seed: warm ONE all-ON stack, then train + eval ALL FOUR arms at the fixed W3 level.

    The single shared warmup is what makes ppo_ree_latent and ppo_latent_plus_localfield a
    matched pair: same encoder, same weights, same episode stream -- the ONLY difference
    between them is whether the 25-dim resource field is concatenated onto the observation.
    """
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
    # route to /failure-autopsy. The latent-reading arms are gated instead via precondition
    # zworld_encoder_trained_in_p0 (scoped OUT of the two non-latent arms).
    assert_world_encoder_trained(
        agent, before, p0=p0, strict=False,
        context=f"{QUEUE_ID} rung={RUNG_ID} seed={seed}",
        escape_hint=(
            "this driver gates the latent-reading arms (ppo_ree_latent, "
            "ppo_latent_plus_localfield) via precondition zworld_encoder_trained_in_p0, "
            "scoped OUT of ppo_raw_obs / ppo_localfield_only which do not read z_world: an "
            "unmoved encoder red-flags only those two arms, not the whole run"
        ),
    )

    _flat, probe_obs = x734._make_env(seed, env_kwargs).reset()
    z_dim = int(x737._agent_zworld(agent, probe_obs).shape[-1])
    field_dim = int(_localfield_vector(probe_obs).shape[-1])
    action_dim = int(warm_env.action_dim)
    _flat2, probe2 = x734._make_env(seed, env_kwargs).reset()
    obs_keys = x734._raw_obs_keys_present(probe2)
    raw_dim = int(x734._raw_obs_vector(probe2, obs_keys, DEVICE).shape[-1])
    print(
        f"  obs dims: z_world={z_dim} local_field={field_dim} "
        f"latent_plus_field={z_dim + field_dim} raw={raw_dim}", flush=True,
    )

    arm_rows: Dict[str, Dict[str, Any]] = {}

    # Per-arm (state_fn factory, on_reset, eval-policy factory, obs_dim, rng offset). One
    # table so every arm goes through the SAME train/eval path and cannot drift.
    arm_specs = [
        (ARM_LATENT, 1000, z_dim,
         (lambda od: x737._agent_zworld(agent, od)),
         agent.reset,
         (lambda net: x737.LatentPPOEvalPolicy(net, agent))),
        (ARM_RAW, 2000, raw_dim,
         (lambda od: x734._raw_obs_vector(od, obs_keys, DEVICE)),
         None,
         (lambda net: x734.PPOEvalPolicy(net, obs_keys, DEVICE))),
        (ARM_LATENT_FIELD, 3000, z_dim + field_dim,
         (lambda od: _state_for_field_arm(agent, od)),
         agent.reset,
         (lambda net: _LocalFieldPPOEvalPolicy(net, agent))),
        (ARM_FIELD, 4000, field_dim,
         (lambda od: _state_for_field_arm(None, od)),
         None,
         (lambda net: _LocalFieldPPOEvalPolicy(net, None))),
    ]

    for arm_id, rng_offset, obs_dim, state_fn, on_reset, eval_policy_of in arm_specs:
        print(f"Seed {seed} Condition {RUNG_ID}:{arm_id}", flush=True)
        torch.manual_seed(seed + rng_offset)
        np.random.seed(seed + rng_offset)
        net = x734.PPOPolicyNet(in_dim=obs_dim, action_dim=action_dim).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=x734.PPO_LR)
        train_decomp = x808._train_ppo_decomposed(
            x734._make_env(seed, env_kwargs), net, opt,
            state_fn=state_fn,
            on_reset=on_reset,
            n_episodes=ppo_eps, rollout_episodes=rollout, steps_per_episode=steps,
            level_id=LEVEL_ID, w_consume=W_CONSUME, w_survival=W_SURVIVAL, seed=seed,
            total_denominator=ppo_eps,
        )
        eval_row = evaluate_seed(
            eval_policy_of(net), x734._make_env(seed, env_kwargs), eval_eps, steps,
        )
        eval_decomp = x808._decompose_eval(
            eval_policy_of(net), x734._make_env(seed, env_kwargs),
            eval_eps, steps, W_CONSUME, W_SURVIVAL,
        )
        arm_rows[arm_id] = _arm_row(
            arm_id, seed, eval_row, train_decomp, eval_decomp, obs_dim)
        print(f"verdict: {'PASS' if eval_row['competence_supra_floor'] else 'FAIL'}", flush=True)

    # --- anchors: random_walk (floor) / local_view_greedy (own-observability ceiling AND the
    #     positive control for harm accrual) / greedy_oracle (global-info ceiling) ----------
    anchor_rows: List[Dict[str, Any]] = []
    anchors = {
        "random_walk": RandomPolicy(seed),
        "local_view_greedy": LocalViewGreedyPolicy(seed),
        "greedy_oracle": OraclePolicy(),
    }
    for aid in ANCHOR_IDS:
        print(f"Seed {seed} Condition {RUNG_ID}:{aid}", flush=True)
        row = evaluate_seed(anchors[aid], x734._make_env(seed, env_kwargs), eval_eps, steps)
        entry = {
            "cell_id": f"{aid}|seed{seed}",
            "anchor_id": aid,
            "seed": int(seed),
            "foraging_competence": float(row["foraging_competence"]),
            "survival_horizon": float(row["survival_horizon"]),
            "mean_episode_reward": float(row["mean_episode_reward"]),
            "competence_supra_floor": bool(row["competence_supra_floor"]),
        }
        if aid == "local_view_greedy":
            # POSITIVE CONTROL for the harm-share readiness precondition. Decomposed under
            # the SAME weighting the arms train on, on a competent forager not under test.
            ctrl_decomp = x808._decompose_eval(
                LocalViewGreedyPolicy(seed), x734._make_env(seed, env_kwargs),
                eval_eps, steps, W_CONSUME, W_SURVIVAL,
            )
            entry["eval_decomposition"] = ctrl_decomp
            entry["eval_harm_share"] = float(ctrl_decomp["shares"]["harm"])
            entry["eval_survival_linked_share"] = float(ctrl_decomp["survival_linked_share"])
        anchor_rows.append(entry)
        print(f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'}", flush=True)

    return {"seed": int(seed), "guard": guard, "arms": arm_rows, "anchors": anchor_rows,
            "agent": agent}


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
    contexts = _arm_contexts()
    # Design-time refusal BEFORE compute: catches a gate no arm could pass from its own
    # pre-registered config, for free at queue time (the V3-EXQ-785 arithmetic).
    audited = assert_no_structurally_unsatisfiable_gate(PRECONDITIONS, contexts, arm_id_key="id")
    print(
        f"structural-vacuity audit: {len(audited)} (spec, arm) pairs checked, "
        f"no unsatisfiable gate", flush=True,
    )

    seed_rows = [
        _run_seed(s, zworld_p0, p0, p1, ppo_eps, eval_eps, steps, rollout, dry_run)
        for s in seeds
    ]

    guards = [r["guard"] for r in seed_rows]
    for g in guards:
        g["cell_id"] = f"warmup|seed{g['seed']}"

    all_arm_rows: Dict[str, List[Dict[str, Any]]] = {
        aid: [r["arms"][aid] for r in seed_rows] for aid in ARM_IDS
    }
    all_anchor_rows = [row for r in seed_rows for row in r["anchors"]]

    per_arm: Dict[str, Dict[str, Any]] = {}
    for aid in ARM_IDS:
        rows = all_arm_rows[aid]
        n_supra = int(sum(1 for r in rows if r["competence_supra_floor"]))
        per_arm[aid] = {
            "arm_id": aid,
            "n_seeds": len(rows),
            "obs_dim": int(rows[0]["obs_dim"]) if rows else 0,
            "foraging_competence_mean": round(_mean([r["foraging_competence"] for r in rows]), 6),
            "foraging_competence_per_seed": [round(r["foraging_competence"], 6) for r in rows],
            "survival_horizon_mean": round(_mean([r["survival_horizon"] for r in rows]), 6),
            "n_seeds_supra_floor": n_supra,
            "majority_supra_floor": bool(n_supra >= (len(rows) + 1) // 2) if rows else False,
            "train_consumption_share_mean": round(
                _mean([r["train_consumption_share"] for r in rows]), 6),
            "train_consumption_share_per_seed": [
                round(r["train_consumption_share"], 6) for r in rows],
            "train_survival_linked_share_per_seed": [
                round(r["train_survival_linked_share"], 9) for r in rows],
            "eval_harm_share_per_seed": [round(r["eval_harm_share"], 6) for r in rows],
            "per_seed_cells": rows,
        }

    per_anchor: Dict[str, Dict[str, Any]] = {}
    for aid in ANCHOR_IDS:
        rows = [r for r in all_anchor_rows if r["anchor_id"] == aid]
        per_anchor[aid] = {
            "anchor_id": aid,
            "n_seeds": len(rows),
            "foraging_competence_mean": round(_mean([r["foraging_competence"] for r in rows]), 6),
            "foraging_competence_per_seed": [round(r["foraging_competence"], 6) for r in rows],
            "survival_horizon_mean": round(_mean([r["survival_horizon"] for r in rows]), 6),
            "per_seed_cells": rows,
        }

    guard_measured, guard_cell = _worst_cell(guards, "world_encoder_max_abs_delta", "min")
    oracle_measured, oracle_cell = _worst_cell(
        [r for r in all_anchor_rows if r["anchor_id"] == "greedy_oracle"],
        "foraging_competence", "min")
    lvg_rows = [r for r in all_anchor_rows if r["anchor_id"] == "local_view_greedy"]
    lvg_measured, lvg_cell = _worst_cell(lvg_rows, "foraging_competence", "min")
    # Worst cell for an UPPER bound is the MAXIMUM -- the same statistic the `met` test
    # quantifies over, so the indexer's recompute reproduces the author's verdict exactly.
    ctrl_harm_measured, ctrl_harm_cell = _worst_cell(lvg_rows, "eval_harm_share", "max")

    arm_gates: List[Dict[str, Any]] = []
    for ctx in contexts:
        aid = str(ctx["id"])
        rows = all_arm_rows[aid]
        sls_measured, sls_cell = _worst_cell(rows, "train_survival_linked_share", "max")
        measured = {
            "zworld_encoder_trained_in_p0": guard_measured,
            "d3_oracle_clears_floor": oracle_measured,
            "d3_local_view_greedy_clears_floor": lvg_measured,
            "survival_linked_share_zeroed": sls_measured,
            "harm_share_negligible_on_control": ctrl_harm_measured,
        }
        gate = evaluate_arm_gate(aid, ctx, PRECONDITIONS, measured)
        gate["offending_cells"] = {
            "zworld_encoder_trained_in_p0": guard_cell,
            "d3_oracle_clears_floor": oracle_cell,
            "d3_local_view_greedy_clears_floor": lvg_cell,
            "survival_linked_share_zeroed": sls_cell,
            "harm_share_negligible_on_control": ctrl_harm_cell,
        }
        arm_gates.append(gate)

    agg = aggregate_arm_gates(arm_gates)

    green = {aid: (aid in agg["green_arms"]) for aid in ARM_IDS}
    clears = {aid: bool(per_arm[aid]["majority_supra_floor"]) for aid in ARM_IDS}

    latent_forage = per_arm[ARM_LATENT]["foraging_competence_mean"]
    field_forage = per_arm[ARM_LATENT_FIELD]["foraging_competence_mean"]
    lift = round(field_forage - latent_forage, 6)

    readiness_met = bool(agg["any_green"])

    # ---- interpretation grid -----------------------------------------------------------
    # Replication guards run FIRST: if the V3-EXQ-813 anchor pair did not reproduce, this run
    # is not comparable to the result the leg rests on and every representation reading below
    # is withheld rather than issued against an unverified baseline.
    if not readiness_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif green[ARM_RAW] and not clears[ARM_RAW]:
        outcome, label = "FAIL", "anchor_replication_failed_raw_obs_subfloor"
    elif green[ARM_LATENT] and clears[ARM_LATENT]:
        outcome, label = "FAIL", "anchor_replication_diverged_latent_arm_supra_floor"
    elif green[ARM_LATENT_FIELD] and clears[ARM_LATENT_FIELD]:
        outcome, label = "PASS", "observation_interface_confirmed_re_representation_lifts_competence"
    elif green[ARM_FIELD] and clears[ARM_FIELD]:
        outcome, label = "FAIL", "latent_actively_obstructs_not_merely_underexposes"
    else:
        outcome, label = "FAIL", "re_representation_does_not_lift_competence"

    criteria_nd = arm_criteria_non_degenerate(
        {
            ARM_LATENT: ["C_latent_clears_floor"],
            ARM_RAW: ["C_raw_obs_clears_floor"],
            ARM_LATENT_FIELD: ["C_latent_plus_localfield_clears_floor"],
            ARM_FIELD: ["C_localfield_only_clears_floor"],
        },
        agg,
    )

    interpretation = {
        "label": label,
        "combination_rule": (
            "PASS is carried by ONE criterion -- C_latent_plus_localfield_clears_floor -- not "
            "by a conjunction. The other three load-bearing criteria are expected to read "
            "passed=false (C_latent_clears_floor, C_localfield_only_clears_floor) or "
            "passed=true (C_raw_obs_clears_floor) as REPLICATION anchors, and their values "
            "select which FAIL label is issued rather than gating PASS. Read the flat "
            "criteria list against this rule: a run with three false criteria and a PASS is "
            "the designed shape, not a mostly-failed run."
        ),
        "declared_null": (
            "ppo_latent_plus_localfield does NOT clear the 1.0 competence floor on a strict "
            "majority of seeds AND ppo_localfield_only does not either -> "
            "re_representation_does_not_lift_competence. Restoring a re-representation whose "
            "sufficiency is independently certified in this same run (the local_view_greedy "
            "anchor clears the floor reading ONLY that channel) does not lift competence, so "
            "the ceiling is NOT the observation interface and H-observation-interface is "
            "REFUTED as a passive-under-exposure account. Live discrimination passes back to "
            "H-substrate-ceiling / H-f-dominance. This does NOT eliminate H-substrate-ceiling "
            "(nothing here tests it positively) and does NOT re-open H-policy-learning."
        ),
        "alternative_outcome_note": (
            "If ppo_latent_plus_localfield clears the floor, H-observation-interface is "
            "CONFIRMED and the missing content is NAMED: the resource gradient, which is "
            "present in z_world's own input (world_state[225:250]) and not exposed by z_world "
            "to a downstream reader. That is actionable at the substrate. If instead arm 3 is "
            "flat while ppo_localfield_only clears, the latent ACTIVELY OBSTRUCTS rather than "
            "merely under-exposing -- a different and sharper claim, whose substrate work is "
            "interference removal, not content addition."
        ),
        "replication_note": (
            "ppo_ree_latent and ppo_raw_obs are exact per-seed replications of V3-EXQ-813's "
            "anchor pair (same rung, same W3_survival_zeroed objective, same seeds, same "
            "budgets), which measured 0.5 and 9.033 res/ep. They are re-run rather than cited "
            "so all four arms share one substrate_hash and one objective; 813's numbers are a "
            "cross-check, not the baseline this run reads against."
        ),
        "dv_symmetry_note": (
            "Per-arm DV-symmetry declaration. DV = D3 foraging_competence, a count of "
            "resources consumed per eval episode. Every arm's manipulation changes the "
            "OBSERVATION VECTOR the PPO policy conditions on, hence the action distribution, "
            "hence the trajectory, hence the count. No arm's manipulation is a broadcast "
            "additive constant over candidate scores (no candidate-score layer exists here -- "
            "PPO emits logits over 5 primitive actions directly), a monotone rescaling of a "
            "rank-valued DV, or a permutation of interchangeable units. So no arm's delta is "
            "an arithmetic identity fixed before the run. Arms 1 and 2 replicate 813 cells "
            "that empirically DID move this DV (0.5 vs 9.033)."
        ),
        "preconditions_scope_note": agg["per_arm_gate"]["preconditions_scope_note"],
        "preconditions": agg["adjudication_preconditions"],
        "criteria_non_degenerate": criteria_nd,
        "criteria": [
            {
                "name": "C_latent_plus_localfield_clears_floor",
                "load_bearing": True,
                "passed": bool(clears[ARM_LATENT_FIELD]),
                "measured": field_forage,
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "role": "THE positive demonstration -- the criterion that carries PASS",
            },
            {
                "name": "C_localfield_only_clears_floor",
                "load_bearing": True,
                "passed": bool(clears[ARM_FIELD]),
                "measured": per_arm[ARM_FIELD]["foraging_competence_mean"],
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "role": "interference discriminator -- separates under-exposure from obstruction",
            },
            {
                "name": "C_latent_clears_floor",
                "load_bearing": True,
                "passed": bool(clears[ARM_LATENT]),
                "measured": latent_forage,
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "role": "813 FLOOR anchor replication -- expected false; true means divergence",
            },
            {
                "name": "C_raw_obs_clears_floor",
                "load_bearing": True,
                "passed": bool(clears[ARM_RAW]),
                "measured": per_arm[ARM_RAW]["foraging_competence_mean"],
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "role": "813 CEILING anchor replication -- expected true; false means divergence",
            },
        ],
    }

    return {
        "outcome": outcome,
        "agents": [r["agent"] for r in seed_rows],
        "interpretation": interpretation,
        "non_degenerate": bool(agg["non_degenerate"]),
        "degeneracy_reason": agg["degeneracy_reason"],
        "per_arm_gate": agg["per_arm_gate"],
        "per_arm": per_arm,
        "per_anchor": per_anchor,
        "per_seed_arms": {aid: all_arm_rows[aid] for aid in ARM_IDS},
        "per_seed_anchors": all_anchor_rows,
        "diagnostics": {
            "zworld_encoder_guard": {
                "policy": "green_gating_latent_reading_arms_only",
                "policy_reason": (
                    "A frozen random projection would make both a lift and a null on the "
                    "latent-reading arms uninterpretable as statements about the TRAINED "
                    "latent, so the guard gates those two arms specifically (scoped OUT of "
                    "ppo_raw_obs / ppo_localfield_only via applies_to)."
                ),
                "n_cells": len(guards),
                "worst_cell_max_abs_delta": round(guard_measured, 9),
                "worst_cell": guard_cell,
                "all_trained": bool(all(
                    float(g.get("world_encoder_max_abs_delta", 0.0)) > ZWORLD_DELTA_FLOOR
                    for g in guards)),
                "per_cell": guards,
            },
            "re_representation_lift": {
                "definition": (
                    "ppo_latent_plus_localfield minus ppo_ree_latent, mean D3 "
                    "foraging_competence. The decisive contrast: identical REE stack, "
                    "identical warmup, identical objective, identical seeds; the ONLY "
                    "difference is the 25-dim resource field on the observation."
                ),
                "latent_forage": latent_forage,
                "latent_plus_localfield_forage": field_forage,
                "lift": lift,
                "per_seed_lift": [
                    round(f["foraging_competence"] - l["foraging_competence"], 6)
                    for l, f in zip(all_arm_rows[ARM_LATENT], all_arm_rows[ARM_LATENT_FIELD])
                ],
                "localfield_only_forage": per_arm[ARM_FIELD]["foraging_competence_mean"],
                "raw_obs_forage": per_arm[ARM_RAW]["foraging_competence_mean"],
            },
            "objective_correction": {
                "note": (
                    "Replaces V3-EXQ-813's circular per-arm consumption_share gate. "
                    "survival_linked_share is 0 by construction at w_survival=0.0; the "
                    "non-consumption residual is harm, which accrues per-tick from the env "
                    "while consumption accrues only when an arm forages -- so consumption "
                    "share is ENDOGENOUS to the competence under test and cannot be a "
                    "readiness statistic here."
                ),
                "survival_linked_share_worst_by_arm": {
                    aid: round(_worst_cell(
                        all_arm_rows[aid], "train_survival_linked_share", "max")[0], 12)
                    for aid in ARM_IDS
                },
                "consumption_share_train_mean_by_arm": {
                    aid: per_arm[aid]["train_consumption_share_mean"] for aid in ARM_IDS},
                "control_eval_harm_share_worst": round(ctrl_harm_measured, 6),
                "control_eval_harm_share_worst_cell": ctrl_harm_cell,
            },
        },
        "headline": {
            "latent_forage": latent_forage,
            "latent_plus_localfield_forage": field_forage,
            "re_representation_lift": lift,
            "localfield_only_forage": per_arm[ARM_FIELD]["foraging_competence_mean"],
            "raw_obs_forage": per_arm[ARM_RAW]["foraging_competence_mean"],
            "clears_majority": clears,
            "arm_green": green,
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
            "adds no ceiling reading to MECH-457 or ARC-065. Matches the exemption V3-EXQ-808 "
            "and V3-EXQ-813 carry for the same hypothesis-space fanout lineage."
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
        "per_arm": result["per_arm"],
        "per_anchor": result["per_anchor"],
        "per_seed_arms": result["per_seed_arms"],
        "per_seed_anchors": result["per_seed_anchors"],
        "config": cfg,
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
            "note": (
                "Hazard-free D3 (num_hazards=0, proximity_harm_scale=0.0); no sleep/replay; "
                "no self-model; V3 pre-ethical instrumentation only (SENT-0 boundary)."
            ),
        },
        "hypothesis_space": {
            "question_id": "conversion_ceiling_root",
            "leg": "H-observation-interface",
            "axis": "representation",
            "role": (
                "positive-demonstration probe (GOV-FANOUT-1); representation-axis successor "
                "to V3-EXQ-813, which ELEVATED this leg but could only establish it "
                "negatively (the latent is not sufficient)"
            ),
            "source_autopsy": "failure_autopsy_backlog_2026-07-24",
            "predecessor_probe": "V3-EXQ-813",
            "sibling_probes": ["V3-EXQ-808 (reward axis)", "V3-EXQ-813 (policy axis)"],
            "reading": (
                "PASS (ppo_latent_plus_localfield clears the floor) CONFIRMS "
                "H-observation-interface and NAMES the missing content as the resource "
                "gradient. A null on BOTH re-representation arms REFUTES the leg as a "
                "passive-under-exposure account and passes live discrimination back to "
                "H-substrate-ceiling / H-f-dominance -- it does NOT eliminate "
                "H-substrate-ceiling, which this design does not test positively. A flat "
                "arm 3 with a clearing arm 4 re-routes to ACTIVE INTERFERENCE, a distinct "
                "claim. Replication guards on the 813 anchor pair fire FIRST: if they do not "
                "reproduce, all representation readings are withheld."
            ),
            "known_substrate_limitations": [
                "mode-governance-engagement (corrupting, implemented_pending_validation; "
                "repair behind default-off salience_affinity_input_cap)",
                "contextmemory-write-path-addressing-degeneracy (corrupting, "
                "implemented_pending_validation; repair behind default-off refractory mode)",
            ],
            "known_substrate_limitations_note": (
                "Both are present IDENTICALLY in ppo_ree_latent and "
                "ppo_latent_plus_localfield -- the pair carrying the decisive contrast -- and "
                "absent from ppo_raw_obs / ppo_localfield_only, which instantiate no REE "
                "agent. Neither can manufacture or mask the arm-3-minus-arm-1 lift."
            ),
        },
        "load_bearing_dv": (
            "C_latent_plus_localfield_clears_floor: D3 foraging_competence mean (strict "
            "majority of seeds) vs the 1.0 competence floor for ppo_latent_plus_localfield, "
            "under the fixed W3_survival_zeroed objective. Read alongside "
            "diagnostics.re_representation_lift (arm 3 minus arm 1), the matched contrast "
            "the leg turns on. C_localfield_only_clears_floor discriminates active "
            "interference from passive under-exposure; the two 813-anchor criteria are "
            "replication guards that select the FAIL label rather than gating PASS (see "
            "interpretation.combination_rule)."
        ),
        "notes": (
            "Representation-axis GOV-FANOUT-1 probe supplying the POSITIVE DEMONSTRATION the "
            "H-observation-interface registry entry names as its own gate. Four arms hold the "
            "learner (PPO) and objective (W3_survival_zeroed, imported verbatim from "
            "V3-EXQ-808) fixed and vary ONLY the observation vector: z_world; raw obs; "
            "z_world + the 25-dim resource_field_view; that field alone. The added channel is "
            "not invented -- the local_view_greedy anchor scores ~45.75 res/ep reading ONLY "
            "it, so its adequacy under the learner's own observability is certified in this "
            "same run, and it is a subset of world_state, which the 9.033 raw-obs arm already "
            "sees. V3-EXQ-813's per-arm consumption_share readiness gate is deliberately NOT "
            "reproduced: it is circular (consumption share falls when an arm forages less, so "
            "it penalises each arm for the failure under test) and forced the "
            "substrate_not_ready_requeue self-route the 813 autopsy had to override. Replaced "
            "by a structural survival_linked_share check and a harm-share check on the "
            "local_view_greedy positive control. DIAGNOSTIC: promotes and demotes nothing; "
            "route to /failure-autopsy before any governance use."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-948 representation-axis GOV-FANOUT-1 positive-demonstration probe for "
            "H-observation-interface (conversion_ceiling_root; diagnostic; claim_ids=[])"
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
        "arms": ARM_IDS,
        "latent_reading_arms": sorted(LATENT_READING_ARMS),
        "localfield_key": LOCALFIELD_KEY,
        "anchors": ANCHOR_IDS,
        "weighting_level_id": LEVEL_ID,
        "w_consume": W_CONSUME,
        "w_survival": W_SURVIVAL,
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
        "zworld_delta_floor": float(ZWORLD_DELTA_FLOOR),
        "survival_linked_share_ceil": float(SURVIVAL_LINKED_SHARE_CEIL),
        "control_harm_share_ceil": float(CONTROL_HARM_SHARE_CEIL),
        "term_families": list(x808.TERM_FAMILIES),
        "survival_linked_families": list(x808.SURVIVAL_LINKED_FAMILIES),
        "predecessor_run": "V3-EXQ-813",
        "predecessor_anchor_values": {"ppo_ree_latent": 0.5, "ppo_raw_obs": 9.033333},
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
        agent=result["agents"],
        elapsed_seconds=(datetime.now(timezone.utc) - _started).total_seconds(),
    )

    hl = result["headline"]
    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} label={result['interpretation']['label']} "
        f"readiness_met={hl['readiness_met']}", flush=True,
    )
    for aid in ARM_IDS:
        pa = result["per_arm"][aid]
        print(
            f"  ARM {aid}: obs_dim={pa['obs_dim']} forage/ep={pa['foraging_competence_mean']} "
            f"clears={hl['clears_majority'][aid]} green={hl['arm_green'][aid]}", flush=True,
        )
    print(
        f"  LIFT (latent+field minus latent): {hl['re_representation_lift']}", flush=True)
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
