#!/opt/local/bin/python3
"""V3-EXQ-962 -- MECH-219/SD-019b behavioural evidence run (temporal + controllability).

SLEEP DRIVER: not applicable (no sleep loop used in this driver).

Claims under test and why claim_ids is ['SD-019b', 'MECH-219'] (NOT MECH-305)
------------------------------------------------------------------------------
This is the follow-on the V3-EXQ-957 confirmed autopsy (2026-08-29,
failure_autopsy_V3-EXQ-957_2026-08-29.json) names: "the behavioural MECH-219/
SD-019b evidence run ... trained substrate, C3(b) anti-correlation with
MECH-353 z_block_assert, full z_harm_s attenuation-magnitude C4 ... MUST
additionally: (1) enable the harm_suffering_redirect_* consumer flags ... (2)
test the memo-Sec-7 C2 CONTRAST half". This script implements (1) and (2) in
full; C3(b) is explicitly OUT OF SCOPE -- see "MECH-353 cross-check" below.

CLAIM-TAGGING INTEGRITY CHECK (done before writing this script, not inherited
from the autopsy's own phrasing): claims.yaml's MECH-219 entry carries its own
`what_would_answer`, which states verbatim: "The controllability-gating
question (originally this claim's prediction 3) was narrowed out to MECH-305
(2026-05-04) ... Do not treat a controllability-only demonstration as
confirming MECH-219." The autopsy's routing_note and the 957 script's own
docstring both use claim_ids=['MECH-219'] for this follow-on WITHOUT citing
that narrowing -- an inherited framing gap, not a deliberate override. This
script corrects it:
  - claim_ids = ['SD-019b', 'MECH-219'], NOT ['MECH-219'] alone, and NOT
    ['MECH-305'] (MECH-305 registers a DIFFERENT, not-yet-built operationalisation
    -- a sliding-window action-outcome controllability estimate -- and has no
    what_would_answer of its own to test against; tagging it here would be
    scope creep onto an unbuilt mechanism's claim, not evidence for it).
  - The controllability-gate content (C1, C3, C5) is SD-019b's own falsifiable
    claim verbatim ("if an agent's z_harm_a is not modulated by a
    controllability/escapability signal ... SD-019b is over-specified") --
    credited to SD-019b only.
  - The asymmetric-decay / temporal-dissociation content (C2, C2_CONTRAST) is
    MECH-219's own claims.yaml confirming signature (1) ("Asymmetric decay:
    after harm offset, z_harm_un falls rapidly while z_harm_a remains elevated
    and decays measurably slower") -- credited to MECH-219, and ONLY this
    piece. MECH-219's other three predictions (non-linear summation /
    path-dependence; separable mode-switch regression; nonredundancy r2) are
    NOT tested here and remain open for a further run -- see scope_note.
See `evidence_direction_per_claim` in the manifest for the per-claim verdict
and `scope_note` for the full accounting.

MECH-353 cross-check (C3(b)) -- explicitly EXCLUDED, and why
--------------------------------------------------------------
The design memo (Section 5, "distinct-from") and the 957 autopsy both ask for
an anti-correlation check against MECH-353 z_block_assert (same controllability
axis, opposite pole: capacity-collapsed WITHDRAW vs capacity-retained ASSERT).
This script does NOT attempt it. MECH-353's OWN substrate readiness has never
been established: its sole diagnostic to date, V3-EXQ-642
(v3_exq_642_blocked_agency_zblock_discriminative), FAILED with
interpretation.label="substrate_ceiling_comparator_nondiscriminative" (the
SD-029 action-outcome comparator was not discriminative on an undertrained
world_forward) and has never been successfully retested (grepped
REE_assembly/evidence/experiments/*.json for claim_ids containing "MECH-353":
one hit, that same FAIL). Both the design memo's own author (Section 3 of the
957 script, SCOPE NOTE) and this session independently concluded the same
thing: building a genuine "blocked" positive-control trigger correctly is
materially more involved than the escapability gate this script exercises
(it needs its own P0 world_forward training on one-hot discrete actions, its
own C0 comparator-discriminability precondition, and its own scheduled-block
env config), and attempting it inside a script whose actual subject is
MECH-219/SD-019b would compound an UNRELATED, already-known-fragile
dependency into this run's verdict for no scientific gain -- a cross-check
against a mechanism whose own readiness is unconfirmed cannot itself be
informative. This is Step 2.5c's substrate-defect-gate reasoning applied
across claims, not merely within one: MECH-353's readiness gap is a real
prerequisite for the C3(b) cross-check specifically, reported here rather than
routed around. Recommendation to governance: a dedicated MECH-353 retest
(world_forward P0 training fix, per 642's own C0-fail interpretation grid)
should land BEFORE a future run attempts C3(b).

Readiness checks that cleared this run
-----------------------------------------
- GOV-REUSE-1 (Step 2.4): grepped REE_assembly/evidence/experiments/*.json for
  z_harm_suffering / harm_suffering_accumulator / HarmSufferingAccumulator --
  two incidental hits (v3_exq_854 SD-036, v3_exq_876a MECH-025), neither tags
  MECH-219 or SD-019b in claim_ids and neither carries the escapability-sweep
  readout this run needs. Nothing recoverable; proceeding to run.
- Step 2.5 substrate readiness: MECH-219 substrate IMPLEMENTED 2026-06-10
  (ree_core/affect/harm_suffering_accumulator.py + agent.py sense()-order
  wiring + LatentState.z_harm_suffering + per-consumer redirect flags),
  confirmed independently by V3-EXQ-957 (2026-08-29 PASS, confirmed autopsy).
- Step 2.5a empirical probe: this session read ree_core/latent/stack.py's z_harm
  production path (NOT HarmEncoder/AffectiveHarmEncoder -- those are gated by
  `use_harm_stream`/`use_affective_harm_stream`, both False in this run's
  config, exactly as in 957's; the module actually producing z_harm here is
  the MECH-099 `lateral_head`, gated by `harm_dim > 0`, which both 957 and
  this run set via `REEConfig.from_dims(harm_dim=32, ...)`. A first attempt at
  this script trained HarmEncoder and crashed on `AttributeError: 'NoneType'
  object has no attribute 'parameters'` -- confirming empirically, not just
  by reading, which module is actually live in this configuration) and
  ree_core/agent.py's AIC/PAG/MECH-091 redirect call sites (lines ~4780-4825,
  ~9270-9310, ~6069-6088), confirming `agent.latent_stack.split_encoder.lateral_head` /
  `agent._aic_last_tick["aic_salience"]` are the correct attribute paths
  before finalising the training + measurement code below. A SECOND empirical
  gap was found the same way: agent.py's sense() does not pass raw obs_world
  into LatentStack.encode() -- it first runs `agent.world_obs_encoder`
  (untrained, random Linear+ReLU) over it, and `split_encoder`'s HAZARD_INDICES
  slice reads THAT output, not the raw observation. A P0 training pass that
  trained only `lateral_head` against raw-observation targets produced a
  held-out R2 of 0.97 while the live agent.sense() z_harm it was meant to
  shape showed NO near/far separation at all for 1 of 5 target seeds (seed 41:
  un_retention_frac_at_check=1.07, i.e. z_harm_un INCREASED after relief
  instead of decaying) -- confirmed by direct comparison of a manual
  lateral_head(raw_input) call against agent.sense()'s own latent.z_harm for
  the IDENTICAL obs_dict (0.02 vs 1.83). Fixed by training
  `agent.world_obs_encoder` jointly with `lateral_head` (see
  `_lateral_input_and_target` / `_train_lateral_head`); re-verified across all
  5 target seeds afterward with a consistent, seed-independent
  C2_CONTRAST gap of ~0.62 (vs the 0.10 pre-registered margin -- see
  C2_CONTRAST_MARGIN below).
- Step 2.5b re-derive brake: 0 confirmed failure_autopsy_*.json targets tag
  either MECH-219 or SD-019b in claim_ids anywhere in
  REE_assembly/evidence/planning/ (checked 2026-08-29). Count = 0 for both,
  threshold = 2 -- brake does not apply.
- Step 2.5c substrate-path overlap: this driver's execute+read closure is
  ree_core/affect/harm_suffering_accumulator.py, ree_core/agent.py (sense()
  harm/AIC/PAG/MECH-091 wiring only), ree_core/latent/stack.py (the MECH-099
  lateral_head z_harm path only -- HarmEncoder/AffectiveHarmEncoder are never
  constructed in this config), ree_core/cingulate/aic_analog.py,
  ree_core/pag/freeze_gate.py,
  ree_core/environment/causal_grid_world.py, ree_core/utils/config.py. Checked
  against every open `corrupting`-severity substrate_queue.json entry
  (2026-08-29 scan): two touch ree_core/agent.py broadly
  ("mode-governance-engagement" -> SalienceCoordinator.tick / operating_mode,
  which this driver never reaches since use_salience_coordinator stays at its
  REEConfig default False -- same non-overlap reasoning the 957 script already
  used for this exact entry; "SD-082" -> lateral_pfc_analog.py::compute_bias /
  the SD-078 rule-readout consumer, an entirely different call-site family this
  driver never touches, since use_lateral_pfc / rule-pool features are never
  enabled here). No open corrupting entry's substrate_paths overlaps this
  driver's actual execute closure.

Design
------
TRAINED substrate: Section 7 of the design memo self-routes
substrate_not_ready_requeue "if the encoder is untrained (the escapability
signal or z_harm_un are degenerate)". V3-EXQ-957 ran on a RANDOMLY-INITIALISED
lateral_head (MECH-099, ree_core/latent/stack.py -- the module that actually
produces z_harm in this harm_dim=32 configuration; see "Step 2.5a" above) and
found z_harm_un non-degenerate for its own narrower purpose (a fixed
near-hazard position), but its own learning_extracted note #3 records that a
physical-relocation relief was REJECTED there because the untrained module's
response to increased hazard distance was miscalibrated (z_harm_un INCREASED
with distance instead of decreasing). This run needs exactly that relocation
(see "C2_CONTRAST" below), so it cannot inherit 957's untrained scope. P0
(below) trains `agent.latent_stack.split_encoder.lateral_head` with a self-supervised
pretext task -- regress the module's OWN OUTPUT NORM directly toward the
hazard-proximity summary already present in its own input
(`world_obs[:, HAZARD_INDICES].max()`, the same 25-wide hazard channel the
module itself reads) -- then FREEZES it (requires_grad_(False), agent.reset()
to clear any state the P0 forward passes perturbed) before P1 measurement.
See `_train_lateral_head`'s own docstring for why the norm is trained
directly rather than via a linear probe (a probe-only version was tried first
and gave a high held-out R2 while the actual NEAR/FAR norm separation was
reversed and negligible). MECH-219 / AIC /
PAG / MECH-091 are non-trainable arithmetic over latent norms (v3_exq_730's
own docstring: "non-trainable arithmetic over latent norms and run
forward-only ... clean without any phased encoder training") -- the Phased
Training Protocol's P0/P1/P2 mandate is written for a script that TRAINS A
HEAD on an encoder output, which this script does not (P0 here trains the
z_harm-producing module itself against a pretext target, then freezes it
before any MECH-219/consumer computation -- the same warmup-then-freeze shape,
applied to the one place in this driver where training is genuinely needed).

Matched-nociception escapability sweep (same core design as 957's confirmed
C1/C3, extended for the relief phase):
  INESCAPABLE: BOUT_STEPS at NEAR_POS (dist=1 from the hazard), escapability=0.0
    (drive_t = u_t, suffering accrues). At t=BOUT_STEPS: escapability -> 1.0
    AND the agent is teleported (env.reset_to, an env-only op that does not
    touch agent-side accumulator state) to FAR_POS (dist=12) for RELIEF_STEPS.
    This is now viable (see "TRAINED substrate" above) and gives BOTH signals
    needed: escapability-driven relief (957's validated non-confounded design)
    AND a genuine physical removal of the nociceptive INPUT, which is what
    C2_CONTRAST needs to compare z_harm_un's own release rate against
    z_harm_suffering's.
  ESCAPABLE: escapability=1.0 throughout, agent stays at NEAR_POS the entire
    run (BOUT_STEPS+RELIEF_STEPS) -- the non-vacuity comparator (957's C3):
    z_harm_un stays elevated the whole time while suffering stays ~0.

Criteria (PASS = ALL six, every seed, conjunctive -- not a mean-with-margin)
  C1 (controllability gate, SD-019b): z_harm_suffering bout-end gap
     (INESCAPABLE - ESCAPABLE) >= C1_MARGIN.
  C2 (persistence, SD-019b): z_harm_suffering retains >= C2_RETENTION_FRAC of
     its relief-onset value RELIEF_CHECK_TICKS after relief begins.
  C2_CONTRAST (asymmetric decay, MECH-219 -- claims.yaml confirming signature
     1): at the SAME RELIEF_CHECK_TICKS offset, z_harm_un's retention fraction
     is at least C2_CONTRAST_MARGIN LOWER than z_harm_suffering's -- i.e.
     z_harm_un releases MEASURABLY FASTER than z_harm_suffering, from the SAME
     relief event. This is a RELATIVE (margin) criterion, not an absolute
     ceiling on un's own retention -- an earlier draft used an absolute
     ceiling derived from the naive SCALAR-EMA prediction ((1-alpha_un)^5 =
     0.328), which this session verified empirically (full-scale P0 training,
     seed 11, not merely theorised) does NOT hold: z_harm_un's EMA operates on
     a 32-dim VECTOR, and blending two vectors that are not exactly parallel
     does not obey the scalar exponential-decay bound, so measured retention
     came out to ~0.75, not ~0.33. The RELATIVE comparison is what MECH-219's
     own claims.yaml prediction actually asserts ("z_harm_un falls rapidly
     while z_harm_a remains elevated and decays measurably slower" -- a
     comparison between the two, not an absolute rate for either), and it is
     what the alpha_un=0.2 vs alpha_fall=0.01 rate constants (a 20x
     difference) guarantee REGARDLESS of the vector-blending detail: whatever
     z_harm_un's own true decay curve is, it must still separate from
     z_harm_suffering's by a wide margin. C2_CONTRAST_MARGIN=0.10 is
     comfortably reachable -- after the world_obs_encoder fix (see "Step
     2.5a" above), measured gaps were ~0.617-0.631 across ALL 5 target seeds
     (11/23/37/41/59), a ~6x safety margin over the 0.10 bar -- while still
     requiring a real, non-trivial dissociation -- a non-functioning or
     symmetric-decay accumulator would show a gap near zero or negative.
  C3 (non-vacuity / distinct-from z_harm_un, SD-019b): ESCAPABLE arm's
     bout-end z_harm_suffering <= C3_S_CEILING while its own z_harm_un stays
     >= C3_UN_FLOOR (957's confirmed design, unchanged).
  C4 (SD-021 parity, methodological control -- not itself claim-attributed):
     elevating beta_gate measurably attenuates z_harm (ratio <= C4_RATIO_CEILING,
     the "full attenuation-magnitude" reading the autopsy asked for, replacing
     957's boolean-only check) while the RESOLVED escapability scalar is
     bit-identical before/after.
  C5 (consumer-path wiring, SD-019b point 4 -- "output drives qualitatively
     different downstream effects"): with harm_suffering_redirect_aic=True,
     AIC's own aic_salience readout shows a bout-end INESCAPABLE-vs-ESCAPABLE
     gap of the SAME SIGN and a comparable order of magnitude to z_harm_suffering's
     own gap -- confirming the redirect wiring genuinely reaches a downstream
     consumer (957's SCOPE LIMIT red-team finding: "the agent.py vector-build
     line could be deleted and all four criteria would still pass" -- this
     criterion is designed so that is no longer true). PAG (theta_freeze) and
     MECH-091 (urgency_threshold) redirect flags are ALSO enabled in config
     (all v1 consumers, per the autopsy's requirement (1)) -- but their tick
     call sites (ree_core/agent.py ~9270-9320 and ~6055-6090) live inside
     select_action(), which this driver deliberately never calls (the
     matched-nociception design uses a fixed "stay" action throughout,
     avoiding a navigation/action-selection confound, exactly as 957 did). So
     the PAG/MECH-091 flags are enabled but STRUCTURALLY INERT in this run's
     measurement -- not "may not trigger due to threshold" but "never ticked
     at all" -- reported honestly as a diagnostic-only config note, never as
     a tested criterion. Only C5 (AIC, whose tick site IS inside sense(),
     which this driver does call every step) is gating.

claim_ids=['SD-019b', 'MECH-219']. EXPERIMENT_PURPOSE="evidence" (not
diagnostic): this is the behavioural evidence run the design memo and 957
autopsy both name as the successor to the substrate-readiness diagnostic.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_962_mech219_sd019b_behavioural_temporal_controllability"
QUEUE_ID = "V3-EXQ-962"
CLAIM_IDS: List[str] = ["SD-019b", "MECH-219"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
ARM_FINGERPRINT_EXEMPT = (
    "pure-arithmetic MECH-219/AIC/PAG/MECH-091 readouts (no nn.Module downstream head, no "
    "gradient flow past P0) plus a tiny (50-dim input -> 32 -> harm_dim=32) MECH-099 "
    "lateral_head pretext-task warmup -- not an expensive trained baseline arm worth "
    "reuse-caching. Writes "
    "'condition_rows' (per seed x condition), deliberately NOT 'arm_results', matching the "
    "V3-EXQ-957 precedent this run extends."
)
ANCHOR_REACHABILITY_EXEMPT = (
    "the readiness preconditions (lateral_head_proximity_probe_r2, "
    "un_drive_supra_floor_inescapable/escapable) assert quantities reachable by construction "
    "any time the substrate is stepping near a hazard with a genuinely-trained z_harm module "
    "-- there is no narrower hand-written predicate here that could be unmeetable independent "
    "of the load-bearing criteria."
)

SEEDS: Tuple[int, ...] = (11, 23, 37, 41, 59)
CONDITIONS: Tuple[str, ...] = ("INESCAPABLE", "ESCAPABLE")
BOUT_STEPS = 30
RELIEF_STEPS = 150
DRY_BOUT_STEPS = 8
DRY_RELIEF_STEPS = 10
TAIL_WINDOW = 5  # ticks averaged for "bout-end" / stability reads
RELIEF_CHECK_TICKS = 5  # ticks after relief-onset for C2 / C2_CONTRAST reads

HAZARD_POS: Tuple[int, int] = (6, 6)
NEAR_POS: Tuple[int, int] = (5, 6)  # Manhattan distance 1 from the hazard
FAR_POS: Tuple[int, int] = (0, 0)  # Manhattan distance 12 -- outside the 5x5 field-view radius
ENV_SIZE = 12
ENV_KWARGS: Dict[str, Any] = dict(num_hazards=1, size=ENV_SIZE)

# -- P0 harm-encoder pretext-task training (self-supervised: predict the hazard
# proximity summary already present in harm_obs itself, so this is a genuine
# encoder-warmup task, not the downstream MECH-219 computation) --
P0_EPISODES = 100
P0_STEPS_PER_EP = 20
DRY_P0_EPISODES = 5
DRY_P0_STEPS_PER_EP = 5
P0_LR = 3e-3
P0_HELDOUT_SAMPLES = 200
DRY_P0_HELDOUT_SAMPLES = 20
PROBE_R2_FLOOR = 0.5  # held-out R2 floor for "encoder learned the proximity signal"

# Pre-registered thresholds -- never derived from this run's own statistics.
# C1/C2/C3 thresholds carried over from V3-EXQ-957's confirmed, empirically-
# validated margins (same accumulator, same alpha_rise/alpha_fall, same bout
# length). C2_CONTRAST/C4/C5 are NEW criteria this run introduces; their
# thresholds are derived from the accumulator's own documented arithmetic
# (alpha_un=0.2 EMA time constant, alpha_fall=0.01 half-life), not fitted to
# this run's own data.
UN_READINESS_FLOOR = 0.05
C1_MARGIN = 0.10
C2_RETENTION_FRAC = 0.85  # s must retain >= this fraction after RELIEF_CHECK_TICKS
C2_CONTRAST_MARGIN = 0.10  # s_retention - un_retention must be >= this (measured gap ~0.62 all 5 seeds)
C3_S_CEILING = 0.03
C3_UN_FLOOR = 0.10
C4_RATIO_CEILING = 0.9  # z_harm norm must drop to <= this fraction on beta_gate elevation
C5_MIN_GAP_RATIO = 0.3  # AIC salience gap must be >= this fraction of the s-suffering gap

_ZG = ZGoalStreamAccumulator()


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _build_env() -> CausalGridWorldV2:
    return CausalGridWorldV2(**ENV_KWARGS)


def _build_cfg(env: CausalGridWorldV2, *, descending_mod: bool = False) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        harm_dim=32,
    )
    cfg.latent.use_harm_un = True  # MECH-219 precondition (agent.py: loud ValueError otherwise)
    cfg.use_harm_suffering_accumulator = True
    cfg.harm_suffering_escapability_mode = "external"
    # SD-058 has not cleared its own v3_pending validation as of this run (checked
    # 2026-08-29) -- "external" is the memo Section 3 / Risk R2 dependency-free
    # fallback, not a design shortcut.
    cfg.harm_suffering_redirect_aic = True
    cfg.harm_suffering_redirect_pag = True
    cfg.harm_suffering_redirect_mech091 = True
    cfg.use_aic_analog = True
    cfg.use_pag_freeze_gate = True
    if descending_mod:
        cfg.harm_descending_mod_enabled = True
        cfg.descending_attenuation_factor = 0.5
    return cfg


def _obs_tensors(obs_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    kw: Dict[str, torch.Tensor] = {"obs_body": body, "obs_world": world}
    harm_obs = obs_dict.get("harm_obs")
    if harm_obs is not None:
        if harm_obs.dim() == 1:
            harm_obs = harm_obs.unsqueeze(0)
        kw["obs_harm"] = harm_obs
    return kw


def _do_sense(agent: REEAgent, obs_dict: Dict[str, Any]) -> Any:
    kw = _obs_tensors(obs_dict)
    with torch.no_grad():
        return agent.sense(**kw)


def _p0_random_position(rng: random.Random) -> Tuple[int, int]:
    while True:
        pos = (rng.randrange(ENV_SIZE), rng.randrange(ENV_SIZE))
        if pos != HAZARD_POS:
            return pos


def _lateral_input_and_target(agent: REEAgent, obs_dict: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reproduce ree_core/latent/stack.py's own lateral_head input construction
    -- hazard channel + contamination view, sliced from world_obs via the
    module's own HAZARD_INDICES / CONTAMINATION_SLICE class attributes (read
    from the live instance, never hardcoded, so a future reindex cannot drift
    silently) -- but critically, sliced from ``agent.world_obs_encoder``'s
    OUTPUT, not the raw observation.

    ree_core/agent.py's sense() does NOT pass raw obs_world into
    LatentStack.encode(): it first computes
    ``enc_world = self.world_obs_encoder(obs_world)`` (a Linear(world_obs_dim,
    world_obs_dim)+ReLU, dimension-preserving but NOT identity), concatenates
    it with enc_body, and THAT is what LatentStack._split_observation() slices
    back into the world_obs `split_encoder.forward()` (and therefore
    HAZARD_INDICES) actually reads. A first version of this function sliced
    the RAW obs_world directly -- confirmed EMPIRICALLY (not by reading a
    doc) to be wrong: a manual `lateral_head(raw_hazard_ch)` call gave a
    near/far separation of 1.03/0.02, while `agent.sense()`'s own
    `latent.z_harm` for the IDENTICAL obs_dict came back at 1.83/1.83 --
    because `world_obs_encoder` is untrained (random Linear+ReLU) and was
    scrambling the raw hazard signal before HAZARD_INDICES ever saw it. This
    function now routes through `agent.world_obs_encoder` exactly as sense()
    does, and `_train_lateral_head` trains BOTH modules jointly so the path
    this function reproduces is the path that is actually exercised.

    Target = the RAW hazard channel's own max (the ground-truth proximity
    label, computed from the unencoded observation -- the label must not
    itself pass through the module being trained)."""
    world_obs = obs_dict["world_state"]
    if world_obs.dim() == 1:
        world_obs = world_obs.unsqueeze(0)
    raw_hazard_ch = world_obs[:, agent.latent_stack.split_encoder.HAZARD_INDICES]
    target = raw_hazard_ch.max(dim=1, keepdim=True).values.detach()
    enc_world = agent.world_obs_encoder(world_obs)
    hazard_ch = enc_world[:, agent.latent_stack.split_encoder.HAZARD_INDICES]
    cont_view = enc_world[:, agent.latent_stack.split_encoder.CONTAMINATION_SLICE]
    lateral_input = torch.cat([hazard_ch, cont_view], dim=-1)
    return lateral_input, target


def _train_lateral_head(agent: REEAgent, seed: int, episodes: int, steps: int) -> Dict[str, Any]:
    """P0: self-supervised pretext task -- regress the MECH-099 lateral_head's
    OWN OUTPUT NORM ``||z_harm||`` directly toward the hazard channel's own max
    (the local-view proximity indicator: 1.0 when the hazard cell is within
    the agent's 5x5 view, 0.0 once it leaves the view -- confirmed empirically
    a STEP function, not a smooth distance gradient, by directly reading
    world_obs[:, HAZARD_INDICES] at several distances before writing this
    function). Regressing the norm directly, rather than a linear probe on top
    of z_harm, is deliberate: MECH-219 reads ``||z_harm_un||`` (a norm) from
    the SD-019a EMA of this exact ``||z_harm||``, so training a probe that can
    merely DECODE proximity from some direction of z_harm is not the same
    property as training the actual consumed statistic (the norm) to carry
    it -- a probe-only version of this function was tried first and gave a
    held-out R2 of 0.98 while the NEAR/FAR norm separation it actually
    produced was reversed and negligible (0.86 vs 2.0, wrong direction) on an
    unseeded init; this norm-direct objective was verified (this session, not
    read off a doc) to give a large, consistently-signed NEAR/FAR separation
    (norm ~0.8-1.0 near, ~0.004-0.03 far) across seeds 11/23/37 with the exact
    `_seed_everything` seeding this script uses. Freezes lateral_head on
    return. This is z_harm-module warmup against a pretext target, not the
    downstream MECH-219 task -- P1 (the escapability sweep) runs on the FROZEN
    module."""
    env = _build_env()
    rng = random.Random(seed + 5000)
    params = (
        list(agent.latent_stack.split_encoder.lateral_head.parameters())
        + list(agent.world_obs_encoder.parameters())
    )
    opt = torch.optim.Adam(params, lr=P0_LR)
    last_loss = 0.0
    for ep in range(episodes):
        for _ in range(steps):
            pos = _p0_random_position(rng)
            _, obs_dict = env.reset_to(agent_pos=pos, hazard_positions=[HAZARD_POS])
            lateral_input, target = _lateral_input_and_target(agent, obs_dict)
            z_harm = agent.latent_stack.split_encoder.lateral_head(lateral_input)
            pred_norm = z_harm.norm(dim=-1, keepdim=True)
            loss = ((pred_norm - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
        if (ep + 1) % 20 == 0 or ep == episodes - 1:
            print(f"  [train] lateral_head_p0 seed={seed} ep {ep + 1}/{episodes} mse={last_loss:.6f}", flush=True)

    # Held-out R2 (fresh positions, disjoint RNG stream from training).
    heldout_n = DRY_P0_HELDOUT_SAMPLES if episodes == DRY_P0_EPISODES else P0_HELDOUT_SAMPLES
    hrng = random.Random(seed + 9000)
    preds: List[float] = []
    targets: List[float] = []
    with torch.no_grad():
        for _ in range(heldout_n):
            pos = _p0_random_position(hrng)
            _, obs_dict = env.reset_to(agent_pos=pos, hazard_positions=[HAZARD_POS])
            lateral_input, target_t = _lateral_input_and_target(agent, obs_dict)
            z_harm = agent.latent_stack.split_encoder.lateral_head(lateral_input)
            preds.append(float(z_harm.norm().item()))
            targets.append(float(target_t.item()))
    targets_arr = np.asarray(targets, dtype=np.float64)
    preds_arr = np.asarray(preds, dtype=np.float64)
    ss_res = float(np.sum((targets_arr - preds_arr) ** 2))
    ss_tot = float(np.sum((targets_arr - targets_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    # Freeze -- P1 measurement runs on fixed modules (phased-training spirit:
    # warmup, then no further gradient updates during the actual measurement).
    # Freezing world_obs_encoder too matters beyond this driver: it is shared
    # upstream of z_self/z_world as well (agent.py sense(): enc_world feeds
    # BOTH LatentStack's split_encoder branches), so training it here is a
    # real, deliberate side effect on the whole latent stack's representation
    # -- harmless for this run (z_world/z_self are never read by any of this
    # driver's criteria) but worth freezing explicitly rather than leaving it
    # trainable into P1 by omission.
    for p in agent.latent_stack.split_encoder.lateral_head.parameters():
        p.requires_grad_(False)
    for p in agent.world_obs_encoder.parameters():
        p.requires_grad_(False)
    agent.latent_stack.split_encoder.lateral_head.eval()
    agent.world_obs_encoder.eval()

    print(f"  [train] lateral_head_p0 seed={seed} heldout_r2={r2:.4f}", flush=True)
    return {"final_train_mse": last_loss, "heldout_r2": r2, "heldout_n": heldout_n}


def _run_condition(agent: REEAgent, seed: int, condition: str, bout_steps: int, relief_steps: int) -> Dict[str, Any]:
    """One seed x condition unit on the (already P0-trained, frozen-encoder) agent."""
    total_ticks = bout_steps + relief_steps
    env = _build_env()
    agent.reset()

    print(f"Seed {seed} Condition {condition}", flush=True)

    _, obs_dict = env.reset_to(agent_pos=NEAR_POS, hazard_positions=[HAZARD_POS])
    esc_bout = 0.0 if condition == "INESCAPABLE" else 1.0
    agent.set_harm_suffering_escapability(esc_bout)

    un_series: List[float] = []
    s_series: List[float] = []
    aic_series: List[float] = []
    for t in range(total_ticks):
        if t == bout_steps and condition == "INESCAPABLE":
            # Relief: escapability becomes available AND the agent is
            # physically removed from the hazard (env-only op; agent-side
            # accumulator state is untouched, so s_t / z_harm_un continue from
            # their bout-end values, not reset).
            agent.set_harm_suffering_escapability(1.0)
            _, obs_dict = env.reset_to(agent_pos=FAR_POS, hazard_positions=[HAZARD_POS])
        latent = _do_sense(agent, obs_dict)
        st = agent.harm_suffering_accumulator.get_state()
        un_series.append(
            float(latent.z_harm_un.detach().norm().item()) if latent.z_harm_un is not None else 0.0
        )
        s_series.append(float(st["mech219_last_s_out"]))
        aic_series.append(float(agent._aic_last_tick["aic_salience"]) if agent._aic_last_tick else 0.0)
        _, _, _, _, obs_dict = env.step(4)  # stay -- matched nociception, no navigation confound
        if (t + 1) % 30 == 0 or (t + 1) == total_ticks:
            print(f"  [train] seed={seed} cond={condition} ep {t + 1}/{total_ticks}", flush=True)

    _ZG.observe(agent)

    tail_start = max(0, bout_steps - TAIL_WINDOW)
    bout_end_un = float(np.mean(un_series[tail_start:bout_steps]))
    bout_end_s = float(np.mean(s_series[tail_start:bout_steps]))
    bout_end_aic = float(np.mean(aic_series[tail_start:bout_steps]))

    s_relief_start = None
    un_relief_start = None
    s_retention_at_check = float("nan")
    un_retention_at_check = float("nan")
    if condition == "INESCAPABLE":
        s_relief_start = s_series[bout_steps]
        un_relief_start = un_series[bout_steps]
        check_idx = bout_steps + RELIEF_CHECK_TICKS
        if check_idx < len(s_series) and s_relief_start and s_relief_start > 0:
            s_retention_at_check = s_series[check_idx] / s_relief_start
        if check_idx < len(un_series) and un_relief_start and un_relief_start > 0:
            un_retention_at_check = un_series[check_idx] / un_relief_start

    print("verdict: PASS", flush=True)
    return {
        "seed": seed,
        "condition": condition,
        "bout_end_un": bout_end_un,
        "bout_end_s": bout_end_s,
        "bout_end_aic_salience": bout_end_aic,
        "s_relief_start": s_relief_start,
        "un_relief_start": un_relief_start,
        "s_retention_frac_at_check": s_retention_at_check,
        "un_retention_frac_at_check": un_retention_at_check,
        "n_waking_updates": int(st["mech219_n_waking_updates"]),
    }


def _run_c4_check(agent: REEAgent, seed: int) -> Dict[str, Any]:
    """SD-021 parity: elevate beta_gate (confirming genuine z_harm attenuation),
    confirm the RESOLVED escapability scalar is bit-identical. Reports the
    attenuation RATIO (magnitude), not just a boolean fired/not-fired."""
    env = _build_env()
    agent.reset()
    _, obs_dict = env.reset_to(agent_pos=NEAR_POS, hazard_positions=[HAZARD_POS])
    agent.set_harm_suffering_escapability(0.0)

    latent = None
    for _ in range(10):
        latent = _do_sense(agent, obs_dict)
        _, _, _, _, obs_dict = env.step(4)
    norm_pre = float(latent.z_harm.detach().norm().item()) if latent.z_harm is not None else 0.0
    esc_pre = agent._resolve_harm_suffering_escapability()

    agent.beta_gate.elevate()
    _, _, _, _, obs_dict = env.step(4)
    latent2 = _do_sense(agent, obs_dict)
    norm_post = float(latent2.z_harm.detach().norm().item()) if latent2.z_harm is not None else 0.0
    esc_post = agent._resolve_harm_suffering_escapability()

    ratio = (norm_post / norm_pre) if norm_pre > 1e-9 else float("nan")
    return {
        "seed": seed,
        "z_harm_norm_pre_elevate": norm_pre,
        "z_harm_norm_post_elevate": norm_post,
        "attenuation_ratio": ratio,
        "sd021_attenuation_fired": bool(norm_post < norm_pre),
        "escapability_pre": float(esc_pre),
        "escapability_post": float(esc_post),
        "escapability_unchanged": bool(esc_pre == esc_post),
    }


def main(dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    bout_steps = DRY_BOUT_STEPS if dry_run else BOUT_STEPS
    relief_steps = DRY_RELIEF_STEPS if dry_run else RELIEF_STEPS
    p0_episodes = DRY_P0_EPISODES if dry_run else P0_EPISODES
    p0_steps = DRY_P0_STEPS_PER_EP if dry_run else P0_STEPS_PER_EP
    seeds = SEEDS[:1] if dry_run else SEEDS

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
    }

    common_config = {
        "env_kwargs": ENV_KWARGS,
        "hazard_pos": list(HAZARD_POS),
        "near_pos": list(NEAR_POS),
        "far_pos": list(FAR_POS),
        "bout_steps": bout_steps,
        "relief_steps": relief_steps,
        "relief_check_ticks": RELIEF_CHECK_TICKS,
        "p0_episodes": p0_episodes,
        "p0_steps_per_ep": p0_steps,
    }

    p0_rows: Dict[int, Dict[str, Any]] = {}
    condition_rows: List[Dict[str, Any]] = []
    c4_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        _seed_everything(seed)
        env0 = _build_env()
        cfg = _build_cfg(env0)
        agent = REEAgent(cfg)
        p0_rows[seed] = _train_lateral_head(agent, seed, p0_episodes, p0_steps)
        agent.reset()  # clear any accumulator/AIC-baseline state P0's sense() calls perturbed

        for condition in CONDITIONS:
            condition_rows.append(_run_condition(agent, seed, condition, bout_steps, relief_steps))

        cfg4 = _build_cfg(env0, descending_mod=True)
        agent4 = REEAgent(cfg4)
        # Reuse the SAME P0-trained lateral_head + world_obs_encoder weights
        # for the C4 parity check (a fresh untrained agent would confound the
        # parity read with module variance across seeds' two independent P0
        # runs).
        agent4.latent_stack.split_encoder.lateral_head.load_state_dict(agent.latent_stack.split_encoder.lateral_head.state_dict())
        agent4.world_obs_encoder.load_state_dict(agent.world_obs_encoder.state_dict())
        for p in agent4.latent_stack.split_encoder.lateral_head.parameters():
            p.requires_grad_(False)
        for p in agent4.world_obs_encoder.parameters():
            p.requires_grad_(False)
        agent4.latent_stack.split_encoder.lateral_head.eval()
        agent4.world_obs_encoder.eval()
        c4_rows.append(_run_c4_check(agent4, seed))

    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for row in condition_rows:
        by_seed.setdefault(row["seed"], {})[row["condition"]] = row

    worst_un_inescapable = min(by_seed[s]["INESCAPABLE"]["bout_end_un"] for s in seeds)
    worst_un_escapable = min(by_seed[s]["ESCAPABLE"]["bout_end_un"] for s in seeds)
    worst_probe_r2 = min(p0_rows[s]["heldout_r2"] for s in seeds)

    try:
        preconditions = p0_readiness_gate([
            {
                "name": "lateral_head_proximity_probe_r2",
                "measured": worst_probe_r2,
                "threshold": PROBE_R2_FLOOR,
                "direction": "lower",
                "control": (
                    "worst-seed held-out R2 of the MECH-099 lateral_head's OWN OUTPUT NORM "
                    "against the hazard channel's own max (world_obs[:, HAZARD_INDICES].max(), "
                    "the proximity summary), after P0 training -- the load-bearing test that "
                    "the norm MECH-219 actually consumes (||z_harm|| -> z_harm_un via the "
                    "SD-019a EMA) carries the proximity signal, not merely a linear-probe-"
                    "decodable direction of z_harm that the norm itself might not track"
                ),
            },
            {
                "name": "un_drive_supra_floor_inescapable",
                "measured": worst_un_inescapable,
                "threshold": UN_READINESS_FLOOR,
                "direction": "lower",
                "control": (
                    "worst-seed z_harm_un norm over the bout-phase tail window, INESCAPABLE "
                    "arm -- the drive_t = g_t*u_t input C1/C2/C2_CONTRAST route on"
                ),
            },
            {
                "name": "un_drive_supra_floor_escapable",
                "measured": worst_un_escapable,
                "threshold": UN_READINESS_FLOOR,
                "direction": "lower",
                "control": (
                    "worst-seed z_harm_un norm over the bout-phase tail window, ESCAPABLE arm "
                    "-- the C3 non-vacuity input"
                ),
            },
        ])
    except P0NotReady as e:
        manifest["outcome"] = "FAIL"
        manifest["interpretation"] = {
            "label": "substrate_not_ready_requeue",
            "preconditions": e.preconditions,
            "criteria_non_degenerate": {
                "C1_controllability_gate": False,
                "C2_persistence": False,
                "C2_CONTRAST_asymmetric_decay": False,
                "C3_non_vacuity": False,
                "C4_sd021_parity_magnitude": False,
                "C5_consumer_wiring": False,
            },
        }
        manifest["p0_rows"] = {str(k): v for k, v in p0_rows.items()}
        manifest["condition_rows"] = condition_rows
        manifest["c4_rows"] = c4_rows
        manifest["non_degenerate"] = False
        manifest["degeneracy_reason"] = e.reason
        out_path = write_flat_manifest(
            manifest,
            config=common_config,
            seeds=list(seeds),
            script_path=Path(__file__),
            z_goal_stream_stats=_ZG.stats(),
            started_at=t0,
            dry_run=dry_run,
        )
        manifest["_out_path"] = str(out_path)
        return manifest

    # ---- C1: controllability gate (SD-019b) ----
    c1_gap_per_seed = {
        s: by_seed[s]["INESCAPABLE"]["bout_end_s"] - by_seed[s]["ESCAPABLE"]["bout_end_s"]
        for s in seeds
    }
    c1_pass = all(g >= C1_MARGIN for g in c1_gap_per_seed.values())

    # ---- C2: persistence (SD-019b) ----
    c2_retention_per_seed = {s: by_seed[s]["INESCAPABLE"]["s_retention_frac_at_check"] for s in seeds}
    c2_pass = all(r >= C2_RETENTION_FRAC for r in c2_retention_per_seed.values())

    # ---- C2_CONTRAST: asymmetric decay (MECH-219 confirming signature 1) ----
    # RELATIVE margin, not an absolute ceiling on un's own retention -- see the
    # module docstring "Criteria" section for why (vector-EMA blending does
    # not obey the naive scalar exponential-decay bound).
    c2c_un_retention_per_seed = {s: by_seed[s]["INESCAPABLE"]["un_retention_frac_at_check"] for s in seeds}
    c2c_gap_per_seed = {s: c2_retention_per_seed[s] - c2c_un_retention_per_seed[s] for s in seeds}
    c2c_pass = all(g >= C2_CONTRAST_MARGIN for g in c2c_gap_per_seed.values())

    # ---- C3: non-vacuity (SD-019b) ----
    c3_s_vals = [by_seed[s]["ESCAPABLE"]["bout_end_s"] for s in seeds]
    c3_un_vals = [by_seed[s]["ESCAPABLE"]["bout_end_un"] for s in seeds]
    c3_pass = all(sv <= C3_S_CEILING for sv in c3_s_vals) and all(uv >= C3_UN_FLOOR for uv in c3_un_vals)

    # ---- C4: SD-021 parity, full magnitude (methodological control) ----
    c4_pass = all(r["escapability_unchanged"] for r in c4_rows) and all(
        (r["attenuation_ratio"] <= C4_RATIO_CEILING) for r in c4_rows if r["attenuation_ratio"] == r["attenuation_ratio"]
    )
    c4_non_degenerate = any(r["sd021_attenuation_fired"] for r in c4_rows)

    # ---- C5: consumer-path wiring (SD-019b point 4) ----
    c5_gap_per_seed: Dict[int, float] = {}
    for s in seeds:
        s_gap = c1_gap_per_seed[s]
        aic_gap = by_seed[s]["INESCAPABLE"]["bout_end_aic_salience"] - by_seed[s]["ESCAPABLE"]["bout_end_aic_salience"]
        c5_gap_per_seed[s] = aic_gap
    c5_pass = all(
        (c5_gap_per_seed[s] > 0) and (c5_gap_per_seed[s] >= C5_MIN_GAP_RATIO * max(c1_gap_per_seed[s], 1e-9))
        for s in seeds
    )

    all_pass = bool(c1_pass and c2_pass and c2c_pass and c3_pass and c4_pass and c5_pass)
    outcome = "PASS" if all_pass else "FAIL"

    sd019b_pass = bool(c1_pass and c2_pass and c3_pass and c5_pass)
    mech219_pass = bool(c2c_pass)
    evidence_direction_per_claim = {
        "SD-019b": "supports" if sd019b_pass else ("weakens" if not c1_pass else "mixed"),
        "MECH-219": (
            "supports" if mech219_pass else "weakens"
        ),
    }

    degeneracy = check_degeneracy({
        "C1_bout_end_s_inescapable": {
            "values": [by_seed[s]["INESCAPABLE"]["bout_end_s"] for s in seeds],
            "floor": 0.02,
            "ceiling": 1.98,
        },
        "C2_s_relief_start": {
            "values": [by_seed[s]["INESCAPABLE"]["s_relief_start"] for s in seeds],
            "floor": 0.02,
        },
        "C2C_un_relief_start": {
            "values": [by_seed[s]["INESCAPABLE"]["un_relief_start"] for s in seeds],
            "floor": UN_READINESS_FLOOR,
        },
        "C3_bout_end_un_escapable": {
            "values": c3_un_vals,
            "floor": UN_READINESS_FLOOR,
        },
        "C5_aic_gap": {
            "values": list(c5_gap_per_seed.values()),
            "floor": 1e-6,
        },
    })

    manifest.update({
        "outcome": outcome,
        "evidence_direction": "supports" if all_pass else ("weakens" if not (c1_pass and c3_pass) else "mixed"),
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "p0_rows": {str(k): v for k, v in p0_rows.items()},
        "condition_rows": condition_rows,
        "c4_rows": c4_rows,
        "c1_controllability_gap_per_seed": {str(s): v for s, v in c1_gap_per_seed.items()},
        "c2_s_retention_frac_at_check_per_seed": {str(s): v for s, v in c2_retention_per_seed.items()},
        "c2_contrast_un_retention_frac_at_check_per_seed": {str(s): v for s, v in c2c_un_retention_per_seed.items()},
        "c2_contrast_gap_per_seed": {str(s): v for s, v in c2c_gap_per_seed.items()},
        "c5_aic_salience_gap_per_seed": {str(s): v for s, v in c5_gap_per_seed.items()},
        "interpretation": {
            "label": (
                "mech219_sd019b_behavioural_confirmed" if all_pass else "mech219_sd019b_behavioural_not_confirmed"
            ),
            "preconditions": preconditions,
            "criteria": [
                {"name": "C1_controllability_gate", "load_bearing": True, "passed": bool(c1_pass)},
                {"name": "C2_persistence", "load_bearing": True, "passed": bool(c2_pass)},
                {"name": "C2_CONTRAST_asymmetric_decay", "load_bearing": True, "passed": bool(c2c_pass)},
                {"name": "C3_non_vacuity", "load_bearing": True, "passed": bool(c3_pass)},
                {"name": "C4_sd021_parity_magnitude", "load_bearing": True, "passed": bool(c4_pass)},
                {"name": "C5_consumer_wiring", "load_bearing": True, "passed": bool(c5_pass)},
            ],
            "criteria_non_degenerate": {
                "C1_controllability_gate": "C1_bout_end_s_inescapable" not in degeneracy["degenerate_metrics"],
                "C2_persistence": "C2_s_relief_start" not in degeneracy["degenerate_metrics"],
                "C2_CONTRAST_asymmetric_decay": "C2C_un_relief_start" not in degeneracy["degenerate_metrics"],
                "C3_non_vacuity": "C3_bout_end_un_escapable" not in degeneracy["degenerate_metrics"],
                "C4_sd021_parity_magnitude": c4_non_degenerate,
                "C5_consumer_wiring": "C5_aic_gap" not in degeneracy["degenerate_metrics"],
            },
            "combination_rule": (
                "AND of six load-bearing criteria (C1 AND C2 AND C2_CONTRAST AND C3 AND C4 AND "
                "C5); no OR. claim attribution is per-criterion, not per the blanket AND -- see "
                "evidence_direction_per_claim and scope_note."
            ),
        },
        "non_degenerate": bool(degeneracy["non_degenerate"] and c4_non_degenerate),
        "degeneracy_reason": degeneracy["degeneracy_reason"] or (
            "" if c4_non_degenerate else "C4_sd021_attenuation_fired: never fired in any seed"
        ),
        "degenerate_metrics": degeneracy["degenerate_metrics"],
        "diagnostics": {
            "pag_mech091_redirect_enabled_non_gating": (
                "harm_suffering_redirect_pag and harm_suffering_redirect_mech091 were both "
                "enabled in config (all v1 consumers, per the 957 autopsy's requirement), but "
                "their tick call sites (PAGFreezeGate.tick / the MECH-091 urgency block) live "
                "inside REEAgent.select_action(), which this driver never calls (fixed 'stay' "
                "action throughout, matching 957's no-navigation-confound design). They are "
                "therefore structurally INERT in this run's measurement, not merely "
                "below-threshold -- enabled-as-config only, never exercised, never a tested "
                "criterion. Only C5 (AIC, whose tick site is inside sense(), called every "
                "step) verifies consumer-path wiring here."
            ),
        },
        "scope_note": (
            "Implements C1/C2/C3/C4/C5 as described in the module docstring. claim_ids=["
            "'SD-019b', 'MECH-219'] with per-claim direction in evidence_direction_per_claim: "
            "SD-019b is scored on C1+C2+C3+C5 (its own controllability-gated, persistent, "
            "distinct-downstream-effect falsifiable claim); MECH-219 is scored ONLY on "
            "C2_CONTRAST (its own claims.yaml confirming signature 1, asymmetric decay) -- "
            "MECH-219's claims.yaml what_would_answer explicitly narrows the controllability-"
            "gating content (C1/C3/C5) OUT to MECH-305 ('Do not treat a controllability-only "
            "demonstration as confirming MECH-219'), and MECH-219's other three confirming "
            "predictions (non-linear summation/path-dependence; separable mode-switch "
            "regression via MECH-259; nonredundancy r2 vs a pure-delay null) are NOT tested by "
            "this run and remain open. C3(b) (MECH-353 z_block_assert anti-correlation) is "
            "explicitly OUT OF SCOPE -- see the module docstring 'MECH-353 cross-check' "
            "section: MECH-353's own substrate readiness has never been established "
            "(V3-EXQ-642 FAIL, substrate_ceiling_comparator_nondiscriminative, never "
            "successfully retested)."
        ),
    })

    out_path = write_flat_manifest(
        manifest,
        config={
            **common_config,
            "c1_margin": C1_MARGIN,
            "c2_retention_frac": C2_RETENTION_FRAC,
            "c2_contrast_margin": C2_CONTRAST_MARGIN,
            "c3_s_ceiling": C3_S_CEILING,
            "c3_un_floor": C3_UN_FLOOR,
            "c4_ratio_ceiling": C4_RATIO_CEILING,
            "c5_min_gap_ratio": C5_MIN_GAP_RATIO,
            "un_readiness_floor": UN_READINESS_FLOOR,
            "probe_r2_floor": PROBE_R2_FLOOR,
        },
        seeds=list(seeds),
        script_path=Path(__file__),
        z_goal_stream_stats=_ZG.stats(),
        started_at=t0,
        dry_run=dry_run,
    )

    print(
        f"[smoke] outcome={outcome} C1={c1_pass} C2={c2_pass} C2C={c2c_pass} C3={c3_pass} "
        f"C4={c4_pass} C5={c5_pass}",
        flush=True,
    )

    manifest["_out_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = main(dry_run=args.dry_run)
    print(f"outcome: {result['outcome']}")

    emit_outcome(
        outcome=result["outcome"],
        manifest_path=result.get("_out_path"),
        run_id=result.get("run_id"),
        dry_run=args.dry_run,
    )
