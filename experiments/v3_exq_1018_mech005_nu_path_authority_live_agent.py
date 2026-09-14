"""V3-EXQ-1018: MECH-005 -- does the substrate's nu-analog modulate PATH AUTHORITY,
or only the RATE of deliberation? Live agent, ENDOGENOUS z_beta.

================================================================================
DO NOT QUEUE -- REFUSED 2026-09-11 (session queue-stranded-1018-1003-20260911)
================================================================================
Step 4.5 red-team (fable, claude-fable-5-1) returned BLOCKING, and its load-bearing
findings were INDEPENDENTLY RE-VERIFIED at source before being accepted.
FULL RECORD, with the reproduction one-liners:
  REE_assembly/evidence/planning/exq1026_mech005_nu_path_authority_refusal_20260911.md

F1 THE ENVIRONMENT IS NEVER STEPPED DURING MEASUREMENT. _measure calls env.reset()
   once (line ~379) and there is NO env.step anywhere in this file
   (grep -c "env.step" -> 0); obs_dict is assigned only at ~342 and ~379 and never
   reassigned. So all MEASURE_TICKS=900 ticks of every arm re-sense ONE FROZEN
   observation, and the selected action is never applied. This falsifies the
   docstring's own central premise below ("the high-arousal arm samples a DIFFERENT
   distribution of world states") -- there is no distribution, only one state.
   The arousal arms then differ ONLY in HOW MANY fresh selects they draw from one
   fixed per-seed sequence (~45 LO vs ~180 HI), so the recorded HI-LO delta is a
   SAMPLE-COUNT artifact: MEASURED, the k-th margin is exactly equal across arousal
   arms for 22/30 and 20/28 k, and truncating HI to LO's draw count collapses the
   delta from -0.00296/+0.01862 to -0.00036/-0.00131. C1 is therefore
   UNATTRIBUTABLE under EVERY outcome -- PASS records "supports" on an artifact,
   FAIL records "weakens" against MECH-005 on an instrument that never varied the
   quantity the claim is about. Null-hypothesis C1 pass rate ~10.5%.
F2 THE COVARIATE-OVERLAP GATE CERTIFIES ITS OWN SUBJECT. It exists solely to guard
   the confound F1 dissolves, but hazard_prox_mean is accumulated (line ~505) from
   the FROZEN obs_dict, so both arms read the same constant and the gap is ~2e-16.
   It cannot fail. (The max()->mean() fix recorded below treated a symptom of this.)
F3 TWO DIAGNOSTIC KEYS ARE NEVER WRITTEN. e3_score_decomp_enabled defaults to False
   (e3_selector.py:614) and this driver never sets it (the cited instrument
   V3-EXQ-785a does, twice). So urgency_applied / committed / commit_variance are
   absent: urgency_fidelity_max_err is nan and line ~698 coerces it to 0.0, which
   PASSES the 1e-6 ceiling -- the urgency-fidelity precondition is VACUOUSLY MET,
   certifying an instrument never read; and n_committed is 0 every run, so
   commit_channel_live is red by instrumentation, not by substrate.
F5 (note) clock.py:211-221 clamps t = min(1, max(0, ||z_beta||*scale)), so at
   scale 10 any ||z_beta|| >= 0.1 pins e3_steps at 5 and at 0.1 it sits at 18-19 --
   one e3_steps value per arm across 22-23 distinct ||z_beta|| values. At these two
   levels the "ENDOGENOUS z_beta" framing is operating a saturated clamp.
ALSO: C1_ABS_FLOOR=1e-3 is BINDING, not the anti-degeneracy guard the docstring
   claims -- c1_required = max(floor, 2*SD), so the uncalibrated floor binds whenever
   between-seed SD is small. Confirmed with stubbed cells: delta 0.0012 -> C1 PASS ->
   "supports", against a measured DV of only ~0.013-0.026.
ALSO: the id in this FILENAME is burned -- V3-EXQ-1018 belongs to the tracked
   experiments/v3_exq_1018_mech222_self_attribution_contamination.py (commit 5cd2882).
   A successor needs a fresh number. V3-EXQ-1026 was reserved for this and released.

F1 is NOT a narrow fix: stepping the env changes the DV's scale and variance and
invalidates every threshold below (all calibrated against the frozen-observation
smoke). A successor is a fresh /queue-experiment pass on a new EXQ number; the
required changes are enumerated in the refusal record above. The SCIENTIFIC QUESTION
REMAINS OPEN AND UNOCCUPIED -- MECH-005 still has 0 of 1013 manifests.

The design below is otherwise careful and mostly checklist-clean (SD-008 alpha_world,
SD-070 zworld_p0, seed-44 exclusion, per-arm gates, E3 latch cleared, truthful reuse
ineligibility, portable pre-sampling DV); read it as a starting point, not a warning.

=== THE CLAIM, AND THE ONE THING NOBODY HAS MEASURED ===
MECH-005 ("Path authority and interruptibility via norepinephrine-like control")
asserts that a norepinephrine-like signal nu does NOT select temporal depth but
modulates three things: (1) the AUTHORITY of the currently committed path, (2) how
INTERRUPTIBLE that path is, and (3) how strongly POST-COMMIT errors drive
restructuring. As of 2026-09-09 MECH-005 has ZERO experimental manifests
(0 of 1013 packs tag it) -- the proposal EVB-1381 flags exactly this:
missing_experimental_evidence + synthetic_signals_only.

The "synthetic_signals_only" half is precise and is this run's reason to exist.
The only adjacent evidence, V3-EXQ-505 (MECH-093 z_beta/precision dissociation,
direction=supports), injected a SYNTHETIC z_beta of a chosen norm directly into a
bare MultiRateClock -- no agent, no candidate set, no commitment latch. It therefore
verified the interpolation ARITHMETIC of clock.update_e3_rate_from_beta and nothing
downstream of it. This run is the live-agent counterpart: endogenous z_beta, real
candidates, real E3 selection.

=== WHERE nu LIVES IN THE V3 SUBSTRATE (read, not assumed) ===
Two DIFFERENT signals implement what MECH-005 attributes to ONE:

  arousal  ||z_beta||  -> clock.update_e3_rate_from_beta (MECH-093, agent.py:5920,
                          inside _e1_tick, unconditional -- no feature flag)
                       -> e3_steps_per_tick in [beta_rate_min_steps=5,
                          beta_rate_max_steps=20]. Higher arousal = E3 re-selects
                          MORE OFTEN = MECH-005's mechanism (2), interruptibility.

  urgency  ||z_harm_a|| * urgency_weight (SD-011, e3_selector.py:3796)
                       -> effective_threshold *= (1 + urgency_applied), and
                          committed = commit_variance < effective_threshold, so
                          urgency makes commitment MORE PERMISSIVE = MECH-005's
                          mechanism (1), commitment pressure.

MECH-005's mechanism (3), post-commit error salience, has NO substrate at all
(grep: no post-commit error weighting anywhere in ree_core). It is NOT tested here
and this run licenses no claim about it.

=== THE QUESTION THIS RUN ACTUALLY ASKS ===
Raising arousal makes E3 run more often. That much is arithmetic and is NOT the
question -- measuring it would be a tautology (it is what 505 already did).
The question is whether that rate change converts into a change in PATH AUTHORITY
PER DELIBERATION OPPORTUNITY:

  MECH-005 supported: nu genuinely modulates how tightly the system is bound to its
    current path, so the per-opportunity authority statistic MOVES.
  MECH-005 weakened as realized in V3: nu modulates deliberation FREQUENCY only;
    per-opportunity path authority is set by machinery nu never touches (the
    beta_gate latch, the commit-variance gate), and the authority statistic is FLAT.

=== THE DV, AND WHY IT IS UPSTREAM OF THE SAMPLER ===
DV = the NORMALIZED E3 score margin at each fresh selection, read off e3.last_scores
(e3_selector.py:3731, `scores.detach()`; REE is lower-is-better so the winner is
argmin):

    normalized_margin = (second_min - min) / (median - min)

i.e. the winner's lead over the runner-up, relative to a TYPICAL candidate's
disadvantage. The RAW margin and the full spread are recorded alongside it but are not
the criterion. Both denominators were tried against the smoke and the median won on
evidence: raw margins are of order 50-100 while the full max-min spread is ~26000,
because a single outlier candidate dominates max -- so a max-normalized DV is ~0.002
and its scale is set by that outlier rather than by the contest among plausible
candidates. The median gap is robust to it.

The normalized form is dimensionless and invariant under affine rescaling of the score
vector -- DESIRABLE here, because an arm-level score-scale shift is not path authority.
The manipulation changes WHICH world states get scored, which is not an affine rescale,
so it is not annihilated by that invariance.

=== HOW C1 IS GATED, AND WHY IT IS SCALE-FREE ===
C1 does NOT use a calibrated absolute threshold. Calibrating one would require knowing
the trained DV scale in advance, which is part of what this run measures -- and the
smoke demonstrated BOTH failure directions of guessing it: an absolute floor of 0.02 in
raw score units passed vacuously (raw deltas are ~49), while a normalized floor of 0.05
was structurally unreachable (normalized values are ~0.002-0.005 under max
normalization). Either would have been a pre-registered artifact.

C1 therefore gates on effect size relative to BETWEEN-SEED NOISE:

    PASS iff  |mean per-seed delta| >= max(C1_ABS_FLOOR, 2 x SD(per-seed deltas))
              AND every seed agrees on the sign of the delta

C1_ABS_FLOOR (1e-3) is only an anti-degeneracy guard against a literally-zero delta;
the SD term is the binding gate. Sign consistency across 3 seeds is required so a
single outlying seed cannot carry the verdict.

This is the SAME quantity the substrate itself calls path readiness --
BetaGate.should_admit_elevation(score_margin) gates commit entry on it against
commit_readiness_floor -- so it is the substrate's own operationalisation of
"how decisively is one path winning", i.e. path authority.

It is deterministic and PRE-SAMPLING. It is deliberately NOT the selected action:
torch.multinomial returns a different category on linux-x86_64 than on darwin-arm64
from a bit-identical probability tensor at the same seed (CLAUDE.md, measured
2026-07-20), and e3_selector has 7 multinomial call sites in the live selection
path. An action-derived DV would not be portable across machine classes; a margin
read off last_scores is.

=== DESIGN: TRAIN ONCE PER SEED, MEASURE FOUR ARMS ON THAT SUBSTRATE ===
BOTH manipulations are read LIVE at measurement time --
clock.beta_magnitude_scale is read inside update_e3_rate_from_beta on every E1
tick, and e3.config.urgency_weight is read inside select() on every selection
(the exact instrument V3-EXQ-785a verified to a fidelity of 2.8e-17). Neither is
baked in at construction.

So each seed trains ONE agent at a NEUTRAL beta_magnitude_scale=1.0, and all four
arms are then measured on that identical trained substrate, with the agent's
mutable surface captured and restored between arms
(experiments/_lib/probe_warmup.capture_agent_surface / restore_agent_surface).

That is not merely cheaper (3 trainings, not 12). It is the stronger design: the
arms differ ONLY in a measurement-phase knob, so an arm difference CANNOT be a
training-history confound. The cost is that the four cells of a seed share a
trained agent, so every cell is stamped reuse-INELIGIBLE
(extra_ineligible_reasons=["shared_trained_agent_across_arms"]) -- truthfully, not
to save effort.

2x2 factorial: arousal (beta_magnitude_scale) x urgency (exogenous), seeds 42/43/45.
Seed 44 is deliberately excluded (recurring early-death instability, EXQ-539-540 /
V3-EXQ-538a); 45 substitutes.

=== URGENCY IS NOT A FACTOR ON THE LOAD-BEARING DV (found by the smoke) ===
The normalized-margin DV is ARITHMETICALLY INVARIANT to the urgency factor.
urgency_applied enters ONLY as effective_threshold *= (1 + urgency_applied)
(e3_selector.py:3796) and thence only the `committed` boolean; it never enters the
candidate `scores` the margin is computed from. The 2026-09-09 smoke confirmed it
empirically -- the URGENCY_LO and URGENCY_HI cells were bit-identical on both margin
and fresh-select count within each arousal level.

So urgency is NOT a C1 factor. Its two levels are pooled as replicates for C1, and the
urgency contrast is reported ONLY through the separately-gated commitment-pressure
diagnostic, whose DV (committed_frac) the manipulation genuinely can reach. Keeping
the factor costs only measurement time (training is once per seed, shared across all
four arms), and it is what makes the dissociation reportable IF the commit channel
turns out to be live.

=== THE CONFOUND THIS RUN CANNOT REMOVE, STATED PLAINLY ===
Raising the E3 rate changes WHEN selection happens, so the high-arousal arm samples
a DIFFERENT distribution of world states than the low-arousal arm. A margin
difference could therefore reflect state sampling rather than path authority. This
is INHERENT to manipulating a rate and is not removable within this design.

It is handled by measurement, not by assertion: per-arm distributions of the state
covariates (hazard proximity from the learner's own 5x5 view -- never a global
oracle, the 732a confound -- and commit_variance) are recorded, and a precondition
requires the arms' covariate means to overlap within a pre-registered band. The
covariate is the MEAN over the 5x5 view, not its max: the smoke showed max() saturated
at exactly 1.000 in every arm, which would have made the check trivially "met" and
certified nothing. If they
do not overlap, C1 is reported as CONFOUNDED rather than as a verdict on MECH-005.

=== DV-SYMMETRY INVARIANCE (mandatory per-arm declaration) ===
Arousal arms: manipulation = beta_magnitude_scale, a positive scalar gain on
||z_beta|| feeding an interpolation to an integer tick period. DV = mean top1-top2
score margin over fresh selections. The DV's symmetry group is (a) permutation of
candidates -- a margin is a symmetric function of the score multiset, and (b)
addition of a broadcast constant to all candidate scores -- which cancels in a
difference. The manipulation is invariant under NEITHER in the relevant direction:
it does not permute candidates, and it does not add a per-candidate constant; it
changes which world states are scored at all. So the DV is not fixed by arithmetic.

Urgency arms: manipulation = urgency_weight -> effective_threshold multiplicatively.
The RECORDED effective_threshold IS an arithmetic image of the manipulation, which
is exactly why effective_threshold is NOT a criterion here. The gated diagnostic is
the REALIZED committed fraction, which depends on where commit_variance sits
relative to the shifted threshold -- an empirical fact, not an identity.

=== WHAT IS AND IS NOT PRE-REGISTERED AS A CRITERION ===
C1 (LOAD-BEARING) is the arousal->margin contrast. Its premise is empirically
CONFIRMED: a 2026-09-09 probe on this substrate measured e3_steps mean 18.05 at
beta_magnitude_scale=0.1 vs 5.00 at 10.0 (range 13.05) with fresh-select fraction
0.060 -> 0.207, so the manipulation demonstrably reaches the selection cadence.

The commitment-pressure half is a RECORDED, GATED DIAGNOSTIC and deliberately NOT a
PASS criterion. The same probe measured `committed == True` on 0 of 59 fresh selects
across all arousal levels on an untrained agent, and beta_gate elevated once and
never released. Whether training lifts the commit channel above its floor is
unresolved at authoring time, and pre-registering a criterion whose premise is
unconfirmed is precisely what produces a starved-not-falsified verdict. It is
therefore measured, gated by its own readiness precondition, and left for a
successor to promote once the channel is shown live. A below-floor commit channel
routes ONLY that diagnostic to substrate_not_ready_requeue and does NOT vacate C1
(the V3-EXQ-785 whole-run-AND defect; gates are per-arm via _lib/precondition_gate).

SLEEP DRIVER: not applicable (no sleep machinery enabled).
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments.pack_writer import write_flat_manifest, flat_readout  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng  # noqa: E402
from experiments._lib.allon_training import _train_all_on_agent  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.probe_warmup import (  # noqa: E402
    capture_agent_surface,
    restore_agent_surface,
)
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    evaluate_arm_gate,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
)

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_1018_mech005_nu_path_authority_live_agent"
CLAIM_IDS = ["MECH-005"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

SEEDS = [42, 43, 45]           # 44 excluded: recurring early-death instability
ALPHA_WORLD = 0.9              # SD-008: z_world fidelity; 0.3 default is a known root cause
STEPS_PER_EPISODE = 40
P0_EPISODES = 12
P1_EPISODES = 12
ZWORLD_P0_EPISODES = 8         # SD-070: without this z_world stays a random projection
# 900, not 600. The LO-arousal arm produces a fresh select roughly every 1/0.060 ticks
# (probe 2026-09-09), so 600 ticks would land ~36 fresh selects against a floor of 30 --
# tight enough that ordinary variation could gate out a well-formed run. 900 gives ~54.
MEASURE_TICKS = 900

AROUSAL_LEVELS = {"LO": 0.1, "HI": 10.0}     # clock.beta_magnitude_scale
URGENCY_LEVELS = {"LO": 0.04, "HI": 0.34}    # exogenous urgency_applied target (785a grid)

# ---- pre-registered thresholds (constants; never derived from this run's stats) ----
# C1 is deliberately SCALE-FREE: the binding gate is the effect size relative to
# BETWEEN-SEED NOISE, not an absolute threshold. Calibrating an absolute threshold
# would require knowing the trained DV scale in advance, which is exactly what this
# run measures -- and the smoke showed both failure directions of guessing it (a raw
# 0.02 floor passed vacuously; a normalized 0.05 floor was unreachable). C1_ABS_FLOOR
# is therefore only an ANTI-DEGENERACY guard against a literally-zero delta.
C1_ABS_FLOOR = 1e-3
C1_SD_MULT = 2.0           # |mean per-seed delta| must exceed 2 x SD of those deltas
C1_REQUIRE_SIGN_CONSISTENCY = True   # ... and every seed must agree on the direction
MIN_FRESH_SELECTS = 30     # per cell, for the margin mean to be worth reporting
MIN_E3_STEPS_SEPARATION = 3.0   # mean e3_steps(LO) - mean e3_steps(HI)
URGENCY_FIDELITY_MAX = 1e-6
COVARIATE_OVERLAP_MAX = 0.25    # max |hazard_prox mean difference| between arousal arms
COMMIT_CHANNEL_FLOOR = 10       # committed==True count for the commitment diagnostic to be live


def _arms() -> List[Dict[str, Any]]:
    out = []
    for a_id, a_val in AROUSAL_LEVELS.items():
        for u_id, u_val in URGENCY_LEVELS.items():
            out.append({
                "arm_id": "AROUSAL_%s_URGENCY_%s" % (a_id, u_id),
                "arousal_level": a_id,
                "beta_magnitude_scale": a_val,
                "urgency_level": u_id,
                "urgency_target": u_val,
            })
    return out


PRECONDITIONS = [
    PreconditionSpec(
        name="e3_rate_separation_between_arousal_arms",
        description=("mean e3_steps_per_tick at AROUSAL_LO minus at AROUSAL_HI; certifies "
                     "that the arousal manipulation reaches the E3 selection cadence. "
                     "CERTIFIES THE AROUSAL CHANNEL ONLY -- it says nothing about the "
                     "urgency channel, which carries its own fidelity precondition."),
        control="the two pre-registered arousal arms at a common seed; probe 2026-09-09 "
                "measured 18.05 vs 5.00 (separation 13.05) on an untrained agent",
        threshold=MIN_E3_STEPS_SEPARATION,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="fresh_selects_per_cell",
        description=("WORST cell's count of fresh E3 selections (latched ticks excluded). "
                     "The margin mean is denominated on this, so the worst cell governs."),
        control="a cell is 600 ticks; the LO-arousal probe yielded ~1 fresh select per 15 "
                "ticks, i.e. ~40 expected in the worst arm",
        threshold=MIN_FRESH_SELECTS,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="urgency_injection_fidelity",
        description=("max |realized urgency_applied - assigned target| over fresh selects. "
                     "CEILING: met when the error stays BELOW the bound."),
        control="V3-EXQ-785a verified this instrument to 2.8e-17 over 600 ticks",
        threshold=URGENCY_FIDELITY_MAX,
        direction="upper",
        kind="readiness",
    ),
    PreconditionSpec(
        name="arousal_arm_state_covariate_overlap",
        description=("|mean hazard-proximity(AROUSAL_HI) - mean hazard-proximity(AROUSAL_LO)|, "
                     "from the learner's own 5x5 view. CEILING: met when the arms sample "
                     "COMPARABLE world states, so a margin difference is not simply a "
                     "different state distribution. This is the stated inherent confound."),
        control="the two arousal arms measured on the SAME trained agent and the same env seed",
        threshold=COVARIATE_OVERLAP_MAX,
        direction="upper",
        kind="readiness",
    ),
]

# The commitment-pressure diagnostic carries its OWN gate, evaluated separately so a
# starved commit channel can never vacate C1.
COMMIT_PRECONDITION = PreconditionSpec(
    name="commit_channel_live",
    description=("count of fresh selects with committed==True in the cell. Below floor the "
                 "commitment-pressure DIAGNOSTIC is starved, not falsified, and routes to "
                 "substrate_not_ready_requeue for ITSELF ONLY."),
    control="probe 2026-09-09 measured 0 of 59 on an UNTRAINED agent; this run trains first",
    threshold=COMMIT_CHANNEL_FLOOR,
    direction="lower",
    kind="readiness",
)


def _build(seed: int):
    """Build env + agent at the NEUTRAL arousal gain used for training.

    reset_all_rng BEFORE constructing the agent: torch.nn.Module weight init draws
    from torch's OWN global RNG, so seeding numpy/random alone would leave the
    trained substrate non-reproducible across runs at the same seed.
    """
    reset_all_rng(seed)
    env = CausalGridWorldV2(use_proxy_fields=True, seed=seed, hazard_harm=0.5)
    _obs, obs_dict = env.reset()
    kw: Dict[str, Any] = dict(
        body_obs_dim=obs_dict["body_state"].shape[-1],
        world_obs_dim=obs_dict["world_state"].shape[-1],
        action_dim=env.action_dim,
        alpha_world=ALPHA_WORLD,
        # SD-011: without BOTH, z_harm_a is None and the urgency branch is skipped
        # entirely -- the exogenous injection would be a silent no-op, not a zero.
        use_harm_stream=True,
        use_affective_harm_stream=True,
    )
    cfg = REEConfig.from_dims(**kw)
    # beta_magnitude_scale is NOT plumbed through from_dims (verified 2026-09-09: it
    # appears only at its HeartbeatConfig definition), so passing it there would be
    # SILENTLY SWALLOWED. It must be set on the sub-config, which agent.py:442 then
    # hands to the clock.
    cfg.heartbeat.beta_magnitude_scale = 1.0
    agent = REEAgent(cfg)
    if agent.clock.beta_magnitude_scale != 1.0:
        raise RuntimeError("beta_magnitude_scale did not reach the clock")
    return agent, env, obs_dict, kw


def _urgency_signal(agent: REEAgent, latent):
    """The tensor E3 actually norms for urgency (SD-019a redirects to z_harm_un)."""
    sig = latent.z_harm_a
    if getattr(agent.config.latent, "use_harm_un", False) and latent.z_harm_un is not None:
        sig = latent.z_harm_un
    return sig


_ZG = ZGoalStreamAccumulator()


def _measure(agent: REEAgent, seed: int, arm: Dict[str, Any], n_ticks: int) -> Dict[str, Any]:
    """Measure ONE arm on an already-trained agent. Pure measurement: no gradients."""
    env = CausalGridWorldV2(use_proxy_fields=True, seed=seed, hazard_harm=0.5)
    _obs, obs_dict = env.reset()
    agent.eval()
    agent.clock.reset()
    agent.clock.beta_magnitude_scale = float(arm["beta_magnitude_scale"])

    margins: List[float] = []
    raw_margins: List[float] = []
    spreads: List[float] = []
    median_gaps: List[float] = []
    e3_steps: List[int] = []
    hazard_prox: List[float] = []
    hazard_prox_mean_view: List[float] = []
    commit_vars: List[float] = []
    eff_thresholds: List[float] = []
    fidelity_errs: List[float] = []
    n_fresh = 0
    n_latched = 0
    n_committed = 0
    n_elevations = 0
    n_releases = 0
    run_lens: List[int] = []
    run_len = 0
    elev_prev = False
    zbeta_norms: List[float] = []
    pred_errs: List[float] = []
    zw_prev = None
    act_prev = None

    print("Seed %d Condition %s" % (seed, arm["arm_id"]), flush=True)

    with torch.no_grad():
        for tick in range(n_ticks):
            latent = agent.sense(
                obs_dict["body_state"].unsqueeze(0),
                obs_dict["world_state"].unsqueeze(0),
                obs_harm=obs_dict.get("harm_obs"),
                obs_harm_a=obs_dict.get("harm_obs_a"),
                obs_harm_history=obs_dict.get("harm_history"),
            )
            zb = float(latent.z_beta.detach().norm(dim=-1).mean().item())
            zbeta_norms.append(zb)

            zw_cur = latent.z_world.detach()
            if zw_prev is not None and act_prev is not None:
                pred = agent.e2.world_forward(zw_prev, act_prev).detach()
                err = zw_cur - pred
                # The agent loop never calls update_running_variance(); the driver must,
                # or commit_variance stays at its init and the commit gate is meaningless.
                agent.e3.update_running_variance(err)
                pred_errs.append(float(err.norm(dim=-1).mean().item()))

            # EXOGENOUS urgency: urgency_applied = min(||sig|| * urgency_weight,
            # urgency_max), and urgency_weight is read live at select(), so solving for
            # the weight lands urgency_applied exactly on the arm's target.
            sig = _urgency_signal(agent, latent)
            sig_norm = float(sig.norm(dim=-1).mean().item()) if sig is not None else 0.0
            target = float(arm["urgency_target"])
            agent.e3.config.urgency_weight = (target / sig_norm) if sig_norm > 1e-9 else 0.0

            # Freshness marker. E3 select() runs only when the commitment latch is open;
            # without this clear a latched tick re-reads the PREVIOUS selection and the
            # row is a duplicate (the ~9x pseudo-replication defect in V3-EXQ-785).
            agent.e3.last_score_diagnostics = None
            agent.e3.last_scores = None

            ticks_d = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks_d["e1_tick"]
                else torch.zeros(1, agent.config.latent.world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks_d)
            action = agent.select_action(candidates, ticks_d, 1.0)

            e3_steps.append(int(agent.clock.e3_steps_per_tick))

            diag = agent.e3.last_score_diagnostics
            scores = agent.e3.last_scores
            if diag is None or scores is None:
                n_latched += 1
            else:
                sv = scores.detach().reshape(-1)
                if sv.numel() >= 2:
                    n_fresh += 1
                    srt, _ = torch.sort(sv)
                    # REE is lower-is-better: winner is argmin, margin = second - min.
                    raw = float((srt[1] - srt[0]).item())
                    spread = float((srt[-1] - srt[0]).item())
                    # MEDIAN-based normalizer, not max-based. The 2026-09-09 smoke
                    # measured a full spread of ~26000 against raw margins of ~50-100:
                    # a single outlier candidate dominates max, so a max-normalized DV
                    # is ~0.002 and its scale is set by the outlier rather than by the
                    # contest among plausible candidates. The median gap is robust to
                    # that and reads as "the winner's lead over the runner-up, relative
                    # to a typical candidate's disadvantage".
                    med = float(srt[srt.numel() // 2].item())
                    denom = med - float(srt[0].item())
                    raw_margins.append(raw)
                    spreads.append(spread)
                    # NORMALIZED margin is the load-bearing DV. The smoke measured raw
                    # margins of order 50-100 (E3 scores are not O(1)), so any absolute
                    # floor is either vacuous or arbitrary. Dividing by the candidate
                    # score spread makes the DV dimensionless and bounded in [0, 1]:
                    # "how decisively does the winner beat the runner-up, relative to how
                    # spread the whole candidate set is". It is invariant under affine
                    # rescaling of scores -- which is DESIRABLE here, since an arm-level
                    # score-scale shift is not path authority -- while the manipulation
                    # (which changes WHICH states get scored) is not an affine rescale
                    # and is therefore not annihilated by it.
                    if denom > 1e-9:
                        margins.append(raw / denom)
                    median_gaps.append(denom)
                    if bool(diag.get("committed", False)):
                        n_committed += 1
                    if "urgency_applied" in diag:
                        fidelity_errs.append(abs(float(diag["urgency_applied"]) - target))
                    commit_vars.append(float(diag.get("commit_variance", float("nan"))))
                    eff_thresholds.append(float(diag.get("effective_threshold", float("nan"))))
                    hz = obs_dict.get("hazard_field_view")
                    if hz is not None:
                        hv = hz.detach().cpu().numpy().astype(float).reshape(-1)
                        # max() saturates at 1.0 in every arm (measured in the 2026-09-09
                        # smoke), which would make the covariate check trivially "met" and
                        # certify nothing. mean() over the same 5x5 learner-observable view
                        # does not saturate, so the overlap test is the mean.
                        hazard_prox.append(float(hv.max()))
                        hazard_prox_mean_view.append(float(hv.mean()))
                else:
                    n_latched += 1

            ev = bool(agent.beta_gate.is_elevated)
            if ev and not elev_prev:
                n_elevations += 1
                run_len = 1
            elif ev and elev_prev:
                run_len += 1
            elif (not ev) and elev_prev:
                n_releases += 1
                run_lens.append(run_len)
                run_len = 0
            elev_prev = ev

            zw_prev = zw_cur
            act_prev = action
            agent._step_count += 1

    # Read the z_goal counters AFTER stepping. Absence of this block in the manifest
    # always means "unmeasured", never "measured zero" -- a dead z_goal stream would
    # otherwise stay invisible (V3-EXQ-626 / V3-EXQ-830).
    _ZG.observe(agent)

    def _m(x):
        return float(np.mean(x)) if len(x) else float("nan")

    corr_zbeta_prederr = float("nan")
    if len(pred_errs) >= 10:
        n = min(len(pred_errs), len(zbeta_norms) - 1)
        if n >= 10:
            a = np.asarray(zbeta_norms[1:n + 1], dtype=float)
            b = np.asarray(pred_errs[:n], dtype=float)
            if a.std() > 1e-12 and b.std() > 1e-12:
                corr_zbeta_prederr = float(np.corrcoef(a, b)[0, 1])

    return {
        "arm_id": arm["arm_id"],
        "seed": seed,
        "arousal_level": arm["arousal_level"],
        "beta_magnitude_scale": arm["beta_magnitude_scale"],
        "urgency_level": arm["urgency_level"],
        "urgency_target": arm["urgency_target"],
        # --- the load-bearing DV ---
        "margin_mean": _m(margins),
        "margin_std": float(np.std(margins)) if margins else float("nan"),
        "margin_min": float(np.min(margins)) if margins else float("nan"),
        "margin_max": float(np.max(margins)) if margins else float("nan"),
        "per_select_margins": [float(x) for x in margins],
        # --- rate (recorded, NOT load-bearing: it is the arithmetic image) ---
        "e3_steps_mean": _m(e3_steps),
        "e3_steps_min": int(np.min(e3_steps)) if e3_steps else -1,
        "e3_steps_max": int(np.max(e3_steps)) if e3_steps else -1,
        "n_fresh_selects": n_fresh,
        "n_latched_ticks": n_latched,
        "fresh_frac": (n_fresh / (n_fresh + n_latched)) if (n_fresh + n_latched) else float("nan"),
        # --- commitment-pressure diagnostic (gated separately) ---
        "n_committed": n_committed,
        "committed_frac": (n_committed / n_fresh) if n_fresh else float("nan"),
        "commit_variance_mean": _m(commit_vars),
        "effective_threshold_mean": _m(eff_thresholds),
        "n_beta_elevations": n_elevations,
        "n_beta_releases": n_releases,
        "mean_committed_run_len": _m(run_lens),
        "committed_run_lens": [int(x) for x in run_lens],
        # --- covariates + prohibition checks ---
        "hazard_prox_max_mean": _m(hazard_prox),
        "hazard_prox_mean": _m(hazard_prox_mean_view),
        "raw_margin_mean": _m(raw_margins),
        "score_spread_mean": _m(spreads),
        "score_median_gap_mean": _m(median_gaps),
        "zbeta_norm_mean": _m(zbeta_norms),
        "zbeta_norm_std": float(np.std(zbeta_norms)) if zbeta_norms else float("nan"),
        "corr_zbeta_prediction_error": corr_zbeta_prederr,
        "urgency_fidelity_max_err": (float(np.max(fidelity_errs))
                                     if fidelity_errs else float("nan")),
    }


def _worst_cell(rows, key, mode="min"):
    """Return (worst value, offending cell id) -- the quantifier the `met` claim tests."""
    vals = [(r[key], "%s/seed%d" % (r["arm_id"], r["seed"])) for r in rows
            if r.get(key) is not None and np.isfinite(r[key])]
    if not vals:
        return float("nan"), None
    return (min(vals) if mode == "min" else max(vals))


def run_experiment(dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = SEEDS[:1] if dry_run else SEEDS
    n_ticks = 25 if dry_run else MEASURE_TICKS
    p0 = 1 if dry_run else P0_EPISODES
    p1 = 1 if dry_run else P1_EPISODES
    zp0 = 1 if dry_run else ZWORLD_P0_EPISODES
    arms = _arms()

    # Design-time proof BEFORE any compute: refuse a gate no arm could satisfy.
    arm_ctxs = [dict(a, n_ticks=n_ticks) for a in arms]
    assert_no_structurally_unsatisfiable_gate(PRECONDITIONS, arm_ctxs)

    arm_results: List[Dict[str, Any]] = []
    full_config = {
        "env": {"cls": "CausalGridWorldV2", "use_proxy_fields": True, "hazard_harm": 0.5},
        "alpha_world": ALPHA_WORLD,
        "steps_per_episode": STEPS_PER_EPISODE,
        "p0_episodes": p0, "p1_episodes": p1, "zworld_p0_episodes": zp0,
        "measure_ticks": n_ticks,
        "arousal_levels": AROUSAL_LEVELS, "urgency_levels": URGENCY_LEVELS,
        "thresholds": {
            "C1_ABS_FLOOR": C1_ABS_FLOOR, "C1_SD_MULT": C1_SD_MULT,
            "MIN_FRESH_SELECTS": MIN_FRESH_SELECTS,
            "MIN_E3_STEPS_SEPARATION": MIN_E3_STEPS_SEPARATION,
            "URGENCY_FIDELITY_MAX": URGENCY_FIDELITY_MAX,
            "COVARIATE_OVERLAP_MAX": COVARIATE_OVERLAP_MAX,
            "COMMIT_CHANNEL_FLOOR": COMMIT_CHANNEL_FLOOR,
        },
    }

    total_denom = p0 + p1
    for seed in seeds:
        agent, train_env, _obs_dict, _kw = _build(seed)
        zenv = CausalGridWorldV2(use_proxy_fields=True, seed=seed, hazard_harm=0.5)
        # zworld_p0_episodes is MANDATORY: it defaults to 0 in the shared helper, and
        # omitting it leaves split_encoder.world_encoder unstepped, so z_world stays a
        # frozen random projection with no error raised (SD-070).
        _train_all_on_agent(
            agent, train_env, seed,
            p0_episodes=p0, p1_episodes=p1,
            steps_per_episode=STEPS_PER_EPISODE,
            rung_id="mech005_nu", total_denominator=total_denom,
            zworld_p0_episodes=zp0, zworld_p0_env=zenv, zworld_p0_dry_run=dry_run,
        )
        for arm in arms:
            surface = capture_agent_surface(agent)
            with arm_cell(
                seed,
                config_slice=full_config,
                script_path=Path(__file__),
                config_slice_declared=True,
                # The four cells of a seed share ONE trained agent by design, so they are
                # not pure functions of (substrate, config, seed) and must never be reused.
                extra_ineligible_reasons=["shared_trained_agent_across_arms"],
            ) as cell:
                row = _measure(agent, seed, arm, n_ticks)
                cell.stamp(row)
            arm_results.append(row)
            restore_agent_surface(agent, surface)
            print("verdict: %s" % ("PASS" if np.isfinite(row["margin_mean"]) else "FAIL"),
                  flush=True)

    # ---------------- analysis ----------------
    def _by(arousal):
        return [r for r in arm_results if r["arousal_level"] == arousal]

    per_seed_delta = []
    for s in seeds:
        hi = [r["margin_mean"] for r in arm_results
              if r["seed"] == s and r["arousal_level"] == "HI" and np.isfinite(r["margin_mean"])]
        lo = [r["margin_mean"] for r in arm_results
              if r["seed"] == s and r["arousal_level"] == "LO" and np.isfinite(r["margin_mean"])]
        if hi and lo:
            per_seed_delta.append(float(np.mean(hi) - np.mean(lo)))

    delta_mean = float(np.mean(per_seed_delta)) if per_seed_delta else float("nan")
    delta_sd = float(np.std(per_seed_delta)) if len(per_seed_delta) > 1 else float("nan")
    # Effect-size gate: absolute floor AND a multiple of the SD of the per-seed delta.
    sd_bar = (C1_SD_MULT * delta_sd) if np.isfinite(delta_sd) else 0.0
    c1_required = max(C1_ABS_FLOOR, sd_bar)
    signs_agree = bool(
        len(per_seed_delta) >= 2
        and (all(d > 0 for d in per_seed_delta) or all(d < 0 for d in per_seed_delta)))
    c1_passed = bool(
        np.isfinite(delta_mean)
        and abs(delta_mean) >= c1_required
        and (signs_agree or not C1_REQUIRE_SIGN_CONSISTENCY))

    # ---------------- preconditions ----------------
    e3_lo = [r["e3_steps_mean"] for r in _by("LO") if np.isfinite(r["e3_steps_mean"])]
    e3_hi = [r["e3_steps_mean"] for r in _by("HI") if np.isfinite(r["e3_steps_mean"])]
    e3_sep = (float(np.mean(e3_lo) - np.mean(e3_hi)) if e3_lo and e3_hi else float("nan"))

    worst_fresh, worst_fresh_cell = _worst_cell(arm_results, "n_fresh_selects", "min")
    worst_fid, worst_fid_cell = _worst_cell(arm_results, "urgency_fidelity_max_err", "max")
    hz_lo = [r["hazard_prox_mean"] for r in _by("LO") if np.isfinite(r["hazard_prox_mean"])]
    hz_hi = [r["hazard_prox_mean"] for r in _by("HI") if np.isfinite(r["hazard_prox_mean"])]
    # hazard_prox_mean is the MEAN over the 5x5 learner view (max() saturates at 1.0).
    cov_gap = (abs(float(np.mean(hz_hi) - np.mean(hz_lo))) if hz_lo and hz_hi else float("nan"))

    measured = {
        "e3_rate_separation_between_arousal_arms": e3_sep,
        "fresh_selects_per_cell": float(worst_fresh),
        "urgency_injection_fidelity": float(worst_fid) if np.isfinite(worst_fid) else 0.0,
        "arousal_arm_state_covariate_overlap": cov_gap,
    }
    gate = evaluate_arm_gate("MECH005_NU_MAIN", {"arm_id": "MECH005_NU_MAIN"},
                             PRECONDITIONS, measured)
    agg = aggregate_arm_gates([gate])

    for p in gate.get("preconditions", []):
        if p.get("name") == "fresh_selects_per_cell":
            p["offending_cell"] = worst_fresh_cell
        if p.get("name") == "urgency_injection_fidelity":
            p["offending_cell"] = worst_fid_cell

    # Commitment diagnostic: its OWN gate. A starved commit channel routes ONLY itself.
    total_committed = int(sum(r["n_committed"] for r in arm_results))
    commit_gate = evaluate_arm_gate(
        "COMMITMENT_PRESSURE_DIAGNOSTIC", {"arm_id": "COMMITMENT_PRESSURE_DIAGNOSTIC"},
        [COMMIT_PRECONDITION], {"commit_channel_live": float(total_committed)})
    commit_live = bool(commit_gate.get("gate_green", False))

    gate_green = bool(agg.get("non_degenerate", False))
    if not gate_green:
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
        outcome = "FAIL"
    elif c1_passed:
        label = ("nu_modulates_path_authority_per_opportunity"
                 if delta_mean > 0 else "nu_modulates_path_authority_inverse")
        direction = "supports"
        outcome = "PASS"
    else:
        label = "nu_modulates_deliberation_rate_only"
        direction = "weakens"
        outcome = "FAIL"

    manifest: Dict[str, Any] = {
        "run_id": "%s_%s_v3" % (EXPERIMENT_TYPE, datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")),
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": direction,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": bool(dry_run),
        "arm_results": arm_results,
        "combination_rule": ("C1 is the ONLY load-bearing criterion. PASS iff the gate is "
                             "green AND |mean per-seed margin delta| >= max(C1_ABS_FLOOR, "
                             "C1_SD_MULT * SD(per-seed delta)). The commitment-pressure "
                             "diagnostic is gated separately and NEVER affects this."),
        "criteria": [
            {"name": "C1_arousal_shifts_path_authority_margin",
             "load_bearing": True, "passed": c1_passed,
             "delta_mean": delta_mean, "required": c1_required,
             "per_seed_signs_agree": signs_agree,
             "factor": "arousal ONLY -- pooled over urgency, see urgency_dv_invariance"},
        ],
        "urgency_dv_invariance": {
            "statement": ("The normalized-margin DV is ARITHMETICALLY INVARIANT to the "
                          "urgency factor. urgency_applied enters ONLY as "
                          "effective_threshold *= (1 + urgency_applied) "
                          "(e3_selector.py:3796) and thence only the `committed` boolean; "
                          "it never enters the candidate `scores` the margin is computed "
                          "from. Confirmed in the 2026-09-09 smoke: the URGENCY_LO and "
                          "URGENCY_HI cells were bit-identical on margin and fresh-select "
                          "count within each arousal level."),
            "consequence": ("Urgency is NOT a C1 factor. The two urgency levels are pooled "
                            "as replicates for C1, and the urgency contrast is reported "
                            "ONLY through the separately-gated commitment-pressure "
                            "diagnostic, whose DV (committed_frac) the manipulation CAN "
                            "reach."),
        },
        "criteria_non_degenerate": {
            "C1_arousal_shifts_path_authority_margin": bool(
                gate_green and len(per_seed_delta) >= 2
                and np.isfinite(delta_sd) and float(np.std(
                    [r["margin_mean"] for r in arm_results
                     if np.isfinite(r["margin_mean"])] or [0.0])) > 0.0),
        },
        "interpretation": {
            "label": label,
            "preconditions": agg.get("adjudication_preconditions",
                                     gate.get("preconditions", [])),
            "criteria_non_degenerate": {
                "C1_arousal_shifts_path_authority_margin": bool(gate_green
                                                                and len(per_seed_delta) >= 2),
            },
            "commitment_pressure_diagnostic": {
                "gate_green": commit_live,
                "route_if_starved": "substrate_not_ready_requeue (THIS DIAGNOSTIC ONLY)",
                "preconditions": commit_gate.get("preconditions", []),
                "total_committed": total_committed,
            },
            "notes": (
                "MECH-005 mechanism (3) (post-commit error salience) has no substrate and is "
                "NOT tested. A 'weakens' here is scoped to nu-as-realized-in-V3 (arousal -> "
                "E3 rate), not to MECH-005 as an architectural proposal. If the covariate "
                "overlap precondition failed, C1 is CONFOUNDED by state sampling, not a "
                "verdict."),
        },
        "per_arm_gate": agg,
        "diagnostics": {
            "per_seed_margin_delta": per_seed_delta,
            "delta_mean": delta_mean, "delta_sd": delta_sd,
            "per_seed_signs_agree": signs_agree,
            "commit_channel_live": commit_live,
        },
    }

    manifest["readout"] = flat_readout({
        "margin_delta_mean": delta_mean,
        "margin_delta_sd": delta_sd,
        "signs_agree": signs_agree,
        "c1_required": c1_required,
        "c1_passed": c1_passed,
        "e3_rate_separation": e3_sep,
        "worst_fresh_selects": float(worst_fresh),
        "covariate_gap": cov_gap,
        "total_committed": float(total_committed),
        "commit_channel_live": commit_live,
        "gate_green": gate_green,
        "n_cells": float(len(arm_results)),
    })

    manifest["_elapsed"] = time.perf_counter() - t0
    manifest["_config"] = full_config
    manifest["_seeds"] = seeds
    manifest["_t0"] = t0
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    result = run_experiment(args.dry_run)
    t0 = result.pop("_t0")
    cfg = result.pop("_config")
    seeds = result.pop("_seeds")
    result.pop("_elapsed", None)

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=cfg,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print("manifest: %s" % out_path, flush=True)
    print("outcome: %s  label: %s" % (result["outcome"],
                                      result["interpretation"]["label"]), flush=True)

    _raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_raw if _raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
