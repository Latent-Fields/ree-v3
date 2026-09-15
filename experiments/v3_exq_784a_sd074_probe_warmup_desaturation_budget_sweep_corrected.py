"""V3-EXQ-784a: SD-074 probe-warmup de-saturation budget sweep -- CORRECTED RE-RUN.

SUPERSEDES V3-EXQ-784. The question, the env, the seeds, the budgets and every
pre-registered threshold are UNCHANGED. What changed is the substrate the
measurement runs on, plus an in-run audit that proves the repair is live.

QUESTION (one sentence): does the SD-074 probe_warmup bring the E3 pre-commit
action-value distribution into a measurable dynamic range, and at what warmup budget?

WHY A CORRECTED RE-RUN EXISTS
-----------------------------
V3-EXQ-784 (PASS, 2026-07-18) was downgraded to PROVISIONAL by 2026-08-16 governance
(GFLAG-0036). Its headline informative_yield rests on a premise its own manifest
states and which was FALSE at the time: sd074_note asserts measure_action_mass is
non-destructive, so reading at one checkpoint leaves the agent bit-identical for
continued training. Measured on ree-cloud-5 2026-08-15: a probe read left 21
non-state_dict attributes changed, 17 surviving agent.reset(). The load-bearing ones
were agent._self_experience_buffer / _world_experience_buffer, which warmup_train
samples via agent.compute_prediction_loss() -- so EVAL-ROLLOUT DATA ENTERED THE E1
TRAINING POOL. Those buffers cap at 1000 while 784 read 1081-1401 env steps, i.e. FULL
DISPLACEMENT of the training pool by evaluation data. A controlled 2-seed replication
of 784's ladder showed all six post-first-read cells differing by 0.011-0.146 on
D_action_mass and TWO OF SIX FLIPPING THE SATURATION REGIME -- exactly the
classification 784's informative_yield headline counts.

The defect was in the INSTRUMENT, not the claim. It is now fixed: measure_action_mass
snapshots and restores the WHOLE plain-Python attribute surface via
capture_agent_surface() / restore_agent_surface() (experiments/_lib/probe_warmup.py,
2026-08-15), replacing the hand-listed four-name set, with
tests/contracts/test_probe_warmup_nondestructive.py requiring an EMPTY diff. This run
re-derives the yield on that fixed instrument.

784'S NUMBERS ARE PROVISIONAL CONTEXT AND ARE NOT A COMPARATOR HERE. They are recorded
under provisional_predecessor for the reader's orientation only. No criterion in this
run is evaluated against them, and no corrected aggregate yield exists anywhere yet --
the 2-seed post-fix replication established that cells MOVE, not what they move to.

WHAT IS DELIBERATELY NOT IMPORTED
---------------------------------
V3-EXQ-777a's "c1_robust" bar, (mean_sin - pstdev_sin) > MARGIN, is NOT used here and
must not be. It belongs to the MECH-063 orthogonal-control-axes lineage
(experiments/v3_exq_777a_*.py:742, v3_exq_777_*.py:504), and SD-074's own claims.yaml
notes list it under "Deliberately out of scope". SD-074's own load-bearing criterion is
yield-based -- the fraction of seeds whose D_action_mass_mean lands strictly inside the
(0.05, 0.95) band -- which is the target condition SD-074's functional_restatement
states verbatim. This run says NOTHING about MECH-063: no control-axis quantity is
measured, and both regulators are held OFF.

THREE CHANGES FROM V3-EXQ-784, EACH STATED SO A READER CAN AUDIT THEM
---------------------------------------------------------------------
(1) NON-DESTRUCTIVENESS IS AUDITED IN-RUN BY CONTENT, NOT ASSERTED. measure_action_mass
    returns a restore_report; 784 discarded it. Every cell here records it AND an
    independent before/after CONTENT DIGEST of the agent -- state_dict, every named
    buffer (including non-persistent ones), the three experience buffers by length AND
    tail content, and e3._running_variance. Any difference means the read did not leave
    the agent as it found it, so it may have leaked into the next training leg, which
    invalidates the ladder: that is a readiness PRECONDITION self-routing to
    substrate_not_ready_requeue, never to a verdict on the warmup.

    THE DIGEST IS NOT DECORATION -- `n_by_reference` ALONE WOULD NOT CATCH GFLAG-0036.
    `restore_report.n_by_reference` counts only attributes capture_agent_surface could
    not COPY (probe_warmup.py:319 unregistered nn.Module, :361 deepcopy raised). It says
    nothing about whether what WAS copied got restored. Buffer LENGTH is a weak witness
    for the same reason in the opposite direction: the three experience buffers are
    trimmed to 1000 (agent.py:5871-5873), so from roughly budget 10 onward they sit AT
    the cap and length is identical before and after no matter what happened to their
    contents -- which is precisely the full-displacement failure GFLAG-0036 recorded.
    Hashing the tail closes that. A passing contract test on a developer box is not
    evidence the repair held on the box that ran this.

    THE DIGEST WAS PROVEN SENSITIVE BEFORE QUEUING, so this gate does not certify its
    own subject. Measured on DLAPTOP 2026-09-15 against the live substrate: a genuine
    restoring read drifts 0 of 7 digest paths (negative control), while a simulated
    restore regression -- restore_agent_surface and agent.load_state_dict both stubbed
    to no-ops, which is the GFLAG-0036 shape -- drifts ALL 7 (state_dict, named_buffers,
    the four experience buffers, e3._running_variance). In that same regression
    `restore_report.n_by_reference` came back 0, i.e. an n_by_reference-only gate
    reports a contaminated ladder as CLEAN. That measurement is why the precondition
    routes on the composite violation count rather than on n_by_reference.

(2) RNG IS RE-PINNED AT EVERY CHECKPOINT BOUNDARY. measure_action_mass restores agent
    state but explicitly does NOT restore process-global RNG, and its own docstring
    says so: "two reads of the same restored agent do NOT return the same
    D_action_mass. A ladder that wants a controlled comparison must re-pin RNG at each
    stage boundary itself." 784 is exactly such a ladder and never discharged that
    obligation, so its within-seed budget contrast carried an uncontrolled stream
    offset. Here torch / numpy / random are re-pinned to a per-seed constant
    immediately before each checkpoint read.

    WHAT THIS DOES NOT PIN, STATED SO THE CONTRAST IS NOT OVER-READ: the ENV keeps its
    own np.random.Generator instance (causal_grid_world.py), and env.reset() draws from
    it every episode, so the layout sequence a checkpoint read sees depends on how far
    intervening training and earlier reads advanced that generator. The per-cell
    env_rng_state_digest records it so the residual is auditable rather than assumed
    away. So change (2) is a NOISE REDUCTION on the agent-side draws, not a guarantee
    that a single seed's regime flip is purely the warmup: it cannot manufacture
    de-saturation, it washes out in a 14-seed yield, and it touches no threshold.

(3) claim_ids TAGS SD-074 AND experiment_purpose IS "evidence". 784 ran with
    claim_ids=[] and purpose="diagnostic" to avoid weighting MECH-063 while the 777
    lineage was braked -- a correct call for MECH-063, but it left SD-074 itself with
    ZERO rows in claim_evidence.v1.json despite a completed run, and SD-074's own notes
    say "NOT VALIDATED YET -- status candidate, validation experiment pending". C1 here
    IS that pending validation: it is SD-074's target condition word for word. So this
    run weights SD-074 and nothing else. MECH-063 remains untagged and untested.

DESIGN: incremental warmup with checkpoint reads (NOT independent arms)
-----------------------------------------------------------------------
Per seed, ONE agent is trained incrementally and D_action_mass is read at each budget
checkpoint [0, 4, 10, 25], against ONE env object held for the whole seed. Deliberate,
and better than four independent arms for two reasons:
  (1) It is ~1.6x cheaper (25 episodes per seed, not 0+4+10+25 = 39).
  (2) It removes a confound: independent arms would compare budgets ACROSS
      differently-initialised agents, entangling a budget effect with a seed-init
      effect. Here every checkpoint is the SAME agent further along one trajectory.
The checkpoint read is only valid because the read is non-destructive -- which is
precisely the property 784 assumed and this run measures (change 1).

CONSEQUENCE FOR ARM REUSE: cells within a seed SHARE a mutable agent, so they are NOT
independent and every cell is stamped reuse-INELIGIBLE via extra_ineligible_reasons.
That is the correctness guard, not an oversight.

WHAT "BUDGET B" MEANS HERE, PRECISELY -- read this before comparing a yield at budget B
against any single-call warmup. The ladder reaches budget B in LEGS (0 -> 4 -> 10 -> 25,
i.e. deltas of 4, 6 and 15), and goal_pipeline_tier1.warmup_train constructs FRESH Adam
optimisers and FRESH empty E2-world-forward / harm-eval minibatch pools at every call
(goal_pipeline_tier1.py:530-540). So "25 episodes" here is three legs with three
optimiser-moment restarts and three pool refills, NOT one 25-episode warmup. The SD-074
consumer path that ships, probe_warmup.warm_agent, makes ONE warmup_train call
(probe_warmup.py:1323-1326), so a later warm_agent(num_episodes=25) could read
differently for reasons of leg structure rather than budget. E1 is unaffected (its
_world_experience_buffer persists on the agent across legs); the E2/harm pools refill
over the first BATCH_SIZE=32 steps of each leg. V3-EXQ-784 has the identical structure,
so the two runs stay comparable, and C1 still discriminates -- what this bounds is what
a PASS at budget B LICENSES about a single-call warmup of the same size. Recorded per
cell as warmup_legs / warmup_leg_episodes and run-level as optimizer_restarts_per_seed
so the qualification travels with the number.

THE BUDGET-0 CHECKPOINT IS THE POSITIVE CONTROL. It is an untrained agent read with the
same instrument, so it must REPRODUCE V3-EXQ-777a's saturation. If budget 0 came back
mostly unsaturated, the instrument or the env would differ from 777a and the whole
comparison would be void -- so that is a readiness PRECONDITION, not a result.

ACCEPTANCE (pre-registered, unchanged from V3-EXQ-784, not derived from this run)
--------------------------------------------------------------------------------
C1 (LOAD-BEARING): at some swept budget > 0, informative-seed yield (fraction of seeds
    with D_action_mass_mean STRICTLY inside (0.05, 0.95)) exceeds 0.5 -- a majority,
    matching SD-074's own target condition.
C2: that yield strictly exceeds V3-EXQ-777a's HEADROOM fraction 5/14 = 0.357 by a
    margin of 0.10.
C3: the budget-0 control reproduces heavy saturation (>= 0.5 of seeds saturated),
    confirming the instrument sees what 777a saw.

C1 CAN GENUINELY FAIL, and is not closed-form-true: 784's own budget-0 cell read 0.357,
below the 0.5 bar, on this same instrument family and these same seeds.

ON THE C2 COMPARATOR -- carried forward from 784 verbatim, because it is a correction
worth restating. The autopsy's headline figure is "informative yield 4 of 14 (28.6%)",
but that is NOT the right comparator. 777a's "informative" required BOTH non-saturation
AND an authority test (a norm_v_score effect floor): it recorded 5 of 14 seeds in
HEADROOM, of which only 4 also cleared authority. THIS run measures saturation ONLY and
computes no authority quantity at all, so its like-for-like comparator is the HEADROOM
fraction 0.357, not 0.286. Comparing against 0.286 would flatter this run by 0.071.

Env and seeds are IDENTICAL to V3-EXQ-777a and V3-EXQ-784 (size 8, 2 hazards, 3
resources; the same 14 seeds), so every figure is drawn from the same cells.

DV-SYMMETRY INVARIANCE. The single arm's DV is informative_yield -- a fraction over
seeds of a band-membership test on D_action_mass_mean, itself a SUM of pre-commit
softmax mass over non-noop candidates. The manipulation is warmup GRADIENT TRAINING,
which changes the E3 score vector non-uniformly across candidates. It is NOT invariant
under this DV's symmetries: a broadcast additive constant across candidates cancels in
the softmax and would leave D fixed, and a monotone rescaling preserves rank but NOT
softmax mass -- warmup does neither, it changes relative scores, which is exactly what
moves probability mass between the noop and non-noop partitions. Confirmed empirically
before queuing: a 3-episode warmup moved a live seed-11 read from D=0.9994 (ceiling) to
D=0.5517 (headroom), i.e. a regime flip the DV registers.

SELF-ROUTE: if the instrument cannot read D at all (cells starved of fresh E3
selections), if the budget-0 control fails to reproduce 777a's saturation, or if any
cell's read failed to restore the agent surface, this self-routes to
substrate_not_ready_requeue with evidence_direction non_contributory -- NEVER to a
substrate verdict. A starved or leaking instrument cannot falsify anything.

MECH-094: N/A -- waking-only gradient training, no simulation, no replay, no memory
write.

RED-TEAM (Step 4.5, fable): CONTESTED -- 2 findings verified at source and FIXED here.
(F4) the non-destructiveness gate as first drafted attested copyability and buffer
LENGTH, not restore fidelity, and would have passed a real restore regression -> replaced
by the content digest above, whose sensitivity is now measured (see change 1); the
precondition's `measured` and `met` were also testing different statistics and now share
one composite count. (F1) "budget B" is three warmup_train legs with optimiser restarts,
not one B-episode warmup -> qualified in the docstring and recorded per cell. Minor:
env RNG is not pinned (claim softened, state digested per cell); unmeasured cells vote
in both criteria (recorded, bars deliberately not moved); C1 implies C2 at n=14
(stated in combination_rule, C2 already non-load-bearing). Full disposition in the
V3-EXQ-784a queue entry note.

ASCII-only output (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import hashlib
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_REE_V3 = _THIS.parent.parent
if str(_REE_V3) not in sys.path:
    sys.path.insert(0, str(_REE_V3))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.probe_warmup import (  # noqa: E402
    D_SAT_HIGH,
    D_SAT_LOW,
    WarmupRecipe,
    measure_action_mass,
    reapply_candidate_capture,
    saturation_regime,
)
from experiments._lib.goal_pipeline_tier1 import warmup_train  # noqa: E402
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_784a_sd074_probe_warmup_desaturation_budget_sweep_corrected"
EXPERIMENT_PURPOSE = "evidence"
# SD-074 ONLY. See "THREE CHANGES" (3) in the docstring: 784 left this empty to avoid
# weighting MECH-063, and SD-074 consequently has zero evidence rows. MECH-063 is
# neither tagged nor tested here.
CLAIM_IDS: List[str] = ["SD-074"]
SUPERSEDES = "V3-EXQ-784"

# The config_slice-declaration lint resolves the slice EXPRESSION at the arm_cell call
# site; here (as in V3-EXQ-784) the slice is threaded in as a function PARAMETER of
# _run_seed, which the lint cannot follow, so it reports every module-level numeric
# constant the cell reads as undeclared. They ARE declared -- see _config_slice's "env",
# "schedule" and "probe" blocks, which this driver widened beyond 784's precisely so the
# caps and floors key the fingerprint. More decisively, the hazard the lint guards (a
# false cache HIT) is structurally impossible for this driver: every cell is stamped
# reuse-INELIGIBLE via extra_ineligible_reasons because the budget checkpoints share one
# incrementally-trained agent, so no consumer can ever reuse these cells.
CONFIG_SLICE_DECLARATION_EXEMPT = (
    "slice is passed through a function parameter the lint cannot resolve; the "
    "constants are declared in _config_slice's env/schedule/probe blocks, and every "
    "cell is reuse-ineligible (shared agent across budget checkpoints) so a false "
    "cache HIT cannot occur"
)

# ---- Pre-registered constants (fixed before the run; not derived post-hoc) ----

# Identical to V3-EXQ-777a and V3-EXQ-784 so the baseline is drawn from the same cells.
SEEDS = [11, 17, 23, 29, 37, 3, 5, 13, 19, 41, 53, 61, 71, 83]
ENV_SIZE = 8
ENV_HAZARDS = 2
ENV_RESOURCES = 3

# Budget checkpoints, cumulative along ONE training trajectory per seed. Unchanged from
# 784, which sized them from a measured cost: warmup_train timed at 780 s for 5 episodes
# x 300 steps = 0.52 s per env step (4 optimisers, batch 32, CPU). A sweep reaching to
# 150 episodes would have been ~91 h of fleet time.
BUDGET_CHECKPOINTS = [0, 4, 10, 25]

# TRAIN cost lever, unchanged from 784. 777a did NO training, so there is no training
# regime to match and nothing is confounded by shortening it.
TRAIN_STEPS_PER_EPISODE = 100

# READ conditions. Held at 777a's STEPS_PER_EPISODE = 300 DELIBERATELY: the
# de-saturation read is the measurement compared against 777a's headroom baseline, so
# its rollout conditions must match that run even though the training regime does not.
READ_STEPS_PER_EPISODE = 300

# De-saturation read at each checkpoint (read-only, non-destructive).
PROBE_SELECTS = 150
PROBE_MAX_ENV_STEPS = 4000
# max_episodes DERIVED from max_env_steps, not set independently. With
# max_episodes < max_env_steps the EPISODE cap binds first for any seed whose episodes
# are shorter than 10 steps, so the read silently spends a fraction of its step budget
# -- the V3-EXQ-779a seed-23 defect (835 of 2400 steps at ~7 steps/episode). Seed 23 is
# in THIS seed list and was floor-pinned with very short episodes in 777a, so the hazard
# is live here, not hypothetical. Setting them equal makes the STEP cap bind for every
# seed.
PROBE_MAX_EPISODES = PROBE_MAX_ENV_STEPS

# CHANGE (2): per-seed RNG pin re-applied immediately before EVERY checkpoint read, so
# within a seed each budget's read draws from the same stream position against the same
# env. measure_action_mass restores agent state but explicitly not process-global RNG;
# its docstring names this as the caller's obligation for a ladder design.
READ_RNG_SEED_BASE = 900000

# Verdict thresholds. UNCHANGED from V3-EXQ-784 -- a corrected re-run must not move its
# own bars, or the correction cannot be separated from a re-specification.
YIELD_MAJORITY = 0.5          # C1: strictly greater than -- "a majority"

# LIKE-FOR-LIKE BASELINE. 777a recorded 5 of 14 seeds in HEADROOM (0.357) but only 4 of
# 14 INFORMATIVE (0.286), because its informative test required BOTH non-saturation AND
# an authority (norm_v_score effect floor) check. THIS run measures saturation ONLY, so
# its like-for-like comparator is the 0.357 headroom fraction.
BASELINE_HEADROOM_777A = 5.0 / 14.0        # 0.3571 -- the comparator C2 actually uses
BASELINE_INFORMATIVE_777A = 4.0 / 14.0     # 0.2857 -- recorded for context, NOT compared
YIELD_MARGIN = 0.10           # C2: must beat the like-for-like baseline by this margin

# C3 / control precondition, expressed as a FLOOR on the SATURATED fraction rather than
# a ceiling on yield. Same statement, but a floor is what assert_anchor_reachable can
# guard, so the gate is proven reachable by 777a's own recorded cells before the run.
CONTROL_MIN_SATURATED_FRAC = 0.5

# 777a's recorded per-seed D_seed_mean, frozen as a literal from
# v3_exq_777a_..._20260718T101635Z_v3.json `per_seed[].D_seed_mean`. The
# known-degenerate positive control the readiness anchor scores with the SHIPPED
# predicate: 9 of 14 saturated = 0.643, which clears the 0.5 gate.
_777A_REFERENCE_D_MEANS = [
    0.99834, 0.497272, 0.009944, 0.959463, 0.882364, 0.90876, 0.000301,
    0.998541, 1.0, 0.995469, 0.148769, 1.0, 0.939666, 0.969998,
]

# V3-EXQ-784's recorded figures. CONTEXT ONLY -- no criterion is evaluated against
# these. Their measurement premise was false (GFLAG-0036), so they are provisional and
# a corrected aggregate yield does not exist anywhere yet.
_784_PROVISIONAL = {
    "run_id": "v3_exq_784_sd074_probe_warmup_desaturation_budget_sweep_20260718T222045Z_v3",
    "outcome_as_recorded": "PASS",
    "best_swept_budget": 25,
    "best_informative_yield": 0.7857142857142857,
    "budget0_informative_yield": 0.35714285714285715,
    "status": "provisional_measurement_premise_false",
    "governance_flag": "GFLAG-0036 (2026-08-16 governance, cycle cranky-driscoll-126a36)",
}

# Readiness floor: a cell must actually collect selections for its read to mean anything.
MIN_SELECTS_FOR_READ = 50
MIN_CELLS_READABLE_FRAC = 0.8

# CHANGE (1): every cell's read must have left the agent as it found it. The bound is
# on the COMPOSITE violation count per cell -- unprotectable attributes
# (restore_report.n_by_reference) PLUS actual before/after content differences
# (n_drifted_paths) PLUS the buffer-length witness -- so that the precondition's
# `measured` is the same statistic its `met` tests and the indexer's recompute cannot
# disagree with the author. Any non-zero value means the read may have leaked into the
# next training leg, which is the GFLAG-0036 defect itself.
MAX_NONDESTRUCTIVE_VIOLATIONS_PER_CELL = 0

DRY_RUN_SEEDS = [11, 17]
DRY_RUN_BUDGETS = [0, 2]
DRY_RUN_TRAIN_STEPS_PER_EPISODE = 30
DRY_RUN_READ_STEPS_PER_EPISODE = 40
DRY_RUN_SELECTS = 15


# --- Opt-in driver-owned env seed (default OFF) ---------------------------
# `CausalGridWorldV2` forwards to `CausalGridWorld.__init__`, whose only RNG is
# `self._rng = np.random.default_rng(seed)`. `np.random.default_rng(None)` takes OS
# entropy AT CONSTRUCTION and does NOT consume the numpy global RNG, so this driver's
# `np.random.seed(...)` never reaches its env. The defect is per CONSTRUCTION, not per
# process. Default None reproduces 784's env seeding exactly, which is what keeps the
# two runs comparable; a pinned run is NOT comparable to a landed one.
#
# NOTE this is orthogonal to CHANGE (2): re-pinning the global RNG before each read
# controls the READ's stochastic draws, and the env OBJECT is held for the whole seed,
# so the within-seed budget contrast is controlled either way.
_ENV_SEED_BASE = None                 # None = OS entropy (the landed default)
_ENV_SEED_STATE = {"n": 0}            # per-CONSTRUCTION counter, not a knob


def _next_env_seed():
    """Seed for the NEXT env construction, or None when the knob is unset."""
    if _ENV_SEED_BASE is None:
        return None
    idx = _ENV_SEED_STATE["n"]
    _ENV_SEED_STATE["n"] = idx + 1
    # stream 2 = driver-owned (the scaffold module reserves 0 for its curriculum builds
    # and 1 for its harm probe).
    return int(_ENV_SEED_BASE) * 1000000 + 2 * 100000 + idx


def _mk_env() -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=_next_env_seed(),
        size=ENV_SIZE, num_hazards=ENV_HAZARDS, num_resources=ENV_RESOURCES
    )


def _pin_read_rng(seed: int) -> int:
    """CHANGE (2). Re-pin every stochastic source the read draws from.

    Returns the pin actually applied so it reaches the manifest. Called immediately
    before each checkpoint read, with a value that depends ONLY on the seed -- not on
    the budget -- which is what makes the within-seed budget contrast controlled.
    """
    pin = READ_RNG_SEED_BASE + int(seed)
    torch.manual_seed(pin)
    np.random.seed(pin)
    random.seed(pin)
    return pin


def _build_config(env: CausalGridWorldV2) -> REEConfig:
    """Shared operating config. Matches V3-EXQ-777a's and V3-EXQ-784's settings exactly.

    Both regulators stay OFF: this run measures the DISTRIBUTION's dynamic range, not
    any regulator's effect on it. Turning them on would confound the de-saturation
    reading with the very modulation a later experiment wants to measure -- and would
    also put this run on ree_core/policy/tonic_vigor.py, which carries an open
    `corrupting` substrate_queue entry (MECH-320).
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )
    cfg.use_control_vector_logging = True
    cfg.hippocampal.use_action_class_scaffold_candidates = True
    cfg.use_tonic_vigor = False
    cfg.use_noise_floor = False
    return cfg


def _config_slice(
    train_spe: int, read_spe: int, budgets: List[int], selects: int
) -> Dict[str, Any]:
    """Fingerprint config slice: env + shared operating settings + the sweep schedule."""
    sl: Dict[str, Any] = {
        "env": {
            "size": ENV_SIZE,
            "num_hazards": ENV_HAZARDS,
            "num_resources": ENV_RESOURCES,
        },
        "shared": {
            "use_control_vector_logging": True,
            "use_action_class_scaffold_candidates": True,
            "use_tonic_vigor": False,
            "use_noise_floor": False,
        },
        "schedule": {
            "train_steps_per_episode": train_spe,
            "read_steps_per_episode": read_spe,
            "budget_checkpoints": list(budgets),
            "probe_selects": selects,
        },
        # The probe caps and read floors are readout-affecting: they decide how much of
        # its budget a read spends and whether a cell counts as readable at all. 784
        # left them out of its slice (validate_experiments config_slice-declaration
        # warning, 5 constants); declared here so the cell's hash cannot collide with a
        # cell computed under different caps.
        "probe": {
            "probe_max_env_steps": PROBE_MAX_ENV_STEPS,
            "probe_max_episodes": PROBE_MAX_EPISODES,
            "min_selects_for_read": MIN_SELECTS_FOR_READ,
            "max_nondestructive_violations_per_cell": MAX_NONDESTRUCTIVE_VIOLATIONS_PER_CELL,
        },
        # CHANGE (2) is readout-affecting -- it fixes the stream each read draws from --
        # so it MUST key the fingerprint, or a 784-style unpinned cell and a pinned cell
        # would share a hash.
        "read_rng_pin_base": READ_RNG_SEED_BASE,
    }
    if _ENV_SEED_BASE is not None:
        sl["env_seed_base"] = _ENV_SEED_BASE
    return sl


def _buffer_len(agent: Any, name: str) -> Optional[int]:
    """Length of one of the buffers GFLAG-0036 named, or None if absent."""
    try:
        buf = getattr(agent, name, None)
        return None if buf is None else int(len(buf))
    except Exception:
        return None


def _tensor_hash(value: Any) -> str:
    """Stable content hash of a tensor-ish value. Never raises."""
    try:
        if torch.is_tensor(value):
            return hashlib.sha256(
                value.detach().to("cpu").contiguous().numpy().tobytes()
            ).hexdigest()
        return hashlib.sha256(repr(value).encode("utf-8", "replace")).hexdigest()
    except Exception:
        return "unhashable"


def _buffer_digest(agent: Any, name: str) -> str:
    """Length AND tail content of one experience buffer.

    LENGTH ALONE IS A VACUOUS WITNESS from roughly budget 10 onward: agent.py:5871-5873
    trims these buffers to 1000, so once training has produced >= 1000 entries the
    length is 1000 before and after any read regardless of whether the contents were
    displaced -- which is exactly the GFLAG-0036 failure (a read of 1081-1401 env steps
    fully displacing a 1000-entry training pool). Hashing the tail is what makes the
    witness real.
    """
    try:
        buf = getattr(agent, name, None)
        if buf is None:
            return "absent"
        tail = list(buf)[-8:]
        h = hashlib.sha256(("len=%d|" % len(buf)).encode())
        for item in tail:
            h.update(_tensor_hash(item).encode())
        return h.hexdigest()
    except Exception:
        return "unreadable"


def _agent_content_digest(agent: Any) -> Dict[str, str]:
    """Content fingerprint of everything a read could plausibly perturb.

    Deliberately INDEPENDENT of probe_warmup's own restore_report: that counts what
    capture_agent_surface could not COPY (probe_warmup.py:319, :361) and therefore
    cannot speak to whether what WAS copied came back. This compares actual content
    either side of the read.

    Covers (a) every parameter and registered buffer via state_dict, (b) every NAMED
    buffer including non-persistent ones -- which state_dict omits and load_state_dict
    therefore cannot restore, and which is where the in-place three-factor plasticity /
    eligibility traces live (e3_selector.py:373-462), (c) the three experience buffers
    by length and tail content, and (d) e3._running_variance, the plain-Python float
    probe_warmup's own docstring records as drifting 0.001839 -> 0.001855 over a
    25-selection read.
    """
    out: Dict[str, str] = {}
    try:
        sd = agent.state_dict()
        h = hashlib.sha256()
        for key in sorted(sd):
            h.update(key.encode())
            h.update(_tensor_hash(sd[key]).encode())
        out["state_dict"] = h.hexdigest()
    except Exception:
        out["state_dict"] = "unreadable"
    try:
        h = hashlib.sha256()
        for key, val in sorted(agent.named_buffers(), key=lambda kv: kv[0]):
            h.update(key.encode())
            h.update(_tensor_hash(val).encode())
        out["named_buffers"] = h.hexdigest()
    except Exception:
        out["named_buffers"] = "unreadable"
    for name in (
        "_self_experience_buffer",
        "_world_experience_buffer",
        "_action_experience_buffer",
        "_e2_transition_buffer",
    ):
        out[name] = _buffer_digest(agent, name)
    try:
        out["e3._running_variance"] = repr(float(getattr(agent.e3, "_running_variance")))
    except Exception:
        out["e3._running_variance"] = "unreadable"
    return out


def _env_rng_digest(env: Any) -> str:
    """Digest of the ENV's own Generator state -- NOT pinned by _pin_read_rng.

    CausalGridWorld holds `self._rng = np.random.default_rng(seed)`, an instance the
    driver's global re-pin does not touch, and env.reset() draws from it every episode.
    Recorded so the residual uncontrolled factor in the within-seed budget contrast is
    auditable rather than assumed away. Not a gate -- it legitimately differs across
    checkpoints, because intervening training advanced it.
    """
    try:
        rng = getattr(env, "_rng", None)
        if rng is None:
            return "absent"
        return hashlib.sha256(
            repr(rng.bit_generator.state).encode("utf-8", "replace")
        ).hexdigest()[:32]
    except Exception:
        return "unreadable"


def _run_seed(
    seed: int,
    budgets: List[int],
    train_spe: int,
    read_spe: int,
    selects: int,
    config_slice: Dict[str, Any],
    zg: ZGoalStreamAccumulator,
) -> List[Dict[str, Any]]:
    """One seed: train ONE agent incrementally, reading D at each budget checkpoint."""
    rows: List[Dict[str, Any]] = []
    total_budget = max(budgets)
    trained_so_far = 0
    # F1 qualification: the deltas actually handed to warmup_train, each of which
    # restarts the Adam moments and the E2/harm minibatch pools.
    leg_episodes: List[int] = []

    # The whole seed trajectory is ONE fingerprinted cell-group. RNG is reset once at
    # entry; the checkpoints share the resulting agent, which is exactly why every row
    # is stamped reuse-ineligible below.
    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=_THIS,
        config_slice_declared=True,
        include_driver_script_in_hash=False,
        extra_ineligible_reasons=[
            "incremental_warmup_shared_agent_across_budget_checkpoints",
        ],
    ) as cell:
        env = _mk_env()
        cfg = _build_config(env)
        agent = REEAgent(cfg)
        captured = reapply_candidate_capture(agent)

        for budget in budgets:
            print(f"Seed {seed} Condition budget_{budget}", flush=True)
            delta = budget - trained_so_far
            if delta > 0:
                # Continue the SAME trajectory rather than restarting.
                warmup_train(
                    agent,
                    env,
                    num_episodes=delta,
                    steps_per_episode=train_spe,
                    label=f"seed={seed} budget={budget}",
                    progress_total_episodes=total_budget,
                )
                leg_episodes.append(int(delta))
                trained_so_far = budget

            # Emit the loop-bound progress line the runner parses. The denominator is
            # the loop bound (total_budget), never a hardcoded constant.
            print(
                f"  [train] seed={seed} budget={budget} "
                f"ep {max(trained_so_far, 1)}/{max(total_budget, 1)}",
                flush=True,
            )

            # CHANGE (1): content digest immediately before the read, so the after-value
            # is a like-for-like comparison. The buffer LENGTHS are kept alongside it as
            # a human-legible quantity, but the digest is what the precondition routes
            # on -- see _agent_content_digest for why length alone is vacuous at the cap.
            digest_before = _agent_content_digest(agent)
            self_buf_before = _buffer_len(agent, "_self_experience_buffer")
            world_buf_before = _buffer_len(agent, "_world_experience_buffer")

            # CHANGE (2): re-pin the stream this read draws from. Records the ENV's own
            # generator state too -- that one is NOT pinned (see _env_rng_digest).
            env_rng_before = _env_rng_digest(env)
            rng_pin = _pin_read_rng(seed)

            read = measure_action_mass(
                agent,
                env,
                seed=seed,
                n_selections=selects,
                max_env_steps=PROBE_MAX_ENV_STEPS,
                max_episodes=PROBE_MAX_EPISODES,
                steps_per_episode=read_spe,
                captured=captured,
                label=f"desat seed={seed} budget={budget}",
            )

            digest_after = _agent_content_digest(agent)
            self_buf_after = _buffer_len(agent, "_self_experience_buffer")
            world_buf_after = _buffer_len(agent, "_world_experience_buffer")
            restore_report = dict(read.get("restore_report") or {})
            n_by_reference = int(restore_report.get("n_by_reference", 0) or 0)
            drifted_paths = sorted(
                k for k, v in digest_before.items() if digest_after.get(k) != v
            )
            buffers_restored = bool(
                self_buf_before == self_buf_after and world_buf_before == world_buf_after
            )
            # ONE INTEGER, and it is the SAME statistic the precondition's `met` tests --
            # the indexer recomputes `met` from (measured, threshold), so a composite
            # `met` over a narrower `measured` would let a component failure recompute as
            # MET. Every component that can mean "this read was not non-destructive"
            # therefore folds into this count:
            #   - n_by_reference: an attribute capture_agent_surface could not protect;
            #   - len(drifted_paths): actual before/after CONTENT differences;
            #   - buffers_restored: the legible length witness, kept as one more vote.
            n_nondestructive_violations = int(
                n_by_reference
                + len(drifted_paths)
                + (0 if buffers_restored else 1)
            )
            nondestructive = bool(
                n_nondestructive_violations <= MAX_NONDESTRUCTIVE_VIOLATIONS_PER_CELL
            )

            d_mean = read["d_action_mass_mean"]
            regime = saturation_regime(d_mean)
            readable = int(read["n_selections"]) >= (
                MIN_SELECTS_FOR_READ if selects >= MIN_SELECTS_FOR_READ else 1
            )
            row = {
                "seed": seed,
                "budget": budget,
                "d_action_mass_mean": d_mean,
                "d_action_mass_std": read["d_action_mass_std"],
                "saturation_regime": regime,
                "informative": bool(regime == "headroom"),
                "n_selections": int(read["n_selections"]),
                "readable": bool(readable),
                "probe_stop_reason": read["stop_reason"],
                # Realised probe BUDGET, not just which cap fired.
                "n_probe_env_steps": int(read["n_env_steps"]),
                "n_probe_episodes": int(read["n_episodes"]),
                "probe_max_env_steps": int(read["max_env_steps"]),
                "probe_max_episodes": int(read["max_episodes"]),
                "probe_episode_cap_can_bind": bool(read["episode_cap_can_bind"]),
                "probe_floors_met": bool(read["floors_met"]),
                "warmup_episodes_cumulative": trained_so_far,
                # ---- CHANGE (1): the non-destructiveness audit, per cell ----
                "restore_report": restore_report,
                "n_by_reference": n_by_reference,
                "drifted_paths": drifted_paths,
                "n_drifted_paths": len(drifted_paths),
                "self_experience_buffer_len_before": self_buf_before,
                "self_experience_buffer_len_after": self_buf_after,
                "world_experience_buffer_len_before": world_buf_before,
                "world_experience_buffer_len_after": world_buf_after,
                "buffers_restored": buffers_restored,
                "n_nondestructive_violations": n_nondestructive_violations,
                "read_nondestructive": nondestructive,
                # ---- CHANGE (2): the pin applied, and what it does NOT pin ----
                "read_rng_pin": int(rng_pin),
                "env_rng_state_digest": env_rng_before,
                # ---- F1 qualification: which legs produced this cumulative budget ----
                "warmup_legs": len(leg_episodes),
                "warmup_leg_episodes": list(leg_episodes),
            }
            cell.stamp(row)
            rows.append(row)
            print(
                f"  budget={budget} D={('None' if d_mean is None else '%.4f' % d_mean)} "
                f"regime={regime} n={row['n_selections']} stop={row['probe_stop_reason']} "
                f"by_ref={n_by_reference} drifted={len(drifted_paths)} "
                f"violations={n_nondestructive_violations}",
                flush=True,
            )
            # One verdict line per (seed x condition) unit, as the runner expects.
            print(f"verdict: {'PASS' if row['informative'] else 'FAIL'}", flush=True)

        # Fold this seed's finished agent into the run-level z_goal tally, then drop it.
        zg.observe(agent)

    return rows


def _yield_at(rows: List[Dict[str, Any]], budget: int) -> Dict[str, Any]:
    at = [r for r in rows if r["budget"] == budget]
    n = len(at)
    inf = [r for r in at if r["informative"]]
    regimes = {"ceiling": 0, "headroom": 0, "floor": 0, "unmeasured": 0}
    for r in at:
        regimes[r["saturation_regime"]] = regimes.get(r["saturation_regime"], 0) + 1
    return {
        "budget": budget,
        "n_seeds": n,
        "n_informative": len(inf),
        "informative_yield": (len(inf) / n) if n else 0.0,
        "informative_seeds": sorted(r["seed"] for r in inf),
        "regimes": regimes,
        "d_means": {r["seed"]: r["d_action_mass_mean"] for r in at},
    }


def _cell_is_saturated(d_mean: Optional[float]) -> bool:
    """THE SHIPPED PREDICATE. Used both to score live cells and to score the frozen
    V3-EXQ-777a reference in the readiness anchor -- deliberately the same callable, so
    the anchor cannot drift from what the run actually measures."""
    return saturation_regime(d_mean) != "headroom"


def _flat_scalar(value: Any) -> Any:
    """Coerce for the flat readout block: bools -> int, non-finite -> dropped."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        f = float(value)
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return value
    return None


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()

    # READINESS ANCHOR (setup-time, fails loudly). Prove the control gate is REACHABLE
    # by a bit-perfect replication of the known-degenerate control before spending fleet
    # hours: score V3-EXQ-777a's own recorded per-seed D values with the shipped
    # predicate and confirm they clear CONTROL_MIN_SATURATED_FRAC. Without this, a
    # hand-written gate narrower than the state it anchors to would report met=false on
    # every run forever and mislabel an instrument-specification gap as a substrate
    # verdict (the V3-EXQ-778d failure mode).
    anchor = assert_anchor_reachable(
        anchor_name="budget0_control_reproduces_777a_saturation",
        reference_cells=_777A_REFERENCE_D_MEANS,
        score_fn=_cell_is_saturated,
        threshold=CONTROL_MIN_SATURATED_FRAC,
        reference_source=(
            "v3_exq_777a_mech063_orthogonal_control_axes_dissociation_"
            "20260718T101635Z_v3.json per_seed[].D_seed_mean (14 cells)"
        ),
    )
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    budgets = DRY_RUN_BUDGETS if dry_run else BUDGET_CHECKPOINTS
    train_spe = DRY_RUN_TRAIN_STEPS_PER_EPISODE if dry_run else TRAIN_STEPS_PER_EPISODE
    read_spe = DRY_RUN_READ_STEPS_PER_EPISODE if dry_run else READ_STEPS_PER_EPISODE
    selects = DRY_RUN_SELECTS if dry_run else PROBE_SELECTS
    cslice = _config_slice(train_spe, read_spe, budgets, selects)

    zg = ZGoalStreamAccumulator()
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        rows.extend(_run_seed(seed, budgets, train_spe, read_spe, selects, cslice, zg))

    by_budget = [_yield_at(rows, b) for b in budgets]
    control = next((b for b in by_budget if b["budget"] == 0), None)
    swept = [b for b in by_budget if b["budget"] > 0]
    best = max(swept, key=lambda b: b["informative_yield"]) if swept else None

    # ---- Readiness precondition: instrument collected enough selections -----------
    # SAME STATISTIC RULE: the load-bearing criterion C1 routes on informative YIELD, a
    # fraction of seeds whose D read is inside the band. The thing that can starve it is
    # a cell that collected too few fresh E3 selections to have a meaningful D at all --
    # so the readiness check asserts the FRACTION OF CELLS THAT ARE READABLE, the same
    # kind of quantity (a fraction over the same cell population), not a magnitude proxy.
    n_cells = len(rows)
    n_readable = sum(1 for r in rows if r["readable"])
    readable_frac = (n_readable / n_cells) if n_cells else 0.0
    instrument_ready = readable_frac >= MIN_CELLS_READABLE_FRAC

    # ---- CHANGE (1): non-destructiveness audit, run-level -------------------------
    # The WORST CELL, not the mean -- `met` is an all() claim, so the reported number
    # must be the extremum the indexer can recompute against (and the offending cell is
    # named beside it).
    leaking_cells = [
        {
            "seed": r["seed"],
            "budget": r["budget"],
            "n_nondestructive_violations": r["n_nondestructive_violations"],
            "n_by_reference": r["n_by_reference"],
            "by_reference": (r.get("restore_report") or {}).get("by_reference"),
            "drifted_paths": r["drifted_paths"],
            "buffers_restored": r["buffers_restored"],
            "self_experience_buffer_len_before": r["self_experience_buffer_len_before"],
            "self_experience_buffer_len_after": r["self_experience_buffer_len_after"],
        }
        for r in rows
        if not r["read_nondestructive"]
    ]
    worst_violations = max(
        (r["n_nondestructive_violations"] for r in rows), default=0
    )
    worst_violations_cell = next(
        (
            {"seed": r["seed"], "budget": r["budget"]}
            for r in rows
            if r["n_nondestructive_violations"] == worst_violations
        ),
        None,
    )
    all_nondestructive = bool(rows) and not leaking_cells
    nondestructiveness_audit = {
        "n_cells": len(rows),
        "n_cells_nondestructive": sum(1 for r in rows if r["read_nondestructive"]),
        "n_leaking_cells": len(leaking_cells),
        "leaking_cells": leaking_cells,
        "worst_n_nondestructive_violations": worst_violations,
        "worst_n_nondestructive_violations_cell": worst_violations_cell,
        "worst_n_by_reference": max((r["n_by_reference"] for r in rows), default=0),
        "worst_n_drifted_paths": max((r["n_drifted_paths"] for r in rows), default=0),
        "all_reads_nondestructive": all_nondestructive,
        "audit_note": (
            "This is the GFLAG-0036 repair, measured on the box that ran this rather "
            "than asserted from a contract test. THREE components, because no one of "
            "them is sufficient: (a) restore_report.n_by_reference counts attributes "
            "capture_agent_surface could not COPY (probe_warmup.py:319, :361) and is "
            "silent about whether what WAS copied came back; (b) n_drifted_paths is an "
            "independent before/after CONTENT comparison over state_dict, every named "
            "buffer including non-persistent ones, the experience buffers by length AND "
            "tail, and e3._running_variance -- this is the component that actually "
            "detects a restore regression; (c) the buffer-length witness, kept because "
            "it is legible, but note it goes vacuous once the buffers reach their "
            "1000-entry cap (agent.py:5871-5873), which is exactly the full-displacement "
            "regime GFLAG-0036 recorded. Any leaking cell means eval-rollout data may "
            "have entered the E1 training pool for the NEXT checkpoint, which is what "
            "made V3-EXQ-784 provisional."
        ),
    }

    # ---- Probe budget audit -----------------------------------------------------
    # Mirrors probe_warmup.saturation_summary()'s budget block, at CELL granularity.
    # It answers the question the readable-cells fraction cannot: a cell can clear
    # MIN_SELECTS_FOR_READ and STILL have been budget-limited. NOT a criterion and NOT a
    # precondition -- this run's budget is safe BY CONSTRUCTION (PROBE_MAX_EPISODES is
    # derived from PROBE_MAX_ENV_STEPS, so the episode cap cannot bind), which is why it
    # is an audit a reader can verify rather than a gate that would always pass.
    measured_cells = [r for r in rows if r["saturation_regime"] != "unmeasured"]
    starved_cells = [
        {"seed": r["seed"], "budget": r["budget"]}
        for r in measured_cells
        if not r["probe_floors_met"]
    ]
    cap_bind_cells = [
        {"seed": r["seed"], "budget": r["budget"]}
        for r in measured_cells
        if r["probe_episode_cap_can_bind"]
    ]
    env_steps_spent = [r["n_probe_env_steps"] for r in measured_cells]
    probe_budget = {
        "n_cells": len(rows),
        "n_cells_measured": len(measured_cells),
        "n_probe_starved": len(starved_cells),
        "probe_starved_cells": starved_cells,
        "n_probe_episode_cap_can_bind": len(cap_bind_cells),
        "probe_episode_cap_can_bind_cells": cap_bind_cells,
        "probe_budget_clean": bool(not starved_cells and not cap_bind_cells),
        "max_env_steps_spent": max(env_steps_spent) if env_steps_spent else None,
        "median_env_steps_spent": (
            statistics.median(env_steps_spent) if env_steps_spent else None
        ),
        "probe_budget_note": (
            "probe_budget_clean=False means at least one cell's de-saturation read was "
            "budget-limited (starved below its selection floor, or run under an episode "
            "cap that can bind before the step cap). Qualify the informative yield "
            "accordingly -- a starved read's saturation verdict is low-confidence."
        ),
    }

    # The budget-0 control must reproduce 777a's saturation, else the instrument or env
    # differs from the run this whole comparison is anchored on. Expressed as a FLOOR on
    # the saturated fraction so it is the same predicate the readiness anchor proved
    # reachable against 777a's recorded cells.
    control_yield = control["informative_yield"] if control else None
    control_saturated_frac = (1.0 - control_yield) if control_yield is not None else None
    control_reproduces = (
        control_saturated_frac is not None
        and control_saturated_frac >= CONTROL_MIN_SATURATED_FRAC
    )

    preconditions = [
        {
            "name": "desaturation_read_cells_readable_frac",
            "description": (
                "fraction of (seed x budget) cells whose de-saturation read collected at "
                "least MIN_SELECTS_FOR_READ fresh E3 selections -- the same fraction-over-"
                "cells statistic the load-bearing yield criterion routes on"
            ),
            "control": "all cells, including the untrained budget-0 control",
            "measured": round(readable_frac, 4),
            "threshold": MIN_CELLS_READABLE_FRAC,
            "direction": "lower",  # FLOOR: met when measured >= threshold
            "met": bool(instrument_ready),
        },
        {
            "name": "every_read_nondestructive",
            "description": (
                "WORST CELL's n_nondestructive_violations -- one integer summing, for "
                "that cell's read, the attributes capture_agent_surface could not protect "
                "(restore_report.n_by_reference), the before/after CONTENT digest paths "
                "that actually differed (n_drifted_paths, over state_dict + named_buffers "
                "+ the experience buffers by length and tail + e3._running_variance), and "
                "the buffer-length witness. This is the GFLAG-0036 premise V3-EXQ-784 "
                "asserted and did not measure; any non-zero value means a read may have "
                "fed eval-rollout data into the E1 training pool for the next checkpoint, "
                "invalidating the ladder. It is deliberately ONE composite count rather "
                "than three separate quantities so that this `measured` is exactly the "
                "statistic `met` tests -- the indexer recomputes met from "
                "(measured, threshold), and a composite met over a narrower measured "
                "would let a component failure recompute as MET."
            ),
            "control": (
                "every cell of the run, including the untrained budget-0 control; the "
                "content digest is computed independently of probe_warmup's own "
                "restore_report, so a regression in capture_agent_surface itself cannot "
                "read as clean"
            ),
            "measured": worst_violations,
            "threshold": MAX_NONDESTRUCTIVE_VIOLATIONS_PER_CELL,
            "direction": "upper",  # CEILING: met when measured <= threshold
            "met": bool(all_nondestructive),
        },
        {
            "name": "budget0_control_reproduces_777a_saturation",
            "description": (
                "the untrained (budget 0) checkpoint must show a MAJORITY of seeds "
                "saturated, reproducing V3-EXQ-777a's 9/14 = 0.643 saturated fraction on "
                "the same env and the same 14 seeds; if the untrained agent were already "
                "unsaturated, the instrument or env would differ from 777a and the whole "
                "comparison would be void. Proven reachable at setup by scoring 777a's "
                "own recorded per-seed D values with the shipped predicate (see "
                "readiness_anchor below)."
            ),
            "control": "budget-0 checkpoint = untrained agent, same env and seeds as 777a",
            "measured": (
                None if control_saturated_frac is None else round(control_saturated_frac, 4)
            ),
            "threshold": CONTROL_MIN_SATURATED_FRAC,
            "direction": "lower",  # FLOOR: met when the saturated fraction >= threshold
            "met": bool(control_reproduces),
        },
    ]

    # ---- Criteria ----------------------------------------------------------------
    c1 = bool(best is not None and best["informative_yield"] > YIELD_MAJORITY)
    c2 = bool(
        best is not None
        and best["informative_yield"] > (BASELINE_HEADROOM_777A + YIELD_MARGIN)
    )
    c3 = bool(control_reproduces)

    yields = [b["informative_yield"] for b in by_budget]
    c1_nondegenerate = bool(len(set(round(y, 6) for y in yields)) > 1)
    c2_nondegenerate = c1_nondegenerate
    c3_nondegenerate = bool(control is not None and control["n_seeds"] > 0)

    # ---- Verdict grid ------------------------------------------------------------
    # ORDER MATTERS. Every instrument failure is checked BEFORE any substantive
    # criterion, and each routes to non_contributory -- never to `weakens`. A control or
    # an instrument failing says nothing about whether the warmup de-saturates.
    if not instrument_ready:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "non_contributory"
        degeneracy_reason = (
            "instrument not ready: only %.3f of cells collected >= %d fresh E3 "
            "selections" % (readable_frac, MIN_SELECTS_FOR_READ)
        )
    elif not all_nondestructive:
        # The GFLAG-0036 defect is still live on this box. The ladder's later
        # checkpoints may be contaminated, so no yield figure from this run is usable.
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "non_contributory"
        degeneracy_reason = (
            "%d cell(s) failed the non-destructiveness audit (worst "
            "n_nondestructive_violations=%d); the GFLAG-0036 measurement premise does "
            "not hold on this box, so the incremental ladder is contaminated and no "
            "yield from this run is usable"
            % (len(leaking_cells), worst_violations)
        )
    elif not control_reproduces:
        # The anchor itself failed; this is an instrument/env mismatch, not a verdict on
        # the warmup.
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "non_contributory"
        degeneracy_reason = (
            "budget-0 control did not reproduce 777a's saturation (saturated_frac=%s < "
            "%.2f); the instrument or env differs from the run this comparison is "
            "anchored on" % (control_saturated_frac, CONTROL_MIN_SATURATED_FRAC)
        )
    elif c1 and c2:
        label = "warmup_desaturates_landscape"
        outcome = "PASS"
        direction = "supports"
        degeneracy_reason = None
    else:
        label = "warmup_insufficient_at_swept_budgets"
        outcome = "FAIL"
        direction = "weakens"
        degeneracy_reason = None

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"SD-074": direction},
        "outcome": outcome,
        "timestamp_utc": ts,
        "dry_run": bool(dry_run),
        "supersedes": SUPERSEDES,
        "substrate_under_test": "SD-074",
        "routed_by": "GFLAG-0036 (2026-08-16 governance) corrected re-run of V3-EXQ-784",
        "provisional_predecessor": dict(_784_PROVISIONAL),
        "baseline_comparator": {
            "run_id": (
                "v3_exq_777a_mech063_orthogonal_control_axes_dissociation_"
                "20260718T101635Z_v3"
            ),
            "headroom_yield": BASELINE_HEADROOM_777A,
            "informative_yield_reported_by_autopsy": BASELINE_INFORMATIVE_777A,
            "compared_against": "headroom_yield",
            "note": (
                "777a recorded 5 of 14 seeds in HEADROOM (0.357) but only 4 of 14 "
                "INFORMATIVE (0.286), because its informative test required BOTH "
                "non-saturation AND an authority (norm_v_score effect floor) check. This "
                "run measures saturation ONLY and computes no authority quantity, so the "
                "like-for-like comparator is the 0.357 headroom fraction. Comparing "
                "against 0.286 would flatter this run by 0.071. Same env and same 14 "
                "seeds either way. V3-EXQ-784's own figures are NOT a comparator here -- "
                "see provisional_predecessor."
            ),
        },
        "readiness_anchor": anchor,
        "d_sat_low": D_SAT_LOW,
        "d_sat_high": D_SAT_HIGH,
        "budget_checkpoints": budgets,
        "read_rng_pin_base": READ_RNG_SEED_BASE,
        "per_cell": rows,
        "per_budget": by_budget,
        "probe_budget": probe_budget,
        "nondestructiveness_audit": nondestructiveness_audit,
        "best_swept_budget": best,
        "control_budget0": control,
        "criteria": [
            {
                "name": "C1_majority_informative_at_some_budget",
                "load_bearing": True,
                "passed": c1,
                "threshold": YIELD_MAJORITY,
                "measured": None if best is None else best["informative_yield"],
            },
            {
                "name": "C2_beats_777a_headroom_baseline_by_margin",
                "load_bearing": False,
                "passed": c2,
                "threshold": BASELINE_HEADROOM_777A + YIELD_MARGIN,
                "measured": None if best is None else best["informative_yield"],
            },
            {
                "name": "C3_control_reproduces_saturation",
                "load_bearing": False,
                "passed": c3,
                "threshold": CONTROL_MIN_SATURATED_FRAC,
                "measured": control_saturated_frac,
            },
        ],
        "combination_rule": (
            "PASS iff (instrument_ready AND all_reads_nondestructive AND C3) AND (C1 AND "
            "C2). The three instrument gates are checked FIRST and any of them failing "
            "routes to substrate_not_ready_requeue / non_contributory, never to weakens: "
            "a failed control or a leaking read says nothing about the warmup. Only C1 is "
            "load-bearing for SD-074; C2 is a like-for-like margin over 777a and C3 is the "
            "control gate. NOTE, so C2 is not read as an independent second test: at "
            "n=14 seeds C1 IMPLIES C2. C1 needs yield > 0.5, i.e. at least 8/14 = 0.5714; "
            "C2 needs yield > 0.4571, i.e. at least 7/14 = 0.5. So on the PASS branch "
            "'C1 and C2' is equivalent to C1 alone, and C2 can only differ from C1 at "
            "exactly 7/14, inside the FAIL branch. That is why C2 is declared "
            "load_bearing=false: it records the margin over 777a, it does not gate."
        ),
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": {
                "C1": c1_nondegenerate,
                "C2": c2_nondegenerate,
                "C3": c3_nondegenerate,
            },
            "null_meaning": (
                "A FAIL at label warmup_insufficient_at_swept_budgets means the swept "
                "budgets (max %d episodes) did not de-saturate a majority of seeds, on a "
                "correctly non-destructive instrument. That WEAKENS SD-074 as stated: the "
                "claim asserts the warmup reaches a majority. It does NOT mean warmup "
                "cannot work at all -- the honest next question is whether a larger "
                "budget, or a different training signal, moves the yield. A FAIL at "
                "substrate_not_ready_requeue means something about the instrument was "
                "wrong and is non_contributory: it weakens nothing. Either way this run "
                "says NOTHING about MECH-063 -- no control-axis quantity is measured."
                % max(budgets)
            ),
        },
        "arm_results": rows,
        "sd074_note": (
            "Cells within a seed share ONE incrementally-trained agent, so every cell is "
            "stamped reuse-INELIGIBLE (incremental_warmup_shared_agent_across_budget_"
            "checkpoints). The checkpoint design is only valid if measure_action_mass is "
            "non-destructive. V3-EXQ-784 ASSERTED that and was wrong (GFLAG-0036); this "
            "run MEASURES it per cell and refuses to report a yield if it does not hold."
        ),
    }

    # Non-degeneracy net (applies on any purpose). Two ways this run can be degenerate:
    # every cell reading the same D, or an instrument gate failing (which makes the yield
    # figure unusable rather than merely unimpressive).
    d_vals = [r["d_action_mass_mean"] for r in rows if r["d_action_mass_mean"] is not None]
    if len(d_vals) > 1 and statistics.pstdev(d_vals) <= 1e-9:
        manifest["non_degenerate"] = False
        manifest["degeneracy_reason"] = (
            "every cell returned an identical D_action_mass_mean; the budget sweep "
            "discriminated nothing"
        )
    elif degeneracy_reason is not None:
        manifest["non_degenerate"] = False
        manifest["degeneracy_reason"] = degeneracy_reason

    # ---- Flat scalar readout (machine-readable verdict projection) ----------------
    readout: Dict[str, Any] = {
        "best_swept_budget": None if best is None else best["budget"],
        "best_informative_yield": None if best is None else best["informative_yield"],
        "control_budget0_informative_yield": control_yield,
        "control_budget0_saturated_frac": control_saturated_frac,
        "readable_cells_frac": readable_frac,
        "worst_n_nondestructive_violations": worst_violations,
        "worst_n_by_reference": max((r["n_by_reference"] for r in rows), default=0),
        "worst_n_drifted_paths": max((r["n_drifted_paths"] for r in rows), default=0),
        "n_leaking_cells": len(leaking_cells),
        "n_cells": len(rows),
        # An UNMEASURED cell still votes: _yield_at counts it non-informative (so it
        # pulls C1 down) and _cell_is_saturated counts it saturated (so it pulls C3 UP,
        # toward met). The readable-cells precondition bounds how many there can be
        # (>= 0.8 of 56 cells readable) but is pooled over all budgets, while C1 and C3
        # are per-budget over 14 -- so these counts are recorded to make the residual
        # auditable. Deliberately NOT used to re-denominate either criterion: both bars
        # are pre-registered and carried unchanged from V3-EXQ-784.
        "n_unmeasured_cells": len(rows) - len(measured_cells),
        "control_budget0_n_unmeasured": (
            0 if control is None else int(control["regimes"].get("unmeasured", 0))
        ),
        "optimizer_restarts_per_seed": max(
            (r["warmup_legs"] for r in rows), default=0
        ),
        "c1_majority_informative": c1,
        "c2_beats_777a_headroom": c2,
        "c3_control_reproduces": c3,
        "all_reads_nondestructive": all_nondestructive,
        "instrument_ready": instrument_ready,
        "yield_majority_threshold": YIELD_MAJORITY,
        "baseline_headroom_777a": BASELINE_HEADROOM_777A,
        "c2_threshold": BASELINE_HEADROOM_777A + YIELD_MARGIN,
        "control_min_saturated_frac": CONTROL_MIN_SATURATED_FRAC,
    }
    for b in by_budget:
        readout["informative_yield_budget_%d" % b["budget"]] = b["informative_yield"]
    manifest["readout"] = {
        k: _flat_scalar(v) for k, v in readout.items() if _flat_scalar(v) is not None
    }

    full_config = {
        "env": {"size": ENV_SIZE, "num_hazards": ENV_HAZARDS, "num_resources": ENV_RESOURCES},
        "budget_checkpoints": budgets,
        "train_steps_per_episode": train_spe,
        "read_steps_per_episode": read_spe,
        "probe_selects": selects,
        "probe_max_env_steps": PROBE_MAX_ENV_STEPS,
        "probe_max_episodes": PROBE_MAX_EPISODES,
        "read_rng_pin_base": READ_RNG_SEED_BASE,
        "thresholds": {
            "D_SAT_LOW": D_SAT_LOW,
            "D_SAT_HIGH": D_SAT_HIGH,
            "YIELD_MAJORITY": YIELD_MAJORITY,
            "BASELINE_HEADROOM_777A": BASELINE_HEADROOM_777A,
            "BASELINE_INFORMATIVE_777A": BASELINE_INFORMATIVE_777A,
            "YIELD_MARGIN": YIELD_MARGIN,
            "CONTROL_MIN_SATURATED_FRAC": CONTROL_MIN_SATURATED_FRAC,
            "MIN_SELECTS_FOR_READ": MIN_SELECTS_FOR_READ,
            "MIN_CELLS_READABLE_FRAC": MIN_CELLS_READABLE_FRAC,
            "MAX_NONDESTRUCTIVE_VIOLATIONS_PER_CELL": MAX_NONDESTRUCTIVE_VIOLATIONS_PER_CELL,
        },
        "warmup_recipe_reference": WarmupRecipe(
            num_episodes=max(budgets), steps_per_episode=train_spe, probe_selections=selects
        ).as_dict(),
    }

    # Multi-arm: stamp AFTER arm_results is assembled so substrate_hash HOISTS from the
    # per-cell fingerprints rather than being recomputed driver-inclusive.
    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=seeds,
        script_path=_THIS,
        started_at=t0,
        z_goal_stream_stats=zg.stats(),
    )
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--env-seed", type=int, default=None,
        help="Opt-in env-seed base. Omitted (the default) reproduces V3-EXQ-784's "
             "OS-entropy env seeding exactly, which is what keeps the two runs "
             "comparable. Set it and every env this run builds is deterministically "
             "seeded. A pinned run is NOT comparable to a landed one.",
    )
    args = ap.parse_args()
    _ENV_SEED_BASE = args.env_seed

    result = run_experiment(dry_run=args.dry_run)
    out_dir = _REE_V3.parent / "REE_assembly" / "evidence" / "experiments"
    # stamp=False: run_experiment already called stamp_recording_core AFTER arm_results
    # was assembled, so substrate_hash is hoisted from the per-cell fingerprints. Letting
    # the writer stamp again would recompute it driver-inclusive and mismatch the cells.
    result["env_seed_base"] = _ENV_SEED_BASE
    out_path = write_flat_manifest(
        result,
        out_dir,
        dry_run=args.dry_run,
        stamp=False,
    )

    print("")
    print("=" * 62)
    print(f"outcome: {result['outcome']}")
    print(f"label:   {result['interpretation']['label']}")
    print(f"direction (SD-074): {result['evidence_direction']}")
    for b in result["per_budget"]:
        print(
            "  budget %-4d yield %.3f (%d/%d) regimes=%s"
            % (
                b["budget"],
                b["informative_yield"],
                b["n_informative"],
                b["n_seeds"],
                b["regimes"],
            )
        )
    _aud = result["nondestructiveness_audit"]
    print(
        "  nondestructive: %d/%d cells, worst violations=%d "
        "(by_reference=%d drifted=%d)"
        % (
            _aud["n_cells_nondestructive"],
            _aud["n_cells"],
            _aud["worst_n_nondestructive_violations"],
            _aud["worst_n_by_reference"],
            _aud["worst_n_drifted_paths"],
        )
    )
    print(f"  777a headroom baseline: {BASELINE_HEADROOM_777A:.3f} (like-for-like)")
    print(
        "  V3-EXQ-784 (PROVISIONAL, not a comparator): best yield %.3f"
        % _784_PROVISIONAL["best_informative_yield"]
    )
    print(f"manifest: {out_path}")
    print("=" * 62)

    _raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_raw if _raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
