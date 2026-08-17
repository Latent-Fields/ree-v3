"""V3-EXQ-936: MECH-439 -- does the CONVERTING F->eligibility demotion reduce F's
committed-selection variance share? (the claim's own pre-registered falsifier)

THE QUESTION, AND WHY IT IS NOT A RE-LITIGATION
-----------------------------------------------
MECH-439 states that the primary harm/goal score F monopolises ~88-89% of E3
committed-selection variance (V3-EXQ-571: 0.886 baseline), and that modulatory
diversity channels therefore cannot convert to committed-action diversity
"unless F's variance share is itself rebalanced". Its registered falsifier is
explicit and two-sided:

  FALSIFYING -- a selection lever lifts committed-action-class entropy
    strict-above BOTH the collapsed-proposer and matched-noise controls on the
    pre-registered majority of seeds WITHOUT reducing F's E3 variance share.
  CONFIRMING -- persistent flat committed entropy under any tested diversity
    channel WHILE F's variance share stays > 0.85.

The FIRST half of the falsifying antecedent is already established and
user-confirmed. V3-EXQ-689i's C_PRIMARY ("on-selected entropy strictly above
BOTH controls") is TRUE, adjudicated 2026-07-24 as "Gate defect, science
upheld" -- the MECH-448 rank-preserving F->eligibility demotion CONVERTS. The
SECOND half -- whether F's variance share moved while it did -- has never been
measured on ANY arm. That single unmeasured quantity is what decides which way
MECH-439's own falsifier resolves, and it is what this experiment measures.

This is therefore NOT one of the 12 braked re-tests. Those all asked "does this
lever convert?" of the near-tie parametric family, learned cross-loop
arbitration, disinhibitory settling and ascending-gain arbitration. This asks a
DIFFERENT question with a DIFFERENT dependent variable (a variance ratio, not an
entropy) about a lever that ALREADY converts, on the substrate face those
autopsies explicitly routed TO.

RE-DERIVE BRAKE: RELEASED (cited per /queue-experiment Step 2.5b)
-----------------------------------------------------------------
MECH-439 carries 12 counted substrate_ceiling autopsies -- 689/689a, the
f-dominance-conversion cluster, 700-cluster/700b/700c/700d-708, 709, 710, 711,
713 and the 711-713 re-adjudication. The most recent (2026-07-20) names
`upstream_substrate: f_dominance_conversion_ceiling` and routes to
implement-substrate.

That substrate is now BUILT AND VALIDATED. `substrate_queue.json`'s
`f_dominance_conversion_ceiling` entry reads: MECH-448 lead lever
BUILT_VALIDATED_PROMOTED_provisional; MECH-449 Go/No-Go leg BUILT with falsifier
V3-EXQ-689g RAN PASS and PROMOTED provisional 2026-06-22; selection-face
conversion ceiling LIFTED on the GAP-A foraging substrate; CONVERSION ROUTE OF
RECORD; **no_new_build_owed**. Its own `ready_blocked_by` states: "NO NEW BUILD
is owed on this entry ... the remaining work is DOWNSTREAM BEHAVIOURAL
VALIDATION, not a buildable substrate item ... what is owed is a FRESH successor
validation (a queue/governance decision) ... its forward motion is validation,
not construction." MECH-448 and MECH-449 each carry a brake count of ZERO.

So the Step 2.5b release condition is met on its own terms, and independently
this run falls under the "not braked" carve-out (new EXQ number; a different
mechanism-face; a probe discriminating WHY the ceiling held rather than
re-posing whether it holds).

WHY THE DV IS A VARIANCE SHARE, AND THE DV-SYMMETRY STATEMENT
-------------------------------------------------------------
Per /queue-experiment Step 3's DV-symmetry rule, for EVERY arm: name the
symmetry group of the arm's DV and state that the manipulation is not invariant
under it.

  DV (ROUTED): `f_variance_share` = the temporal variance fraction of the
  primary reality-cost components (f + harm_weighted), computed by V3-EXQ-571's
  EXACT method -- per-step MEAN ACROSS CANDIDATES, denominator var(SUM of the
  six SCORE_COMPONENTS) with covariance retained. Reproduced faithfully because
  MECH-439's ">0.85" bar is anchored to that number.

  DV (RECORDED, not routed): `f_variance_share_committed`, the same
  decomposition taken at the SELECTED candidate. Arguably the more faithful
  reading of "committed-selection variance", kept for a successor -- but
  re-anchoring the pre-registered bar to it would move the goalposts.

  Note the routed share is NOT bounded to [0, 1]: because the denominator
  var(SUM) retains covariance, negatively-correlated components make it smaller
  than sum(var), so a share above 1.0 is legitimate rather than a defect.

  Symmetry group of a variance RATIO over a component decomposition: joint
  positive affine rescaling of ALL components simultaneously (which cancels in
  the ratio), and permutation of the observations. It is NOT invariant under
  (a) reweighting one component relative to the others, nor (b) restriction of
  the candidate set over which the selection is taken.

  C2 is a PER-SEED PAIRED reduction vs ARM_OFF rather than an absolute bar,
  which additionally makes the verdict invariant to the run's overall scale --
  see MIN_F_SHARE_REDUCTION for why that matters here.

  ARM_OFF: no manipulation (reference cell).
  ARM_DEMOTION: MECH-448 moves F from a score CONTRIBUTOR to an eligibility
    MASK, which is precisely (a) reweighting F relative to the modulatory
    channels and (b) restricting the candidate set to the F-eligible prefix.
    Not invariant.
  (There is no MECH-449 arm -- see the next section for why.)

This is a deliberate contrast with the 12 braked runs, whose DVs were entropies
over the argmax/committed-class sequence -- RANK statistics, invariant under any
monotone transform of the scores. A rank-invariant DV structurally cannot see a
change in F's relative variance contribution; a variance ratio can. That
mismatch is one plausible reason the lineage kept reading "no conversion" while
the selection-face lever did in fact lift entropy in 689i.

WHY THERE IS NO MECH-449 ARM (a vacuous arm found and removed at smoke time)
---------------------------------------------------------------------------
A third arm enabling `use_go_nogo_constitution=True` on top of the demotion was
built and smoke-tested. It was REMOVED because it is structurally vacuous in a
full-agent driver, and the smoke proved it rather than merely suggesting it:
the arm's `f_variance_share` came out BIT-IDENTICAL to ARM_DEMOTION's
(1.4327509068 to 10 decimal places), while its `constitution_active_ticks` read
15/15 -- i.e. the gate REPORTS active but changes nothing.

The mechanism: MECH-449 consumes PER-CANDIDATE Go/No-Go signals supplied as
`E3Selector.select(..., go_nogo_signals=...)`. `REEAgent.select_action()` does
not plumb that argument, so a full-agent loop necessarily passes None and the
constitution has no signal to act on. V3-EXQ-689g -- the run that PASSED for
MECH-449 -- exercises it by calling `agent.e3.select(...)` DIRECTLY with
constructed `{"safety": ..., "staleness": ...}` signals, i.e. a selector-level
probe rather than an agent-level one.

Keeping the arm would have added apparent coverage with zero information, and
would have let a bit-identical duplicate be read as an independent
corroboration -- the artifact-laundering failure the /queue-experiment
DV-symmetry rule exists to prevent. Exercising MECH-449 properly needs either a
selector-level driver (the 689g architecture) or an agent-side plumbing change;
either is a legitimate SUCCESSOR, not something to fake here. This run is
therefore scoped to the ONE lever V3-EXQ-689i actually proved converts:
the MECH-448 rank-preserving F->eligibility demotion.

E3 LATCH / PSEUDO-REPLICATION
-----------------------------
`last_score_decomp` is populated ONLY inside `E3Selector.select()` and LATCHES:
on a tick where E3 did not re-select, it still holds the previous selection's
values. `agent.select_action` returns the held action before reaching
`e3.select()` whenever `not ticks["e3_tick"]`, and the E3 cadence
(`heartbeat.e3_steps_per_tick`) defaults to 10 -- so a naive per-env-step read
inflates n by ~10x (V3-EXQ-785: 600 rows behind 67 genuine selections). Every
decomposition sample here is gated on the SHARED freshness probe
(`experiments/_lib/fresh_select.py`), and `n_fresh_select` / `n_latched` /
`fresh_select_yield` / `replication_factor` are emitted per cell so the true
denominator is auditable rather than trusted.

SEED 44 is deliberately absent (reef-config per-seed instability; see the
baseline module docstring).

ASCII-only output (repo rule).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng  # noqa: E402
from experiments._lib.fresh_select import (  # noqa: E402
    FreshSelectCounter,
    FreshSelectProbe,
)
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.baselines.mech439_f_variance_share import (  # noqa: E402
    CONTRASTIVE_BATCH_K,
    E2_CONTRASTIVE_LR,
    E2_TRAIN_EVERY_K_TICKS,
    ENV_KWARGS,
    MAX_GRAD_NORM,
    P0_WARMUP_EPISODES,
    SD056_WEIGHT,
    SEEDS,
    STEPS_PER_EPISODE,
    TRANSITION_BUFFER_MAX,
    make_agent_kwargs,
    make_env,
    off_path_config_slice,
    sd056_training_slice,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_936_mech439_f_variance_share_under_f_demotion"
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-439"]

# ---------------------------------------------------------------------------
# PRE-REGISTERED CONSTANTS (defined here, never derived from the run's own data)
# ---------------------------------------------------------------------------

# MECH-439's stated CONFIRMING-signature bar: F's share staying "> 0.85".
# Recorded as a secondary absolute reading -- NOT what C2 routes on. See below.
F_MONOPOLY_THRESHOLD = 0.85

# WHY C2 IS A WITHIN-RUN PAIRED REDUCTION AND NOT AN ABSOLUTE BAR.
# MECH-439's falsifier is worded "... WITHOUT REDUCING F's E3 variance share".
# "Reducing" is inherently relative, and a within-run paired contrast
# (treatment vs ARM_OFF at the SAME seed, same config, same instrument) is
# scale-free: it is immune to the absolute level differing from 571's 0.886
# because 571 ran a different config (its diversity-stack stack, not this
# GAP-A/689i-derived stack). Routing C2 on the absolute 0.85 instead would make
# the verdict hostage to a cross-experiment calibration this run does not
# control -- and, since a 571-method share legitimately EXCEEDS 1.0 when
# components covary negatively (the denominator var(SUM) can be smaller than
# sum(var)), the absolute scale is not even bounded to [0,1].
#
# The minimum meaningful reduction is pre-registered FROM THE CLAIM'S OWN DATA,
# not from this run: V3-EXQ-571 measured 0.886 baseline vs 0.894 under the full
# diversity stack, a +0.008 move the claim itself describes as "the stack does
# not dent F's share". A genuine dent must therefore be materially larger than
# that null-level movement; 0.05 is ~6x it.
MIN_F_SHARE_REDUCTION = 0.05

# Context only (recorded, never gated on): the 571 figure the claim cites.
F_SHARE_REFERENCE_571 = 0.886

# Conversion bar, matching 689i's repaired 3-of-4 power bar.
MIN_SEEDS_CONVERTING = 3

# The primary reality-cost components. MECH-439's "F" is the primary harm/goal
# score; V3-EXQ-571 reports the monopoly figure as f + harm_weighted (its
# `raw_total_frac`), so the same pair is used here for comparability.
F_COMPONENTS = ("f", "harm_weighted")

# The FIXED component set V3-EXQ-571 decomposes over. Held identical here so the
# denominator spans the same components and the resulting share is numerically
# comparable to the 0.886 figure MECH-439 cites. Do NOT replace this with
# "whatever keys the decomp happens to emit" -- a different component set is a
# different denominator and therefore a different statistic.
SCORE_COMPONENTS = (
    "f", "harm_weighted", "residue_weighted",
    "benefit_weighted", "novelty_weighted", "goal_weighted",
)

# Fresh-selection budget per cell (689i's proven figure). Equalises the sample
# size across arms BY CONSTRUCTION rather than modelling exposure asymmetry.
N_FRESH_SELECT_TARGET = 200
P1_EPISODE_CAP = 40

# Readiness floors.
MIN_FRESH_SELECT_PER_CELL = 60      # P1: below this the variance is under-sampled
MIN_F_VARIANCE = 1e-9               # P2: F must actually vary across selections
MIN_NONF_VARIANCE = 1e-9            # P3: SAME statistic C2 routes on (a variance)
MIN_DEMOTION_ACTIVE_FRAC = 0.5      # P4: the lever must actually engage

# Online SD-056 contrastive training (689i parity) is declared in the canonical
# baseline module and imported above, so the fingerprint config_slice and the
# values the cell actually trains with cannot drift apart.

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1_CAP = 2
DRY_RUN_STEPS = 40
DRY_RUN_FRESH_TARGET = 4

OFF_ARM = "ARM_OFF"
DEMOTION_ARM = "ARM_DEMOTION"

ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": OFF_ARM,
        "flags": {
            "use_f_eligibility_demotion": False,
            "use_go_nogo_constitution": False,
        },
        "expects_demotion": False,
    },
    {
        "arm_id": DEMOTION_ARM,
        "flags": {
            "use_f_eligibility_demotion": True,
            "use_go_nogo_constitution": False,
        },
        "expects_demotion": True,
    },
]

# ARM_CONSTITUTION (MECH-449 Go/No-Go) is DELIBERATELY ABSENT -- see the
# "WHY THERE IS NO MECH-449 ARM" section of the module docstring. It was built,
# smoke-tested, measured BIT-IDENTICAL to ARM_DEMOTION, and removed rather than
# shipped.
TREATMENT_ARMS = [DEMOTION_ARM]

_FRESH_SELECT = FreshSelectProbe("exq936")
_ZG = ZGoalStreamAccumulator()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    v = d.get(key)
    if v is None or not torch.is_tensor(v):
        return None
    return v.float().unsqueeze(0) if v.dim() == 1 else v.float()


def _mean(xs: List[float], default: float = 0.0) -> float:
    return float(statistics.fmean(xs)) if xs else default


def _entropy_from_counts(counts: Counter) -> float:
    """Shannon entropy (nats) of a class histogram."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for n in counts.values():
        if n <= 0:
            continue
        p = n / total
        h -= p * math.log(p)
    return float(h)


def _variance_share(
    component_series: Dict[str, List[float]],
) -> Tuple[float, Dict[str, float], float]:
    """The V3-EXQ-571 temporal variance fraction -- REPRODUCED EXACTLY.

    Returns (f_share, per_component_fraction, total_variance).

    METHOD FIDELITY (this is the part that must be exactly right). MECH-439's
    ">0.85 F-share" bar refers to the number V3-EXQ-571 produced (0.886), so
    this statistic is computed the way 571 computes it, NOT a plausible variant:

      * the per-step value of a component is its MEAN ACROSS CANDIDATES
        (571: `step_comps[comp] = float(np.mean(vals))`), not its value at the
        selected candidate;
      * the DENOMINATOR is `var(SUM of all components)` -- which retains the
        cross-component COVARIANCE -- not `sum(var(component))`, which drops it.
        The two differ whenever components are correlated, and they are;
      * the component set is the FIXED `SCORE_COMPONENTS` list, so the
        denominator spans the same six components 571 used.

    A same-shaped-but-different statistic here would silently make the
    pre-registered 0.85 bar mean something other than what the claim registered,
    which is the failure mode /queue-experiment's "same statistic" rule exists
    to prevent. The alternative committed-selection operationalisation is
    computed separately by `_committed_variance_share` and RECORDED, but is
    deliberately not what the load-bearing criterion routes on.
    """
    n = 0
    for comp in SCORE_COMPONENTS:
        n = max(n, len(component_series.get(comp, [])))
    if n < 2:
        return 0.0, {c: 0.0 for c in SCORE_COMPONENTS}, 0.0

    arrays: Dict[str, List[float]] = {}
    for comp in SCORE_COMPONENTS:
        series = list(component_series.get(comp, []))
        if len(series) < n:
            series = series + [0.0] * (n - len(series))
        arrays[comp] = series[:n]

    # 571's denominator: variance of the per-step TOTAL (covariance retained).
    total_per_step = [
        float(sum(arrays[comp][i] for comp in SCORE_COMPONENTS)) for i in range(n)
    ]
    total_var = float(statistics.pvariance(total_per_step)) + 1e-12

    fractions: Dict[str, float] = {}
    for comp in SCORE_COMPONENTS:
        v = float(statistics.pvariance(arrays[comp]))
        if not math.isfinite(v):
            v = 0.0
        fractions[comp] = v / total_var
    f_share = float(sum(fractions.get(c, 0.0) for c in F_COMPONENTS))
    return f_share, fractions, total_var


def _committed_variance_share(
    component_series: Dict[str, List[float]],
) -> Tuple[float, Dict[str, float], float]:
    """SECONDARY, recorded-not-routed: the same decomposition taken at the
    SELECTED candidate rather than the pool mean.

    This is arguably the more faithful reading of MECH-439's phrase
    "committed-selection variance" -- it measures the variation in what actually
    drove the committed choice. It is recorded because a successor may want it
    and because a divergence between the two is itself informative, but it is
    NOT what C2 routes on: the claim's pre-registered 0.85 bar is anchored to
    571's pool-mean number, and re-anchoring it here would move the goalposts.
    """
    variances: Dict[str, float] = {}
    for comp in SCORE_COMPONENTS:
        series = component_series.get(comp, [])
        v = float(statistics.pvariance(series)) if len(series) > 1 else 0.0
        variances[comp] = v if math.isfinite(v) else 0.0
    total = float(sum(variances.values()))
    if total <= 0.0:
        return 0.0, {c: 0.0 for c in SCORE_COMPONENTS}, 0.0
    fractions = {k: (v / total) for k, v in variances.items()}
    f_share = float(sum(fractions.get(c, 0.0) for c in F_COMPONENTS))
    return f_share, fractions, total


def _sample_class_diverse_batch(
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    k: int,
    rng: random.Random,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    if len(buffer) < k:
        return None
    return rng.sample(list(buffer), k)


def _e2_contrastive_step(
    agent: REEAgent,
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimiser: torch.optim.Optimizer,
    rng: random.Random,
) -> Optional[float]:
    batch = _sample_class_diverse_batch(buffer, CONTRASTIVE_BATCH_K, rng)
    if batch is None:
        return None
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)
    optimiser.zero_grad(set_to_none=True)
    loss = agent.e2.world_forward_contrastive_loss(
        z_world_0=z0_K, actions=actions_K,
        z_world_1_targets=z1_K, simulation_mode=False,
    )
    if not torch.is_tensor(loss):
        return None
    loss_val = float(loss.detach().item())
    if not math.isfinite(loss_val):
        return loss_val
    if not loss.requires_grad or loss_val == 0.0:
        return loss_val
    weighted = SD056_WEIGHT * loss
    weighted.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return loss_val


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    cfg = REEConfig.from_dims(**make_agent_kwargs(env, arm["flags"]))
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Per-(seed, arm) runner
# ---------------------------------------------------------------------------

def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episode_cap: int,
    steps_per_episode: int,
    fresh_target: int,
) -> Dict[str, Any]:
    arm_id = str(arm["arm_id"])
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    # MINT AS YOU GO: the OFF arm is emitted reuse-ELIGIBLE with the driver
    # EXCLUDED from the hash, so a later, different-driver sibling can match
    # this cell by construction. Treatment arms declare their own slice.
    is_off = arm_id == OFF_ARM
    slice_for_cell = (
        off_path_config_slice() if is_off
        else {
            "env_kwargs": dict(ENV_KWARGS),
            "schedule": {
                "p0_warmup_episodes": p0_episodes,
                "steps_per_episode": steps_per_episode,
            },
            "sd056_training": sd056_training_slice(),
            "arm_flags": dict(arm["flags"]),
        }
    )

    with arm_cell(
        seed,
        config_slice=slice_for_cell,
        script_path=Path(__file__),
        config_slice_declared=True,
        include_driver_script_in_hash=False,
    ) as cell:
        # arm_cell's entry already performed the complete RNG reset.
        env = make_env(seed)
        agent = _make_agent(env, arm)
        # Diagnostics-only, gated; the selection path is bit-identical when OFF.
        agent.e3.e3_score_decomp_enabled = True
        e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)

        transition_buffer: Deque[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = deque(maxlen=TRANSITION_BUFFER_MAX)
        sample_rng = random.Random(seed)

        total_train_eps = p0_episodes + p1_episode_cap

        # Per-component series, one entry per GENUINE selection (never per env
        # step -- see the module docstring on the E3 latch).
        #   pool_component_series -- MEAN ACROSS CANDIDATES, the V3-EXQ-571
        #     method, and what the load-bearing C2 routes on.
        #   sel_component_series  -- value at the SELECTED candidate, recorded
        #     as the secondary committed-selection operationalisation.
        pool_component_series: Dict[str, List[float]] = {}
        sel_component_series: Dict[str, List[float]] = {}
        selected_class_counts: Counter = Counter()
        demotion_active_ticks = 0
        constitution_active_ticks = 0
        envelope_sizes: List[float] = []
        n_p1_fresh = 0
        # ENV-STEP count over the measured phase. This -- NOT len(rows) -- is the
        # denominator that makes the hold-replication factor auditable.
        n_p1_ticks = 0
        harm_p1_abs_sum = 0.0
        harm_p1_ticks = 0
        n_contrastive_steps = 0

        fs = FreshSelectCounter()
        p1_episodes_run = 0
        target_met = False
        error_note: Optional[str] = None

        for ep in range(total_train_eps):
            is_p1 = ep >= p0_episodes
            phase_label = "P1" if is_p1 else "P0"
            if is_p1:
                p1_episodes_run += 1

            _, obs_dict = env.reset()
            agent.reset()

            z_self_prev: Optional[torch.Tensor] = None
            action_prev: Optional[torch.Tensor] = None
            pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
            tick_in_ep = 0

            for _step in range(steps_per_episode):
                body = obs_dict["body_state"].float()
                world = obs_dict["world_state"].float()
                if body.dim() == 1:
                    body = body.unsqueeze(0)
                if world.dim() == 1:
                    world = world.unsqueeze(0)

                latent = agent.sense(
                    obs_body=body, obs_world=world,
                    obs_harm=_obs(obs_dict, "harm_obs"),
                    obs_harm_a=_obs(obs_dict, "harm_obs_a"),
                    obs_harm_history=_obs(obs_dict, "harm_history"),
                )

                if pending_capture is not None:
                    z0_prev, a_prev = pending_capture
                    z1_obs = latent.z_world.detach().reshape(-1).clone()
                    if (
                        torch.isfinite(z0_prev).all()
                        and torch.isfinite(a_prev).all()
                        and torch.isfinite(z1_obs).all()
                    ):
                        transition_buffer.append((z0_prev, a_prev, z1_obs))
                    pending_capture = None

                if z_self_prev is not None and action_prev is not None:
                    agent.record_transition(
                        z_self_prev, action_prev, latent.z_self.detach()
                    )

                ticks = agent.clock.advance()
                wdim = latent.z_world.shape[-1]
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, wdim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

                if agent.goal_state is not None:
                    try:
                        energy = float(body[0, 3].item())
                    except Exception:
                        energy = 1.0
                    agent.update_z_goal(
                        benefit_exposure=0.0,
                        drive_level=max(0.0, 1.0 - energy),
                    )

                # --- FRESHNESS-GATED SELECTION -------------------------------
                # Everything read out of e3.last_* below is valid ONLY when this
                # tick genuinely re-selected. See the module docstring.
                with _FRESH_SELECT.watch(agent) as _sel:
                    action = agent.select_action(candidates, ticks)
                fresh_select = _sel.fresh
                if is_p1:
                    n_p1_ticks += 1
                    fs.record(fresh_select)

                if is_p1 and fresh_select:
                    diag = agent.e3.last_score_diagnostics or {}
                    decomp = agent.e3.last_score_decomp or {}
                    per_cand = decomp.get("per_candidate")
                    sel_idx = decomp.get("selected_idx")

                    if isinstance(per_cand, list) and per_cand:
                        dict_cands = [c for c in per_cand if isinstance(c, dict)]
                        if dict_cands:
                            n_p1_fresh += 1
                            # V3-EXQ-571 method: per-step MEAN across candidates.
                            for comp in SCORE_COMPONENTS:
                                vals = []
                                for c in dict_cands:
                                    try:
                                        v = float(c.get(comp, 0.0))
                                    except (TypeError, ValueError):
                                        continue
                                    if math.isfinite(v):
                                        vals.append(v)
                                pool_component_series.setdefault(comp, []).append(
                                    float(statistics.fmean(vals)) if vals else 0.0
                                )
                            # Secondary: the SELECTED candidate.
                            if (
                                isinstance(sel_idx, int)
                                and 0 <= sel_idx < len(per_cand)
                                and isinstance(per_cand[sel_idx], dict)
                            ):
                                chosen = per_cand[sel_idx]
                                for comp in SCORE_COMPONENTS:
                                    try:
                                        v = float(chosen.get(comp, 0.0))
                                    except (TypeError, ValueError):
                                        continue
                                    if math.isfinite(v):
                                        sel_component_series.setdefault(
                                            comp, []
                                        ).append(v)

                    if bool(diag.get("f_eligibility_demotion_active", False)):
                        demotion_active_ticks += 1
                        env_size = float(diag.get("f_eligibility_envelope_size", -1))
                        if math.isfinite(env_size) and env_size >= 0:
                            envelope_sizes.append(env_size)
                    if bool(diag.get("go_nogo_constitution_active", False)):
                        constitution_active_ticks += 1

                    try:
                        cls = int(action.argmax(dim=-1).reshape(-1)[0].item())
                        selected_class_counts[cls] += 1
                    except Exception:
                        pass

                if (
                    torch.isfinite(latent.z_world).all()
                    and torch.isfinite(action).all()
                ):
                    pending_capture = (
                        latent.z_world.detach().reshape(-1).clone(),
                        action.detach().reshape(-1).clone(),
                    )

                if tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                    loss_val = _e2_contrastive_step(
                        agent=agent, buffer=transition_buffer,
                        optimiser=e2_opt, rng=sample_rng,
                    )
                    if loss_val is not None and math.isfinite(loss_val) and is_p1:
                        n_contrastive_steps += 1

                _, harm_signal, done, info, next_obs_dict = env.step(action)
                if is_p1:
                    hv = abs(float(harm_signal))
                    if math.isfinite(hv):
                        harm_p1_abs_sum += hv
                        harm_p1_ticks += 1
                with torch.no_grad():
                    agent.update_residue(
                        harm_signal=float(harm_signal), world_delta=None,
                        hypothesis_tag=False, owned=True,
                    )

                z_self_prev = latent.z_self.detach()
                action_prev = action
                obs_dict = next_obs_dict
                tick_in_ep += 1
                if done:
                    break

            fs.flush()

            if ep == 0 or (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
                print(
                    f"  [train] arm={arm_id} seed={seed} phase={phase_label} "
                    f"ep {ep + 1}/{total_train_eps} fresh={n_p1_fresh}/{fresh_target}",
                    flush=True,
                )

            if error_note is not None:
                break

            if is_p1 and n_p1_fresh >= fresh_target:
                target_met = True
                print(
                    f"  [p1-done] arm={arm_id} seed={seed} "
                    f"fresh={n_p1_fresh} after {p1_episodes_run} P1 episode(s)",
                    flush=True,
                )
                break

        # ROUTED statistic (V3-EXQ-571 method, pool mean, covariance denominator).
        f_share, fractions, total_var = _variance_share(pool_component_series)
        # RECORDED secondary (selected candidate).
        sel_share, sel_fractions, sel_total_var = _committed_variance_share(
            sel_component_series
        )

        nonf_var_frac = float(
            sum(v for k, v in fractions.items() if k not in F_COMPONENTS)
        )
        committed_entropy = _entropy_from_counts(selected_class_counts)

        f_series_len = max(
            (len(pool_component_series.get(c, [])) for c in F_COMPONENTS), default=0
        )
        # Absolute variances, used by the readiness preconditions. These are
        # variances of the component series themselves -- NOT shares -- so a
        # starved (flat) channel is detected regardless of what the ratio says.
        f_variance_abs = 0.0
        for c in F_COMPONENTS:
            s = pool_component_series.get(c, [])
            if len(s) > 1:
                v = float(statistics.pvariance(s))
                if math.isfinite(v):
                    f_variance_abs += v
        nonf_variance_abs = 0.0
        for c in SCORE_COMPONENTS:
            if c in F_COMPONENTS:
                continue
            s = pool_component_series.get(c, [])
            if len(s) > 1:
                v = float(statistics.pvariance(s))
                if math.isfinite(v):
                    nonf_variance_abs += v

        denom = max(1, n_p1_fresh)
        row: Dict[str, Any] = {
            "arm_id": arm_id,
            "seed": seed,
            # --- PRIMARY DV (V3-EXQ-571 method; what C2 routes on) ---
            "f_variance_share": f_share,
            "f_variance_share_method": (
                "exq571: per-step MEAN across candidates; denominator "
                "var(SUM of SCORE_COMPONENTS) with covariance retained"
            ),
            "component_variance_fractions": fractions,
            "total_component_variance": total_var,
            "f_variance_abs": f_variance_abs,
            "nonf_variance_abs": nonf_variance_abs,
            # --- SECONDARY DV (recorded, NOT routed) ---
            "f_variance_share_committed": sel_share,
            "f_variance_share_committed_method": (
                "selected-candidate value per genuine selection; denominator "
                "sum(var(component))"
            ),
            "committed_component_variance_fractions": sel_fractions,
            "committed_total_component_variance": sel_total_var,
            "n_selected_component_samples": max(
                (len(sel_component_series.get(c, [])) for c in F_COMPONENTS),
                default=0,
            ),
            # --- conversion DV (the 689i replication check) ---
            "committed_action_class_entropy": committed_entropy,
            "n_committed_classes": len(selected_class_counts),
            "selected_class_counts": dict(selected_class_counts),
            # --- sample-size integrity (E3 latch) ---
            # Canonical fragment from the shared helper: n_fresh_select,
            # n_latched, fresh_select_yield, replication_factor and the hold
            # duration stats, all against the ENV-STEP denominator n_p1_ticks.
            **fs.as_dict(n_p1_ticks),
            "n_p1_ticks": int(n_p1_ticks),
            "n_p1_decomp_samples": n_p1_fresh,
            "n_f_component_samples": f_series_len,
            "fresh_target_met": bool(target_met),
            # --- lever engagement ---
            "demotion_active_ticks": demotion_active_ticks,
            "demotion_active_frac": demotion_active_ticks / denom,
            "constitution_active_ticks": constitution_active_ticks,
            "constitution_active_frac": constitution_active_ticks / denom,
            "mean_envelope_size": _mean(envelope_sizes),
            # --- safety / context ---
            "p1_mean_abs_harm": (
                harm_p1_abs_sum / harm_p1_ticks if harm_p1_ticks else 0.0
            ),
            "p1_episodes_run": p1_episodes_run,
            "n_contrastive_steps": n_contrastive_steps,
            "error_note": error_note,
        }
        cell.stamp(row)

    _ZG.observe(agent)
    print(f"verdict: {'PASS' if error_note is None else 'FAIL'}", flush=True)
    return row


# ---------------------------------------------------------------------------
# Criteria + interpretation
# ---------------------------------------------------------------------------

def _evaluate(rows: List[Dict[str, Any]], seeds: List[int]) -> Dict[str, Any]:
    by_arm: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for r in rows:
        by_arm.setdefault(str(r["arm_id"]), {})[int(r["seed"])] = r

    off_rows = by_arm.get(OFF_ARM, {})

    # --- Readiness preconditions --------------------------------------------
    # P1: sample-size integrity -- the latch must not have starved the variance.
    worst_fresh_cell, worst_fresh = None, None
    for r in rows:
        v = int(r["n_p1_decomp_samples"])
        if worst_fresh is None or v < worst_fresh:
            worst_fresh, worst_fresh_cell = v, f"{r['arm_id']}/seed{r['seed']}"

    # P2: F must actually vary (the numerator of the share).
    worst_fvar_cell, worst_fvar = None, None
    for r in rows:
        v = float(r["f_variance_abs"])
        if worst_fvar is None or v < worst_fvar:
            worst_fvar, worst_fvar_cell = v, f"{r['arm_id']}/seed{r['seed']}"

    # P3: the NON-F components must carry variance -- SAME statistic (a variance)
    # that C2's share routes on, per the 643 magnitude-vs-range rule. If the
    # modulatory channels are flat, f_share is ~1.0 by starvation, not by
    # monopoly, and the falsifier is unmeasurable rather than confirmed.
    worst_nonf_cell, worst_nonf = None, None
    for r in rows:
        v = float(r["nonf_variance_abs"])
        if worst_nonf is None or v < worst_nonf:
            worst_nonf, worst_nonf_cell = v, f"{r['arm_id']}/seed{r['seed']}"

    # P4: the lever must engage on the treatment arms.
    treat_rows = [r for r in rows if r["arm_id"] in TREATMENT_ARMS]
    worst_dem_cell, worst_dem = None, None
    for r in treat_rows:
        v = float(r["demotion_active_frac"])
        if worst_dem is None or v < worst_dem:
            worst_dem, worst_dem_cell = v, f"{r['arm_id']}/seed{r['seed']}"

    # P5: instrument sanity -- ARM_OFF should reproduce the 571 reference band.
    off_share_mean = _mean([float(r["f_variance_share"]) for r in off_rows.values()])

    preconditions = [
        {
            "name": "decomp_samples_sufficient",
            "kind": "readiness",
            "description": (
                "Every cell banked at least MIN_FRESH_SELECT_PER_CELL GENUINE E3 "
                "selections (fresh-gated, never per-env-step) behind its variance "
                "decomposition. Guards the ~10x E3-cadence pseudo-replication."
            ),
            "control": "FreshSelectProbe-gated decomp reads; n_latched emitted per cell",
            "measured": float(worst_fresh if worst_fresh is not None else 0),
            "threshold": float(MIN_FRESH_SELECT_PER_CELL),
            "direction": "lower",
            "comparator": ">=",
            "offending_cell": worst_fresh_cell,
            "met": bool(
                worst_fresh is not None and worst_fresh >= MIN_FRESH_SELECT_PER_CELL
            ),
        },
        {
            "name": "f_variance_nondegenerate",
            "kind": "readiness",
            "description": (
                "The F components carry non-zero variance across committed "
                "selections in every cell -- the numerator of the share statistic."
            ),
            "control": "worst cell across all (arm, seed)",
            "measured": float(worst_fvar if worst_fvar is not None else 0.0),
            "threshold": float(MIN_F_VARIANCE),
            "direction": "lower",
            "comparator": ">=",
            "offending_cell": worst_fvar_cell,
            "met": bool(worst_fvar is not None and worst_fvar >= MIN_F_VARIANCE),
        },
        {
            "name": "nonf_variance_nondegenerate",
            "kind": "readiness",
            "description": (
                "The NON-F (modulatory) components carry non-zero variance in "
                "every cell. SAME STATISTIC (a variance) that the load-bearing "
                "C2 share routes on -- if these are flat, f_variance_share is "
                "~1.0 by STARVATION and the falsifier is unmeasurable, not "
                "confirmed."
            ),
            "control": "worst cell across all (arm, seed)",
            "measured": float(worst_nonf if worst_nonf is not None else 0.0),
            "threshold": float(MIN_NONF_VARIANCE),
            "direction": "lower",
            "comparator": ">=",
            "offending_cell": worst_nonf_cell,
            "met": bool(worst_nonf is not None and worst_nonf >= MIN_NONF_VARIANCE),
        },
        {
            "name": "demotion_lever_engaged",
            "kind": "readiness",
            "description": (
                "MECH-448 f_eligibility demotion reports active on at least "
                "MIN_DEMOTION_ACTIVE_FRAC of genuine selections in every "
                "treatment cell."
            ),
            "control": "e3.last_score_diagnostics f_eligibility_demotion_active, "
                       "fresh-gated; worst treatment cell",
            "measured": float(worst_dem if worst_dem is not None else 0.0),
            "threshold": float(MIN_DEMOTION_ACTIVE_FRAC),
            "direction": "lower",
            "comparator": ">=",
            "offending_cell": worst_dem_cell,
            "met": bool(
                worst_dem is not None and worst_dem >= MIN_DEMOTION_ACTIVE_FRAC
            ),
        },
    ]
    # NOTE: comparability against V3-EXQ-571's 0.886 is deliberately NOT a
    # gating precondition. It was one in an earlier draft and was demoted: 571
    # ran a DIFFERENT config, so a mismatch means "the configs differ", not
    # "the substrate is not ready" -- gating on it would self-route
    # substrate_not_ready_requeue and vacate an otherwise valid run, mislabelling
    # the cause (the /queue-experiment saturation-guard rule: record it, do not
    # gate on it). C2 is a within-run PAIRED contrast and does not depend on the
    # absolute level matching 571 at all. The number is recorded under
    # summary.off_arm_mean_f_variance_share for the reader.
    readiness_all_met = all(bool(p["met"]) for p in preconditions)

    # --- C1: does a treatment arm CONVERT (entropy strictly above OFF)? ------
    per_arm_conversion: Dict[str, Any] = {}
    for arm_id in TREATMENT_ARMS:
        arm_rows = by_arm.get(arm_id, {})
        converting_seeds = []
        deltas = []
        for seed in seeds:
            a = arm_rows.get(seed)
            o = off_rows.get(seed)
            if a is None or o is None:
                continue
            d = float(a["committed_action_class_entropy"]) - float(
                o["committed_action_class_entropy"]
            )
            deltas.append(d)
            if d > 0.0:
                converting_seeds.append(seed)
        per_arm_conversion[arm_id] = {
            "converting_seeds": converting_seeds,
            "n_converting": len(converting_seeds),
            "n_seeds": len(deltas),
            "entropy_deltas_vs_off": deltas,
            "converts": bool(len(converting_seeds) >= MIN_SEEDS_CONVERTING),
        }
    converting_arms = [a for a, v in per_arm_conversion.items() if v["converts"]]
    c1_pass = bool(converting_arms)

    # --- C2: did a CONVERTING arm REDUCE F's variance share vs ARM_OFF? ------
    # PER-SEED PAIRED contrast (same seed, same config, same instrument), which
    # is what makes it scale-free -- see MIN_F_SHARE_REDUCTION's rationale.
    per_arm_share: Dict[str, Any] = {}
    for arm_id in TREATMENT_ARMS:
        arm_rows = by_arm.get(arm_id, {})
        shares = [float(r["f_variance_share"]) for r in arm_rows.values()]
        mean_share = _mean(shares)
        paired_reductions: Dict[int, float] = {}
        reducing_seeds: List[int] = []
        for seed in seeds:
            a = arm_rows.get(seed)
            o = off_rows.get(seed)
            if a is None or o is None:
                continue
            red = float(o["f_variance_share"]) - float(a["f_variance_share"])
            paired_reductions[int(seed)] = red
            if red >= MIN_F_SHARE_REDUCTION:
                reducing_seeds.append(int(seed))
        per_arm_share[arm_id] = {
            "f_variance_share_per_seed": {
                int(s): float(r["f_variance_share"]) for s, r in arm_rows.items()
            },
            "mean_f_variance_share": mean_share,
            "paired_reduction_vs_off_per_seed": paired_reductions,
            "reducing_seeds": reducing_seeds,
            "n_reducing": len(reducing_seeds),
            # THE ROUTED PREDICATE.
            "reduced_f_share": bool(len(reducing_seeds) >= MIN_SEEDS_CONVERTING),
            # Secondary absolute reading against the claim's stated confirming
            # signature ("F's share stays > 0.85"). Recorded, never routed.
            "mean_share_above_monopoly_bar": bool(
                mean_share > F_MONOPOLY_THRESHOLD
            ),
            "mean_delta_vs_off": mean_share - off_share_mean,
        }

    c2_reduced_in_converting = None
    if c1_pass:
        c2_reduced_in_converting = all(
            per_arm_share[a]["reduced_f_share"] for a in converting_arms
        )

    # --- Route ---------------------------------------------------------------
    if not readiness_all_met:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        non_degenerate = False
        degeneracy_reason = (
            "Readiness precondition unmet: "
            + ", ".join(p["name"] for p in preconditions if not p["met"])
            + " -- F's variance share was not validly measurable, so MECH-439's "
              "pre-registered falsifier was NOT evaluated in either direction."
        )
    elif not c1_pass:
        # No arm converted, so the falsifying antecedent ("a lever that lifts
        # committed entropy") is absent. Per the claim's CONFIRMING signature
        # this is only informative if F's share also stayed above the bar.
        label = "no_conversion_falsifier_antecedent_absent"
        outcome = "FAIL"
        # The claim's CONFIRMING signature: flat committed entropy WHILE F's
        # share stays above the monopoly bar. Absolute here on purpose -- that
        # is how the claim words the confirming case.
        held_above = all(
            per_arm_share[a]["mean_share_above_monopoly_bar"]
            for a in TREATMENT_ARMS
        )
        if held_above:
            evidence_direction = "supports"
            non_degenerate = True
            degeneracy_reason = None
        else:
            evidence_direction = "non_contributory"
            non_degenerate = False
            degeneracy_reason = (
                "No arm converted AND F's share did not stay above "
                f"{F_MONOPOLY_THRESHOLD} -- neither the confirming nor the "
                "falsifying signature is present, so this run does not bear on "
                "MECH-439 in either direction."
            )
    elif c2_reduced_in_converting:
        # Converted AND F's share fell: conversion required rebalancing F.
        label = "conversion_accompanied_by_f_share_reduction"
        outcome = "PASS"
        evidence_direction = "supports"
        non_degenerate = True
        degeneracy_reason = None
    else:
        # THE FALSIFIER: converted WITHOUT reducing F's variance share.
        label = "conversion_without_f_share_reduction_falsifies_monopoly"
        outcome = "PASS"
        evidence_direction = "weakens"
        non_degenerate = True
        degeneracy_reason = None

    criteria = [
        {
            "name": "C1_a_treatment_arm_converts",
            "load_bearing": True,
            "passed": bool(c1_pass),
            "description": (
                "At least one treatment arm lifts committed-action-class entropy "
                f"above ARM_OFF on >= {MIN_SEEDS_CONVERTING} seeds."
            ),
        },
        {
            "name": "C2_converting_arm_reduces_f_variance_share",
            "load_bearing": True,
            "passed": bool(c2_reduced_in_converting) if c1_pass else False,
            "description": (
                "Every converting arm REDUCES F's variance share vs ARM_OFF by "
                f">= {MIN_F_SHARE_REDUCTION} on >= {MIN_SEEDS_CONVERTING} "
                "PER-SEED PAIRED comparisons. FALSE while C1 is TRUE is "
                "MECH-439's own falsifying signature: a lever that converts "
                "WITHOUT reducing F's variance share."
            ),
        },
    ]

    criteria_non_degenerate = {
        "C1_a_treatment_arm_converts": bool(readiness_all_met),
        "C2_converting_arm_reduces_f_variance_share": bool(
            readiness_all_met and c1_pass
        ),
    }

    return {
        "label": label,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "preconditions": preconditions,
        "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
        "combination_rule": (
            "PASS requires C1 (a treatment arm lifts committed-class entropy "
            "above ARM_OFF on the pre-registered majority of seeds). The "
            "DIRECTION is then decided by C2, a PER-SEED PAIRED reduction of "
            "F's variance share vs ARM_OFF: C2 true -> supports (conversion "
            "required rebalancing F's share, as MECH-439 predicts); C2 false "
            "-> weakens (a lever converted WITHOUT reducing F's share, which is "
            "the claim's own registered falsifying signature). C1 false with "
            "F's share held above the monopoly bar -> supports (the confirming "
            "signature). C1 false otherwise, or any readiness precondition "
            "unmet -> the falsifier was NOT evaluated and the run is scored "
            "non_contributory / non_degenerate."
        ),
        "off_arm_mean_f_variance_share": off_share_mean,
        "per_arm_conversion": per_arm_conversion,
        "per_arm_f_variance_share": per_arm_share,
        "converting_arms": converting_arms,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()

    seeds = list(DRY_RUN_SEEDS if dry_run else SEEDS)
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1_cap = DRY_RUN_P1_CAP if dry_run else P1_EPISODE_CAP
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    fresh_target = DRY_RUN_FRESH_TARGET if dry_run else N_FRESH_SELECT_TARGET

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            rows.append(
                _run_seed_arm(
                    arm=arm, seed=seed, p0_episodes=p0,
                    p1_episode_cap=p1_cap, steps_per_episode=steps,
                    fresh_target=fresh_target,
                )
            )

    summary = _evaluate(rows, seeds)

    full_config: Dict[str, Any] = {
        "env_kwargs": dict(ENV_KWARGS),
        "schedule": {
            "p0_warmup_episodes": p0,
            "p1_episode_cap": p1_cap,
            "steps_per_episode": steps,
            "fresh_select_target": fresh_target,
        },
        "arms": [{"arm_id": a["arm_id"], "flags": a["flags"]} for a in ARMS],
        "pre_registered_thresholds": {
            "F_MONOPOLY_THRESHOLD": F_MONOPOLY_THRESHOLD,
            "MIN_F_SHARE_REDUCTION": MIN_F_SHARE_REDUCTION,
            "F_SHARE_REFERENCE_571": F_SHARE_REFERENCE_571,
            "SCORE_COMPONENTS": list(SCORE_COMPONENTS),
            "MIN_SEEDS_CONVERTING": MIN_SEEDS_CONVERTING,
            "MIN_FRESH_SELECT_PER_CELL": MIN_FRESH_SELECT_PER_CELL,
            "MIN_F_VARIANCE": MIN_F_VARIANCE,
            "MIN_NONF_VARIANCE": MIN_NONF_VARIANCE,
            "MIN_DEMOTION_ACTIVE_FRAC": MIN_DEMOTION_ACTIVE_FRAC,
            "F_COMPONENTS": list(F_COMPONENTS),
        },
        "sd056": {
            "weight": SD056_WEIGHT, "lr": E2_CONTRASTIVE_LR,
            "batch_k": CONTRASTIVE_BATCH_K,
            "train_every_k_ticks": E2_TRAIN_EVERY_K_TICKS,
        },
        "dry_run": bool(dry_run),
    }

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": list(CLAIM_IDS),
        "outcome": summary["outcome"],
        "evidence_direction": summary["evidence_direction"],
        "evidence_class": "exp:v3_selection_variance_decomposition",
        "timestamp_utc": timestamp,
        "non_degenerate": bool(summary["non_degenerate"]),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "combination_rule": summary["combination_rule"],
        },
        "criteria": summary["criteria"],
        "acceptance_criteria": {
            "C1_a_treatment_arm_converts": bool(
                summary["criteria"][0]["passed"]
            ),
            "C2_converting_arm_reduces_f_variance_share": bool(
                summary["criteria"][1]["passed"]
            ),
        },
        "summary": {
            "label": summary["label"],
            "off_arm_mean_f_variance_share": summary["off_arm_mean_f_variance_share"],
            "per_arm_f_variance_share": summary["per_arm_f_variance_share"],
            "per_arm_conversion": summary["per_arm_conversion"],
            "converting_arms": summary["converting_arms"],
            "f_monopoly_threshold": F_MONOPOLY_THRESHOLD,
            "reference_571_baseline": F_SHARE_REFERENCE_571,
        },
        "per_seed_results": rows,
        "arm_results": rows,
        "custom_information": {
            "dv_symmetry_statement": (
                "DV = a variance RATIO over the E3 score decomposition at the "
                "selected candidate. Its symmetry group is joint positive affine "
                "rescaling of ALL components (cancels in the ratio) plus "
                "permutation of observations. The MECH-448/449 manipulation "
                "reweights F relative to the modulatory channels AND restricts "
                "the eligible candidate set, so it is NOT invariant under that "
                "group. Deliberately contrasted with the 12 braked runs, whose "
                "entropy-over-argmax DVs are RANK statistics and are invariant "
                "under any monotone transform of the scores."
            ),
            "re_derive_brake_release": (
                "MECH-439 brake count 12; upstream_substrate "
                "'f_dominance_conversion_ceiling' is BUILT+VALIDATED "
                "(MECH-448 promoted provisional, MECH-449 689g PASS, "
                "no_new_build_owed, 'forward motion is validation, not "
                "construction'). MECH-448/MECH-449 brake counts are 0."
            ),
            "gov_reuse_1_check": (
                "reanalysis_query over readouts f_variance_share / "
                "temporal_variance / bias_fraction / variance_fraction against "
                "claim MECH-439: 12 manifests matched, 0 carry any "
                "variance-share readout; only 689a has a recoverable "
                "substrate_hash and it carries none. Not recoverable -> run."
            ),
        },
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
    }
    if summary["degeneracy_reason"]:
        manifest["degeneracy_reason"] = summary["degeneracy_reason"]

    stamp_recording_core(
        manifest, config=full_config, seeds=seeds,
        script_path=Path(__file__), started_at=t0,
    )

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=bool(dry_run),
        config=full_config, seeds=seeds,
        script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"Manifest written: {out_path}", flush=True)
    print(f"Result written to: {out_path}", flush=True)
    print(
        f"Outcome: {summary['outcome']} (label={summary['label']}, "
        f"evidence_direction={summary['evidence_direction']})",
        flush=True,
    )
    for c in summary["criteria"]:
        print(f"  {c['name']}: {c['passed']}", flush=True)
    for p in summary["preconditions"]:
        print(f"  precondition {p['name']}: met={p['met']} "
              f"measured={p['measured']}", flush=True)
    print(
        f"  OFF f_variance_share={summary['off_arm_mean_f_variance_share']:.4f} "
        f"(571 reference {F_SHARE_REFERENCE_571})",
        flush=True,
    )

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-936 MECH-439: does a converting selection-face lever reduce "
            "F's committed-selection variance share?"
        )
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run.")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
        dry_run=args.dry_run,
    )
