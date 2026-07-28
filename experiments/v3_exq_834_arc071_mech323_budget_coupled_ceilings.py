"""V3-EXQ-834 -- ARC-071 / MECH-323: do chunk SIZE and chunk DEPTH move together
as the agent's deliberation budget changes?

PURPOSE (diagnostic -- a substrate-property DISCRIMINATION, not governance evidence).
The 2026-07-27 landings turned both fiat constants on ARC-071's compute-versus-
efficiency trade-off into growable ceilings DERIVED from one shared budget knob:
chunk size (MECH-323, ree-v3 c74434f) and chunk depth (ARC-071, ree-v3 7c201f7),
both default OFF, both reading REEConfig.chunk_deliberation_horizon. Until then the
independent variable could not move at all, which is why the coupled-parameter
question was registered as NOT YET QUEUED. It is now buildable. The readout here is
an internal ceiling, not behaviour, so this is classified diagnostic and is excluded
from confidence scoring on purpose: ARC-071 must not be promoted on ceiling
arithmetic.

WHICH QUESTION THIS ASKS (option (b), stated explicitly). Building the depth knob
surfaced a STRUCTURAL coupling that the literature framing does not contain: a chunk
composes another only by CONTAINING it contiguously, so a depth-D chain needs D
distinct sequence lengths and the deepest hierarchy the substrate can physically
mint is

    structural_max_depth = effective_max_chunk_size - min_chunk_size + 1

which is 4 at the inherited 2-5 budget. consider_depth_growth() refuses to grow past
that bound and reads the LIVE size ceiling. So the two parameters are not
independently manipulable, and a naive 2x2 has a cell that cannot be realised. This
run therefore does NOT test the literature prediction that both move with compute
(that prediction is nearly tautological here -- both derive from one knob). It tests
the sharper question: DOES DEPTH GROWTH ADD ANYTHING BEYOND WHAT THE SIZE CEILING
ALREADY LICENSES? The control that answers it is a size-growth-only arm, and the
substantive readout is WHICH BOUND BINDS -- chunk_depth_structural_max <
chunk_depth_derived_max means the size ceiling, not the deliberation budget, is what
is holding depth back.

WHAT THE IV IS, AND ITS SCOPE LIMIT (stated because it bounds the claim).
chunk_deliberation_horizon is a SENTINEL-DEFAULTED override that feeds the two
derivations ONLY; it does NOT change HippocampalConfig.horizon, so the agent's real
rollout budget and hence its planning quality are IDENTICAL in every arm. That is
deliberate -- moving the real horizon would change behaviour everywhere and confound
the ceiling manipulation with a policy-quality manipulation. The consequence is that
this run tests whether the DERIVATION couples size and depth, not whether genuinely
added compute buys deeper chunks. The latter needs the real horizon as the IV and is
a separate, confounded, more expensive design.

HORIZON LADDER (the derivations, computed at authoring time, not measured):
    H30: derived size floor(30*0.1667) = 5, derived depth floor(30*0.1) = 3
         -- exactly the inherited Sakai budget and the inherited R3/R4 cap. Both
         derived bounds EQUAL the starting constants, so consider_*_growth returns
         False at its first guard and NOTHING can grow. This is the anchoring claim
         made falsifiable.
    H50: derived size floor(50*0.1667) = 8, derived depth floor(50*0.1) = 5
         -- and 5 > 4 = the structural bound at a pinned size ceiling, which is
         exactly the regime where the two bounds disagree and the binding one is
         identifiable.

ARMS (5 conditions x 5 seeds = 25 cells). All share one baseline construction from
experiments/_lib/baselines/arc071_chunk_budget.py; the ONLY differences are the two
growth flags and the shared budget.
    STATIC_H50  flags off/off, horizon 50  -- the budget knob must be INERT without
                the flags. Ceilings pinned at 5 / 3.
    BOTH_H30    flags on/on,  horizon 30   -- the ANCHOR control. Flag-ON at today's
                budget must be indistinguishable from flag-OFF.
    SIZE_H50    size on, depth off, h 50   -- the (b) CONTROL. Size may reach 8;
                depth is a hard cap at 3. What does size growth alone license?
    DEPTH_H50   size off, depth on, h 50   -- the BINDING-BOUND arm. Size pinned at
                5, so structural_max_depth is pinned at 4 while the budget derives
                5. Depth must saturate at 4 however large the budget gets.
    BOTH_H50    flags on/on, horizon 50    -- the COUPLING test. Depth can exceed 4
                ONLY because the size ceiling grew and raised the structural bound.

WHY DEPTH_H50 IS AN INFORMATIVE ARM AND NOT THE IMPOSSIBLE 2x2 CELL. It is the arm
in which structural < derived, which is the reading the design exists to produce. Its
pre-registered prediction is SATURATION at 4, not growth to 5 -- so no criterion in
this run demands depth > 4 from it, and it is not scoped out as structurally vacuous.

DV-SYMMETRY (Step 3 mandatory declaration, per arm).
  Shared DV for the three measured arms: the final effective ceilings
  (chunk_acc_effective_ceiling, chunk_effective_max_depth) and their growth counts.
  These are integer bounds reached by a growth process over a tally of sequences, so
  the symmetry group is permutation of the tallied sub-sequences and relabelling of
  the action-class indices -- a ceiling is a symmetric function of its tally. The
  manipulation is NOT invariant under that group: raising the declared budget raises
  the SUPREMUM of the reachable set, which no permutation or relabelling preserves.
  Not a broadcast scalar on a rank readout either (the 604c hazard): with proposal
  injection OFF the chunking machinery never touches selection, so no DV in this run
  derives from an argmax or any order statistic over candidates.
  SIZE_H50  measured. Manipulation = the size flag; DV = the size ceiling and its
    growth count, whose cardinality the flag changes from a fixed 5.
  DEPTH_H50 measured. Manipulation = the depth flag against a pinned size ceiling;
    DV = the depth ceiling AND the pair (structural bound, derived bound), whose
    INEQUALITY the manipulation creates. Note the saturation prediction is a
    structural fact, not a null: chunk_depth_structural_max = 4 < 5 is directly
    reported, so a saturating result is a positive reading, not an absence.
  BOTH_H50  measured. Manipulation = both flags; DV as above plus the co-occurrence
    of depth > 4 with ceiling > 5 within a cell.
  STATIC_H50 and BOTH_H30 are STRUCTURAL CONTROLS, and their deltas are ARITHMETIC
    IDENTITIES declared as such rather than measured. STATIC_H50: with both flags off
    effective_max_chunk_size / effective_max_depth return the config constants
    unread, so the horizon cannot reach the DV by any path. BOTH_H30: the derivations
    return the starting constants, so consider_*_growth returns False at its first
    guard. Neither is cited as evidence in either direction -- they assert the
    ABSENCE of an effect that must not be present, which is what makes C0/C1
    meaningful whatever the measured arms do (the same footing V3-EXQ-810 gave its
    ARM_OFF).

WHY THE 810 EPISODE LENGTH WOULD HAVE MADE THIS VACUOUS, and what was changed.
V3-EXQ-810 FAILed with the accumulator silent (label chunk_accumulator_silent), and
its post-mortem (ree-v3/CLAUDE.md, credit-rule entry) found three causes, all of
which would have made THIS run's ceilings inert rather than merely unfired:
  (1) 24-step episodes give ~3 symbols per episode (record_step is reached only on
      the E3 deliberation path and E3 ticks every e3_steps_per_tick = 10 env steps),
      so note_outcome breaks out at len(actions) < size and every size above 3 is
      STRUCTURALLY unreachable -- both growable ceilings are inert at that length
      whatever the flags say. FIX: 60-step episodes, set from a MEASURED symbol
      rate rather than from steps/e3_steps_per_tick -- this script's smoke gave
      5.50 symbols per 20-step episode (0.275/step, well above the 0.1 a periodic
      tick would give), so 60 steps predicts ~16 symbols, about a 2x margin over
      the largest ceiling this ladder derives (8). The measured buffer is
      reported and GATED as a precondition rather than assumed.
  (2) 810's loop never called update_residue, so the MECH-091 salient-event E3
      phase_reset could never fire and the tick was perfectly periodic; compounded by
      num_hazards=0. FIX: experiments/_harness.StepHarness, whose invariant 3 pins
      exactly one update_residue per env step, plus num_hazards=2 (phase_reset is
      reached only on harm_signal < 0, so hazards are load-bearing, not decoration).
  (3) 3 seeds is too few for a stochastic formation event. FIX: 5 seeds.
  Also constant across every arm: use_chunk_all_position_credit=True. Trailing-only
  credit tallies at most one key per size per outcome and never a leading
  sub-sequence, starving the very tally the growth rule reads. It is held constant,
  not manipulated -- it is not this run's IV.

GOV-REUSE-1 (Step 2.4). Decisive readouts: chunk_acc_effective_ceiling,
chunk_acc_ceiling_derived_max, chunk_acc_n_ceiling_growths, chunk_effective_max_depth,
chunk_depth_derived_max, chunk_depth_structural_max, chunk_n_depth_growths. The only
recorded ARC-071 run in evidence/ is
v3_exq_810_arc071_chunk_accumulator_readiness_20260723T222726Z_v3 on substrate_hash
e228d8f303ccb09e..., which predates both landings by four days and carries NONE of
those seven keys (verified by key search on the manifest). reanalysis_query for
chunk_effective_max_depth returns no MATCH on any substrate_hash. Not recoverable ->
run.

ETHICS PREFLIGHT (Step 2.6): all involvement flags false, decision allow. Pure
arithmetic over a chunk tally on a resource-and-hazard grid; no self-model, no
suffering-like accumulator manipulation, no inescapability, no offline replay over
harm (the MECH-322 carve-out is OFF and asserted so by C6), no human data.

MECH-094 (SAFETY-CRITICAL). The waking path is the only writer into the accumulator,
so chunk_acc_n_replay_formed and chunk_lib_n_replay_origin must be 0 in every cell.
C6 pins it through agent.py.

SLEEP DRIVER: not applicable -- no sleep phase is entered, and the MECH-322
sleep-replay carve-out is OFF in every arm.
"""
from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.baselines import arc071_chunk_budget as BASE  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_834_arc071_mech323_budget_coupled_ceilings"
EXPERIMENT_PURPOSE = "diagnostic"

E3_HOLD_WEIGHTED_READOUT_EXEMPT = (
    "The one per-step accumulation from select_action's return value is the "
    "action_stream list, whose SOLE consumer is a SHA-1 EXACT-EQUALITY test across "
    "arms at a fixed seed (C5). Hold weighting is not a contaminant there -- it is "
    "the object of comparison: C5 asks whether the executed ENV-STEP sequence is "
    "identical across arms, so the held repeats must be present in the compared "
    "object, and an equality test on a hash is maximally threshold-invariant (the "
    "SAFE class the lint itself names). It is not a margin against a floor and not a "
    "distribution-shape statistic. Every DV that DOES bear on this experiment's "
    "verdict -- the ceiling readouts, the growth counts, the gain-evaluability "
    "counters -- is read once per note_outcome at an EPISODE boundary, never per env "
    "step, so no hold duration can weight it. The lint's own suggested counters are "
    "emitted anyway: n_fresh_select (= chunk_acc_n_steps, which record_step "
    "increments only on the E3 deliberation path) and n_latched, so the fresh-vs-held "
    "denominator is auditable from the manifest rather than inferred."
)

# ARC-071 owns the depth ceiling (ree-v3 7c201f7), MECH-323 the size ceiling
# (ree-v3 c74434f). MECH-324 maintenance runs but is held CONSTANT across every arm
# and is therefore not tagged -- nothing here tests it.
CLAIM_IDS = ["ARC-071", "MECH-323"]

SEEDS = [101, 202, 303, 404, 505]
N_EPISODES = BASE.N_EPISODES
STEPS_PER_EPISODE = BASE.STEPS_PER_EPISODE

# --- Conditions ------------------------------------------------------------
# (arm_id, use_growable_chunk_ceiling, use_growable_chunk_depth, horizon)
ANCHOR_HORIZON = 30
TREAT_HORIZON = 50
CONDITIONS: List[Tuple[str, bool, bool, int]] = [
    ("STATIC_H50", False, False, TREAT_HORIZON),
    ("BOTH_H30", True, True, ANCHOR_HORIZON),
    ("SIZE_H50", True, False, TREAT_HORIZON),
    ("DEPTH_H50", False, True, TREAT_HORIZON),
    ("BOTH_H50", True, True, TREAT_HORIZON),
]
ARMS = [c[0] for c in CONDITIONS]
GROWTH_ARMS = ["SIZE_H50", "DEPTH_H50", "BOTH_H50"]

# --- Pre-registered derivations (computed here at authoring time, ASSERTED below
# against what the substrate reports -- a mismatch is a substrate defect, not a
# scientific result).
EXPECTED_DERIVED = {
    ANCHOR_HORIZON: {"size": 5, "depth": 3},
    TREAT_HORIZON: {"size": 8, "depth": 5},
}
START_SIZE_CEILING = BASE.CHUNK_MAX_SIZE          # 5
START_DEPTH_CEILING = BASE.CHUNK_MAX_DEPTH        # 3
# structural_max_depth at the PINNED size ceiling = 5 - 2 + 1.
PINNED_STRUCTURAL_MAX_DEPTH = BASE.CHUNK_MAX_SIZE - BASE.CHUNK_MIN_SIZE + 1  # 4

# --- Pre-registered thresholds (defined HERE, never inferred post-hoc) -----
SEED_PASS_FRACTION = 3.0 / 5.0
OUTCOME_SPREAD_FLOOR = 0.05      # readiness: the evaluative gate is RELATIVE
MIN_FORMED_FLOOR = 0.5           # readiness: >= 1 chunk formed (strict-> semantics)
GAIN_EVALUABLE_FLOOR = 0.5       # readiness: >= 1 trial where the growth test ran
SYMBOL_BUFFER_FLOOR = 7.5        # readiness: >= ~8 symbols/episode so size 8 is
                                 # reachable at all (the V3-EXQ-810 structural bound)


# ---------------------------------------------------------------------------
# Preconditions -- regime-conditioned. A precondition that is not MEANINGFUL for
# an arm is scoped OUT of it (disposition (a)), never failed by it. This is the
# V3-EXQ-785 rule: one arm's inapplicable gate must not vacate another arm.
# ---------------------------------------------------------------------------
def _size_growth_possible(ctx: Dict[str, Any]) -> bool:
    """True only where a size growth could occur at all: flag on AND headroom."""
    return bool(ctx["grow_size"]) and ctx["derived_size"] > START_SIZE_CEILING


def _depth_growth_possible(ctx: Dict[str, Any]) -> bool:
    """True only where a depth growth could occur at all: flag on AND headroom.

    Headroom is against the DERIVED bound only. DEPTH_H50 has derived 5 > start 3,
    so the depth gain test is genuinely meaningful there even though the STRUCTURAL
    bound will stop it at 4 -- which is the arm's whole point.
    """
    return bool(ctx["grow_depth"]) and ctx["derived_depth"] > START_DEPTH_CEILING


PRECONDITIONS = [
    PreconditionSpec(
        name="episode_outcome_spread_supra_floor",
        description=(
            "MECH-323's evaluative gate is RELATIVE to a running baseline, so a flat "
            "outcome stream makes formation structurally impossible and every "
            "downstream ceiling reading meaningless. Asserts the SAME statistic the "
            "gate routes on (spread), never a mean or a magnitude proxy for it."),
        control=("worst cell in the arm; the task is a resource-and-hazard grid whose "
                 "per-episode collection genuinely varies episode to episode"),
        threshold=OUTCOME_SPREAD_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="symbol_buffer_per_episode",
        description=(
            "record_step is reached only on the E3 deliberation path and E3 ticks "
            "every e3_steps_per_tick env steps, so the accumulator sees "
            "steps_per_episode/10 symbols per episode. note_outcome breaks out at "
            "len(actions) < size, so every ceiling at or above that count is "
            "STRUCTURALLY inert -- the exact defect that made V3-EXQ-810's 24-step "
            "episodes unable to reach sizes 4-5. Asserts the buffer clears the "
            "largest ceiling this ladder derives."),
        control="worst cell in the arm; measured as chunk_acc_n_steps / n_episodes",
        threshold=SYMBOL_BUFFER_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="chunks_formed",
        description=(
            "No chunk, no ceiling reading: the growth rules read the accumulator's "
            "live tally and the registered chunk set, so a silent accumulator makes "
            "every criterion here unmeasurable rather than false. This is the "
            "V3-EXQ-810 failure mode carried forward as a gate."),
        control="worst cell in the arm",
        threshold=MIN_FORMED_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="size_gain_evaluable_trials",
        description=(
            "SAME-STATISTIC readiness for the size criterion (the V3-EXQ-643 rule). "
            "consider_ceiling_growth grows only when marginal_return_at_ceiling() is "
            "non-None, which needs a sequence AT the ceiling attested to "
            "min_repetitions. This counts the trials on which that test was actually "
            "evaluable. Zero means the growth test NEVER RAN, which is substrate "
            "not-ready -- not a refusal of the coupling."),
        control=("worst cell in the arm; counted once per note_outcome by calling the "
                 "same accessor the growth rule consumes"),
        threshold=GAIN_EVALUABLE_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=_size_growth_possible,
        applies_note=(
            "Not meaningful where a size growth cannot occur at any setting: the flag "
            "is off (STATIC_H50, DEPTH_H50) or the derived bound equals the starting "
            "ceiling so the first guard in consider_ceiling_growth refuses (BOTH_H30)."),
    ),
    PreconditionSpec(
        name="depth_gain_evaluable_trials",
        description=(
            "SAME-STATISTIC readiness for the depth criteria. consider_depth_growth "
            "grows only when marginal_return_at_depth_ceiling() is non-None, which "
            "needs a chunk AT the depth ceiling attested to min_repetitions with a "
            "judgeable shallower chunk it contains. This counts the trials on which "
            "that test was evaluable. Zero means the depth growth test never ran, so a "
            "saturating depth reading would be starvation, not the structural bound."),
        control=("worst cell in the arm; counted once per note_outcome by calling the "
                 "same accessor the growth rule consumes"),
        threshold=GAIN_EVALUABLE_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=_depth_growth_possible,
        applies_note=(
            "Not meaningful where a depth growth cannot occur at any setting: the flag "
            "is off (STATIC_H50, SIZE_H50) or the derived bound equals the starting "
            "depth so the first guard in consider_depth_growth refuses (BOTH_H30)."),
    ),
]


def _arm_ctx(arm: str) -> Dict[str, Any]:
    """Pre-registered context for one arm -- config only, no measured values."""
    grow_size, grow_depth, horizon = next(
        (c[1], c[2], c[3]) for c in CONDITIONS if c[0] == arm)
    return {
        "arm": arm,
        "grow_size": grow_size,
        "grow_depth": grow_depth,
        "horizon": horizon,
        "derived_size": EXPECTED_DERIVED[horizon]["size"],
        "derived_depth": EXPECTED_DERIVED[horizon]["depth"],
    }


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------
_ZG = ZGoalStreamAccumulator()


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **BASE.env_kwargs())


def _build_agent(env: CausalGridWorldV2, arm: str) -> REEAgent:
    ctx = _arm_ctx(arm)
    return REEAgent(REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        **BASE.agent_kwargs(
            use_growable_chunk_ceiling=ctx["grow_size"],
            use_growable_chunk_depth=ctx["grow_depth"],
            chunk_deliberation_horizon=ctx["horizon"],
        ),
    ))


def _config_slice(arm: str) -> Dict[str, Any]:
    ctx = _arm_ctx(arm)
    return BASE.config_slice(
        use_growable_chunk_ceiling=ctx["grow_size"],
        use_growable_chunk_depth=ctx["grow_depth"],
        chunk_deliberation_horizon=ctx["horizon"],
        n_episodes=N_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
    )


def _chunk_shape(agent: REEAgent) -> Dict[str, Any]:
    """Length and depth histograms over the registered, non-dissolved chunk set.

    get_state() reports the ceilings but not the SHAPE they produced, and the shape
    is what shows whether a raised ceiling was actually used rather than merely
    permitted. Recorded generously per the Experimental Recording Standard.
    """
    pc = agent.policy_chunking
    if pc is None:
        return {"length_hist": {}, "depth_hist": {}, "max_length": 0, "max_depth": 0}
    live = [c for c in pc.library.all_chunks() if c.state.value != "dissolved"]
    lengths = Counter(len(c.sequence) for c in live)
    depths = Counter(int(c.depth) for c in live)
    return {
        "length_hist": {str(k): int(v) for k, v in sorted(lengths.items())},
        "depth_hist": {str(k): int(v) for k, v in sorted(depths.items())},
        "max_length": max(lengths) if lengths else 0,
        "max_depth": max(depths) if depths else 0,
        "n_live_chunks": len(live),
    }


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm}", flush=True)
    with arm_cell(seed,
                  config_slice=_config_slice(arm),
                  script_path=Path(__file__),
                  config_slice_declared=True,
                  # MINT AS YOU GO: the driver script is EXCLUDED from the hash so a
                  # later sibling of this lineage, with its own driver, can match this
                  # cell and cite it via reuse_baseline_from. With the default (True)
                  # no cross-driver reuse is ever possible.
                  include_driver_script_in_hash=False) as cell:
        env = _build_env(seed)
        agent = _build_agent(env, arm)
        pc = agent.policy_chunking

        episode_outcomes: List[float] = []
        action_stream: List[int] = []
        n_size_gain_evaluable = 0
        n_depth_gain_evaluable = 0
        max_size_gain: Optional[float] = None
        max_depth_gain: Optional[float] = None
        ceiling_trace: List[int] = []
        depth_trace: List[int] = []

        # THE CANONICAL LOOP, not a hand-rolled one. experiments/_harness.StepHarness
        # invariant 3 pins exactly one agent.update_residue per env step, and MECH-091's
        # salient-event clock.phase_reset() lives INSIDE update_residue. V3-EXQ-810's
        # driver omitted that call, so the E3 tick was perfectly periodic, the symbol
        # buffer was a constant 3.00 on every seed, and both growable ceilings were
        # structurally inert. Using the harness rather than re-deriving the loop is what
        # stops this run inheriting that defect.
        harness = StepHarness(agent, env, train_mode=True, seed=seed)

        for ep in range(N_EPISODES):
            results = harness.run_episode(max_steps=STEPS_PER_EPISODE)
            ep_reward = 0.0
            for r in results:
                action_stream.append(int(r.action.argmax(dim=-1).item()))
                ep_reward += float(r.harm_signal)

            agent.note_chunk_outcome(ep_reward)
            episode_outcomes.append(ep_reward)

            # SAME-STATISTIC readiness instrumentation: call the very accessors the
            # growth rules consume, once per trial, and count how often the test was
            # EVALUABLE. A None is "no evidence either way", not a gain of zero, so it
            # is counted as not-evaluable rather than folded into the maximum.
            if pc is not None:
                sg = pc.accumulator.marginal_return_at_ceiling()
                if sg is not None:
                    n_size_gain_evaluable += 1
                    max_size_gain = sg if max_size_gain is None else max(max_size_gain, sg)
                dg = pc.marginal_return_at_depth_ceiling()
                if dg is not None:
                    n_depth_gain_evaluable += 1
                    max_depth_gain = dg if max_depth_gain is None else max(max_depth_gain, dg)
                ceiling_trace.append(int(pc.accumulator.effective_max_chunk_size))
                depth_trace.append(int(pc.effective_max_depth))

            # Every 25 episodes AND on the last one. The "or last" leg is not
            # cosmetic: without it a short --dry-run (4 episodes) emits no
            # `ep N/M` line at all, so the smoke could not verify the runner
            # progress contract it exists to verify.
            if (ep + 1) % 25 == 0 or ep == N_EPISODES - 1:
                st = agent.get_chunking_state()
                print(f"  [train] chunk seed={seed} arm={arm} ep {ep+1}/{N_EPISODES} "
                      f"formed={st.get('chunk_acc_n_formed', 0)} "
                      f"ceil={st.get('chunk_acc_effective_ceiling', 0)} "
                      f"depth={st.get('chunk_effective_max_depth', 0)}", flush=True)

        _ZG.observe(agent)
        st = agent.get_chunking_state()
        shape = _chunk_shape(agent)
        n_steps = int(st.get("chunk_acc_n_steps", 0))
        ctx = _arm_ctx(arm)
        row = {
            "arm": arm,
            "seed": seed,
            "horizon": ctx["horizon"],
            "grow_size": ctx["grow_size"],
            "grow_depth": ctx["grow_depth"],
            # --- the decisive ceiling readouts
            "chunk_acc_effective_ceiling": int(st.get("chunk_acc_effective_ceiling", 0)),
            "chunk_acc_ceiling_derived_max": int(st.get("chunk_acc_ceiling_derived_max", 0)),
            "chunk_acc_n_ceiling_growths": int(st.get("chunk_acc_n_ceiling_growths", 0)),
            "chunk_acc_last_ceiling_gain": float(st.get("chunk_acc_last_ceiling_gain", 0.0)),
            "chunk_effective_max_depth": int(st.get("chunk_effective_max_depth", 0)),
            "chunk_depth_derived_max": int(st.get("chunk_depth_derived_max", 0)),
            "chunk_depth_structural_max": int(st.get("chunk_depth_structural_max", 0)),
            "chunk_n_depth_growths": int(st.get("chunk_n_depth_growths", 0)),
            "chunk_last_depth_gain": float(st.get("chunk_last_depth_gain", 0.0)),
            # --- growth-test evaluability (the same-statistic readiness measures)
            "n_size_gain_evaluable_trials": n_size_gain_evaluable,
            "n_depth_gain_evaluable_trials": n_depth_gain_evaluable,
            "max_size_gain_seen": max_size_gain,
            "max_depth_gain_seen": max_depth_gain,
            "ceiling_trace": ceiling_trace,
            "depth_trace": depth_trace,
            # --- formation / maintenance
            "chunk_acc_n_formed": int(st.get("chunk_acc_n_formed", 0)),
            "chunk_acc_n_steps": n_steps,
            "chunk_acc_n_outcomes": int(st.get("chunk_acc_n_outcomes", 0)),
            "chunk_acc_n_credit_events": int(st.get("chunk_acc_n_credit_events", 0)),
            "chunk_acc_all_position_credit": bool(
                st.get("chunk_acc_all_position_credit", False)),
            "chunk_acc_n_tracked_sequences": int(st.get("chunk_acc_n_tracked_sequences", 0)),
            "chunk_acc_n_replay_formed": int(st.get("chunk_acc_n_replay_formed", 0)),
            "chunk_acc_n_simulation_skips": int(st.get("chunk_acc_n_simulation_skips", 0)),
            "chunk_lib_size": int(st.get("chunk_lib_size", 0)),
            "chunk_lib_n_crystallised": int(st.get("chunk_lib_n_crystallised", 0)),
            "chunk_lib_n_dissolved": int(st.get("chunk_lib_n_dissolved", 0)),
            "chunk_lib_n_selectable": int(st.get("chunk_lib_n_selectable", 0)),
            "chunk_lib_n_replay_origin": int(st.get("chunk_lib_n_replay_origin", 0)),
            "chunk_lib_by_state": st.get("chunk_lib_by_state", {}),
            "chunk_shape": shape,
            # --- structural-reachability readouts (the V3-EXQ-810 bound).
            # record_step is reached ONLY on the E3 deliberation path (the hold path
            # returns earlier in select_action), so chunk_acc_n_steps IS the fresh-
            # selection count and symbols_per_episode is the accumulator's real symbol
            # buffer -- not an env-step count divided by an assumed cadence. n_latched
            # is the complement, emitted so the fresh-vs-held denominator is auditable
            # from the manifest instead of inferred from e3_steps_per_tick.
            "symbols_per_episode": (n_steps / len(episode_outcomes)
                                    if episode_outcomes else 0.0),
            "n_fresh_select": n_steps,
            "n_latched": max(0, len(action_stream) - n_steps),
            "fresh_select_yield": (n_steps / len(action_stream)) if action_stream else 0.0,
            # --- task-side readouts (the outcome stream IS the evaluative gate's
            # substrate, so its distribution is load-bearing)
            "episode_outcome_mean": (statistics.fmean(episode_outcomes)
                                     if episode_outcomes else 0.0),
            "episode_outcome_spread": (statistics.pstdev(episode_outcomes)
                                       if len(episode_outcomes) > 1 else 0.0),
            "episode_outcome_min": min(episode_outcomes) if episode_outcomes else 0.0,
            "episode_outcome_max": max(episode_outcomes) if episode_outcomes else 0.0,
            "per_episode_outcomes": episode_outcomes,
            "n_episodes": len(episode_outcomes),
            # --- behavioural-parity witness. With proposal injection OFF the chunking
            # machinery never touches selection, so this hash must be IDENTICAL across
            # every arm at a given seed. Reported (C5) rather than assumed.
            "action_stream_sha1": hashlib.sha1(
                bytes(bytearray(a & 0xFF for a in action_stream))).hexdigest(),
            "action_stream_len": len(action_stream),
        }
        cell.stamp(row)
    # One verdict line per seed x condition cell. A cell "passes" when its own arm's
    # pre-registered structural prediction holds; this is the runner progress
    # contract, not the experiment's verdict.
    cell_ok = _cell_prediction_holds(row)
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row


def _cell_prediction_holds(row: Dict[str, Any]) -> bool:
    """The arm's own pre-registered per-cell prediction (progress-line verdict)."""
    arm = row["arm"]
    if arm in ("STATIC_H50", "BOTH_H30"):
        return (row["chunk_acc_effective_ceiling"] == START_SIZE_CEILING
                and row["chunk_effective_max_depth"] == START_DEPTH_CEILING)
    if arm == "SIZE_H50":
        return row["chunk_effective_max_depth"] == START_DEPTH_CEILING
    if arm == "DEPTH_H50":
        return row["chunk_effective_max_depth"] <= PINNED_STRUCTURAL_MAX_DEPTH
    return True


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _worst_cell(rows: List[Dict[str, Any]], key: str) -> Tuple[float, Optional[str]]:
    """Minimum of `key` across rows plus the offending cell id.

    The EXTREMUM, never the mean: every readiness `met` here quantifies over cells
    (all(...)), so the indexer's recompute must see the same statistic the quantifier
    tests, and it must name the culprit.
    """
    if not rows:
        return 0.0, None
    worst = min(rows, key=lambda r: float(r[key]))
    return float(worst[key]), f"{worst['arm']}/seed{worst['seed']}"


def _frac(rows: List[Dict[str, Any]], pred) -> float:
    return (sum(1 for r in rows if pred(r)) / len(rows)) if rows else 0.0


def run_experiment() -> Dict[str, Any]:
    # Design-time refusal: refuses the run BEFORE compute is spent if any arm's
    # pre-registered config makes an applicable precondition unsatisfiable, or if
    # every arm is vacuous. Both dispositions are deliberate here, so this should
    # pass -- it is the check, not a formality.
    assert_no_structurally_unsatisfiable_gate(
        PRECONDITIONS, [_arm_ctx(a) for a in ARMS])

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            rows.append(_run_cell(arm, seed))

    by_arm = {a: [r for r in rows if r["arm"] == a] for a in ARMS}

    # --- Per-arm gates, regime-conditioned. A red arm does NOT vacate a green one.
    arm_gates = []
    for arm in ARMS:
        cells = by_arm[arm]
        ctx = _arm_ctx(arm)
        spread_w, spread_c = _worst_cell(cells, "episode_outcome_spread")
        buf_w, buf_c = _worst_cell(cells, "symbols_per_episode")
        formed_w, formed_c = _worst_cell(cells, "chunk_acc_n_formed")
        sgain_w, _sgain_c = _worst_cell(cells, "n_size_gain_evaluable_trials")
        dgain_w, _dgain_c = _worst_cell(cells, "n_depth_gain_evaluable_trials")
        measured = {
            "episode_outcome_spread_supra_floor": spread_w,
            "symbol_buffer_per_episode": buf_w,
            "chunks_formed": formed_w,
        }
        if _size_growth_possible(ctx):
            measured["size_gain_evaluable_trials"] = sgain_w
        if _depth_growth_possible(ctx):
            measured["depth_gain_evaluable_trials"] = dgain_w
        gate = evaluate_arm_gate(arm, ctx, PRECONDITIONS, measured)
        for p in gate["preconditions"]:
            if p["precondition"] == "episode_outcome_spread_supra_floor":
                p["offending_cell"] = spread_c
            elif p["precondition"] == "symbol_buffer_per_episode":
                p["offending_cell"] = buf_c
            elif p["precondition"] == "chunks_formed":
                p["offending_cell"] = formed_c
        arm_gates.append(gate)
    agg = aggregate_arm_gates(arm_gates)
    green = set(agg["green_arms"])

    # --- Substrate-derivation assertions. These are ARITHMETIC checks on what the
    # substrate reports against what was derived at authoring time. A mismatch is a
    # substrate/config defect (a dead knob), never a scientific finding.
    derivation_ok = all(
        r["chunk_acc_ceiling_derived_max"] == EXPECTED_DERIVED[r["horizon"]]["size"]
        and r["chunk_depth_derived_max"] == EXPECTED_DERIVED[r["horizon"]]["depth"]
        for r in rows)

    # --- C0 (STRUCTURAL CONTROL, identity): the budget knob is inert without flags.
    c0_pass = all(r["chunk_acc_effective_ceiling"] == START_SIZE_CEILING
                  and r["chunk_effective_max_depth"] == START_DEPTH_CEILING
                  and r["chunk_acc_n_ceiling_growths"] == 0
                  and r["chunk_n_depth_growths"] == 0
                  for r in by_arm["STATIC_H50"])

    # --- C1 (STRUCTURAL CONTROL, identity): the anchor. Flag-ON at today's budget
    # derives exactly the inherited constants and grows nothing.
    c1_pass = all(r["chunk_acc_ceiling_derived_max"] == START_SIZE_CEILING
                  and r["chunk_depth_derived_max"] == START_DEPTH_CEILING
                  and r["chunk_acc_effective_ceiling"] == START_SIZE_CEILING
                  and r["chunk_effective_max_depth"] == START_DEPTH_CEILING
                  and r["chunk_acc_n_ceiling_growths"] == 0
                  and r["chunk_n_depth_growths"] == 0
                  for r in by_arm["BOTH_H30"])

    # --- C2 (LOAD-BEARING, MECH-323): the size ceiling responds to the budget.
    size_rows = by_arm["SIZE_H50"] + by_arm["BOTH_H50"]
    c2_frac = _frac(size_rows, lambda r: r["chunk_acc_effective_ceiling"] > START_SIZE_CEILING)
    c2_pass = c2_frac >= SEED_PASS_FRACTION

    # --- C3 (LOAD-BEARING, ARC-071): with the size ceiling pinned, the STRUCTURAL
    # bound binds below the derived one and depth saturates at it.
    c3a_pass = all(r["chunk_depth_structural_max"] == PINNED_STRUCTURAL_MAX_DEPTH
                   and r["chunk_depth_derived_max"] > PINNED_STRUCTURAL_MAX_DEPTH
                   for r in by_arm["DEPTH_H50"])
    c3b_pass = all(r["chunk_effective_max_depth"] <= PINNED_STRUCTURAL_MAX_DEPTH
                   for r in by_arm["DEPTH_H50"])
    c3_pass = c3a_pass and c3b_pass

    # --- C4 (LOAD-BEARING, ARC-071): depth beyond the size-pinned bound requires the
    # size ceiling to have grown. Two legs: the structural bound must RISE in the
    # coupled arm, and no cell may show a depth above the pinned bound without a
    # ceiling above the pinned size.
    c4_frac = _frac(by_arm["BOTH_H50"],
                    lambda r: r["chunk_depth_structural_max"] > PINNED_STRUCTURAL_MAX_DEPTH)
    c4_no_orphan = all(
        (r["chunk_effective_max_depth"] <= PINNED_STRUCTURAL_MAX_DEPTH)
        or (r["chunk_acc_effective_ceiling"] > START_SIZE_CEILING)
        for r in rows)
    c4_pass = (c4_frac >= SEED_PASS_FRACTION) and c4_no_orphan

    # --- C5 (validity witness): with proposal injection OFF, chunking must not
    # perturb selection, so the action stream is identical across arms at a seed.
    by_seed_hashes = {
        s: {r["action_stream_sha1"] for r in rows if r["seed"] == s} for s in SEEDS}
    c5_pass = all(len(h) == 1 for h in by_seed_hashes.values())

    # --- C6 (SAFETY, MECH-094): no chunk from replayed or imagined content.
    c6_pass = all(r["chunk_acc_n_replay_formed"] == 0
                  and r["chunk_lib_n_replay_origin"] == 0 for r in rows)

    scored_ok = (("SIZE_H50" in green or "BOTH_H50" in green) and c2_pass) and \
                ("DEPTH_H50" in green and c3_pass) and \
                ("BOTH_H50" in green and c4_pass)
    overall_pass = bool(agg["any_green"] and derivation_ok and c0_pass and c1_pass
                        and scored_ok and c5_pass and c6_pass)

    # --- Label routing. A readiness failure NEVER routes to a substrate verdict.
    if not agg["any_green"]:
        label = "substrate_not_ready_requeue"
    elif not derivation_ok:
        label = "budget_derivation_mismatch_substrate_defect"
    elif not (c0_pass and c1_pass):
        label = "growable_ceiling_knob_defect"
    elif not c6_pass:
        label = "mech094_replay_origin_violation"
    elif overall_pass:
        label = "size_and_depth_coupled_via_structural_bound"
    elif not c2_pass:
        label = "size_ceiling_growth_not_realised"
    elif not c4_pass:
        label = "size_grows_depth_does_not_follow"
    else:
        label = "coupling_partial"

    # --- Non-degeneracy, keyed per criterion to its owning arm's gate. C0/C1/C5/C6
    # assert the ABSENCE of something that must not appear, so they are meaningful
    # whatever the measured arms did -- the same footing V3-EXQ-810 gave its
    # inertness and safety criteria.
    crit_nd = dict(arm_criteria_non_degenerate(
        {"DEPTH_H50": ["C3"], "BOTH_H50": ["C4"]}, agg))
    # C2 is owned JOINTLY by SIZE_H50 and BOTH_H50, which arm_criteria_non_degenerate's
    # arm -> criteria map cannot express (a second arm listing C2 would overwrite the
    # first). Either owning arm being green makes it scorable, so it is keyed by hand.
    crit_nd["C2"] = ("SIZE_H50" in green) or ("BOTH_H50" in green)
    crit_nd["C0"] = True
    crit_nd["C1"] = True
    crit_nd["C5"] = True
    crit_nd["C6"] = True

    metrics = {
        "c0_pass": c0_pass, "c1_pass": c1_pass, "c2_pass": c2_pass,
        "c3_pass": c3_pass, "c4_pass": c4_pass, "c5_pass": c5_pass,
        "c6_pass": c6_pass,
        "c3a_structural_below_derived": c3a_pass,
        "c3b_depth_saturates_at_structural": c3b_pass,
        "c4_structural_rose_frac": c4_frac,
        "c4_no_orphan_depth": c4_no_orphan,
        "c2_size_grew_frac": c2_frac,
        "derivation_matches_preregistered": derivation_ok,
        "green_arms": sorted(green),
        "red_arms": sorted(agg["red_arms"]),
        # Per-arm ceiling summaries -- the substantive table.
        "per_arm": {
            a: {
                "effective_ceiling_max": max((r["chunk_acc_effective_ceiling"]
                                              for r in by_arm[a]), default=0),
                "effective_ceiling_mean": statistics.fmean(
                    [r["chunk_acc_effective_ceiling"] for r in by_arm[a]]) if by_arm[a] else 0.0,
                "effective_depth_max": max((r["chunk_effective_max_depth"]
                                            for r in by_arm[a]), default=0),
                "effective_depth_mean": statistics.fmean(
                    [r["chunk_effective_max_depth"] for r in by_arm[a]]) if by_arm[a] else 0.0,
                "derived_size": EXPECTED_DERIVED[_arm_ctx(a)["horizon"]]["size"],
                "derived_depth": EXPECTED_DERIVED[_arm_ctx(a)["horizon"]]["depth"],
                "structural_depth_max": max((r["chunk_depth_structural_max"]
                                             for r in by_arm[a]), default=0),
                "n_ceiling_growths_mean": statistics.fmean(
                    [r["chunk_acc_n_ceiling_growths"] for r in by_arm[a]]) if by_arm[a] else 0.0,
                "n_depth_growths_mean": statistics.fmean(
                    [r["chunk_n_depth_growths"] for r in by_arm[a]]) if by_arm[a] else 0.0,
                "n_formed_mean": statistics.fmean(
                    [r["chunk_acc_n_formed"] for r in by_arm[a]]) if by_arm[a] else 0.0,
                "symbols_per_episode_mean": statistics.fmean(
                    [r["symbols_per_episode"] for r in by_arm[a]]) if by_arm[a] else 0.0,
                "max_chunk_length": max((r["chunk_shape"]["max_length"]
                                         for r in by_arm[a]), default=0),
                "max_chunk_depth": max((r["chunk_shape"]["max_depth"]
                                        for r in by_arm[a]), default=0),
            }
            for a in ARMS
        },
        "n_replay_formed_total": sum(r["chunk_acc_n_replay_formed"] for r in rows),
        "action_stream_hash_sets": {str(s): sorted(h) for s, h in by_seed_hashes.items()},
    }

    interpretation = {
        "label": label,
        "preconditions": agg["adjudication_preconditions"],
        "preconditions_scope_note": agg["per_arm_gate"]["preconditions_scope_note"],
        "criteria": [
            {"name": "C0_budget_inert_without_flags", "load_bearing": False,
             "passed": c0_pass},
            {"name": "C1_anchor_reproduces_inherited_constants", "load_bearing": False,
             "passed": c1_pass},
            {"name": "C2_size_ceiling_responds_to_budget", "load_bearing": True,
             "passed": c2_pass},
            {"name": "C3_depth_bound_by_pinned_size_ceiling", "load_bearing": True,
             "passed": c3_pass},
            {"name": "C4_depth_beyond_pinned_bound_requires_size_growth",
             "load_bearing": True, "passed": c4_pass},
            {"name": "C5_chunking_does_not_perturb_selection", "load_bearing": False,
             "passed": c5_pass},
            {"name": "C6_no_replay_origin_chunks", "load_bearing": False,
             "passed": c6_pass},
        ],
        "criteria_non_degenerate": crit_nd,
    }

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "per_arm_gate": agg["per_arm_gate"],
        "interpretation": interpretation,
        "non_degenerate": bool(agg["non_degenerate"]),
        "degeneracy_reason": agg["degeneracy_reason"] or None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()

    global SEEDS, N_EPISODES, STEPS_PER_EPISODE
    if args.dry_run:
        # Smoke only: the MINIMUM that still reaches every code path -- all five
        # arms constructed, the regime-conditioned gate evaluated per arm and
        # aggregated, the StepHarness loop stepped, both growth accessors called at
        # an episode boundary, the row assembled, the manifest written and the
        # sentinel emitted. STEPS_PER_EPISODE stays >= e3_steps_per_tick so at least
        # one E3 deliberation occurs and record_step actually runs. NOT a scientific
        # run: nothing can form at this length, so the gate is EXPECTED to be red and
        # the outcome FAIL with label substrate_not_ready_requeue. A green gate here
        # would be the surprising result.
        SEEDS = [101]
        N_EPISODES = 2
        STEPS_PER_EPISODE = 20
        assert_no_structurally_unsatisfiable_gate(
            PRECONDITIONS, [_arm_ctx(a) for a in ARMS])

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS,
        "arms": ARMS,
        "conditions": [
            {"arm": a, "grow_size": g, "grow_depth": d, "horizon": h}
            for (a, g, d, h) in CONDITIONS
        ],
        "n_episodes": N_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "anchor_horizon": ANCHOR_HORIZON,
        "treat_horizon": TREAT_HORIZON,
        "expected_derived": {str(k): v for k, v in EXPECTED_DERIVED.items()},
        "start_size_ceiling": START_SIZE_CEILING,
        "start_depth_ceiling": START_DEPTH_CEILING,
        "pinned_structural_max_depth": PINNED_STRUCTURAL_MAX_DEPTH,
        "seed_pass_fraction": SEED_PASS_FRACTION,
        "outcome_spread_floor": OUTCOME_SPREAD_FLOOR,
        "min_formed_floor": MIN_FORMED_FLOOR,
        "gain_evaluable_floor": GAIN_EVALUABLE_FLOOR,
        "symbol_buffer_floor": SYMBOL_BUFFER_FLOOR,
        "baseline_module": BASE.CANONICAL_BASELINE_ID,
        "arm_config_slices": {a: _config_slice(a) for a in ARMS},
    }
    m = result["metrics"]
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if result["outcome"] == "PASS" else "unknown",
        "evidence_direction_per_claim": {
            # MECH-323 owns the SIZE ceiling: C2 is its readout.
            "MECH-323": "supports" if m["c2_pass"] else "unknown",
            # ARC-071 owns the DEPTH ceiling and the structural coupling: C3 + C4.
            "ARC-071": "supports" if (m["c3_pass"] and m["c4_pass"]) else "unknown",
        },
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": m,
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "per_arm_gate": result["per_arm_gate"],
        "interpretation": result["interpretation"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "sleep_driver_pattern": "not_applicable_no_sleep_phase",
    }
    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"green_arms: {m['green_arms']} red_arms: {m['red_arms']}", flush=True)
    for a in ARMS:
        pa = m["per_arm"][a]
        print(f"  {a}: ceiling {pa['effective_ceiling_mean']:.2f} "
              f"(derived {pa['derived_size']}) depth {pa['effective_depth_mean']:.2f} "
              f"(derived {pa['derived_depth']}, structural {pa['structural_depth_max']}) "
              f"formed {pa['n_formed_mean']:.2f} buf {pa['symbols_per_episode_mean']:.2f}",
              flush=True)
    print(f"C0={m['c0_pass']} C1={m['c1_pass']} C2={m['c2_pass']} C3={m['c3_pass']} "
          f"C4={m['c4_pass']} C5={m['c5_pass']} C6={m['c6_pass']} "
          f"derivation_ok={m['derivation_matches_preregistered']}", flush=True)
    print(f"wrote: {out_path}", flush=True)
    return result, out_path, args.dry_run


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
