"""V3-EXQ-1019: MECH-464 -- is the ARC-109 D1/D2 opponent-gain split ORDER-CHANGING?

CLAIM UNDER TEST (MECH-464, claims.yaml): `_d1_d2_split` (ree-v3/ree_core/predictors/
e3_selector.py, ~line 2376, applied ~line 2594) splits each non-motor loop's accumulator
into Go/D1 (relu(-accum), potentiated 1+d1_da_gain*da) and No-Go/D2 (relu(+accum),
depressed max(0, 1-d2_da_gain*da)) populations. Because the two populations are gained
ASYMMETRICALLY about zero, a non-zero da can reorder any candidate pair whose PRE-split
loop accumulators straddle zero -- contradicting MECH-463's premise that every "global
scalar" arousal route (D1/D2 among them) applies one scalar uniformly and therefore cannot
change an argmax/argmin.

=== WHY THIS RUN EXISTS NOW (2026-08-26 digestion_note, and what changed since) ===
MECH-464 sat unrun for five weeks because its ORIGINAL what_would_answer was not
executable, per three defects the digestion_note (claims.yaml, 2026-08-26) found by
reading source directly:
  (1) THE INSTRUMENT DID NOT EXIST. The straddle-fraction non-vacuity gate and the da=0
      shadow-argmin counterfactual (`loop_d1_d2_reorder_vs_da0`) were not in source.
  (2) NO EXOGENOUS `da` KNOB. `da = math.tanh(self._lcg_value_baseline)` and
      `_lcg_value_baseline` is normally written only under an ARC-108 learning flag; with
      that off, da == 0.0 forever and the split is PROVABLY bit-identical to the additive
      scalar, so a run in that state returns a spurious REFUTES. Prescribed fix: since
      `d1_gain`/`d2_gain` only ever appear as `1 +/- g*da`, sweep `d1_da_gain`/`d2_da_gain`
      at a FIXED non-zero da instead -- spans the identical surface, moves the two gains
      independently, and needs no ARC-108 learning.
  (3) Stale line references (cosmetic).

VERIFIED BY GREP + a live one-tick probe (this session, 2026-09-10) that defect (1) is
NOW RESOLVED: `_pair_straddle_frac` (~e3_selector.py:2395), the PRE-split accumulator
capture (~2578-2585), `loop_assoc_straddle_frac` / `loop_limbic_straddle_frac` /
`loop_d1_d2_d2_gain_zero` (~2571-2585, 2778-2782) and `loop_d1_d2_reorder_vs_da0`
(~2725-2745) all exist, are wired into `last_score_diagnostics`, and are already read by
`experiments/_lib/gate_dv.py`'s `GateDVRecorder` (added for MECH-464 specifically --
`GATE_DIAGNOSTIC_KEYS` / `DEFAULT_STRADDLE_FRAC_FLOOR`, never previously consumed by any
queued experiment; this is the first). So the substrate side of this claim is READY.

Defect (2) is addressed here via the prescribed reparametrisation: `_lcg_value_baseline`
is a plain instance attribute (not a property), so it is set directly post-construction
(the same post-construction attribute-assignment idiom used for commitment_threshold
injection in v3_exq_018b/050b) to a fixed value giving da = tanh(_lcg_value_baseline) a
comfortably non-trivial magnitude, side-stepping the ARC-108 learning requirement entirely.
`use_d1_d2_population_split=True`, `d1_da_gain` / `d2_da_gain` set per DESIGN below.

=== A FOURTH, PREVIOUSLY UNDOCUMENTED SUBSTRATE GAP FOUND BY THIS SESSION'S PROBE ===
`use_loop_segregation=True` alone does NOT reach `_segregated_loop_arbitrate` at all.
Reading e3_selector.py's `select()` (~line 3914-4028) directly: the entire
`eligible_idx` / `n_eligible` / `use_loop_segregation` branch is nested INSIDE
`if (use_modulatory_shortlist_then_modulate OR use_f_eligibility_demotion) and
_modulatory_accum is not None and raw_scores.shape[0] >= 2:` -- omit both flags and
`loop_segregation_active` stays permanently False (confirmed empirically: a probe with
`use_loop_segregation=True` + `use_d1_d2_population_split=True` alone produced
`loop_segregation_active=False` on every one of 8 fresh selections over 60 ticks). This
run therefore also sets `use_f_eligibility_demotion=True` (+ its matched envelope/floor
config) and `use_go_nogo_constitution=True`, mirroring the validated v3_exq_707c
"A1_LOOPS" matched-stack recipe, plus `use_named_channel_routing=True` (the 707b/707c C2
release repair -- without it the limbic loop carries no live per-candidate range at all,
per e3_selector.py's own inline comment at ~2524-2533) and the three limbic-loop value
SOURCES (`use_ofc_analog`, `use_mech295_liking_bridge`, `use_tonic_vigor`) 707c documents
as matched-constant prerequisites for the limbic channels to carry any live representation.
Once these are added, the live probe (2026-09-10, seed 0, untrained agent, 500 ticks,
d1_da_gain=3.0/d2_da_gain=3.0/da=0.9) measured `loop_segregation_active=True`,
`loop_d1_d2_active=True` on every fresh selection, straddle fraction up to 0.53
(non-vacuous), and 10/51 (19.6%) genuine `loop_d1_d2_reorder_vs_da0` events -- confirming
the mechanism is empirically reachable, not just true in code.

=== DESIGN: SINGLE CONDITION, STRADDLE-CONDITIONED ANALYSIS (not a level sweep) ===
The da=0 shadow argmin is computed WITHIN EVERY TICK against the SAME pre-split
accumulators, so "above the da==0 baseline" from the claim's what_would_answer is true
BY CONSTRUCTION for any single non-zero-da run -- a separate da==0 "control arm" would be
redundant. What is NOT guaranteed by construction, and IS the falsifier, is whether
reordering is actually CONFINED to (and graded by) straddle: `_loop_normalize` is
affine-invariant, so when a loop's pre-split accumulator values all share one sign, the
D1/D2 split there reduces to a pure positive rescale the zscore cancels EXACTLY -- meaning
a reorder should be near-IMPOSSIBLE on a tick where straddle_frac (both loops) is exactly
0, and should become MORE likely as straddle grows on ticks where it is not. This run
therefore pools EVERY d1d2-active fresh selection (across seeds) into one per-tick table
of (straddle_max, reorder_flag, d2_gain_zero_flag) and tests three separable predictions:

  C1 EXISTENCE     -- total reorder count > 0 (the split can reorder at all).
  C2 CONFINEMENT   -- reorder rate on straddle>0 ticks exceeds reorder rate on straddle==0
                      ticks (structurally near-zero on straddle==0 per the affine-invariance
                      argument above; a large straddle==0 rate would indicate either a
                      second reorder channel or an instrument defect, flagged but not
                      scored as falsifying MECH-464).
  C3 GRADATION     -- among straddle>0 ticks only, reorder likelihood rises with straddle
                      magnitude (point-biserial correlation r(straddle_max, reorder) > 0).
                      This is the "scales with straddle fraction" half of the claim's own
                      what_would_answer.

C1 and C2 are LOAD-BEARING (together they are "reordering exists and is where the
mechanism predicts it can be"); C3 is LOAD-BEARING for the graded half of the claim but
scored separately so a C1+C2-only pass (existence confirmed, gradation underpowered) is
distinguishable in the record from a full three-for-three pass.

d1_da_gain=2.0 / d2_da_gain=1.2 / da=0.6 (giving d1_gain=2.2, d2_gain=0.28 -- meaningfully
asymmetric, ~7.9x differential, deliberately short of the d2_gain==0 saturation edge
`max(0, 1 - d2_da_gain*da) == 0` at d2_da_gain*da >= 1) was chosen over the probe's
stronger da=0.9/gain=3.0 (which saturates d2_gain to exactly 0 -- the
`loop_d1_d2_d2_gain_zero` confound state MECH-464's own notes flag as a DIFFERENT real
effect to exclude, not treat as reordering) so the bulk of the run sits in the genuine
opponent-gain regime rather than the fully-silenced-No-Go regime. `d2_gain_zero_frac` is
still recorded and C3 is re-computed excluding those ticks as a sensitivity check.

=== NON-VACUITY GATE (MECH-464's own MANDATORY clause) ===
"report the straddle fraction. If it is ~0 the run is vacuous and MUST be scored
precondition_unmet, not as a null." Implemented as P2 below (pooled mean straddle_max
across d1d2-active ticks >= 0.01, `gate_dv.DEFAULT_STRADDLE_FRAC_FLOOR`). P1/P3-P5 are
supporting sample-size floors so a thin sample cannot manufacture a spurious PASS or FAIL.

=== INSTRUMENT: GateDVRecorder (never before consumed by a queued experiment) ===
`experiments/_lib/gate_dv.py` was built for exactly this claim (module comments cite
MECH-464 throughout) but had zero consumers until this script. It fresh-gates every read
via the shared `_lib/fresh_select.py` sentinel (the 699/689d/785 pseudo-replication
repair: E3 only reselects every `heartbeat.e3_steps_per_tick` ticks, default 10; reading
`last_score_diagnostics` on a latched tick silently re-records the PREVIOUS selection).
Used here for the run-level readiness block (`gate_readiness()`); the per-tick
(straddle, reorder) table for the C1-C3 analysis is collected in parallel from the SAME
fresh/latched determination via the recorder's own public `probe.diagnostics()` call
(read-only; no double-counting).

SCOPE: this is an `evidence` run testing MECH-464 only. It does not reopen MECH-463
(whose harm-urgency-threshold falsifier stands, per that claim's notes) and does not
speak to the softmax-temperature route (MECH-087, already evidenced separately).
GOV-REUSE-1: the decisive readout (per-tick D1/D2 reorder-vs-da0 conditioned on the
pre-split straddle fraction) has never been recorded -- no manifest carries
`loop_d1_d2_reorder_vs_da0` or `loop_assoc_straddle_frac` at all (grep of
evidence/experiments/ finds no prior consumer of these diagnostic keys) -- so it is not
recoverable by reanalysis and must run.

=== RED-TEAM DESIGN REVIEW (Step 4.5) ===
red-team (same-model fallback, sonnet-5 -- the fable spawn hit an account spend limit
mid-review and was not resumed, per the skill's "re-spawn once with no model argument"
rule): CONTESTED. Finding: C2's straddle==0 comparison bucket (`reorder_rate_zero`) had
NO minimum-N precondition anywhere in the original gate, even though
`criteria_non_degenerate` already computed `len(zero_rows) >= MIN_TERCILE_N_FLOOR` and
never wired it into `c2_pass` -- a thin zero-straddle bucket could pass C2 on sampling
luck alone. FIXED: promoted the same check to a gating precondition
(`straddle_zero_bucket_scorable`), mirroring `tercile_bands_scorable`'s existing floor on
the straddle>0 side. The reviewer separately traced and RULED OUT a hypothesized ARC-108
EMA mutation of `_lcg_value_baseline` mid-run (confirmed `post_action_update()` -- the
EMA's only caller -- is reached only via `agent.update_residue()`, which this driver's
tick loop never calls, unlike the 707c reference; da is genuinely frozen for the whole
run) and a suspected exploration-term contamination of the shadow-argmin comparison
(both `use_noisy_selection_head` / `use_model_disagreement_curiosity` are off).

Run:
  /opt/local/bin/python3 experiments/v3_exq_1019_mech464_d1d2_reorder_dagain_sweep.py --dry-run
"""

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.gate_dv import GateDVRecorder, DEFAULT_STRADDLE_FRAC_FLOOR  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

# ------------------------------------------------------------------ #
# Identity                                                            #
# ------------------------------------------------------------------ #
EXPERIMENT_TYPE = "v3_exq_1019_mech464_d1d2_reorder_dagain_sweep"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["MECH-464"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

OUT_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# The config_slice-declaration check cannot statically resolve a config_slice built
# inside a callee and returned by value (`_build_agent_and_env`'s `fingerprint_slice`,
# not a literal dict at the `arm_cell(config_slice=...)` call site) -- it flags
# ALPHA_WORLD/D1_DA_GAIN/D2_DA_GAIN/DA_TARGET/MODULATORY_AUTHORITY_GAIN as undeclared
# even though `fingerprint_slice` explicitly carries alpha_world/d1_da_gain/d2_da_gain/
# da_target/modulatory_authority_gain (verified by inspection: every readout-affecting
# constant IS a key in the dict actually passed to arm_cell -- see _build_agent_and_env).
CONFIG_SLICE_DECLARATION_EXEMPT = (
    "fingerprint_slice is built and returned inside _build_agent_and_env, not as a "
    "literal dict at the arm_cell() call site, so the static resolver cannot trace it; "
    "alpha_world/d1_da_gain/d2_da_gain/da_target/modulatory_authority_gain are all "
    "present as declared keys in that returned dict (confirmed by inspection)"
)

# ------------------------------------------------------------------ #
# Pre-registered constants (NOT derived from the run's own statistics) #
# ------------------------------------------------------------------ #
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
TICKS_PER_SEED = 2000
DRY_RUN_SEEDS = [0, 1]
DRY_RUN_TICKS = 150

D1_DA_GAIN = 2.0
D2_DA_GAIN = 1.2
DA_TARGET = 0.6  # da = tanh(_lcg_value_baseline); |da| >= 0.01 required (P3)
ALPHA_WORLD = 0.9
MODULATORY_AUTHORITY_GAIN = 0.5

assert abs(DA_TARGET) >= 0.01, "DA_TARGET must clear MECH-464's |da|>=0.01 precondition"
assert D2_DA_GAIN * abs(DA_TARGET) < 1.0, (
    "D2_DA_GAIN * DA_TARGET must stay below 1.0 or d2_gain saturates to exactly 0 on "
    "every tick, collapsing the run into the d2_gain_zero confound state instead of the "
    "genuine opponent-gain regime this design targets"
)

# Non-vacuity gate floors (FLOORS unless marked).
MIN_D1D2_ACTIVE_FLOOR = 200          # P1: enough d1d2-active fresh selections, pooled
STRADDLE_FRAC_FLOOR = DEFAULT_STRADDLE_FRAC_FLOOR  # P2: MECH-464's own mandatory gate (0.01)
MIN_FRESH_SELECT_FLOOR = 200         # P4: enough genuine E3 selections, pooled
MIN_TERCILE_N_FLOOR = 20             # P5: each straddle tercile needs enough n to score

# PASS criteria.
MIN_REORDER_COUNT_FLOOR = 20         # C1 existence: enough reorders to trust the count
POINT_BISERIAL_R_FLOOR = 0.03        # C3 gradation: modest but non-trivial positive floor
STRADDLE_ZERO_ANOMALY_CEILING = 0.05  # C2: reorder rate on straddle==0 ticks should be tiny


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #
def _pearson(x: List[float], y: List[float]) -> float:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return 0.0
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def _build_agent_and_env(seed: int):
    """Matched config: env/agent scaffold that reaches _segregated_loop_arbitrate with a
    live, non-degenerate D1/D2 split -- see the module docstring's "fourth substrate gap"
    section for why each of the loop-segregation-adjacent flags below is required."""
    env = CausalGridWorldV2(use_proxy_fields=True, seed=seed, hazard_harm=0.5)
    _obs, obs_dict = env.reset()

    kw: Dict[str, Any] = dict(
        body_obs_dim=obs_dict["body_state"].shape[-1],
        world_obs_dim=obs_dict["world_state"].shape[-1],
        action_dim=env.action_dim,
        alpha_world=ALPHA_WORLD,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        use_support_preserving_cem=True,
        support_preserving_min_first_action_classes=2,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=0.1,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=4.0,
        use_structured_curiosity=True,
        use_curiosity_novelty=True,
        curiosity_bias_scale=0.1,
        curiosity_novelty_weight=0.05,
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_min_range_floor=1e-6,
        # required to REACH the use_loop_segregation branch at all (see docstring).
        use_f_eligibility_demotion=True,
        f_eligibility_envelope_floor=0.30,
        f_eligibility_dn_sigma=0.0,
        use_f_eligibility_adaptive_floor=True,
        f_eligibility_adaptive_mean_factor=1.0,
        use_go_nogo_constitution=True,
        gng_perseveration_floor=0.5,
        gng_safety_floor=0.5,
        gng_protect_min_eligible=1,
        # finer channels + loops + D1/D2 (ARC-109/ARC-110/MECH-451).
        use_finer_channel_gating=True,
        use_dacc=True,
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,
        use_ofc_analog=True,
        use_mech295_liking_bridge=True,
        use_tonic_vigor=True,
        use_loop_segregation=True,
        loop_segregation_channel_map={},
        loop_segregation_normalize="zscore",
        use_named_channel_routing=True,
        use_d1_d2_population_split=True,
        d1_da_gain=D1_DA_GAIN,
        d2_da_gain=D2_DA_GAIN,
        use_loop_local_eligibility_traces=True,
        use_learned_settling_step=True,
    )
    cfg = REEConfig.from_dims(**kw)
    agent = REEAgent(cfg)
    agent.eval()
    # da = tanh(_lcg_value_baseline). Plain instance attribute (not a config field, not a
    # property), so setting it post-construction sidesteps the ARC-108 learning
    # requirement entirely -- the digestion note's prescribed exogenous-da workaround.
    agent.e3._lcg_value_baseline = math.atanh(max(min(DA_TARGET, 0.999999), -0.999999))
    # Declared fingerprint slice: `kw` (passed to from_dims) plus DA_TARGET, which is
    # applied post-construction via _lcg_value_baseline and so is NOT otherwise present
    # in `kw` at all -- an under-declared slice here would be a false-cache-HIT bug
    # (arm_reuse_fingerprint_plan.md 7b): a future consumer with a different DA_TARGET
    # would match this cell's fingerprint and silently reuse a readout computed under a
    # different exogenous da.
    fingerprint_slice = dict(kw)
    fingerprint_slice["da_target"] = DA_TARGET
    return agent, env, obs_dict, fingerprint_slice


def _collect_cell(seed: int, n_ticks: int, rec: GateDVRecorder, zg: ZGoalStreamAccumulator):
    """One seed's worth of ticks. Returns per-tick (straddle_max, reorder, d2_gain_zero)
    rows for d1d2-active fresh selections, plus the built config slice for fingerprinting."""
    agent, env, obs_dict, cfg_slice = _build_agent_and_env(seed)
    rows: List[Dict[str, Any]] = []

    print(f"Seed {seed} Condition d1d2_reorder_probe", flush=True)
    rec.begin_episode()
    for tick in range(n_ticks):
        with torch.no_grad():
            latent = agent.sense(
                obs_dict["body_state"].unsqueeze(0),
                obs_dict["world_state"].unsqueeze(0),
                obs_harm=obs_dict.get("harm_obs"),
                obs_harm_a=obs_dict.get("harm_obs_a"),
                obs_harm_history=obs_dict.get("harm_history"),
            )
            ticks_d = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks_d["e1_tick"]
                else torch.zeros(1, agent.config.latent.world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks_d)
            with rec.watch(agent) as sel:
                action = agent.select_action(candidates, ticks_d, 1.0)
        agent._step_count += 1

        diag = rec.probe.diagnostics(agent, bool(sel))
        if diag and bool(diag.get("loop_d1_d2_active", False)):
            assoc_s = float(diag.get("loop_assoc_straddle_frac", 0.0) or 0.0)
            limbic_s = float(diag.get("loop_limbic_straddle_frac", 0.0) or 0.0)
            rows.append({
                "seed": seed,
                "tick": tick,
                "straddle_max": max(assoc_s, limbic_s),
                "reorder": bool(diag.get("loop_d1_d2_reorder_vs_da0", False)),
                "d2_gain_zero": bool(diag.get("loop_d1_d2_d2_gain_zero", False)),
                "conflict_signal": float(diag.get("loop_d1_d2_conflict_signal", 0.0) or 0.0),
            })

        committed_class = int(action.argmax().item()) if isinstance(action, torch.Tensor) else int(action)
        rec.record(agent, sel, committed_class=committed_class, fallback=False)

        if (tick + 1) % 500 == 0 or tick == n_ticks - 1:
            n_reorder_so_far = sum(1 for r in rows if r["reorder"])
            print(f"  [train] mech464 seed={seed} ep {tick + 1}/{n_ticks} "
                  f"d1d2_rows={len(rows)} reorders={n_reorder_so_far}", flush=True)

        act_idx = int(action.argmax().item()) if isinstance(action, torch.Tensor) else int(action)
        _obs, _r, done, _info, obs_dict = env.step(act_idx % env.action_dim)
        if done:
            _obs, obs_dict = env.reset()
    rec.end_episode()
    zg.observe(agent)
    print(f"verdict: {'PASS' if len(rows) > 0 else 'FAIL'}", flush=True)
    return rows, cfg_slice


def run_experiment(dry_run: bool):
    t0 = time.perf_counter()
    n_ticks = DRY_RUN_TICKS if dry_run else TICKS_PER_SEED
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS

    rec = GateDVRecorder("mech464x1019", straddle_frac_floor=STRADDLE_FRAC_FLOOR)
    zg = ZGoalStreamAccumulator()
    all_rows: List[Dict[str, Any]] = []
    arm_results: List[Dict[str, Any]] = []

    for seed in seeds:
        probe_slice = _build_agent_and_env(seed)[3]
        with arm_cell(
            seed,
            config_slice=probe_slice,
            script_path=Path(__file__),
            config_slice_declared=True,
            include_driver_script_in_hash=False,
        ) as cell:
            rows, _cfg_slice = _collect_cell(seed, n_ticks, rec, zg)
            cell_row: Dict[str, Any] = {
                "arm_id": "d1d2_reorder_probe", "seed": seed,
                "n_d1d2_rows": len(rows),
                "n_reorder": sum(1 for r in rows if r["reorder"]),
            }
            cell.stamp(cell_row)
        arm_results.append(cell_row)
        all_rows.extend(rows)

    # ---------------- pooled analysis ---------------- #
    n_d1d2_active = len(all_rows)
    straddle_vals = [r["straddle_max"] for r in all_rows]
    reorder_flags = [1.0 if r["reorder"] else 0.0 for r in all_rows]
    straddle_frac_mean = float(np.mean(straddle_vals)) if straddle_vals else 0.0
    d2_gain_zero_frac = (
        float(np.mean([1.0 if r["d2_gain_zero"] else 0.0 for r in all_rows]))
        if all_rows else 0.0
    )

    zero_rows = [r for r in all_rows if r["straddle_max"] <= 0.0]
    pos_rows = [r for r in all_rows if r["straddle_max"] > 0.0]
    n_reorder_total = sum(1 for r in all_rows if r["reorder"])
    reorder_rate_zero = (
        float(sum(1 for r in zero_rows if r["reorder"]) / len(zero_rows))
        if zero_rows else 0.0
    )
    reorder_rate_pos = (
        float(sum(1 for r in pos_rows if r["reorder"]) / len(pos_rows))
        if pos_rows else 0.0
    )

    # C3: point-biserial correlation among straddle>0 ticks only (excludes the
    # structurally-non-reorderable straddle==0 ticks, which would only dilute/deflate r).
    pb_r = _pearson(
        [r["straddle_max"] for r in pos_rows], [1.0 if r["reorder"] else 0.0 for r in pos_rows]
    )
    # Sensitivity: same correlation excluding d2_gain_zero-saturated ticks.
    pos_rows_no_sat = [r for r in pos_rows if not r["d2_gain_zero"]]
    pb_r_no_sat = _pearson(
        [r["straddle_max"] for r in pos_rows_no_sat],
        [1.0 if r["reorder"] else 0.0 for r in pos_rows_no_sat],
    )

    # Tercile profile (reporting/diagnostic; not itself load-bearing).
    tercile_profile: List[Dict[str, Any]] = []
    if len(pos_rows) >= 3 * MIN_TERCILE_N_FLOOR:
        svals = np.asarray([r["straddle_max"] for r in pos_rows])
        lo, hi = float(np.quantile(svals, 1 / 3)), float(np.quantile(svals, 2 / 3))
        bands = [("low", -np.inf, lo), ("mid", lo, hi), ("high", hi, np.inf)]
        for name, a, b in bands:
            sel = [r for r, s in zip(pos_rows, svals) if (a <= s < b) or (name == "high" and s >= a)]
            if len(sel) < MIN_TERCILE_N_FLOOR:
                tercile_profile.append({"band": name, "n": len(sel), "scored": False})
                continue
            tercile_profile.append({
                "band": name, "n": len(sel), "scored": True,
                "straddle_mean": float(np.mean([r["straddle_max"] for r in sel])),
                "reorder_rate": float(sum(1 for r in sel if r["reorder"]) / len(sel)),
            })

    gate_dict = rec.as_dict(n_ticks=None)
    gate_readiness = rec.gate_readiness()
    n_fresh_select = int(gate_dict["n_fresh_select"])

    def _p(name, desc, control, measured, threshold, met=None, direction="lower"):
        d = {"name": name, "kind": "readiness", "description": desc, "control": control,
             "direction": direction, "measured": float(measured), "threshold": float(threshold)}
        d["met"] = bool(met) if met is not None else (
            bool(measured < threshold) if direction == "upper" else bool(measured > threshold)
        )
        return d

    preconditions = [
        _p("n_d1d2_active_pooled",
           "pooled count of fresh, d1d2-active E3 selections across all seeds",
           "use_d1_d2_population_split=True + loop_segregation reaching every fresh tick",
           n_d1d2_active, MIN_D1D2_ACTIVE_FLOOR),
        _p("straddle_frac_mean_nonvacuous",
           "MECH-464's OWN MANDATORY non-vacuity gate: mean straddle fraction (max of "
           "assoc/limbic) across d1d2-active ticks. If ~0, no candidate pair straddles "
           "zero and the run cannot answer the reordering question -- self-route "
           "precondition_unmet, never a null",
           "candidates whose loop accumulators genuinely straddle zero (measured live "
           "probe, 2026-09-10: up to 0.53 on a single seed)",
           straddle_frac_mean, STRADDLE_FRAC_FLOOR),
        _p("da_target_clears_floor",
           "design-time |da| clears MECH-464's own |da|>=0.01 precondition (da is a "
           "pre-registered constant here, not measured per-tick, since it is fixed by "
           "the exogenous injection, not endogenously computed)",
           "post-construction _lcg_value_baseline injection", abs(DA_TARGET), 0.01),
        _p("fresh_select_count_sufficient",
           "pooled genuine (non-latched) E3 selections -- the fresh_select repair "
           "denominator, guards against the 785-class ~9x pseudo-replication defect "
           "(not applicable here since GateDVRecorder fresh-gates by construction, but "
           "reported as a sample-size floor)",
           "fresh_select.py sentinel-key detection", n_fresh_select, MIN_FRESH_SELECT_FLOOR),
        _p("tercile_bands_scorable",
           "at least 2 of 3 straddle terciles (among straddle>0 ticks) have enough n to "
           "report a rate -- guards the diagnostic tercile profile against being computed "
           "from near-empty bands",
           f"straddle>0 pool, n={len(pos_rows)}",
           float(sum(1 for t in tercile_profile if t.get("scored"))), 1.5),
        # Red-team finding (this session, same-model fallback after the fable spawn hit
        # a spend limit): C2's reorder_rate_straddle_zero side had NO floor on
        # len(zero_rows) anywhere in the ORIGINAL preconditions list, even though
        # criteria_non_degenerate already computed exactly this check and then never
        # wired it into c2_pass -- a thin zero-straddle bucket could pass C2 on sampling
        # luck alone (near-zero rate over e.g. 3-10 rows) and the manifest's own
        # non-degeneracy flag would be the only place a reader could discover it. Fixed
        # by promoting the SAME check to a gating precondition, so a thin bucket
        # self-routes substrate_not_ready_requeue instead of silently passing.
        _p("straddle_zero_bucket_scorable",
           "the straddle==0 bucket C2 compares AGAINST needs enough n for its near-zero "
           "reorder rate to be a real confirmation rather than sampling noise -- "
           "MIRRORS tercile_bands_scorable's floor on the straddle>0 side, applied to "
           "the straddle==0 side C2 actually reads",
           f"straddle==0 pool, n={len(zero_rows)}",
           float(len(zero_rows)), float(MIN_TERCILE_N_FLOOR)),
    ]
    gate_green = all(p["met"] for p in preconditions)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    if not gate_green:
        outcome, evidence_direction = "FAIL", "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        failed = [p["name"] for p in preconditions if not p["met"]]
        degeneracy_reason = (
            "MECH-464 non-vacuity gate RED: " + ", ".join(failed) + ". A thin sample or "
            "a vacuous (near-zero) straddle fraction yields a profile indistinguishable "
            "from a genuine result, so this run is NOT scored and is NOT a refutation of "
            "MECH-464."
        )
        c1_pass = c2_pass = c3_pass = False
    else:
        non_degenerate, degeneracy_reason = True, ""
        c1_pass = n_reorder_total >= MIN_REORDER_COUNT_FLOOR
        c2_pass = bool(
            reorder_rate_zero <= STRADDLE_ZERO_ANOMALY_CEILING
            and reorder_rate_pos > reorder_rate_zero
        )
        c3_pass = pb_r > POINT_BISERIAL_R_FLOOR

        if c1_pass and c2_pass and c3_pass:
            outcome, evidence_direction = "PASS", "supports"
            label = "d1_d2_split_order_changing_straddle_conditioned"
        elif not c1_pass:
            outcome, evidence_direction = "FAIL", "does_not_support"
            label = "d1_d2_split_never_reorders_despite_straddle"
        elif not c2_pass and reorder_rate_zero > STRADDLE_ZERO_ANOMALY_CEILING:
            # Non-trivial reordering on straddle==0 ticks contradicts the affine-invariance
            # argument in the docstring -- flag as an instrument anomaly, not a claim verdict.
            outcome, evidence_direction = "FAIL", "unknown"
            label = "straddle_zero_reorder_anomaly_instrument_check_needed"
            non_degenerate = False
            degeneracy_reason = (
                f"reorder_rate_straddle_zero={reorder_rate_zero:.4f} exceeds the "
                f"{STRADDLE_ZERO_ANOMALY_CEILING} anomaly ceiling -- theory predicts this "
                "should be ~0 (loop_normalize's affine invariance makes a same-sign "
                "accumulator's D1/D2 split a pure rescale the zscore cancels exactly). "
                "This is either a second reorder channel or a straddle-fraction "
                "measurement defect; needs autopsy before it can be read as evidence "
                "either way."
            )
        else:
            outcome, evidence_direction = "FAIL", "mixed"
            label = "d1_d2_reorder_exists_but_ungraded_by_straddle"

    criteria = [
        {"name": "C1_existence_nonzero_reorder_count", "load_bearing": True,
         "passed": c1_pass, "measured": n_reorder_total, "threshold": MIN_REORDER_COUNT_FLOOR,
         "null_note": (
             "a null here (0 or near-0 reorders despite adequate straddle) means the "
             "D1/D2 split, though asymmetric in code, never actually flips the committed "
             "cross-loop winner on this substrate -- a clean REFUTES of MECH-464")},
        {"name": "C2_confinement_reorder_confined_to_straddle_positive", "load_bearing": True,
         "passed": c2_pass,
         "measured_reorder_rate_straddle_zero": reorder_rate_zero,
         "measured_reorder_rate_straddle_positive": reorder_rate_pos,
         "threshold_zero_ceiling": STRADDLE_ZERO_ANOMALY_CEILING,
         "n_straddle_zero": len(zero_rows), "n_straddle_positive": len(pos_rows),
         "null_note": (
             "structurally, a straddle==0 tick's D1/D2 split is a pure positive rescale "
             "the per-loop zscore cancels exactly, so reordering there should be near-"
             "impossible. A high straddle-zero rate is scored as an INSTRUMENT ANOMALY "
             "(unknown direction), not a MECH-464 verdict")},
        {"name": "C3_gradation_point_biserial_r_straddle_vs_reorder", "load_bearing": True,
         "passed": c3_pass, "measured": pb_r, "threshold": POINT_BISERIAL_R_FLOOR,
         "measured_excluding_d2_gain_zero_sensitivity": pb_r_no_sat,
         "n_straddle_positive": len(pos_rows),
         "null_note": (
             "a null here (r <= floor among straddle>0 ticks) means reordering, though it "
             "exists and is confined to straddle>0 ticks, does not further scale with "
             "straddle MAGNITUDE -- a MIXED finding: existence supported, graded "
             "'scales with straddle fraction' half of the claim's what_would_answer not "
             "supported")},
    ]
    criteria_non_degenerate = {
        "C1_existence_nonzero_reorder_count": bool(n_d1d2_active >= MIN_D1D2_ACTIVE_FLOOR),
        "C2_confinement_reorder_confined_to_straddle_positive": bool(
            len(zero_rows) >= MIN_TERCILE_N_FLOOR and len(pos_rows) >= MIN_TERCILE_N_FLOOR),
        "C3_gradation_point_biserial_r_straddle_vs_reorder": bool(len(pos_rows) >= 30),
    }

    full_config = {
        "seeds": seeds,
        "ticks_per_seed": n_ticks,
        "d1_da_gain": D1_DA_GAIN,
        "d2_da_gain": D2_DA_GAIN,
        "da_target": DA_TARGET,
        "alpha_world": ALPHA_WORLD,
        "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
        "env": {"cls": "CausalGridWorldV2", "use_proxy_fields": True, "hazard_harm": 0.5},
        "thresholds": {
            "MIN_D1D2_ACTIVE_FLOOR": MIN_D1D2_ACTIVE_FLOOR,
            "STRADDLE_FRAC_FLOOR": STRADDLE_FRAC_FLOOR,
            "MIN_FRESH_SELECT_FLOOR": MIN_FRESH_SELECT_FLOOR,
            "MIN_TERCILE_N_FLOOR": MIN_TERCILE_N_FLOOR,
            "MIN_REORDER_COUNT_FLOOR": MIN_REORDER_COUNT_FLOOR,
            "POINT_BISERIAL_R_FLOOR": POINT_BISERIAL_R_FLOOR,
            "STRADDLE_ZERO_ANOMALY_CEILING": STRADDLE_ZERO_ANOMALY_CEILING,
        },
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "timestamp_utc": ts,
        "dry_run": dry_run,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "config": full_config,
        "seeds": seeds,
        "arm_results": arm_results,
        "gate_dv": gate_dict,
        "gate_readiness": gate_readiness,
        "readout": {
            "outcome_pass": 1 if outcome == "PASS" else 0,
            "n_d1d2_active": n_d1d2_active,
            "n_fresh_select": n_fresh_select,
            "straddle_frac_mean": round(straddle_frac_mean, 6),
            "d2_gain_zero_frac": round(d2_gain_zero_frac, 6),
            "n_reorder_total": n_reorder_total,
            "reorder_rate_straddle_zero": round(reorder_rate_zero, 6),
            "reorder_rate_straddle_positive": round(reorder_rate_pos, 6),
            "point_biserial_r": round(pb_r, 6),
            "point_biserial_r_excl_d2_gain_zero": round(pb_r_no_sat, 6),
            "c1_existence_pass": 1 if c1_pass else 0,
            "c2_confinement_pass": 1 if c2_pass else 0,
            "c3_gradation_pass": 1 if c3_pass else 0,
            "gate_green": 1 if gate_green else 0,
        },
        "criteria": criteria,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "design_note": (
                "Single condition (no da==0 control arm): the da=0 shadow argmin "
                "(loop_d1_d2_reorder_vs_da0) is computed WITHIN every tick against the "
                "SAME pre-split accumulators, so 'above the da==0 baseline' from the "
                "claim's what_would_answer holds by construction for any nonzero-da run. "
                "The substantive test is confinement (C2) and gradation (C3) against the "
                "measured straddle fraction, pooled across all seeds."
            ),
            "d2_gain_zero_note": (
                f"d2_gain_zero_frac={d2_gain_zero_frac:.4f} at the chosen "
                f"d1_da_gain={D1_DA_GAIN}/d2_da_gain={D2_DA_GAIN}/da={DA_TARGET} (design-"
                "time: d2_gain=max(0, 1-d2_da_gain*da) computed to stay just above 0 -- "
                "see the module-level assert). Ticks in this state have the No-Go "
                "population fully silenced, a distinct real effect MECH-464's own notes "
                "say must be excluded from being read as reordering; C3 is reported both "
                "including and excluding these ticks."
            ),
            "tercile_profile": tercile_profile,
            "scope_note": (
                "Tests MECH-464 only. Does not reopen MECH-463 (whose harm-urgency-"
                "threshold falsifier stands per that claim's own notes) and does not "
                "speak to the softmax-temperature arousal route (MECH-087, already "
                "evidenced by V3-EXQ-674)."
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
        "custom_information": {
            "gov_reuse_1_check": (
                "Decisive readout: per-tick D1/D2 reorder-vs-da0 conditioned on the "
                "pre-split straddle fraction (loop_d1_d2_reorder_vs_da0 x "
                "loop_assoc_straddle_frac / loop_limbic_straddle_frac). No prior manifest "
                "carries either key -- GateDVRecorder (which reads them) had zero prior "
                "consumers. Not recoverable by reanalysis -> must run."
            ),
            "substrate_gap_found_this_session": (
                "use_loop_segregation=True + use_d1_d2_population_split=True ALONE never "
                "reaches _segregated_loop_arbitrate (loop_segregation_active stays False "
                "on every tick) -- the whole eligible_idx/n_eligible block in select() is "
                "nested inside `(use_modulatory_shortlist_then_modulate OR "
                "use_f_eligibility_demotion) and _modulatory_accum is not None`. Confirmed "
                "empirically (live probe, 2026-09-10) before this script was written; "
                "fixed here by matching the validated v3_exq_707c 'A1_LOOPS' config "
                "recipe (use_f_eligibility_demotion + use_go_nogo_constitution + "
                "use_named_channel_routing + the three limbic-loop value-source flags)."
            ),
            "per_tick_rows": all_rows,
        },
    }

    zg_stats = zg.stats()
    stamp_recording_core(
        manifest, config=full_config, seeds=seeds, script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=zg_stats,
    )
    return manifest, zg_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("V3-EXQ-1019: MECH-464 D1/D2 opponent-gain reorder-vs-straddle probe", flush=True)
    print(f"  dry_run={args.dry_run}", flush=True)

    t_start = time.perf_counter()
    manifest, zg_stats = run_experiment(args.dry_run)

    out_path = write_flat_manifest(
        manifest,
        OUT_DIR,
        dry_run=args.dry_run,
        config=manifest.get("config"),
        seeds=manifest.get("seeds"),
        script_path=Path(__file__),
        started_at=t_start,
        z_goal_stream_stats=zg_stats,
    )

    print(f"  manifest: {out_path}", flush=True)
    print(f"  outcome={manifest['outcome']} direction={manifest['evidence_direction']} "
          f"label={manifest['interpretation']['label']}", flush=True)
    r = manifest["readout"]
    print(f"  n_d1d2_active={r['n_d1d2_active']} straddle_frac_mean={r['straddle_frac_mean']} "
          f"n_reorder={r['n_reorder_total']} rate_zero={r['reorder_rate_straddle_zero']} "
          f"rate_pos={r['reorder_rate_straddle_positive']} r={r['point_biserial_r']}",
          flush=True)
    for p in manifest["interpretation"]["preconditions"]:
        if not p["met"]:
            print(f"  RED {p['name']}: measured={p['measured']:.6g} "
                  f"threshold={p['threshold']:.6g} ({p['direction']})", flush=True)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
