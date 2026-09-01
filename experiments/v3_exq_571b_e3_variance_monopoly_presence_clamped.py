"""V3-EXQ-571b: E3 committed-selection variance-MONOPOLY PRESENCE audit, clamp armed.

Re-measures V3-EXQ-571's variance-monopoly PRESENCE question in 571's OWN config
with the SD-056 E2 rollout output-norm clamp ARMED, so that MECH-439's quantitative
premise is -- for the first time -- measured by the fixed instrument.

WHY THIS RUN EXISTS. MECH-439 asserts "the primary harm/goal score (F) monopolises
~88-89% of E3 committed-selection variance". That 0.886 figure comes from
V3-EXQ-571's ARM_0_baseline (temporal_fraction_f = 0.8863118651034253), a DIAGNOSTIC
that ran 2026-05-15/16 -- BEFORE the rollout clamp landed (ree-v3 d327b89, 2026-05-31)
and with the clamp flag at its config default of False. Its successor family then
measured the opposite: V3-EXQ-936a, with the clamp armed but in a DIFFERENT config
(GAP-A/689i stack), found residue_weighted carrying ~99.9998% of committed variance
and F's share at 6.33e-06. Neither run can settle whether that inversion is caused by
THE CLAMP or by THE CONFIG, because the two differ in both at once.

This audit closes that by making the clamp a WITHIN-RUN, WITHIN-CONFIG manipulation:
571's own config, clamp OFF vs clamp ON, matched seeds. A clamp-ON-only re-run could
not separate the clamp from three months of substrate drift; this design can.

  Ha  no monopoly, or the monopoly leaves F, under the clamp
      -> MECH-439's quantitative premise does not survive the fixed instrument;
         governance re-words / narrows the claim.
  Hb  571's config genuinely exhibits an F-monopoly clamped
      -> MECH-439 is regime-scoped, and the 936-family falsifier may be re-posed
         HERE, with a monopoly-presence precondition and a RELATIVE bar.

Source: failure_autopsy_V3-EXQ-936a_2026-08-30 (REE_assembly b0c0b8df72), Section 8
routing, governance-ratified 2026-08-30 (REE_assembly 54dbe477be). Reports back to
closure node behavioral_diversity_isolation:GAP-I.

ARMS (2 x 2; the CLAMP contrast is load-bearing, the stack contrast is secondary):
  A0_baseline_clamp_off   571 ARM_0 config, clamp OFF  -- same-substrate replication
                          of the 0.886 reference cell
  A1_baseline_clamp_on    571 ARM_0 config, clamp ON   -- LOAD-BEARING cell
  A2_divstack_clamp_off   571 ARM_1 config, clamp OFF  -- secondary, 571-comparability
  A3_divstack_clamp_on    571 ARM_1 config, clamp ON   -- secondary

THREE INSTRUMENT DEFECTS IN THE PREDECESSORS, ALL FIXED HERE. Each was confirmed by
reading the substrate, not inferred:

 (1) PSEUDO-REPLICATION. 571 read `agent.e3.last_score_decomp` once per ENV STEP for
     200 steps with no freshness gate. `heartbeat.e3_steps_per_tick` defaults to 10
     (ree_core/heartbeat/clock.py:52), and the E3 `last_*` attributes LATCH between
     selections, so 571's ~200 recorded rows sit behind only ~20 genuine selections --
     roughly 10x replication. Measured here on this exact config: 6 fresh selections
     in 40 env steps (yield ~0.15). This driver gates every read through
     `experiments/_lib/fresh_select.FreshSelectProbe` (sentinel-key, NOT attribute
     nulling -- nulling `last_score_diagnostics` would change substrate behaviour via
     the ARC-016 fallback in post_action_update) and records `n_latched` /
     `replication_factor` so the true denominator is auditable.

 (2) THE DECOMPOSITION SUMMED A TERM THAT IS NOT IN THE SCORE. The E3 score is
     `score = f_weight * f + lambda_eff * m + rho_residue * phi`
     (ree_core/predictors/e3_selector.py:1310). The per-candidate component dict
     (e3_selector.py:1375-1383) therefore exposes `f_weighted` (= f_weight * f) as
     the actual additive F term, alongside a RAW, UNWEIGHTED `f` and the scalar
     multiplier `lambda_eff`. 571's SCORE_COMPONENTS list -- inherited unchanged by
     936/936a -- uses the raw `f`, so its "total" is not the score. At the default
     `f_weight = 1.0` (ree_core/utils/config.py:817) the two coincide numerically, so
     this did NOT corrupt the 0.886; it is a latent defect that would misreport
     silently under any `f_weight != 1.0`. This driver decomposes on BOTH sets --
     the corrected one and 571's legacy one -- and records `f_weight` in the config
     slice, so the 0.886 stays directly comparable AND the defect can never bite.

 (3) NO SCALE ANCHORS -- the question the 936a autopsy left open. Its prescription (b)
     was not implemented, so its manifest "cannot distinguish naturally sub-ceiling
     rollouts from clamp-PINNED rollouts (a norm-pinned rollout population would crush
     var(F) by construction)". This driver measures that directly and read-only, off
     the committed rollout retained at `agent.e3._last_selected_trajectory`: the clamp
     renormalises exactly to `ratio * ||z_world_0||` when it binds
     (e3/e2_fast.py:744-750), so a pinned step is detectable as
     `||z_w_t|| >= ratio * ||z_w_0|| * (1 - eps)`. No substrate change and no
     behaviour change. A one-shot probe on this config already found the committed
     rollout pinned EXACTLY at the ceiling on its first selection
     (||z_w_last|| = 0.7405 = 2.0 * ||z_w_0||), so this readout is expected to be
     load-bearing rather than decorative. On clamp-OFF arms the same statistic is the
     "would have been clamped" fraction, which is what makes the pair interpretable.

DV-SYMMETRY DECLARATION (one line per arm; Step 4.5 discipline). The DV is a set of
variance SHARES over per-selection component series. Its symmetry group: additive
per-channel constants (variance is translation-invariant), permutation of the
selection index (variance is a symmetric function), and a COMMON positive rescaling
of every channel (a share is invariant in the common factor).
  A0/A1 (the load-bearing CLAMP contrast): the manipulation is NOT invariant. The
    clamp bounds ||z_world|| along the rollout, which rescales the world-state path
    feeding f/residue NON-uniformly across channels, changing the RATIO of per-channel
    variances rather than a common factor or a per-channel constant. Confirmed
    empirically, not assumed: the clamp binds on this config (norm pinned at the
    ceiling), so it is an active non-affine transformation of the rollout.
  A2/A3 (the secondary STACK contrast): the diversity stack reaches this DV only
    INDIRECTLY, and that is declared rather than glossed. Its bias is composed into
    `score_bias` at the select() call site, NOT inside score_trajectory
    (e3_selector.py:1288-1296), and `benefit_weighted` / `novelty_weighted` /
    `goal_weighted` are hardcoded 0.0 in the component dict -- so the stack
    contributes NOTHING directly to any decomposed channel. Probed on this config,
    `last_channel_terms` carried a single `score_bias` entry with per-candidate
    range = 0.0 -- a BROADCAST SCALAR, which is argmin-invariant by construction.
    The stack can therefore move this DV only by changing which candidate is
    committed (via the noise-floor temperature path), i.e. indirectly. These arms are
    retained for 571-comparability and to RECORD that fact; they are explicitly NOT
    load-bearing, and no verdict routes on them.

KNOWN OPEN SUBSTRATE LIMITATIONS THIS RUN EXECUTES UNDER (Step 2.5c, recorded not
blocking): `contextmemory-write-path-addressing-degeneracy` (corrupting,
implemented_pending_validation, ree_core/predictors/e1_deep.py::ContextMemory.write)
and `SD-e1-rollout-consistency-training` (corrupting, item1_validated_item2_pending,
e1_deep.py::forward/predict_long_horizon) both overlap paths this driver exercises.
Both are fix-landed-pending-validation, and both bear on representation/rollout
QUALITY, whereas this audit's readout is a channel-SCALE property of the score sum.
The other open corrupting entries do not apply: `use_lateral_pfc_analog`,
`use_salience_coordinator` and `use_blocked_agency` are all config-default False and
this driver does not enable them.

red-team (fable): see queue entry note for V3-EXQ-571b.

EXPERIMENT_PURPOSE = "diagnostic" -- excluded from governance confidence scoring.
"""

import sys
import json
import math
import time
import argparse
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.fresh_select import FreshSelectProbe, FreshSelectCounter  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_571b_e3_variance_monopoly_presence_clamped"

# Each readiness predicate here measures EXACTLY the quantity that must be
# non-degenerate for its criterion to mean anything -- it is the degeneracy
# definition, not a hand-written narrower proxy for it, which is the V3-EXQ-778d
# failure mode this check guards (scoring one rail of a two-rail degeneracy).
# `pool_total_variance_nondegenerate` IS the share's own denominator;
# `decomp_samples_sufficient` IS the sample count the variances are taken over;
# `clamp_config_landed_every_cell` is a boolean config-landing identity. All three
# are additionally demonstrated reachable rather than assumed: a one-shot probe on
# this exact config landed the clamp (True on agent.e2.config) and yielded 6 fresh
# selections in 40 env steps (~300 at the shipped 2000), and V3-EXQ-936a cleared the
# same 60-sample floor at 201 with a non-degenerate total variance.
ANCHOR_REACHABILITY_EXEMPT = (
    "each predicate IS the degeneracy definition for the criterion it gates (share "
    "denominator / sample count / config-landing identity), not a narrower proxy; "
    "reachability separately demonstrated by the Step 2.5a probe and by 936a's 201 "
    "samples against the same 60 floor"
)

# 571's own seeds, plus one, for direct comparability with the 0.886 reference.
SEEDS = [42, 123, 456, 43]

# 571 ran 200 env steps per cell with no freshness gate. At e3_steps_per_tick=10 that
# is ~20 genuine selections. 2000 steps at the measured yield (~0.15) gives ~300.
N_EPISODES = 20
STEPS_PER_EPISODE = 100
N_STEPS = N_EPISODES * STEPS_PER_EPISODE

GRID_SIZE = 8
NUM_HAZARDS = 1
ALPHA_WORLD = 0.9  # SD-008: high-fidelity z_world (571 set this)

CLAMP_RATIO = 2.0  # matches the 689i / 936a parity value

# ENV SEEDING -- a DELIBERATE deviation from V3-EXQ-571's unseeded default.
# 571 passes seed=None (OS entropy at construction), so two cells at "seed 42"
# share an agent init but NOT a world layout. The load-bearing comparison here is
# A0 vs A1 AT MATCHED SEED, and under 571's scheme that contrast is confounded by
# layout: "matched seeds" would match the agent and not the world. The env seed is
# therefore derived from the RUN seed ONLY and is arm-independent, so all four arms
# at seed 42 see the identical layout and the clamp contrast is genuinely
# within-layout. Cost, stated: the ABSOLUTE shares are no longer produced under
# 571's exact env-sampling scheme, so the 0.886 comparison (already loose across
# three months of substrate drift) is comparability-by-config, not by layout draw.
ENV_SEED_BASE = 5710000

# --- Pre-registered thresholds (constants, never derived from this run) ---
F_MONOPOLY_THRESHOLD = 0.85   # 936a's own monopoly bar
MIN_FRESH_SELECTIONS = 60     # 936a's established decomp-sample floor
MIN_TOTAL_VARIANCE = 1e-9     # 936a's non-degeneracy floor
PIN_EPS = 1e-6                # relative tolerance for "at the clamp ceiling"

# The TRUE additive terms of the E3 score (e3_selector.py:1310, 1375-1383).
SCORE_COMPONENTS = (
    "f_weighted", "harm_weighted", "residue_weighted",
    "benefit_weighted", "novelty_weighted", "goal_weighted",
)
# EXACTLY V3-EXQ-571's list (raw unweighted `f`), kept so the 0.886 stays comparable.
LEGACY_571_COMPONENTS = (
    "f", "harm_weighted", "residue_weighted",
    "benefit_weighted", "novelty_weighted", "goal_weighted",
)
F_COMPONENTS = ("f_weighted", "harm_weighted")
LEGACY_F_COMPONENTS = ("f", "harm_weighted")

ARMS = [
    {"id": "A0_baseline_clamp_off", "stack": False, "clamp": False, "load_bearing": False},
    {"id": "A1_baseline_clamp_on",  "stack": False, "clamp": True,  "load_bearing": True},
    {"id": "A2_divstack_clamp_off", "stack": True,  "clamp": False, "load_bearing": False},
    {"id": "A3_divstack_clamp_on",  "stack": True,  "clamp": True,  "load_bearing": False},
]
LOAD_BEARING_ARM = "A1_baseline_clamp_on"
REFERENCE_ARM = "A0_baseline_clamp_off"

_FRESH_SELECT = FreshSelectProbe("v3_exq_571b")
_ZG = ZGoalStreamAccumulator()


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

def _stack_flags() -> Dict[str, Any]:
    """571's ARM_1 diversity stack, verbatim."""
    return {
        "use_noise_floor": True,
        "noise_floor_alpha": 0.1,
        "noise_floor_min_temperature": 1.0,
        "use_structured_curiosity": True,
        "use_curiosity_novelty": True,
        "use_curiosity_uncertainty": True,
        "use_curiosity_learning_progress": True,
        "curiosity_novelty_weight": 0.05,
        "curiosity_uncertainty_weight": 0.05,
        "curiosity_learning_progress_weight": 0.05,
        "curiosity_bias_scale": 0.1,
        "use_tonic_vigor": True,
        "tonic_vigor_half_life": 100.0,
        "tonic_vigor_w_action": 0.1,
        "tonic_vigor_w_passive": 0.1,
        "tonic_vigor_bias_scale": 0.1,
    }


def config_slice_for(arm: Dict[str, Any]) -> Dict[str, Any]:
    """Every readout-affecting constant, declared.

    MANDATORY per the 2026-08-30 C0 adjudication: the clamp flag AND ratio appear
    explicitly, because the entire point of this run is that they changed between
    V3-EXQ-571 and now. The score-composition weights are declared too, since they
    are literally the coefficients of the quantity being decomposed.
    """
    return {
        "env_kwargs": {
            "size": GRID_SIZE,
            "num_hazards": NUM_HAZARDS,
            "use_proxy_fields": True,
            "env_seed_base": ENV_SEED_BASE,
            "env_seed_scheme": "ENV_SEED_BASE + run_seed; arm-independent so arms share a layout",
        },
        "schedule": {
            "n_episodes": N_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "n_steps_total": N_STEPS,
        },
        "clamp": {
            "e2_rollout_output_norm_clamp_enabled": bool(arm["clamp"]),
            "e2_rollout_output_norm_clamp_ratio": CLAMP_RATIO,
        },
        "score_composition_weights": {
            "f_weight": 1.0,
            "lambda_ethical": 1.0,
            "rho_residue": 0.5,
        },
        "substrate": {
            "alpha_world": ALPHA_WORLD,
            "use_support_preserving_cem": True,
            "support_preserving_min_first_action_classes": 2,
            "support_preserving_stratified_elites": True,
            "support_preserving_ao_std_floor": 0.2,
            "harm_obs_dim": 51,
        },
        # Readout-affecting constants that the CELL itself records against
        # (`monopoly_present`, `cell_decidable`, `rollout_pinned_frac_*`). They must
        # be in the slice: this fingerprint is cross-driver reusable
        # (include_driver_script_in_hash=False), so omitting them would let a
        # consumer using different values HIT these cells and silently read
        # readouts computed under a different scheme (arm_reuse_fingerprint_plan 7b).
        "cell_readout_constants": {
            "f_monopoly_threshold": F_MONOPOLY_THRESHOLD,
            "min_fresh_selections": MIN_FRESH_SELECTIONS,
            "min_total_variance": MIN_TOTAL_VARIANCE,
            "pin_eps": PIN_EPS,
        },
        "arm_flags": _stack_flags() if arm["stack"] else {},
    }


def make_env_and_agent(seed: int, arm: Dict[str, Any]):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = CausalGridWorld(
        seed=ENV_SEED_BASE + int(seed),  # arm-INDEPENDENT: matches layout across arms
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        use_proxy_fields=True,
    )
    _, obs_dict = env.reset()

    config = REEConfig().from_dims(
        body_obs_dim=obs_dict["body_state"].shape[0],
        world_obs_dim=obs_dict["world_state"].shape[0],
        action_dim=env.action_dim,
        harm_obs_dim=51,
        use_support_preserving_cem=True,
        support_preserving_min_first_action_classes=2,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        e2_rollout_output_norm_clamp_enabled=bool(arm["clamp"]),
        e2_rollout_output_norm_clamp_ratio=CLAMP_RATIO,
    )
    config.latent.alpha_world = ALPHA_WORLD

    if arm["stack"]:
        for k, v in _stack_flags().items():
            setattr(config, k, v)

    agent = REEAgent(config=config)
    agent.eval()
    agent.e3.e3_score_decomp_enabled = True
    return agent, env, obs_dict


# --------------------------------------------------------------------------- #
# Variance-share methods                                                       #
# --------------------------------------------------------------------------- #

def _pool_variance_share(
    series: Dict[str, List[float]], components: Sequence[str], f_components: Sequence[str]
) -> Tuple[float, Dict[str, float], float]:
    """V3-EXQ-571's temporal variance fraction, reproduced exactly.

    Denominator is the variance of the per-selection TOTAL across `components`
    (covariance retained -- variance-of-sum, not sum-of-variances).
    """
    n = 0
    for comp in components:
        n = max(n, len(series.get(comp, [])))
    if n < 2:
        return 0.0, {c: 0.0 for c in components}, 0.0

    arrays: Dict[str, List[float]] = {}
    for comp in components:
        s = list(series.get(comp, []))
        if len(s) < n:
            s = s + [0.0] * (n - len(s))
        arrays[comp] = s[:n]

    total_per_step = [float(sum(arrays[c][i] for c in components)) for i in range(n)]
    total_var = float(statistics.pvariance(total_per_step)) + 1e-12

    fractions: Dict[str, float] = {}
    for comp in components:
        v = float(statistics.pvariance(arrays[comp]))
        if not math.isfinite(v):
            v = 0.0
        fractions[comp] = v / total_var
    f_share = float(sum(fractions.get(c, 0.0) for c in f_components))
    return f_share, fractions, total_var


def _committed_variance_share(
    series: Dict[str, List[float]], components: Sequence[str], f_components: Sequence[str]
) -> Tuple[float, Dict[str, float], float]:
    """The same decomposition taken AT THE SELECTED candidate.

    Denominator is the sum of per-component variances (covariance dropped), so the
    fractions are a genuine partition summing to 1.0.
    """
    variances: Dict[str, float] = {}
    for comp in components:
        s = series.get(comp, [])
        v = float(statistics.pvariance(s)) if len(s) > 1 else 0.0
        variances[comp] = v if math.isfinite(v) else 0.0
    total = float(sum(variances.values()))
    if total <= 0.0:
        return 0.0, {c: 0.0 for c in components}, 0.0
    fractions = {k: (v / total) for k, v in variances.items()}
    f_share = float(sum(fractions.get(c, 0.0) for c in f_components))
    return f_share, fractions, total


def _top_channel(fractions: Dict[str, float]) -> Tuple[str, float]:
    if not fractions:
        return "none", 0.0
    top = max(fractions.items(), key=lambda kv: kv[1])
    return str(top[0]), float(top[1])


# --------------------------------------------------------------------------- #
# Per-cell run                                                                 #
# --------------------------------------------------------------------------- #

def _pin_stats(traj: Any, clamp_ratio: float) -> Optional[Dict[str, float]]:
    """Read-only clamp-binding statistics off the COMMITTED rollout.

    The clamp renormalises z_world exactly to `ratio * ||z_world_0||` when it binds
    (e2_fast.py:744-750), so a bound step is `||z_w_t|| >= ratio * ||z_w_0||(1-eps)`.
    On a clamp-OFF arm the identical statistic is the "would have been clamped"
    fraction, which is what makes the OFF/ON pair interpretable.
    """
    ws = getattr(traj, "world_states", None) if traj is not None else None
    if not ws or len(ws) < 2:
        return None
    try:
        n0 = float(ws[0].detach().norm(dim=-1).reshape(-1)[0].item())
    except Exception:
        return None
    if not math.isfinite(n0) or n0 <= 0.0:
        return None
    ceiling = clamp_ratio * n0
    norms: List[float] = []
    for t in range(1, len(ws)):
        try:
            norms.append(float(ws[t].detach().norm(dim=-1).reshape(-1)[0].item()))
        except Exception:
            continue
    if not norms:
        return None
    # THREE separate statistics, because a single ">= ceiling" test is INVARIANT
    # under the manipulation it is supposed to measure: an unclamped rollout that
    # EXCEEDS the ceiling and a clamped one PINNED AT it both satisfy it. Measured
    # in the smoke: a ">= ceiling" fraction read 0.9667 on BOTH the clamp-OFF and
    # clamp-ON arm at seed 42, i.e. bit-identical and useless.
    #   at_ceiling_frac  ~0 when unclamped (an exact hit is a measure-zero
    #                    coincidence), high when the clamp is actively binding
    #   above_ceiling_frac  >0 only when unclamped -- the clamp forbids exceeding
    #   max_over_ceiling continuous, non-saturating: >1 unclamped, ==1 clamped
    tol = ceiling * PIN_EPS
    n_at = sum(1 for v in norms if math.isfinite(v) and abs(v - ceiling) <= tol)
    n_above = sum(1 for v in norms if math.isfinite(v) and v > ceiling + tol)
    finite = [v for v in norms if math.isfinite(v)]
    return {
        "zw0_norm": n0,
        "zw_last_norm": float(norms[-1]),
        "zw_max_norm": float(max(finite)) if finite else 0.0,
        "ceiling": float(ceiling),
        "at_ceiling_frac": float(n_at) / float(len(norms)),
        "above_ceiling_frac": float(n_above) / float(len(norms)),
        "max_over_ceiling": (float(max(finite)) / ceiling) if finite else 0.0,
    }


def run_cell(arm: Dict[str, Any], seed: int, dry_run: bool = False) -> Dict[str, Any]:
    arm_id = arm["id"]
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    n_eps = 2 if dry_run else N_EPISODES
    # Long enough that the smoke actually exercises the decisive readout (variance
    # shares need several fresh selections to exist at all). The pre-registered
    # 60-sample floor is deliberately NOT relaxed for the dry run, so a smoke
    # legitimately routes substrate_not_ready_requeue -- the smoke's job here is to
    # prove ENGAGEMENT (shares computed, clamp binding observed), which the
    # [smoke] line below reports explicitly.
    steps_per_ep = 60 if dry_run else STEPS_PER_EPISODE

    slice_for_cell = config_slice_for(arm)

    with arm_cell(
        seed,
        config_slice=slice_for_cell,
        script_path=Path(__file__),
        config_slice_declared=True,
        include_driver_script_in_hash=False,
    ) as cell:
        agent, env, obs_dict = make_env_and_agent(seed, arm)

        # Config-landing assert: REEConfig.from_dims silently swallows unknown
        # kwargs, so a typo would leave the clamp OFF with no error at all. This
        # reads the value the E2 rollout actually consults.
        clamp_live = bool(
            getattr(agent.e2.config, "e2_rollout_output_norm_clamp_enabled", False)
        )
        ratio_live = float(
            getattr(agent.e2.config, "e2_rollout_output_norm_clamp_ratio", CLAMP_RATIO)
        )

        pool_series: Dict[str, List[float]] = {}
        sel_series: Dict[str, List[float]] = {}
        score_ranges: List[float] = []
        at_ceiling_vals: List[float] = []
        above_ceiling_vals: List[float] = []
        max_over_ceiling_vals: List[float] = []
        zw0_vals: List[float] = []
        zw_max_vals: List[float] = []
        lambda_eff_vals: List[float] = []

        fs = FreshSelectCounter()
        n_ticks = 0

        all_components = tuple(sorted(set(SCORE_COMPONENTS) | set(LEGACY_571_COMPONENTS)))

        for ep in range(n_eps):
            for _ in range(steps_per_ep):
                ob = torch.tensor(np.asarray(obs_dict["body_state"])).float()
                ow = torch.tensor(np.asarray(obs_dict["world_state"])).float()

                with _FRESH_SELECT.watch(agent) as sel:
                    with torch.no_grad():
                        action = agent.act_with_split_obs(ob, ow)
                n_ticks += 1
                fs.record(bool(sel.fresh))

                if sel.fresh:
                    decomp = getattr(agent.e3, "last_score_decomp", None) or {}
                    per_cand = decomp.get("per_candidate")
                    sel_idx = decomp.get("selected_idx")
                    if isinstance(per_cand, list) and per_cand:
                        dict_cands = [c for c in per_cand if isinstance(c, dict)]
                        if dict_cands:
                            # 571 method: per-selection MEAN across candidates.
                            for comp in all_components:
                                vals = []
                                for c in dict_cands:
                                    try:
                                        v = float(c.get(comp, 0.0))
                                    except (TypeError, ValueError):
                                        continue
                                    if math.isfinite(v):
                                        vals.append(v)
                                pool_series.setdefault(comp, []).append(
                                    float(statistics.fmean(vals)) if vals else 0.0
                                )
                            # Committed method: AT the selected candidate.
                            if (
                                isinstance(sel_idx, int)
                                and 0 <= sel_idx < len(per_cand)
                                and isinstance(per_cand[sel_idx], dict)
                            ):
                                chosen = per_cand[sel_idx]
                                for comp in all_components:
                                    try:
                                        v = float(chosen.get(comp, 0.0))
                                    except (TypeError, ValueError):
                                        v = 0.0
                                    sel_series.setdefault(comp, []).append(
                                        v if math.isfinite(v) else 0.0
                                    )
                                try:
                                    le = float(chosen.get("lambda_eff", 0.0))
                                    if math.isfinite(le):
                                        lambda_eff_vals.append(le)
                                except (TypeError, ValueError):
                                    pass

                    # Scale anchor: raw E3 score RANGE across candidates.
                    ls = getattr(agent.e3, "last_scores", None)
                    if ls is not None and getattr(ls, "numel", lambda: 0)() > 1:
                        try:
                            r = float((ls.max() - ls.min()).item())
                            if math.isfinite(r):
                                score_ranges.append(r)
                        except Exception:
                            pass

                    # Scale anchor: clamp binding on the COMMITTED rollout.
                    ps = _pin_stats(
                        getattr(agent.e3, "_last_selected_trajectory", None), ratio_live
                    )
                    if ps is not None:
                        at_ceiling_vals.append(ps["at_ceiling_frac"])
                        above_ceiling_vals.append(ps["above_ceiling_frac"])
                        max_over_ceiling_vals.append(ps["max_over_ceiling"])
                        zw0_vals.append(ps["zw0_norm"])
                        zw_max_vals.append(ps["zw_max_norm"])

                if isinstance(action, torch.Tensor):
                    action_idx = int(action.argmax().item())
                else:
                    action_idx = int(action)
                _, _, terminated, _, obs_dict = env.step(action_idx % env.action_dim)
                if terminated:
                    agent.reset()
                    _, obs_dict = env.reset()

            fs.flush()
            print(
                f"  [train] monopoly-audit seed={seed} arm={arm_id} "
                f"ep {ep + 1}/{n_eps} fresh={fs.n_fresh_select} latched={fs.n_latched}",
                flush=True,
            )

        _ZG.observe(agent)

        # FABLE RED-TEAM FINDING 1, resolved by MEASUREMENT rather than by assumption.
        # V3-EXQ-936a -- the run in which residue_weighted took ~99.9998% of the
        # variance -- calls agent.update_residue() every env step (936a driver:760).
        # Neither V3-EXQ-571 nor this driver does (grep update_residue: 0/0/1). The
        # reviewer inferred from that that the RBF half of phi is "exactly zero
        # forever", which would make residue_weighted structurally unable to hold the
        # monopoly in EITHER arm here -- i.e. this run could not tell "the clamp does
        # not cause the residue inversion" from "this driver starved the channel".
        # That inference is NOT safe as stated: residue_field.accumulate() is also
        # reached from e3_selector.py:4083 whenever harm_occurred, and this config runs
        # NUM_HAZARDS=1, so the accumulation path is live even with no update_residue
        # call. Rather than assume either way, record the fact: active_mask.sum() is
        # the number of RBF centers actually recruited, and it is what a later reader
        # needs to know whether the residue channel was ever a candidate occupant.
        try:
            _rf = getattr(agent, "residue_field", None)
            _rbf = getattr(_rf, "rbf_field", None) if _rf is not None else None
            _mask = getattr(_rbf, "active_mask", None) if _rbf is not None else None
            n_residue_active = int(_mask.sum().item()) if _mask is not None else -1
        except Exception:
            n_residue_active = -1

        # --- decompose, both component sets, both methods ---
        pool_f, pool_fr, pool_total_var = _pool_variance_share(
            pool_series, SCORE_COMPONENTS, F_COMPONENTS
        )
        comm_f, comm_fr, comm_total = _committed_variance_share(
            sel_series, SCORE_COMPONENTS, F_COMPONENTS
        )
        legacy_pool_f, legacy_pool_fr, legacy_pool_total_var = _pool_variance_share(
            pool_series, LEGACY_571_COMPONENTS, LEGACY_F_COMPONENTS
        )
        legacy_comm_f, legacy_comm_fr, _ = _committed_variance_share(
            sel_series, LEGACY_571_COMPONENTS, LEGACY_F_COMPONENTS
        )

        pool_top, pool_top_share = _top_channel(pool_fr)
        comm_top, comm_top_share = _top_channel(comm_fr)
        legacy_pool_top, legacy_pool_top_share = _top_channel(legacy_pool_fr)

        row: Dict[str, Any] = {
            "arm": arm_id,
            "seed": int(seed),
            "load_bearing_arm": bool(arm["load_bearing"]),
            "clamp_requested": bool(arm["clamp"]),
            "clamp_live_on_e2": clamp_live,
            "clamp_ratio_live": ratio_live,
            "clamp_config_landed": bool(clamp_live == bool(arm["clamp"])),
            "stack_enabled": bool(arm["stack"]),
            "n_env_ticks": int(n_ticks),

            # corrected component set (the true additive score terms)
            "pool_f_share": float(pool_f),
            "pool_fractions": {k: float(v) for k, v in pool_fr.items()},
            "pool_total_variance": float(pool_total_var),
            "pool_top_channel": pool_top,
            "pool_top_share": float(pool_top_share),
            "committed_f_share": float(comm_f),
            "committed_fractions": {k: float(v) for k, v in comm_fr.items()},
            "committed_total_variance": float(comm_total),
            "committed_top_channel": comm_top,
            "committed_top_share": float(comm_top_share),

            # 571's legacy component set, for direct comparison with 0.886
            "legacy571_pool_f_share": float(legacy_pool_f),
            "legacy571_pool_fractions": {k: float(v) for k, v in legacy_pool_fr.items()},
            "legacy571_pool_total_variance": float(legacy_pool_total_var),
            "legacy571_pool_top_channel": legacy_pool_top,
            "legacy571_pool_top_share": float(legacy_pool_top_share),
            "legacy571_committed_f_share": float(legacy_comm_f),
            "legacy571_committed_fractions": {k: float(v) for k, v in legacy_comm_fr.items()},

            # scale anchors (the 936a prescription (b), implemented here)
            "score_range_mean": float(statistics.fmean(score_ranges)) if score_ranges else 0.0,
            "score_range_max": float(max(score_ranges)) if score_ranges else 0.0,
            "zw0_norm_mean": float(statistics.fmean(zw0_vals)) if zw0_vals else 0.0,
            "zw_max_norm_mean": float(statistics.fmean(zw_max_vals)) if zw_max_vals else 0.0,
            "rollout_at_ceiling_frac_mean": (
                float(statistics.fmean(at_ceiling_vals)) if at_ceiling_vals else 0.0
            ),
            "rollout_above_ceiling_frac_mean": (
                float(statistics.fmean(above_ceiling_vals)) if above_ceiling_vals else 0.0
            ),
            "rollout_max_over_ceiling_mean": (
                float(statistics.fmean(max_over_ceiling_vals)) if max_over_ceiling_vals else 0.0
            ),
            "rollout_max_over_ceiling_max": (
                float(max(max_over_ceiling_vals)) if max_over_ceiling_vals else 0.0
            ),
            "n_pin_samples": int(len(at_ceiling_vals)),
            "residue_rbf_active_centers": int(n_residue_active),
            "lambda_eff_mean": (
                float(statistics.fmean(lambda_eff_vals)) if lambda_eff_vals else 0.0
            ),
        }
        row.update(fs.as_dict(n_ticks))

        # MONOPOLY ROUTES ON THE COMMITTED PARTITION, NOT ON 571's POOL METHOD.
        # 571's method (inherited unchanged by 936/936a) divides each component's
        # variance by the variance of the SUM, retaining covariance -- so an
        # anti-correlated pair makes the denominator SMALLER than a numerator and
        # the "share" exceeds 1.0. Measured in this driver's own smoke:
        # harm_weighted "share" of 1.06 and 1.92. A quantity that can exceed 1 is
        # not a fraction of variance, so `>= 0.85` is not a monopoly test on it.
        # The committed method divides by the SUM of per-component variances -- a
        # genuine partition, bounded [0,1], summing to 1.0 -- which is what
        # "monopolises the variance" actually means. The pool/571 numbers are still
        # recorded in full, because they are what makes the 0.886 comparable.
        row["pool_share_exceeds_unity"] = bool(row["pool_top_share"] > 1.0)
        monopoly_present = bool(row["committed_top_share"] >= F_MONOPOLY_THRESHOLD)
        row["monopoly_present"] = monopoly_present
        row["monopoly_occupant_is_f"] = bool(
            monopoly_present and row["committed_top_channel"] in F_COMPONENTS
        )
        row["cell_decidable"] = bool(
            row["n_fresh_select"] >= MIN_FRESH_SELECTIONS
            and row["committed_total_variance"] > MIN_TOTAL_VARIANCE
        )

        cell.stamp(row)

    if dry_run:
        # Engagement assertion: the decisive readout must be non-trivially live
        # BEFORE the full 16-cell grid is committed to.
        print(
            f"  [smoke] arm={arm_id} seed={seed} clamp_live={row['clamp_live_on_e2']} "
            f"n_fresh={row['n_fresh_select']} n_latched={row['n_latched']} "
            f"repl={row['replication_factor']:.2f} "
            f"committed_var={row['committed_total_variance']:.6g} "
            f"ROUTED_top={row['committed_top_channel']}"
            f"({row['committed_top_share']:.6g}) "
            f"pool_top={row['pool_top_channel']}({row['pool_top_share']:.6g}"
            f"{',>1!' if row['pool_share_exceeds_unity'] else ''}) "
            f"at_ceil={row['rollout_at_ceiling_frac_mean']:.4f} "
            f"above_ceil={row['rollout_above_ceiling_frac_mean']:.4f} "
            f"max/ceil={row['rollout_max_over_ceiling_mean']:.4f} "
            f"score_range={row['score_range_mean']:.6g}",
            flush=True,
        )

    print(f"verdict: {'PASS' if row['cell_decidable'] else 'FAIL'}", flush=True)
    return row


# --------------------------------------------------------------------------- #
# Aggregation + interpretation                                                 #
# --------------------------------------------------------------------------- #

def _worst_cell(rows: List[Dict[str, Any]], key: str, use_min: bool = True):
    """Extremum plus the offending cell id, so `measured` recomputes exactly."""
    if not rows:
        return 0.0, "none"
    fn = min if use_min else max
    best = fn(rows, key=lambda r: float(r.get(key, 0.0)))
    return float(best.get(key, 0.0)), f"{best.get('arm')}/seed{best.get('seed')}"


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = SEEDS[:2] if dry_run else SEEDS
    rows: List[Dict[str, Any]] = []

    for arm in ARMS:
        for seed in seeds:
            rows.append(run_cell(arm, seed, dry_run=dry_run))

    lb_rows = [r for r in rows if r["arm"] == LOAD_BEARING_ARM]
    ref_rows = [r for r in rows if r["arm"] == REFERENCE_ARM]

    # --- readiness preconditions -------------------------------------------
    # SCOPED TO THE VERDICT-BEARING ARMS (A0 + A1), NOT to all four. Every
    # criterion routes on those two arms only; the div-stack pair is secondary and
    # recorded. Taking the worst cell across ALL arms would let a secondary arm's
    # shortfall vacate a clean, well-powered load-bearing result and self-route the
    # whole run substrate_not_ready_requeue -- the V3-EXQ-785 defect (an AND'd
    # whole-run gate burying a good arm). The secondary arms' readiness is recorded
    # below as a NON-GATING diagnostic so nothing is hidden either.
    verdict_rows = lb_rows + ref_rows
    secondary_rows = [r for r in rows if r not in verdict_rows]

    fresh_worst, fresh_cell = _worst_cell(verdict_rows, "n_fresh_select", use_min=True)
    var_worst, var_cell = _worst_cell(verdict_rows, "committed_total_variance", use_min=True)
    # Config landing is checked across ALL cells on purpose: it is a correctness
    # identity (did the manipulation exist at all), not a power question, and a
    # mis-landed clamp anywhere means the script is wrong rather than under-powered.
    landed = 1.0 if all(bool(r["clamp_config_landed"]) for r in rows) else 0.0

    preconditions = [
        {
            "name": "decomp_samples_sufficient",
            "description": (
                "genuine fresh E3 selections per cell (latch-gated, never per-env-step) "
                "-- the true denominator behind every variance share"
            ),
            "kind": "readiness",
            "measured": float(fresh_worst),
            "threshold": float(MIN_FRESH_SELECTIONS),
            "direction": "lower",
            "control": "worst cell across the VERDICT-BEARING arms (A0, A1)",
            "offending_cell": fresh_cell,
            "met": bool(fresh_worst >= MIN_FRESH_SELECTIONS),
        },
        {
            "name": "committed_total_variance_nondegenerate",
            "description": (
                "sum of per-component variances at the committed candidate -- "
                "literally the denominator the load-bearing share criterion divides by"
            ),
            "kind": "readiness",
            "measured": float(var_worst),
            "threshold": float(MIN_TOTAL_VARIANCE),
            "direction": "lower",
            "control": "worst cell across the VERDICT-BEARING arms (A0, A1)",
            "offending_cell": var_cell,
            "met": bool(var_worst > MIN_TOTAL_VARIANCE),
        },
        {
            "name": "clamp_config_landed_every_cell",
            "description": (
                "the clamp flag actually reached agent.e2.config in every cell and "
                "matches the arm's request (REEConfig.from_dims swallows unknown "
                "kwargs silently, so this is the only proof the manipulation exists)"
            ),
            "kind": "readiness",
            "measured": float(landed),
            "threshold": 1.0,
            "direction": "lower",
            "control": "all cells; 1.0 = every cell's live clamp equals its arm spec",
            "met": bool(landed >= 1.0),
        },
    ]
    readiness_ok = all(bool(p["met"]) for p in preconditions)

    def _mean(rs, k):
        vals = [float(r.get(k, 0.0)) for r in rs]
        return float(statistics.fmean(vals)) if vals else 0.0

    lb_top_share = _mean(lb_rows, "committed_top_share")
    ref_top_share = _mean(ref_rows, "committed_top_share")
    lb_tops = [r["committed_top_channel"] for r in lb_rows]
    ref_tops = [r["committed_top_channel"] for r in ref_rows]
    lb_top_modal = max(set(lb_tops), key=lb_tops.count) if lb_tops else "none"
    ref_top_modal = max(set(ref_tops), key=ref_tops.count) if ref_tops else "none"

    c1_monopoly_present = bool(lb_top_share >= F_MONOPOLY_THRESHOLD)
    c2_occupant_is_f = bool(c1_monopoly_present and lb_top_modal in F_COMPONENTS)
    n_flip = sum(
        1
        for a in ref_rows
        for b in lb_rows
        if a["seed"] == b["seed"]
        and a["committed_top_channel"] != b["committed_top_channel"]
    )
    c3_clamp_flips = bool(n_flip >= max(1, (len(lb_rows) + 1) // 2))

    criteria = [
        {
            "name": "C1_monopoly_present_clamped_baseline",
            "load_bearing": True,
            "passed": c1_monopoly_present,
            "detail": (
                f"mean top-channel share in {LOAD_BEARING_ARM} = {lb_top_share:.6g} "
                f"vs bar {F_MONOPOLY_THRESHOLD}"
            ),
        },
        {
            "name": "C2_monopoly_occupant_is_f",
            "load_bearing": False,
            "passed": c2_occupant_is_f,
            "detail": f"modal top channel clamped = {lb_top_modal}",
        },
        {
            "name": "C3_clamp_flips_monopoly_occupant",
            "load_bearing": False,
            "passed": c3_clamp_flips,
            "detail": (
                f"top channel OFF={ref_top_modal} ON={lb_top_modal}; "
                f"{n_flip}/{len(lb_rows)} matched seeds flip"
            ),
        },
    ]

    criteria_non_degenerate = {
        "C1_monopoly_present_clamped_baseline": bool(
            readiness_ok and bool(lb_rows) and all(r["cell_decidable"] for r in lb_rows)
        ),
        # The occupant question is only meaningful where a monopoly exists at all.
        "C2_monopoly_occupant_is_f": bool(readiness_ok and c1_monopoly_present),
        "C3_clamp_flips_monopoly_occupant": bool(
            readiness_ok
            and all(r["cell_decidable"] for r in lb_rows)
            and all(r["cell_decidable"] for r in ref_rows)
        ),
    }

    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
    elif c1_monopoly_present and c2_occupant_is_f and c3_clamp_flips:
        # The occupant is still inside F, but it is NOT the same channel the
        # unclamped reference holds. Reporting that as a bare
        # "f_monopoly_regime_present_clamped" would hide the shift from governance,
        # which is the one thing this audit exists to surface.
        label = "f_monopoly_present_clamped_occupant_shifted"
        outcome = "PASS"
    elif c1_monopoly_present and c2_occupant_is_f:
        label = "f_monopoly_regime_present_clamped"
        outcome = "PASS"
    elif c1_monopoly_present:
        label = "monopoly_present_non_f_occupant"
        outcome = "PASS"
    else:
        label = "no_monopoly_regime_under_fixed_instrument"
        outcome = "PASS"

    note_map = {
        "f_monopoly_present_clamped_occupant_shifted": (
            "Hb, with a caveat governance must not miss: an F-channel monopoly IS "
            "present with the clamp armed, but the channel holding it is not the one "
            "the unclamped reference holds -- the occupant shifted WITHIN F when the "
            "clamp was armed. MECH-439's F-level premise survives; its channel-level "
            "reading does not transfer across the instrument change. Any re-posed "
            "936-family falsifier must name WHICH F channel it targets."
        ),
        "f_monopoly_regime_present_clamped": (
            "Hb: 571's config exhibits an F-monopoly WITH the clamp armed, so MECH-439 "
            "is regime-scoped rather than refuted. A 936-family falsifier may be re-posed "
            "in THIS regime, with a monopoly-presence precondition and a RELATIVE bar."
        ),
        "monopoly_present_non_f_occupant": (
            "The structural monopoly (one channel dominates the un-normalised score sum) "
            "is conserved under the clamp, but its occupant is not F. MECH-439's "
            "F-SPECIFIC premise does not obtain under the fixed instrument in 571's own "
            "config; governance should re-word/narrow the claim to the structural form."
        ),
        "no_monopoly_regime_under_fixed_instrument": (
            "Ha: no channel monopolises committed-selection variance once the clamp is "
            "armed. MECH-439's quantitative premise (0.886) is an artefact of the "
            "unclamped instrument; governance re-words/narrows."
        ),
        "substrate_not_ready_requeue": (
            "Readiness precondition unmet -- the shares are not measurable in at least "
            "one cell. Re-queue at an adequate n; this run routes to no verdict."
        ),
    }

    # Whether the 936a monopolist channel was even RECRUITED in this run. If no RBF
    # center was ever activated in any cell, residue_weighted could not have held the
    # monopoly here regardless of the clamp -- so this run measures the monopoly in
    # 571's regime but does NOT adjudicate the 936a residue inversion, and any
    # governance reading must stop short of that. Stated from the measurement, not
    # from a premise about the driver.
    residue_active_max = max(
        [int(r.get("residue_rbf_active_centers", -1)) for r in rows] or [-1]
    )
    residue_scope_caveat = (
        " SCOPE LIMIT (residue channel never recruited): no RBF center was activated "
        "in any cell, so residue_weighted -- the channel that held ~99.9998% in "
        "V3-EXQ-936a -- was not an available occupant here at all. This run therefore "
        "characterises the monopoly WITHIN 571's non-accumulating regime and does NOT "
        "adjudicate whether the clamp causes the 936a residue inversion; that question "
        "needs a residue-feeding arm and is explicitly NOT answered by this run."
        if residue_active_max == 0
        else (
            f" Residue channel WAS recruited (max active RBF centers across cells = "
            f"{residue_active_max}), so residue_weighted was an available occupant and "
            f"the comparison against 936a's residue monopoly is meaningful."
            if residue_active_max > 0
            else " Residue recruitment UNMEASURED (probe unavailable); treat any "
                 "comparison against 936a's residue monopoly as unsupported."
        )
    )

    outcome_note = (
        f"{label}: clamped top channel {lb_top_modal} at share {lb_top_share:.6g}; "
        f"unclamped reference top channel {ref_top_modal} at share {ref_top_share:.6g}; "
        f"V3-EXQ-571 reported f=0.8863 unclamped in this config (571 pool method; "
        f"this run's pool-method figure for the same arm is "
        f"{_mean(ref_rows, 'legacy571_pool_top_share'):.6g} on channel "
        f"{(ref_rows[0]['legacy571_pool_top_channel'] if ref_rows else 'none')}). "
        f"Clamp binding: max||z_w||/ceiling {_mean(lb_rows, 'rollout_max_over_ceiling_mean'):.4f} "
        f"clamped vs {_mean(ref_rows, 'rollout_max_over_ceiling_mean'):.4f} unclamped; "
        f"above-ceiling fraction {_mean(lb_rows, 'rollout_above_ceiling_frac_mean'):.4f} "
        f"clamped vs {_mean(ref_rows, 'rollout_above_ceiling_frac_mean'):.4f} unclamped. "
        f"{note_map[label]}"
        f"{residue_scope_caveat}"
    )

    return {
        "outcome": outcome,
        "outcome_note": outcome_note,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "combination_rule": (
                "C1 alone routes the verdict (load-bearing). C2 refines WHICH channel "
                "holds a monopoly C1 found; C3 is descriptive attribution of any change "
                "to the clamp. C2 and C3 never override C1."
            ),
        },
        "criteria": criteria,
        "summary": {
            "clamped_top_channel_modal": lb_top_modal,
            "clamped_top_share_mean": lb_top_share,
            "unclamped_top_channel_modal": ref_top_modal,
            "unclamped_top_share_mean": ref_top_share,
            "v3_exq_571_reference_f_share": 0.8863118651034253,
            "clamped_at_ceiling_frac_mean": _mean(lb_rows, "rollout_at_ceiling_frac_mean"),
            "unclamped_at_ceiling_frac_mean": _mean(ref_rows, "rollout_at_ceiling_frac_mean"),
            "clamped_above_ceiling_frac_mean": _mean(
                lb_rows, "rollout_above_ceiling_frac_mean"
            ),
            "unclamped_above_ceiling_frac_mean": _mean(
                ref_rows, "rollout_above_ceiling_frac_mean"
            ),
            "clamped_max_over_ceiling_mean": _mean(lb_rows, "rollout_max_over_ceiling_mean"),
            "unclamped_max_over_ceiling_mean": _mean(
                ref_rows, "rollout_max_over_ceiling_mean"
            ),
            "n_pool_shares_exceeding_unity": int(
                sum(1 for r in rows if r.get("pool_share_exceeds_unity"))
            ),
            "n_seed_flips": int(n_flip),
            "residue_rbf_active_centers_max": int(residue_active_max),
            "adjudicates_936a_residue_inversion": bool(residue_active_max > 0),
        },
        "diagnostics": {
            # NON-GATING readiness for the secondary (div-stack) arms. Recorded so a
            # reader can see their power without letting them vacate the verdict.
            "secondary_arm_readiness": [
                {
                    "cell": f"{r['arm']}/seed{r['seed']}",
                    "n_fresh_select": int(r["n_fresh_select"]),
                    "committed_total_variance": float(r["committed_total_variance"]),
                    "decidable": bool(r["cell_decidable"]),
                }
                for r in secondary_rows
            ],
            "share_method_note": (
                "The ROUTED statistic is the committed partition "
                "(sum-of-per-component-variances denominator, bounded [0,1]). "
                "V3-EXQ-571's 0.886 was computed with the covariance-retained "
                "pool method, which is a DIFFERENT statistic and can exceed 1.0 "
                "(this run flags each such cell as pool_share_exceeds_unity). "
                "Compare the 0.886 against legacy571_pool_top_share, NEVER against "
                "the C1 bar. The 0.85 bar is deliberately left where 936a set it "
                "rather than re-derived, because moving a pre-registered threshold "
                "to fit a new statistic would fabricate it."
            ),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-571b: clamped E3 variance-monopoly presence audit"
    )
    parser.add_argument("--dry-run", action="store_true", help="Short run for smoke testing")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = run_experiment(dry_run=args.dry_run)

    timestamp = __import__("datetime").datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    out_dir = (
        Path(__file__).parent.parent.parent / "REE_assembly" / "evidence" / "experiments"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds_used = SEEDS[:2] if args.dry_run else SEEDS
    full_config = {
        "seeds": seeds_used,
        "n_episodes": N_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "n_steps": N_STEPS,
        "grid_size": GRID_SIZE,
        "num_hazards": NUM_HAZARDS,
        "alpha_world": ALPHA_WORLD,
        "clamp_ratio": CLAMP_RATIO,
        "f_monopoly_threshold": F_MONOPOLY_THRESHOLD,
        "min_fresh_selections": MIN_FRESH_SELECTIONS,
        "min_total_variance": MIN_TOTAL_VARIANCE,
        "arms": [dict(a) for a in ARMS],
        "score_components": list(SCORE_COMPONENTS),
        "legacy571_components": list(LEGACY_571_COMPONENTS),
        "arm_config_slices": {a["id"]: config_slice_for(a) for a in ARMS},
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": ["MECH-439"],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "diagnostic",
        "outcome": result["outcome"],
        "outcome_note": result["outcome_note"],
        "timestamp_utc": timestamp,
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
        "criteria": result["criteria"],
        "summary": result["summary"],
        "diagnostics": result["diagnostics"],
        "custom_information": {
            "audits": "V3-EXQ-571 monopoly premise underlying MECH-439",
            "reference_run": "v3_exq_571_e3_score_variance_decomp_20260516T004017Z_v3",
            "reference_f_share_arm0": 0.8863118651034253,
            "predecessor_autopsy": "failure_autopsy_V3-EXQ-936a_2026-08-30",
            "closure_node": "behavioral_diversity_isolation:GAP-I",
            "instrument_fixes": [
                "latch-gated fresh selection (571 was ~10x pseudo-replicated)",
                "decomposes f_weighted (the true additive score term) as well as 571's raw f",
                "clamp-binding + score-range + z_world scale anchors (936a prescription b)",
            ],
            "brake_count_note": (
                "MECH-439 carries 13 counted ceiling autopsies, but this is a DIAGNOSTIC "
                "on the MEASUREMENT axis, which the 936a autopsy Section 8 explicitly "
                "exempted from the re-derive brake. Reconcile note carried forward from "
                "that autopsy Section 9: it counts 9 ceiling hits (matching claims.yaml's "
                "2026-08-10 correction) while the 936a driver's inherited "
                "custom_information says 12/13. NOT silently resolved here -- reported "
                "for GOV-APPLY-1, and no ceiling hit is added by this run either way."
            ),
        },
    }

    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=seeds_used,
        script_path=Path(__file__),
        started_at=t0,
    )

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=bool(args.dry_run),
        config=full_config,
        seeds=seeds_used,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
        json_default=str,
    )
    print(f"Manifest written: {out_path}")

    s = result["summary"]
    print(
        "SUMMARY: clamped top="
        f"{s['clamped_top_channel_modal']}({s['clamped_top_share_mean']:.6g}) "
        f"unclamped top={s['unclamped_top_channel_modal']}"
        f"({s['unclamped_top_share_mean']:.6g}) "
        f"571 ref f={s['v3_exq_571_reference_f_share']:.4f} "
        f"max/ceil on={s['clamped_max_over_ceiling_mean']:.4f} "
        f"off={s['unclamped_max_over_ceiling_mean']:.4f} "
        f"above_ceil on={s['clamped_above_ceiling_frac_mean']:.4f} "
        f"off={s['unclamped_above_ceiling_frac_mean']:.4f} "
        f"residue_active_max={s['residue_rbf_active_centers_max']} "
        f"adjudicates_936a_residue={s['adjudicates_936a_residue_inversion']}",
        flush=True,
    )
    print(f"LABEL: {result['interpretation']['label']}", flush=True)

    if args.dry_run:
        print("DRY RUN complete.")

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=bool(args.dry_run),
    )
