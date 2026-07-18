"""V3-EXQ-785: MECH-463 -- arousal as a channel-agnostic VARIANCE AMPLIFIER of the
incumbent selection channel. Two-regime incumbent-share decomposition.

CLAIM UNDER TEST (MECH-463): the three affective routes that modulate E3 commitment
GLOBALLY -- D1/D2 dopamine gain, harm-urgency threshold shrinkage, LC-NE / phasic-surprise
temperature -- each apply ONE SCALAR uniformly across all K candidates. A uniform scalar
cannot reorder an argmax, but shrinking the commit threshold makes commitment fire on
whichever candidate already has the largest score separation. That is CHANNEL-AGNOSTIC
variance amplification: arousal does not bias toward any particular channel, it selects
for whichever channel already dominates.

=== WHY TWO REGIMES (the design correction, 2026-07-18) ===
MECH-463 as registered adds "...which, under F-dominance (MECH-439), IS F". That last
clause is a CONTINGENT FACT supplied by MECH-439, not part of the mechanism. The
mechanism is channel-agnostic, so the right test does not require F to be the incumbent
-- it requires only that SOME channel is, and it is far sharper if run at more than one
incumbent identity.

Measured cross-candidate incumbent shares on committed ticks (250 ticks, seed 0,
241 committed; shares sum to exactly 1.0 -- full covariance attribution):

    regime                     incumbent          share     f
    entropy_bias_scale=0.0     harm_weighted      +0.8230   +0.1458
    entropy_bias_scale=1.0     CH:mech341         +1.0429   -0.0089

So the substrate offers TWO well-formed regimes with DIFFERENT incumbents. Note F is NOT
the cross-candidate incumbent in EITHER -- harm_weighted is. MECH-439's ~88-89% F-share is
a TEMPORAL statistic (V3-EXQ-571's compute_variance_decomp(): np.var over per-STEP arrays
of candidate-MEAN components), whereas MECH-463's mechanism is CROSS-CANDIDATE. Those are
incommensurable, so this experiment does NOT import the 88-89% figure as a null; it
pre-registers each regime's incumbent identity from the measurement above and asserts it.

DISCRIMINATING PREDICTION: the INCUMBENT's share of committed-selection score variance
rises monotonically with the urgency_applied decile, IN BOTH REGIMES. Three outcomes,
all distinguishable (which the original single-regime F-share design could not do):

  SUPPORTS    -- incumbent share rises in BOTH regimes -> channel-agnostic variance
                 amplification, the claim's actual mechanism.
  MIXED/REFINE-- rises only in the harm-incumbent regime -> the effect is channel-SPECIFIC,
                 not channel-agnostic; MECH-463 needs narrowing, not killing.
  REFUTES     -- flat in BOTH with the non-vacuity gate GREEN -> arousal is causally inert
                 w.r.t. the selection-variance distribution. Kills MECH-463 cleanly and
                 leaves MECH-439 untouched.
  INVALID     -- gate RED -> substrate_not_ready_requeue, NOT a refutation.

SCOPE NOTE ON WHAT A PASS LICENSES: MECH-463 as registered ties its CONSEQUENCE to
F-entrenchment ("this makes affective engagement ENTRENCH F rather than convert
diversity"). This experiment establishes the MECHANISM (channel-agnostic amplification of
the incumbent) but NOT that applied consequence, because the substrate offers no
F-incumbent cross-candidate regime at all (F is 0.146 at best). Do not read a PASS as
direct evidence about the conversion ceiling; it is evidence about the mechanism the
ceiling argument invokes.

WHY THIS NEEDED NEW INSTRUMENTATION (audit 2026-07-18): urgency_applied escaped only as
SelectionResult.urgency; effective_threshold and commit_variance were pure locals that
died at the end of select(); per-channel bias was retained only as the SUMMED tensor
(agent.py:6816-6845 keeps reduced scalars, losing channel-incumbent covariance). Landed
2026-07-18 in e3_selector.py (ree-v3 435322f), behind e3_score_decomp_enabled.
GOV-REUSE-1: reanalysis_query.py --readout urgency_applied returns no MATCH (all
candidates UNVERIFIABLE, no recoverable substrate_hash) and the field did not exist as a
recorded quantity before that commit -> not recoverable, must run.

=== MANDATORY NON-VACUITY GATE (inherited from V3-EXQ-643a, plus two additions) ===
V3-EXQ-643 FAILed precondition_unmet purely from float32 catastrophic cancellation (E3
scores ~1e32, modulatory range below float32 ULP); 643a added this gate. Without it a
NUMERICAL degeneracy returns a spurious FLAT profile INDISTINGUISHABLE from a genuine
refutation. Assertions, all measured on the run, all FLOORS:
  P1  modulatory_authority_active_frac > floor  (inherited from 643a, but applied ONLY to
      regimes whose incumbent is a modulatory CHANNEL. In a primary-component regime the
      modulatory channels legitimately contribute ~0 and the authority gate never fires,
      so asserting it there would make that regime structurally un-passable and silently
      collapse the two-regime design back to one.)
  P2  cross-candidate RANGE of the INCUMBENT component > floor (643a's channel-range check
      generalised to whichever component the decomposition actually routes on. Still a
      RANGE, never a magnitude/mean-abs proxy -- that substitution is the 643 GAP.)
  P7  >= 2 components hold |share| > 0.01 (a decomposition where only one component is
      non-trivial is arithmetically forced, not measured)
  P3  urgency_applied NON-CONSTANT across the run       (new: no variation -> no deciles)
  P4  total cross-candidate score variance > floor      (a share of ~0 variance is undefined)
  P5  COMMITTED-tick count > floor                      (new, and the one the inherited
      gate misses entirely -- see the 396a note below; a probe about COMMITTED-selection
      variance with zero committed ticks is vacuous in a way P1-P4 all read as GREEN)
  P6  the pre-registered incumbent IS the incumbent in its regime (guards against
      scoring "the incumbent's share rises" when the incumbent is not who we think)
Any RED -> outcome FAIL, label substrate_not_ready_requeue, non_degenerate=false.
NEVER a substrate-verdict label.

=== TWO DRIVER CONSTRAINTS THAT SILENTLY VACATE THIS PROBE ===
(1) act_with_split_obs() calls sense(obs_body, obs_world) with NO obs_harm_a, so z_harm_a
    is None on that path and urgency_applied is pinned at 0. The harm stream MUST be fed
    explicitly and the wrapper replicated: clock.advance() -> _e1_tick ->
    generate_trajectories -> select_action.
(2) V3-EXQ-396a: update_running_variance() has NO CALLER ANYWHERE IN ree_core. So
    _running_variance stays pinned at precision_init (0.5) forever while
    effective_threshold tops out near 0.39 -- the commit gate NEVER fires and every tick
    is uncommitted. The driver must drive the EMA itself. Measured: commit rate
    0.000 -> 0.964. Do not "simplify" either of these away.

CONFIG NOTES:
  - urgency_weight=0.12, NOT 0.5. At 0.5 urgency SATURATES against urgency_max
    (median == p90 == 0.5), collapsing the top deciles into a single value.
  - SD-056 e2_action_contrastive + rollout clamp keep E3 scores bounded (~0.034 raw range)
    rather than the ~1e32 that killed 643.

INTERPRETATION CAVEAT (carried into the manifest): under an under-differentiated z_world
(participation ratio ~1.06 at world_dim=128) the per-candidate channels have little to
range over. That objection bites the FIX, not the measurement -- but a REFUTES verdict
must be read against it rather than as unconditional.

SCOPE: NOT gated on MECH-457 or INV-088 (a decomposition probe on score geometry, not a
behavioural conversion test, so it does not require the agent to be competent), and does
NOT re-open the 689/485/445/625 selection-face lineages -- the conversion-ceiling
re-derive brake stays respected. claim_ids = [MECH-463] ONLY; MECH-439 carries 11
substrate_ceiling autopsies and is deliberately NOT tagged.

Arms: 2 regimes x 5 seeds, observational within each regime (the conditioning variable
urgency_applied is ENDOGENOUS). The regime is a substrate-config contrast, not a
manipulation of arousal.

Run:
  /opt/local/bin/python3 experiments/v3_exq_785_mech463_arousal_variance_amplifier_decomp.py --dry-run
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# ------------------------------------------------------------------ #
# Identity                                                           #
# ------------------------------------------------------------------ #
EXPERIMENT_TYPE = "v3_exq_785_mech463_arousal_variance_amplifier_decomp"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["MECH-463"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

OUT_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# ------------------------------------------------------------------ #
# Pre-registered constants (NOT derived from the run's own statistics) #
# ------------------------------------------------------------------ #
SEEDS = [0, 1, 2, 3, 4]
TICKS_PER_SEED = 800
DRY_RUN_TICKS = 60

# Primary per-candidate score components exposed by last_score_decomp["per_candidate"].
PRIMARY_COMPONENTS = [
    "f", "harm_weighted", "residue_weighted",
    "benefit_weighted", "novelty_weighted", "goal_weighted",
]

# The two regimes, each with its PRE-REGISTERED incumbent (measured 2026-07-18, seed 0,
# 241 committed ticks; asserted at run time by precondition P6).
REGIMES = [
    {"id": "harm_incumbent", "entropy_bias_scale": 0.0,
     "expected_incumbent": "harm_weighted", "expected_incumbent_share": 0.823},
    {"id": "entropy_incumbent", "entropy_bias_scale": 1.0,
     "expected_incumbent": "CH:mech341", "expected_incumbent_share": 1.043},
]

N_DECILES = 10
MIN_TICKS_PER_DECILE = 15
MIN_DECILES_POPULATED = 6

# Non-vacuity gate floors (all FLOORS -- direction "lower").
AUTHORITY_ACTIVE_FRAC_FLOOR = 0.05
CHANNEL_RANGE_FLOOR = 1e-4      # same statistic + floor family as 643a
URGENCY_STD_FLOOR = 1e-3
URGENCY_MIN_DISTINCT = 10
SCORE_VARIANCE_FLOOR = 1e-12
COMMITTED_TICK_FLOOR = float(N_DECILES * MIN_TICKS_PER_DECILE)  # P5
INCUMBENT_MARGIN_FLOOR = 0.10   # P6: incumbent must lead the runner-up by this much

# PASS criteria.
MONOTONIC_RHO_FLOOR = 0.60
GAP_ABS_FLOOR = 0.02
GAP_SD_MULTIPLIER = 2.0

URGENCY_WEIGHT = 0.12
MODULATORY_AUTHORITY_GAIN = 0.5
ALPHA_WORLD = 0.9


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #
def _spearman_rho(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation. Returns 0.0 when undefined (n < 3 or a flat side)."""
    n = len(x)
    if n < 3:
        return 0.0

    def _rank(v: List[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return float(num / (dx * dy))


def _component_shares(components: Dict[str, np.ndarray]) -> Optional[Dict[str, float]]:
    """Covariance-correct cross-candidate variance share for EVERY component.

    score_k = sum_c C_ck. Attributing to component c its own variance plus its full
    covariance with the rest:
        share_c = (Var(C_c) + Cov(C_c, sum_{d != c} C_d)) / Var(score)
    These sum to EXACTLY 1, since
        sum_c [Var(C_c) + Cov(C_c, rest_c)] = sum_c Var(C_c) + 2 sum_{c<d} Cov(C_c, C_d)
                                            = Var(sum_c C_c).
    Retaining the per-candidate [K] tensors unreduced is what makes the covariance terms
    computable at all -- marginal variances alone cannot form them, and dropping the
    cross-terms makes the "shares" fail to sum to 1 (the reduced-scalar failure mode).
    """
    keys = [k for k, v in components.items() if v.size >= 2]
    if not keys:
        return None
    n = components[keys[0]].size
    total = np.zeros(n, dtype=float)
    for k in keys:
        if components[k].size != n:
            return None
        total = total + components[k]
    var_total = float(np.var(total))
    if not np.isfinite(var_total) or var_total <= SCORE_VARIANCE_FLOOR:
        return None
    out: Dict[str, float] = {}
    for k in keys:
        c = components[k]
        rest = total - c
        cov = float(np.mean((c - c.mean()) * (rest - rest.mean())))
        out[k] = (float(np.var(c)) + cov) / var_total
    out["__var_total__"] = var_total
    return out


def _build_agent_and_env(seed: int, regime: Dict[str, Any]):
    # CausalGridWorld carries its OWN np.random.default_rng(seed) -- omitting seed= here
    # makes the run non-reproducible (and any bit-identity check meaningless).
    env = CausalGridWorldV2(use_proxy_fields=True, seed=seed, hazard_harm=0.5)
    _obs, obs_dict = env.reset()

    kw: Dict[str, Any] = dict(
        body_obs_dim=obs_dict["body_state"].shape[-1],
        world_obs_dim=obs_dict["world_state"].shape[-1],
        action_dim=env.action_dim,
        alpha_world=ALPHA_WORLD,
        # SD-011 affective harm stream -> z_harm_a -> urgency_applied. Without BOTH,
        # urgency_applied is pinned at 0 and the probe is vacuous (P3 RED).
        use_harm_stream=True,
        use_affective_harm_stream=True,
        urgency_weight=URGENCY_WEIGHT,
        use_support_preserving_cem=True,
        support_preserving_min_first_action_classes=2,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        # SD-056: action-divergent candidates + the clamp keeping E3 scores bounded.
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
    )
    es = float(regime["entropy_bias_scale"])
    if es > 0.0:
        # MECH-341 entropy bonus ON -> it becomes the incumbent (measured share ~1.04).
        kw.update(
            use_e3_score_diversity=True,
            use_e3_diversity_entropy_bonus=True,
            use_e3_diversity_stratified_select=False,
            e3_diversity_entropy_bias_scale=es,
        )
    else:
        # OFF -> harm_weighted is the incumbent (measured share ~0.82) and the modulatory
        # channels contribute ~0 cross-candidate variance.
        kw.update(use_e3_score_diversity=False, use_e3_diversity_entropy_bonus=False)

    cfg = REEConfig.from_dims(**kw)
    # Split last_channel_terms into NAMED channels rather than one compressed
    # "score_bias" slot -- the decomposition needs the parts separable.
    cfg.e3.use_finer_channel_gating = True

    agent = REEAgent(cfg)
    agent.eval()
    agent.e3.e3_score_decomp_enabled = True  # MECH-463 instrumentation gate
    return agent, env, obs_dict, kw


def _collect_cell(seed: int, regime: Dict[str, Any], n_ticks: int):
    """One (regime, seed) cell. Returns per-tick rows + telemetry + the config slice."""
    agent, env, obs_dict, cfg_slice = _build_agent_and_env(seed, regime)

    rows: List[Dict[str, Any]] = []
    authority_hits = 0
    channel_ranges: List[float] = []
    incumbent_ranges: List[float] = []
    n_select = 0
    # 396a: the agent loop never calls update_running_variance(), so the driver must.
    z_world_prev: Optional[torch.Tensor] = None
    action_prev: Optional[torch.Tensor] = None

    print(f"Seed {seed} Condition {regime['id']}", flush=True)

    for tick in range(n_ticks):
        with torch.no_grad():
            # Explicit harm feed -- act_with_split_obs() would drop obs_harm_a.
            latent = agent.sense(
                obs_dict["body_state"].unsqueeze(0),
                obs_dict["world_state"].unsqueeze(0),
                obs_harm=obs_dict.get("harm_obs"),
                obs_harm_a=obs_dict.get("harm_obs_a"),
                obs_harm_history=obs_dict.get("harm_history"),
            )
            z_world_cur = latent.z_world.detach()
            if z_world_prev is not None and action_prev is not None:
                _pred = agent.e2.world_forward(z_world_prev, action_prev)
                agent.e3.update_running_variance(z_world_cur - _pred.detach())
            ticks_d = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks_d["e1_tick"]
                else torch.zeros(1, agent.config.latent.world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks_d)
            action = agent.select_action(candidates, ticks_d, 1.0)
        agent._step_count += 1

        diag = agent.e3.last_score_diagnostics or {}
        decomp = agent.e3.last_score_decomp or {}
        chan = agent.e3.last_channel_terms or {}

        if "urgency_applied" in diag:
            n_select += 1
            if float(diag.get("modulatory_authority_active", 0.0) or 0.0) > 0.0:
                authority_hits += 1

            per_cand = decomp.get("per_candidate") or []
            comps: Dict[str, np.ndarray] = {}
            for name in PRIMARY_COMPONENTS:
                comps[name] = np.asarray(
                    [float(c.get(name, 0.0) or 0.0) for c in per_cand], dtype=float
                )
            for cname, cvec in chan.items():
                v = cvec.detach().cpu().numpy().astype(float)
                comps["CH:" + cname] = v
                if v.size >= 2:
                    channel_ranges.append(float(v.max() - v.min()))

            inc = regime["expected_incumbent"]
            if inc in comps and comps[inc].size >= 2:
                incumbent_ranges.append(float(comps[inc].max() - comps[inc].min()))
            shares = _component_shares(comps) if len(per_cand) >= 2 else None
            if shares is not None:
                var_total = shares.pop("__var_total__")
                rows.append({
                    "regime": regime["id"],
                    "seed": seed,
                    "tick": tick,
                    "urgency_applied": float(diag["urgency_applied"]),
                    "effective_threshold": float(diag.get("effective_threshold", 0.0)),
                    "commit_variance": float(diag.get("commit_variance", 0.0)),
                    "commit_gate_mode": str(diag.get("commit_gate_mode", "")),
                    "committed": bool(diag.get("committed", False)),
                    "temperature": float(
                        diag.get("gap_scaled_commit_temperature_eff", 1.0) or 1.0
                    ),
                    "da": float(diag.get("loop_d1_d2_conflict_signal", 0.0) or 0.0),
                    "var_total": round(var_total, 12),
                    "shares": {k: round(v, 6) for k, v in shares.items()},
                    "per_candidate": {
                        k: [round(x, 6) for x in v.tolist()] for k, v in comps.items()
                    },
                })

        if (tick + 1) % 50 == 0 or tick == n_ticks - 1:
            print(f"  [train] mech463 {regime['id']} seed={seed} ep {tick + 1}/{n_ticks} "
                  f"rows={len(rows)}", flush=True)

        act_idx = int(action.argmax().item()) if isinstance(action, torch.Tensor) else int(action)
        action_prev = torch.zeros(1, env.action_dim)
        action_prev[0, act_idx % env.action_dim] = 1.0
        z_world_prev = z_world_cur
        _obs, _r, done, _info, obs_dict = env.step(act_idx % env.action_dim)
        if done:
            _obs, obs_dict = env.reset()

    telemetry = {
        "regime": regime["id"],
        "seed": seed,
        "n_select_calls": n_select,
        "n_rows": len(rows),
        "n_committed": sum(1 for r in rows if r["committed"]),
        "authority_active_frac": (authority_hits / n_select) if n_select else 0.0,
        "channel_range_mean": float(np.mean(channel_ranges)) if channel_ranges else 0.0,
        "incumbent_range_mean": (
            float(np.mean(incumbent_ranges)) if incumbent_ranges else 0.0),
    }
    return rows, telemetry, cfg_slice


def _decile_profile(rows: List[Dict[str, Any]], incumbent: str) -> Dict[str, Any]:
    """Bin COMMITTED ticks by urgency decile; average the incumbent's share in each."""
    committed = [r for r in rows if r["committed"] and incumbent in r["shares"]]
    if not committed:
        return {"deciles": [], "n_committed": 0}
    urg = np.asarray([r["urgency_applied"] for r in committed], dtype=float)
    edges = np.quantile(urg, np.linspace(0.0, 1.0, N_DECILES + 1))
    deciles = []
    for d in range(N_DECILES):
        lo, hi = edges[d], edges[d + 1]
        if d == N_DECILES - 1:
            sel = [r for r, u in zip(committed, urg) if lo <= u <= hi]
        else:
            sel = [r for r, u in zip(committed, urg) if lo <= u < hi]
        if len(sel) < MIN_TICKS_PER_DECILE:
            deciles.append({"decile": d, "n": len(sel), "scored": False,
                            "urgency_lo": float(lo), "urgency_hi": float(hi)})
            continue
        vals = [r["shares"][incumbent] for r in sel]
        deciles.append({
            "decile": d, "n": len(sel), "scored": True,
            "urgency_lo": float(lo), "urgency_hi": float(hi),
            "urgency_mean": float(np.mean([r["urgency_applied"] for r in sel])),
            "incumbent_share_mean": float(np.mean(vals)),
            "incumbent_share_sd": float(np.std(vals)),
            "var_total_mean": float(np.mean([r["var_total"] for r in sel])),
        })
    return {"deciles": deciles, "n_committed": len(committed)}


def _mean_shares(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    committed = [r for r in rows if r["committed"]]
    acc: Dict[str, List[float]] = {}
    for r in committed:
        for k, v in r["shares"].items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _analyse_regime(regime, rows, telem_list) -> Dict[str, Any]:
    """Per-regime gate + monotonicity analysis."""
    incumbent = regime["expected_incumbent"]
    mean_shares = _mean_shares(rows)

    # P6: is the pre-registered incumbent actually the incumbent?
    ranked = sorted(mean_shares.items(), key=lambda kv: -kv[1])
    actual_incumbent = ranked[0][0] if ranked else ""
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    incumbent_margin = (mean_shares.get(incumbent, 0.0) - runner_up) if ranked else 0.0

    auth = float(np.mean([t["authority_active_frac"] for t in telem_list])) if telem_list else 0.0
    chan = float(np.mean([t["channel_range_mean"] for t in telem_list])) if telem_list else 0.0
    urg_all = [r["urgency_applied"] for r in rows]
    urg_std = float(np.std(urg_all)) if urg_all else 0.0
    urg_distinct = len({round(u, 9) for u in urg_all})
    var_total_mean = float(np.mean([r["var_total"] for r in rows])) if rows else 0.0
    n_committed = sum(1 for r in rows if r["committed"])

    def _p(name, desc, control, measured, threshold, met=None):
        return {"name": f"{regime['id']}::{name}", "kind": "readiness",
                "description": desc, "control": control,
                # All FLOORs. Declared explicitly so the indexer recompute cannot
                # default-misread them (the 2026-06-07 648a/649 directionality bug).
                "direction": "lower",
                "measured": float(measured), "threshold": float(threshold),
                "met": bool(measured > threshold) if met is None else bool(met)}

    inc_range = (float(np.mean([t["incumbent_range_mean"] for t in telem_list]))
                 if telem_list else 0.0)
    n_nontrivial = sum(1 for v in mean_shares.values() if abs(v) > 0.01)

    preconditions = []
    # P1 applies ONLY where the incumbent is a MODULATORY CHANNEL. In a
    # primary-component regime (harm_incumbent) the modulatory channels legitimately
    # contribute ~0 and the authority gate never fires -- asserting it there would make
    # that regime's gate structurally un-passable and collapse the two-regime design.
    if incumbent.startswith("CH:"):
        preconditions.append(_p(
            "modulatory_authority_active_frac",
            "fraction of selection ticks where the authority gate fired (channel-incumbent "
            "regimes only -- N/A when the incumbent is a primary score component)",
            "candidates that genuinely differ", auth, AUTHORITY_ACTIVE_FRAC_FLOOR))
    preconditions += [
        # Generalised from 643a's channel-range check: assert the range of whichever
        # component the decomposition actually routes on. Still a RANGE, never a
        # magnitude/mean-abs proxy -- that substitution is the 643 GAP.
        _p("incumbent_cross_candidate_range",
           f"mean cross-candidate RANGE of the incumbent component '{incumbent}' -- the "
           f"SAME statistic the decomposition routes on",
           "SD-056 action-contrastive candidates", inc_range, CHANNEL_RANGE_FLOOR),
        _p("n_components_with_nontrivial_share",
           "number of components holding |share| > 0.01; a decomposition where only one "
           "component is non-trivial is arithmetically forced, not measured",
           "multi-component primary score", float(n_nontrivial), 1.5),
        _p("urgency_applied_non_constant",
           "SD of urgency_applied; no variation means no deciles and a vacuous probe",
           "SD-011 harm stream fed explicitly via sense()",
           urg_std, URGENCY_STD_FLOOR,
           met=(urg_std > URGENCY_STD_FLOOR and urg_distinct >= URGENCY_MIN_DISTINCT)),
        _p("cross_candidate_score_variance",
           "mean total cross-candidate score variance (a share of ~0 is undefined)",
           "SD-056 action-contrastive candidates + rollout clamp",
           var_total_mean, SCORE_VARIANCE_FLOOR),
        _p("committed_tick_count",
           "committed ticks available to score. The inherited 643a gate MISSES this: "
           "with update_running_variance uncalled (396a) every tick is uncommitted while "
           "P1-P4 all read GREEN, so a probe about COMMITTED-selection variance silently "
           "scores nothing",
           "driver drives the running-variance EMA per 396a",
           float(n_committed), COMMITTED_TICK_FLOOR),
        _p("incumbent_identity_as_preregistered",
           f"margin of pre-registered incumbent '{incumbent}' over the runner-up; guards "
           f"against scoring 'the incumbent's share rises' for the wrong incumbent "
           f"(observed top: '{actual_incumbent}')",
           "measured 2026-07-18 seed 0, 241 committed ticks",
           incumbent_margin, INCUMBENT_MARGIN_FLOOR,
           met=(actual_incumbent == incumbent and incumbent_margin > INCUMBENT_MARGIN_FLOOR)),
    ]
    gate_green = all(p["met"] for p in preconditions)

    pooled = _decile_profile(rows, incumbent)
    scored = [d for d in pooled["deciles"] if d.get("scored")]
    rho = gap = gap_sd = 0.0
    gap_floor = GAP_ABS_FLOOR
    strictly_monotonic = False
    per_seed_gaps: List[float] = []
    for t in telem_list:
        srows = [r for r in rows if r["seed"] == t["seed"]]
        sp = _decile_profile(srows, incumbent)
        ss = [d for d in sp["deciles"] if d.get("scored")]
        if len(ss) >= 2:
            per_seed_gaps.append(
                float(ss[-1]["incumbent_share_mean"] - ss[0]["incumbent_share_mean"]))
    if len(scored) >= 2:
        xs = [float(d["decile"]) for d in scored]
        ys = [float(d["incumbent_share_mean"]) for d in scored]
        rho = _spearman_rho(xs, ys)
        gap = float(ys[-1] - ys[0])
        strictly_monotonic = all(ys[i + 1] >= ys[i] for i in range(len(ys) - 1))
        gap_sd = float(np.std(per_seed_gaps)) if len(per_seed_gaps) >= 2 else 0.0
        # Effect-size floor: scale noise on the SD of the DELTA, plus an absolute floor.
        gap_floor = max(GAP_ABS_FLOOR, GAP_SD_MULTIPLIER * gap_sd)

    rises = bool(len(scored) >= MIN_DECILES_POPULATED
                 and rho >= MONOTONIC_RHO_FLOOR and gap > gap_floor)

    return {
        "regime": regime["id"],
        "expected_incumbent": incumbent,
        "observed_incumbent": actual_incumbent,
        "incumbent_margin": incumbent_margin,
        "mean_shares": mean_shares,
        "gate_green": gate_green,
        "preconditions": preconditions,
        "decile_profile": pooled,
        "deciles_scored": len(scored),
        "spearman_rho": rho,
        "top_minus_bottom": gap,
        "gap_floor_applied": gap_floor,
        "gap_sd_across_seeds": gap_sd,
        "strictly_monotonic": strictly_monotonic,
        "incumbent_share_rises": rises,
        "n_committed": n_committed,
        "per_seed_gaps": per_seed_gaps,
    }


def run_experiment(dry_run: bool):
    t0 = time.perf_counter()
    n_ticks = DRY_RUN_TICKS if dry_run else TICKS_PER_SEED
    seeds = SEEDS[:2] if dry_run else SEEDS

    all_rows: List[Dict[str, Any]] = []
    arm_results: List[Dict[str, Any]] = []
    regime_analyses: List[Dict[str, Any]] = []

    for regime in REGIMES:
        regime_rows: List[Dict[str, Any]] = []
        regime_telem: List[Dict[str, Any]] = []
        for seed in seeds:
            # arm_cell discharges BOTH per-cell obligations: full RNG reset on enter and
            # fingerprint stamp on .stamp(). include_driver_script_in_hash=False so this
            # cell is MINTED reuse-eligible for a future, different-driver consumer.
            probe = _build_agent_and_env(seed, regime)[3]
            with arm_cell(
                seed,
                config_slice=probe,
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=False,
            ) as cell:
                rows, telem, _slice = _collect_cell(seed, regime, n_ticks)
                row: Dict[str, Any] = {
                    "arm_id": regime["id"],
                    "seed": seed,
                    **telem,
                }
                cell.stamp(row)
            regime_rows.extend(rows)
            regime_telem.append(telem)
            arm_results.append(row)
            print(f"verdict: {'PASS' if telem['n_committed'] > 0 else 'FAIL'}", flush=True)

        all_rows.extend(regime_rows)
        regime_analyses.append(_analyse_regime(regime, regime_rows, regime_telem))

    # ---------------- cross-regime outcome routing ---------------- #
    gate_green = all(a["gate_green"] for a in regime_analyses)
    rises = [a["incumbent_share_rises"] for a in regime_analyses]
    preconditions = [p for a in regime_analyses for p in a["preconditions"]]

    if not gate_green:
        outcome, evidence_direction = "FAIL", "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        failed = [p["name"] for p in preconditions if not p["met"]]
        degeneracy_reason = (
            "MECH-463 non-vacuity gate RED: " + ", ".join(failed)
            + ". A numerical/config degeneracy yields a spurious FLAT profile that is "
              "indistinguishable from a genuine refutation, so this run is NOT scored "
              "and is NOT a refutation of MECH-463."
        )
    elif all(rises):
        outcome, evidence_direction = "PASS", "supports"
        label = "channel_agnostic_incumbent_variance_amplification"
        non_degenerate, degeneracy_reason = True, ""
    elif any(rises):
        outcome, evidence_direction = "FAIL", "mixed"
        label = "incumbent_amplification_channel_specific_not_agnostic"
        non_degenerate, degeneracy_reason = True, ""
    else:
        outcome, evidence_direction = "FAIL", "does_not_support"
        label = "incumbent_share_flat_across_arousal_arousal_causally_inert"
        non_degenerate, degeneracy_reason = True, ""

    criteria = []
    for a in regime_analyses:
        criteria.append({
            "name": f"C_{a['regime']}_incumbent_share_rises_with_urgency",
            # Both regimes are load-bearing: channel-agnosticism is the conjunction.
            "load_bearing": True,
            "passed": a["incumbent_share_rises"],
            "measured_rho": a["spearman_rho"],
            "threshold_rho": MONOTONIC_RHO_FLOOR,
            "measured_gap": a["top_minus_bottom"],
            "threshold_gap": a["gap_floor_applied"],
            "deciles_scored": a["deciles_scored"],
        })
    criteria_non_degenerate = {
        f"C_{a['regime']}_incumbent_share_rises_with_urgency": bool(
            a["deciles_scored"] >= MIN_DECILES_POPULATED and len(a["per_seed_gaps"]) >= 2
        )
        for a in regime_analyses
    }

    elapsed = time.perf_counter() - t0
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    full_config = {
        "seeds": seeds,
        "ticks_per_seed": n_ticks,
        "regimes": REGIMES,
        "urgency_weight": URGENCY_WEIGHT,
        "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
        "alpha_world": ALPHA_WORLD,
        "env": {"cls": "CausalGridWorldV2", "use_proxy_fields": True, "hazard_harm": 0.5},
        "thresholds": {
            "MONOTONIC_RHO_FLOOR": MONOTONIC_RHO_FLOOR,
            "GAP_ABS_FLOOR": GAP_ABS_FLOOR,
            "GAP_SD_MULTIPLIER": GAP_SD_MULTIPLIER,
            "AUTHORITY_ACTIVE_FRAC_FLOOR": AUTHORITY_ACTIVE_FRAC_FLOOR,
            "CHANNEL_RANGE_FLOOR": CHANNEL_RANGE_FLOOR,
            "URGENCY_STD_FLOOR": URGENCY_STD_FLOOR,
            "COMMITTED_TICK_FLOOR": COMMITTED_TICK_FLOOR,
            "INCUMBENT_MARGIN_FLOOR": INCUMBENT_MARGIN_FLOOR,
            "MIN_TICKS_PER_DECILE": MIN_TICKS_PER_DECILE,
            "MIN_DECILES_POPULATED": MIN_DECILES_POPULATED,
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
        "regime_analyses": regime_analyses,
        "metrics": {
            f"{a['regime']}_incumbent": a["expected_incumbent"] for a in regime_analyses
        },
        "criteria": criteria,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "design_note": (
                "Two regimes with DIFFERENT incumbents (harm_weighted vs CH:mech341). "
                "MECH-463's mechanism is channel-agnostic -- the 'which IS F' clause is a "
                "contingent fact from MECH-439, not part of the mechanism -- so the test "
                "is whether the INCUMBENT's share rises with urgency regardless of who "
                "the incumbent is. Rising in both = channel-agnostic; rising in one = "
                "channel-specific (refine, not kill); flat in both = clean refutation."
            ),
            "null_note": (
                "Does NOT import MECH-439's ~88-89% F-share as a null. That figure is a "
                "TEMPORAL variance fraction (V3-EXQ-571 compute_variance_decomp: np.var "
                "over per-STEP candidate-MEAN arrays); this measures CROSS-CANDIDATE "
                "variance within a tick. The two are incommensurable. Nulls are "
                "pre-registered per regime from measurement instead."
            ),
            "licensing_note": (
                "MECH-463 as registered ties its CONSEQUENCE to F-entrenchment. This "
                "establishes the MECHANISM, not that consequence: the substrate offers no "
                "F-incumbent cross-candidate regime (F share is 0.146 at best, and "
                "harm_weighted is the actual cross-candidate incumbent at baseline). A "
                "PASS is evidence about the mechanism the conversion-ceiling argument "
                "invokes, NOT direct evidence about the ceiling."
            ),
            "caveat_z_world_under_differentiation": (
                "Under an under-differentiated z_world (participation ratio ~1.06 at "
                "world_dim=128) the per-candidate channels have little to range over. "
                "That objection bites the FIX, not the measurement -- but a "
                "does_not_support verdict must be read against it, not as unconditional."
            ),
            "scope_note": (
                "NOT gated on MECH-457 or INV-088 (decomposition probe on score geometry, "
                "not a behavioural conversion test). Does NOT re-open the 689/485/445/625 "
                "selection-face lineages; the conversion-ceiling re-derive brake is "
                "respected. MECH-439 deliberately NOT tagged (11 substrate_ceiling "
                "autopsies); a refutation here leaves it untouched."
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
                "reanalysis_query.py query --readout urgency_applied -> no MATCH (all "
                "UNVERIFIABLE, no recoverable substrate_hash); the field did not exist as "
                "a recorded quantity before ree-v3 435322f (2026-07-18) -> must run."
            ),
            "instrumentation_commit": "ree-v3 435322f",
            "per_tick_sink": f"{run_id}_per_tick.jsonl",
            "driver_constraints": (
                "(1) act_with_split_obs() drops obs_harm_a -> urgency pinned at 0; "
                "(2) update_running_variance() has no caller in ree_core (396a) -> commit "
                "gate never fires (rate 0.000 -> 0.964 once the driver drives the EMA)."
            ),
        },
    }
    manifest["_elapsed_seconds_measured"] = elapsed
    return manifest, all_rows


# ------------------------------------------------------------------ #
# Entry point                                                        #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("V3-EXQ-785: MECH-463 two-regime incumbent-share variance decomposition",
          flush=True)
    print(f"  dry_run={args.dry_run}", flush=True)

    t_start = time.perf_counter()
    manifest, rows = run_experiment(args.dry_run)

    out_path = write_flat_manifest(
        manifest,
        OUT_DIR,
        dry_run=args.dry_run,
        config=manifest.get("config"),
        seeds=manifest.get("seeds"),
        script_path=Path(__file__),
        started_at=t_start,
    )

    # Per-tick sink: the novel harness piece (no existing E3 experiment persists per-tick
    # rows -- 689d/571/609 all reduce to arm means first). Written as .jsonl so the
    # indexer's *.json glob cannot mistake it for a second manifest. Skipped on a dry run
    # so a smoke never drops an artifact under evidence/.
    if not args.dry_run:
        sink = Path(out_path).parent / f"{manifest['run_id']}_per_tick.jsonl"
        with open(sink, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"  per-tick sink: {sink} ({len(rows)} rows)", flush=True)

    print(f"  outcome={manifest['outcome']} direction={manifest['evidence_direction']} "
          f"label={manifest['interpretation']['label']}", flush=True)
    for a in manifest["regime_analyses"]:
        print(f"  [{a['regime']}] incumbent={a['observed_incumbent']} "
              f"(expected {a['expected_incumbent']}) rho={a['spearman_rho']:.4f} "
              f"gap={a['top_minus_bottom']:.4f} (floor {a['gap_floor_applied']:.4f}) "
              f"deciles={a['deciles_scored']} committed={a['n_committed']} "
              f"gate={'GREEN' if a['gate_green'] else 'RED'}", flush=True)
        for p in a["preconditions"]:
            if not p["met"]:
                print(f"      RED {p['name']}: measured={p['measured']:.6g} "
                      f"threshold={p['threshold']:.6g}", flush=True)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
