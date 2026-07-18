"""V3-EXQ-785: MECH-463 -- arousal-conditioned E3 selection-score variance decomposition.

!!! DRAFT -- NOT QUEUED. DO NOT QUEUE WITHOUT RESOLVING THE OPEN DESIGN QUESTION BELOW. !!!

Authored 2026-07-18 alongside the MECH-463 instrumentation (ree-v3 435322f). The harness,
the non-vacuity gate and the covariance-correct decomposition are implemented and the
gate measures GREEN. It is NOT queued because pre-queue validation surfaced a
DECOMPOSITION-TARGET mismatch that changes what the probe measures:

  MECH-463's stated null is "MECH-439 alone predicts flat-and-high (~88-89%) F-share".
  That 88-89% figure comes from V3-EXQ-571, whose compute_variance_decomp() is
  explicitly a "temporal variance fraction": np.var over per-STEP arrays whose entries
  are the MEAN across candidates (v3_exq_571_e3_score_variance_decomp.py). It is a
  CROSS-STEP statistic over candidate-averaged components.

  MECH-463's mechanism -- "commitment fires on whichever candidate already has the
  largest score separation" -- is inherently CROSS-CANDIDATE (within a tick).

  These are incommensurable. Measuring a cross-candidate F-share and comparing it to
  88-89% compares different quantities, so "flat-and-high" is not the right null for
  this measurement and a REFUTES verdict would be ill-posed.

Measured cross-candidate F-share in this substrate (250 ticks, seed 0, commit rate 0.964),
F vs the other PRIMARY score components + modulatory channels:

    entropy_bias_scale   F_share    channel_range
    0.0 / 0.01           0.1471     0.0            (P2 RED: non-F channels contribute
                                                    exactly zero cross-candidate variance)
    0.05                 0.0811     0.0163
    0.2                 -0.0245     0.0911
    1.0                 -0.0092     0.2318

  So (a) F is NOT cross-candidate dominant here (0.147, not ~0.88), and (b) there is NO
  window where F dominates AND the non-F channels are non-degenerate -- the transition is
  a cliff, not a gradient. The MECH-341 entropy bonus keys on first-action class and
  produces a large discrete bias, while F's cross-candidate range is tiny (~0.034), so the
  moment the entropy channel is non-degenerate at all it swamps F.

OPEN DESIGN QUESTION (needs a decision before queuing):
  (1) Re-pose the probe on the TEMPORAL statistic, to match MECH-439's 88-89% baseline; or
  (2) keep the cross-candidate statistic (which is what MECH-463's mechanism is actually
      about) and drop the 88-89% null, pre-registering the flat-null from a measured
      entropy-off baseline instead; or
  (3) generalise to the claim's own channel-agnostic wording -- "the DOMINANT channel's
      share rises with urgency", whichever channel dominates -- which is robust to F not
      being the incumbent and is arguably the stronger test of the mechanism; or
  (4) treat the empty F-dominant window as itself the finding and register the probe as
      substrate-blocked.

Everything below this line is implemented and smoke-tested; only the decomposition target
(the definition of F's "rest" term and the null) is unresolved.



CLAIM UNDER TEST (MECH-463): the three affective routes that modulate E3 commitment
GLOBALLY -- D1/D2 dopamine gain, harm-urgency threshold shrinkage, LC-NE / phasic-surprise
temperature -- each apply ONE SCALAR uniformly across all K candidates. A uniform scalar
cannot reorder an argmax, but shrinking the commit threshold makes commitment fire on
whichever candidate already has the largest score separation. That is channel-agnostic
VARIANCE AMPLIFICATION: arousal does not bias TOWARD F, it selects for whichever channel
already dominates -- which, under F-dominance (MECH-439), IS F.

DISCRIMINATING PREDICTION (the whole point -- this is what separates MECH-463 from
MECH-439 alone): F's share of COMMITTED-selection score variance should be a MONOTONE
INCREASING function of the urgency_applied decile, and materially lower in the bottom
decile. MECH-439 as stated predicts flat-and-high (~88-89%) at every arousal level.

  SUPPORTS  -- F-share rises monotonically across urgency deciles AND the bottom-vs-top
               decile gap clears the effect-size floor.
  REFUTES   -- F-share is FLAT with the non-vacuity gate GREEN. This kills MECH-463
               cleanly and leaves MECH-439 untouched: F-dominance would then be
               structural in the score geometry, with arousal causally inert w.r.t. it.
  INVALID   -- non-vacuity gate RED -> substrate_not_ready_requeue, NOT a refutation.

WHY THIS NEEDED NEW INSTRUMENTATION (audit 2026-07-18): urgency_applied escaped only as
SelectionResult.urgency; effective_threshold and commit_variance were pure locals that
died at the end of select(); per-channel bias was retained only as the SUMMED tensor
(agent.py:6816-6845 keeps reduced scalars, losing channel-F covariance). Landed
2026-07-18 in e3_selector.py (ree-v3 435322f), all behind e3_score_decomp_enabled.
GOV-REUSE-1: reanalysis_query.py --readout urgency_applied returns no MATCH (every
candidate UNVERIFIABLE, no recoverable substrate_hash) and the field did not exist as a
recorded quantity before that commit -> not recoverable, must run.

=== MANDATORY NON-VACUITY GATE (inherited from V3-EXQ-643a) ===
V3-EXQ-643 FAILed precondition_unmet purely from float32 catastrophic cancellation (E3
scores ~1e32, modulatory range below float32 ULP) and 643a added this gate. Without it a
NUMERICAL degeneracy returns a spurious FLAT F-share that is INDISTINGUISHABLE from a
genuine refutation. Three assertions, all measured on the run itself:
  P1  modulatory_authority_active_frac > floor
  P2  cross-candidate RANGE of the channel terms > floor  (the SAME statistic the
      variance decomposition routes on -- NOT a magnitude/mean-abs proxy; that
      mismatch is exactly the 643 GAP)
  P3  urgency_applied is NON-CONSTANT across the run (new for MECH-463: if urgency
      never varies there are no deciles and the probe is vacuous)
  P4  total cross-candidate score variance > floor (a share of ~0 variance is undefined)
Any RED -> outcome FAIL, label substrate_not_ready_requeue, non_degenerate=false
(excluded from confidence scoring). NEVER a substrate-verdict label.

CONFIG NOTE (empirically validated GREEN at 400 ticks before authoring):
  - urgency_weight=0.12, NOT 0.5. At 0.5 urgency SATURATES against urgency_max and the
    median == p90 == 0.5, collapsing the top deciles into one value.
  - MECH-341 entropy bonus is the ONLY channel carrying real cross-candidate range
    (~0.465). Curiosity / tonic-vigor biases are UNIFORM across candidates
    (score_bias_abs_mean ~0.025, score_bias_range_mean 0.0), so modulatory authority
    never fires on them and every non-F channel would contribute ZERO variance --
    making F-share trivially 100% at every decile (a false REFUTES).
  - SD-056 e2_action_contrastive + rollout clamp keep E3 scores bounded (~0.034 raw
    range) rather than the ~1e32 that killed 643.

DRIVER CONSTRAINT (do not "simplify" this loop): act_with_split_obs() calls
sense(obs_body, obs_world) with NO obs_harm_a, so z_harm_a is None on that path and
urgency_applied is pinned at 0 -- vacuous. The harm stream MUST be fed explicitly and the
wrapper replicated: clock.advance() -> _e1_tick -> generate_trajectories -> select_action.

INTERPRETATION CAVEAT (carried into the manifest): under an under-differentiated z_world
(participation ratio ~1.06 at world_dim=128) the per-candidate channels have little to
range over. That objection bites the FIX, not the measurement -- but a REFUTES verdict
must be read against it rather than as unconditional.

SCOPE: NOT gated on MECH-457 or INV-088 (this is a decomposition probe on score geometry,
not a behavioural conversion test, so it does not require the agent to be competent), and
does NOT re-open the 689/485/445/625 selection-face lineages -- the conversion-ceiling
re-derive brake stays respected. claim_ids = [MECH-463] ONLY; MECH-439 carries 11
substrate_ceiling autopsies and is deliberately NOT tagged.

Arms: single-arm observational. The conditioning variable (urgency_applied) is ENDOGENOUS,
so no manipulation arm is needed for the first pass. An urgency_weight=0 low-arousal
control arm is a sanctioned confirmatory follow-up if this SUPPORTS.

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
DRY_RUN_TICKS = 40

N_DECILES = 10
MIN_TICKS_PER_DECILE = 15      # a decile with fewer committed ticks is not scored
MIN_DECILES_POPULATED = 6      # need most deciles to speak of a monotone trend

# Non-vacuity gate floors (P1-P4).
AUTHORITY_ACTIVE_FRAC_FLOOR = 0.05
CHANNEL_RANGE_FLOOR = 1e-4     # same statistic + floor family as 643a MODULATORY_RANGE_FLOOR
URGENCY_STD_FLOOR = 1e-3
URGENCY_MIN_DISTINCT = 10
SCORE_VARIANCE_FLOOR = 1e-12

# PASS criteria.
MONOTONIC_RHO_FLOOR = 0.60     # Spearman rho of decile index vs mean F-share
GAP_ABS_FLOOR = 0.02           # absolute floor: 2 percentage points of F-share
GAP_SD_MULTIPLIER = 2.0        # gap must clear GAP_SD_MULTIPLIER * SD-of-delta across seeds

# Validated config values (see CONFIG NOTE in the docstring).
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


def _decompose_tick(f_vec: np.ndarray, r_vec: np.ndarray) -> Optional[Dict[str, float]]:
    """Covariance-correct split of cross-candidate score variance into F vs rest.

    score_k = F_k + R_k, so
        Var(score) = Var(F) + Var(R) + 2 Cov(F, R).
    Attributing HALF the covariance to each side gives two shares that sum to exactly 1:
        share_F = (Var(F) + Cov(F,R)) / Var(score)
        share_R = (Var(R) + Cov(F,R)) / Var(score)
    This is the point of retaining the per-candidate [K] channel tensors unreduced --
    marginal variances alone cannot form Cov(F, R), and dropping the cross-term makes the
    two "shares" not sum to 1 (the failure mode of the reduced-scalar decomp).
    """
    if f_vec.shape != r_vec.shape or f_vec.size < 2:
        return None
    total = f_vec + r_vec
    var_total = float(np.var(total))
    if not np.isfinite(var_total) or var_total <= SCORE_VARIANCE_FLOOR:
        return None
    var_f = float(np.var(f_vec))
    var_r = float(np.var(r_vec))
    cov_fr = float(np.mean((f_vec - f_vec.mean()) * (r_vec - r_vec.mean())))
    share_f = (var_f + cov_fr) / var_total
    return {
        "var_total": var_total,
        "var_f": var_f,
        "var_r": var_r,
        "cov_fr": cov_fr,
        "share_f": float(share_f),
    }


def _build_agent_and_env(seed: int) -> Tuple[REEAgent, Any, Dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    # CausalGridWorld carries its OWN np.random.default_rng(seed) -- omitting seed= here
    # makes the whole run non-reproducible.
    env = CausalGridWorldV2(use_proxy_fields=True, seed=seed, hazard_harm=0.5)
    _obs, obs_dict = env.reset()

    cfg = REEConfig.from_dims(
        body_obs_dim=obs_dict["body_state"].shape[-1],
        world_obs_dim=obs_dict["world_state"].shape[-1],
        action_dim=env.action_dim,
        alpha_world=ALPHA_WORLD,
        # SD-011 affective harm stream -> z_harm_a -> urgency_applied. Without BOTH of
        # these urgency_applied is pinned at 0 and the probe is vacuous (P3 RED).
        use_harm_stream=True,
        use_affective_harm_stream=True,
        urgency_weight=URGENCY_WEIGHT,
        # Candidate diversity.
        use_support_preserving_cem=True,
        support_preserving_min_first_action_classes=2,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        # SD-056: action-divergent candidates + the clamp that keeps E3 scores bounded.
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=0.1,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=4.0,
        # MECH-314 structured curiosity (a modulatory channel; uniform on its own).
        use_structured_curiosity=True,
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=True,
        use_curiosity_learning_progress=True,
        curiosity_bias_scale=0.1,
        curiosity_novelty_weight=0.05,
        curiosity_uncertainty_weight=0.05,
        curiosity_learning_progress_weight=0.05,
        # MECH-341 entropy bonus: the channel that genuinely carries cross-candidate
        # range. Without it every non-F channel contributes zero variance (P2 RED).
        use_e3_score_diversity=True,
        use_e3_diversity_entropy_bonus=True,
        use_e3_diversity_stratified_select=False,
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_min_range_floor=1e-6,
    )
    # Split last_channel_terms into NAMED channels rather than one compressed
    # "score_bias" slot -- the decomposition needs the parts separable.
    cfg.e3.use_finer_channel_gating = True

    agent = REEAgent(cfg)
    agent.eval()
    agent.e3.e3_score_decomp_enabled = True  # MECH-463 instrumentation gate
    return agent, env, obs_dict


def _collect_seed(seed: int, n_ticks: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run one seed, returning per-tick rows plus that seed's gate telemetry."""
    agent, env, obs_dict = _build_agent_and_env(seed)

    rows: List[Dict[str, Any]] = []
    authority_active_hits = 0
    channel_ranges: List[float] = []
    n_select = 0
    # V3-EXQ-396a: the agent loop NEVER calls update_running_variance() -- no caller
    # exists anywhere in ree_core. _running_variance therefore stays pinned at
    # precision_init (0.5) forever, and since effective_threshold tops out around 0.39
    # the commit gate NEVER fires: committed is False on every tick and the probe (which
    # is about COMMITTED-selection variance) has zero rows to score. The driver must
    # drive the EMA itself, per the 396a fix. Measured: commit rate 0.000 -> 0.964.
    z_world_prev: Optional[torch.Tensor] = None
    action_prev: Optional[torch.Tensor] = None

    print(f"Seed {seed} Condition endogenous_arousal", flush=True)

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
            # 396a: drive the running-variance EMA or the commit gate never fires.
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
                authority_active_hits += 1

            per_cand = decomp.get("per_candidate") or []
            f_list = [float(c.get("f", 0.0)) for c in per_cand]
            chan_vecs = {k: v.detach().cpu().numpy().astype(float)
                         for k, v in chan.items()}
            for v in chan_vecs.values():
                if v.size >= 2:
                    channel_ranges.append(float(v.max() - v.min()))

            if len(f_list) >= 2 and chan_vecs:
                f_arr = np.asarray(f_list, dtype=float)
                r_arr = np.zeros_like(f_arr)
                usable = True
                for v in chan_vecs.values():
                    if v.shape != f_arr.shape:
                        usable = False
                        break
                    r_arr = r_arr + v
                if usable:
                    dec = _decompose_tick(f_arr, r_arr)
                    if dec is not None:
                        rows.append({
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
                            "f_per_candidate": [round(x, 6) for x in f_arr.tolist()],
                            "channel_per_candidate": {
                                k: [round(x, 6) for x in v.tolist()]
                                for k, v in chan_vecs.items()
                            },
                            **{k: round(val, 9) for k, val in dec.items()},
                        })

        # Emit on the final tick too, so a short dry-run still exercises the pattern.
        if (tick + 1) % 50 == 0 or tick == n_ticks - 1:
            print(f"  [train] mech463 seed={seed} ep {tick + 1}/{n_ticks} "
                  f"rows={len(rows)}", flush=True)

        act_idx = int(action.argmax().item()) if isinstance(action, torch.Tensor) else int(action)
        action_prev = torch.zeros(1, env.action_dim)
        action_prev[0, act_idx % env.action_dim] = 1.0
        z_world_prev = z_world_cur
        _obs, _r, done, _info, obs_dict = env.step(act_idx % env.action_dim)
        if done:
            _obs, obs_dict = env.reset()

    telemetry = {
        "seed": seed,
        "n_select_calls": n_select,
        "n_rows": len(rows),
        "authority_active_frac": (authority_active_hits / n_select) if n_select else 0.0,
        "channel_range_mean": float(np.mean(channel_ranges)) if channel_ranges else 0.0,
        "channel_range_max": float(np.max(channel_ranges)) if channel_ranges else 0.0,
    }
    return rows, telemetry


def _decile_profile(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Bin COMMITTED ticks by urgency_applied decile and average F-share within each."""
    committed = [r for r in rows if r["committed"]]
    if len(committed) < N_DECILES * MIN_TICKS_PER_DECILE:
        # Still profile what we have; the populated-decile count gates scoring.
        pass
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
            deciles.append({
                "decile": d, "n": len(sel), "scored": False,
                "urgency_lo": float(lo), "urgency_hi": float(hi),
            })
            continue
        shares = [r["share_f"] for r in sel]
        deciles.append({
            "decile": d,
            "n": len(sel),
            "scored": True,
            "urgency_lo": float(lo),
            "urgency_hi": float(hi),
            "urgency_mean": float(np.mean([r["urgency_applied"] for r in sel])),
            "share_f_mean": float(np.mean(shares)),
            "share_f_sd": float(np.std(shares)),
            "var_total_mean": float(np.mean([r["var_total"] for r in sel])),
            "cov_fr_mean": float(np.mean([r["cov_fr"] for r in sel])),
        })
    return {"deciles": deciles, "n_committed": len(committed)}


def run_experiment(dry_run: bool) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    t0 = time.perf_counter()
    n_ticks = DRY_RUN_TICKS if dry_run else TICKS_PER_SEED
    seeds = SEEDS[:2] if dry_run else SEEDS

    all_rows: List[Dict[str, Any]] = []
    per_seed_telemetry: List[Dict[str, Any]] = []
    per_seed_profiles: List[Dict[str, Any]] = []
    per_seed_gaps: List[float] = []

    for seed in seeds:
        rows, telem = _collect_seed(seed, n_ticks)
        all_rows.extend(rows)
        per_seed_telemetry.append(telem)

        prof = _decile_profile(rows)
        scored = [d for d in prof["deciles"] if d.get("scored")]
        seed_gap = None
        if len(scored) >= 2:
            seed_gap = float(scored[-1]["share_f_mean"] - scored[0]["share_f_mean"])
            per_seed_gaps.append(seed_gap)
        prof["seed"] = seed
        prof["top_minus_bottom_share_f"] = seed_gap
        per_seed_profiles.append(prof)

        print(f"verdict: {'PASS' if seed_gap is not None else 'FAIL'}", flush=True)

    # ---------------- non-vacuity gate (P1-P4) ---------------- #
    urg_all = [r["urgency_applied"] for r in all_rows]
    auth_frac = (float(np.mean([t["authority_active_frac"] for t in per_seed_telemetry]))
                 if per_seed_telemetry else 0.0)
    chan_range = (float(np.mean([t["channel_range_mean"] for t in per_seed_telemetry]))
                  if per_seed_telemetry else 0.0)
    urg_std = float(np.std(urg_all)) if urg_all else 0.0
    urg_distinct = len({round(u, 9) for u in urg_all})
    var_total_mean = (float(np.mean([r["var_total"] for r in all_rows]))
                      if all_rows else 0.0)

    preconditions = [
        {
            "name": "modulatory_authority_active_frac",
            "kind": "readiness",
            # All four are FLOORs (met when measured >= threshold). Declared
            # explicitly so the indexer recompute cannot default-misread them.
            "direction": "lower",
            "description": "fraction of selection ticks where the authority gate fired",
            "control": "MECH-341 entropy bonus ON: candidates that genuinely differ",
            "measured": auth_frac,
            "threshold": AUTHORITY_ACTIVE_FRAC_FLOOR,
            "met": auth_frac > AUTHORITY_ACTIVE_FRAC_FLOOR,
        },
        {
            "name": "channel_cross_candidate_range",
            "kind": "readiness",
            # All four are FLOORs (met when measured >= threshold). Declared
            # explicitly so the indexer recompute cannot default-misread them.
            "direction": "lower",
            "description": (
                "mean cross-candidate RANGE of the per-channel bias vectors -- the SAME "
                "statistic the variance decomposition routes on, deliberately NOT a "
                "magnitude/mean-abs proxy (the V3-EXQ-643 GAP)"
            ),
            "control": "MECH-341 entropy bonus ON",
            "measured": chan_range,
            "threshold": CHANNEL_RANGE_FLOOR,
            "met": chan_range > CHANNEL_RANGE_FLOOR,
        },
        {
            "name": "urgency_applied_non_constant",
            "kind": "readiness",
            # All four are FLOORs (met when measured >= threshold). Declared
            # explicitly so the indexer recompute cannot default-misread them.
            "direction": "lower",
            "description": (
                "SD of urgency_applied across the run; if urgency never varies there "
                "are no deciles and the probe is vacuous"
            ),
            "control": "SD-011 affective harm stream fed explicitly via sense()",
            "measured": urg_std,
            "threshold": URGENCY_STD_FLOOR,
            "met": (urg_std > URGENCY_STD_FLOOR) and (urg_distinct >= URGENCY_MIN_DISTINCT),
        },
        {
            "name": "cross_candidate_score_variance",
            "kind": "readiness",
            # All four are FLOORs (met when measured >= threshold). Declared
            # explicitly so the indexer recompute cannot default-misread them.
            "direction": "lower",
            "description": "mean total cross-candidate score variance (a share of ~0 is undefined)",
            "control": "SD-056 action-contrastive candidates + rollout clamp",
            "measured": var_total_mean,
            "threshold": SCORE_VARIANCE_FLOOR,
            "met": var_total_mean > SCORE_VARIANCE_FLOOR,
        },
    ]
    gate_green = all(p["met"] for p in preconditions)

    # ---------------- pooled decile profile + criteria ---------------- #
    pooled = _decile_profile(all_rows)
    scored = [d for d in pooled["deciles"] if d.get("scored")]
    n_scored = len(scored)

    rho = 0.0
    gap = 0.0
    gap_sd = 0.0
    gap_floor = GAP_ABS_FLOOR
    strictly_monotonic = False
    if n_scored >= 2:
        xs = [float(d["decile"]) for d in scored]
        ys = [float(d["share_f_mean"]) for d in scored]
        rho = _spearman_rho(xs, ys)
        gap = float(ys[-1] - ys[0])
        strictly_monotonic = all(ys[i + 1] >= ys[i] for i in range(len(ys) - 1))
        gap_sd = float(np.std(per_seed_gaps)) if len(per_seed_gaps) >= 2 else 0.0
        # Effect-size floor: scale noise on the SD of the DELTA, plus an absolute floor.
        gap_floor = max(GAP_ABS_FLOOR, GAP_SD_MULTIPLIER * gap_sd)

    c1_monotone = bool(n_scored >= MIN_DECILES_POPULATED and rho >= MONOTONIC_RHO_FLOOR)
    c2_effect = bool(n_scored >= 2 and gap > gap_floor)

    criteria = [
        {"name": "C1_f_share_rises_monotonically_with_urgency",
         "load_bearing": True, "passed": c1_monotone,
         "measured_rho": rho, "threshold_rho": MONOTONIC_RHO_FLOOR,
         "deciles_scored": n_scored, "deciles_required": MIN_DECILES_POPULATED},
        {"name": "C2_bottom_vs_top_gap_clears_effect_floor",
         "load_bearing": False, "passed": c2_effect,
         "measured_gap": gap, "threshold_gap": gap_floor,
         "gap_sd_across_seeds": gap_sd},
    ]

    # Degeneracy: a criterion that could not discriminate.
    share_values = [d["share_f_mean"] for d in scored]
    shares_flat_identical = bool(
        len(share_values) >= 2 and (max(share_values) - min(share_values)) < 1e-12
    )
    criteria_non_degenerate = {
        "C1_f_share_rises_monotonically_with_urgency": bool(
            n_scored >= MIN_DECILES_POPULATED and not shares_flat_identical
        ),
        "C2_bottom_vs_top_gap_clears_effect_floor": bool(
            n_scored >= 2 and len(per_seed_gaps) >= 2
        ),
    }

    # ---------------- outcome routing ---------------- #
    if not gate_green:
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = (
            "MECH-463 non-vacuity gate RED: "
            + "; ".join(f"{p['name']}={p['measured']:.6g} (floor {p['threshold']:.6g})"
                        for p in preconditions if not p["met"])
            + ". A numerical/config degeneracy yields a spurious FLAT F-share that is "
              "indistinguishable from a genuine refutation, so this run is NOT scored "
              "and is NOT a refutation of MECH-463."
        )
    elif c1_monotone and c2_effect:
        outcome = "PASS"
        evidence_direction = "supports"
        label = "arousal_amplifies_dominant_channel_variance"
        non_degenerate = True
        degeneracy_reason = ""
    elif n_scored < MIN_DECILES_POPULATED:
        outcome = "FAIL"
        evidence_direction = "inconclusive"
        label = "insufficient_decile_coverage"
        non_degenerate = False
        degeneracy_reason = (
            f"only {n_scored}/{N_DECILES} deciles reached "
            f"{MIN_TICKS_PER_DECILE} committed ticks; monotonicity not assessable"
        )
    else:
        # Gate GREEN, deciles populated, no rise -> the clean refutation.
        outcome = "FAIL"
        evidence_direction = "does_not_support"
        label = "f_share_flat_across_arousal_arousal_causally_inert"
        non_degenerate = True
        degeneracy_reason = ""

    elapsed = time.perf_counter() - t0
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    full_config = {
        "seeds": seeds,
        "ticks_per_seed": n_ticks,
        "urgency_weight": URGENCY_WEIGHT,
        "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
        "alpha_world": ALPHA_WORLD,
        "use_affective_harm_stream": True,
        "use_finer_channel_gating": True,
        "use_e3_diversity_entropy_bonus": True,
        "e2_action_contrastive_enabled": True,
        "e2_rollout_output_norm_clamp_ratio": 4.0,
        "env": {"cls": "CausalGridWorldV2", "use_proxy_fields": True, "hazard_harm": 0.5},
        "thresholds": {
            "MONOTONIC_RHO_FLOOR": MONOTONIC_RHO_FLOOR,
            "GAP_ABS_FLOOR": GAP_ABS_FLOOR,
            "GAP_SD_MULTIPLIER": GAP_SD_MULTIPLIER,
            "AUTHORITY_ACTIVE_FRAC_FLOOR": AUTHORITY_ACTIVE_FRAC_FLOOR,
            "CHANNEL_RANGE_FLOOR": CHANNEL_RANGE_FLOOR,
            "URGENCY_STD_FLOOR": URGENCY_STD_FLOOR,
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
        "metrics": {
            "f_share_spearman_rho": rho,
            "f_share_top_minus_bottom_decile": gap,
            "f_share_gap_floor_applied": gap_floor,
            "f_share_gap_sd_across_seeds": gap_sd,
            "f_share_strictly_monotonic": strictly_monotonic,
            "deciles_scored": n_scored,
            "n_committed_ticks": pooled["n_committed"],
            "n_rows_total": len(all_rows),
            "modulatory_authority_active_frac": auth_frac,
            "channel_cross_candidate_range_mean": chan_range,
            "urgency_applied_sd": urg_std,
            "urgency_applied_distinct": urg_distinct,
            "cross_candidate_score_variance_mean": var_total_mean,
        },
        "decile_profile": pooled,
        "per_seed_profiles": per_seed_profiles,
        "per_seed_telemetry": per_seed_telemetry,
        "criteria": criteria,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "caveat_z_world_under_differentiation": (
                "Under an under-differentiated z_world (participation ratio ~1.06 at "
                "world_dim=128) the per-candidate channels have little to range over. "
                "That objection bites the FIX, not the measurement -- but a "
                "does_not_support (REFUTES) verdict here must be read against it "
                "rather than as unconditional."
            ),
            "scope_note": (
                "NOT gated on MECH-457 or INV-088: this is a decomposition probe on "
                "score geometry, not a behavioural conversion test, so it does not "
                "require the agent to be competent. Does NOT re-open the "
                "689/485/445/625 selection-face lineages; the conversion-ceiling "
                "re-derive brake is respected. MECH-439 is deliberately NOT tagged "
                "(11 substrate_ceiling autopsies); a REFUTES here leaves it untouched."
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
                "candidates UNVERIFIABLE, no recoverable substrate_hash). The field did "
                "not exist as a recorded quantity before ree-v3 435322f (2026-07-18), "
                "so the decisive readout is not recoverable -> run."
            ),
            "instrumentation_commit": "ree-v3 435322f",
            "per_tick_sink": f"{run_id}_per_tick.jsonl",
            "driver_constraint": (
                "act_with_split_obs() calls sense() without obs_harm_a, pinning "
                "urgency_applied at 0; this driver feeds the harm stream explicitly and "
                "replicates the wrapper loop."
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

    print("V3-EXQ-785: MECH-463 arousal-conditioned E3 variance decomposition", flush=True)
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

    # Per-tick sink: the novel harness piece. Written beside the manifest as .jsonl so
    # the indexer's *.json glob does not pick it up as a second manifest. Skipped on a
    # dry run so a smoke never drops an artifact under evidence/.
    if not args.dry_run:
        sink = Path(out_path).parent / f"{manifest['run_id']}_per_tick.jsonl"
        with open(sink, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"  per-tick sink: {sink} ({len(rows)} rows)", flush=True)

    m = manifest["metrics"]
    print(
        f"  outcome={manifest['outcome']} direction={manifest['evidence_direction']} "
        f"label={manifest['interpretation']['label']}",
        flush=True,
    )
    print(
        f"  rho={m['f_share_spearman_rho']:.4f} gap={m['f_share_top_minus_bottom_decile']:.4f} "
        f"(floor {m['f_share_gap_floor_applied']:.4f}) deciles={m['deciles_scored']}",
        flush=True,
    )
    print(
        f"  gate: authority_frac={m['modulatory_authority_active_frac']:.3f} "
        f"chan_range={m['channel_cross_candidate_range_mean']:.6g} "
        f"urgency_sd={m['urgency_applied_sd']:.6g} "
        f"distinct={m['urgency_applied_distinct']}",
        flush=True,
    )
    for p in manifest["interpretation"]["preconditions"]:
        print(f"    P {p['name']}: measured={p['measured']:.6g} "
              f"threshold={p['threshold']:.6g} met={p['met']}", flush=True)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
