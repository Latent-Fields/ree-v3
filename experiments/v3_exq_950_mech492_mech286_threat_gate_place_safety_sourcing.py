"""V3-EXQ-950: MECH-492 falsifier -- does MECH-286's sleep-permission THREAT conjunct
discriminate place safety, and does that depend on which z_harm_a sourcing mode the
driver happens to set?

SLEEP DRIVER: n/a (no sleep loop used -- the gate is evaluated directly on a waking
agent at real world positions; see HARNESS SCOPE below).

CLAIM UNDER TEST: MECH-492 (registered 2026-08-14, candidate, never tested).
MECH-286's sleep-onset gate (ree_core/sleep/sleep_onset_gate.py) ANDs three conditions;
the third is `z_harm_a.norm() < mech286_threat_tonic_threshold` (default 0.4) -- the
identical expression MECH-303's contextual-safety gate reads at a threshold 8x tighter.
MECH-286 declares NO sourcing mode of its own and inherits whichever the driver sets.
V3-EXQ-917 measured that shared expression at chance (AUC <= 0.52) under damage_sourced
and at AUC 0.84-0.97 under the proximity_ema_sourced default, and located the
reachable-AND-discriminating band at 0.55-0.6 -- ABOVE MECH-286's uncalibrated 0.4.
MECH-492 asserts the threat conjunct therefore does not reliably gate on place safety.

NON-DEGENERACY (the claim's own explicit precondition, and the reason V3-EXQ-891 could
NOT be reused as the harness). 891 set the threat signal BY HAND --
`agent._current_latent = types.SimpleNamespace(z_harm_a=v)` with v hand-normed to 0.10
(safe) / 0.90 (threat), line 182 -- never running the encoder, never placing the agent in
a world, never touching ground-truth hazard. Its PASS establishes only that the boolean
AND is wired correctly. THIS driver instead walks a live agent through REAL world
positions, calls the real `agent.sense()` (which sets `agent._current_latent`, agent.py
line 5029, confirmed by direct read), then calls the REAL
`evaluate_sleep_onset_permit(agent)` and reads the gate's OWN emitted diagnostics
(`mech286_threat_ok`, `mech286_z_harm_a_norm`, `mech286_threat_tonic_threshold`). No
substrate change is needed and none is made: the gate already emits all three.

GROUND TRUTH. Place safety is the Chebyshev distance from the agent's actual cell to the
nearest persistent hazard, recomputed EVERY tick from `env.hazards` (so hazard movement is
handled): d <= HAZARD_NEAR_DIST is a HAZARDOUS place, d >= HAZARD_FAR_DIST is a SAFE
place, the band between is excluded as ambiguous. Pre-registered ABSOLUTE cuts, not a
within-run median split -- a within-run split would derive the threshold from the run's own
statistics, which the pre-registration rule forbids. `env.hazard_field` at the agent cell is
ALSO recorded per tick (continuous, includes transient hazards) as a generous-recording
secondary ground truth, but is not the scored label.

DESIGN. 2 sourcing-mode arms x N_SEEDS seeds. Each cell random-walks the agent for
EPISODE_TICKS ticks. Following the V3-EXQ-917/764/520 precedent, the full internal pipeline
is ticked every step (`_act`) so the live path is exercised as in a real driver, while the
ENVIRONMENT is stepped with a RANDOM action -- this decouples "internal computation" from
"which way the walk goes" and avoids a learned-policy confound (a policy that avoided
hazards would starve the HAZARDOUS class).

  damage_sourced        limb_damage_enabled=True   (SD-022 body-damage re-sourcing)
  proximity_ema_sourced limb_damage_enabled=False  (the direct proximity-EMA path; per
                        GFLAG-0026 this is what 18 of the 20 drivers exercising the
                        related MECH-303 gate actually run, i.e. the de facto default)

UNTRAINED ENCODER IS DELIBERATE, and is the 917 precedent. z_harm_a is an unconditional
every-tick forward pass over harm_obs_a (SD-011) -- never gated on training or convergence
-- and 917, whose AUC numbers this claim is built on, measured it exactly this way. Keeping
the apparatus identical is what makes this run's AUC directly comparable to the 0.84-0.97 /
<=0.52 band the claim cites. A trained-encoder replication is the obvious successor and is
named as such in the queue note; it is NOT run here, because a different apparatus would
produce numbers that could not be read against the cited band.

CRITERIA -- taken verbatim from MECH-492's `what_would_answer`, not paraphrased.
Let discriminates(M) := auc_threat_ok(M) >= AUC_BAR, where auc_threat_ok is the mean
per-seed AUC of the gate's own `mech286_threat_ok` as a SAFE-vs-HAZARDOUS place classifier
at the DEFAULT threshold the gate itself reports.

  FALSIFYING (claim REFUTED, evidence_direction "weakens"):
      discriminates(damage_sourced) AND discriminates(proximity_ema_sourced)
      -- the term gates on place safety regardless of driver configuration.
  CONFIRMING (claim SUPPORTED, evidence_direction "supports"), any of:
      (a) fails to discriminate above chance under the mode(s) tested; OR
      (b) is invariant across the ground-truth hazard gradient (threat_ok near-constant
          0 or 1 across all locations, |mean - 0| or |mean - 1| < INVARIANCE_EPS); OR
      (c) discriminates under ONE sourcing mode and not the other -- the claim's exact
          registered form, and the leg this design is built to resolve.

  ALSO FALSIFYING (weaker, partial -> NARROW rather than support): the term is
  near-constant BUT the conjunction's behaviour is unaffected because another conjunct is
  always the binding one. **THIS CRITERION IS SCOPED OUT OF SCORING HERE, on a measured
  reason, and is NOT reported as a narrowing verdict.** Measured in this session's Step
  2.5a probe: with the gate and its named dependent flags (`use_anchor_sets`,
  `use_staleness_accumulator`, `use_broadcast_override`) all enabled, the hippocampal
  staleness accumulator's snapshot stays EMPTY across 400 ticks of the full pipeline, so
  `mech286_staleness_max` is identically 0.0 and `staleness_ok = (0.0 > 0.3)` is
  structurally False on every tick -- making the staleness conjunct trivially "always
  binding" for a reason that is an ARTIFACT OF A WAKE-ONLY HARNESS, not a property of the
  gate. Reporting a narrowing verdict off that would be exactly the structurally-vacuous
  gate the precondition machinery exists to refuse. What WOULD answer it is a driver that
  runs a real sleep loop with episode-end notifications so regions accrue staleness; that
  is named in the queue note as follow-on and is deliberately not built here.

  This scoping does NOT weaken the primary result. `threat_ok` is computed from z_harm_a
  ALONE, upstream of and independently of the AND (sleep_onset_gate.py: `threat_ok =
  harm_a_norm < cfg.threat_tonic_threshold`), so its discrimination against place hazard
  is well-defined whatever the other two conjuncts do.

READINESS (self-routes `substrate_not_ready_requeue`, NEVER a substrate verdict). The
load-bearing criterion routes on an AUC over a binary place label, so the readiness
statistic is the SAME statistic that AUC needs in order to exist: both label classes
present at the visited positions. A worst-cell value is reported (never a mean), so one
starved cell cannot hide behind a healthy average.
  **A near-constant `threat_ok` is NOT a readiness failure -- it is criterion (b), i.e. a
  CONFIRMING measurement.** The readiness gate deliberately asserts nothing about
  threat_ok's variance; asserting it would convert the claim's own predicted signature into
  a substrate-not-ready self-route and make the claim untestable.

claim_ids=[MECH-492] ONLY. MECH-286 is deliberately NOT tagged: this run measures the
information content of a term MECH-286 consumes and says nothing about MECH-286's own
sleep-onset mechanism (MECH-492 lists it as a CONCEPTUAL cross-ref, not a prerequisite).
SD-011 is not tagged either -- 917 already carries that encoder-discrimination question.
experiment_purpose="evidence": MECH-492 is registered `epistemic_category: standard`
EXPLICITLY so it stays in the experiment lane, and its own notes name MECH-475 -- a measured
substrate-defect claim retired by its own pre-registered falsifier via the normal evidence
lane -- as "the lifecycle wanted here".

NOTE ON `outcome`: PASS/FAIL here is APPARATUS decisiveness, not the scientific verdict.
outcome=PASS means the run was ready and reached a determinate verdict; the verdict itself
is carried by `evidence_direction`. A PASS with evidence_direction="supports" therefore
means the defect claim was CONFIRMED.
"""
import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.sleep.sleep_onset_gate import evaluate_sleep_onset_permit  # noqa: E402

from _lib.arm_fingerprint import arm_cell  # noqa: E402
from _lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from _lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from pack_writer import write_flat_manifest  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402

_ZG = ZGoalStreamAccumulator()

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_950_mech492_mech286_threat_gate_place_safety_sourcing"
ARCH_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-492"]

DEVICE = torch.device("cpu")

# --- pre-registered constants -------------------------------------------------
N_SEEDS = 12
SEED_BASE = 4950
EPISODE_TICKS = 400          # progress denominator M (per seed x arm)
GRID_SIZE = 10
NUM_HAZARDS = 4              # chosen in the Step 2.5a probe for the best near/far label
NUM_RESOURCES = 3            # balance (near_frac 0.30-0.42, far_frac 0.29-0.32 at 300 ticks)

# Ground-truth place-safety label: Chebyshev distance to the nearest persistent hazard.
HAZARD_NEAR_DIST = 1         # d <= 1  -> HAZARDOUS place
HAZARD_FAR_DIST = 3          # d >= 3  -> SAFE place; 1 < d < 3 excluded as ambiguous

# SD-022 damage-sourcing vs the direct proximity-EMA path. Dims mirror V3-EXQ-917.
SOURCING_MODES = {
    "damage_sourced": dict(limb_damage_enabled=True, heal_rate=0.4, harm_obs_a_dim=7),
    "proximity_ema_sourced": dict(limb_damage_enabled=False, harm_obs_a_dim=50),
}
ARM_ORDER = ["damage_sourced", "proximity_ema_sourced"]

AUC_BAR = 0.60               # above-chance discrimination with margin
INVARIANCE_EPS = 0.02        # threat_ok "near-constant" band around 0.0 or 1.0
MIN_CLASS_FRAC = 0.10        # each place-label class must cover >= 10% of scored ticks
MIN_APPARATUS_NORM = 1e-6    # z_harm_a must be measurably nonzero
APPARATUS_VALID_FRAC_FLOOR = 0.99

_SUBSTRATE_COMMON = dict(use_harm_stream=True, harm_obs_dim=51, use_affective_harm_stream=True)
# The gate under test plus the dependent flags REEConfig's own MECH-286 comment names
# ("Requires use_staleness_accumulator for the staleness leg; use_broadcast_override for
# hyperarousal lesion tests"). Enabled HERE ONLY -- every one is REEConfig-default False and
# this driver changes no default, so no other experiment is affected.
_GATE_FLAGS = dict(
    use_mech286_sleep_onset_gate=True,
    use_anchor_sets=True,
    use_staleness_accumulator=True,
    use_broadcast_override=True,
)

# --- preconditions ------------------------------------------------------------
# Both apply to both arms: the design is symmetric and neither is scoped out. The
# structural bounds are the pre-registered best case (a label can at most be 50/50
# balanced; an apparatus fraction can at most be 1.0), so neither gate is
# structurally unsatisfiable -- asserted before compute is spent.
PRECONDITIONS = [
    PreconditionSpec(
        name="place_hazard_label_both_classes_present",
        description=(
            "min(SAFE frac, HAZARDOUS frac) among scored ticks, worst cell in the arm. "
            "This is the SAME statistic the load-bearing AUC criterion routes on: an AUC "
            "over a binary label does not exist unless both classes are populated."
        ),
        control=(
            "positive control is the random walk itself, which is not hazard-avoiding "
            "(the env is stepped with a RANDOM action) so both classes are reachable by "
            "construction. Asserts NOTHING about threat_ok's own variance -- a constant "
            "threat_ok is criterion (b), a CONFIRMING result, not a readiness failure."
        ),
        threshold=MIN_CLASS_FRAC,
        direction="lower",
        kind="readiness",
        structural_max=lambda ctx: 0.5,
    ),
    PreconditionSpec(
        name="z_harm_a_apparatus_valid",
        description=(
            "fraction of ticks with a finite z_harm_a norm > MIN_APPARATUS_NORM, worst "
            "cell in the arm. Below floor means z_harm_a is wired off/None -- a genuine "
            "substrate defect, never a discrimination verdict."
        ),
        control=(
            "z_harm_a is an unconditional every-tick forward pass over harm_obs_a "
            "(SD-011), never gated on training/convergence, so a finite nonzero norm is "
            "the only way sense() returns in practice (V3-EXQ-917 precedent)."
        ),
        threshold=APPARATUS_VALID_FRAC_FLOOR,
        direction="lower",
        kind="readiness",
        structural_max=lambda ctx: 1.0,
    ),
]


def _env_kwargs(mode):
    kw = dict(SOURCING_MODES[mode])
    kw.pop("harm_obs_a_dim", None)
    out = dict(size=GRID_SIZE, num_hazards=NUM_HAZARDS, num_resources=NUM_RESOURCES,
               use_proxy_fields=True)
    out.update(kw)
    return out


def _cfg_kwargs(env, mode):
    kw = dict(body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim, action_dim=5)
    kw.update(_SUBSTRATE_COMMON)
    kw.update(_GATE_FLAGS)
    kw["harm_obs_a_dim"] = SOURCING_MODES[mode]["harm_obs_a_dim"]
    return kw


def _config_slice(mode, episode_ticks):
    return {
        "mode": mode,
        "env_kwargs": _env_kwargs(mode),
        "substrate_common": _SUBSTRATE_COMMON,
        "gate_flags": _GATE_FLAGS,
        "harm_obs_a_dim": SOURCING_MODES[mode]["harm_obs_a_dim"],
        "episode_ticks": episode_ticks,
        "hazard_near_dist": HAZARD_NEAR_DIST,
        "hazard_far_dist": HAZARD_FAR_DIST,
        # Declared explicitly even where already implied by env_kwargs: this cell emits a
        # CROSS-DRIVER-reusable fingerprint (include_driver_script_in_hash=False), so any
        # readout-affecting constant left out of the slice is a false-cache-HIT bug
        # (arm_reuse_fingerprint_plan.md 7b), which corrupts a conclusion rather than
        # merely wasting compute.
        "grid_size": GRID_SIZE,
        "num_hazards": NUM_HAZARDS,
        "num_resources": NUM_RESOURCES,
        "seed_base": SEED_BASE,
        "min_apparatus_norm": MIN_APPARATUS_NORM,
    }


def _sense(agent, obs):
    return agent.sense(
        obs["body_state"], obs["world_state"],
        obs_harm=obs.get("harm_obs"), obs_harm_a=obs.get("harm_obs_a"),
    )


def _act(agent, latent):
    """Ticks the full internal pipeline (V3-EXQ-917/764/520 precedent) so the live path is
    exercised as in a real driver. The returned action is DISCARDED -- the environment is
    stepped randomly, so this only advances agent-internal state."""
    ticks = agent.clock.advance()
    world_dim = agent.config.latent.world_dim
    e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, world_dim, device=DEVICE))
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    action = agent.select_action(candidates, ticks)
    return int(action.argmax(dim=-1).item())


def _nearest_hazard_dist(env):
    """Chebyshev distance from the agent cell to the nearest PERSISTENT hazard, recomputed
    every tick so hazard movement is handled. Returns None when the world has no hazard."""
    hz = getattr(env, "hazards", None) or []
    if not hz:
        return None
    ax, ay = env.agent_x, env.agent_y
    return min(max(abs(ax - int(h[0])), abs(ay - int(h[1]))) for h in hz)


def _auc(pos_scores, neg_scores):
    """Rank-biserial AUC (Mann-Whitney U / (n_pos*n_neg)) = P(pos > neg), ties 0.5.
    Vectorised: the tick counts here make the naive O(n*m) double loop too slow."""
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    # average ranks within ties so a constant score yields exactly 0.5
    allv = np.concatenate([pos, neg])
    srt = np.sort(allv, kind="mergesort")
    i = 0
    while i < srt.size:
        j = i
        while j + 1 < srt.size and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            mask = allv == srt[i]
            ranks[mask] = float(np.mean(np.arange(i + 1, j + 2, dtype=np.float64)))
        i = j + 1
    r_pos = float(np.sum(ranks[:n_pos]))
    u = r_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _run_cell(mode, seed, episode_ticks):
    env = CausalGridWorldV2(seed=SEED_BASE + seed, **_env_kwargs(mode))
    cfg = REEConfig.from_dims(**_cfg_kwargs(env, mode))
    agent = REEAgent(cfg)
    agent.reset()
    _, obs = env.reset()
    rng = random.Random(SEED_BASE + seed)

    ticks = []
    n_valid = 0
    reported_threshold = None
    # Runner parses `ep N/M` for its progress bar; M MUST be the loop bound. Interval is
    # derived from the bound so the line still fires on a short --dry-run.
    print_every = max(1, min(100, episode_ticks // 4))
    for step in range(episode_ticks):
        latent = _sense(agent, obs)
        _ = _act(agent, latent)
        # The REAL gate, on the REAL current latent that sense() just installed.
        permitted, diag = evaluate_sleep_onset_permit(agent)
        if reported_threshold is None:
            reported_threshold = float(diag["mech286_threat_tonic_threshold"])
        norm = float(diag["mech286_z_harm_a_norm"])
        if np.isfinite(norm) and norm > MIN_APPARATUS_NORM:
            n_valid += 1
        d = _nearest_hazard_dist(env)
        ticks.append({
            "step": step,
            "agent_x": int(env.agent_x),
            "agent_y": int(env.agent_y),
            "hazard_dist": (None if d is None else int(d)),
            "hazard_field_at_agent": float(env.hazard_field[env.agent_x, env.agent_y]),
            "z_harm_a_norm": norm,
            "threat_ok": float(diag["mech286_threat_ok"]),
            "override_signal": float(diag["mech286_override_signal"]),
            "staleness_max": float(diag["mech286_staleness_max"]),
            "override_ok": float(diag["mech286_override_ok"]),
            "staleness_ok": float(diag["mech286_staleness_ok"]),
            "sleep_permitted": float(diag["mech286_sleep_permitted"]),
        })
        _flat, _harm, done, _info, obs = env.step(rng.randint(0, 4))
        if done:
            _, obs = env.reset()
        if (step + 1) % print_every == 0:
            print(f"  [train] arm={mode} seed={seed} ep {step + 1}/{episode_ticks}",
                  flush=True)

    _ZG.observe(agent)   # genuine goal-OFF measured zero (never leave it "unmeasured")

    safe = [t for t in ticks if t["hazard_dist"] is not None and t["hazard_dist"] >= HAZARD_FAR_DIST]
    hazd = [t for t in ticks if t["hazard_dist"] is not None and t["hazard_dist"] <= HAZARD_NEAR_DIST]
    n_scored = len(safe) + len(hazd)
    safe_frac = (len(safe) / n_scored) if n_scored else 0.0
    hazd_frac = (len(hazd) / n_scored) if n_scored else 0.0

    # threat_ok as a SAFE-predicting score: higher (=1, "permissive") should mean SAFER
    # place if the term gates on place safety.
    auc_threat_ok = _auc([t["threat_ok"] for t in safe], [t["threat_ok"] for t in hazd])
    # -norm as a SAFE-predicting score: says whether ANY threshold could discriminate,
    # separating "signal carries no place information" from "threshold is miscalibrated".
    auc_norm = _auc([-t["z_harm_a_norm"] for t in safe], [-t["z_harm_a_norm"] for t in hazd])

    tok_all = np.array([t["threat_ok"] for t in ticks], dtype=np.float64)
    norms_all = np.array([t["z_harm_a_norm"] for t in ticks], dtype=np.float64)
    stale_all = np.array([t["staleness_max"] for t in ticks], dtype=np.float64)
    perm_all = np.array([t["sleep_permitted"] for t in ticks], dtype=np.float64)

    return {
        "mode": mode,
        "seed": seed,
        "n_ticks": len(ticks),
        "n_valid_apparatus": n_valid,
        "apparatus_valid_frac": (n_valid / len(ticks)) if ticks else 0.0,
        "reported_threat_threshold": reported_threshold,
        "n_scored": n_scored,
        "n_safe": len(safe),
        "n_hazardous": len(hazd),
        "n_ambiguous_excluded": len(ticks) - n_scored,
        "safe_frac": safe_frac,
        "hazardous_frac": hazd_frac,
        "label_min_class_frac": min(safe_frac, hazd_frac),
        "auc_threat_ok": auc_threat_ok,
        "auc_neg_norm": auc_norm,
        "threat_ok_mean": float(tok_all.mean()) if tok_all.size else 0.0,
        "threat_ok_std": float(tok_all.std()) if tok_all.size else 0.0,
        "threat_ok_mean_safe": float(np.mean([t["threat_ok"] for t in safe])) if safe else None,
        "threat_ok_mean_hazardous": float(np.mean([t["threat_ok"] for t in hazd])) if hazd else None,
        "norm_mean": float(norms_all.mean()) if norms_all.size else 0.0,
        "norm_std": float(norms_all.std()) if norms_all.size else 0.0,
        "norm_min": float(norms_all.min()) if norms_all.size else 0.0,
        "norm_max": float(norms_all.max()) if norms_all.size else 0.0,
        "norm_range": float(norms_all.max() - norms_all.min()) if norms_all.size else 0.0,
        "norm_mean_safe": float(np.mean([t["z_harm_a_norm"] for t in safe])) if safe else None,
        "norm_mean_hazardous": float(np.mean([t["z_harm_a_norm"] for t in hazd])) if hazd else None,
        # conjunction context -- recorded, NOT scored (see SCOPED-OUT criterion in docstring)
        "staleness_max_max": float(stale_all.max()) if stale_all.size else 0.0,
        "staleness_ok_frac": float(np.mean([t["staleness_ok"] for t in ticks])) if ticks else 0.0,
        "override_ok_frac": float(np.mean([t["override_ok"] for t in ticks])) if ticks else 0.0,
        "sleep_permitted_frac": float(perm_all.mean()) if perm_all.size else 0.0,
        "per_tick": ticks,   # generous recording: full per-tick raw record
    }


def _run_arm(mode, seeds, episode_ticks):
    rows = []
    for seed in seeds:
        print(f"Seed {seed} Condition {mode}", flush=True)
        with arm_cell(
            seed,
            config_slice=_config_slice(mode, episode_ticks),
            script_path=Path(__file__),
            config_slice_declared=True,
            # Mint-as-you-go: cross-driver reusable, so a successor (e.g. the
            # trained-encoder or sleep-loop replication named in the queue note) can
            # reuse these cells rather than re-walking them.
            include_driver_script_in_hash=False,
        ) as cell:
            row = _run_cell(mode, seed, episode_ticks)
            cell.stamp(row)
        print(f"  [{mode} seed={seed}] auc_threat_ok={row['auc_threat_ok']:.4f} "
              f"auc_neg_norm={row['auc_neg_norm']:.4f} "
              f"threat_ok_mean={row['threat_ok_mean']:.3f} "
              f"min_class_frac={row['label_min_class_frac']:.3f}", flush=True)
        print("verdict: PASS", flush=True)   # per-cell run completion marker
        rows.append(row)
    return rows


def _worst_cell(rows, key):
    """Worst (minimum) value plus the cell that carries it. Never a mean -- the gate's
    `met` is an all(...) over cells, so `measured` must be the extremum or the indexer's
    recompute would pass a starved cell hidden behind a healthy average."""
    vals = [(r[key], f"{r['mode']}_seed{r['seed']}") for r in rows if r.get(key) is not None]
    if not vals:
        return 0.0, None
    v, cell = min(vals, key=lambda t: t[0])
    return float(v), cell


def _mode_analysis(rows):
    aucs = [r["auc_threat_ok"] for r in rows]
    aucs_norm = [r["auc_neg_norm"] for r in rows]
    tok_means = [r["threat_ok_mean"] for r in rows]
    mean_auc = float(np.mean(aucs)) if aucs else 0.5
    mean_tok = float(np.mean(tok_means)) if tok_means else 0.0
    invariant = bool(abs(mean_tok - 0.0) < INVARIANCE_EPS or abs(mean_tok - 1.0) < INVARIANCE_EPS)
    # AUC is DIRECTIONAL: threat_ok=1 ("permissive") is scored as a SAFE prediction, so
    # auc < 0.5 means the term is MORE permissive at HAZARDOUS places than at safe ones --
    # anti-discrimination, which is a distinct and worse finding than a null and must not
    # be collapsed into one. It still counts as "does not discriminate" for the claim's
    # criterion (a); this flag only makes it visible to a reader.
    anti = bool(mean_auc <= (1.0 - AUC_BAR))
    return {
        "anti_discriminates": anti,
        "mean_auc_threat_ok": mean_auc,
        "std_auc_threat_ok": float(np.std(aucs)) if aucs else 0.0,
        "per_seed_auc_threat_ok": aucs,
        "mean_auc_neg_norm": float(np.mean(aucs_norm)) if aucs_norm else 0.5,
        "per_seed_auc_neg_norm": aucs_norm,
        "mean_threat_ok": mean_tok,
        "per_seed_threat_ok_mean": tok_means,
        "threat_ok_invariant": invariant,
        "discriminates": bool(mean_auc >= AUC_BAR),
        "mean_norm": float(np.mean([r["norm_mean"] for r in rows])) if rows else 0.0,
        "mean_norm_range": float(np.mean([r["norm_range"] for r in rows])) if rows else 0.0,
        "mean_sleep_permitted_frac": float(np.mean([r["sleep_permitted_frac"] for r in rows])) if rows else 0.0,
        "mean_staleness_ok_frac": float(np.mean([r["staleness_ok_frac"] for r in rows])) if rows else 0.0,
        "mean_override_ok_frac": float(np.mean([r["override_ok_frac"] for r in rows])) if rows else 0.0,
        "max_staleness_max": float(np.max([r["staleness_max_max"] for r in rows])) if rows else 0.0,
    }


def run_experiment(n_seeds, episode_ticks, dry_run=False):
    t0 = time.perf_counter()
    seeds = list(range(n_seeds))

    # Design-time proof that no arm's gate is structurally unsatisfiable, BEFORE compute.
    arm_contexts = {m: {"id": m, "arm_id": m, "mode": m} for m in ARM_ORDER}
    assert_no_structurally_unsatisfiable_gate(
        PRECONDITIONS, [arm_contexts[m] for m in ARM_ORDER])

    rows_by_mode = {}
    for mode in ARM_ORDER:
        rows_by_mode[mode] = _run_arm(mode, seeds, episode_ticks)

    analyses = {m: _mode_analysis(rows_by_mode[m]) for m in ARM_ORDER}

    # --- per-arm readiness gates (a red arm must never vacate a green one) ---
    arm_gates = []
    for mode in ARM_ORDER:
        rows = rows_by_mode[mode]
        lbl, lbl_cell = _worst_cell(rows, "label_min_class_frac")
        app, app_cell = _worst_cell(rows, "apparatus_valid_frac")
        gate = evaluate_arm_gate(
            mode, arm_contexts[mode], PRECONDITIONS,
            measured={
                "place_hazard_label_both_classes_present": lbl,
                "z_harm_a_apparatus_valid": app,
            },
        )
        gate["offending_cells"] = {
            "place_hazard_label_both_classes_present": lbl_cell,
            "z_harm_a_apparatus_valid": app_cell,
        }
        arm_gates.append(gate)
    aggregate = aggregate_arm_gates(arm_gates)

    green = set(aggregate.get("green_arms") or [])
    d_dam = analyses["damage_sourced"]["discriminates"]
    d_pro = analyses["proximity_ema_sourced"]["discriminates"]
    inv_dam = analyses["damage_sourced"]["threat_ok_invariant"]
    inv_pro = analyses["proximity_ema_sourced"]["threat_ok_invariant"]

    both_green = {"damage_sourced", "proximity_ema_sourced"} <= green
    ready = bool(aggregate.get("non_degenerate")) and both_green

    if not ready:
        label = "substrate_not_ready_requeue"
        verdict = "apparatus not ready"
        direction = "non_contributory"
        outcome = "FAIL"
    elif d_dam and d_pro:
        label = "mech492_refuted_discriminates_under_both_sourcing_modes"
        verdict = ("FALSIFYING: threat_ok discriminates place safety above chance under "
                   "BOTH sourcing modes at the gate's default threshold")
        direction = "weakens"
        outcome = "PASS"
    elif d_dam != d_pro:
        label = "mech492_confirmed_registered_form_sourcing_dependent_discrimination"
        verdict = ("CONFIRMING criterion (c): threat_ok discriminates under ONE sourcing "
                   "mode and not the other -- the claim's exact registered form")
        direction = "supports"
        outcome = "PASS"
    else:
        which = []
        if inv_dam or inv_pro:
            which.append("(b) near-constant across the ground-truth hazard gradient")
        which.append("(a) fails to discriminate above chance")
        anti_modes = [m for m in ARM_ORDER if analyses[m]["anti_discriminates"]]
        if anti_modes:
            which.append("with ANTI-discrimination (more permissive at HAZARDOUS than at "
                         "safe places) in: " + ", ".join(anti_modes))
        label = "mech492_confirmed_no_place_safety_discrimination_under_either_mode"
        verdict = "CONFIRMING criteria " + " and ".join(which) + " under BOTH sourcing modes"
        direction = "supports"
        outcome = "PASS"

    criteria = [
        {"name": "C1_discriminates_damage_sourced", "load_bearing": True, "passed": bool(d_dam),
         "measured_auc": analyses["damage_sourced"]["mean_auc_threat_ok"], "bar": AUC_BAR},
        {"name": "C2_discriminates_proximity_ema_sourced", "load_bearing": True, "passed": bool(d_pro),
         "measured_auc": analyses["proximity_ema_sourced"]["mean_auc_threat_ok"], "bar": AUC_BAR},
        {"name": "C3_threat_ok_invariant_damage_sourced", "load_bearing": False, "passed": bool(inv_dam)},
        {"name": "C4_threat_ok_invariant_proximity_ema_sourced", "load_bearing": False, "passed": bool(inv_pro)},
    ]
    combination_rule = (
        "NOT a plain AND. C1 and C2 are read as a PAIR and the verdict is a 3-way routing "
        "on their pattern, taken verbatim from MECH-492's what_would_answer: (C1 and C2) -> "
        "REFUTED/weakens; (C1 xor C2) -> CONFIRMED criterion (c)/supports, the claim's exact "
        "registered form; (neither) -> CONFIRMED criteria (a)/(b)/supports. A `passed: false` "
        "on C1 or C2 is therefore NOT a failed run -- for this claim it is a positive finding."
    )

    nd = arm_criteria_non_degenerate(
        {"damage_sourced": ["C1_discriminates_damage_sourced",
                            "C3_threat_ok_invariant_damage_sourced"],
         "proximity_ema_sourced": ["C2_discriminates_proximity_ema_sourced",
                                   "C4_threat_ok_invariant_proximity_ema_sourced"]},
        aggregate,
    )

    arm_results = []
    for mode in ARM_ORDER:
        for r in rows_by_mode[mode]:
            arm_results.append(r)

    all_rows = [r for m in ARM_ORDER for r in rows_by_mode[m]]
    label_balance = {
        "place_hazard_safe_vs_hazardous": {
            f"{m}_eval_safe_frac": float(np.mean([r["safe_frac"] for r in rows_by_mode[m]]))
            for m in ARM_ORDER
        }
    }
    label_balance["place_hazard_safe_vs_hazardous"].update({
        f"{m}_eval_hazardous_frac": float(np.mean([r["hazardous_frac"] for r in rows_by_mode[m]]))
        for m in ARM_ORDER
    })

    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCH_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": list(CLAIM_IDS),
        "outcome": outcome,
        "evidence_direction": direction,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "sleep_driver_pattern": "n/a (no sleep loop used)",
        "verdict": verdict,
        "combination_rule": combination_rule,
        "criteria": criteria,
        "pre_registered": {
            "auc_bar": AUC_BAR,
            "invariance_eps": INVARIANCE_EPS,
            "min_class_frac": MIN_CLASS_FRAC,
            "hazard_near_dist": HAZARD_NEAR_DIST,
            "hazard_far_dist": HAZARD_FAR_DIST,
            "mech286_threat_tonic_threshold_default": 0.4,
            "apparatus_valid_frac_floor": APPARATUS_VALID_FRAC_FLOOR,
        },
        "mode_analyses": analyses,
        "arm_results": arm_results,
        "per_arm_gate": aggregate.get("per_arm_gate"),
        "non_degenerate": bool(aggregate.get("non_degenerate")),
        "degeneracy_reason": aggregate.get("degeneracy_reason") or "",
        "label_balance": label_balance,
        "interpretation": {
            "label": label,
            "preconditions": aggregate.get("adjudication_preconditions"),
            "preconditions_scope_note": aggregate.get("per_arm_gate", {}).get(
                "preconditions_scope_note", ""),
            "criteria_non_degenerate": nd,
            "scoped_out_criteria": [{
                "name": "ALSO_FALSIFYING_another_conjunct_always_binding",
                "source": "MECH-492 what_would_answer, third (narrowing) criterion",
                "disposition": "scoped out of scoring -- structurally unsatisfiable in this harness",
                "measured_reason": (
                    "max staleness_max across every cell = "
                    f"{max(a['max_staleness_max'] for a in analyses.values()):.6f}; the "
                    "hippocampal staleness accumulator snapshot stays EMPTY in a wake-only "
                    "walk harness even with use_staleness_accumulator/use_anchor_sets on, so "
                    "staleness_ok = (0.0 > 0.3) is structurally False every tick and the "
                    "staleness conjunct is trivially always-binding for a HARNESS reason, "
                    "not a gate property. Reporting a narrowing verdict from that would be a "
                    "vacuous gate."
                ),
                "what_would_answer_it": (
                    "a driver running a real sleep loop with episode-end notifications so "
                    "regions accrue MECH-284 staleness -- named as follow-on, not built here"
                ),
            }],
        },
        "custom_information": {
            "gate_conjunction_context_not_scored": {
                "note": (
                    "Recorded for completeness. sleep_permitted is identically 0 in this "
                    "harness because the staleness leg never engages (see scoped_out_criteria); "
                    "it is NOT a DV here and carries no weight for MECH-492. threat_ok is "
                    "computed upstream of and independently of the AND, so the primary "
                    "measurement is unaffected."
                ),
                "per_mode": {m: {
                    "mean_sleep_permitted_frac": analyses[m]["mean_sleep_permitted_frac"],
                    "mean_staleness_ok_frac": analyses[m]["mean_staleness_ok_frac"],
                    "mean_override_ok_frac": analyses[m]["mean_override_ok_frac"],
                } for m in ARM_ORDER},
            },
            "comparability_to_v3_exq_917": (
                "Same untrained live-agent apparatus and same two sourcing modes as "
                "V3-EXQ-917, so mean_auc_neg_norm is readable against 917's 0.84-0.97 "
                "(proximity_ema_sourced) / <=0.52 (damage_sourced) band. The NEW quantity "
                "here is auc_threat_ok -- the gate's own BOOLEAN at its own uncalibrated "
                "0.4 threshold, against a PER-POSITION place-safety label rather than 917's "
                "arm-level hazard-density grouping."
            ),
            "reported_threat_threshold_seen": sorted({
                r["reported_threat_threshold"] for r in all_rows
                if r.get("reported_threat_threshold") is not None
            }),
        },
        "n_seeds": n_seeds,
        "episode_ticks": episode_ticks,
        "dry_run": bool(dry_run),
    }

    full_config = {
        "n_seeds": n_seeds,
        "episode_ticks": episode_ticks,
        "grid_size": GRID_SIZE,
        "num_hazards": NUM_HAZARDS,
        "num_resources": NUM_RESOURCES,
        "seed_base": SEED_BASE,
        "sourcing_modes": SOURCING_MODES,
        "substrate_common": _SUBSTRATE_COMMON,
        "gate_flags": _GATE_FLAGS,
        "auc_bar": AUC_BAR,
        "invariance_eps": INVARIANCE_EPS,
        "min_class_frac": MIN_CLASS_FRAC,
        "hazard_near_dist": HAZARD_NEAR_DIST,
        "hazard_far_dist": HAZARD_FAR_DIST,
    }

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config=full_config,
        seeds=[SEED_BASE + s for s in seeds],
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    return manifest, out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--ticks", type=int, default=EPISODE_TICKS)
    args = ap.parse_args()

    n_seeds = 2 if args.dry_run else args.seeds
    ticks = 60 if args.dry_run else args.ticks

    result, out_path = run_experiment(n_seeds, ticks, dry_run=args.dry_run)

    print("")
    print(f"outcome: {result['outcome']}")
    print(f"evidence_direction: {result['evidence_direction']}")
    print(f"interpretation: {result['interpretation']['label']}")
    print(f"verdict: {result['verdict']}")
    for m in ARM_ORDER:
        a = result["mode_analyses"][m]
        print(f"  {m}: auc_threat_ok={a['mean_auc_threat_ok']:.4f} "
              f"auc_neg_norm={a['mean_auc_neg_norm']:.4f} "
              f"threat_ok_mean={a['mean_threat_ok']:.3f} "
              f"invariant={a['threat_ok_invariant']} discriminates={a['discriminates']}")
    print(f"manifest: {out_path}")

    return result, out_path, args


if __name__ == "__main__":
    _result, _out_path, _args = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=_args.dry_run,
    )
