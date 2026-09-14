"""BUILT AND DELIBERATELY NOT QUEUED (2026-09-11). Renamed off the v3_exq_*.py pattern so it
cannot trip another session's queue atomicity audit; V3-EXQ-1026 was reserved and RELEASED.

This is a complete, validated 3-arm extension of V3-EXQ-598b asking MECH-309's question: does a
rule-apprehension layer break the monomodal policy collapse that gradient training falls into?
It was not queued because its own `dv_responds_to_arm` precondition would refuse a verdict:
the DV (598b's p2_reef_visit_fraction) is INERT with respect to the manipulation. Measured at
two independent budgets, all three arms returned a bit-identical fraction (0.483 at 3/4/2/30;
0.1034 at 10/20/8/60; arm spread 0.0000) while the arms differed demonstrably inside -- the
gradient arm's mean |bias| reached 0.100000 and the CRF arm minted a live 2-rule pool. That is
the MECH-439 F-dominance conversion ceiling, recorded as GFLAG-0271.

It becomes queueable IF that ceiling is lifted, or on a training regime where the rule signal
reaches committed action. Until then, running it buys a substrate_not_ready_requeue.

Context: GFLAG-0265 / GFLAG-0266 / GFLAG-0268 / GFLAG-0271; chip
chip-20260911-mech349-validation-redesign.
"""

"""V3-EXQ-1026: MECH-309 -- does a rule-apprehension layer BREAK the monomodal policy
collapse that gradient training falls into?

red-team: PENDING (filled in before queueing).

=== THE QUESTION ===
MECH-309 asserts that monomodal policy collapse is the EQUILIBRIUM of a parametric-policy
agent WITHOUT a rule-apprehension layer: "Bayesian update and gradient descent revise weights
over a hypothesis space they do not invent; without a non-Bayesian rule-creator that proposes
discriminative policy modes ... the trainer collapses to the smoothest single regime". Its own
notes route the fix to ARC-062 / ARC-063.

V3-EXQ-598b (2026-05-27) measured the first half and nothing has ever measured the second.
With the SD-033a rule-bias head TRAINABLE under outcome-coupled REINFORCE, 598b recorded
`C2_trainable_nonzero=True` (the head moves) but `C3_trainable_not_monomodal=False` -- the
policy collapsed anyway. This run reproduces that arm and adds the one arm 598b did not have:
the ARC-063 CandidateRuleField, a NON-GRADIENT rule creator, with the bias head left frozen.

  ARM_0_FROZEN    train_rule_bias_head=False, CRF off -- 598b's silent baseline
  ARM_1_GRADIENT  train_rule_bias_head=True,  CRF off -- 598b's ARM_1 (collapse under gradient)
  ARM_2_MINTED    train_rule_bias_head=False, CRF ON  -- the non-gradient creator

DV: 598b's own `p2_reef_visit_fraction`, unchanged. "Not monomodal" = the fraction lies inside
(REEF_LO, REEF_HI) -- the agent divides its time between the reef and forage regimes rather
than collapsing onto one.

=== WHY THIS RUN EXISTS, AND WHY IT IS NOT THE FOUR THAT PRECEDED IT ===
Four prior designs (all BLOCKING at adversarial review, see GFLAG-0265/0266/0268) tried to
evidence MECH-349 by measuring the MINT EVENT itself. That is not testable: MECH-349's trigger
clauses restate `CandidateRuleField._maybe_mint` (a sub-threshold mint is unreachable by
construction), its distinctness diagnostic is a deterministic lookup on live-rule count off the
fixed pinned-direction matrix, and its retirement diagnostic is structurally zero under the
maintenance stack. Measured 2026-09-11: on the REPRESENTATIONAL DV those attempts fell back to
(`rule_state` RSA margin), the legacy non-CRF EMA already scores 0.586 against a 0.15 floor --
there is nothing for minting to rescue -- and the CRF arm's higher margin (0.982) is dominated
by the geometry of 2 fixed pinned directions, minting only 2 rules whether the ecology offers
2 contexts or 4.

So this run changes the TARGET, not the design. It asks MECH-309's question, on MECH-309's own
DV, where an answer can come back either way.

=== PRE-REGISTERED CRITERIA ===
C1 (LOAD-BEARING, MECH-309) -- COLLAPSE UNDER GRADIENT IS REAL AND STILL REPRODUCES.
    ARM_1_GRADIENT's bias head moves (mean |bias| >= TRAINABLE_BIAS_MIN) AND its
    p2_reef_visit_fraction falls OUTSIDE (REEF_LO, REEF_HI), on >= MIN_PASS_SEEDS seeds.
    This is a reproduction of V3-EXQ-598b's C2+C3 and it is two-sided: if the gradient arm no
    longer collapses, MECH-309's asserted equilibrium does not hold on this substrate and the
    claim is WEAKENED. It is deliberately a CRITERION and not a precondition, so that outcome
    is reported rather than suppressed as "instrument not ready".
C2 (LOAD-BEARING, MECH-309 routing + MECH-349 answer-hood) -- THE RULE LAYER BREAKS IT.
    ARM_2_MINTED's p2_reef_visit_fraction falls INSIDE (REEF_LO, REEF_HI) on
    >= MIN_PASS_SEEDS seeds, i.e. the non-gradient creator produces the bimodal regime split
    that gradient training could not.

PASS = preconditions met AND C1 AND C2.

=== PER-CLAIM DIRECTION, AND THE ASYMMETRY MECH-349's OWN TEXT REQUIRES ===
MECH-309: supports iff C1; weakens iff not C1. Symmetric -- this is its own DV.
MECH-349: supports iff (C1 AND C2). If C1 holds and C2 FAILS, MECH-349 is recorded
    NON_CONTRIBUTORY, never `weakens`. That is not a hedge: MECH-349's `what_would_answer`
    states "NOT FALSIFYING. A null on any downstream committed-action or behavioural DV ...
    Routing such a null to MECH-349 would repeat the ARC-062 GAP-B C1-holds/C2-fails
    misattribution 21 autopsies have already recorded." A behavioural null here is exactly that
    shape, so the claim's own text forbids scoring it against itself. A behavioural POSITIVE is
    not foreclosed by that sentence and is recorded as support.

=== NON-DEGENERACY PRECONDITIONS ===
(a) BASELINE INTACT -- ARM_0_FROZEN's mean |bias| < FROZEN_BIAS_MAX (598b's C1). If the
    zero-init head is not silent, the substrate baseline has drifted and nothing is comparable.
(b) THE RULE LAYER EXISTS -- ARM_2_MINTED actually minted a pool: crf_n_minted_total >= 2 and
    crf_live_rules_final >= 2 on >= MIN_PASS_SEEDS seeds. Without this, C2 would be measuring an
    INERT layer and a null would be unattributable. (Design-time probing measured the CRF
    minting only 2 rules regardless of ecology richness, so this floor is set at 2, not higher,
    and the realised count is recorded rather than assumed.)
(c) THE DV IS MEASURABLE -- every arm records a non-trivial number of P2 steps.

=== WHAT THIS RUN DOES NOT DO ===
It does not measure the mint event, distinctness, or retirement -- all three are structural
under this stack (GFLAG-0265/0266). It does not tag MECH-350/351/352. A C2 null is a fact about
this behavioural DV under this training signal; per MECH-439's F-dominance conversion ceiling it
does not by itself establish that the minted pool is representationally inert.

SLEEP DRIVER: not applicable (no sleep loop).
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.run_id import make_run_id
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# --- the reviewed instrument, imported VERBATIM rather than re-implemented -------------
# V3-EXQ-598b supplies _encoder_step (P0), _p1_train (P1 outcome-coupled REINFORCE) and
# _p2_reef_fraction (the DV). Re-deriving them would make any difference from 598b's
# published result unattributable. 598b itself is NOT modified.
_X598B_PATH = _REPO_ROOT / "experiments" / "v3_exq_598b_gap1_sd033a_bias_head_trainable_ablation.py"
_spec = importlib.util.spec_from_file_location("_x598b", _X598B_PATH)
X598B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X598B)

EXPERIMENT_TYPE = "v3_exq_1026_mech309_rule_layer_breaks_monomodal_collapse"
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-309"]
RELATED_EXQ = ["V3-EXQ-598b", "V3-EXQ-598c", "V3-EXQ-543l"]

SEEDS = [0, 1, 2]

# --- inherited from 598b, deliberately UNCHANGED (the DV and its band) ----------------
REEF_LO = X598B.REEF_LO                     # 0.20
REEF_HI = X598B.REEF_HI                     # 0.80
FROZEN_BIAS_MAX = X598B.FROZEN_BIAS_MAX     # 1e-4
TRAINABLE_BIAS_MIN = X598B.TRAINABLE_BIAS_MIN  # 0.002
MIN_PASS_SEEDS = X598B.MIN_PASS_SEEDS       # 2
P0_EPISODES = X598B.P0_EPISODES
P1_EPISODES = X598B.P1_EPISODES
P2_EPISODES = X598B.P2_EPISODES
STEPS_PER_EPISODE = X598B.STEPS_PER_EPISODE

# --- this run's own pre-registered floors ---------------------------------------------
CRF_MIN_MINTED = 2          # precondition (b): the rule layer must exist at all
CRF_MIN_LIVE = 2
# Earned by the smoke: all three arms returned a bit-identical reef_frac at reduced budget.
DV_MIN_ARM_SPREAD = 0.02   # min spread (max-min) across per-arm mean reef_visit_fraction

ARM_FROZEN = "ARM_0_FROZEN"
ARM_GRADIENT = "ARM_1_GRADIENT"
ARM_MINTED = "ARM_2_MINTED"


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _not_monomodal(frac: float) -> bool:
    return bool(REEF_LO < frac < REEF_HI)


def _make_agent(seed: int, train_rule_bias_head: bool, crf: bool):
    """598b's _make_agent, with the ONE added factor: the ARC-063 CandidateRuleField.

    Every other kwarg is copied from X598B._make_agent so the arms differ in exactly the
    factor each is named for. The CRF stack is the one MECH-349 names (SD-078 centered cue
    key + mature pool + availability maintenance); persist-across-episode-reset is ON
    because the P1/P2 loops reset the agent per episode and a pool wiped every episode
    cannot be a rule-apprehension LAYER at P2 -- precondition (b) records what it achieved.
    """
    torch.manual_seed(seed)
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **X598B.ENV_KWARGS)
    kw = dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=X598B.SELF_DIM,
        world_dim=X598B.WORLD_DIM,
        harm_dim=X598B.HARM_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        use_harm_stream=True,
        z_harm_dim=X598B.HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=X598B.HARM_A_DIM,
        harm_history_len=X598B.HARM_HISTORY_LEN,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        use_gated_policy=True,
        gated_policy_use_first_action_onehot=True,
        use_dacc=False,
        dacc_weight=0.0,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        use_lateral_pfc_analog=True,
        lateral_pfc_use_discriminator_source=True,
        lateral_pfc_discriminator_pool_weight=0.3,
        lateral_pfc_train_rule_bias_head=train_rule_bias_head,
    )
    if crf:
        kw.update(
            use_candidate_rule_field=True,
            crf_cue_centering=True,
            crf_mature_pool_dynamics=True,
            crf_availability_maintenance=True,
            crf_persist_rules_across_episode_reset=True,
        )
    config = REEConfig.from_dims(**kw)
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    config.gated_policy_use_differential_heads = True
    config.gated_policy_differential_bias_scale = 0.1
    config.gated_policy_mode_separation_floor = X598B.MODE_SEPARATION_FLOOR
    config.gated_policy_p1_w_deviation_aux_weight = X598B.P1_W_DEVIATION_AUX_WEIGHT
    return REEAgent(config), env


def config_slice(train_bias: bool, crf: bool, p0: int, p1: int, p2: int, steps: int) -> Dict[str, Any]:
    return {
        "lateral_pfc_train_rule_bias_head": bool(train_bias),
        "use_candidate_rule_field": bool(crf),
        "crf_cue_centering": bool(crf),
        "crf_mature_pool_dynamics": bool(crf),
        "crf_availability_maintenance": bool(crf),
        "crf_persist_rules_across_episode_reset": bool(crf),
        "env_kwargs": dict(X598B.ENV_KWARGS),
        "self_dim": X598B.SELF_DIM, "world_dim": X598B.WORLD_DIM,
        "harm_dim": X598B.HARM_DIM, "harm_a_dim": X598B.HARM_A_DIM,
        "mode_separation_floor": X598B.MODE_SEPARATION_FLOOR,
        "p1_w_deviation_aux_weight": X598B.P1_W_DEVIATION_AUX_WEIGHT,
        "p0_episodes": p0, "p1_episodes": p1, "p2_episodes": p2,
        "steps_per_episode": steps,
    }


def _run_cell(arm: str, seed: int, train_bias: bool, crf: bool,
              p0: int, p1: int, p2: int, steps: int) -> Dict[str, Any]:
    total = p0 + p1 + p2
    print("Seed %d Condition %s" % (seed, arm), flush=True)
    agent, env = _make_agent(seed, train_bias, crf)
    for ep in range(p0):
        X598B._encoder_step(agent, env, steps, total, arm, ep)
        if (ep + 1) % 10 == 0 or (ep + 1) == p0:
            print("  [train] %s seed=%d ep %d/%d" % (arm, seed, ep + 1, total), flush=True)
    p1m = X598B._p1_train(agent, env, p1, steps, total, arm, train_bias)
    print("  [train] %s seed=%d ep %d/%d" % (arm, seed, p0 + p1, total), flush=True)
    reef_frac = X598B._p2_reef_fraction(agent, env, p2, steps)
    print("  [train] %s seed=%d ep %d/%d" % (arm, seed, total, total), flush=True)

    row: Dict[str, Any] = {
        "arm": arm,
        "seed": int(seed),
        "train_rule_bias_head": bool(train_bias),
        "crf_enabled": bool(crf),
        **p1m,
        "p2_reef_visit_fraction": float(reef_frac),
        "not_monomodal": _not_monomodal(float(reef_frac)),
        "p2_episodes": p2,
        "p2_steps_per_episode": steps,
    }
    if agent.candidate_rule_field is not None:
        st = agent.candidate_rule_field.get_state()
        row.update({
            "crf_n_minted_total": int(st["crf_n_minted_total"]),
            "crf_live_rules_final": int(st["crf_n_slots_minted"]),
            "crf_n_retired_total": int(st["crf_n_retired_total"]),
            "crf_max_pairwise_rule_dist": float(st["crf_max_pairwise_rule_dist"]),
            "crf_frac_active": float(st["crf_frac_active"]),
        })
    else:
        row.update({"crf_n_minted_total": 0, "crf_live_rules_final": 0})
    print("verdict: %s  reef_frac=%.3f  bias=%.6f" % (
        "PASS" if row["not_monomodal"] else "FAIL", reef_frac,
        p1m.get("p1_mean_abs_lpfc_bias", float("nan"))), flush=True)
    return row


def _by_seed(rows: List[Dict[str, Any]], arm: str, key: str) -> Dict[int, Any]:
    return {r["seed"]: r[key] for r in rows if r["arm"] == arm}


def main(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = SEEDS[:1] if dry_run else SEEDS
    p0 = 3 if dry_run else P0_EPISODES
    p1 = 4 if dry_run else P1_EPISODES
    p2 = 2 if dry_run else P2_EPISODES
    steps = 30 if dry_run else STEPS_PER_EPISODE
    n_seeds = len(seeds)
    need = min(MIN_PASS_SEEDS, n_seeds)

    arms = [(ARM_FROZEN, False, False), (ARM_GRADIENT, True, False), (ARM_MINTED, False, True)]
    rows: List[Dict[str, Any]] = []
    for arm, tb, crf in arms:
        for sd in seeds:
            slice_ = config_slice(tb, crf, p0, p1, p2, steps)
            with arm_cell(
                sd, config_slice=slice_, script_path=Path(__file__),
                config_slice_declared=True,
                # 598b supplies P0/P1/P2 and lives in experiments/, which is NOT in the
                # default substrate-hash glob -- declare it so an edit to the instrument
                # busts these fingerprints instead of silently matching.
                extra_substrate_paths=[_X598B_PATH],
                include_driver_script_in_hash=True,
            ) as cell:
                row = _run_cell(arm, sd, tb, crf, p0, p1, p2, steps)
                cell.stamp(row)
            rows.append(row)

    froz = [r for r in rows if r["arm"] == ARM_FROZEN]
    grad = [r for r in rows if r["arm"] == ARM_GRADIENT]
    mint = [r for r in rows if r["arm"] == ARM_MINTED]

    # ---------------- preconditions ----------------------------------------
    froz_bias = {r["seed"]: r.get("p1_mean_abs_lpfc_bias", float("nan")) for r in froz}
    worst_froz = max(froz_bias.values()) if froz_bias else float("nan")
    crf_ok = sum(1 for r in mint
                 if r["crf_n_minted_total"] >= CRF_MIN_MINTED
                 and r["crf_live_rules_final"] >= CRF_MIN_LIVE)
    min_minted = min((r["crf_n_minted_total"] for r in mint), default=0)
    p2_steps_ok = all(r["p2_episodes"] * r["p2_steps_per_episode"] > 0 for r in rows)
    _arm_means = [statistics.fmean([r["p2_reef_visit_fraction"] for r in grp])
                  for grp in (froz, grad, mint) if grp]
    _arm_spread = (max(_arm_means) - min(_arm_means)) if len(_arm_means) > 1 else 0.0

    preconditions = [
        {"name": "baseline_head_silent", "kind": "readiness",
         "description": ("598b C1: the zero-init rule-bias head must remain silent in the "
                         "frozen arm, else the substrate baseline has drifted and the arms "
                         "are not comparable. Worst (max) seed reported."),
         "measured": float(worst_froz), "threshold": float(FROZEN_BIAS_MAX),
         "direction": "upper", "comparator": "<",
         "control": "zero-initialised head with train_rule_bias_head=False",
         "met": bool(_finite(worst_froz) and worst_froz < FROZEN_BIAS_MAX)},
        {"name": "rule_layer_actually_minted", "kind": "readiness",
         "description": ("C2 would measure an INERT layer if the CRF never built a pool. "
                         "Seeds with crf_n_minted_total >= %d AND crf_live_rules_final >= %d."
                         % (CRF_MIN_MINTED, CRF_MIN_LIVE)),
         "measured": float(crf_ok), "threshold": float(need), "direction": "lower",
         "control": ("design-time probing measured the CRF minting only 2 rules regardless of "
                     "ecology richness, so this floor is 2; realised counts are recorded"),
         "met": bool(crf_ok >= need)},
        {"name": "dv_responds_to_arm", "kind": "readiness",
         "description": ("The DV must actually respond to the manipulation. If every arm "
                         "returns a bit-identical reef_visit_fraction the measurement is inert "
                         "and no arm contrast means anything -- observed in the --dry-run smoke, "
                         "where all three arms returned 0.483 at reduced budget. Reported value "
                         "is the spread (max-min) across arm means."),
         "measured": float(_arm_spread), "threshold": float(DV_MIN_ARM_SPREAD),
         "direction": "lower",
         "control": "three arms whose rule-signal source differs by construction",
         "met": bool(_finite(_arm_spread) and _arm_spread >= DV_MIN_ARM_SPREAD)},
        {"name": "dv_measurable", "kind": "readiness",
         "description": "every arm ran a non-zero number of P2 steps",
         "measured": 1.0 if p2_steps_ok else 0.0, "threshold": 1.0, "direction": "lower",
         "control": "p2_episodes * steps_per_episode > 0 in every row",
         "met": bool(p2_steps_ok)},
    ]

    # ---------------- C1: gradient collapse reproduces ----------------------
    grad_moved = {r["seed"]: r.get("p1_mean_abs_lpfc_bias", 0.0) >= TRAINABLE_BIAS_MIN
                  for r in grad}
    grad_mono = {r["seed"]: (not r["not_monomodal"]) for r in grad}
    c1_seed = {s: bool(grad_moved.get(s) and grad_mono.get(s)) for s in seeds}
    c1_n = sum(1 for v in c1_seed.values() if v)
    c1_pass = c1_n >= need

    # ---------------- C2: the rule layer breaks it --------------------------
    # SINGLE-FACTOR contrast. ARM_MINTED and ARM_FROZEN both have the bias head FROZEN and
    # differ only in the CRF, so the pair isolates the rule layer. Comparing against
    # ARM_GRADIENT instead would vary two factors at once (bias head AND CRF) and a
    # difference could not be attributed to minting.
    froz_mono = {r["seed"]: (not r["not_monomodal"]) for r in froz}
    mint_nonmono = {r["seed"]: bool(r["not_monomodal"]) for r in mint}
    c2_seed = {s_: bool(mint_nonmono.get(s_) and froz_mono.get(s_)) for s_ in seeds}
    c2_n = sum(1 for v in c2_seed.values() if v)
    c2_pass = c2_n >= need

    preconditions_met = all(p["met"] for p in preconditions)
    load_bearing_pass = bool(c1_pass and c2_pass)
    outcome = "PASS" if (preconditions_met and load_bearing_pass) else "FAIL"

    criteria = [
        {"name": "C1_gradient_collapse_reproduces", "load_bearing": True, "passed": c1_pass,
         "measured": float(c1_n), "threshold": float(need),
         "units": "seeds where the trainable head MOVED and the policy was MONOMODAL",
         "per_seed": c1_seed,
         "bias_by_seed": {r["seed"]: r.get("p1_mean_abs_lpfc_bias") for r in grad},
         "reef_frac_by_seed": _by_seed(rows, ARM_GRADIENT, "p2_reef_visit_fraction"),
         "trainable_bias_min": TRAINABLE_BIAS_MIN,
         "monomodal_band": [REEF_LO, REEF_HI],
         "tests": "MECH-309's asserted equilibrium (reproduction of V3-EXQ-598b C2+C3)"},
        {"name": "C2_rule_layer_breaks_collapse", "load_bearing": True, "passed": c2_pass,
         "measured": float(c2_n), "threshold": float(need),
         "units": ("seeds where the CRF arm was INSIDE the band AND the matched "
                      "bias-frozen no-CRF arm was OUTSIDE it (single-factor contrast)"),
         "per_seed": c2_seed,
         "reef_frac_minted_by_seed": _by_seed(rows, ARM_MINTED, "p2_reef_visit_fraction"),
         "reef_frac_frozen_by_seed": _by_seed(rows, ARM_FROZEN, "p2_reef_visit_fraction"),
         "per_seed_frozen_monomodal": froz_mono,
         "monomodal_band": [REEF_LO, REEF_HI],
         "crf_minted_by_seed": _by_seed(rows, ARM_MINTED, "crf_n_minted_total"),
         "tests": "whether a NON-GRADIENT rule creator produces the split gradient could not"},
    ]

    combination_rule = ("PASS iff all preconditions met AND C1 AND C2. C1 is a two-sided "
                        "reproduction of MECH-309's own equilibrium claim and is a CRITERION, "
                        "not a precondition, so a non-collapsing gradient arm is REPORTED as "
                        "weakening MECH-309 rather than suppressed as instrument-not-ready.")

    criteria_non_degenerate = {
        "C1": bool(len({r["p2_reef_visit_fraction"] for r in grad}) > 1 or len(grad) == 1),
        "C2": bool(len({r["p2_reef_visit_fraction"] for r in mint}) > 1 or len(mint) == 1),
    }

    # ---------------- per-claim direction (see the docstring asymmetry) -----
    if not preconditions_met:
        label = "substrate_not_ready_requeue"
        summary = "A non-degeneracy precondition was not met; no verdict is admissible."
        dir_309 = "unknown"
        dir_349 = "non_contributory"
        claim_ids_out: List[str] = []
    elif c1_pass and c2_pass:
        label = "rule_layer_breaks_monomodal_collapse"
        summary = ("Gradient training collapsed the policy to a single regime (C1, reproducing "
                   "V3-EXQ-598b) and the non-gradient CandidateRuleField arm did not (C2): the "
                   "rule-apprehension layer produced the regime split gradient could not.")
        dir_309, dir_349 = "supports", "supports"
        claim_ids_out = list(CLAIM_IDS)
    elif c1_pass and not c2_pass:
        label = "collapse_reproduces_rule_layer_does_not_break_it"
        summary = ("Gradient collapse reproduced (C1), but the CandidateRuleField arm collapsed "
                   "too (C2 failed). MECH-309's equilibrium is supported; MECH-349 is recorded "
                   "NON_CONTRIBUTORY, not weakened -- its own what_would_answer forecloses "
                   "scoring a behavioural null against it (the MECH-439 F-dominance ceiling).")
        dir_309, dir_349 = "supports", "non_contributory"
        claim_ids_out = list(CLAIM_IDS)
    else:
        label = "gradient_arm_did_not_collapse"
        summary = ("The gradient arm did NOT reproduce monomodal collapse, so MECH-309's "
                   "asserted equilibrium does not hold on this substrate as configured. "
                   "MECH-349 is non_contributory: with no collapse to break, C2 answers nothing.")
        dir_309, dir_349 = "weakens", "non_contributory"
        claim_ids_out = list(CLAIM_IDS)

    def _flat(v: Any) -> Optional[float]:
        if isinstance(v, bool):
            return float(int(v))
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
        return None

    _readout_raw: Dict[str, Any] = {
        "c1_seeds": c1_n, "c1_passed": c1_pass,
        "c2_seeds": c2_n, "c2_passed": c2_pass,
        "seeds_required": need, "n_seeds": n_seeds,
        "reef_lo": REEF_LO, "reef_hi": REEF_HI,
        "mean_reef_frac_frozen": statistics.fmean([r["p2_reef_visit_fraction"] for r in froz]) if froz else 0.0,
        "mean_reef_frac_gradient": statistics.fmean([r["p2_reef_visit_fraction"] for r in grad]) if grad else 0.0,
        "mean_reef_frac_minted": statistics.fmean([r["p2_reef_visit_fraction"] for r in mint]) if mint else 0.0,
        "worst_frozen_bias": worst_froz,
        "frozen_bias_max": FROZEN_BIAS_MAX,
        "min_crf_minted": min_minted,
        "crf_min_minted_floor": CRF_MIN_MINTED,
        "preconditions_met": preconditions_met,
        "load_bearing_pass": load_bearing_pass,
    }
    readout = {k: v for k, v in ((k, _flat(v)) for k, v in _readout_raw.items()) if v is not None}

    full_config = {
        "seeds": seeds, "p0_episodes": p0, "p1_episodes": p1, "p2_episodes": p2,
        "steps_per_episode": steps, "reef_lo": REEF_LO, "reef_hi": REEF_HI,
        "frozen_bias_max": FROZEN_BIAS_MAX, "trainable_bias_min": TRAINABLE_BIAS_MIN,
        "min_pass_seeds": MIN_PASS_SEEDS, "crf_min_minted": CRF_MIN_MINTED,
        "env_kwargs": dict(X598B.ENV_KWARGS),
        "mode_separation_floor": X598B.MODE_SEPARATION_FLOOR,
        "p1_w_deviation_aux_weight": X598B.P1_W_DEVIATION_AUX_WEIGHT,
        "instrument_source": "V3-EXQ-598b (_encoder_step/_p1_train/_p2_reef_fraction), unmodified",
    }

    manifest: Dict[str, Any] = {
        "run_id": make_run_id(EXPERIMENT_TYPE),
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": claim_ids_out,
        "claim_ids_intended": list(CLAIM_IDS),
        "related_exq": RELATED_EXQ,
        "outcome": outcome,
        "evidence_direction": dir_309,
        "evidence_direction_per_claim": {"MECH-309": dir_309},
        "sleep_driver_pattern": "not_applicable_no_sleep_loop",
        "readout": readout,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "interpretation": {
            "label": label, "summary": summary,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "mech349_implication_not_tagged": (
                "MECH-349 is deliberately NOT in claim_ids, and this run scores it NOWHERE. "
                "Its own what_would_answer forecloses scoring a behavioural null against it "
                "(a MECH-439 F-dominance ceiling null; the ARC-062 GAP-B misattribution 21 "
                "autopsies have recorded), so this run could only ever HELP it -- a one-way "
                "ratchet, which is not evidence. What a C2 PASS would imply for MECH-349's "
                "claim to be 'the literal MECH-309 answer' is a governance reading, routed via "
                "GFLAG-0268, not a row this run writes. dir_349 is retained in the "
                "interpretation only as that reading, never as an evidence_direction."),
            "mech349_reading_if_scored": dir_349,
            "structural_facts_not_measurements": {
                "mint_distinctness_and_retirement_not_measured": (
                    "crf_max_pairwise_rule_dist is a deterministic lookup on live-rule count "
                    "(pinned_seed=6063) and crf_n_retired_total is structurally 0 under the "
                    "maintenance stack; both are recorded per cell for continuity and neither "
                    "is scored. See GFLAG-0265 / GFLAG-0266."),
            },
        },
        "claim_scope_note": (
            "MECH-350/351/352 are NOT tagged: neither representation, conflict nor credit is "
            "manipulated here. This run measures a BEHAVIOURAL DV (598b's reef_visit_fraction) "
            "and tags MECH-309, whose own claim that DV is, plus MECH-349 asymmetrically."),
        "arm_results": rows,
        "cell_summary": {
            arm: {"reef_frac_by_seed": _by_seed(rows, arm, "p2_reef_visit_fraction"),
                  "not_monomodal_by_seed": _by_seed(rows, arm, "not_monomodal"),
                  "crf_minted_by_seed": _by_seed(rows, arm, "crf_n_minted_total")}
            for arm in (ARM_FROZEN, ARM_GRADIENT, ARM_MINTED)
        },
    }

    out_path = write_flat_manifest(manifest, dry_run=dry_run, config=full_config, seeds=seeds,
                                   script_path=Path(__file__), started_at=t0, agent=None)
    manifest["_out_path"] = out_path
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="V3-EXQ-1026 MECH-309 rule-layer vs monomodal collapse")
    ap.add_argument("--dry-run", action="store_true",
                    help="1 seed, tiny budgets; manifest relocated out of evidence/")
    args = ap.parse_args()

    result = main(dry_run=args.dry_run)
    out_path = result["_out_path"]
    print()
    print("=== V3-EXQ-1026 MECH-309 rule-layer vs monomodal collapse ===")
    print("label:   %s" % result["interpretation"]["label"])
    print("outcome: %s" % result["outcome"])
    print("summary: %s" % result["interpretation"]["summary"])
    print("per-claim: %s" % result["evidence_direction_per_claim"])
    print("--- preconditions ---")
    for p in result["interpretation"]["preconditions"]:
        print("  %-28s measured=%-12.6f thr=%-10.6f met=%s"
              % (p["name"], p["measured"], p["threshold"], p["met"]))
    print("--- criteria ---")
    for c in result["criteria"]:
        print("  %-34s [LOAD-BEARING] measured=%-5.1f thr=%-5.1f passed=%s"
              % (c["name"], c["measured"], c["threshold"], c["passed"]))
    print("--- reef_visit_fraction by arm x seed (band %.2f-%.2f) ---" % (REEF_LO, REEF_HI))
    for arm, c in result["cell_summary"].items():
        vals = " ".join("s%d=%.3f" % (s, v) for s, v in sorted(c["reef_frac_by_seed"].items()))
        print("  %-16s %s" % (arm, vals))
    print("manifest: %s" % out_path)
    print("overall_outcome: %s" % result["outcome"])

    _o = str(result["outcome"]).upper()
    emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                 manifest_path=out_path, dry_run=args.dry_run)
