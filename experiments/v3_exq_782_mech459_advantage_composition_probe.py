#!/opt/local/bin/python3
"""V3-EXQ-782 -- MECH-459 probe R: advantage composition (pre/post standardisation) + critic
calibration on BC-visited states.

DIAGNOSTIC probe on INTERMEDIATE QUANTITIES (experiment_purpose=diagnostic -> excluded from
governance confidence/conflict scoring). PROMOTES / DEMOTES NOTHING. Routes to /failure-autopsy.
MECH-459 stays candidate (NARROWED). This run is explicitly NOT the adjudicating run for the
H-return-scale registry leg -- see PRE-REGISTRATION HYGIENE below.

SANCTIONED BY: REE_assembly/evidence/decisions/decision_MECH-459_registry_and_brake_2026-07-18.md
(Decision 2b). That adjudication:
  * REFUTED the STRONG form of MECH-459 (Finding V2). Global multiplicative reward-scale
    invariance is real (Finding V1) but NONE of the 770/771/772 legs is a global rescaling, so
    the eliminations were NOT laundered "by construction". Do not restate the strong form.
  * Left the WEAK form live: per-episode standardisation re-amplifies co-present novelty noise
    (NOVELTY_COEF 0.1/sqrt(visit_count), firing EVERY step) back to unit variance, swamping
    whatever shape change a treatment introduced. That is a NOISE-DOMINANCE claim -- empirical,
    not architectural -- and it is what probe R tests.
  * Ruled the shape-preserving normaliser KNOCK-OUT (probe K: (adv-mean)/(std+1e-8) ->
    adv / max(1, Per95(adv)-Per5(adv))) PERMITTED by the re-derive brake but the WRONG FIRST
    SPEND: it tests only the weak form, and a knock-out NULL is compatible with a
    bootstrap-signal-absence account too. Probe K is HELD, to be queued ONLY if R-(a) below
    returns the parity-rescaling signature.

LIVE PORTFOLIO IS UNTOUCHED. V3-EXQ-780 (claimed, ree-cloud-2) and V3-EXQ-781 (claimed,
ree-cloud-3) are NOT modified, and neither are mech457_explorer_classes.py /
mech457_bootstrap_explorer.py / mech457_fanout.py -- the instrumentation lives in a separate
mirror module (experiments/_lib/mech459_probe_r.py) precisely so the live portfolio's bytes,
substrate_hash and running processes are undisturbed. The substrate_queue entry
mech457_competence_bootstrap_explorer stays blocked_pending_discrimination.

MEASURED PATH. The REFERENCE composed-bootstrap ctrl arm -- 128-wide actor, 3x budget (3000 ep),
z_world DETACHED, credit-replay 3/topk 32, developmental drive anneal 1.0->0.05, RND on, NO BC,
NO approach drive. This is the SHARED CONTROL arm of both 780 and 781, which is what makes the
readout speak to both. Both representations (z_world detached + raw 5x5).

=== HYPOTHESES UNDER TEST ===
H-weak (MECH-459 surviving weak form). The two-sided normaliser
(running-std divide at explorer_classes.py:688, per-episode standardisation at :697) re-amplifies
the every-step novelty noise back to unit variance, so a reward-side shape change lands in the
gradient at parity with noise instead of dominating it. If true, forage-contact steps carry LESS
than their per-step share of |adv| mass BEFORE standardisation and are rescaled to roughly their
share (or above) AFTER it.
H-absence (bootstrap-signal-absence rival). There is no competence signal in the gradient at all
-- the actor never accumulates usable credit for forage contacts, normaliser or no normaliser.
If true, forage steps carry a tiny share of |adv| mass on BOTH sides of the standardiser, and the
critic is uninformative everywhere.
H-mse (bimodal-collapse half of the same story, tested by R-(b)). The critic minimises MSE
against a bimodal return distribution and settles on its never-observed MEAN, so it cannot
separate pre-reward from post-reward states even on a state distribution that contains genuinely
high-return states.

ANALYTIC CONSTRAINT THE GRID RESPECTS (stated so a reading is not over-claimed). The
standardisation is (a - mean) / (std + eps). Division by a positive scalar is a GLOBAL rescale
and leaves every |adv| mass FRACTION exactly invariant. Only the MEAN SUBTRACTION can move the
composition. So the parity-rescaling signature is a demanding, specific prediction (it requires
a large negative episode-mean advantage relative to forage-step advantages), not a formality --
which is what makes R-(a) a genuine discriminator. mean_episode_mean_adv is recorded so the
mechanism is auditable.

=== INTERPRETATION GRID ===
Notation. s = fraction of measured steps that are forage contacts. f_pre / f_post = forage share
of total |adv| mass, before / after standardisation. C_pre = f_pre/s, C_post = f_post/s (the
CONCENTRATION -- prevalence-normalised, so rarity alone cannot produce a "tiny" reading). C == 1
means forage steps carry exactly their per-step share of the gradient.

R-(a) ADVANTAGE COMPOSITION -- evidence directions BOTH ways:
  * PARITY-RESCALING (evidence FOR H-weak, AGAINST H-absence):
      C_pre < 0.5 (forage carries less than half its share pre) AND C_post >= 1.0 (at or above
      its share post) AND (C_post - C_pre) >= 0.25.
      -> the normaliser IS moving competence signal into parity with noise. MECH-459's weak form
         graduates to a rival ANSWER; probe K becomes worth queuing; H-return-scale gets
         registered (see PRE-REGISTRATION HYGIENE). SELF-ROUTE: parity_rescaling_signature.
  * STAYS-TINY (evidence AGAINST H-weak, FOR H-absence):
      C_post < 1.0 AND (C_post - C_pre) < 0.25 -- the forage share is small on BOTH sides.
      -> the normaliser is EXONERATED; the weak form falls with it. Probe K should NOT be
         queued. MECH-459 routes to `weakened`. SELF-ROUTE: forage_mass_tiny_both_sides.
  * ALREADY-CONCENTRATED (evidence AGAINST H-weak by a different route):
      C_pre >= 0.5 -- forage steps already carry near/above their share BEFORE standardisation,
      so the weak form's premise (competence signal is swamped pre-normalisation) is false.
      -> MECH-459 weak form falls; the deficit is downstream of the advantage composition.
      SELF-ROUTE: forage_mass_already_concentrated.

R-(b) CRITIC CALIBRATION ON BC-VISITED STATES -- evidence directions BOTH ways. The
demonstrator (LocalViewGreedyPolicy, 48.05 @D3; the policy 748 distilled to 32.72) visits
genuinely high-return states, so G is bimodal by construction. V = trained critic value,
G = realized discounted return-to-go on the SAME reward scale the critic regressed against.
  * COLLAPSED-TO-MEAN (evidence FOR H-mse):
      std(V)/std(G) < 0.25 AND |mean(V)-mean(G)| <= 0.5*std(G) AND separation_ratio < 0.25.
      -> the critic sits on the never-observed mean of a bimodal return distribution and does
         not separate pre-reward from far states. SELF-ROUTE component: critic_collapsed_to_mean.
  * FLAT-UNINFORMED (evidence FOR H-absence, AGAINST H-mse):
      std(V)/std(G) < 0.25 AND |mean(V)-mean(G)| > 0.5*std(G).
      -> a flat critic that did not even learn the mean; nothing to rescale.
      SELF-ROUTE component: critic_flat_uninformed.
  * CALIBRATED (evidence AGAINST both H-mse and H-absence):
      std(V)/std(G) >= 0.25 AND separation_ratio >= 0.25 -- the critic does separate.
      -> the value pathway is not the locus. SELF-ROUTE component: critic_separates.
  * PARTIAL-NO-SEPARATION (residual; neither collapse account is clean):
      std(V)/std(G) >= 0.25 but separation_ratio < 0.25 -- V varies but not along the pre-vs-far
      axis, so it is tracking something other than return-relevant structure.
      SELF-ROUTE component: critic_partial_no_separation.

JOINT ROUTING. The manifest reports both readouts independently (adv_composition_verdict and
critic_calibration_verdict) and a combined interpretation.label; /failure-autopsy adjudicates.
A DISAGREEMENT (e.g. parity-rescaling with a flat critic) is itself informative and is NOT
collapsed into one number.

SECOND, INDEPENDENT PAYOFF -- is a V3-EXQ-781 NULL readable? (Finding V3 of the adjudication.)
  * V3-EXQ-780's BC auxiliary is a LOSS-side cross-entropy term (explorer_classes.py:703-708).
    It never enters `shaped`, so the normaliser cannot swamp it: a 780 null is NOT confounded by
    MECH-459 either way.
  * V3-EXQ-781's approach drive is added directly INTO `shaped` (explorer_classes.py:660-662),
    INSIDE the normaliser. So a 781 NULL is confounded by the weak form while a 781 POSITIVE is
    not. R-(a) resolves which: the parity-rescaling signature means a 781 null must be read as
    "confounded, not eliminating" (H-approach-primitive stays live pending probe K); the
    stays-tiny / already-concentrated signatures mean a 781 null can be read at face value.
  This is recorded as `exq_781_null_readable` in the manifest headline and is decision-relevant
  WITHIN 781's runtime, which no other pending work delivers.

PRE-REGISTRATION HYGIENE (stated now so the sequence cannot be challenged later). Probe R is a
diagnostic on INTERMEDIATE quantities and is explicitly NOT the adjudicating run for the
H-return-scale registry leg -- the adjudicating run would be a subsequent competence-LIFTING
experiment. So if R-(a) returns the parity signature, registering H-return-scale in
hypothesis_space_registry.v1.json (question competence_floor, axis normalisation-pathway, via a
labelled fanout_growth_events[] entry citing decision_MECH-459_registry_and_brake_2026-07-18.md
as fanout_source) AFTER this run still satisfies the invariant pre_registered_utc <= resolved_utc
of the adjudicating run. Per labelled_fanout_growth, the surviving ratio must then be reported
BOTH ways: against the ORIGINAL frozen count 7 and against the CURRENT count (12, or 13 with
H-return-scale added).

RE-DERIVE BRAKE. MECH-459's substrate_ceiling/non_contributory autopsy count is 0 -- the brake
does not fire on this claim. Independently, a diagnostic whose purpose is to discriminate WHY a
ceiling holds is named as NOT braked. This is not another MECH-457 same-axis re-pose: it changes
no config/env/credit/capacity knob and runs no treatment arm at all -- it MEASURES the reference
control path.

GOV-REUSE-1. Decisive readouts = per-step |adv| mass fractions on both sides of the per-episode
standardiser, and trained-critic V vs realized return-to-go on demonstrator-visited states. These
are INTERMEDIATE TRAINING QUANTITIES that no manifest has ever recorded (grep over
evidence/experiments/*.json for advantage_composition / critic_calibration /
pre_standardisation returns nothing; 770/771/772/780/781 record only end-of-run foraging
competence and guard aggregates). Not recoverable by reanalysis -> run.

READINESS (P0 readiness-assert; SAME statistic as each load-bearing criterion):
  * anchors -- local_view_greedy AND greedy_oracle clear the 1.0 floor @D3 (env solvable from
    the learner's own view; the shared lineage gate).
  * R-(a) is COUNT-gated (mass fractions over forage steps), so the readiness check is a COUNT:
    n_forage_steps in the measured window >= 30 per cell. With zero forage steps both C_pre and
    C_post are trivially 0 and "stays tiny" would be vacuous rather than falsifying.
  * R-(b) is SPREAD-gated (std(V)/std(G) and a mean-separation), so the readiness check is a
    SPREAD on the positive control: std(G) over demonstrator-visited states >= 0.25. A degenerate
    return distribution cannot show a collapse.
  Below any readiness floor -> substrate_not_ready_requeue (FAIL; NEVER a substrate-verdict
  label such as substrate_ceiling / does_not_support / *_nondiscriminative).

MINT (mint-as-you-go). Both probe arms emit reuse-ELIGIBLE per representation (rng_fully_reset
via arm_cell + config_slice_declared + include_driver_script_in_hash=False); the probe mechanism
lives in experiments/_lib/mech459_probe_r.py (inside the substrate hash).

evidence_direction = "unknown" (DIAGNOSTIC; the verdict lives in interpretation.label and the
two per-readout verdicts, adjudicated by /failure-autopsy).

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

Shared machinery: experiments/_lib/mech459_probe_r.py (the instrumented mirror) +
mech457_explorer_classes.py (RepAgent / RNDModule / credit replay -- imported, NOT modified) +
mech457_bootstrap_explorer.py (config + anneal) + mech457_fanout.py (anchors / readiness /
budgets) + capability_eval.LocalViewGreedyPolicy (the demonstrator for R-(b)).
ASCII-only in all runtime strings.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.capability_eval import (  # noqa: E402
    COMPETENCE_RESOURCE_FLOOR,
    LocalViewGreedyPolicy,
    evaluate_seed,
)
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
import experiments._lib.mech457_bootstrap_explorer as boot  # noqa: E402
import experiments._lib.mech457_explorer_classes as mech  # noqa: E402
import experiments._lib.mech457_fanout as fan  # noqa: E402
import experiments._lib.mech459_probe_r as probe  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_782_mech459_advantage_composition_probe"
QUEUE_ID = "V3-EXQ-782"
CLAIM_IDS: List[str] = ["MECH-459"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

DEVICE = fan.DEVICE

# ---- Reference (non-regressed) composed-bootstrap build: the SHARED ctrl arm of 780 and 781.
REF_ACTOR_CRITIC_HIDDEN = fan.ACTOR_CRITIC_HIDDEN      # 128 (reference width)
REF_BUDGET_MULTIPLIER = 3                              # 3x the 1000-ep plateau budget
REF_CREDIT_PASSES = mech.CREDIT_REPLAY_PASSES          # 3
REF_CREDIT_TOPK = mech.CREDIT_TOPK                     # 32
REF_COTRAIN_ENCODER = False                            # z_world DETACHED

# ---- Pre-registered thresholds (declared here; NEVER derived from the run's own statistics) ----
# R-(a): prevalence-normalised concentration C = (forage |adv| mass fraction) / (forage step
# fraction). C == 1 means forage steps carry exactly their per-step share of the gradient.
CONC_TINY_PRE = 0.5          # C_pre below this = forage carries less than half its share pre
CONC_PARITY_POST = 1.0       # C_post at/above this = forage at or above its share post
CONC_RESCALE_DELTA = 0.25    # minimum C_post - C_pre for a real rescaling (not measurement noise)
# R-(b): critic-vs-return calibration on demonstrator-visited states.
COLLAPSE_STD_RATIO = 0.25    # std(V)/std(G) below this = the critic has collapsed
MEAN_TRACK_TOL_SD = 0.5      # |mean(V)-mean(G)| <= this many std(G) = tracking the mean
SEPARATION_RATIO_FLOOR = 0.25  # value_separation / return_separation below this = no separation
# Readiness floors (same statistic as the criterion each one gates).
MIN_FORAGE_STEPS = 30        # COUNT floor for the count-gated R-(a) mass fractions
MIN_RETURN_STD = 0.25        # SPREAD floor for the spread-gated R-(b) collapse test
# R-(b) rollout budget + horizon.
DEMO_EPISODES = 20
PRE_REWARD_HORIZON = 10      # a state is "pre-reward" if a contact occurs within this many steps
MEASURE_WINDOW = 200         # episodes accumulated at the start (early) and end (late) of RL

REPRESENTATIONS: Tuple[str, ...] = ("z_world", "raw_view")
_REP_TAG = {"z_world": "zworld", "raw_view": "raw"}


def _arm_id(rep: str) -> str:
    return f"probeR_ctrl_{_REP_TAG[rep]}"


PROBE_ARMS: Tuple[str, ...] = tuple(_arm_id(r) for r in REPRESENTATIONS)
ARM_ORDER: Tuple[str, ...] = PROBE_ARMS + fan.ANCHOR_ARMS


def _make_cfg(on_budget: int) -> boot.BootstrapExplorerConfig:
    """The reference composed bootstrap ctrl -- identical to the 770/780/781 ctrl arm.
    NO BC, NO approach drive: probe R measures the shared control path, it manipulates nothing."""
    return boot.BootstrapExplorerConfig(
        use_rnd=True,
        intrinsic_coef_start=1.0, intrinsic_coef_end=boot.ON_INTRINSIC_COEF_END,
        anneal_fraction=boot.ON_ANNEAL_FRACTION,
        warm_start_fraction=0.0,
        entropy_beta_start=boot.ON_ENTROPY_BETA_START, entropy_beta_end=boot.ON_ENTROPY_BETA_END,
        credit_replay=True, credit_replay_passes=REF_CREDIT_PASSES, credit_topk=REF_CREDIT_TOPK,
        n_episodes=int(on_budget),
        actor_critic_hidden=REF_ACTOR_CRITIC_HIDDEN, cotrain_encoder=REF_COTRAIN_ENCODER,
    )


def _config_slice(arm_id: str, rep: str, cfg: boot.BootstrapExplorerConfig,
                  env_kwargs: Dict[str, Any], p0: int, eval_eps: int, steps: int,
                  demo_eps: int, measure_window: int) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "arm_id": arm_id, "rung_id": fan.RUNG_ID, "env_kwargs": dict(env_kwargs),
        "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
        "kind": "mech459_advantage_composition_probe", "representation": rep,
        "measure_window_episodes": int(measure_window),
        "demo_episodes": int(demo_eps), "pre_reward_horizon": int(PRE_REWARD_HORIZON),
        "p0_warmup_episodes": int(p0) if rep == "z_world" else 0,
    }
    base.update(cfg.as_slice())
    return base


def _run_probe_cell(rep: str, env_kwargs: Dict[str, Any], seed: int, p0: int, on_budget: int,
                    eval_eps: int, steps: int, demo_eps: int,
                    measure_window: int) -> Dict[str, Any]:
    arm_id = _arm_id(rep)
    cfg = _make_cfg(on_budget)

    warm_env = x734._make_env(seed, env_kwargs)
    rep_agent = mech.make_rep(
        rep, warm_env, seed=seed, p0=p0, steps=steps,
        actor_critic_hidden=int(cfg.actor_critic_hidden),
        cotrain_encoder=bool(cfg.cotrain_encoder),
    )

    train_env = x734._make_env(seed, env_kwargs)
    guard = probe.train_probed_bootstrap(
        rep_agent, train_env, seed=seed, steps=steps, arm_label=arm_id, cfg=cfg,
        denom=cfg.n_episodes, measure_window=int(measure_window),
    )

    # R-(b): the trained critic read on the demonstrator's own state distribution.
    demo = LocalViewGreedyPolicy(seed=seed)
    demo_env = x734._make_env(seed, env_kwargs)
    calib = probe.critic_calibration_on_demo_states(
        rep_agent, demo_env, demo, seed=seed, episodes=int(demo_eps), steps=steps,
        reward_scale=float(guard["final_reward_std_scale"]),
        pre_reward_horizon=int(PRE_REWARD_HORIZON),
    )

    eval_env = x734._make_env(seed, env_kwargs)
    row = evaluate_seed(rep_agent.eval_policy(arm_id), eval_env, eval_eps, steps)
    row["mean_train_forage_recent"] = float(guard.get("mean_train_forage_recent", 0.0))
    row["mean_intrinsic_reward_recent"] = float(guard.get("mean_intrinsic_reward_recent", 0.0))
    row["n_credit_replay_passes"] = int(guard.get("n_credit_replay_passes", 0))
    row["final_reward_std_scale"] = float(guard.get("final_reward_std_scale", 0.0))
    row["adv_composition"] = guard["adv_composition"]
    row["critic_calibration"] = calib
    return row


def _late(row: Dict[str, Any]) -> Dict[str, Any]:
    return dict(((row.get("adv_composition") or {}).get("late")) or {})


def _forage_conc(row: Dict[str, Any]) -> Tuple[float, float, int]:
    """(C_pre, C_post, n_forage_steps) from a cell's LATE measurement window."""
    late = _late(row)
    n_forage = int((late.get("n_steps_by_class") or {}).get("forage", 0))
    c_pre = float((late.get("concentration_pre") or {}).get("forage", 0.0))
    c_post = float((late.get("concentration_post") or {}).get("forage", 0.0))
    return c_pre, c_post, n_forage


def _classify_composition(c_pre: float, c_post: float) -> str:
    if c_pre >= CONC_TINY_PRE:
        return "forage_mass_already_concentrated"
    if c_post >= CONC_PARITY_POST and (c_post - c_pre) >= CONC_RESCALE_DELTA:
        return "parity_rescaling_signature"
    return "forage_mass_tiny_both_sides"


def _classify_critic(calib: Dict[str, Any]) -> str:
    std_ratio = float(calib.get("value_std_over_return_std", 0.0))
    mean_gap_sd = float(calib.get("value_minus_return_mean_in_return_sd", 0.0))
    sep_ratio = abs(float(calib.get("separation_ratio", 0.0)))
    if std_ratio >= COLLAPSE_STD_RATIO and sep_ratio >= SEPARATION_RATIO_FLOOR:
        return "critic_separates"
    if std_ratio < COLLAPSE_STD_RATIO and mean_gap_sd <= MEAN_TRACK_TOL_SD:
        return "critic_collapsed_to_mean"
    if std_ratio < COLLAPSE_STD_RATIO:
        return "critic_flat_uninformed"
    return "critic_partial_no_separation"


def run_experiment(seeds: List[int], p0: int, on_budget: int, eval_eps: int, steps: int,
                   demo_eps: int, measure_window: int) -> Dict[str, Any]:
    print(
        f"MECH-459 probe R: advantage composition + critic calibration "
        f"({len(ARM_ORDER)} arms x 1 rung [{fan.RUNG_ID}] x {len(seeds)} seeds; "
        f"P0={p0}, RL_budget={on_budget}, measure_window={measure_window}, "
        f"demo_eps={demo_eps}, eval={eval_eps}, steps={steps}; "
        f"ref_hidden={REF_ACTOR_CRITIC_HIDDEN}, budget_mult={REF_BUDGET_MULTIPLIER}, "
        f"z_world_detached={not REF_COTRAIN_ENCODER}; measured path = the 780/781 SHARED ctrl)",
        flush=True,
    )
    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    per_arm_forage: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    all_cells: List[Dict[str, Any]] = []

    def _run_cell(arm_id: str, seed: int, rep: Optional[str]) -> Dict[str, Any]:
        print(f"Seed {seed} Condition {fan.RUNG_ID}:{arm_id}", flush=True)
        if arm_id in PROBE_ARMS:
            cfg = _make_cfg(on_budget)
            slice_cfg = _config_slice(arm_id, rep, cfg, env_kwargs, p0, eval_eps, steps,
                                      demo_eps, measure_window)
        else:
            slice_cfg = {"arm_id": arm_id, "rung_id": fan.RUNG_ID, "env_kwargs": dict(env_kwargs),
                         "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
                         "kind": "anchor"}
        with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__),
                      config_slice_declared=True, include_driver_script_in_hash=False) as cell:
            if arm_id in PROBE_ARMS:
                row = _run_probe_cell(rep, env_kwargs, seed, p0, on_budget, eval_eps, steps,
                                      demo_eps, measure_window)
            else:
                anchor_env = x734._make_env(seed, env_kwargs)
                row = fan.run_anchor_cell(arm_id, anchor_env, seed, eval_eps, steps)
            row["rung_id"] = fan.RUNG_ID
            row["arm_id"] = arm_id
            row["seed"] = int(seed)
            cell.stamp(row)
        forage = float(row["foraging_competence"])
        per_arm_forage[arm_id].append(forage)
        all_cells.append(row)
        if arm_id in PROBE_ARMS:
            c_pre, c_post, n_f = _forage_conc(row)
            cal = row["critic_calibration"]
            print(
                f"  probeR {arm_id} seed={seed}: forage_steps={n_f} "
                f"C_pre={c_pre} C_post={c_post} | "
                f"std(V)/std(G)={cal['value_std_over_return_std']} "
                f"sep_ratio={cal['separation_ratio']}",
                flush=True,
            )
        print(
            f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'} "
            f"(arm={arm_id} seed={seed} forage/ep={forage})", flush=True,
        )
        return row

    for arm_id in fan.ANCHOR_ARMS:
        for seed in seeds:
            _run_cell(arm_id, seed, None)

    def _mean(arm: str) -> float:
        vals = per_arm_forage[arm]
        return float(sum(vals) / len(vals)) if vals else 0.0

    local_view_mean = _mean("local_view_greedy")
    oracle_mean = _mean("greedy_oracle")
    anchors_ready = bool(
        local_view_mean >= COMPETENCE_RESOURCE_FLOOR and oracle_mean >= COMPETENCE_RESOURCE_FLOOR
    )

    if anchors_ready:
        for rep in REPRESENTATIONS:
            for seed in seeds:
                _run_cell(_arm_id(rep), seed, rep)
    else:
        print(
            f"readiness UNMET (local_view={local_view_mean} oracle={oracle_mean}); "
            f"skipping probe training -> substrate_not_ready_requeue", flush=True,
        )

    # ---------------- per-representation readouts ----------------
    def _cells_for(arm_id: str) -> List[Dict[str, Any]]:
        return [c for c in all_cells if c.get("arm_id") == arm_id]

    per_rep: Dict[str, Any] = {}
    all_forage_steps: List[int] = []
    all_return_stds: List[float] = []
    comp_verdicts: List[str] = []
    crit_verdicts: List[str] = []

    for rep in REPRESENTATIONS:
        arm_id = _arm_id(rep)
        cells = _cells_for(arm_id)
        c_pres, c_posts, n_forages = [], [], []
        std_ratios, mean_gaps, sep_ratios, ret_stds, corrs = [], [], [], [], []
        for c in cells:
            cp, cq, nf = _forage_conc(c)
            c_pres.append(cp); c_posts.append(cq); n_forages.append(nf)
            cal = c.get("critic_calibration") or {}
            std_ratios.append(float(cal.get("value_std_over_return_std", 0.0)))
            mean_gaps.append(float(cal.get("value_minus_return_mean_in_return_sd", 0.0)))
            sep_ratios.append(float(cal.get("separation_ratio", 0.0)))
            ret_stds.append(float(cal.get("return_std", 0.0)))
            corrs.append(float(cal.get("value_return_correlation", 0.0)))

        def _m(vals: List[float]) -> float:
            return round(float(sum(vals) / len(vals)), 6) if vals else 0.0

        c_pre_mean, c_post_mean = _m(c_pres), _m(c_posts)
        comp_verdict = _classify_composition(c_pre_mean, c_post_mean)
        crit_verdict = _classify_critic({
            "value_std_over_return_std": _m(std_ratios),
            "value_minus_return_mean_in_return_sd": _m(mean_gaps),
            "separation_ratio": _m(sep_ratios),
        })
        comp_verdicts.append(comp_verdict)
        crit_verdicts.append(crit_verdict)
        all_forage_steps.extend(n_forages)
        all_return_stds.extend(ret_stds)

        late_step_fracs = [
            float((_late(c).get("step_fraction_by_class") or {}).get("forage", 0.0)) for c in cells
        ]
        per_rep[rep] = {
            "arm_id": arm_id,
            # R-(a)
            "forage_concentration_pre_per_seed": [round(v, 6) for v in c_pres],
            "forage_concentration_post_per_seed": [round(v, 6) for v in c_posts],
            "forage_concentration_pre_mean": c_pre_mean,
            "forage_concentration_post_mean": c_post_mean,
            "forage_concentration_delta": round(c_post_mean - c_pre_mean, 6),
            "forage_step_fraction_per_seed": [round(v, 8) for v in late_step_fracs],
            "n_forage_steps_per_seed": [int(v) for v in n_forages],
            "adv_composition_verdict": comp_verdict,
            # R-(b)
            "value_std_over_return_std_per_seed": [round(v, 6) for v in std_ratios],
            "value_std_over_return_std_mean": _m(std_ratios),
            "value_mean_gap_in_return_sd_mean": _m(mean_gaps),
            "separation_ratio_per_seed": [round(v, 6) for v in sep_ratios],
            "separation_ratio_mean": _m(sep_ratios),
            "return_std_per_seed": [round(v, 6) for v in ret_stds],
            "value_return_correlation_mean": _m(corrs),
            "critic_calibration_verdict": crit_verdict,
        }

    # ---------------- readiness (count-gated R-(a), spread-gated R-(b)) ----------------
    min_forage_steps = int(min(all_forage_steps)) if all_forage_steps else 0
    min_return_std = float(min(all_return_stds)) if all_return_stds else 0.0
    forage_count_ready = bool(anchors_ready and min_forage_steps >= MIN_FORAGE_STEPS)
    return_spread_ready = bool(anchors_ready and min_return_std >= MIN_RETURN_STD)
    readiness_met = bool(anchors_ready and forage_count_ready and return_spread_ready)

    # ---------------- joint self-route ----------------
    def _agree(vs: List[str]) -> str:
        uniq = sorted(set(vs))
        return uniq[0] if len(uniq) == 1 else "mixed_across_representations"

    comp_overall = _agree(comp_verdicts) if comp_verdicts else "unmeasured"
    crit_overall = _agree(crit_verdicts) if crit_verdicts else "unmeasured"
    any_parity = bool("parity_rescaling_signature" in comp_verdicts)

    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    elif any_parity:
        outcome = "PASS"
        label = f"parity_rescaling_signature__{crit_overall}"
    else:
        outcome = "FAIL"
        label = f"{comp_overall}__{crit_overall}"

    # Finding V3 corollary: is a V3-EXQ-781 null readable at face value?
    exq_781_null_readable = bool(readiness_met and not any_parity)

    degeneracy = check_degeneracy({
        "forage_adv_concentration_pre_post": {
            "values": (
                [per_rep[r]["forage_concentration_pre_mean"] for r in REPRESENTATIONS]
                + [per_rep[r]["forage_concentration_post_mean"] for r in REPRESENTATIONS]
            )
        },
        "demo_state_return_spread": {
            "values": all_return_stds, "floor": float(MIN_RETURN_STD),
        },
    })

    interpretation = {
        "label": label,
        "preconditions": [
            fan.readiness_precondition(local_view_mean),
            {"name": "greedy_oracle_clears_floor_at_d3", "kind": "readiness",
             "description": "Env is floor-achievable with global info (achievability anchor).",
             "control": "greedy_oracle foraging_competence @D3 vs the 1.0 floor",
             "measured": round(oracle_mean, 6), "threshold": float(COMPETENCE_RESOURCE_FLOOR),
             "met": bool(oracle_mean >= COMPETENCE_RESOURCE_FLOOR)},
            {"name": "forage_steps_in_measured_window_supra_count_floor", "kind": "readiness",
             "description": (
                 "COUNT readiness matching the count-gated R-(a) criterion: the forage |adv| "
                 "mass fractions are computed OVER forage-contact steps, so with too few such "
                 "steps a 'stays tiny' reading is vacuous (no signal to swamp) rather than a "
                 "falsification of the weak form. Minimum over all probe cells."
             ),
             "control": "n forage-contact steps in the LATE measurement window, per probe cell",
             "measured": int(min_forage_steps), "threshold": int(MIN_FORAGE_STEPS),
             "met": forage_count_ready},
            {"name": "demo_state_return_spread_supra_floor", "kind": "readiness",
             "description": (
                 "SPREAD readiness matching the spread-gated R-(b) criterion: the collapse test "
                 "routes on std(V)/std(G), so a degenerate G distribution over demonstrator-"
                 "visited states makes the ratio uninterpretable. The demonstrator forages "
                 "(48.05 @D3), so its state distribution MUST carry a bimodal return spread."
             ),
             "control": "std(return-to-go) over LocalViewGreedyPolicy-visited states, min over cells",
             "measured": round(min_return_std, 6), "threshold": float(MIN_RETURN_STD),
             "met": return_spread_ready},
        ],
        "criteria": [
            {"name": "R_a_forage_adv_mass_parity_rescaled_post_standardisation",
             "load_bearing": True, "passed": bool(any_parity)},
            {"name": "R_b_critic_separates_pre_from_post_reward_states",
             "load_bearing": False, "passed": bool(crit_overall == "critic_separates")},
        ],
        "criteria_non_degenerate": {
            "local_view_clears_floor_at_d3": bool(local_view_mean >= COMPETENCE_RESOURCE_FLOOR),
            "oracle_clears_floor_at_d3": bool(oracle_mean >= COMPETENCE_RESOURCE_FLOOR),
            "forage_steps_present_in_window": forage_count_ready,
            "demo_return_distribution_has_spread": return_spread_ready,
            "concentration_pre_post_spread": bool(degeneracy["non_degenerate"]),
        },
    }

    result: Dict[str, Any] = {
        "outcome": outcome,
        "interpretation": interpretation,
        "interpretation_label": label,
        "adv_composition_verdict": comp_overall,
        "critic_calibration_verdict": crit_overall,
        "evidence_direction": "unknown",
        "evidence_direction_per_claim": {"MECH-459": "unknown"},
        "readiness": {
            "readiness_met": readiness_met,
            "anchors_ready": anchors_ready,
            "forage_count_ready": forage_count_ready,
            "return_spread_ready": return_spread_ready,
            "local_view_greedy_d3": round(local_view_mean, 6),
            "greedy_oracle_d3": round(oracle_mean, 6),
            "min_forage_steps_in_window": int(min_forage_steps),
            "min_demo_return_std": round(min_return_std, 6),
        },
        "headline": {
            "adv_composition_verdict": comp_overall,
            "critic_calibration_verdict": crit_overall,
            "any_rep_parity_rescaling_signature": any_parity,
            "per_representation": per_rep,
            "exq_781_null_readable": exq_781_null_readable,
            "exq_781_null_readable_note": (
                "V3-EXQ-781's approach drive enters `shaped` INSIDE the normaliser "
                "(explorer_classes.py:660-662), so a 781 NULL is confounded by the MECH-459 weak "
                "form while a 781 POSITIVE is not. V3-EXQ-780's BC auxiliary is a LOSS-side CE "
                "term (:703-708) that never enters `shaped` and is NOT confounded either way. "
                "true here = no parity-rescaling signature -> a 781 null reads at face value; "
                "false = parity signature (or readiness unmet) -> a 781 null must be read as "
                "confounded, not eliminating."
            ),
            "probe_k_recommended": bool(any_parity),
            "probe_k_note": (
                "Probe K (the shape-preserving normaliser knock-out, adv / max(1, Per95-Per5)) "
                "is sanctioned by Decision 2a but HELD. Queue it ONLY on the parity-rescaling "
                "signature. On forage_mass_tiny_both_sides or forage_mass_already_concentrated "
                "the weak form falls, K should NOT be queued, and MECH-459 routes to `weakened`."
            ),
            "hypothesis_registry_action": (
                "register H-return-scale (axis normalisation-pathway) under question "
                "competence_floor via a labelled fanout_growth_events[] entry citing "
                "decision_MECH-459_registry_and_brake_2026-07-18.md as fanout_source; report the "
                "surviving ratio BOTH ways (vs the ORIGINAL frozen count 7 and the CURRENT count "
                "12 -> 13)" if any_parity else
                "NO registry write: MECH-459 does not graduate to a rival answer; counts stay "
                "7 / 12 as ruled by Decision 1."
            ),
            "d3_local_view_greedy_denominator": round(local_view_mean, 6),
            "d3_greedy_oracle": round(oracle_mean, 6),
            "d3_random_walk": round(_mean("random_walk"), 6),
        },
        "thresholds": {
            "concentration_tiny_pre": CONC_TINY_PRE,
            "concentration_parity_post": CONC_PARITY_POST,
            "concentration_rescale_delta": CONC_RESCALE_DELTA,
            "collapse_std_ratio": COLLAPSE_STD_RATIO,
            "mean_track_tol_in_return_sd": MEAN_TRACK_TOL_SD,
            "separation_ratio_floor": SEPARATION_RATIO_FLOOR,
            "min_forage_steps": MIN_FORAGE_STEPS,
            "min_return_std": MIN_RETURN_STD,
        },
        "per_arm": {a: fan.summarize(per_arm_forage[a]) for a in ARM_ORDER},
        "reference_band": boot.reference_band(),
        "denominators": {
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
            "local_view_greedy_d3_live": round(local_view_mean, 6),
            "local_view_greedy_d3_738_reference": float(fan.DENOM_738_D3_REFERENCE),
        },
        "arm_results": all_cells,
        "non_degenerate": bool(degeneracy["non_degenerate"]),
        "degeneracy_reason": degeneracy["degeneracy_reason"],
        "degenerate_metrics": degeneracy["degenerate_metrics"],
    }
    return result


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool,
                    cfg: Dict[str, Any]) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": timestamp_utc,
        "dry_run": bool(dry_run),
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "interpretation": result["interpretation"],
        "interpretation_label": result["interpretation_label"],
        "adv_composition_verdict": result["adv_composition_verdict"],
        "critic_calibration_verdict": result["critic_calibration_verdict"],
        "readiness": result["readiness"],
        "headline": result["headline"],
        "thresholds": result["thresholds"],
        "denominators": result["denominators"],
        "per_arm": result["per_arm"],
        "reference_band": result["reference_band"],
        "arm_results": result["arm_results"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "degenerate_metrics": result["degenerate_metrics"],
        "sleep_driver_pattern": "none",
        "reuse_mint": {
            "reusable_arms": list(PROBE_ARMS),
            "reuse_eligible": True,
            "note": (
                "Both probe arms emitted reuse-eligible per representation (rng_fully_reset via "
                "arm_cell + config_slice_declared + include_driver_script_in_hash=False). The "
                "probe instrumentation lives in experiments/_lib/mech459_probe_r.py (inside the "
                "substrate hash). NOTE: this arm is the REFERENCE ctrl of 780/781 but its "
                "fingerprint will NOT match theirs -- the probed trainer is a distinct _lib "
                "module, which correctly refuses a cross-lineage reuse."
            ),
        },
        "config": cfg,
        "load_bearing_dv": (
            "R-(a): the prevalence-normalised CONCENTRATION of forage-contact |adv| mass, "
            "C = (forage |adv| mass fraction) / (forage step fraction), measured on the LATE "
            "window of the reference composed-bootstrap ctrl path, once BEFORE the per-episode "
            "standardisation adv=(adv-mean)/(std+1e-8) (explorer_classes.py:697) and once AFTER. "
            f"PARITY-RESCALING (MECH-459 weak form supported) = C_pre < {CONC_TINY_PRE} AND "
            f"C_post >= {CONC_PARITY_POST} AND delta >= {CONC_RESCALE_DELTA}; STAYS-TINY (weak "
            "form falls, normaliser exonerated) = C_post below parity with delta below the "
            f"rescale floor; ALREADY-CONCENTRATED = C_pre >= {CONC_TINY_PRE} (the weak form's "
            "premise fails). Only the MEAN SUBTRACTION can move the composition -- the std "
            "divide is provably fraction-preserving -- so mean_episode_mean_adv is recorded as "
            "the audit of the mechanism. R-(b) covariate: std(V)/std(G) and the pre-reward vs "
            "far separation ratio on LocalViewGreedyPolicy-visited states, separating "
            "collapsed-to-the-bimodal-mean (MSE half) from flat-uninformed (signal-absence)."
        ),
        "notes": (
            "MECH-459 probe R, sanctioned by decision_MECH-459_registry_and_brake_2026-07-18.md "
            "Decision 2b. DIAGNOSTIC on INTERMEDIATE QUANTITIES (excluded from scoring); "
            "PROMOTES/DEMOTES NOTHING; routes to /failure-autopsy. MECH-459 stays candidate "
            "(NARROWED): the STRONG form (architectural scale-invariance laundering the "
            "770/771/772 eliminations) is REFUTED by that decision's Finding V2 and is not "
            "restated here; only the WEAK form (novelty-noise re-amplification swamping shape "
            "changes) is under test. Measures the reference composed-bootstrap CTRL path -- the "
            "SHARED control arm of 780 and 781 -- with NO manipulation and NO treatment arm. "
            "LIVE PORTFOLIO UNTOUCHED: V3-EXQ-780/781 scripts and the mech457_* _lib modules are "
            "byte-unchanged (instrumentation is a separate mirror module), and "
            "substrate_queue mech457_competence_bootstrap_explorer stays "
            "blocked_pending_discrimination. RE-DERIVE BRAKE: does not fire -- MECH-459's "
            "substrate_ceiling/non_contributory autopsy count is 0, and a diagnostic "
            "discriminating WHY a ceiling holds is named not-braked; no config/env/credit/"
            "capacity knob is moved. GOV-REUSE-1: the decisive readouts (per-step |adv| mass "
            "fractions either side of the standardiser; trained-critic V vs realized return on "
            "demonstrator-visited states) are intermediate training quantities recorded by NO "
            "manifest -> not recoverable by reanalysis -> run. SECOND PAYOFF: R-(a) resolves "
            "whether a V3-EXQ-781 NULL is readable (781's approach drive enters `shaped` INSIDE "
            "the normaliser, so a null is confounded under the weak form; 780's BC auxiliary is "
            "a loss-side CE term outside it and is NOT confounded) -- decision-relevant within "
            "781's runtime. CONDITIONAL FOLLOW-ONS: parity signature -> queue probe K (the "
            "normaliser knock-out, sanctioned but HELD) AND register H-return-scale (axis "
            "normalisation-pathway) under competence_floor via a labelled fanout_growth_events[] "
            "entry citing the decision artifact as fanout_source, reporting the surviving ratio "
            "BOTH ways (vs ORIGINAL frozen 7 and CURRENT 12->13) per labelled_fanout_growth; "
            "stays-tiny or already-concentrated -> do NOT queue probe K, no registry write "
            "(counts stay 7/12), route MECH-459 to `weakened`. Probe R is NOT the adjudicating "
            "run for H-return-scale (that is a later competence-lifting experiment), so "
            "registering after this run still satisfies pre_registered_utc <= resolved_utc."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-782 MECH-459 probe R: advantage composition + critic calibration"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(fan.DRY_SEEDS)
        p0 = fan.DRY_P0
        on_budget = fan.DRY_RL
        eval_eps, steps = fan.DRY_EVAL, fan.DRY_STEPS
        demo_eps = 2
        measure_window = 2
    else:
        seeds = list(fan.SEEDS)
        p0 = fan.P0_WARMUP_EPISODES
        on_budget = int(fan.RL_EPISODES * REF_BUDGET_MULTIPLIER)   # 3000 -- reference budget
        eval_eps, steps = fan.EVAL_EPISODES, fan.STEPS_PER_EPISODE
        demo_eps = DEMO_EPISODES
        measure_window = MEASURE_WINDOW

    result = run_experiment(seeds=seeds, p0=p0, on_budget=on_budget, eval_eps=eval_eps,
                            steps=steps, demo_eps=demo_eps, measure_window=measure_window)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = {
        "seeds": seeds, "rung": fan.RUNG_ID, "arms": list(ARM_ORDER),
        "representations": list(REPRESENTATIONS),
        "p0_warmup_episodes": p0, "on_budget_episodes": on_budget,
        "measure_window_episodes": measure_window,
        "demo_episodes": demo_eps, "pre_reward_horizon": PRE_REWARD_HORIZON,
        "bc_demonstrator": "local_view_greedy",
        "budget_multiplier": REF_BUDGET_MULTIPLIER,
        "eval_episodes": eval_eps, "steps_per_episode": steps,
        "ref_actor_critic_hidden": REF_ACTOR_CRITIC_HIDDEN,
        "ref_credit_replay_passes": REF_CREDIT_PASSES, "ref_credit_topk": REF_CREDIT_TOPK,
        "ref_cotrain_encoder": REF_COTRAIN_ENCODER,
        "ac_lr": fan.AC_LR, "ac_gamma": fan.AC_GAMMA,
        "ppo_gamma": x734.PPO_GAMMA, "ppo_gae_lambda": x734.PPO_GAE_LAMBDA,
        "forage_bonus": x734.FORAGE_BONUS, "novelty_coef": x734.NOVELTY_COEF,
        "ctrl_config": _make_cfg(on_budget).as_slice(),
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "thresholds": result["thresholds"],
        "probe": "MECH-459 probe R (advantage composition + critic calibration)",
        "routed_by": "decision_MECH-459_registry_and_brake_2026-07-18 (Decision 2b)",
    }
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run), cfg=cfg)

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=args.dry_run, config=cfg, seeds=seeds,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - started).total_seconds(),
    )

    print(f"manifest: {out_path}", flush=True)
    hl = result["headline"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"readiness_met={result['readiness']['readiness_met']} "
        f"non_degenerate={result['non_degenerate']}", flush=True,
    )
    for rep in REPRESENTATIONS:
        pr = hl["per_representation"][rep]
        print(
            f"  {rep}: C_pre={pr['forage_concentration_pre_mean']} "
            f"C_post={pr['forage_concentration_post_mean']} "
            f"(delta={pr['forage_concentration_delta']}) -> {pr['adv_composition_verdict']} | "
            f"std(V)/std(G)={pr['value_std_over_return_std_mean']} "
            f"sep={pr['separation_ratio_mean']} -> {pr['critic_calibration_verdict']}",
            flush=True,
        )
    print(
        f"  exq_781_null_readable={hl['exq_781_null_readable']} "
        f"probe_k_recommended={hl['probe_k_recommended']}", flush=True,
    )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = str(result["outcome"]).upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
